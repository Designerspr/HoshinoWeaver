#include "star_shrink_ops.h"

#include "common/cpu_compat.h"
#include "common/median_histogram.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct LabPixel {
    float l;
    float a;
    float b;
};

int64_t reflect101(int64_t idx, const int64_t length) {
    if (length <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= length) {
        if (idx < 0) {
            // OpenCV BORDER_REFLECT_101 maps -1 -> 1; BORDER_REFLECT maps -1 -> 0.
            idx = -idx;
        } else {
            idx = 2 * length - idx - 2;
        }
    }
    return idx;
}

template <typename T> float dtype_max_value() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return 255.0f;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return 65535.0f;
    } else {
        return 1.0f;
    }
}

float srgb_to_linear(const float value) {
    const float x = std::clamp(value, 0.0f, 1.0f);
    if (x <= 0.04045f) {
        return x / 12.92f;
    }
    return std::pow((x + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb(const float value) {
    const float x = std::clamp(value, 0.0f, 1.0f);
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

float lab_f(const float t) {
    constexpr float delta = 6.0f / 29.0f;
    constexpr float delta3 = delta * delta * delta;
    if (t > delta3) {
        return std::cbrt(t);
    }
    return t / (3.0f * delta * delta) + 4.0f / 29.0f;
}

float lab_f_inv(const float t) {
    constexpr float delta = 6.0f / 29.0f;
    if (t > delta) {
        return t * t * t;
    }
    return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

LabPixel bgr_to_lab(const float b, const float g, const float r) {
    const float rl = srgb_to_linear(r);
    const float gl = srgb_to_linear(g);
    const float bl = srgb_to_linear(b);

    const float x = 0.4124564f * rl + 0.3575761f * gl + 0.1804375f * bl;
    const float y = 0.2126729f * rl + 0.7151522f * gl + 0.0721750f * bl;
    const float z = 0.0193339f * rl + 0.1191920f * gl + 0.9503041f * bl;

    const float fx = lab_f(x / 0.95047f);
    const float fy = lab_f(y);
    const float fz = lab_f(z / 1.08883f);
    return {
        116.0f * fy - 16.0f,
        500.0f * (fx - fy),
        200.0f * (fy - fz),
    };
}

void lab_to_bgr(const float l, const float a, const float b_lab, float* b, float* g, float* r) {
    const float fy = (l + 16.0f) / 116.0f;
    const float fx = fy + a / 500.0f;
    const float fz = fy - b_lab / 200.0f;

    const float x = 0.95047f * lab_f_inv(fx);
    const float y = lab_f_inv(fy);
    const float z = 1.08883f * lab_f_inv(fz);

    const float rl = 3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
    const float gl = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
    const float bl = 0.0556434f * x - 0.2040259f * y + 1.0572252f * z;

    *b = std::clamp(linear_to_srgb(bl), 0.0f, 1.0f);
    *g = std::clamp(linear_to_srgb(gl), 0.0f, 1.0f);
    *r = std::clamp(linear_to_srgb(rl), 0.0f, 1.0f);
}

std::vector<int> make_kernel(const std::string& shape, const int ksize) {
    if (ksize <= 0 || ksize % 2 == 0) {
        throw std::invalid_argument(
            "star_shrink_process: kernel sizes must be positive odd values");
    }
    std::vector<int> values(static_cast<size_t>(ksize * ksize), 0);
    const int radius = ksize / 2;
    for (int y = 0; y < ksize; ++y) {
        for (int x = 0; x < ksize; ++x) {
            const int dy = y - radius;
            const int dx = x - radius;
            bool active = false;
            if (shape == "RECT") {
                active = true;
            } else if (shape == "CROSS") {
                active = dx == 0 || dy == 0;
            } else if (shape == "CIRCLE") {
                active = dx * dx + dy * dy <= radius * radius;
            } else {
                throw std::invalid_argument("star_shrink_process: unknown shrink_shape");
            }
            values[static_cast<size_t>(y * ksize + x)] = active ? 1 : 0;
        }
    }
    return values;
}

template <typename T>
float normalized_sample(const T* ptr, const int64_t idx, const float max_value) {
    if constexpr (std::is_same_v<T, float>) {
        return std::clamp(ptr[idx], 0.0f, 1.0f);
    } else {
        return static_cast<float>(ptr[idx]) / max_value;
    }
}

template <typename T> T cast_output(const float value, const float max_value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    if constexpr (std::is_same_v<T, float>) {
        return clamped;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return static_cast<uint8_t>(
            std::clamp(std::nearbyint(clamped * max_value), 0.0f, max_value));
    } else {
        return static_cast<uint16_t>(
            std::clamp(std::nearbyint(clamped * max_value), 0.0f, max_value));
    }
}

using hnw::cpu::MedianHistogram;

int64_t clamp_index(const int64_t value, const int64_t low, const int64_t high) {
    return std::max(low, std::min(value, high));
}

template <typename T>
T gray_integer_sample(const T* image, const int64_t pixel_idx, const int64_t channels,
                      const float max_value) {
    if (channels == 1) {
        return image[pixel_idx];
    }
    const int64_t base = pixel_idx * 3;
    const double gray = 0.114 * static_cast<double>(image[base]) +
                        0.587 * static_cast<double>(image[base + 1]) +
                        0.299 * static_cast<double>(image[base + 2]);
    return static_cast<T>(std::clamp(std::nearbyint(gray), 0.0, static_cast<double>(max_value)));
}

template <typename T>
T gray_at(const std::vector<T>& gray, const int64_t height, const int64_t width, const int64_t y,
          const int64_t x) {
    const int64_t yy = clamp_index(y, 0, height - 1);
    const int64_t xx = clamp_index(x, 0, width - 1);
    return gray[static_cast<size_t>(yy * width + xx)];
}

template <typename T>
void median_filter_gray(const std::vector<T>& gray, std::vector<T>& background,
                        const int64_t height, const int64_t width, const int ksize) {
    const int64_t radius = ksize / 2;
    const uint64_t window_area = static_cast<uint64_t>(ksize) * static_cast<uint64_t>(ksize);
    const uint32_t target_rank = static_cast<uint32_t>(window_area / 2 + 1);
    const auto at = [&](const int64_t y, const int64_t x) {
        return gray_at(gray, height, width, y, x);
    };
#if defined(_OPENMP)
#pragma omp parallel
    {
        MedianHistogram<T> hist;
#pragma omp for schedule(static)
        for (int64_t y = 0; y < height; ++y) {
            hnw::cpu::build_median_histogram_for_row(hist, at, y, radius);
            for (int64_t x = 0; x < width; ++x) {
                background[static_cast<size_t>(y * width + x)] = hist.median(target_rank);
                if (x + 1 < width) {
                    hnw::cpu::slide_median_histogram_right(hist, at, y, x, radius);
                }
            }
        }
    }
#else
    MedianHistogram<T> hist;
    for (int64_t y = 0; y < height; ++y) {
        hnw::cpu::build_median_histogram_for_row(hist, at, y, radius);
        for (int64_t x = 0; x < width; ++x) {
            background[static_cast<size_t>(y * width + x)] = hist.median(target_rank);
            if (x + 1 < width) {
                hnw::cpu::slide_median_histogram_right(hist, at, y, x, radius);
            }
        }
    }
#endif
}

bool cross_active(const int ksize, const int ky, const int kx) {
    const int radius = ksize / 2;
    return (ky - radius) == 0 || (kx - radius) == 0;
}

void erode_cross_mask(const uint8_t* input, uint8_t* output, const int64_t height,
                      const int64_t width, const int ksize) {
    const int radius = ksize / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            bool keep = true;
            for (int ky = 0; ky < ksize && keep; ++ky) {
                const int64_t yy = y + ky - radius;
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int kx = 0; kx < ksize; ++kx) {
                    if (!cross_active(ksize, ky, kx)) {
                        continue;
                    }
                    const int64_t xx = x + kx - radius;
                    if (xx < 0 || xx >= width) {
                        continue;
                    }
                    if (input[yy * width + xx] == 0) {
                        keep = false;
                        break;
                    }
                }
            }
            output[y * width + x] = keep ? 1 : 0;
        }
    }
}

void dilate_cross_mask(const uint8_t* input, uint8_t* output, const int64_t height,
                       const int64_t width, const int ksize) {
    const int radius = ksize / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            bool keep = false;
            for (int ky = 0; ky < ksize && !keep; ++ky) {
                const int64_t yy = y + ky - radius;
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int kx = 0; kx < ksize; ++kx) {
                    if (!cross_active(ksize, ky, kx)) {
                        continue;
                    }
                    const int64_t xx = x + kx - radius;
                    if (xx < 0 || xx >= width) {
                        continue;
                    }
                    if (input[yy * width + xx] != 0) {
                        keep = true;
                        break;
                    }
                }
            }
            output[y * width + x] = keep ? 1 : 0;
        }
    }
}

void apply_cross_morphology(std::vector<uint8_t>& mask, std::vector<uint8_t>& scratch,
                            const int64_t height, const int64_t width, const int open_ksize,
                            const int dilate_ksize) {
    if (open_ksize > 0) {
        erode_cross_mask(mask.data(), scratch.data(), height, width, open_ksize);
        dilate_cross_mask(scratch.data(), mask.data(), height, width, open_ksize);
    }
    if (dilate_ksize > 0) {
        dilate_cross_mask(mask.data(), scratch.data(), height, width, dilate_ksize);
        mask.swap(scratch);
    }
}

template <typename T>
py::array_t<uint8_t>
star_shrink_detect_mask_impl(const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
                             const int ksize, const double threshold_ratio, const int open_ksize,
                             const int dilate_ksize) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "star_shrink_detect_mask: image must have shape (H, W) or (H, W, C)");
    }
    if (image.ndim() == 3 && image.shape(2) != 3) {
        throw std::invalid_argument(
            "star_shrink_detect_mask: 3D image must have exactly 3 channels");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument(
            "star_shrink_detect_mask: image height and width must be positive");
    }
    if (ksize <= 0 || ksize % 2 == 0) {
        throw std::invalid_argument("star_shrink_detect_mask: ksize must be a positive odd value");
    }
    if ((open_ksize < 0) || (open_ksize > 0 && open_ksize % 2 == 0) || (dilate_ksize < 0) ||
        (dilate_ksize > 0 && dilate_ksize % 2 == 0)) {
        throw std::invalid_argument(
            "star_shrink_detect_mask: morphology kernel sizes must be zero or positive odd values");
    }

    const int64_t height = static_cast<int64_t>(image.shape(0));
    const int64_t width = static_cast<int64_t>(image.shape(1));
    const int64_t channels = image.ndim() == 3 ? 3 : 1;
    const int64_t plane_size = height * width;
    const float max_value = dtype_max_value<T>();
    const T* image_ptr = image.data();

    std::vector<T> gray(static_cast<size_t>(plane_size));
    std::vector<T> background(static_cast<size_t>(plane_size));
    std::vector<float> diff(static_cast<size_t>(plane_size));
    std::vector<uint8_t> mask(static_cast<size_t>(plane_size));
    std::vector<uint8_t> scratch(static_cast<size_t>(plane_size));

    {
        py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int64_t idx = 0; idx < plane_size; ++idx) {
            gray[static_cast<size_t>(idx)] =
                gray_integer_sample(image_ptr, idx, channels, max_value);
        }

        median_filter_gray(gray, background, height, width, ksize);

        double sum = 0.0;
        double sum_sq = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : sum, sum_sq) schedule(static)
#endif
        for (int64_t idx = 0; idx < plane_size; ++idx) {
            const float value = (static_cast<float>(gray[static_cast<size_t>(idx)]) -
                                 static_cast<float>(background[static_cast<size_t>(idx)])) /
                                max_value;
            diff[static_cast<size_t>(idx)] = value;
            sum += static_cast<double>(value);
            sum_sq += static_cast<double>(value) * static_cast<double>(value);
        }
        const double mean = sum / static_cast<double>(plane_size);
        const double variance =
            std::max(0.0, sum_sq / static_cast<double>(plane_size) - mean * mean);
        const float threshold = static_cast<float>(std::sqrt(variance) * threshold_ratio);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int64_t idx = 0; idx < plane_size; ++idx) {
            mask[static_cast<size_t>(idx)] = diff[static_cast<size_t>(idx)] > threshold ? 1 : 0;
        }
        apply_cross_morphology(mask, scratch, height, width, open_ksize, dilate_ksize);
    }

    py::array_t<uint8_t> output({height, width});
    std::copy(mask.begin(), mask.end(), output.mutable_data());
    return output;
}

void erode_luma(std::vector<float>& current, std::vector<float>& scratch, const int64_t height,
                const int64_t width, const std::vector<int>& kernel, const int ksize,
                const int times, const float ratio) {
    const int radius = ksize / 2;
    const int64_t plane_size = height * width;
    for (int iter = 0; iter < times; ++iter) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                float minimum = std::numeric_limits<float>::infinity();
                for (int ky = 0; ky < ksize; ++ky) {
                    const int64_t yy = y + ky - radius;
                    if (yy < 0 || yy >= height) {
                        continue;
                    }
                    for (int kx = 0; kx < ksize; ++kx) {
                        if (!kernel[static_cast<size_t>(ky * ksize + kx)]) {
                            continue;
                        }
                        const int64_t xx = x + kx - radius;
                        if (xx < 0 || xx >= width) {
                            continue;
                        }
                        minimum = std::min(minimum, current[static_cast<size_t>(yy * width + xx)]);
                    }
                }
                const int64_t idx = y * width + x;
                scratch[static_cast<size_t>(idx)] =
                    minimum * ratio + current[static_cast<size_t>(idx)] * (1.0f - ratio);
            }
        }
        current.swap(scratch);
    }
    if (current.size() != static_cast<size_t>(plane_size)) {
        throw std::runtime_error("star_shrink_process: internal luma buffer error");
    }
}

template <typename T>
void build_channel_integral(const T* image, std::vector<double>& integral, const int64_t height,
                            const int64_t width, const int64_t channels, const int64_t channel,
                            const int64_t kernel_size, const float max_value) {
    const int64_t radius = kernel_size / 2;
    const int64_t ext_height = height + 2 * radius;
    const int64_t ext_width = width + 2 * radius;
    const int64_t integral_width = ext_width + 1;
    integral.assign(static_cast<size_t>((ext_height + 1) * integral_width), 0.0);

    for (int64_t y = 0; y < ext_height; ++y) {
        double row_sum = 0.0;
        const int64_t yy = reflect101(y - radius, height);
        for (int64_t x = 0; x < ext_width; ++x) {
            const int64_t xx = reflect101(x - radius, width);
            const int64_t src_idx = (yy * width + xx) * channels + channel;
            row_sum += normalized_sample(image, src_idx, max_value);
            const int64_t out_idx = (y + 1) * integral_width + (x + 1);
            integral[static_cast<size_t>(out_idx)] =
                integral[static_cast<size_t>(out_idx - integral_width)] + row_sum;
        }
    }
}

double rect_sum(const std::vector<double>& integral, const int64_t integral_width, const int64_t y0,
                const int64_t x0, const int64_t y1, const int64_t x1) {
    const int64_t a = y0 * integral_width + x0;
    const int64_t b = y0 * integral_width + x1;
    const int64_t c = y1 * integral_width + x0;
    const int64_t d = y1 * integral_width + x1;
    return integral[static_cast<size_t>(d)] - integral[static_cast<size_t>(b)] -
           integral[static_cast<size_t>(c)] + integral[static_cast<size_t>(a)];
}

template <typename T>
py::array_t<T> star_shrink_process_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    const int shrink_ksize, const std::string& shrink_shape, const int shrink_times,
    const float shrink_ratio, const int deringing_ksize) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "star_shrink_process: image must have shape (H, W) or (H, W, C)");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument("star_shrink_process: image height and width must be positive");
    }
    if (image.ndim() == 3 && image.shape(2) != 3) {
        throw std::invalid_argument("star_shrink_process: 3D image must have exactly 3 channels");
    }
    if (mask.ndim() != 2 || mask.shape(0) != image.shape(0) || mask.shape(1) != image.shape(1)) {
        throw std::invalid_argument("star_shrink_process: star_mask must have shape (H, W)");
    }
    if (shrink_times <= 0) {
        throw std::invalid_argument("star_shrink_process: shrink_times must be positive");
    }
    if (!(shrink_ratio > 0.0f && shrink_ratio <= 1.0f)) {
        throw std::invalid_argument("star_shrink_process: shrink_ratio must be in (0, 1]");
    }
    if (deringing_ksize <= 0 || deringing_ksize % 2 == 0) {
        throw std::invalid_argument(
            "star_shrink_process: deringing_ksize must be a positive odd value");
    }

    const int64_t height = static_cast<int64_t>(image.shape(0));
    const int64_t width = static_cast<int64_t>(image.shape(1));
    const int64_t channels = image.ndim() == 3 ? 3 : 1;
    const int64_t plane_size = height * width;
    const int64_t total = plane_size * channels;
    const float max_value = dtype_max_value<T>();
    const auto kernel = make_kernel(shrink_shape, shrink_ksize);

    py::array_t<T> output(image.request().shape);
    const T* HNW_RESTRICT image_ptr = image.data();
    const uint8_t* HNW_RESTRICT mask_ptr = mask.data();
    T* HNW_RESTRICT out_ptr = output.mutable_data();

    std::vector<float> luma(static_cast<size_t>(plane_size));
    std::vector<float> luma_scratch(static_cast<size_t>(plane_size));
    std::vector<float> lab_a;
    std::vector<float> lab_b;
    std::vector<float> shrunk(static_cast<size_t>(total));
    if (channels == 3) {
        lab_a.resize(static_cast<size_t>(plane_size));
        lab_b.resize(static_cast<size_t>(plane_size));
    }

    {
        py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int64_t idx = 0; idx < plane_size; ++idx) {
            if (channels == 1) {
                luma[static_cast<size_t>(idx)] = normalized_sample(image_ptr, idx, max_value);
            } else {
                const int64_t base = idx * 3;
                const LabPixel lab = bgr_to_lab(normalized_sample(image_ptr, base, max_value),
                                                normalized_sample(image_ptr, base + 1, max_value),
                                                normalized_sample(image_ptr, base + 2, max_value));
                luma[static_cast<size_t>(idx)] = lab.l;
                lab_a[static_cast<size_t>(idx)] = lab.a;
                lab_b[static_cast<size_t>(idx)] = lab.b;
            }
        }

        erode_luma(luma, luma_scratch, height, width, kernel, shrink_ksize, shrink_times,
                   shrink_ratio);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int64_t idx = 0; idx < plane_size; ++idx) {
            if (channels == 1) {
                shrunk[static_cast<size_t>(idx)] = luma[static_cast<size_t>(idx)];
            } else {
                float b = 0.0f;
                float g = 0.0f;
                float r = 0.0f;
                lab_to_bgr(luma[static_cast<size_t>(idx)], lab_a[static_cast<size_t>(idx)],
                           lab_b[static_cast<size_t>(idx)], &b, &g, &r);
                const int64_t base = idx * 3;
                shrunk[static_cast<size_t>(base)] = b;
                shrunk[static_cast<size_t>(base + 1)] = g;
                shrunk[static_cast<size_t>(base + 2)] = r;
            }
        }

        const int64_t radius = deringing_ksize / 2;
        const int64_t ext_width = width + 2 * radius;
        const int64_t integral_width = ext_width + 1;
        const double denom =
            static_cast<double>(deringing_ksize) * static_cast<double>(deringing_ksize);
        std::vector<double> integral;
        for (int64_t c = 0; c < channels; ++c) {
            build_channel_integral(image_ptr, integral, height, width, channels, c, deringing_ksize,
                                   max_value);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const int64_t pixel_idx = y * width + x;
                    const int64_t out_idx = pixel_idx * channels + c;
                    if (mask_ptr[pixel_idx] == 0) {
                        out_ptr[out_idx] = image_ptr[out_idx];
                        continue;
                    }
                    const double blur = rect_sum(integral, integral_width, y, x,
                                                 y + deringing_ksize, x + deringing_ksize) /
                                        denom;
                    const float value =
                        std::max(shrunk[static_cast<size_t>(out_idx)], static_cast<float>(blur));
                    out_ptr[out_idx] = cast_output<T>(value, max_value);
                }
            }
        }
    }

    return output;
}

py::array star_shrink_process_dispatch(const py::array& image, const py::array& star_mask,
                                       const int shrink_ksize, const std::string& shrink_shape,
                                       const int shrink_times, const float shrink_ratio,
                                       const int deringing_ksize) {
    auto mask_arr =
        star_mask.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_shrink_process_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), mask_arr,
            shrink_ksize, shrink_shape, shrink_times, shrink_ratio, deringing_ksize);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_shrink_process_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            mask_arr, shrink_ksize, shrink_shape, shrink_times, shrink_ratio, deringing_ksize);
    }
    throw std::invalid_argument("star_shrink_process: unsupported dtype; expected uint8 or uint16");
}

py::array star_shrink_detect_mask_dispatch(const py::array& image, const int ksize,
                                           const double threshold_ratio, const int open_ksize,
                                           const int dilate_ksize) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_shrink_detect_mask_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), ksize,
            threshold_ratio, open_ksize, dilate_ksize);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_shrink_detect_mask_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(), ksize,
            threshold_ratio, open_ksize, dilate_ksize);
    }
    throw std::invalid_argument(
        "star_shrink_detect_mask: unsupported dtype; expected uint8 or uint16");
}

} // namespace

void bind_star_shrink_ops(py::module_& m) {
    m.def("star_shrink_process", &star_shrink_process_dispatch, py::arg("image"),
          py::arg("star_mask"), py::arg("shrink_ksize"), py::arg("shrink_shape"),
          py::arg("shrink_times"), py::arg("shrink_ratio"), py::arg("deringing_ksize"),
          "Fused CPU star shrink luma erosion, deringing, and mask apply.");
    m.def("star_shrink_detect_mask", &star_shrink_detect_mask_dispatch, py::arg("image"),
          py::arg("ksize"), py::arg("threshold_ratio"), py::arg("open_ksize"),
          py::arg("dilate_ksize"), "CPU star shrink threshold detector mask.");
}
