#include "star_mask_dog_ops.h"

#include "common/compat.h"
#include "common/cpu_compat.h"
#include "common/gaussian_kernel.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

int64_t reflect101(int64_t index, const int64_t length) {
    if (length <= 1) {
        return 0;
    }
    while (index < 0 || index >= length) {
        index = index < 0 ? -index : 2 * length - index - 2;
    }
    return index;
}

template <typename T>
void convert_to_gray(const T* image, float* gray, const int64_t plane_size, const int channels) {
    constexpr float max_value = std::is_same_v<T, uint8_t> ? 255.0f : 65535.0f;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t index = 0; index < plane_size; ++index) {
        if (channels == 1) {
            gray[index] = static_cast<float>(image[index]) / max_value;
            continue;
        }
        const int64_t base = index * 3;
        gray[index] = (0.114f * static_cast<float>(image[base]) +
                       0.587f * static_cast<float>(image[base + 1]) +
                       0.299f * static_cast<float>(image[base + 2])) /
                      max_value;
    }
}

void gaussian_horizontal(const float* input, float* output, const float* kernel, const int radius,
                         const int64_t height, const int64_t width) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        const float* input_row = input + y * width;
        float* output_row = output + y * width;
        const int64_t interior_begin = std::min<int64_t>(radius, width);
        const int64_t interior_end = std::max<int64_t>(interior_begin, width - radius);
        for (int64_t x = 0; x < interior_begin; ++x) {
            float sum = 0.0f;
            for (int offset = -radius; offset <= radius; ++offset) {
                const int64_t xx = reflect101(x + offset, width);
                sum += input_row[xx] * kernel[offset + radius];
            }
            output_row[x] = sum;
        }
        for (int64_t x = interior_begin; x < interior_end; ++x) {
            float sum = 0.0f;
#if HNW_ENABLE_OMP_SIMD
#pragma omp simd reduction(+ : sum)
#endif
            for (int offset = -radius; offset <= radius; ++offset) {
                sum += input_row[x + offset] * kernel[offset + radius];
            }
            output_row[x] = sum;
        }
        for (int64_t x = interior_end; x < width; ++x) {
            float sum = 0.0f;
            for (int offset = -radius; offset <= radius; ++offset) {
                const int64_t xx = reflect101(x + offset, width);
                sum += input_row[xx] * kernel[offset + radius];
            }
            output_row[x] = sum;
        }
    }
}

void gaussian_vertical(const float* input, float* output, const float* kernel, const int radius,
                       const int64_t height, const int64_t width) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        float* output_row = output + y * width;
        std::fill_n(output_row, static_cast<size_t>(width), 0.0f);
        for (int offset = -radius; offset <= radius; ++offset) {
            const int64_t yy = reflect101(y + offset, height);
            const float* input_row = input + yy * width;
            const float weight = kernel[offset + radius];
#if HNW_ENABLE_OMP_SIMD
#pragma omp simd
#endif
            for (int64_t x = 0; x < width; ++x) {
                output_row[x] += input_row[x] * weight;
            }
        }
    }
}

bool cross_active(const int kernel_size, const int y, const int x) {
    const int radius = kernel_size / 2;
    return y == radius || x == radius;
}

void erode_cross(const uint8_t* input, uint8_t* output, const int64_t height, const int64_t width,
                 const int kernel_size) {
    const int radius = kernel_size / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            bool keep = true;
            for (int kernel_y = 0; kernel_y < kernel_size && keep; ++kernel_y) {
                const int64_t yy = y + kernel_y - radius;
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
                    if (!cross_active(kernel_size, kernel_y, kernel_x)) {
                        continue;
                    }
                    const int64_t xx = x + kernel_x - radius;
                    if (xx >= 0 && xx < width && input[yy * width + xx] == 0) {
                        keep = false;
                        break;
                    }
                }
            }
            output[y * width + x] = keep ? 1 : 0;
        }
    }
}

void dilate_cross(const uint8_t* input, uint8_t* output, const int64_t height, const int64_t width,
                  const int kernel_size) {
    const int radius = kernel_size / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            bool keep = false;
            for (int kernel_y = 0; kernel_y < kernel_size && !keep; ++kernel_y) {
                const int64_t yy = y + kernel_y - radius;
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
                    if (!cross_active(kernel_size, kernel_y, kernel_x)) {
                        continue;
                    }
                    const int64_t xx = x + kernel_x - radius;
                    if (xx >= 0 && xx < width && input[yy * width + xx] != 0) {
                        keep = true;
                        break;
                    }
                }
            }
            output[y * width + x] = keep ? 1 : 0;
        }
    }
}

template <typename T>
void star_mask_dog_cpu(const T* image, uint8_t* output, const int64_t height, const int64_t width,
                       const int channels, const float sigma_small, const float sigma_large,
                       const float threshold_ratio, const int open_kernel_size,
                       const int dilate_kernel_size) {
    const int64_t plane_size = height * width;
    const std::vector<float> small_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_small, "star_mask_dog_cpu");
    const std::vector<float> large_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_large, "star_mask_dog_cpu");
    const int small_radius = static_cast<int>(small_kernel.size() / 2);
    const int large_radius = static_cast<int>(large_kernel.size() / 2);
    std::vector<float> gray(static_cast<size_t>(plane_size));
    std::vector<float> temporary(static_cast<size_t>(plane_size));
    std::vector<float> small_blur(static_cast<size_t>(plane_size));
    std::vector<float> large_blur(static_cast<size_t>(plane_size));

    convert_to_gray(image, gray.data(), plane_size, channels);
    gaussian_horizontal(gray.data(), temporary.data(), small_kernel.data(), small_radius, height,
                        width);
    gaussian_vertical(temporary.data(), small_blur.data(), small_kernel.data(), small_radius,
                      height, width);
    gaussian_horizontal(gray.data(), temporary.data(), large_kernel.data(), large_radius, height,
                        width);
    gaussian_vertical(temporary.data(), large_blur.data(), large_kernel.data(), large_radius,
                      height, width);

    double sum = 0.0;
    double square_sum = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : sum, square_sum) schedule(static)
#endif
    for (int64_t index = 0; index < plane_size; ++index) {
        const float value =
            small_blur[static_cast<size_t>(index)] - large_blur[static_cast<size_t>(index)];
        small_blur[static_cast<size_t>(index)] = value;
        sum += static_cast<double>(value);
        square_sum += static_cast<double>(value) * static_cast<double>(value);
    }
    const double mean = sum / static_cast<double>(plane_size);
    const double variance =
        std::max(0.0, square_sum / static_cast<double>(plane_size) - mean * mean);
    const float threshold =
        static_cast<float>(std::sqrt(variance) * static_cast<double>(threshold_ratio));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t index = 0; index < plane_size; ++index) {
        output[index] = small_blur[static_cast<size_t>(index)] > threshold ? 1 : 0;
    }

    std::vector<uint8_t> scratch(static_cast<size_t>(plane_size));
    if (open_kernel_size > 0) {
        erode_cross(output, scratch.data(), height, width, open_kernel_size);
        dilate_cross(scratch.data(), output, height, width, open_kernel_size);
    }
    if (dilate_kernel_size > 0) {
        dilate_cross(output, scratch.data(), height, width, dilate_kernel_size);
        std::copy(scratch.begin(), scratch.end(), output);
    }
}

void validate(const py::array& image, const float threshold_ratio, const int open_kernel_size,
              const int dilate_kernel_size) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument("star_mask_dog_cpu: image must have shape (H, W) or (H, W, C)");
    }
    if (image.ndim() == 3 && image.shape(2) != 3) {
        throw std::invalid_argument("star_mask_dog_cpu: 3D image must have exactly 3 channels");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0 ||
        image.shape(0) > std::numeric_limits<int>::max() / image.shape(1)) {
        throw std::invalid_argument("star_mask_dog_cpu: invalid image shape");
    }
    if (!std::isfinite(threshold_ratio)) {
        throw std::invalid_argument("star_mask_dog_cpu: threshold_ratio must be finite");
    }
    if (open_kernel_size < 0 || (open_kernel_size > 0 && open_kernel_size % 2 == 0) ||
        dilate_kernel_size < 0 || (dilate_kernel_size > 0 && dilate_kernel_size % 2 == 0)) {
        throw std::invalid_argument(
            "star_mask_dog_cpu: morphology kernel sizes must be zero or positive odd values");
    }
}

template <typename T>
py::array_t<uint8_t>
star_mask_dog_cpu_impl(const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
                       const float sigma_small, const float sigma_large,
                       const float threshold_ratio, const int open_kernel_size,
                       const int dilate_kernel_size) {
    validate(image, threshold_ratio, open_kernel_size, dilate_kernel_size);
    const int64_t height = image.shape(0);
    const int64_t width = image.shape(1);
    const int channels = image.ndim() == 3 ? 3 : 1;
    py::array_t<uint8_t> output({height, width});
    {
        py::gil_scoped_release release;
        star_mask_dog_cpu(image.data(), output.mutable_data(), height, width, channels, sigma_small,
                          sigma_large, threshold_ratio, open_kernel_size, dilate_kernel_size);
    }
    return output;
}

py::array star_mask_dog_cpu_dispatch(const py::array& image, const float sigma_small,
                                     const float sigma_large, const float threshold_ratio,
                                     const int open_kernel_size, const int dilate_kernel_size) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_mask_dog_cpu_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_kernel_size, dilate_kernel_size);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_mask_dog_cpu_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_kernel_size, dilate_kernel_size);
    }
    throw std::invalid_argument("star_mask_dog_cpu: unsupported dtype; expected uint8 or uint16");
}

} // namespace

void bind_star_mask_dog_cpu_ops(py::module_& m) {
    m.def("star_mask_dog_cpu", &star_mask_dog_cpu_dispatch, py::arg("image"),
          py::arg("sigma_small"), py::arg("sigma_large"), py::arg("threshold_ratio"),
          py::arg("open_ksize"), py::arg("dilate_ksize"));
}
