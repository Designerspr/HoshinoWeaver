#include "star_detect_fused_pixel_components_ops.h"

#include "common/compat.h"
#include "common/cpu_compat.h"
#include "common/wavelet_geometry.h"
#include "ops/cpu/wavelet/wavelet_ops.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
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

void gaussian_rows(const double* input, double* output, const double* kernel, const int64_t height,
                   const int64_t width, const int64_t kernel_size) {
    const int64_t radius = kernel_size / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            double value = 0.0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                const int64_t xx = reflect101(x + k - radius, width);
                value += input[y * width + xx] * kernel[k];
            }
            output[y * width + x] = value;
        }
    }
}

void gaussian_cols(const double* input, double* output, const double* kernel, const int64_t height,
                   const int64_t width, const int64_t kernel_size) {
    const int64_t radius = kernel_size / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            double value = 0.0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                const int64_t yy = reflect101(y + k - radius, height);
                value += input[yy * width + x] * kernel[k];
            }
            output[y * width + x] = value;
        }
    }
}

void resize_linear(const double* input, double* output, const int64_t input_height,
                   const int64_t input_width, const int64_t output_height,
                   const int64_t output_width, const uint8_t* output_mask = nullptr) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < output_height; ++y) {
        const double source_y = (static_cast<double>(y) + 0.5) * static_cast<double>(input_height) /
                                    static_cast<double>(output_height) -
                                0.5;
        const int64_t y0_raw = static_cast<int64_t>(std::floor(source_y));
        const double wy = source_y - static_cast<double>(y0_raw);
        const int64_t y0 = std::clamp<int64_t>(y0_raw, 0, input_height - 1);
        const int64_t y1 = std::clamp<int64_t>(y0_raw + 1, 0, input_height - 1);
        for (int64_t x = 0; x < output_width; ++x) {
            const int64_t offset = y * output_width + x;
            if (output_mask != nullptr && output_mask[offset] == 0) {
                output[offset] = 0.0;
                continue;
            }
            const double source_x = (static_cast<double>(x) + 0.5) *
                                        static_cast<double>(input_width) /
                                        static_cast<double>(output_width) -
                                    0.5;
            const int64_t x0_raw = static_cast<int64_t>(std::floor(source_x));
            const double wx = source_x - static_cast<double>(x0_raw);
            const int64_t x0 = std::clamp<int64_t>(x0_raw, 0, input_width - 1);
            const int64_t x1 = std::clamp<int64_t>(x0_raw + 1, 0, input_width - 1);
            const double top = input[y0 * input_width + x0] +
                               (input[y0 * input_width + x1] - input[y0 * input_width + x0]) * wx;
            const double bottom =
                input[y1 * input_width + x0] +
                (input[y1 * input_width + x1] - input[y1 * input_width + x0]) * wx;
            output[offset] = top + (bottom - top) * wy;
        }
    }
}

double percentile_995(std::vector<double>* values) {
    if (values->empty()) {
        throw std::invalid_argument("star_detect_fused_pixel_components: mask selects no pixels");
    }
    const double rank = 0.995 * static_cast<double>(values->size() - 1);
    const size_t lower_index = static_cast<size_t>(std::floor(rank));
    const size_t upper_index = static_cast<size_t>(std::ceil(rank));
    std::nth_element(values->begin(), values->begin() + lower_index, values->end());
    const double lower = (*values)[lower_index];
    if (lower_index == upper_index) {
        return lower;
    }
    std::nth_element(values->begin() + static_cast<ptrdiff_t>(lower_index + 1),
                     values->begin() + static_cast<ptrdiff_t>(upper_index), values->end());
    const double upper = (*values)[upper_index];
    return lower + (upper - lower) * (rank - static_cast<double>(lower_index));
}

void threshold_open(const double* image, const uint8_t* mask, uint8_t* output, const int64_t height,
                    const int64_t width, const double threshold) {
    std::vector<uint8_t> eroded(static_cast<size_t>(height * width));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            bool keep = true;
            for (int64_t dy = -1; dy <= 1 && keep; ++dy) {
                const int64_t yy = y + dy;
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int64_t dx = -1; dx <= 1; ++dx) {
                    const int64_t xx = x + dx;
                    if (xx < 0 || xx >= width) {
                        continue;
                    }
                    const int64_t offset = yy * width + xx;
                    if (mask[offset] == 0 || !(image[offset] > threshold)) {
                        keep = false;
                        break;
                    }
                }
            }
            eroded[static_cast<size_t>(y * width + x)] = keep ? 255 : 0;
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            uint8_t value = 0;
            for (int64_t dy = -1; dy <= 1 && value == 0; ++dy) {
                const int64_t yy = y + dy;
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int64_t dx = -1; dx <= 1; ++dx) {
                    const int64_t xx = x + dx;
                    if (xx >= 0 && xx < width &&
                        eroded[static_cast<size_t>(yy * width + xx)] != 0) {
                        value = 255;
                        break;
                    }
                }
            }
            output[y * width + x] = value;
        }
    }
}

void connected_component_stats(const uint8_t* binary_mask, const double* image,
                               const int64_t height, const int64_t width,
                               std::vector<double>* positions, std::vector<double>* intensities) {
    const int64_t total = height * width;
    std::vector<uint8_t> visited(static_cast<size_t>(total), 0);
    std::vector<int64_t> queue;
    positions->clear();
    intensities->clear();

    for (int64_t start = 0; start < total; ++start) {
        if (binary_mask[start] == 0 || visited[static_cast<size_t>(start)] != 0) {
            continue;
        }
        queue.clear();
        queue.push_back(start);
        visited[static_cast<size_t>(start)] = 1;
        int64_t count = 0;
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_intensity = 0.0;
        for (size_t head = 0; head < queue.size(); ++head) {
            const int64_t index = queue[head];
            const int64_t y = index / width;
            const int64_t x = index - y * width;
            ++count;
            sum_x += static_cast<double>(x);
            sum_y += static_cast<double>(y);
            sum_intensity += image[index];
            for (int64_t dy = -1; dy <= 1; ++dy) {
                const int64_t yy = y + dy;
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int64_t dx = -1; dx <= 1; ++dx) {
                    const int64_t xx = x + dx;
                    if (xx < 0 || xx >= width) {
                        continue;
                    }
                    const int64_t neighbor = yy * width + xx;
                    if (binary_mask[neighbor] != 0 && visited[static_cast<size_t>(neighbor)] == 0) {
                        visited[static_cast<size_t>(neighbor)] = 1;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        const double inv_count = 1.0 / static_cast<double>(count);
        positions->push_back(sum_x * inv_count);
        positions->push_back(sum_y * inv_count);
        intensities->push_back(sum_intensity * inv_count);
    }
}

void launch_star_detect_fused_pixel_components_cpu(
    const double* image, const uint8_t* external_mask, const double* gaussian_kernel,
    std::vector<double>* positions, std::vector<double>* intensities, uint8_t* binary_mask,
    const int64_t height, const int64_t width, const int64_t small_height,
    const int64_t small_width, const int64_t level, const int64_t gaussian_kernel_size) {
    const int64_t total = height * width;
    std::vector<uint8_t> full_mask;
    const uint8_t* mask = external_mask;
    if (mask == nullptr) {
        full_mask.assign(static_cast<size_t>(total), 1);
        mask = full_mask.data();
    }

    std::vector<double> rows(static_cast<size_t>(total));
    std::vector<double> normalized(static_cast<size_t>(total));
    gaussian_rows(image, rows.data(), gaussian_kernel, height, width, gaussian_kernel_size);
    gaussian_cols(rows.data(), normalized.data(), gaussian_kernel, height, width,
                  gaussian_kernel_size);
    rows.clear();
    rows.shrink_to_fit();

    double sum = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : sum) schedule(static)
#endif
    for (int64_t index = 0; index < total; ++index) {
        sum += normalized[static_cast<size_t>(index)];
    }
    const auto [minimum_iterator, maximum_iterator] =
        std::minmax_element(normalized.begin(), normalized.end());
    const double minimum = *minimum_iterator;
    const double maximum = *maximum_iterator;
    const double range = maximum - minimum;
    if (!(range > 0.0)) {
        throw std::runtime_error(
            "star_detect_fused_pixel_components: blurred image has zero range");
    }
    const double mean = sum / static_cast<double>(total);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t index = 0; index < total; ++index) {
        normalized[static_cast<size_t>(index)] =
            (normalized[static_cast<size_t>(index)] - mean) / range;
    }

    std::vector<double> small(static_cast<size_t>(small_height * small_width));
    resize_linear(normalized.data(), small.data(), height, width, small_height, small_width);
    normalized.clear();
    normalized.shrink_to_fit();
    hnw::wavelet::CpuImage reconstructed =
        hnw::wavelet::dec_rec_cpu(small.data(), small_height, small_width, level);
    small.clear();
    small.shrink_to_fit();

    std::vector<double> image_rec(static_cast<size_t>(total));
    resize_linear(reconstructed.values.data(), image_rec.data(), reconstructed.height,
                  reconstructed.width, height, width, mask);

    std::vector<double> percentile_values;
    percentile_values.reserve(static_cast<size_t>(total));
    for (int64_t index = 0; index < total; ++index) {
        if (mask[index] != 0) {
            percentile_values.push_back(image_rec[static_cast<size_t>(index)]);
        }
    }
    const double threshold = percentile_995(&percentile_values);
    threshold_open(image_rec.data(), mask, binary_mask, height, width, threshold);
    connected_component_stats(binary_mask, image_rec.data(), height, width, positions, intensities);
}

py::tuple star_detect_fused_pixel_components_cpu_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& image,
    py::object mask_object, const ssize_t small_height, const ssize_t small_width,
    const ssize_t level,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& gaussian_kernel) {
    if (image.ndim() != 2) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image must be 2D");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0 || small_height <= 0 || small_width <= 0) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: image dimensions must be positive");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() / image.shape(1) ||
        small_height > std::numeric_limits<int>::max() / small_width) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image is too large");
    }
    if (level <= 0) {
        throw std::invalid_argument("star_detect_fused_pixel_components: invalid wavelet level");
    }
    if (gaussian_kernel.ndim() != 1 || gaussian_kernel.shape(0) <= 0) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: gaussian kernel must be 1D");
    }

    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask;
    const uint8_t* mask_pointer = nullptr;
    if (!mask_object.is_none()) {
        mask = mask_object.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (mask.ndim() != 2 || mask.shape(0) != image.shape(0) ||
            mask.shape(1) != image.shape(1)) {
            throw std::invalid_argument(
                "star_detect_fused_pixel_components: mask shape must match image");
        }
        mask_pointer = mask.data();
    }

    const auto [output_height, output_width] =
        hnw::wavelet::reconstructed_shape(small_height, small_width, level);
    if (output_height <= 0 || output_width <= 0) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: wavelet output shape is invalid");
    }

    std::vector<double> positions;
    std::vector<double> intensities;
    py::array_t<uint8_t> binary_mask({image.shape(0), image.shape(1)});
    {
        py::gil_scoped_release release;
        launch_star_detect_fused_pixel_components_cpu(
            image.data(), mask_pointer, gaussian_kernel.data(), &positions, &intensities,
            binary_mask.mutable_data(), image.shape(0), image.shape(1), small_height, small_width,
            level, gaussian_kernel.shape(0));
    }

    const ssize_t count = static_cast<ssize_t>(intensities.size());
    py::array_t<double> positions_output({count, static_cast<ssize_t>(2)});
    py::array_t<double> intensities_output({count});
    std::copy(positions.begin(), positions.end(), positions_output.mutable_data());
    std::copy(intensities.begin(), intensities.end(), intensities_output.mutable_data());
    return py::make_tuple(positions_output, intensities_output, binary_mask);
}

} // namespace

void bind_star_detect_fused_pixel_components_cpu_ops(py::module_& m) {
    m.def("star_detect_fused_pixel_components_cpu", &star_detect_fused_pixel_components_cpu_impl,
          py::arg("image"), py::arg("mask"), py::arg("small_height"), py::arg("small_width"),
          py::arg("level"), py::arg("gaussian_kernel"));
}
