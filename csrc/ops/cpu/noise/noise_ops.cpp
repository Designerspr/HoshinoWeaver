#include "noise_ops.h"

#include "common/cpu_compat.h"
#include "common/py_array_utils.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

double select_quantile(std::vector<double>& values, const double q) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double clamped_q = std::clamp(q, 0.0, 1.0);
    const double position = (static_cast<double>(values.size()) - 1.0) * clamped_q;
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    auto begin = values.begin();
    std::nth_element(begin, begin + static_cast<std::ptrdiff_t>(lo), values.end());
    const double lo_value = values[lo];
    if (lo == hi) {
        return lo_value;
    }
    std::nth_element(begin, begin + static_cast<std::ptrdiff_t>(hi), values.end());
    const double hi_value = values[hi];
    const double fraction = position - static_cast<double>(lo);
    return lo_value * (1.0 - fraction) + hi_value * fraction;
}

int64_t reflect_border(int64_t idx, const int64_t length) {
    if (length <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= length) {
        if (idx < 0) {
            idx = -idx - 1;
        } else {
            idx = 2 * length - idx - 1;
        }
    }
    return idx;
}

template <typename T>
void equalize_noise_correct_kernel(py::buffer_info& out_info, const py::buffer_info& max_info,
                                   const py::buffer_info& filled_std_info, const double sigma_ref,
                                   const double c_n_eff, const double max_value,
                                   const double highlight_preserve) {
    auto* HNW_RESTRICT out_ptr = static_cast<T*>(out_info.ptr);
    const auto* HNW_RESTRICT max_ptr = static_cast<const T*>(max_info.ptr);
    const auto* HNW_RESTRICT filled_std_ptr = static_cast<const T*>(filled_std_info.ptr);
    const ssize_t total = out_info.size;
    const T sigma_ref_value = static_cast<T>(sigma_ref);
    const T c_n_eff_value = static_cast<T>(c_n_eff);
    const T max_value_value = static_cast<T>(max_value);
    const T highlight_value = static_cast<T>(highlight_preserve);
    const T denom = max_value_value * (static_cast<T>(1) - highlight_value);

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
    HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(schedule(static))
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        const T max_pixel = max_ptr[i];
        const T numerator =
            std::min(static_cast<T>(0), max_value_value * highlight_value - max_pixel);
        const T fix_strength = numerator / denom + static_cast<T>(1);
        const T fixed_std = fix_strength * filled_std_ptr[i];
        const T corrected = max_pixel - (fixed_std - sigma_ref_value) * c_n_eff_value;
        out_ptr[i] = std::clamp(corrected, static_cast<T>(0), max_value_value);
    }
}

template <typename T>
py::array_t<T> equalize_noise_correct_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& max_img,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& filled_std_img,
    const double sigma_ref, const double c_n_eff, const double max_value,
    const double highlight_preserve) {
    hnw::require_same_shape(max_img, filled_std_img, "equalize_noise_correct");
    if (!(highlight_preserve >= 0.0 && highlight_preserve < 1.0)) {
        throw std::invalid_argument("equalize_noise_correct: highlight_preserve must be in [0, 1)");
    }

    auto max_info = max_img.request();
    auto filled_std_info = filled_std_img.request();
    py::array_t<T> out(max_info.shape);
    auto out_info = out.request();
    equalize_noise_correct_kernel<T>(out_info, max_info, filled_std_info, sigma_ref, c_n_eff,
                                     max_value, highlight_preserve);
    return out;
}

py::array equalize_noise_correct_dispatch(const py::array& max_img, const py::array& filled_std_img,
                                          const double sigma_ref, const double c_n_eff,
                                          const double max_value, const double highlight_preserve) {
    hnw::require_same_dtype(max_img, filled_std_img, "equalize_noise_correct");

    if (py::isinstance<py::array_t<float>>(max_img)) {
        return equalize_noise_correct_impl<float>(
            max_img.cast<py::array_t<float>>(), filled_std_img.cast<py::array_t<float>>(),
            sigma_ref, c_n_eff, max_value, highlight_preserve);
    }
    if (py::isinstance<py::array_t<double>>(max_img)) {
        return equalize_noise_correct_impl<double>(
            max_img.cast<py::array_t<double>>(), filled_std_img.cast<py::array_t<double>>(),
            sigma_ref, c_n_eff, max_value, highlight_preserve);
    }

    throw std::invalid_argument("equalize_noise_correct: unsupported dtype");
}

template <typename T>
void noise_fill_local_mean_kernel(py::buffer_info& out_info, const py::buffer_info& img_info,
                                  const py::buffer_info& mask_info, const int64_t kernel_size) {
    const auto* HNW_RESTRICT img_ptr = static_cast<const T*>(img_info.ptr);
    const auto* HNW_RESTRICT mask_ptr = static_cast<const bool*>(mask_info.ptr);
    auto* HNW_RESTRICT out_ptr = static_cast<T*>(out_info.ptr);

    const int64_t height = static_cast<int64_t>(img_info.shape[0]);
    const int64_t width = static_cast<int64_t>(img_info.shape[1]);
    const int64_t channels = img_info.ndim == 3 ? static_cast<int64_t>(img_info.shape[2]) : 1;
    const int64_t anchor = kernel_size / 2;
    const int64_t before = anchor;
    const int64_t after = kernel_size - anchor - 1;
    const int64_t ext_height = height + before + after;
    const int64_t ext_width = width + before + after;
    const int64_t integral_width = ext_width + 1;
    const int64_t integral_size = (ext_height + 1) * integral_width;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t c = 0; c < channels; ++c) {
        std::vector<double> sum_integral(static_cast<size_t>(integral_size), 0.0);
        std::vector<int32_t> count_integral(static_cast<size_t>(integral_size), 0);

        for (int64_t ey = 0; ey < ext_height; ++ey) {
            const int64_t sy = reflect_border(ey - before, height);
            double row_sum = 0.0;
            int32_t row_count = 0;
            const int64_t integral_row = (ey + 1) * integral_width;
            const int64_t previous_row = ey * integral_width;

            for (int64_t ex = 0; ex < ext_width; ++ex) {
                const int64_t sx = reflect_border(ex - before, width);
                const int64_t src_idx = (sy * width + sx) * channels + c;
                if (!mask_ptr[src_idx]) {
                    row_sum += static_cast<double>(img_ptr[src_idx]);
                    row_count += 1;
                }
                const int64_t integral_idx = integral_row + ex + 1;
                sum_integral[static_cast<size_t>(integral_idx)] =
                    sum_integral[static_cast<size_t>(previous_row + ex + 1)] + row_sum;
                count_integral[static_cast<size_t>(integral_idx)] =
                    count_integral[static_cast<size_t>(previous_row + ex + 1)] + row_count;
            }
        }

        for (int64_t y = 0; y < height; ++y) {
            const int64_t y0 = y;
            const int64_t y1 = y + kernel_size;
            for (int64_t x = 0; x < width; ++x) {
                const int64_t dst_idx = (y * width + x) * channels + c;
                if (!mask_ptr[dst_idx]) {
                    out_ptr[dst_idx] = img_ptr[dst_idx];
                    continue;
                }

                const int64_t x0 = x;
                const int64_t x1 = x + kernel_size;
                const int64_t bottom_right = y1 * integral_width + x1;
                const int64_t top_right = y0 * integral_width + x1;
                const int64_t bottom_left = y1 * integral_width + x0;
                const int64_t top_left = y0 * integral_width + x0;
                const double sum = sum_integral[static_cast<size_t>(bottom_right)] -
                                   sum_integral[static_cast<size_t>(top_right)] -
                                   sum_integral[static_cast<size_t>(bottom_left)] +
                                   sum_integral[static_cast<size_t>(top_left)];
                const int32_t count = count_integral[static_cast<size_t>(bottom_right)] -
                                      count_integral[static_cast<size_t>(top_right)] -
                                      count_integral[static_cast<size_t>(bottom_left)] +
                                      count_integral[static_cast<size_t>(top_left)];
                out_ptr[dst_idx] = count > 0 ? static_cast<T>(sum / static_cast<double>(count))
                                             : static_cast<T>(0);
            }
        }
    }
}

template <typename T>
py::array_t<T>
noise_fill_local_mean_impl(const py::array_t<T, py::array::c_style | py::array::forcecast>& img,
                           const py::array_t<bool, py::array::c_style | py::array::forcecast>& mask,
                           const int64_t kernel_size) {
    hnw::require_same_shape(img, mask, "noise_fill_local_mean");
    if (img.ndim() != 2 && img.ndim() != 3) {
        throw std::invalid_argument("noise_fill_local_mean: expected 2D or 3D array");
    }
    if (kernel_size <= 0) {
        throw std::invalid_argument("noise_fill_local_mean: kernel_size must be positive");
    }

    auto img_info = img.request();
    auto mask_info = mask.request();
    py::array_t<T> out(img_info.shape);
    auto out_info = out.request();
    noise_fill_local_mean_kernel<T>(out_info, img_info, mask_info, kernel_size);
    return out;
}

py::array noise_fill_local_mean_dispatch(const py::array& img, const py::array& mask,
                                         const int64_t kernel_size) {
    if (py::isinstance<py::array_t<float>>(img)) {
        return noise_fill_local_mean_impl<float>(img.cast<py::array_t<float>>(),
                                                 mask.cast<py::array_t<bool>>(), kernel_size);
    }
    if (py::isinstance<py::array_t<double>>(img)) {
        return noise_fill_local_mean_impl<double>(img.cast<py::array_t<double>>(),
                                                  mask.cast<py::array_t<bool>>(), kernel_size);
    }

    throw std::invalid_argument("noise_fill_local_mean: unsupported dtype");
}

template <typename T>
py::object noise_equalization_params_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& max_img,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& mean_img,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& std_img,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& n_img,
    const double top_fraction, const double sigma_reject, const bool minus_only) {
    auto max_info = max_img.request();
    auto mean_info = mean_img.request();
    auto std_info = std_img.request();
    auto n_info = n_img.request();
    hnw::require_same_shape(max_info, mean_info, "noise_equalization_params");
    hnw::require_same_shape(max_info, std_info, "noise_equalization_params");
    if (max_info.ndim != 2 && max_info.ndim != 3) {
        throw std::invalid_argument("noise_equalization_params: expected 2D or 3D array");
    }
    bool n_matches_image = n_info.ndim == max_info.ndim;
    if (n_matches_image) {
        for (ssize_t i = 0; i < max_info.ndim; ++i) {
            if (n_info.shape[i] != max_info.shape[i]) {
                n_matches_image = false;
                break;
            }
        }
    }
    const bool n_matches_pixels = max_info.ndim == 3 && n_info.ndim == 2 &&
                                  n_info.shape[0] == max_info.shape[0] &&
                                  n_info.shape[1] == max_info.shape[1];
    if (!n_matches_image && !n_matches_pixels) {
        throw std::invalid_argument("noise_equalization_params: n_img shape mismatch");
    }
    if (!(top_fraction >= 0.0 && top_fraction <= 1.0)) {
        throw std::invalid_argument("noise_equalization_params: top_fraction must be in [0, 1]");
    }
    if (sigma_reject < 0.0) {
        throw std::invalid_argument("noise_equalization_params: sigma_reject must be non-negative");
    }

    const auto* HNW_RESTRICT max_ptr = static_cast<const T*>(max_info.ptr);
    const auto* HNW_RESTRICT mean_ptr = static_cast<const T*>(mean_info.ptr);
    const auto* HNW_RESTRICT std_ptr = static_cast<const T*>(std_info.ptr);
    const auto* HNW_RESTRICT n_ptr = static_cast<const double*>(n_info.ptr);
    const ssize_t total = max_info.size;
    const ssize_t channels = max_info.ndim == 3 ? max_info.shape[2] : 1;
    const ssize_t pixels = total / channels;
    const bool n_per_pixel = n_matches_pixels && !n_matches_image;
    const ssize_t n_total = n_info.size;

    std::vector<double> n_values(static_cast<size_t>(n_total));
    std::vector<double> channel_sum(static_cast<size_t>(channels), 0.0);
    std::vector<double> channel_sq_sum(static_cast<size_t>(channels), 0.0);

    {
        py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t i = 0; i < n_total; ++i) {
            n_values[static_cast<size_t>(i)] = n_ptr[i];
        }

#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
            std::vector<double> local_sum(static_cast<size_t>(channels), 0.0);
            std::vector<double> local_sq_sum(static_cast<size_t>(channels), 0.0);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (ssize_t i = 0; i < total; ++i) {
                const double std_value = static_cast<double>(std_ptr[i]);
                const ssize_t c = i % channels;
                local_sum[static_cast<size_t>(c)] += std_value;
                local_sq_sum[static_cast<size_t>(c)] += std_value * std_value;
            }
#if defined(_OPENMP)
#pragma omp critical
#endif
            {
                for (ssize_t c = 0; c < channels; ++c) {
                    channel_sum[static_cast<size_t>(c)] += local_sum[static_cast<size_t>(c)];
                    channel_sq_sum[static_cast<size_t>(c)] += local_sq_sum[static_cast<size_t>(c)];
                }
            }
        }
    }

    const double threshold = select_quantile(n_values, 1.0 - top_fraction);
    std::vector<double> sigma_values;
    std::vector<double> ratio_values;
    sigma_values.reserve(static_cast<size_t>(total));
    ratio_values.reserve(static_cast<size_t>(total));

    {
        py::gil_scoped_release release;
        for (ssize_t i = 0; i < total; ++i) {
            const ssize_t n_idx = n_per_pixel ? i / channels : i;
            if (n_ptr[n_idx] < threshold) {
                continue;
            }
            const double sigma = static_cast<double>(std_ptr[i]);
            if (!(sigma > 0.0)) {
                continue;
            }
            sigma_values.push_back(sigma);
            ratio_values.push_back(
                (static_cast<double>(max_ptr[i]) - static_cast<double>(mean_ptr[i])) / sigma);
        }
    }
    if (sigma_values.empty()) {
        return py::none();
    }

    const double c_n_eff = select_quantile(ratio_values, 0.5);
    const double sigma_ref = minus_only ? 0.0 : select_quantile(sigma_values, 0.5);

    py::array_t<bool> mask(max_info.shape);
    auto mask_info = mask.request();
    auto* HNW_RESTRICT mask_ptr = static_cast<bool*>(mask_info.ptr);
    std::vector<double> channel_mean(static_cast<size_t>(channels), 0.0);
    std::vector<double> channel_std(static_cast<size_t>(channels), 0.0);
    for (ssize_t c = 0; c < channels; ++c) {
        const double mean = channel_sum[static_cast<size_t>(c)] / static_cast<double>(pixels);
        const double variance =
            std::max(0.0, channel_sq_sum[static_cast<size_t>(c)] / static_cast<double>(pixels) -
                              mean * mean);
        channel_mean[static_cast<size_t>(c)] = mean;
        channel_std[static_cast<size_t>(c)] = std::sqrt(variance);
    }

    {
        py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t i = 0; i < total; ++i) {
            const ssize_t c = i % channels;
            const double threshold_value = channel_mean[static_cast<size_t>(c)] +
                                           sigma_reject * channel_std[static_cast<size_t>(c)];
            mask_ptr[i] = static_cast<double>(std_ptr[i]) > threshold_value;
        }
    }

    return py::make_tuple(sigma_ref, c_n_eff, mask);
}

py::object noise_equalization_params_dispatch(const py::array& max_img, const py::array& mean_img,
                                              const py::array& std_img, const py::array& n_img,
                                              const double top_fraction, const double sigma_reject,
                                              const bool minus_only) {
    hnw::require_same_dtype(max_img, mean_img, "noise_equalization_params");
    hnw::require_same_dtype(max_img, std_img, "noise_equalization_params");

    if (py::isinstance<py::array_t<float>>(max_img)) {
        return noise_equalization_params_impl<float>(
            max_img.cast<py::array_t<float>>(), mean_img.cast<py::array_t<float>>(),
            std_img.cast<py::array_t<float>>(), n_img.cast<py::array_t<double>>(), top_fraction,
            sigma_reject, minus_only);
    }
    if (py::isinstance<py::array_t<double>>(max_img)) {
        return noise_equalization_params_impl<double>(
            max_img.cast<py::array_t<double>>(), mean_img.cast<py::array_t<double>>(),
            std_img.cast<py::array_t<double>>(), n_img.cast<py::array_t<double>>(), top_fraction,
            sigma_reject, minus_only);
    }

    throw std::invalid_argument("noise_equalization_params: unsupported dtype");
}

} // namespace

void bind_noise_ops(py::module_& m) {
    m.def("equalize_noise_correct", &equalize_noise_correct_dispatch, py::arg("max_img"),
          py::arg("filled_std_img"), py::arg("sigma_ref"), py::arg("c_n_eff"), py::arg("max_value"),
          py::arg("highlight_preserve"),
          "Apply highlight-preserving equalize-noise correction with a C++ pixel kernel.");
    m.def("noise_fill_local_mean", &noise_fill_local_mean_dispatch, py::arg("img"), py::arg("mask"),
          py::arg("kernel_size") = 21, "Fill masked pixels with a reflected-border local mean.");
    m.def("noise_equalization_params", &noise_equalization_params_dispatch, py::arg("max_img"),
          py::arg("mean_img"), py::arg("std_img"), py::arg("n_img"), py::arg("top_fraction") = 0.02,
          py::arg("sigma_reject") = 3.0, py::arg("minus_only") = false,
          "Estimate noise equalization scalar parameters and std outlier mask.");
}
