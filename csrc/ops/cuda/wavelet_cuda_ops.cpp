#include "wavelet_cuda_ops.h"

#include "common/compat.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>

void launch_wavelet_dec_rec_cuda_core(
    const double* image_host,
    double* out_host,
    int height,
    int width,
    int level);

void launch_star_detect_full_connected_components(
    const double* image_host,
    const uint8_t* external_mask_host,
    const double* gaussian_kernel_host,
    const uint8_t* dilate_kernel_host,
    std::vector<double>* positions_xy_host,
    std::vector<double>* areas_host,
    std::vector<double>* intensities_host,
    std::vector<double>* eccentricities_host,
    int height,
    int width,
    int small_height,
    int small_width,
    int level,
    int gaussian_ksize,
    int dilate_height,
    int dilate_width);

namespace {

constexpr ssize_t DB8_FILTER_LEN = 16;

ssize_t dwt_len(const ssize_t n) {
    return (n + DB8_FILTER_LEN - 1) / 2;
}

ssize_t idwt_len(const ssize_t n) {
    return 2 * n - DB8_FILTER_LEN + 2;
}

std::pair<ssize_t, ssize_t> wavelet_output_shape(
    const ssize_t height,
    const ssize_t width,
    const ssize_t level) {
    std::vector<std::pair<ssize_t, ssize_t>> details;
    details.reserve(static_cast<size_t>(level));
    ssize_t current_h = height;
    ssize_t current_w = width;
    for (ssize_t idx = 0; idx < level; ++idx) {
        current_h = dwt_len(current_h);
        current_w = dwt_len(current_w);
        details.emplace_back(current_h, current_w);
    }
    for (ssize_t idx = level - 1; idx >= 0; --idx) {
        current_h = idwt_len(details[static_cast<size_t>(idx)].first);
        current_w = idwt_len(details[static_cast<size_t>(idx)].second);
    }
    return {current_h, current_w};
}

py::array_t<double> wavelet_dec_rec_cuda_core_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& image,
    const ssize_t level) {
    if (image.ndim() != 2) {
        throw std::invalid_argument("wavelet_dec_rec_cuda_core: image must be 2D");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument(
            "wavelet_dec_rec_cuda_core: image height and width must be positive");
    }
    if (level <= 0) {
        throw std::invalid_argument("wavelet_dec_rec_cuda_core: invalid wavelet level");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() ||
        image.shape(1) > std::numeric_limits<int>::max() ||
        level > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("wavelet_dec_rec_cuda_core: input is too large");
    }
    const ssize_t input_size = image.shape(0) * image.shape(1);
    if (input_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("wavelet_dec_rec_cuda_core: input is too large");
    }
    const auto [out_h, out_w] = wavelet_output_shape(
        image.shape(0), image.shape(1), level);
    const ssize_t output_size = out_h * out_w;
    if (out_h <= 0 || out_w <= 0 ||
        out_h > std::numeric_limits<int>::max() ||
        out_w > std::numeric_limits<int>::max() ||
        output_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "wavelet_dec_rec_cuda_core: output shape is invalid");
    }

    py::array_t<double> output({out_h, out_w});
    launch_wavelet_dec_rec_cuda_core(
        image.data(),
        output.mutable_data(),
        static_cast<int>(image.shape(0)),
        static_cast<int>(image.shape(1)),
        static_cast<int>(level));
    return output;
}

py::tuple star_detect_full_connected_components_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& image,
    py::object mask_obj,
    const ssize_t small_height,
    const ssize_t small_width,
    const ssize_t level,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& gaussian_kernel,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& dilate_kernel) {
    if (image.ndim() != 2) {
        throw std::invalid_argument(
            "star_detect_full_connected_components: image must be 2D");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0 ||
        small_height <= 0 || small_width <= 0) {
        throw std::invalid_argument(
            "star_detect_full_connected_components: image dimensions must be positive");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() ||
        image.shape(1) > std::numeric_limits<int>::max() ||
        small_height > std::numeric_limits<int>::max() ||
        small_width > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "star_detect_full_connected_components: image shape is too large");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() / image.shape(1) ||
        small_height > std::numeric_limits<int>::max() / small_width) {
        throw std::invalid_argument(
            "star_detect_full_connected_components: image is too large");
    }
    if (level <= 0 || level > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "star_detect_full_connected_components: invalid wavelet level");
    }
    if (gaussian_kernel.ndim() != 1 || gaussian_kernel.shape(0) <= 0 ||
        gaussian_kernel.shape(0) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "star_detect_full_connected_components: gaussian kernel must be 1D");
    }
    if (dilate_kernel.ndim() != 2 || dilate_kernel.shape(0) <= 0 ||
        dilate_kernel.shape(1) <= 0 ||
        dilate_kernel.shape(0) > std::numeric_limits<int>::max() ||
        dilate_kernel.shape(1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "star_detect_full_connected_components: dilate kernel must be 2D");
    }

    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask;
    const uint8_t* mask_ptr = nullptr;
    if (!mask_obj.is_none()) {
        mask = mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (mask.ndim() != 2 || mask.shape(0) != image.shape(0) ||
            mask.shape(1) != image.shape(1)) {
            throw std::invalid_argument(
                "star_detect_full_connected_components: mask shape must match image");
        }
        mask_ptr = mask.data();
    }

    const auto [small_out_h, small_out_w] = wavelet_output_shape(
        small_height, small_width, level);
    const ssize_t small_output_size = small_out_h * small_out_w;
    if (small_out_h <= 0 || small_out_w <= 0 ||
        small_out_h > std::numeric_limits<int>::max() ||
        small_out_w > std::numeric_limits<int>::max() ||
        small_output_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "star_detect_full_connected_components: wavelet output shape is invalid");
    }

    std::vector<double> positions_xy;
    std::vector<double> areas;
    std::vector<double> intensities;
    std::vector<double> eccentricities;
    launch_star_detect_full_connected_components(
        image.data(),
        mask_ptr,
        gaussian_kernel.data(),
        dilate_kernel.data(),
        &positions_xy,
        &areas,
        &intensities,
        &eccentricities,
        static_cast<int>(image.shape(0)),
        static_cast<int>(image.shape(1)),
        static_cast<int>(small_height),
        static_cast<int>(small_width),
        static_cast<int>(level),
        static_cast<int>(gaussian_kernel.shape(0)),
        static_cast<int>(dilate_kernel.shape(0)),
        static_cast<int>(dilate_kernel.shape(1)));

    const ssize_t out_count = static_cast<ssize_t>(areas.size());
    py::array_t<double> positions({out_count, static_cast<ssize_t>(2)});
    py::array_t<double> areas_out({out_count});
    py::array_t<double> intensities_out({out_count});
    py::array_t<double> eccentricities_out({out_count});
    std::copy(positions_xy.begin(), positions_xy.end(), positions.mutable_data());
    std::copy(areas.begin(), areas.end(), areas_out.mutable_data());
    std::copy(intensities.begin(), intensities.end(), intensities_out.mutable_data());
    std::copy(
        eccentricities.begin(), eccentricities.end(), eccentricities_out.mutable_data());
    return py::make_tuple(
        positions, areas_out, intensities_out, eccentricities_out);
}

}  // namespace

void bind_wavelet_cuda_ops(py::module_& m) {
    m.def(
        "wavelet_dec_rec_cuda_core",
        &wavelet_dec_rec_cuda_core_impl,
        py::arg("image"),
        py::arg("level"));
    m.def(
        "star_detect_full_connected_components_core",
        &star_detect_full_connected_components_impl,
        py::arg("image"),
        py::arg("mask"),
        py::arg("small_height"),
        py::arg("small_width"),
        py::arg("level"),
        py::arg("gaussian_kernel"),
        py::arg("dilate_kernel"));
}
