#include "star_detect_fused_pixel_components_ops.h"

#include "common/compat.h"
#include "common/wavelet_geometry.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

py::tuple star_detect_fused_pixel_components_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& image,
    py::object mask_obj, const ssize_t small_height, const ssize_t small_width, const ssize_t level,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& gaussian_kernel) {
    if (image.ndim() != 2) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image must be 2D");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0 || small_height <= 0 || small_width <= 0) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image "
                                    "dimensions must be positive");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() ||
        image.shape(1) > std::numeric_limits<int>::max() ||
        small_height > std::numeric_limits<int>::max() ||
        small_width > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image shape is too large");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() / image.shape(1) ||
        small_height > std::numeric_limits<int>::max() / small_width) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image is too large");
    }
    if (level <= 0 || level > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("star_detect_fused_pixel_components: invalid wavelet level");
    }
    if (gaussian_kernel.ndim() != 1 || gaussian_kernel.shape(0) <= 0 ||
        gaussian_kernel.shape(0) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: gaussian kernel must be 1D");
    }
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask;
    const uint8_t* mask_ptr = nullptr;
    if (!mask_obj.is_none()) {
        mask = mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (mask.ndim() != 2 || mask.shape(0) != image.shape(0) ||
            mask.shape(1) != image.shape(1)) {
            throw std::invalid_argument(
                "star_detect_fused_pixel_components: mask shape must match image");
        }
        mask_ptr = mask.data();
    }

    const auto [small_out_h, small_out_w] =
        hnw::wavelet::reconstructed_shape(small_height, small_width, level);
    const ssize_t small_output_size = small_out_h * small_out_w;
    if (small_out_h <= 0 || small_out_w <= 0 || small_out_h > std::numeric_limits<int>::max() ||
        small_out_w > std::numeric_limits<int>::max() ||
        small_output_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: wavelet output shape is invalid");
    }

    std::vector<double> positions_xy;
    std::vector<double> intensities;
    py::array_t<uint8_t> binary_mask(std::vector<ssize_t>{image.shape(0), image.shape(1)});
    {
        py::gil_scoped_release release;
        launch_star_detect_fused_pixel_components(
            image.data(), mask_ptr, gaussian_kernel.data(), &positions_xy, &intensities,
            binary_mask.mutable_data(), static_cast<int>(image.shape(0)),
            static_cast<int>(image.shape(1)), static_cast<int>(small_height),
            static_cast<int>(small_width), static_cast<int>(level),
            static_cast<int>(gaussian_kernel.shape(0)));
    }

    const ssize_t out_count = static_cast<ssize_t>(intensities.size());
    py::array_t<double> positions({out_count, static_cast<ssize_t>(2)});
    py::array_t<double> intensities_out({out_count});
    std::copy(positions_xy.begin(), positions_xy.end(), positions.mutable_data());
    std::copy(intensities.begin(), intensities.end(), intensities_out.mutable_data());
    return py::make_tuple(positions, intensities_out, binary_mask);
}

} // namespace

void bind_star_detect_fused_pixel_components_ops(py::module_& m) {
    m.def("star_detect_fused_pixel_components_cuda", &star_detect_fused_pixel_components_impl,
          py::arg("image"), py::arg("mask"), py::arg("small_height"), py::arg("small_width"),
          py::arg("level"), py::arg("gaussian_kernel"));
}
