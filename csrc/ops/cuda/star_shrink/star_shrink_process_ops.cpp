#include "star_shrink_process_ops.h"

#include "common/compat.h"

#include <pybind11/numpy.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

void launch_star_shrink_process_cuda_u8(const uint8_t* image_host, const uint8_t* mask_host,
                                        uint8_t* out_host, int height, int width, int channels,
                                        int shrink_ksize, int shrink_shape, int shrink_times,
                                        float shrink_ratio, int deringing_ksize);

void launch_star_shrink_process_cuda_u16(const uint16_t* image_host, const uint8_t* mask_host,
                                         uint16_t* out_host, int height, int width, int channels,
                                         int shrink_ksize, int shrink_shape, int shrink_times,
                                         float shrink_ratio, int deringing_ksize);

namespace {

template <typename T>
using launch_fn_t = void (*)(const T* image_host, const uint8_t* mask_host, T* out_host, int height,
                             int width, int channels, int shrink_ksize, int shrink_shape,
                             int shrink_times, float shrink_ratio, int deringing_ksize);

int parse_shape(const std::string& shape) {
    if (shape == "RECT") {
        return 0;
    }
    if (shape == "CROSS") {
        return 1;
    }
    if (shape == "CIRCLE") {
        return 2;
    }
    throw std::invalid_argument("star_shrink_process_cuda: unknown shrink_shape");
}

void validate_common(const py::array& image, const py::array& mask, const int shrink_ksize,
                     const int shrink_times, const float shrink_ratio, const int deringing_ksize) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "star_shrink_process_cuda: image must have shape (H, W) or (H, W, C)");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument(
            "star_shrink_process_cuda: image height and width must be positive");
    }
    if (image.ndim() == 3 && image.shape(2) != 3) {
        throw std::invalid_argument(
            "star_shrink_process_cuda: 3D image must have exactly 3 channels");
    }
    if (mask.ndim() != 2 || mask.shape(0) != image.shape(0) || mask.shape(1) != image.shape(1)) {
        throw std::invalid_argument("star_shrink_process_cuda: star_mask must have shape (H, W)");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() ||
        image.shape(1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("star_shrink_process_cuda: image is too large");
    }
    if (shrink_ksize <= 0 || shrink_ksize % 2 == 0) {
        throw std::invalid_argument(
            "star_shrink_process_cuda: shrink_ksize must be a positive odd value");
    }
    if (shrink_times <= 0) {
        throw std::invalid_argument("star_shrink_process_cuda: shrink_times must be positive");
    }
    if (!(shrink_ratio > 0.0f && shrink_ratio <= 1.0f)) {
        throw std::invalid_argument("star_shrink_process_cuda: shrink_ratio must be in (0, 1]");
    }
    if (deringing_ksize <= 0 || deringing_ksize % 2 == 0) {
        throw std::invalid_argument(
            "star_shrink_process_cuda: deringing_ksize must be a positive odd value");
    }
}

template <typename T>
py::array_t<T> star_shrink_process_cuda_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    const int shrink_ksize, const int shrink_shape, const int shrink_times,
    const float shrink_ratio, const int deringing_ksize, launch_fn_t<T> launcher) {
    validate_common(image, mask, shrink_ksize, shrink_times, shrink_ratio, deringing_ksize);
    const int height = static_cast<int>(image.shape(0));
    const int width = static_cast<int>(image.shape(1));
    const int channels = image.ndim() == 3 ? 3 : 1;
    py::array_t<T> output(image.request().shape);
    {
        py::gil_scoped_release release;
        launcher(image.data(), mask.data(), output.mutable_data(), height, width, channels,
                 shrink_ksize, shrink_shape, shrink_times, shrink_ratio, deringing_ksize);
    }
    return output;
}

py::array star_shrink_process_cuda_dispatch(const py::array& image, const py::array& star_mask,
                                            const int shrink_ksize, const std::string& shrink_shape,
                                            const int shrink_times, const float shrink_ratio,
                                            const int deringing_ksize) {
    auto mask_arr =
        star_mask.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    const int shape_code = parse_shape(shrink_shape);
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_shrink_process_cuda_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), mask_arr,
            shrink_ksize, shape_code, shrink_times, shrink_ratio, deringing_ksize,
            launch_star_shrink_process_cuda_u8);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_shrink_process_cuda_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            mask_arr, shrink_ksize, shape_code, shrink_times, shrink_ratio, deringing_ksize,
            launch_star_shrink_process_cuda_u16);
    }
    throw std::invalid_argument(
        "star_shrink_process_cuda: unsupported dtype; expected uint8 or uint16");
}

} // namespace

void bind_star_shrink_process_cuda_ops(py::module_& m) {
    m.def("star_shrink_process_cuda", &star_shrink_process_cuda_dispatch, py::arg("image"),
          py::arg("star_mask"), py::arg("shrink_ksize"), py::arg("shrink_shape"),
          py::arg("shrink_times"), py::arg("shrink_ratio"), py::arg("deringing_ksize"),
          "CUDA host-in/out fused star shrink luma erosion, deringing, and mask apply.");
}
