#include "filter_ops.h"

#include "common/cpu_compat.h"
#include "ops/cpu/filter/median_filter_internal.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr ssize_t MAX_MEDIAN_FILTER_KSIZE = 65535;

void validate_image_shape(const py::buffer_info& info, const char* op_name) {
    if (info.ndim != 2 && info.ndim != 3) {
        throw std::invalid_argument(std::string(op_name) +
                                    ": image must have shape (H, W) or (H, W, C)");
    }
    if (info.shape[0] <= 0 || info.shape[1] <= 0) {
        throw std::invalid_argument(std::string(op_name) +
                                    ": image height and width must be positive");
    }
    if (info.ndim == 3) {
        const ssize_t channels = info.shape[2];
        if (channels != 1 && channels != 3 && channels != 4) {
            throw std::invalid_argument(std::string(op_name) +
                                        ": channel count must be 1, 3, or 4");
        }
    }
}

void validate_ksize(ssize_t ksize, const char* op_name) {
    if (ksize <= 0 || (ksize % 2) == 0) {
        throw std::invalid_argument(std::string(op_name) +
                                    ": ksize must be a positive odd integer");
    }
    if (ksize > MAX_MEDIAN_FILTER_KSIZE) {
        throw std::invalid_argument(std::string(op_name) + ": ksize is too large");
    }
}

template <typename T>
py::array_t<T>
median_filter_2d_impl(const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
                      ssize_t ksize) {
    constexpr const char* op_name = "median_filter_2d";
    validate_ksize(ksize, op_name);

    auto input_info = image.request();
    validate_image_shape(input_info, op_name);

    std::vector<ssize_t> out_shape(input_info.shape.begin(), input_info.shape.end());
    py::array_t<T> output(out_shape);
    auto output_info = output.request();

    const ssize_t h = input_info.shape[0];
    const ssize_t w = input_info.shape[1];
    const ssize_t channels = input_info.ndim == 3 ? input_info.shape[2] : 1;

    const auto* input_ptr = static_cast<const T*>(input_info.ptr);
    auto* output_ptr = static_cast<T*>(output_info.ptr);

    py::gil_scoped_release release;
    hnw::cpu::median_filter_2d_kernel<T>(input_ptr, output_ptr, h, w, channels, ksize);
    return output;
}

py::array median_filter_2d_dispatch(const py::array& image, ssize_t ksize) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return median_filter_2d_impl<uint8_t>(image.cast<py::array_t<uint8_t>>(), ksize);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return median_filter_2d_impl<uint16_t>(image.cast<py::array_t<uint16_t>>(), ksize);
    }
    throw std::invalid_argument("median_filter_2d: unsupported dtype; expected uint8/uint16");
}

} // namespace

void bind_filter_ops(py::module_& m) {
    m.def("median_filter_2d", &median_filter_2d_dispatch, py::arg("image"), py::arg("ksize"),
          "Apply exact 2D median filtering with replicate borders.");
}
