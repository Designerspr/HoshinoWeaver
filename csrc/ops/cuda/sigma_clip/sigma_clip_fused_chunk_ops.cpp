#include "sigma_clip_fused_chunk_ops.h"

#include "common/compat.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <pybind11/numpy.h>

void launch_sigma_clip_fused_chunk_cuda_u8(
    const uint8_t* stack_host,
    const uint8_t* mask_host,
    double* out_sum_host,
    double* out_sq_host,
    double* out_n_host,
    int64_t n_frames,
    int64_t plane_size,
    double rej_high,
    double rej_low,
    int max_iter,
    bool skip_zero_rgb,
    int64_t channels);

void launch_sigma_clip_fused_chunk_cuda_u16(
    const uint16_t* stack_host,
    const uint8_t* mask_host,
    double* out_sum_host,
    double* out_sq_host,
    double* out_n_host,
    int64_t n_frames,
    int64_t plane_size,
    double rej_high,
    double rej_low,
    int max_iter,
    bool skip_zero_rgb,
    int64_t channels);

namespace {

void validate_shape(const py::array& stack,
                    ssize_t channels,
                    const bool skip_zero_rgb) {
    if (stack.ndim() != 2) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk_cuda: stack must be 2D (n_frames, plane_size)");
    }
    if (stack.shape(0) <= 0 || stack.shape(1) <= 0) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk_cuda: stack dimensions must be positive");
    }
    if (stack.shape(0) > std::numeric_limits<int>::max() ||
        stack.shape(1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk_cuda: stack is too large");
    }
    if (channels <= 0 || channels > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk_cuda: channels must be positive");
    }
    if (skip_zero_rgb && channels >= 3 && stack.shape(1) % channels != 0) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk_cuda: plane_size must be divisible by channels when skip_zero_rgb is true");
    }
}

py::array_t<uint8_t, py::array::c_style | py::array::forcecast> parse_mask(
    const py::object& mask_obj,
    const ssize_t n_frames,
    const ssize_t plane_size,
    const uint8_t** mask_ptr) {
    *mask_ptr = nullptr;
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask_arr;
    if (mask_obj.is_none()) {
        return mask_arr;
    }
    mask_arr = mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    if (mask_arr.ndim() != 2 ||
        mask_arr.shape(0) != n_frames ||
        mask_arr.shape(1) != plane_size) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk_cuda: mask must have shape (n_frames, plane_size)");
    }
    *mask_ptr = mask_arr.data();
    return mask_arr;
}

template <typename T, typename Launcher>
py::tuple sigma_clip_fused_chunk_cuda_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& stack,
    const double rej_high,
    const double rej_low,
    const int max_iter,
    const py::object& mask_obj,
    const bool skip_zero_rgb,
    const ssize_t channels,
    Launcher launcher) {
    validate_shape(stack, channels, skip_zero_rgb);

    const ssize_t n_frames = stack.shape(0);
    const ssize_t plane_size = stack.shape(1);
    const uint8_t* mask_ptr = nullptr;
    auto mask_arr = parse_mask(mask_obj, n_frames, plane_size, &mask_ptr);

    py::array_t<double> out_sum({plane_size});
    py::array_t<double> out_sq({plane_size});
    py::array_t<double> out_n({plane_size});

    {
        py::gil_scoped_release release;
        launcher(
            stack.data(),
            mask_ptr,
            out_sum.mutable_data(),
            out_sq.mutable_data(),
            out_n.mutable_data(),
            n_frames,
            plane_size,
            rej_high,
            rej_low,
            max_iter,
            skip_zero_rgb,
            channels);
    }
    return py::make_tuple(out_sum, out_sq, out_n);
}

py::tuple sigma_clip_fused_chunk_cuda_dispatch(
    const py::array& stack,
    const double rej_high,
    const double rej_low,
    const int max_iter,
    const py::object& mask_obj,
    const bool skip_zero_rgb,
    const ssize_t channels) {
    if (py::isinstance<py::array_t<uint8_t>>(stack)) {
        auto stack_arr =
            stack.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        return sigma_clip_fused_chunk_cuda_impl(
            stack_arr,
            rej_high,
            rej_low,
            max_iter,
            mask_obj,
            skip_zero_rgb,
            channels,
            launch_sigma_clip_fused_chunk_cuda_u8);
    }
    if (py::isinstance<py::array_t<uint16_t>>(stack)) {
        auto stack_arr =
            stack.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        return sigma_clip_fused_chunk_cuda_impl(
            stack_arr,
            rej_high,
            rej_low,
            max_iter,
            mask_obj,
            skip_zero_rgb,
            channels,
            launch_sigma_clip_fused_chunk_cuda_u16);
    }
    throw std::invalid_argument(
        "sigma_clip_fused_chunk_cuda: unsupported stack dtype; expected uint8/uint16");
}

}  // namespace

void bind_sigma_clip_fused_chunk_cuda_ops(py::module_& m) {
    m.def("sigma_clip_fused_chunk_cuda",
          &sigma_clip_fused_chunk_cuda_dispatch,
          py::arg("stack"),
          py::arg("rej_high"),
          py::arg("rej_low"),
          py::arg("max_iter"),
          py::arg("mask") = py::none(),
          py::arg("skip_zero_rgb") = false,
          py::arg("channels") = static_cast<ssize_t>(1),
          "CUDA host-in/out fused sigma clipping on a chunk stack.");
}
