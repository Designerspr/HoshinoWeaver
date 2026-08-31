#include "huber_weighted_chunk_ops.h"

#include "common/compat.h"

#include <pybind11/numpy.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

void launch_huber_weighted_chunk_cuda_u8(const uint8_t* stack_host, const double* ref_mean_host,
                                         const double* ref_std_host, const double* weights_host,
                                         double* weighted_sum_host, double* weight_total_host,
                                         int64_t n_frames, int64_t plane_size, double huber_c);

void launch_huber_weighted_chunk_cuda_u16(const uint16_t* stack_host, const double* ref_mean_host,
                                          const double* ref_std_host, const double* weights_host,
                                          double* weighted_sum_host, double* weight_total_host,
                                          int64_t n_frames, int64_t plane_size, double huber_c);

namespace {

void validate_shape(const py::array& stack, const py::array_t<double>& ref_mean,
                    const py::array_t<double>& ref_std, const py::object& weights_obj) {
    if (stack.ndim() != 2) {
        throw std::invalid_argument(
            "huber_weighted_chunk_cuda: stack must be 2D (n_frames, plane_size)");
    }
    if (stack.shape(0) <= 0 || stack.shape(1) <= 0) {
        throw std::invalid_argument("huber_weighted_chunk_cuda: stack dimensions must be positive");
    }
    if (stack.shape(0) > std::numeric_limits<int>::max() ||
        stack.shape(1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("huber_weighted_chunk_cuda: stack is too large");
    }
    if (ref_mean.ndim() != 1 || ref_std.ndim() != 1 || ref_mean.shape(0) != stack.shape(1) ||
        ref_std.shape(0) != stack.shape(1)) {
        throw std::invalid_argument(
            "huber_weighted_chunk_cuda: ref_mean/ref_std must match plane_size");
    }
    if (!weights_obj.is_none()) {
        auto weights =
            weights_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (weights.ndim() != 1 || weights.shape(0) != stack.shape(0)) {
            throw std::invalid_argument(
                "huber_weighted_chunk_cuda: weights must have shape (n_frames,)");
        }
    }
}

py::array_t<double, py::array::c_style | py::array::forcecast>
parse_weights(const py::object& weights_obj, const double** weights_ptr) {
    *weights_ptr = nullptr;
    py::array_t<double, py::array::c_style | py::array::forcecast> weights_arr;
    if (weights_obj.is_none()) {
        return weights_arr;
    }
    weights_arr =
        weights_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    *weights_ptr = weights_arr.data();
    return weights_arr;
}

template <typename T, typename Launcher>
py::tuple huber_weighted_chunk_cuda_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& stack,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ref_mean,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ref_std,
    const double huber_c, const py::object& weights_obj, Launcher launcher) {
    validate_shape(stack, ref_mean, ref_std, weights_obj);

    const ssize_t n_frames = stack.shape(0);
    const ssize_t plane_size = stack.shape(1);
    const double* weights_ptr = nullptr;
    auto weights_arr = parse_weights(weights_obj, &weights_ptr);

    py::array_t<double> weighted_sum({plane_size});
    py::array_t<double> weight_total({plane_size});

    {
        py::gil_scoped_release release;
        launcher(stack.data(), ref_mean.data(), ref_std.data(), weights_ptr,
                 weighted_sum.mutable_data(), weight_total.mutable_data(), n_frames, plane_size,
                 huber_c);
    }
    return py::make_tuple(weighted_sum, weight_total);
}

py::tuple huber_weighted_chunk_cuda_dispatch(
    const py::array& stack,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ref_mean,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ref_std,
    const double huber_c, const py::object& weights_obj) {
    if (py::isinstance<py::array_t<uint8_t>>(stack)) {
        auto stack_arr =
            stack.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        return huber_weighted_chunk_cuda_impl(stack_arr, ref_mean, ref_std, huber_c, weights_obj,
                                              launch_huber_weighted_chunk_cuda_u8);
    }
    if (py::isinstance<py::array_t<uint16_t>>(stack)) {
        auto stack_arr =
            stack.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        return huber_weighted_chunk_cuda_impl(stack_arr, ref_mean, ref_std, huber_c, weights_obj,
                                              launch_huber_weighted_chunk_cuda_u16);
    }
    throw std::invalid_argument(
        "huber_weighted_chunk_cuda: unsupported stack dtype; expected uint8/uint16");
}

} // namespace

void bind_huber_weighted_chunk_cuda_ops(py::module_& m) {
    m.def("huber_weighted_chunk_cuda", &huber_weighted_chunk_cuda_dispatch, py::arg("stack"),
          py::arg("ref_mean"), py::arg("ref_std"), py::arg("huber_c"),
          py::arg("weights") = py::none(), "CUDA host-in/out Huber weighted chunk accumulation.");
}
