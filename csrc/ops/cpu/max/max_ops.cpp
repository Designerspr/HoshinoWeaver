#include "max_ops.h"
#include "common/cpu_compat.h"
#include "common/py_array_utils.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>

namespace {

template <typename T>
inline T elementwise_max(T lhs, T rhs) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::max(lhs, rhs);
    } else {
        return lhs > rhs ? lhs : rhs;
    }
}

template <typename T>
void max_combine_inplace_kernel(py::buffer_info& base_info,
                                const py::buffer_info& fresh_info) {
    auto* HNW_RESTRICT base_ptr = static_cast<T*>(base_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr = static_cast<const T*>(fresh_info.ptr);
    const ssize_t total = base_info.size;

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
    HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(schedule(static))
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        base_ptr[i] = elementwise_max(base_ptr[i], fresh_ptr[i]);
    }
}

template <typename T>
void threshold_max_merge_inplace_kernel(
    py::buffer_info& result_info,
    const py::buffer_info& frame_info,
    const py::buffer_info& mean_info,
    const py::buffer_info& std_info,
    const double n_sigma,
    const double weight) {
    auto* HNW_RESTRICT result_ptr = static_cast<T*>(result_info.ptr);
    const auto* HNW_RESTRICT frame_ptr = static_cast<const T*>(frame_info.ptr);
    const auto* HNW_RESTRICT mean_ptr = static_cast<const T*>(mean_info.ptr);
    const auto* HNW_RESTRICT std_ptr = static_cast<const T*>(std_info.ptr);
    const ssize_t total = result_info.size;
    const T n_sigma_value = static_cast<T>(n_sigma);
    const T weight_value = static_cast<T>(weight);

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
    HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(schedule(static))
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        const T frame_value = frame_ptr[i];
        const T candidate =
            frame_value > mean_ptr[i] + n_sigma_value * std_ptr[i]
                ? static_cast<T>(frame_value * weight_value)
                : mean_ptr[i];
        result_ptr[i] = elementwise_max(result_ptr[i], candidate);
    }
}

template <typename T>
py::array_t<T> max_combine_inplace_impl(
    hnw::MutableCArray<T> base,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& fresh) {
    hnw::require_same_shape(base, fresh, "max_combine");

    auto base_info = base.request();
    auto fresh_info = fresh.request();
    if (base_info.ptr == fresh_info.ptr) {
        return base;
    }
    max_combine_inplace_kernel<T>(base_info, fresh_info);
    return base;
}

template <typename T>
py::array_t<T> threshold_max_merge_inplace_impl(
    hnw::MutableCArray<T> result,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& frame,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& mean_img,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& std_img,
    const double n_sigma,
    const double weight) {
    hnw::require_same_shape(result, frame, "threshold_max_merge");
    hnw::require_same_shape(result, mean_img, "threshold_max_merge");
    hnw::require_same_shape(result, std_img, "threshold_max_merge");

    auto result_info = result.request();
    auto frame_info = frame.request();
    auto mean_info = mean_img.request();
    auto std_info = std_img.request();
    threshold_max_merge_inplace_kernel<T>(
        result_info,
        frame_info,
        mean_info,
        std_info,
        n_sigma,
        weight);
    return result;
}

py::array max_combine_dispatch(py::array base, const py::array& fresh) {
    hnw::require_mutable_c_array(base, "max_combine", "base");
    hnw::require_same_dtype(base, fresh, "max_combine");

    if (py::isinstance<py::array_t<uint8_t>>(base)) {
        return max_combine_inplace_impl<uint8_t>(
            base.cast<hnw::MutableCArray<uint8_t>>(),
            fresh.cast<py::array_t<uint8_t>>());
    }
    if (py::isinstance<py::array_t<uint16_t>>(base)) {
        return max_combine_inplace_impl<uint16_t>(
            base.cast<hnw::MutableCArray<uint16_t>>(),
            fresh.cast<py::array_t<uint16_t>>());
    }
    if (py::isinstance<py::array_t<uint32_t>>(base)) {
        return max_combine_inplace_impl<uint32_t>(
            base.cast<hnw::MutableCArray<uint32_t>>(),
            fresh.cast<py::array_t<uint32_t>>());
    }
    if (py::isinstance<py::array_t<uint64_t>>(base)) {
        return max_combine_inplace_impl<uint64_t>(
            base.cast<hnw::MutableCArray<uint64_t>>(),
            fresh.cast<py::array_t<uint64_t>>());
    }
    if (py::isinstance<py::array_t<float>>(base)) {
        return max_combine_inplace_impl<float>(
            base.cast<hnw::MutableCArray<float>>(),
            fresh.cast<py::array_t<float>>());
    }
    if (py::isinstance<py::array_t<double>>(base)) {
        return max_combine_inplace_impl<double>(
            base.cast<hnw::MutableCArray<double>>(),
            fresh.cast<py::array_t<double>>());
    }

    throw std::invalid_argument("max_combine: unsupported dtype");
}

py::array threshold_max_merge_dispatch(py::array result,
                                       const py::array& frame,
                                       const py::array& mean_img,
                                       const py::array& std_img,
                                       const double n_sigma,
                                       py::object weight_obj) {
    hnw::require_mutable_c_array(result, "threshold_max_merge", "result");
    const double weight = weight_obj.is_none() ? 1.0 : weight_obj.cast<double>();
    hnw::require_same_dtype(result, frame, "threshold_max_merge");
    hnw::require_same_dtype(result, mean_img, "threshold_max_merge");
    hnw::require_same_dtype(result, std_img, "threshold_max_merge");
    const std::string result_dtype = py::str(result.dtype()).cast<std::string>();

    if (py::isinstance<py::array_t<float>>(result)) {
        return threshold_max_merge_inplace_impl<float>(
            result.cast<hnw::MutableCArray<float>>(),
            frame.cast<py::array_t<float>>(),
            mean_img.cast<py::array_t<float>>(),
            std_img.cast<py::array_t<float>>(),
            n_sigma,
            weight);
    }
    if (py::isinstance<py::array_t<double>>(result)) {
        return threshold_max_merge_inplace_impl<double>(
            result.cast<hnw::MutableCArray<double>>(),
            frame.cast<py::array_t<double>>(),
            mean_img.cast<py::array_t<double>>(),
            std_img.cast<py::array_t<double>>(),
            n_sigma,
            weight);
    }

    throw std::invalid_argument("threshold_max_merge: unsupported dtype");
}

}  // namespace

void bind_max_ops(py::module_& m) {
    m.def("max_combine", &max_combine_dispatch, py::arg("base"), py::arg("fresh"),
          "Update base in-place with elementwise max(base, fresh) using a C++ combine kernel.");
    m.def("threshold_max_merge",
          &threshold_max_merge_dispatch,
          py::arg("result"),
          py::arg("frame"),
          py::arg("mean_img"),
          py::arg("std_img"),
          py::arg("n_sigma"),
          py::arg("weight") = py::none(),
          "Update result in-place with threshold-max candidate pixels using a C++ combine kernel.");
}
