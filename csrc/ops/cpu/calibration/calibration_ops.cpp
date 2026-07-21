#include "calibration_ops.h"

#include "common/cpu_compat.h"
#include "common/py_array_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <pybind11/numpy.h>

namespace {

template <typename T>
constexpr double max_value_for() {
    if constexpr (std::is_floating_point_v<T>) {
        return 0.0;
    } else {
        return static_cast<double>(std::numeric_limits<T>::max());
    }
}

template <typename T>
T cast_divide_output(const double value) {
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(value);
    } else {
        const double clipped =
            std::clamp(value, 0.0, static_cast<double>(std::numeric_limits<T>::max()));
        return static_cast<T>(clipped);
    }
}

template <typename T>
void calibration_subtract_kernel(py::buffer_info& out_info,
                                 const py::buffer_info& frame_info,
                                 const py::buffer_info& ref_info) {
    auto* HNW_RESTRICT out_ptr = static_cast<T*>(out_info.ptr);
    const auto* HNW_RESTRICT frame_ptr = static_cast<const T*>(frame_info.ptr);
    const auto* HNW_RESTRICT ref_ptr = static_cast<const T*>(ref_info.ptr);
    const ssize_t total = out_info.size;

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
    HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(schedule(static))
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t idx = 0; idx < total; ++idx) {
        if constexpr (std::is_floating_point_v<T>) {
            out_ptr[idx] = std::max(frame_ptr[idx] - ref_ptr[idx], static_cast<T>(0));
        } else {
            const int64_t diff =
                static_cast<int64_t>(frame_ptr[idx]) - static_cast<int64_t>(ref_ptr[idx]);
            const int64_t clipped =
                std::clamp<int64_t>(diff, 0, static_cast<int64_t>(std::numeric_limits<T>::max()));
            out_ptr[idx] = static_cast<T>(clipped);
        }
    }
}

template <typename T>
double mean_as_float64(const py::buffer_info& ref_info) {
    const auto* HNW_RESTRICT ref_ptr = static_cast<const T*>(ref_info.ptr);
    const ssize_t total = ref_info.size;
    double sum = 0.0;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : sum) schedule(static)
#endif
    for (ssize_t idx = 0; idx < total; ++idx) {
        sum += static_cast<double>(ref_ptr[idx]);
    }
    return sum / static_cast<double>(total);
}

template <typename T>
void calibration_divide_kernel(py::buffer_info& out_info,
                               const py::buffer_info& frame_info,
                               const py::buffer_info& ref_info,
                               const double ref_mean) {
    auto* HNW_RESTRICT out_ptr = static_cast<T*>(out_info.ptr);
    const auto* HNW_RESTRICT frame_ptr = static_cast<const T*>(frame_info.ptr);
    const auto* HNW_RESTRICT ref_ptr = static_cast<const T*>(ref_info.ptr);
    const ssize_t total = out_info.size;
    const double max_value = max_value_for<T>();

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
    HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(schedule(static))
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t idx = 0; idx < total; ++idx) {
        const double ref_value = static_cast<double>(ref_ptr[idx]);
        const double ref_safe = ref_value > 0.0 ? ref_value : 1.0;
        double value = static_cast<double>(frame_ptr[idx]) / ref_safe * ref_mean;
        if constexpr (!std::is_floating_point_v<T>) {
            value = std::clamp(value, 0.0, max_value);
        }
        out_ptr[idx] = cast_divide_output<T>(value);
    }
}

template <typename T>
py::array_t<T> calibration_subtract_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& frame,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& reference) {
    hnw::require_same_shape(frame, reference, "calibration_subtract");
    auto frame_info = frame.request();
    auto ref_info = reference.request();
    py::array_t<T> out(frame_info.shape);
    auto out_info = out.request();
    calibration_subtract_kernel<T>(out_info, frame_info, ref_info);
    return out;
}

template <typename T>
py::array_t<T> calibration_divide_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& frame,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& reference) {
    hnw::require_same_shape(frame, reference, "calibration_divide");
    auto frame_info = frame.request();
    auto ref_info = reference.request();
    py::array_t<T> out(frame_info.shape);
    auto out_info = out.request();
    const double ref_mean = mean_as_float64<T>(ref_info);
    calibration_divide_kernel<T>(out_info, frame_info, ref_info, ref_mean);
    return out;
}

py::array calibration_subtract_dispatch(const py::array& frame,
                                        const py::array& reference) {
    hnw::require_same_dtype(frame, reference, "calibration_subtract");
    const std::string frame_dtype = py::str(frame.dtype()).cast<std::string>();

    if (py::isinstance<py::array_t<unsigned char>>(frame)) {
        return calibration_subtract_impl<unsigned char>(
            frame.cast<py::array_t<unsigned char>>(),
            reference.cast<py::array_t<unsigned char>>());
    }
    if (py::isinstance<py::array_t<unsigned short>>(frame)) {
        return calibration_subtract_impl<unsigned short>(
            frame.cast<py::array_t<unsigned short>>(),
            reference.cast<py::array_t<unsigned short>>());
    }
    if (py::isinstance<py::array_t<uint32_t>>(frame)) {
        return calibration_subtract_impl<uint32_t>(
            frame.cast<py::array_t<uint32_t>>(),
            reference.cast<py::array_t<uint32_t>>());
    }
    if (py::isinstance<py::array_t<float>>(frame)) {
        return calibration_subtract_impl<float>(
            frame.cast<py::array_t<float>>(),
            reference.cast<py::array_t<float>>());
    }
    if (py::isinstance<py::array_t<double>>(frame)) {
        return calibration_subtract_impl<double>(
            frame.cast<py::array_t<double>>(),
            reference.cast<py::array_t<double>>());
    }

    throw std::invalid_argument("calibration_subtract: unsupported dtype");
}

py::array calibration_divide_dispatch(const py::array& frame,
                                      const py::array& reference) {
    hnw::require_same_dtype(frame, reference, "calibration_divide");
    const std::string frame_dtype = py::str(frame.dtype()).cast<std::string>();

    if (py::isinstance<py::array_t<unsigned char>>(frame)) {
        return calibration_divide_impl<unsigned char>(
            frame.cast<py::array_t<unsigned char>>(),
            reference.cast<py::array_t<unsigned char>>());
    }
    if (py::isinstance<py::array_t<unsigned short>>(frame)) {
        return calibration_divide_impl<unsigned short>(
            frame.cast<py::array_t<unsigned short>>(),
            reference.cast<py::array_t<unsigned short>>());
    }
    if (py::isinstance<py::array_t<uint32_t>>(frame)) {
        return calibration_divide_impl<uint32_t>(
            frame.cast<py::array_t<uint32_t>>(),
            reference.cast<py::array_t<uint32_t>>());
    }
    if (py::isinstance<py::array_t<float>>(frame)) {
        return calibration_divide_impl<float>(
            frame.cast<py::array_t<float>>(),
            reference.cast<py::array_t<float>>());
    }
    if (py::isinstance<py::array_t<double>>(frame)) {
        return calibration_divide_impl<double>(
            frame.cast<py::array_t<double>>(),
            reference.cast<py::array_t<double>>());
    }

    throw std::invalid_argument("calibration_divide: unsupported dtype");
}

}  // namespace

void bind_calibration_ops(py::module_& m) {
    m.def("calibration_subtract",
          &calibration_subtract_dispatch,
          py::arg("frame"),
          py::arg("reference"),
          "Apply calibration subtraction with dtype-preserving clip.");
    m.def("calibration_divide",
          &calibration_divide_dispatch,
          py::arg("frame"),
          py::arg("reference"),
          "Apply flat-field calibration division with dtype-preserving clip.");
}
