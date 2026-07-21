#include "fgp_internal.h"

#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

template <typename FreshT>
void huber_weighted_accumulate_inplace_kernel(py::buffer_info& weighted_sum_info,
                                              py::buffer_info& weight_total_info,
                                              const py::buffer_info& fresh_info,
                                              const py::buffer_info& ref_mean_info,
                                              const py::buffer_info& ref_std_info,
                                              const double huber_c, const double frame_weight) {
    auto* HNW_RESTRICT weighted_sum_ptr = static_cast<double*>(weighted_sum_info.ptr);
    auto* HNW_RESTRICT weight_total_ptr = static_cast<double*>(weight_total_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr = static_cast<const FreshT*>(fresh_info.ptr);
    const auto* HNW_RESTRICT ref_mean_ptr = static_cast<const float*>(ref_mean_info.ptr);
    const auto* HNW_RESTRICT ref_std_ptr = static_cast<const float*>(ref_std_info.ptr);
    const ssize_t total = fresh_info.size;
    const float huber_c_f = static_cast<float>(huber_c);

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
    HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(schedule(static))
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        const double pixel_value = static_cast<double>(fresh_ptr[i]);
        const float residual_value = static_cast<float>(fresh_ptr[i]);
        const float residual = (residual_value - ref_mean_ptr[i]) / (ref_std_ptr[i] + 1.0e-10f);
        const float abs_residual = std::fabs(residual);
        const float huber_weight =
            abs_residual <= huber_c_f ? 1.0f : huber_c_f / (abs_residual + 1.0e-10f);
        const double effective_weight = static_cast<double>(huber_weight) * frame_weight;
        weighted_sum_ptr[i] += pixel_value * effective_weight;
        weight_total_ptr[i] += effective_weight;
    }
}

template <typename FreshT>
void dispatch_huber_weighted_accumulate(
    py::array weighted_sum, py::array weight_total,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& ref_mean,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& ref_std,
    const double huber_c, const double frame_weight) {
    auto weighted_sum_t = weighted_sum.cast<hnw::MutableCArray<double>>();
    auto weight_total_t = weight_total.cast<hnw::MutableCArray<double>>();
    auto weighted_sum_info = weighted_sum_t.request();
    auto weight_total_info = weight_total_t.request();
    auto fresh_info = fresh.request();
    auto ref_mean_info = ref_mean.request();
    auto ref_std_info = ref_std.request();
    huber_weighted_accumulate_inplace_kernel<FreshT>(weighted_sum_info, weight_total_info,
                                                     fresh_info, ref_mean_info, ref_std_info,
                                                     huber_c, frame_weight);
}

void validate_huber_shapes(const py::array& weighted_sum, const py::array& weight_total,
                           const py::array& fresh, const py::array& ref_mean,
                           const py::array& ref_std) {
    hnw::require_mutable_c_array(weighted_sum, "huber_weighted_accumulate", "weighted_sum");
    hnw::require_mutable_c_array(weight_total, "huber_weighted_accumulate", "weight_total");
    if (py::str(weighted_sum.dtype()).cast<std::string>() != "float64" ||
        py::str(weight_total.dtype()).cast<std::string>() != "float64") {
        throw std::invalid_argument("huber_weighted_accumulate: accumulators must be float64");
    }
    hnw::require_same_shape(weighted_sum, fresh, "huber_weighted_accumulate");
    hnw::require_same_shape(weight_total, fresh, "huber_weighted_accumulate");
    hnw::require_same_shape(ref_mean, fresh, "huber_weighted_accumulate");
    hnw::require_same_shape(ref_std, fresh, "huber_weighted_accumulate");
}

void huber_weighted_accumulate_dispatch(py::array weighted_sum, py::array weight_total,
                                        const py::array& fresh, const py::array& ref_mean,
                                        const py::array& ref_std, double huber_c,
                                        const py::object& weight_obj) {
    validate_huber_shapes(weighted_sum, weight_total, fresh, ref_mean, ref_std);
    double frame_weight = 1.0;
    if (!weight_obj.is_none()) {
        frame_weight = py::cast<double>(weight_obj);
    }
    auto ref_mean_t =
        ref_mean.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto ref_std_t = ref_std.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        dispatch_huber_weighted_accumulate<uint8_t>(
            weighted_sum, weight_total,
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t, ref_std_t, huber_c, frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        dispatch_huber_weighted_accumulate<uint16_t>(
            weighted_sum, weight_total,
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t, ref_std_t, huber_c, frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        dispatch_huber_weighted_accumulate<uint32_t>(
            weighted_sum, weight_total,
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t, ref_std_t, huber_c, frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        dispatch_huber_weighted_accumulate<uint64_t>(
            weighted_sum, weight_total,
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t, ref_std_t, huber_c, frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        dispatch_huber_weighted_accumulate<float>(
            weighted_sum, weight_total,
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(), ref_mean_t,
            ref_std_t, huber_c, frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        dispatch_huber_weighted_accumulate<double>(
            weighted_sum, weight_total,
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t, ref_std_t, huber_c, frame_weight);
        return;
    }
    throw std::invalid_argument("huber_weighted_accumulate: unsupported fresh dtype");
}

} // namespace

void bind_huber_accumulate_ops(py::module_& m) {
    m.def("huber_weighted_accumulate", &huber_weighted_accumulate_dispatch, py::arg("weighted_sum"),
          py::arg("weight_total"), py::arg("fresh"), py::arg("ref_mean"), py::arg("ref_std"),
          py::arg("huber_c"), py::arg("weight") = py::none(),
          "Update HuberMeanParam buffers in-place with one more frame.");
}
