#include "fgp_internal.h"

#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using hnw::fgp_internal::is_pixel_zero_rgb;
using hnw::fgp_internal::validate_accumulator_shapes;

template <typename FreshT, typename SumT, typename SquareT, typename CountT>
void fgp_accumulate_inplace_kernel(py::buffer_info& sum_info, py::buffer_info& square_info,
                                   py::buffer_info& count_info, const py::buffer_info& fresh_info,
                                   const uint64_t weight, const bool skip_zero_rgb,
                                   const ssize_t channels) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr = static_cast<const FreshT*>(fresh_info.ptr);
    const ssize_t total = fresh_info.size;

    using SumAccumT = SumT;
    using SquareAccumT = SquareT;
    using CountAccumT = CountT;
    const SumAccumT weight_sum = static_cast<SumAccumT>(weight);
    const SquareAccumT weight_square = static_cast<SquareAccumT>(weight);
    const CountAccumT weight_count = static_cast<CountAccumT>(weight);

    py::gil_scoped_release release;

    if (!skip_zero_rgb || channels < 3) {
        // Original flat loop — no zero-pixel detection needed
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
        HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(schedule(static))
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t i = 0; i < total; ++i) {
            const SumAccumT value_sum = static_cast<SumAccumT>(fresh_ptr[i]);
            const SquareAccumT value_square = static_cast<SquareAccumT>(fresh_ptr[i]);
            sum_ptr[i] = static_cast<SumT>(static_cast<SumAccumT>(sum_ptr[i]) +
                                           static_cast<SumAccumT>(value_sum * weight_sum));
            square_ptr[i] = static_cast<SquareT>(
                static_cast<SquareAccumT>(square_ptr[i]) +
                static_cast<SquareAccumT>(value_square * value_square * weight_square));
            count_ptr[i] =
                static_cast<CountT>(static_cast<CountAccumT>(count_ptr[i]) + weight_count);
        }
    } else {
        // Spatial+channel loop with per-pixel zero detection
        const ssize_t spatial = total / channels;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t idx = 0; idx < spatial; ++idx) {
            const ssize_t base = idx * channels;
            if (is_pixel_zero_rgb(fresh_ptr, base, channels))
                continue;
#if defined(HNW_ENABLE_OMP_SIMD) && HNW_ENABLE_OMP_SIMD
#pragma omp simd
#endif
            for (ssize_t c = 0; c < channels; ++c) {
                const ssize_t offset = base + c;
                const SumAccumT value_sum = static_cast<SumAccumT>(fresh_ptr[offset]);
                const SquareAccumT value_square = static_cast<SquareAccumT>(fresh_ptr[offset]);
                sum_ptr[offset] = static_cast<SumT>(static_cast<SumAccumT>(sum_ptr[offset]) +
                                                    static_cast<SumAccumT>(value_sum * weight_sum));
                square_ptr[offset] = static_cast<SquareT>(
                    static_cast<SquareAccumT>(square_ptr[offset]) +
                    static_cast<SquareAccumT>(value_square * value_square * weight_square));
                count_ptr[offset] =
                    static_cast<CountT>(static_cast<CountAccumT>(count_ptr[offset]) + weight_count);
            }
        }
    }
}
template <typename FreshT, typename SumT, typename SquareT>
void dispatch_accumulate_count_dtype(
    hnw::MutableCArray<SumT> sum_mu, hnw::MutableCArray<SquareT> square_sum, py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const uint64_t weight, const bool skip_zero_rgb, const ssize_t channels) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint16_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        fgp_accumulate_inplace_kernel<FreshT, SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, fresh_info, weight, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint32_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        fgp_accumulate_inplace_kernel<FreshT, SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, fresh_info, weight, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint64_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        fgp_accumulate_inplace_kernel<FreshT, SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, fresh_info, weight, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<hnw::MutableCArray<double>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        fgp_accumulate_inplace_kernel<FreshT, SumT, SquareT, double>(
            sum_info, square_info, count_info, fresh_info, weight, skip_zero_rgb, channels);
        return;
    }
    throw std::invalid_argument("fgp_accumulate: unsupported n dtype");
}

template <typename FreshT, typename SumT>
void dispatch_accumulate_square_dtype(
    hnw::MutableCArray<SumT> sum_mu, py::array square_sum, py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const uint64_t weight, const bool skip_zero_rgb, const ssize_t channels) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_accumulate_count_dtype<FreshT, SumT, uint32_t>(
            sum_mu, square_sum.cast<hnw::MutableCArray<uint32_t>>(), n, fresh, weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_accumulate_count_dtype<FreshT, SumT, uint64_t>(
            sum_mu, square_sum.cast<hnw::MutableCArray<uint64_t>>(), n, fresh, weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_accumulate_count_dtype<FreshT, SumT, double>(
            sum_mu, square_sum.cast<hnw::MutableCArray<double>>(), n, fresh, weight, skip_zero_rgb,
            channels);
        return;
    }
    throw std::invalid_argument("fgp_accumulate: unsupported square_sum dtype");
}

template <typename FreshT>
void dispatch_accumulate_sum_dtype(
    py::array sum_mu, py::array square_sum, py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const uint64_t weight, const bool skip_zero_rgb, const ssize_t channels) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_accumulate_square_dtype<FreshT, uint16_t>(
            sum_mu.cast<hnw::MutableCArray<uint16_t>>(), square_sum, n, fresh, weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_accumulate_square_dtype<FreshT, uint32_t>(
            sum_mu.cast<hnw::MutableCArray<uint32_t>>(), square_sum, n, fresh, weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_accumulate_square_dtype<FreshT, uint64_t>(
            sum_mu.cast<hnw::MutableCArray<uint64_t>>(), square_sum, n, fresh, weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_accumulate_square_dtype<FreshT, double>(sum_mu.cast<hnw::MutableCArray<double>>(),
                                                         square_sum, n, fresh, weight,
                                                         skip_zero_rgb, channels);
        return;
    }
    throw std::invalid_argument("fgp_accumulate: unsupported sum_mu dtype");
}

void validate_accumulate_shapes(const py::array& sum_mu, const py::array& square_sum,
                                const py::array& n, const py::array& fresh) {
    validate_accumulator_shapes(sum_mu, square_sum, n, fresh, "fgp_accumulate");
}

void fgp_accumulate_dispatch(py::array sum_mu, py::array square_sum, py::array n,
                             const py::array& fresh, const py::object& weight_obj,
                             bool skip_zero_rgb) {
    validate_accumulate_shapes(sum_mu, square_sum, n, fresh);
    uint64_t weight = 1;
    if (!weight_obj.is_none()) {
        weight = py::cast<uint64_t>(weight_obj);
    }
    const ssize_t channels = (fresh.ndim() >= 3) ? fresh.shape(fresh.ndim() - 1) : 1;

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        dispatch_accumulate_sum_dtype<uint8_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        dispatch_accumulate_sum_dtype<uint16_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(), weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        dispatch_accumulate_sum_dtype<uint32_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(), weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        dispatch_accumulate_sum_dtype<uint64_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(), weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        dispatch_accumulate_sum_dtype<float>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(), weight,
            skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        dispatch_accumulate_sum_dtype<double>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(), weight,
            skip_zero_rgb, channels);
        return;
    }
    throw std::invalid_argument("fgp_accumulate: unsupported fresh dtype");
}

} // namespace

void bind_fgp_accumulate_ops(py::module_& m) {
    m.def("fgp_accumulate", &fgp_accumulate_dispatch, py::arg("sum_mu"), py::arg("square_sum"),
          py::arg("n"), py::arg("fresh"), py::arg("weight") = py::none(),
          py::arg("skip_zero_rgb") = false,
          "Update FastGaussianParam buffers in-place with one more frame.");
}
