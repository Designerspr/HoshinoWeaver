#include "fgp_internal.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <pybind11/numpy.h>

namespace {

using hnw::fgp_internal::is_pixel_zero_rgb;
using hnw::fgp_internal::validate_accumulator_shapes;
using hnw::fgp_internal::validate_masked_shapes;

template <typename SumT, typename SquareT, typename CountT>
void fgp_add_inplace_kernel(py::buffer_info& sum_info,
                            py::buffer_info& square_info,
                            py::buffer_info& count_info,
                            const py::buffer_info& other_sum_info,
                            const py::buffer_info& other_square_info,
                            const py::buffer_info& other_count_info) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT other_sum_ptr =
        static_cast<const SumT*>(other_sum_info.ptr);
    const auto* HNW_RESTRICT other_square_ptr =
        static_cast<const SquareT*>(other_square_info.ptr);
    const auto* HNW_RESTRICT other_count_ptr =
        static_cast<const CountT*>(other_count_info.ptr);
    const ssize_t total = sum_info.size;

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
    HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(schedule(static))
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        sum_ptr[i] = static_cast<SumT>(sum_ptr[i] + other_sum_ptr[i]);
        square_ptr[i] =
            static_cast<SquareT>(square_ptr[i] + other_square_ptr[i]);
        count_ptr[i] = static_cast<CountT>(count_ptr[i] + other_count_ptr[i]);
    }
}

template <typename FreshT, typename SumT, typename SquareT, typename CountT>
void fgp_masked_mean_inplace_kernel(py::buffer_info& sum_info,
                                    py::buffer_info& square_info,
                                    py::buffer_info& count_info,
                                    const py::buffer_info& fresh_info,
                                    const py::buffer_info& mask_info,
                                    const bool skip_zero_rgb) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr =
        static_cast<const FreshT*>(fresh_info.ptr);
    const auto* HNW_RESTRICT mask_ptr = static_cast<const uint8_t*>(mask_info.ptr);
    const ssize_t spatial = mask_info.size;
    const ssize_t channels = spatial == 0 ? 0 : fresh_info.size / spatial;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t idx = 0; idx < spatial; ++idx) {
        if (mask_ptr[idx] == 0) {
            continue;
        }
        const ssize_t base = idx * channels;
        if (skip_zero_rgb && channels >= 3 &&
            is_pixel_zero_rgb(fresh_ptr, base, channels)) {
            continue;
        }
#if defined(HNW_ENABLE_OMP_SIMD) && HNW_ENABLE_OMP_SIMD
#pragma omp simd
#endif
        for (ssize_t c = 0; c < channels; ++c) {
            const ssize_t offset = base + c;
            const SumT value_sum = static_cast<SumT>(fresh_ptr[offset]);
            const SquareT value_square = static_cast<SquareT>(fresh_ptr[offset]);
            sum_ptr[offset] = static_cast<SumT>(sum_ptr[offset] + value_sum);
            square_ptr[offset] = static_cast<SquareT>(
                square_ptr[offset] + value_square * value_square);
            count_ptr[offset] = static_cast<CountT>(count_ptr[offset] + 1);
        }
    }
}

template <typename SumT, typename SquareT>
void dispatch_add_count_dtype(
    hnw::MutableCArray<SumT> sum_mu,
    hnw::MutableCArray<SquareT> square_sum,
    py::array n,
    const py::array& other_sum_mu,
    const py::array& other_square_sum,
    const py::array& other_n) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint16_t>>();
        auto other_count =
            other_n.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        auto other_sum =
            other_sum_mu.cast<py::array_t<SumT, py::array::c_style | py::array::forcecast>>();
        auto other_square = other_square_sum.cast<
            py::array_t<SquareT, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto other_sum_info = other_sum.request();
        auto other_square_info = other_square.request();
        auto other_count_info = other_count.request();
        fgp_add_inplace_kernel<SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, other_sum_info, other_square_info, other_count_info);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint32_t>>();
        auto other_count =
            other_n.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        auto other_sum =
            other_sum_mu.cast<py::array_t<SumT, py::array::c_style | py::array::forcecast>>();
        auto other_square = other_square_sum.cast<
            py::array_t<SquareT, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto other_sum_info = other_sum.request();
        auto other_square_info = other_square.request();
        auto other_count_info = other_count.request();
        fgp_add_inplace_kernel<SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, other_sum_info, other_square_info, other_count_info);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint64_t>>();
        auto other_count =
            other_n.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        auto other_sum =
            other_sum_mu.cast<py::array_t<SumT, py::array::c_style | py::array::forcecast>>();
        auto other_square = other_square_sum.cast<
            py::array_t<SquareT, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto other_sum_info = other_sum.request();
        auto other_square_info = other_square.request();
        auto other_count_info = other_count.request();
        fgp_add_inplace_kernel<SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, other_sum_info, other_square_info, other_count_info);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<hnw::MutableCArray<double>>();
        auto other_count =
            other_n.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto other_sum =
            other_sum_mu.cast<py::array_t<SumT, py::array::c_style | py::array::forcecast>>();
        auto other_square = other_square_sum.cast<
            py::array_t<SquareT, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto other_sum_info = other_sum.request();
        auto other_square_info = other_square.request();
        auto other_count_info = other_count.request();
        fgp_add_inplace_kernel<SumT, SquareT, double>(
            sum_info, square_info, count_info, other_sum_info, other_square_info, other_count_info);
        return;
    }
    throw std::invalid_argument("fgp_add: unsupported n dtype");
}

template <typename SumT>
void dispatch_add_square_dtype(
    hnw::MutableCArray<SumT> sum_mu,
    py::array square_sum,
    py::array n,
    const py::array& other_sum_mu,
    const py::array& other_square_sum,
    const py::array& other_n) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_add_count_dtype<SumT, uint32_t>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<uint32_t>>(),
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_add_count_dtype<SumT, uint64_t>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<uint64_t>>(),
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_add_count_dtype<SumT, double>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<double>>(),
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    throw std::invalid_argument("fgp_add: unsupported square_sum dtype");
}

void dispatch_add_sum_dtype(py::array sum_mu,
                            py::array square_sum,
                            py::array n,
                            const py::array& other_sum_mu,
                            const py::array& other_square_sum,
                            const py::array& other_n) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_add_square_dtype<uint16_t>(
            sum_mu.cast<hnw::MutableCArray<uint16_t>>(),
            square_sum,
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_add_square_dtype<uint32_t>(
            sum_mu.cast<hnw::MutableCArray<uint32_t>>(),
            square_sum,
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_add_square_dtype<uint64_t>(
            sum_mu.cast<hnw::MutableCArray<uint64_t>>(),
            square_sum,
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_add_square_dtype<double>(
            sum_mu.cast<hnw::MutableCArray<double>>(),
            square_sum,
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    throw std::invalid_argument("fgp_add: unsupported sum_mu dtype");
}

template <typename FreshT, typename SumT, typename SquareT>
void dispatch_masked_mean_count_dtype(
    hnw::MutableCArray<SumT> sum_mu,
    hnw::MutableCArray<SquareT> square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    const bool skip_zero_rgb) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint16_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto mask_info = mask.request();
        fgp_masked_mean_inplace_kernel<FreshT, SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, fresh_info, mask_info,
            skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint32_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto mask_info = mask.request();
        fgp_masked_mean_inplace_kernel<FreshT, SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, fresh_info, mask_info,
            skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint64_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto mask_info = mask.request();
        fgp_masked_mean_inplace_kernel<FreshT, SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, fresh_info, mask_info,
            skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<hnw::MutableCArray<double>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto mask_info = mask.request();
        fgp_masked_mean_inplace_kernel<FreshT, SumT, SquareT, double>(
            sum_info, square_info, count_info, fresh_info, mask_info,
            skip_zero_rgb);
        return;
    }
    throw std::invalid_argument("fgp_masked_mean_merge: unsupported n dtype");
}

template <typename FreshT, typename SumT>
void dispatch_masked_mean_square_dtype(
    hnw::MutableCArray<SumT> sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    const bool skip_zero_rgb) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_masked_mean_count_dtype<FreshT, SumT, uint32_t>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<uint32_t>>(),
            n,
            fresh,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_masked_mean_count_dtype<FreshT, SumT, uint64_t>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<uint64_t>>(),
            n,
            fresh,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_masked_mean_count_dtype<FreshT, SumT, double>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<double>>(),
            n,
            fresh,
            mask, skip_zero_rgb);
        return;
    }
    throw std::invalid_argument("fgp_masked_mean_merge: unsupported square_sum dtype");
}

template <typename FreshT>
void dispatch_masked_mean_sum_dtype(
    py::array sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    const bool skip_zero_rgb) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_masked_mean_square_dtype<FreshT, uint16_t>(
            sum_mu.cast<hnw::MutableCArray<uint16_t>>(),
            square_sum,
            n,
            fresh,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_masked_mean_square_dtype<FreshT, uint32_t>(
            sum_mu.cast<hnw::MutableCArray<uint32_t>>(),
            square_sum,
            n,
            fresh,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_masked_mean_square_dtype<FreshT, uint64_t>(
            sum_mu.cast<hnw::MutableCArray<uint64_t>>(),
            square_sum,
            n,
            fresh,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_masked_mean_square_dtype<FreshT, double>(
            sum_mu.cast<hnw::MutableCArray<double>>(),
            square_sum,
            n,
            fresh,
            mask, skip_zero_rgb);
        return;
    }
    throw std::invalid_argument("fgp_masked_mean_merge: unsupported sum_mu dtype");
}

void validate_add_shapes(const py::array& sum_mu,
                         const py::array& square_sum,
                         const py::array& n,
                         const py::array& other_sum_mu,
                         const py::array& other_square_sum,
                         const py::array& other_n) {
    validate_accumulator_shapes(sum_mu, square_sum, n, other_sum_mu, "fgp_add");
    hnw::require_same_shape(other_square_sum, other_sum_mu, "fgp_add");
    hnw::require_same_shape(other_n, other_sum_mu, "fgp_add");
}

void fgp_add_dispatch(py::array sum_mu,
                      py::array square_sum,
                      py::array n,
                      const py::array& other_sum_mu,
                      const py::array& other_square_sum,
                      const py::array& other_n) {
    validate_add_shapes(sum_mu, square_sum, n, other_sum_mu, other_square_sum,
                        other_n);
    dispatch_add_sum_dtype(
        sum_mu, square_sum, n, other_sum_mu, other_square_sum, other_n);
}

void fgp_masked_mean_dispatch(py::array sum_mu,
                              py::array square_sum,
                              py::array n,
                              const py::array& fresh,
                              const py::array& mask,
                              bool skip_zero_rgb) {
    validate_masked_shapes(sum_mu, square_sum, n, fresh, mask,
                           "fgp_masked_mean_merge");
    auto mask_t =
        mask.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        dispatch_masked_mean_sum_dtype<uint8_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        dispatch_masked_mean_sum_dtype<uint16_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        dispatch_masked_mean_sum_dtype<uint32_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        dispatch_masked_mean_sum_dtype<uint64_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        dispatch_masked_mean_sum_dtype<float>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        dispatch_masked_mean_sum_dtype<double>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    throw std::invalid_argument("fgp_masked_mean_merge: unsupported fresh dtype");
}

}  // namespace

void bind_fgp_merge_ops(py::module_& m) {
    m.def(
        "fgp_add",
        &fgp_add_dispatch,
        py::arg("sum_mu"),
        py::arg("square_sum"),
        py::arg("n"),
        py::arg("other_sum_mu"),
        py::arg("other_square_sum"),
        py::arg("other_n"),
        "Update FastGaussianParam buffers in-place with another FastGaussianParam.");
    m.def(
        "fgp_masked_mean_merge",
        &fgp_masked_mean_dispatch,
        py::arg("sum_mu"),
        py::arg("square_sum"),
        py::arg("n"),
        py::arg("fresh"),
        py::arg("mask"),
        py::arg("skip_zero_rgb") = false,
        "Update FastGaussianParam buffers in-place for spatial-mask mean merge.");
}
