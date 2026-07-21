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

template <typename FreshT, typename SumT, typename SquareT, typename CountT>
void sigma_clip_fused_inplace_kernel(py::buffer_info& sum_info,
                                     py::buffer_info& square_info,
                                     py::buffer_info& count_info,
                                     const py::buffer_info& fresh_info,
                                     const py::buffer_info& rej_high_info,
                                     const py::buffer_info& rej_low_info,
                                     const bool skip_zero_rgb,
                                     const ssize_t channels) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr =
        static_cast<const FreshT*>(fresh_info.ptr);
    const auto* HNW_RESTRICT rej_high_ptr =
        static_cast<const FreshT*>(rej_high_info.ptr);
    const auto* HNW_RESTRICT rej_low_ptr =
        static_cast<const FreshT*>(rej_low_info.ptr);
    const ssize_t total = fresh_info.size;

    py::gil_scoped_release release;

    if (!skip_zero_rgb || channels < 3) {
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
        HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(schedule(static))
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t i = 0; i < total; ++i) {
            const FreshT value = fresh_ptr[i];
            if (value < rej_low_ptr[i] || value > rej_high_ptr[i]) {
                const SumT value_sum = static_cast<SumT>(value);
                const SquareT value_square = static_cast<SquareT>(value);
                sum_ptr[i] = static_cast<SumT>(sum_ptr[i] + value_sum);
                square_ptr[i] = static_cast<SquareT>(
                    square_ptr[i] + value_square * value_square);
                count_ptr[i] = static_cast<CountT>(count_ptr[i] + 1);
            }
        }
    } else {
        const ssize_t spatial = total / channels;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t idx = 0; idx < spatial; ++idx) {
            const ssize_t base = idx * channels;
            if (is_pixel_zero_rgb(fresh_ptr, base, channels)) continue;
#if defined(HNW_ENABLE_OMP_SIMD) && HNW_ENABLE_OMP_SIMD
#pragma omp simd
#endif
            for (ssize_t c = 0; c < channels; ++c) {
                const ssize_t offset = base + c;
                const FreshT value = fresh_ptr[offset];
                if (value < rej_low_ptr[offset] || value > rej_high_ptr[offset]) {
                    const SumT value_sum = static_cast<SumT>(value);
                    const SquareT value_square = static_cast<SquareT>(value);
                    sum_ptr[offset] = static_cast<SumT>(sum_ptr[offset] + value_sum);
                    square_ptr[offset] = static_cast<SquareT>(
                        square_ptr[offset] + value_square * value_square);
                    count_ptr[offset] = static_cast<CountT>(count_ptr[offset] + 1);
                }
            }
        }
    }
}

template <typename FreshT, typename SumT, typename SquareT, typename CountT>
void sigma_clip_fused_masked_inplace_kernel(py::buffer_info& sum_info,
                                            py::buffer_info& square_info,
                                            py::buffer_info& count_info,
                                            const py::buffer_info& fresh_info,
                                            const py::buffer_info& rej_high_info,
                                            const py::buffer_info& rej_low_info,
                                            const py::buffer_info& mask_info,
                                            const bool skip_zero_rgb) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr =
        static_cast<const FreshT*>(fresh_info.ptr);
    const auto* HNW_RESTRICT rej_high_ptr =
        static_cast<const FreshT*>(rej_high_info.ptr);
    const auto* HNW_RESTRICT rej_low_ptr =
        static_cast<const FreshT*>(rej_low_info.ptr);
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
            const FreshT value = fresh_ptr[offset];
            if (value < rej_low_ptr[offset] || value > rej_high_ptr[offset]) {
                const SumT value_sum = static_cast<SumT>(value);
                const SquareT value_square = static_cast<SquareT>(value);
                sum_ptr[offset] = static_cast<SumT>(sum_ptr[offset] + value_sum);
                square_ptr[offset] = static_cast<SquareT>(
                    square_ptr[offset] + value_square * value_square);
                count_ptr[offset] = static_cast<CountT>(count_ptr[offset] + 1);
            }
        }
    }
}

template <typename FreshT, typename SumT, typename SquareT>
void dispatch_sigma_clip_count_dtype(
    hnw::MutableCArray<SumT> sum_mu,
    hnw::MutableCArray<SquareT> square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low,
    const bool skip_zero_rgb,
    const ssize_t channels) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint16_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        sigma_clip_fused_inplace_kernel<FreshT, SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint32_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        sigma_clip_fused_inplace_kernel<FreshT, SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint64_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        sigma_clip_fused_inplace_kernel<FreshT, SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<hnw::MutableCArray<double>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        sigma_clip_fused_inplace_kernel<FreshT, SumT, SquareT, double>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, skip_zero_rgb, channels);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_merge: unsupported n dtype");
}

template <typename FreshT, typename SumT>
void dispatch_sigma_clip_square_dtype(
    hnw::MutableCArray<SumT> sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low,
    const bool skip_zero_rgb,
    const ssize_t channels) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_sigma_clip_count_dtype<FreshT, SumT, uint32_t>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<uint32_t>>(),
            n,
            fresh,
            rej_high,
            rej_low, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_sigma_clip_count_dtype<FreshT, SumT, uint64_t>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<uint64_t>>(),
            n,
            fresh,
            rej_high,
            rej_low, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_sigma_clip_count_dtype<FreshT, SumT, double>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<double>>(),
            n,
            fresh,
            rej_high,
            rej_low, skip_zero_rgb, channels);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_merge: unsupported square_sum dtype");
}

template <typename FreshT>
void dispatch_sigma_clip_sum_dtype(
    py::array sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low,
    const bool skip_zero_rgb,
    const ssize_t channels) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_sigma_clip_square_dtype<FreshT, uint16_t>(
            sum_mu.cast<hnw::MutableCArray<uint16_t>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_sigma_clip_square_dtype<FreshT, uint32_t>(
            sum_mu.cast<hnw::MutableCArray<uint32_t>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_sigma_clip_square_dtype<FreshT, uint64_t>(
            sum_mu.cast<hnw::MutableCArray<uint64_t>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low, skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_sigma_clip_square_dtype<FreshT, double>(
            sum_mu.cast<hnw::MutableCArray<double>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low, skip_zero_rgb, channels);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_merge: unsupported sum_mu dtype");
}

template <typename FreshT, typename SumT, typename SquareT>
void dispatch_sigma_clip_masked_count_dtype(
    hnw::MutableCArray<SumT> sum_mu,
    hnw::MutableCArray<SquareT> square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    const bool skip_zero_rgb) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint16_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        auto mask_info = mask.request();
        sigma_clip_fused_masked_inplace_kernel<FreshT, SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, mask_info, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint32_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        auto mask_info = mask.request();
        sigma_clip_fused_masked_inplace_kernel<FreshT, SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, mask_info, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<hnw::MutableCArray<uint64_t>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        auto mask_info = mask.request();
        sigma_clip_fused_masked_inplace_kernel<FreshT, SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, mask_info, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<hnw::MutableCArray<double>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        auto mask_info = mask.request();
        sigma_clip_fused_masked_inplace_kernel<FreshT, SumT, SquareT, double>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, mask_info, skip_zero_rgb);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_masked_merge: unsupported n dtype");
}

template <typename FreshT, typename SumT>
void dispatch_sigma_clip_masked_square_dtype(
    hnw::MutableCArray<SumT> sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    const bool skip_zero_rgb) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_sigma_clip_masked_count_dtype<FreshT, SumT, uint32_t>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<uint32_t>>(),
            n,
            fresh,
            rej_high,
            rej_low,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_sigma_clip_masked_count_dtype<FreshT, SumT, uint64_t>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<uint64_t>>(),
            n,
            fresh,
            rej_high,
            rej_low,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_sigma_clip_masked_count_dtype<FreshT, SumT, double>(
            sum_mu,
            square_sum.cast<hnw::MutableCArray<double>>(),
            n,
            fresh,
            rej_high,
            rej_low,
            mask, skip_zero_rgb);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_masked_merge: unsupported square_sum dtype");
}

template <typename FreshT>
void dispatch_sigma_clip_masked_sum_dtype(
    py::array sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    const bool skip_zero_rgb) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_sigma_clip_masked_square_dtype<FreshT, uint16_t>(
            sum_mu.cast<hnw::MutableCArray<uint16_t>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_sigma_clip_masked_square_dtype<FreshT, uint32_t>(
            sum_mu.cast<hnw::MutableCArray<uint32_t>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_sigma_clip_masked_square_dtype<FreshT, uint64_t>(
            sum_mu.cast<hnw::MutableCArray<uint64_t>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low,
            mask, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_sigma_clip_masked_square_dtype<FreshT, double>(
            sum_mu.cast<hnw::MutableCArray<double>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low,
            mask, skip_zero_rgb);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_masked_merge: unsupported sum_mu dtype");
}

void validate_sigma_shapes(const py::array& sum_mu,
                           const py::array& square_sum,
                           const py::array& n,
                           const py::array& fresh,
                           const py::array& rej_high,
                           const py::array& rej_low,
                           const char* op_name) {
    validate_accumulator_shapes(sum_mu, square_sum, n, fresh, op_name);
    if (rej_high.ndim() != fresh.ndim() || rej_low.ndim() != fresh.ndim()) {
        throw std::invalid_argument(std::string(op_name) + ": rejection image ndim mismatch");
    }
    for (ssize_t i = 0; i < fresh.ndim(); ++i) {
        if (rej_high.shape(i) != fresh.shape(i) ||
            rej_low.shape(i) != fresh.shape(i)) {
            throw std::invalid_argument(std::string(op_name) + ": rejection image shape mismatch");
        }
    }
    if (py::str(rej_high.dtype()).cast<std::string>() !=
            py::str(fresh.dtype()).cast<std::string>() ||
        py::str(rej_low.dtype()).cast<std::string>() !=
            py::str(fresh.dtype()).cast<std::string>()) {
        throw std::invalid_argument(std::string(op_name) + ": rejection image dtype mismatch");
    }
}

void sigma_clip_fused_dispatch(py::array sum_mu,
                               py::array square_sum,
                               py::array n,
                               const py::array& fresh,
                               const py::array& rej_high,
                               const py::array& rej_low,
                               bool skip_zero_rgb) {
    validate_sigma_shapes(sum_mu, square_sum, n, fresh, rej_high, rej_low,
                          "sigma_clip_fused_merge");
    const ssize_t channels = (fresh.ndim() >= 3) ? fresh.shape(fresh.ndim() - 1) : 1;

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<uint8_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<uint16_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(), skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<uint32_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(), skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<uint64_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(), skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<float>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(), skip_zero_rgb, channels);
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<double>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(), skip_zero_rgb, channels);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_merge: unsupported fresh dtype");
}

void sigma_clip_fused_masked_dispatch(py::array sum_mu,
                                      py::array square_sum,
                                      py::array n,
                                      const py::array& fresh,
                                      const py::array& rej_high,
                                      const py::array& rej_low,
                                      const py::array& mask,
                                      bool skip_zero_rgb) {
    validate_sigma_shapes(sum_mu, square_sum, n, fresh, rej_high, rej_low,
                          "sigma_clip_fused_masked_merge");
    validate_masked_shapes(sum_mu, square_sum, n, fresh, mask,
                           "sigma_clip_fused_masked_merge");
    auto mask_t =
        mask.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<uint8_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<uint16_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<uint32_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<uint64_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<float>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<double>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            mask_t, skip_zero_rgb);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_masked_merge: unsupported fresh dtype");
}

}  // namespace

void bind_sigma_clip_merge_ops(py::module_& m) {
    m.def(
        "sigma_clip_fused_merge",
        &sigma_clip_fused_dispatch,
        py::arg("sum_mu"),
        py::arg("square_sum"),
        py::arg("n"),
        py::arg("fresh"),
        py::arg("rej_high"),
        py::arg("rej_low"),
        py::arg("skip_zero_rgb") = false,
        "Update rejected FastGaussianParam buffers in-place for sigma clip.");
    m.def(
        "sigma_clip_fused_masked_merge",
        &sigma_clip_fused_masked_dispatch,
        py::arg("sum_mu"),
        py::arg("square_sum"),
        py::arg("n"),
        py::arg("fresh"),
        py::arg("rej_high"),
        py::arg("rej_low"),
        py::arg("mask"),
        py::arg("skip_zero_rgb") = false,
        "Update rejected FastGaussianParam buffers in-place for masked sigma clip.");
}
