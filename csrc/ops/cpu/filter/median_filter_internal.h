#pragma once

#include "common/cpu_compat.h"
#include "common/median_histogram.h"

#include <algorithm>
#include <cstdint>

namespace hnw::cpu {

inline ssize_t clamp_median_index(ssize_t value, ssize_t low, ssize_t high) {
    return std::max(low, std::min(value, high));
}

template <typename T>
inline T load_median_pixel(const T* HNW_RESTRICT input, ssize_t h, ssize_t w, ssize_t channels,
                           ssize_t y, ssize_t x, ssize_t c) {
    const ssize_t yy = clamp_median_index(y, 0, h - 1);
    const ssize_t xx = clamp_median_index(x, 0, w - 1);
    return input[(yy * w + xx) * channels + c];
}

template <typename T>
void median_filter_2d_kernel(const T* HNW_RESTRICT input, T* HNW_RESTRICT output, ssize_t h,
                             ssize_t w, ssize_t channels, ssize_t ksize) {
    const ssize_t radius = ksize / 2;
    const uint64_t window_area = static_cast<uint64_t>(ksize) * static_cast<uint64_t>(ksize);
    const uint32_t target_rank = static_cast<uint32_t>(window_area / 2 + 1);
    const ssize_t task_count = h * channels;

#if defined(_OPENMP)
#pragma omp parallel
    {
        MedianHistogram<T> hist;
#pragma omp for schedule(static)
        for (ssize_t task = 0; task < task_count; ++task) {
            const ssize_t y = task / channels;
            const ssize_t c = task % channels;
            const auto at = [&](int64_t yy, int64_t xx) {
                return load_median_pixel(input, h, w, channels, static_cast<ssize_t>(yy),
                                         static_cast<ssize_t>(xx), c);
            };
            build_median_histogram_for_row(hist, at, y, radius);
            for (ssize_t x = 0; x < w; ++x) {
                output[(y * w + x) * channels + c] = hist.median(target_rank);
                if (x + 1 < w) {
                    slide_median_histogram_right(hist, at, y, x, radius);
                }
            }
        }
    }
#else
    MedianHistogram<T> hist;
    for (ssize_t task = 0; task < task_count; ++task) {
        const ssize_t y = task / channels;
        const ssize_t c = task % channels;
        const auto at = [&](int64_t yy, int64_t xx) {
            return load_median_pixel(input, h, w, channels, static_cast<ssize_t>(yy),
                                     static_cast<ssize_t>(xx), c);
        };
        build_median_histogram_for_row(hist, at, y, radius);
        for (ssize_t x = 0; x < w; ++x) {
            output[(y * w + x) * channels + c] = hist.median(target_rank);
            if (x + 1 < w) {
                slide_median_histogram_right(hist, at, y, x, radius);
            }
        }
    }
#endif
}

} // namespace hnw::cpu
