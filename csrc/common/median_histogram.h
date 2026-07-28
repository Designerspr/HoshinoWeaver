#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace hnw::cpu {

// Rank-selection histograms for sliding-window median filters.  uint8 uses a
// flat 256-bin count; uint16 adds a 256-bin coarse level over the fine bins so
// median() scans at most 256 + 256 buckets instead of 65536.
template <typename T> struct MedianHistogram;

template <> struct MedianHistogram<uint8_t> {
    std::array<uint32_t, 256> bins{};

    void clear() { bins.fill(0); }
    void add(uint8_t value) { ++bins[value]; }
    void remove(uint8_t value) { --bins[value]; }

    uint8_t median(uint32_t target_rank) const {
        uint32_t count = 0;
        for (uint32_t value = 0; value < bins.size(); ++value) {
            count += bins[value];
            if (count >= target_rank) {
                return static_cast<uint8_t>(value);
            }
        }
        return 255;
    }
};

template <> struct MedianHistogram<uint16_t> {
    std::array<uint32_t, 256> coarse{};
    std::vector<uint32_t> fine;

    MedianHistogram() : fine(65536, 0) {}

    void clear() {
        coarse.fill(0);
        std::fill(fine.begin(), fine.end(), 0);
    }

    void add(uint16_t value) {
        ++coarse[value >> 8];
        ++fine[value];
    }

    void remove(uint16_t value) {
        --coarse[value >> 8];
        --fine[value];
    }

    uint16_t median(uint32_t target_rank) const {
        uint32_t count = 0;
        uint32_t high_bin = 0;
        for (; high_bin < coarse.size(); ++high_bin) {
            const uint32_t next = count + coarse[high_bin];
            if (next >= target_rank) {
                break;
            }
            count = next;
        }

        const uint32_t start = high_bin << 8;
        const uint32_t end = start + 256;
        for (uint32_t value = start; value < end; ++value) {
            count += fine[value];
            if (count >= target_rank) {
                return static_cast<uint16_t>(value);
            }
        }
        return 65535;
    }
};

// Sliding-window helpers shared by row-major median kernels.  `at(y, x)` must
// return the (border-clamped) source sample the histogram tracks.
template <typename T, typename At>
void build_median_histogram_for_row(MedianHistogram<T>& hist, At&& at, int64_t y, int64_t radius) {
    hist.clear();
    for (int64_t dy = -radius; dy <= radius; ++dy) {
        for (int64_t dx = -radius; dx <= radius; ++dx) {
            hist.add(at(y + dy, dx));
        }
    }
}

template <typename T, typename At>
void slide_median_histogram_right(MedianHistogram<T>& hist, At&& at, int64_t y, int64_t x,
                                  int64_t radius) {
    const int64_t remove_x = x - radius;
    const int64_t add_x = x + radius + 1;
    for (int64_t dy = -radius; dy <= radius; ++dy) {
        hist.remove(at(y + dy, remove_x));
        hist.add(at(y + dy, add_x));
    }
}

} // namespace hnw::cpu
