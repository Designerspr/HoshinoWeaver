#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hnw::wavelet {

constexpr int kDb8FilterLength = 16;

template <typename Integer> constexpr Integer dwt_length(const Integer value) {
    return (value + static_cast<Integer>(kDb8FilterLength - 1)) / static_cast<Integer>(2);
}

template <typename Integer> constexpr Integer idwt_length(const Integer value) {
    return static_cast<Integer>(2) * value - static_cast<Integer>(kDb8FilterLength - 2);
}

template <typename Integer>
std::pair<Integer, Integer> reconstructed_shape(const Integer height, const Integer width,
                                                const Integer level) {
    static_assert(std::is_signed_v<Integer>);
    std::vector<std::pair<Integer, Integer>> details;
    details.reserve(static_cast<size_t>(level));
    Integer current_h = height;
    Integer current_w = width;
    for (Integer idx = 0; idx < level; ++idx) {
        current_h = dwt_length(current_h);
        current_w = dwt_length(current_w);
        details.emplace_back(current_h, current_w);
    }
    for (Integer idx = level - 1; idx >= 0; --idx) {
        current_h = idwt_length(details[static_cast<size_t>(idx)].first);
        current_w = idwt_length(details[static_cast<size_t>(idx)].second);
    }
    return {current_h, current_w};
}

} // namespace hnw::wavelet
