#pragma once

#include "common/compat.h"

namespace hnw::cuda {

// Keep this aligned with the oldest architecture emitted by HnwCuda.cmake.
inline constexpr int kMinimumComputeCapabilityMajor = 6;
inline constexpr int kMinimumComputeCapabilityMinor = 0;

inline constexpr bool compute_capability_supported(const int major, const int minor) {
    return major > kMinimumComputeCapabilityMajor ||
           (major == kMinimumComputeCapabilityMajor && minor >= kMinimumComputeCapabilityMinor);
}

} // namespace hnw::cuda
