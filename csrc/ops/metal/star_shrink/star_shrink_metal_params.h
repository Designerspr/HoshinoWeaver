#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace hnw::metal::star_shrink {

// Host mirrors of the MSL parameter structs. The fused DoG+shrink op dispatches
// kernels from both shader files, so both layouts live here rather than in one
// translation unit. Field order must match the .metal definitions exactly.

// star_mask.metal: ksize doubles as the Gaussian radius or morphology kernel
// size, and threshold is only meaningful for the mask kernel.
struct MaskParams {
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t ksize;
    float threshold;
};

// star_shrink_process.metal
struct ShrinkParams {
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t shrink_ksize;
    uint32_t shrink_shape;
    uint32_t deringing_ksize;
    float shrink_ratio;
};

static_assert(sizeof(MaskParams) == 20, "MaskParams must match the MSL struct layout");
static_assert(sizeof(ShrinkParams) == 28, "ShrinkParams must match the MSL struct layout");

inline int parse_shrink_shape(const std::string& shape, const char* context) {
    if (shape == "RECT") {
        return 0;
    }
    if (shape == "CROSS") {
        return 1;
    }
    if (shape == "CIRCLE") {
        return 2;
    }
    throw std::invalid_argument(std::string(context) + ": unknown shrink_shape");
}

inline void validate_shrink_params(const int shrink_ksize, const int shrink_times,
                                   const float shrink_ratio, const int deringing_ksize,
                                   const char* context) {
    const std::string prefix(context);
    if (shrink_ksize <= 0 || shrink_ksize % 2 == 0) {
        throw std::invalid_argument(prefix + ": shrink_ksize must be a positive odd value");
    }
    if (shrink_times <= 0) {
        throw std::invalid_argument(prefix + ": shrink_times must be positive");
    }
    if (!(shrink_ratio > 0.0f && shrink_ratio <= 1.0f)) {
        throw std::invalid_argument(prefix + ": shrink_ratio must be in (0, 1]");
    }
    if (deringing_ksize <= 0 || deringing_ksize % 2 == 0) {
        throw std::invalid_argument(prefix + ": deringing_ksize must be a positive odd value");
    }
}

} // namespace hnw::metal::star_shrink
