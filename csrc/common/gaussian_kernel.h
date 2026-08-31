#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnw::camera {

// Single source for the DoG separable Gaussian taps shared by the OpenMP, CUDA
// and Metal star-mask backends; `context` prefixes the error message. Radius and
// the double-accumulated normalisation must stay identical across backends or
// their masks diverge.
inline std::vector<float> make_dog_gaussian_kernel(const float sigma, const char* context) {
    if (!std::isfinite(sigma) || !(sigma > 0.0f)) {
        throw std::invalid_argument(std::string(context) +
                                    ": sigma values must be positive and finite");
    }
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    const float denom = 2.0f * sigma * sigma;
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const float value = std::exp(-(static_cast<float>(i * i)) / denom);
        kernel[static_cast<size_t>(i + radius)] = value;
        sum += value;
    }
    for (float& value : kernel) {
        value = static_cast<float>(static_cast<double>(value) / sum);
    }
    return kernel;
}

} // namespace hnw::camera
