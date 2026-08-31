#pragma once

#include "common/compat.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace hnw {

class StarDetectCapacityError : public std::runtime_error {
public:
    explicit StarDetectCapacityError(const std::string& message) : std::runtime_error(message) {}
};

} // namespace hnw

void launch_star_detect_fused_pixel_components(
    const double* image_host, const uint8_t* external_mask_host, const double* gaussian_kernel_host,
    std::vector<double>* positions_xy_host, std::vector<double>* intensities_host,
    uint8_t* binary_mask_host, int height, int width, int small_height, int small_width, int level,
    int gaussian_ksize);

void bind_star_detect_fused_pixel_components_ops(py::module_& m);
