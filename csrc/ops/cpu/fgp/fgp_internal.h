#pragma once

#include "common/cpu_compat.h"
#include "common/py_array_utils.h"

#include <algorithm>
#include <string>

#include <pybind11/numpy.h>

namespace hnw::fgp_internal {

inline void validate_accumulator_shapes(const py::array& sum_mu,
                                        const py::array& square_sum,
                                        const py::array& n,
                                        const py::array& fresh,
                                        const char* op_name) {
    require_mutable_c_array(sum_mu, op_name, "sum_mu");
    require_mutable_c_array(square_sum, op_name, "square_sum");
    require_mutable_c_array(n, op_name, "n");
    require_same_shape(sum_mu, fresh, op_name);
    require_same_shape(square_sum, fresh, op_name);
    require_same_shape(n, fresh, op_name);
}

inline void validate_masked_shapes(const py::array& sum_mu,
                                   const py::array& square_sum,
                                   const py::array& n,
                                   const py::array& fresh,
                                   const py::array& mask,
                                   const char* op_name) {
    validate_accumulator_shapes(sum_mu, square_sum, n, fresh, op_name);
    if (mask.ndim() == fresh.ndim()) {
        for (ssize_t idx = 0; idx < fresh.ndim(); ++idx) {
            if (mask.shape(idx) != fresh.shape(idx)) {
                throw std::invalid_argument(
                    std::string(op_name) + ": mask shape mismatch");
            }
        }
        return;
    }
    if (mask.ndim() + 1 == fresh.ndim()) {
        for (ssize_t idx = 0; idx < mask.ndim(); ++idx) {
            if (mask.shape(idx) != fresh.shape(idx)) {
                throw std::invalid_argument(
                    std::string(op_name) + ": mask shape mismatch");
            }
        }
        return;
    }
    throw std::invalid_argument(std::string(op_name) + ": mask ndim mismatch");
}

template <typename FreshT>
inline bool is_pixel_zero_rgb(const FreshT* HNW_RESTRICT ptr,
                              ssize_t base,
                              ssize_t channels) {
    for (ssize_t channel = 0; channel < std::min<ssize_t>(channels, 3);
         ++channel) {
        if (ptr[base + channel] != static_cast<FreshT>(0)) {
            return false;
        }
    }
    return true;
}

}  // namespace hnw::fgp_internal

void bind_fgp_accumulate_ops(py::module_& m);
void bind_fgp_merge_ops(py::module_& m);
void bind_sigma_clip_merge_ops(py::module_& m);
void bind_huber_accumulate_ops(py::module_& m);
