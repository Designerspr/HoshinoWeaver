#pragma once

#include "common/compat.h"

#include <pybind11/numpy.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace hnw {

inline void require_mutable_c_array(const py::array& array,
                                    const char* op_name,
                                    const char* argument_name) {
    if ((array.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument(
            std::string(op_name) + ": " + argument_name + " must be C-contiguous");
    }
    if (!array.writeable()) {
        throw std::invalid_argument(
            std::string(op_name) + ": " + argument_name + " must be writeable");
    }
}

inline void require_same_shape(const py::array& first,
                               const py::array& second,
                               const char* op_name) {
    if (first.ndim() != second.ndim()) {
        throw std::invalid_argument(std::string(op_name) + ": ndim mismatch");
    }
    for (ssize_t idx = 0; idx < first.ndim(); ++idx) {
        if (first.shape(idx) != second.shape(idx)) {
            throw std::invalid_argument(std::string(op_name) + ": shape mismatch");
        }
    }
}

inline void require_same_shape(const py::buffer_info& first,
                               const py::buffer_info& second,
                               const char* op_name) {
    if (first.ndim != second.ndim) {
        throw std::invalid_argument(std::string(op_name) + ": ndim mismatch");
    }
    for (ssize_t idx = 0; idx < first.ndim; ++idx) {
        if (first.shape[idx] != second.shape[idx]) {
            throw std::invalid_argument(std::string(op_name) + ": shape mismatch");
        }
    }
}

inline void require_same_dtype(const py::array& first,
                               const py::array& second,
                               const char* op_name) {
    if (py::str(first.dtype()).cast<std::string>() !=
        py::str(second.dtype()).cast<std::string>()) {
        throw std::invalid_argument(std::string(op_name) + ": dtype mismatch");
    }
}

template <typename T>
using MutableCArray = py::array_t<T, py::array::c_style>;

}  // namespace hnw
