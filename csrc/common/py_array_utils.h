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

template <typename T>
using MutableCArray = py::array_t<T, py::array::c_style>;

}  // namespace hnw
