#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_matching_bidirectional_nearest_cpu_ops(py::module_& m);
