#pragma once

#include "common/compat.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_star_mask_dog_cpu_ops(py::module_& m);
