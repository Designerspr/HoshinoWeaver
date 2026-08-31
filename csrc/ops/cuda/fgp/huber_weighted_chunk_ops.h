#pragma once

#include "common/compat.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_huber_weighted_chunk_cuda_ops(py::module_& m);
