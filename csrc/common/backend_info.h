#pragma once

#include "common/compat.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

py::dict build_info_dict();
py::dict cuda_memory_info_dict();
#if HNW_ENABLE_CUDA
py::dict cuda_memory_info_cuda_dict();
#endif
int get_openmp_max_threads();
bool set_openmp_threads(int num_threads);
void bind_backend_info(py::module_& m);
