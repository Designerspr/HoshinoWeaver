#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>

namespace py = pybind11;

bool launch_matching_cosine_bidirectional_nearest_cuda(
    const double* features1_host, const double* features2_host, int64_t n1, int64_t n2, int64_t dim,
    int64_t* row_indices_host, double* row_distances_host, int64_t* col_indices_host,
    double* col_distances_host);

void bind_matching_bidirectional_nearest_cuda_ops(py::module_& m);
