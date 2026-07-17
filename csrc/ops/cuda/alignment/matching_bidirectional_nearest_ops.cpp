#include "matching_bidirectional_nearest_ops.h"

#include "common/compat.h"

#include <pybind11/numpy.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

using FloatArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::object matching_cosine_bidirectional_nearest_cuda_impl(const FloatArray& features1,
                                                           const FloatArray& features2) {
    if (features1.ndim() != 2 || features2.ndim() != 2 ||
        features1.shape(1) != features2.shape(1)) {
        throw std::invalid_argument(
            "matching_cosine_bidirectional_nearest: features must have shape "
            "(N, D) and (M, D) with matching D");
    }
    const ssize_t n1 = features1.shape(0);
    const ssize_t n2 = features2.shape(0);
    const ssize_t dim = features1.shape(1);
    if (n1 <= 0 || n2 <= 0 || dim <= 0) {
        throw std::invalid_argument(
            "matching_cosine_bidirectional_nearest: feature dimensions must be positive");
    }
    if (n1 > std::numeric_limits<int>::max() || n2 > std::numeric_limits<int>::max() ||
        dim > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "matching_cosine_bidirectional_nearest: feature dimensions are too large");
    }
    if (n1 > std::numeric_limits<int64_t>::max() / n2 ||
        n1 * n2 > static_cast<int64_t>(std::numeric_limits<int>::max()) * 256) {
        throw std::invalid_argument(
            "matching_cosine_bidirectional_nearest: distance matrix is too large");
    }

    py::array_t<int64_t> row_indices(n1);
    py::array_t<double> row_distances(n1);
    py::array_t<int64_t> col_indices(n2);
    py::array_t<double> col_distances(n2);
    const auto features1_info = features1.request();
    const auto features2_info = features2.request();
    auto row_indices_info = row_indices.request();
    auto row_distances_info = row_distances.request();
    auto col_indices_info = col_indices.request();
    auto col_distances_info = col_distances.request();

    bool exact = false;
    {
        py::gil_scoped_release release;
        exact = launch_matching_cosine_bidirectional_nearest_cuda(
            static_cast<const double*>(features1_info.ptr),
            static_cast<const double*>(features2_info.ptr), static_cast<int64_t>(n1),
            static_cast<int64_t>(n2), static_cast<int64_t>(dim),
            static_cast<int64_t*>(row_indices_info.ptr),
            static_cast<double*>(row_distances_info.ptr),
            static_cast<int64_t*>(col_indices_info.ptr),
            static_cast<double*>(col_distances_info.ptr));
    }
    if (!exact) {
        return py::none();
    }
    return py::make_tuple(row_indices, row_distances, col_indices, col_distances);
}

} // namespace

void bind_matching_bidirectional_nearest_cuda_ops(py::module_& m) {
    m.def("matching_cosine_bidirectional_nearest_cuda",
          &matching_cosine_bidirectional_nearest_cuda_impl, py::arg("features1"),
          py::arg("features2"),
          "Return bidirectional cosine nearest-neighbor indices and distances using CUDA; "
          "returns None for ambiguous NumPy ordering cases.");
}
