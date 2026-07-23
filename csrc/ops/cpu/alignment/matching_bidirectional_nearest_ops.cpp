#include "matching_bidirectional_nearest_ops.h"

#include "common/compat.h"
#include "common/cpu_compat.h"
#include "common/matching_cosine_contract.h"

#include <pybind11/numpy.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using FloatArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

struct BestCandidate {
    double distance = std::numeric_limits<double>::infinity();
    int64_t index = -1;
    int tied = 0;
};

constexpr long double MIN_OPENMP_SCALAR_PRODUCTS = 20'000'000.0L;
constexpr ssize_t ROW_BLOCK_SIZE = 16;

py::object matching_cosine_bidirectional_nearest_cpu_impl(const FloatArray& features1,
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
    const auto* HNW_RESTRICT f1 = static_cast<const double*>(features1_info.ptr);
    const auto* HNW_RESTRICT f2 = static_cast<const double*>(features2_info.ptr);
    auto* HNW_RESTRICT row_idx = static_cast<int64_t*>(row_indices_info.ptr);
    auto* HNW_RESTRICT row_dist = static_cast<double*>(row_distances_info.ptr);
    auto* HNW_RESTRICT col_idx = static_cast<int64_t*>(col_indices_info.ptr);
    auto* HNW_RESTRICT col_dist = static_cast<double*>(col_distances_info.ptr);

    std::vector<double> norms1(static_cast<size_t>(n1));
    std::vector<double> norms2(static_cast<size_t>(n2));
    std::vector<BestCandidate> column_best(static_cast<size_t>(n2));
    int ambiguous = 0;

    {
        py::gil_scoped_release release;

        for (ssize_t i = 0; i < n1; ++i) {
            double squared_norm = 0.0;
            const double* feature = f1 + i * dim;
#if HNW_ENABLE_OMP_SIMD || !defined(_MSC_VER)
#pragma omp simd reduction(+ : squared_norm)
#endif
            for (ssize_t k = 0; k < dim; ++k) {
                squared_norm += feature[k] * feature[k];
            }
            norms1[static_cast<size_t>(i)] = std::sqrt(squared_norm);
            if (!(norms1[static_cast<size_t>(i)] > 0.0) ||
                !std::isfinite(norms1[static_cast<size_t>(i)])) {
                ambiguous = 1;
            }
        }
        for (ssize_t j = 0; j < n2; ++j) {
            double squared_norm = 0.0;
            const double* feature = f2 + j * dim;
#if HNW_ENABLE_OMP_SIMD || !defined(_MSC_VER)
#pragma omp simd reduction(+ : squared_norm)
#endif
            for (ssize_t k = 0; k < dim; ++k) {
                squared_norm += feature[k] * feature[k];
            }
            norms2[static_cast<size_t>(j)] = std::sqrt(squared_norm);
            if (!(norms2[static_cast<size_t>(j)] > 0.0) ||
                !std::isfinite(norms2[static_cast<size_t>(j)])) {
                ambiguous = 1;
            }
        }

        if (ambiguous == 0) {
#if defined(_OPENMP)
            const bool use_parallel = static_cast<long double>(n1) * static_cast<long double>(n2) *
                                          static_cast<long double>(dim) >=
                                      MIN_OPENMP_SCALAR_PRODUCTS;
#pragma omp parallel if (use_parallel) reduction(| : ambiguous)
            {
                std::vector<BestCandidate> local_column_best(static_cast<size_t>(n2));
#pragma omp for schedule(static)
                for (ssize_t block_start = 0; block_start < n1; block_start += ROW_BLOCK_SIZE) {
                    const ssize_t block_size =
                        n1 - block_start < ROW_BLOCK_SIZE ? n1 - block_start : ROW_BLOCK_SIZE;
                    std::array<BestCandidate, ROW_BLOCK_SIZE> row_bests;
                    for (ssize_t j = 0; j < n2; ++j) {
                        const double* feature2 = f2 + j * dim;
                        for (ssize_t row_offset = 0; row_offset < block_size; ++row_offset) {
                            const ssize_t i = block_start + row_offset;
                            const double* feature1 = f1 + i * dim;
                            double dot = 0.0;
#if HNW_ENABLE_OMP_SIMD || !defined(_MSC_VER)
#pragma omp simd reduction(+ : dot)
#endif
                            for (ssize_t k = 0; k < dim; ++k) {
                                dot += feature1[k] * feature2[k];
                            }
                            const double distance = 1.0 - dot / (norms1[static_cast<size_t>(i)] *
                                                                 norms2[static_cast<size_t>(j)]);
                            if (!std::isfinite(distance)) {
                                ambiguous = 1;
                                continue;
                            }
                            BestCandidate& row_best = row_bests[static_cast<size_t>(row_offset)];
                            hnw::matching::update_best(
                                row_best.distance, row_best.index, row_best.tied, distance,
                                static_cast<int64_t>(j), static_cast<int64_t>(dim));
                            BestCandidate& column = local_column_best[static_cast<size_t>(j)];
                            hnw::matching::update_best(column.distance, column.index, column.tied,
                                                       distance, static_cast<int64_t>(i),
                                                       static_cast<int64_t>(dim));
                        }
                    }
                    for (ssize_t row_offset = 0; row_offset < block_size; ++row_offset) {
                        const ssize_t i = block_start + row_offset;
                        const BestCandidate& row_best = row_bests[static_cast<size_t>(row_offset)];
                        row_idx[i] = row_best.index;
                        row_dist[i] = row_best.distance;
                        if (row_best.index < 0 || row_best.tied) {
                            ambiguous = 1;
                        }
                    }
                }
#pragma omp critical
                {
                    for (ssize_t j = 0; j < n2; ++j) {
                        BestCandidate& best = column_best[static_cast<size_t>(j)];
                        const BestCandidate& candidate = local_column_best[static_cast<size_t>(j)];
                        hnw::matching::merge_best(best.distance, best.index, best.tied,
                                                  candidate.distance, candidate.index,
                                                  candidate.tied, static_cast<int64_t>(dim));
                    }
                }
            }
#else
            for (ssize_t i = 0; i < n1; ++i) {
                BestCandidate row_best;
                const double* feature1 = f1 + i * dim;
                for (ssize_t j = 0; j < n2; ++j) {
                    const double* feature2 = f2 + j * dim;
                    double dot = 0.0;
#if HNW_ENABLE_OMP_SIMD || !defined(_MSC_VER)
#pragma omp simd reduction(+ : dot)
#endif
                    for (ssize_t k = 0; k < dim; ++k) {
                        dot += feature1[k] * feature2[k];
                    }
                    const double distance = 1.0 - dot / (norms1[static_cast<size_t>(i)] *
                                                         norms2[static_cast<size_t>(j)]);
                    if (!std::isfinite(distance)) {
                        ambiguous = 1;
                        continue;
                    }
                    hnw::matching::update_best(row_best.distance, row_best.index, row_best.tied,
                                               distance, static_cast<int64_t>(j),
                                               static_cast<int64_t>(dim));
                    BestCandidate& column = column_best[static_cast<size_t>(j)];
                    hnw::matching::update_best(column.distance, column.index, column.tied, distance,
                                               static_cast<int64_t>(i), static_cast<int64_t>(dim));
                }
                row_idx[i] = row_best.index;
                row_dist[i] = row_best.distance;
                if (row_best.index < 0 || row_best.tied) {
                    ambiguous = 1;
                }
            }
#endif
            for (ssize_t j = 0; j < n2; ++j) {
                const BestCandidate& best = column_best[static_cast<size_t>(j)];
                col_idx[j] = best.index;
                col_dist[j] = best.distance;
                if (best.index < 0 || best.tied) {
                    ambiguous = 1;
                }
            }
        }
    }

    if (ambiguous != 0) {
        return py::none();
    }
    return py::make_tuple(row_indices, row_distances, col_indices, col_distances);
}

} // namespace

void bind_matching_bidirectional_nearest_cpu_ops(py::module_& m) {
    m.def("matching_cosine_bidirectional_nearest_cpu",
          &matching_cosine_bidirectional_nearest_cpu_impl, py::arg("features1"),
          py::arg("features2"),
          "Return bidirectional cosine nearest-neighbor indices and distances using OpenMP; "
          "returns None for ambiguous NumPy ordering cases.");
}
