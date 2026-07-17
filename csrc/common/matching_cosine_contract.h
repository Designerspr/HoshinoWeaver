#pragma once

#include <cfloat>
#include <cstdint>

#if defined(__CUDACC__)
#define HNW_MATCHING_HOST_DEVICE __host__ __device__
#else
#define HNW_MATCHING_HOST_DEVICE
#endif

namespace hnw::matching {

// Sequential, vectorized, and GPU dot products may accumulate in different
// orders. Treat nearest distances inside this conservative error envelope as
// ambiguous so the Python wrapper can recover the exact SciPy argsort contract.
constexpr double COSINE_NEAR_TIE_ERROR_FACTOR = 8.0;

HNW_MATCHING_HOST_DEVICE inline bool cosine_distances_are_near(const double lhs, const double rhs,
                                                               const int64_t feature_dim) {
    const double difference = lhs >= rhs ? lhs - rhs : rhs - lhs;
    const double lhs_abs = lhs >= 0.0 ? lhs : -lhs;
    const double rhs_abs = rhs >= 0.0 ? rhs : -rhs;
    double scale = lhs_abs > rhs_abs ? lhs_abs : rhs_abs;
    if (scale < 1.0) {
        scale = 1.0;
    }
    const double tolerance =
        COSINE_NEAR_TIE_ERROR_FACTOR * DBL_EPSILON * static_cast<double>(feature_dim) * scale;
    return difference <= tolerance;
}

HNW_MATCHING_HOST_DEVICE inline void update_best(double& best_distance, int64_t& best_index,
                                                 int& tied, const double distance,
                                                 const int64_t index, const int64_t feature_dim) {
    if (best_index < 0 || distance < best_distance) {
        const bool near_previous =
            best_index >= 0 && cosine_distances_are_near(distance, best_distance, feature_dim);
        best_distance = distance;
        best_index = index;
        tied = near_previous ? 1 : 0;
        return;
    }
    if (index != best_index && cosine_distances_are_near(distance, best_distance, feature_dim)) {
        tied = 1;
        if (distance == best_distance && index < best_index) {
            best_index = index;
        }
    }
}

HNW_MATCHING_HOST_DEVICE inline void
merge_best(double& best_distance, int64_t& best_index, int& tied, const double candidate_distance,
           const int64_t candidate_index, const int candidate_tied, const int64_t feature_dim) {
    if (candidate_index < 0) {
        return;
    }
    if (best_index < 0 || candidate_distance < best_distance) {
        const bool near_previous =
            best_index >= 0 &&
            cosine_distances_are_near(candidate_distance, best_distance, feature_dim);
        best_distance = candidate_distance;
        best_index = candidate_index;
        tied = candidate_tied || near_previous;
        return;
    }
    if (cosine_distances_are_near(candidate_distance, best_distance, feature_dim)) {
        tied = tied || candidate_tied || candidate_index != best_index;
        if (candidate_distance == best_distance && candidate_index < best_index) {
            best_index = candidate_index;
        }
    }
}

} // namespace hnw::matching

#undef HNW_MATCHING_HOST_DEVICE
