#include "matching_bidirectional_nearest_ops.h"

#include "common/compat.h"
#include "common/cuda_host_io_workspace.cuh"
#include "common/matching_cosine_contract.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

__global__ void feature_norms_kernel(const double* features, double* norms, const int64_t count,
                                     const int64_t dim, int* ambiguous) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= count) {
        return;
    }
    const double* feature = features + row * dim;
    double squared_norm = 0.0;
    for (int64_t k = 0; k < dim; ++k) {
        squared_norm += feature[k] * feature[k];
    }
    const double norm = sqrt(squared_norm);
    norms[row] = norm;
    if (!(norm > 0.0) || !isfinite(norm)) {
        atomicExch(ambiguous, 1);
    }
}

__global__ void cosine_distance_matrix_kernel(const double* features1, const double* features2,
                                              const double* norms1, const double* norms2,
                                              double* distances, const int64_t n1, const int64_t n2,
                                              const int64_t dim, int* ambiguous) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n1 * n2;
    if (index >= total) {
        return;
    }
    const int64_t row = index / n2;
    const int64_t col = index - row * n2;
    const double* feature1 = features1 + row * dim;
    const double* feature2 = features2 + col * dim;
    double dot = 0.0;
    for (int64_t k = 0; k < dim; ++k) {
        dot += feature1[k] * feature2[k];
    }
    const double distance = 1.0 - dot / (norms1[row] * norms2[col]);
    distances[index] = distance;
    if (!isfinite(distance)) {
        atomicExch(ambiguous, 1);
    }
}

template <bool ReduceRows>
__global__ void nearest_reduce_kernel(const double* distances, const int64_t n1, const int64_t n2,
                                      const int64_t feature_dim, int64_t* indices,
                                      double* nearest_distances, int* ambiguous) {
    __shared__ double shared_distances[THREADS_PER_BLOCK];
    __shared__ int64_t shared_indices[THREADS_PER_BLOCK];
    __shared__ int shared_ties[THREADS_PER_BLOCK];

    const int64_t output_index = static_cast<int64_t>(blockIdx.x);
    const int64_t reduce_count = ReduceRows ? n2 : n1;
    double best_distance = DBL_MAX;
    int64_t best_index = -1;
    int tied = 0;
    for (int64_t candidate = threadIdx.x; candidate < reduce_count; candidate += blockDim.x) {
        const int64_t matrix_index =
            ReduceRows ? output_index * n2 + candidate : candidate * n2 + output_index;
        const double distance = distances[matrix_index];
        if (!isfinite(distance)) {
            continue;
        }
        hnw::matching::update_best(best_distance, best_index, tied, distance, candidate,
                                   feature_dim);
    }
    shared_distances[threadIdx.x] = best_distance;
    shared_indices[threadIdx.x] = best_index;
    shared_ties[threadIdx.x] = tied;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            hnw::matching::merge_best(shared_distances[threadIdx.x], shared_indices[threadIdx.x],
                                      shared_ties[threadIdx.x],
                                      shared_distances[threadIdx.x + stride],
                                      shared_indices[threadIdx.x + stride],
                                      shared_ties[threadIdx.x + stride], feature_dim);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        indices[output_index] = shared_indices[0];
        nearest_distances[output_index] = shared_distances[0];
        if (shared_indices[0] < 0 || shared_ties[0] != 0) {
            atomicExch(ambiguous, 1);
        }
    }
}

} // namespace

bool launch_matching_cosine_bidirectional_nearest_cuda(
    const double* features1_host, const double* features2_host, const int64_t n1, const int64_t n2,
    const int64_t dim, int64_t* row_indices_host, double* row_distances_host,
    int64_t* col_indices_host, double* col_distances_host) {
    const size_t features1_bytes =
        static_cast<size_t>(n1) * static_cast<size_t>(dim) * sizeof(double);
    const size_t features2_bytes =
        static_cast<size_t>(n2) * static_cast<size_t>(dim) * sizeof(double);
    const size_t norms1_bytes = static_cast<size_t>(n1) * sizeof(double);
    const size_t norms2_bytes = static_cast<size_t>(n2) * sizeof(double);
    const size_t distance_bytes =
        static_cast<size_t>(n1) * static_cast<size_t>(n2) * sizeof(double);
    const size_t row_indices_bytes = static_cast<size_t>(n1) * sizeof(int64_t);
    const size_t row_distances_bytes = static_cast<size_t>(n1) * sizeof(double);
    const size_t col_indices_bytes = static_cast<size_t>(n2) * sizeof(int64_t);
    const size_t col_distances_bytes = static_cast<size_t>(n2) * sizeof(double);

    auto workspace =
        hnw::cuda::acquire_host_io_workspace("matching bidirectional nearest cudaGetDevice");
    try {
        auto* features1_device = static_cast<double*>(workspace.device_buffer(
            features1_bytes, "matching bidirectional nearest cudaMalloc(features1)"));
        auto* features2_device = static_cast<double*>(workspace.device_buffer(
            features2_bytes, "matching bidirectional nearest cudaMalloc(features2)"));
        auto* norms1_device = static_cast<double*>(workspace.device_buffer(
            norms1_bytes, "matching bidirectional nearest cudaMalloc(norms1)"));
        auto* norms2_device = static_cast<double*>(workspace.device_buffer(
            norms2_bytes, "matching bidirectional nearest cudaMalloc(norms2)"));
        auto* distances_device = static_cast<double*>(workspace.device_buffer(
            distance_bytes, "matching bidirectional nearest cudaMalloc(distances)"));
        auto* row_indices_device = static_cast<int64_t*>(workspace.device_buffer(
            row_indices_bytes, "matching bidirectional nearest cudaMalloc(row_indices)"));
        auto* row_distances_device = static_cast<double*>(workspace.device_buffer(
            row_distances_bytes, "matching bidirectional nearest cudaMalloc(row_distances)"));
        auto* col_indices_device = static_cast<int64_t*>(workspace.device_buffer(
            col_indices_bytes, "matching bidirectional nearest cudaMalloc(col_indices)"));
        auto* col_distances_device = static_cast<double*>(workspace.device_buffer(
            col_distances_bytes, "matching bidirectional nearest cudaMalloc(col_distances)"));
        auto* ambiguous_device = static_cast<int*>(workspace.device_buffer(
            sizeof(int), "matching bidirectional nearest cudaMalloc(ambiguous)"));

        void* pinned_features1 = workspace.pinned_buffer(
            features1_bytes, "matching bidirectional nearest cudaMallocHost(features1)");
        void* pinned_features2 = workspace.pinned_buffer(
            features2_bytes, "matching bidirectional nearest cudaMallocHost(features2)");
        void* pinned_row_indices = workspace.pinned_buffer(
            row_indices_bytes, "matching bidirectional nearest cudaMallocHost(row_indices)");
        void* pinned_row_distances = workspace.pinned_buffer(
            row_distances_bytes, "matching bidirectional nearest cudaMallocHost(row_distances)");
        void* pinned_col_indices = workspace.pinned_buffer(
            col_indices_bytes, "matching bidirectional nearest cudaMallocHost(col_indices)");
        void* pinned_col_distances = workspace.pinned_buffer(
            col_distances_bytes, "matching bidirectional nearest cudaMallocHost(col_distances)");
        void* pinned_ambiguous = workspace.pinned_buffer(
            sizeof(int), "matching bidirectional nearest cudaMallocHost(ambiguous)");

        std::memcpy(pinned_features1, features1_host, features1_bytes);
        std::memcpy(pinned_features2, features2_host, features2_bytes);
        *static_cast<int*>(pinned_ambiguous) = 0;
        cudaStream_t stream = workspace.stream();
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(features1_device, pinned_features1,
                                                   features1_bytes, cudaMemcpyHostToDevice, stream),
                                   "matching bidirectional nearest cudaMemcpyAsync(features1)");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(features2_device, pinned_features2,
                                                   features2_bytes, cudaMemcpyHostToDevice, stream),
                                   "matching bidirectional nearest cudaMemcpyAsync(features2)");
        hnw::cuda::throw_if_failed(cudaMemsetAsync(ambiguous_device, 0, sizeof(int), stream),
                                   "matching bidirectional nearest cudaMemsetAsync(ambiguous)");

        const int feature1_blocks =
            static_cast<int>((n1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        const int feature2_blocks =
            static_cast<int>((n2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        feature_norms_kernel<<<feature1_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            features1_device, norms1_device, n1, dim, ambiguous_device);
        feature_norms_kernel<<<feature2_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            features2_device, norms2_device, n2, dim, ambiguous_device);
        const int64_t pair_count = n1 * n2;
        const int distance_blocks =
            static_cast<int>((pair_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        cosine_distance_matrix_kernel<<<distance_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            features1_device, features2_device, norms1_device, norms2_device, distances_device, n1,
            n2, dim, ambiguous_device);
        nearest_reduce_kernel<true><<<static_cast<int>(n1), THREADS_PER_BLOCK, 0, stream>>>(
            distances_device, n1, n2, dim, row_indices_device, row_distances_device,
            ambiguous_device);
        nearest_reduce_kernel<false><<<static_cast<int>(n2), THREADS_PER_BLOCK, 0, stream>>>(
            distances_device, n1, n2, dim, col_indices_device, col_distances_device,
            ambiguous_device);
        hnw::cuda::throw_if_failed(cudaGetLastError(),
                                   "matching bidirectional nearest kernel launch");

        hnw::cuda::throw_if_failed(cudaMemcpyAsync(pinned_row_indices, row_indices_device,
                                                   row_indices_bytes, cudaMemcpyDeviceToHost,
                                                   stream),
                                   "matching bidirectional nearest cudaMemcpyAsync(row_indices)");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(pinned_row_distances, row_distances_device,
                                                   row_distances_bytes, cudaMemcpyDeviceToHost,
                                                   stream),
                                   "matching bidirectional nearest cudaMemcpyAsync(row_distances)");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(pinned_col_indices, col_indices_device,
                                                   col_indices_bytes, cudaMemcpyDeviceToHost,
                                                   stream),
                                   "matching bidirectional nearest cudaMemcpyAsync(col_indices)");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(pinned_col_distances, col_distances_device,
                                                   col_distances_bytes, cudaMemcpyDeviceToHost,
                                                   stream),
                                   "matching bidirectional nearest cudaMemcpyAsync(col_distances)");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(pinned_ambiguous, ambiguous_device, sizeof(int),
                                                   cudaMemcpyDeviceToHost, stream),
                                   "matching bidirectional nearest cudaMemcpyAsync(ambiguous)");
        hnw::cuda::throw_if_failed(cudaStreamSynchronize(stream),
                                   "matching bidirectional nearest cudaStreamSynchronize");

        std::memcpy(row_indices_host, pinned_row_indices, row_indices_bytes);
        std::memcpy(row_distances_host, pinned_row_distances, row_distances_bytes);
        std::memcpy(col_indices_host, pinned_col_indices, col_indices_bytes);
        std::memcpy(col_distances_host, pinned_col_distances, col_distances_bytes);
        return *static_cast<const int*>(pinned_ambiguous) == 0;
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}
