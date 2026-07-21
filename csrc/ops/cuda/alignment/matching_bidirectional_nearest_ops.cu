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

constexpr int REDUCTION_THREADS = 256;
constexpr int TILE_SIZE = hnw::matching::COSINE_TILE_SIZE;
constexpr int TILE_THREADS = TILE_SIZE * TILE_SIZE;

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

__global__ void tiled_cosine_partial_nearest_kernel(
    const double* features1, const double* features2, const double* norms1, const double* norms2,
    double* row_partial_distances, int64_t* row_partial_indices, int* row_partial_ties,
    double* col_partial_distances, int64_t* col_partial_indices, int* col_partial_ties,
    const int64_t n1, const int64_t n2, const int64_t dim, const int64_t row_tiles,
    const int64_t col_tiles, int* ambiguous) {
    __shared__ double feature1_tile[TILE_SIZE][TILE_SIZE];
    __shared__ double feature2_tile[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_distances[TILE_THREADS];

    const int64_t tile_index = static_cast<int64_t>(blockIdx.x);
    const int64_t tile_row = tile_index / col_tiles;
    const int64_t tile_col = tile_index - tile_row * col_tiles;
    const int local_row = threadIdx.x / TILE_SIZE;
    const int local_col = threadIdx.x - local_row * TILE_SIZE;
    const int64_t row = tile_row * TILE_SIZE + local_row;
    const int64_t col = tile_col * TILE_SIZE + local_col;
    double dot = 0.0;
    for (int64_t base = 0; base < dim; base += TILE_SIZE) {
        const int64_t feature_index = base + local_col;
        feature1_tile[local_row][local_col] =
            row < n1 && feature_index < dim ? features1[row * dim + feature_index] : 0.0;
        feature2_tile[local_row][local_col] =
            tile_col * TILE_SIZE + local_row < n2 && feature_index < dim
                ? features2[(tile_col * TILE_SIZE + local_row) * dim + feature_index]
                : 0.0;
        __syncthreads();
        if (row < n1 && col < n2) {
            const int64_t chunk_size = dim - base < TILE_SIZE ? dim - base : TILE_SIZE;
            for (int64_t k = 0; k < chunk_size; ++k) {
                dot += feature1_tile[local_row][k] * feature2_tile[local_col][k];
            }
        }
        __syncthreads();
    }

    double distance = DBL_MAX;
    if (row < n1 && col < n2) {
        distance = 1.0 - dot / (norms1[row] * norms2[col]);
    }
    tile_distances[threadIdx.x] = distance;
    if (row < n1 && col < n2 && !isfinite(distance)) {
        atomicExch(ambiguous, 1);
    }
    __syncthreads();

    if (local_col == 0 && row < n1) {
        double best_distance = DBL_MAX;
        int64_t best_index = -1;
        int tied = 0;
        for (int candidate = 0; candidate < TILE_SIZE; ++candidate) {
            const int64_t candidate_index = tile_col * TILE_SIZE + candidate;
            if (candidate_index >= n2) {
                break;
            }
            hnw::matching::update_best(best_distance, best_index, tied,
                                       tile_distances[local_row * TILE_SIZE + candidate],
                                       candidate_index, dim);
        }
        const int64_t partial_index = row * col_tiles + tile_col;
        row_partial_distances[partial_index] = best_distance;
        row_partial_indices[partial_index] = best_index;
        row_partial_ties[partial_index] = tied;
    }
    if (local_row == 0 && col < n2) {
        double best_distance = DBL_MAX;
        int64_t best_index = -1;
        int tied = 0;
        for (int candidate = 0; candidate < TILE_SIZE; ++candidate) {
            const int64_t candidate_index = tile_row * TILE_SIZE + candidate;
            if (candidate_index >= n1) {
                break;
            }
            hnw::matching::update_best(best_distance, best_index, tied,
                                       tile_distances[candidate * TILE_SIZE + local_col],
                                       candidate_index, dim);
        }
        const int64_t partial_index = col * row_tiles + tile_row;
        col_partial_distances[partial_index] = best_distance;
        col_partial_indices[partial_index] = best_index;
        col_partial_ties[partial_index] = tied;
    }
}

__global__ void partial_nearest_reduce_kernel(const double* partial_distances,
                                              const int64_t* partial_indices,
                                              const int* partial_ties, const int64_t partial_count,
                                              const int64_t feature_dim, int64_t* indices,
                                              double* nearest_distances, int* ambiguous) {
    __shared__ double shared_distances[REDUCTION_THREADS];
    __shared__ int64_t shared_indices[REDUCTION_THREADS];
    __shared__ int shared_ties[REDUCTION_THREADS];

    const int64_t output_index = static_cast<int64_t>(blockIdx.x);
    double best_distance = DBL_MAX;
    int64_t best_index = -1;
    int tied = 0;
    for (int64_t candidate = threadIdx.x; candidate < partial_count; candidate += blockDim.x) {
        const int64_t partial_index = output_index * partial_count + candidate;
        hnw::matching::merge_best(best_distance, best_index, tied, partial_distances[partial_index],
                                  partial_indices[partial_index], partial_ties[partial_index],
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
    const int64_t row_tiles = (n1 + TILE_SIZE - 1) / TILE_SIZE;
    const int64_t col_tiles = (n2 + TILE_SIZE - 1) / TILE_SIZE;
    const size_t row_partial_count = static_cast<size_t>(n1) * static_cast<size_t>(col_tiles);
    const size_t col_partial_count = static_cast<size_t>(n2) * static_cast<size_t>(row_tiles);
    const size_t row_partial_distances_bytes = row_partial_count * sizeof(double);
    const size_t row_partial_indices_bytes = row_partial_count * sizeof(int64_t);
    const size_t row_partial_ties_bytes = row_partial_count * sizeof(int);
    const size_t col_partial_distances_bytes = col_partial_count * sizeof(double);
    const size_t col_partial_indices_bytes = col_partial_count * sizeof(int64_t);
    const size_t col_partial_ties_bytes = col_partial_count * sizeof(int);
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
        auto* row_partial_distances_device = static_cast<double*>(workspace.device_buffer(
            row_partial_distances_bytes,
            "matching bidirectional nearest cudaMalloc(row partial distances)"));
        auto* row_partial_indices_device = static_cast<int64_t*>(workspace.device_buffer(
            row_partial_indices_bytes,
            "matching bidirectional nearest cudaMalloc(row partial indices)"));
        auto* row_partial_ties_device = static_cast<int*>(workspace.device_buffer(
            row_partial_ties_bytes, "matching bidirectional nearest cudaMalloc(row partial ties)"));
        auto* col_partial_distances_device = static_cast<double*>(workspace.device_buffer(
            col_partial_distances_bytes,
            "matching bidirectional nearest cudaMalloc(col partial distances)"));
        auto* col_partial_indices_device = static_cast<int64_t*>(workspace.device_buffer(
            col_partial_indices_bytes,
            "matching bidirectional nearest cudaMalloc(col partial indices)"));
        auto* col_partial_ties_device = static_cast<int*>(workspace.device_buffer(
            col_partial_ties_bytes, "matching bidirectional nearest cudaMalloc(col partial ties)"));
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
            static_cast<int>((n1 + REDUCTION_THREADS - 1) / REDUCTION_THREADS);
        const int feature2_blocks =
            static_cast<int>((n2 + REDUCTION_THREADS - 1) / REDUCTION_THREADS);
        feature_norms_kernel<<<feature1_blocks, REDUCTION_THREADS, 0, stream>>>(
            features1_device, norms1_device, n1, dim, ambiguous_device);
        feature_norms_kernel<<<feature2_blocks, REDUCTION_THREADS, 0, stream>>>(
            features2_device, norms2_device, n2, dim, ambiguous_device);
        const int64_t tile_count = row_tiles * col_tiles;
        tiled_cosine_partial_nearest_kernel<<<static_cast<int>(tile_count), TILE_THREADS, 0,
                                              stream>>>(
            features1_device, features2_device, norms1_device, norms2_device,
            row_partial_distances_device, row_partial_indices_device, row_partial_ties_device,
            col_partial_distances_device, col_partial_indices_device, col_partial_ties_device, n1,
            n2, dim, row_tiles, col_tiles, ambiguous_device);
        partial_nearest_reduce_kernel<<<static_cast<int>(n1), REDUCTION_THREADS, 0, stream>>>(
            row_partial_distances_device, row_partial_indices_device, row_partial_ties_device,
            col_tiles, dim, row_indices_device, row_distances_device, ambiguous_device);
        partial_nearest_reduce_kernel<<<static_cast<int>(n2), REDUCTION_THREADS, 0, stream>>>(
            col_partial_distances_device, col_partial_indices_device, col_partial_ties_device,
            row_tiles, dim, col_indices_device, col_distances_device, ambiguous_device);
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
