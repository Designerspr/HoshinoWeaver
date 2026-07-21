#include "common/compat.h"
#include "common/cuda_host_io_workspace.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

template <typename T>
__global__ void huber_weighted_chunk_kernel(const T* stack, const double* ref_mean,
                                            const double* ref_std, const double* weights,
                                            double* weighted_sum, double* weight_total,
                                            const int64_t n_frames, const int64_t plane_size,
                                            const double huber_c) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane_size) {
        return;
    }

    const float mean = static_cast<float>(ref_mean[idx]);
    const float std_val = static_cast<float>(ref_std[idx]);
    const float c = static_cast<float>(huber_c);
    double sum = 0.0;
    double total = 0.0;
    for (int64_t frame = 0; frame < n_frames; ++frame) {
        const float value = static_cast<float>(stack[frame * plane_size + idx]);
        const float residual = (value - mean) / (std_val + 1.0e-10f);
        const float abs_residual = fabsf(residual);
        float huber_weight = abs_residual <= c ? 1.0f : c / (abs_residual + 1.0e-10f);
        double final_weight = static_cast<double>(huber_weight);
        if (weights != nullptr) {
            final_weight *= weights[frame];
        }
        sum += static_cast<double>(value) * final_weight;
        total += final_weight;
    }
    weighted_sum[idx] = sum;
    weight_total[idx] = total;
}

template <typename T>
void launch_huber_weighted_chunk_cuda_impl(const T* stack_host, const double* ref_mean_host,
                                           const double* ref_std_host, const double* weights_host,
                                           double* weighted_sum_host, double* weight_total_host,
                                           const int64_t n_frames, const int64_t plane_size,
                                           const double huber_c, const char* op_name) {
    if (n_frames <= 0 || plane_size <= 0) {
        throw std::invalid_argument("huber_weighted_chunk_cuda: stack dimensions must be positive");
    }
    const size_t stack_count = static_cast<size_t>(n_frames) * static_cast<size_t>(plane_size);
    if (stack_count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::invalid_argument("huber_weighted_chunk_cuda: stack is too large");
    }

    auto workspace =
        hnw::cuda::acquire_host_io_workspace("huber_weighted_chunk_cuda cudaGetDevice");
    try {
        const size_t stack_bytes = stack_count * sizeof(T);
        const size_t plane_bytes = static_cast<size_t>(plane_size) * sizeof(double);
        const size_t weights_bytes = static_cast<size_t>(n_frames) * sizeof(double);
        cudaStream_t stream = workspace.stream();
        void* stack_device =
            workspace.device_buffer(stack_bytes, "huber_weighted_chunk_cuda cudaMalloc(stack)");
        auto* ref_mean_device = static_cast<double*>(
            workspace.device_buffer(plane_bytes, "huber_weighted_chunk_cuda cudaMalloc(ref_mean)"));
        auto* ref_std_device = static_cast<double*>(
            workspace.device_buffer(plane_bytes, "huber_weighted_chunk_cuda cudaMalloc(ref_std)"));
        auto* weighted_sum_device = static_cast<double*>(workspace.device_buffer(
            plane_bytes, "huber_weighted_chunk_cuda cudaMalloc(weighted_sum)"));
        auto* weight_total_device = static_cast<double*>(workspace.device_buffer(
            plane_bytes, "huber_weighted_chunk_cuda cudaMalloc(weight_total)"));

        double* weights_device = nullptr;
        if (weights_host != nullptr) {
            weights_device = static_cast<double*>(workspace.device_buffer(
                weights_bytes, "huber_weighted_chunk_cuda cudaMalloc(weights)"));
        }

        hnw::cuda::throw_if_failed(
            cudaMemcpyAsync(stack_device, stack_host, stack_bytes, cudaMemcpyHostToDevice, stream),
            "huber_weighted_chunk_cuda cudaMemcpy(stack)");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(ref_mean_device, ref_mean_host, plane_bytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "huber_weighted_chunk_cuda cudaMemcpy(ref_mean)");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(ref_std_device, ref_std_host, plane_bytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "huber_weighted_chunk_cuda cudaMemcpy(ref_std)");
        if (weights_host != nullptr) {
            hnw::cuda::throw_if_failed(cudaMemcpyAsync(weights_device, weights_host, weights_bytes,
                                                       cudaMemcpyHostToDevice, stream),
                                       "huber_weighted_chunk_cuda cudaMemcpy(weights)");
        }

        const int blocks =
            static_cast<int>((plane_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        huber_weighted_chunk_kernel<T><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            static_cast<const T*>(stack_device), ref_mean_device, ref_std_device, weights_device,
            weighted_sum_device, weight_total_device, n_frames, plane_size, huber_c);
        hnw::cuda::throw_if_failed(cudaGetLastError(), op_name);

        hnw::cuda::throw_if_failed(cudaMemcpyAsync(weighted_sum_host, weighted_sum_device,
                                                   plane_bytes, cudaMemcpyDeviceToHost, stream),
                                   "huber_weighted_chunk_cuda cudaMemcpy(weighted_sum)");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(weight_total_host, weight_total_device,
                                                   plane_bytes, cudaMemcpyDeviceToHost, stream),
                                   "huber_weighted_chunk_cuda cudaMemcpy(weight_total)");
        hnw::cuda::throw_if_failed(cudaStreamSynchronize(stream),
                                   "huber_weighted_chunk_cuda cudaStreamSynchronize");
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}

} // namespace

void launch_huber_weighted_chunk_cuda_u8(const uint8_t* stack_host, const double* ref_mean_host,
                                         const double* ref_std_host, const double* weights_host,
                                         double* weighted_sum_host, double* weight_total_host,
                                         const int64_t n_frames, const int64_t plane_size,
                                         const double huber_c) {
    launch_huber_weighted_chunk_cuda_impl<uint8_t>(
        stack_host, ref_mean_host, ref_std_host, weights_host, weighted_sum_host, weight_total_host,
        n_frames, plane_size, huber_c, "huber_weighted_chunk_cuda kernel launch");
}

void launch_huber_weighted_chunk_cuda_u16(const uint16_t* stack_host, const double* ref_mean_host,
                                          const double* ref_std_host, const double* weights_host,
                                          double* weighted_sum_host, double* weight_total_host,
                                          const int64_t n_frames, const int64_t plane_size,
                                          const double huber_c) {
    launch_huber_weighted_chunk_cuda_impl<uint16_t>(
        stack_host, ref_mean_host, ref_std_host, weights_host, weighted_sum_host, weight_total_host,
        n_frames, plane_size, huber_c, "huber_weighted_chunk_cuda kernel launch");
}
