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

template <typename T> __device__ inline double dtype_max_value() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return 255.0;
    }
    return 65535.0;
}

template <typename T>
__device__ inline bool is_zero_rgb_sample(const T* stack, const int64_t frame_offset,
                                          const int64_t idx, const int64_t channels) {
    const int64_t base = (idx / channels) * channels;
    const int64_t limit = channels < 3 ? channels : 3;
    for (int64_t channel = 0; channel < limit; ++channel) {
        if (stack[frame_offset + base + channel] != static_cast<T>(0)) {
            return false;
        }
    }
    return true;
}

template <typename T>
__device__ inline bool sample_is_valid(const T* stack, const uint8_t* mask, const int64_t frame,
                                       const int64_t plane_size, const int64_t idx,
                                       const bool skip_zero_rgb, const int64_t channels) {
    const int64_t frame_offset = frame * plane_size;
    if (mask != nullptr && !mask[frame_offset + idx]) {
        return false;
    }
    if (skip_zero_rgb && channels >= 3 && is_zero_rgb_sample(stack, frame_offset, idx, channels)) {
        return false;
    }
    return true;
}

template <typename T>
__global__ void sigma_clip_fused_chunk_kernel(const T* stack, const uint8_t* mask, double* out_sum,
                                              double* out_sq, double* out_n, const int64_t n_frames,
                                              const int64_t plane_size, const double rej_high,
                                              const double rej_low, const int max_iter,
                                              const bool skip_zero_rgb, const int64_t channels) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane_size) {
        return;
    }

    double total_sum = 0.0;
    double total_sq = 0.0;
    double total_n = 0.0;
    for (int64_t frame = 0; frame < n_frames; ++frame) {
        if (!sample_is_valid(stack, mask, frame, plane_size, idx, skip_zero_rgb, channels)) {
            continue;
        }
        const double value = static_cast<double>(stack[frame * plane_size + idx]);
        total_sum += value;
        total_sq += value * value;
        total_n += 1.0;
    }

    double cur_sum = total_sum;
    double cur_sq = total_sq;
    double cur_n = total_n;
    const double dtype_min = 0.0;
    const double dtype_max = dtype_max_value<T>();

    for (int iter = 0; iter < max_iter; ++iter) {
        if (cur_n <= 1.0) {
            break;
        }
        const double mu = cur_sum / cur_n;
        const double var = (cur_sq - cur_sum * cur_sum / cur_n) / (cur_n - 1.0);
        const double std_val = sqrt(fmax(var, 0.0));
        const double high = fmin(floor(mu + std_val * rej_high), dtype_max);
        const double low = fmax(ceil(mu - std_val * rej_low), dtype_min);

        double rej_sum = 0.0;
        double rej_sq = 0.0;
        double rej_n = 0.0;
        for (int64_t frame = 0; frame < n_frames; ++frame) {
            if (!sample_is_valid(stack, mask, frame, plane_size, idx, skip_zero_rgb, channels)) {
                continue;
            }
            const double value = static_cast<double>(stack[frame * plane_size + idx]);
            if (value < low || value > high) {
                rej_sum += value;
                rej_sq += value * value;
                rej_n += 1.0;
            }
        }

        const double new_n = total_n - rej_n;
        const double new_sum = total_sum - rej_sum;
        const double new_sq = total_sq - rej_sq;
        if (new_n == cur_n && new_sum == cur_sum && new_sq == cur_sq) {
            break;
        }
        if (new_n <= 0.0) {
            cur_sum = total_sum;
            cur_sq = total_sq;
            cur_n = total_n;
            break;
        }
        cur_sum = new_sum;
        cur_sq = new_sq;
        cur_n = new_n;
    }

    out_sum[idx] = cur_sum;
    out_sq[idx] = cur_sq;
    out_n[idx] = cur_n;
}

template <typename T>
void launch_sigma_clip_fused_chunk_cuda_impl(const T* stack_host, const uint8_t* mask_host,
                                             double* out_sum_host, double* out_sq_host,
                                             double* out_n_host, const int64_t n_frames,
                                             const int64_t plane_size, const double rej_high,
                                             const double rej_low, const int max_iter,
                                             const bool skip_zero_rgb, const int64_t channels,
                                             const char* op_name) {
    if (n_frames <= 0 || plane_size <= 0) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk_cuda: stack dimensions must be positive");
    }
    if (channels <= 0) {
        throw std::invalid_argument("sigma_clip_fused_chunk_cuda: channels must be positive");
    }
    const size_t stack_count = static_cast<size_t>(n_frames) * static_cast<size_t>(plane_size);
    if (stack_count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::invalid_argument("sigma_clip_fused_chunk_cuda: stack is too large");
    }

    auto workspace =
        hnw::cuda::acquire_host_io_workspace("sigma_clip_fused_chunk_cuda cudaGetDevice");
    try {
        const size_t stack_bytes = stack_count * sizeof(T);
        const size_t mask_bytes = stack_count * sizeof(uint8_t);
        const size_t out_bytes = static_cast<size_t>(plane_size) * sizeof(double);
        cudaStream_t stream = workspace.stream();
        void* stack_device =
            workspace.device_buffer(stack_bytes, "sigma_clip_fused_chunk_cuda cudaMalloc(stack)");
        auto* out_sum_device = static_cast<double*>(
            workspace.device_buffer(out_bytes, "sigma_clip_fused_chunk_cuda cudaMalloc(out_sum)"));
        auto* out_sq_device = static_cast<double*>(
            workspace.device_buffer(out_bytes, "sigma_clip_fused_chunk_cuda cudaMalloc(out_sq)"));
        auto* out_n_device = static_cast<double*>(
            workspace.device_buffer(out_bytes, "sigma_clip_fused_chunk_cuda cudaMalloc(out_n)"));

        uint8_t* mask_device = nullptr;
        if (mask_host != nullptr) {
            mask_device = static_cast<uint8_t*>(workspace.device_buffer(
                mask_bytes, "sigma_clip_fused_chunk_cuda cudaMalloc(mask)"));
        }

        hnw::cuda::throw_if_failed(
            cudaMemcpyAsync(stack_device, stack_host, stack_bytes, cudaMemcpyHostToDevice, stream),
            "sigma_clip_fused_chunk_cuda cudaMemcpy(stack)");
        if (mask_host != nullptr) {
            hnw::cuda::throw_if_failed(
                cudaMemcpyAsync(mask_device, mask_host, mask_bytes, cudaMemcpyHostToDevice, stream),
                "sigma_clip_fused_chunk_cuda cudaMemcpy(mask)");
        }

        const int blocks =
            static_cast<int>((plane_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        sigma_clip_fused_chunk_kernel<T><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            static_cast<const T*>(stack_device), mask_device, out_sum_device, out_sq_device,
            out_n_device, n_frames, plane_size, rej_high, rej_low, max_iter, skip_zero_rgb,
            channels);
        hnw::cuda::throw_if_failed(cudaGetLastError(), op_name);

        hnw::cuda::throw_if_failed(cudaMemcpyAsync(out_sum_host, out_sum_device, out_bytes,
                                                   cudaMemcpyDeviceToHost, stream),
                                   "sigma_clip_fused_chunk_cuda cudaMemcpy(out_sum)");
        hnw::cuda::throw_if_failed(
            cudaMemcpyAsync(out_sq_host, out_sq_device, out_bytes, cudaMemcpyDeviceToHost, stream),
            "sigma_clip_fused_chunk_cuda cudaMemcpy(out_sq)");
        hnw::cuda::throw_if_failed(
            cudaMemcpyAsync(out_n_host, out_n_device, out_bytes, cudaMemcpyDeviceToHost, stream),
            "sigma_clip_fused_chunk_cuda cudaMemcpy(out_n)");
        hnw::cuda::throw_if_failed(cudaStreamSynchronize(stream),
                                   "sigma_clip_fused_chunk_cuda cudaStreamSynchronize");
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}

} // namespace

void launch_sigma_clip_fused_chunk_cuda_u8(const uint8_t* stack_host, const uint8_t* mask_host,
                                           double* out_sum_host, double* out_sq_host,
                                           double* out_n_host, const int64_t n_frames,
                                           const int64_t plane_size, const double rej_high,
                                           const double rej_low, const int max_iter,
                                           const bool skip_zero_rgb, const int64_t channels) {
    launch_sigma_clip_fused_chunk_cuda_impl<uint8_t>(
        stack_host, mask_host, out_sum_host, out_sq_host, out_n_host, n_frames, plane_size,
        rej_high, rej_low, max_iter, skip_zero_rgb, channels,
        "sigma_clip_fused_chunk_cuda kernel launch");
}

void launch_sigma_clip_fused_chunk_cuda_u16(const uint16_t* stack_host, const uint8_t* mask_host,
                                            double* out_sum_host, double* out_sq_host,
                                            double* out_n_host, const int64_t n_frames,
                                            const int64_t plane_size, const double rej_high,
                                            const double rej_low, const int max_iter,
                                            const bool skip_zero_rgb, const int64_t channels) {
    launch_sigma_clip_fused_chunk_cuda_impl<uint16_t>(
        stack_host, mask_host, out_sum_host, out_sq_host, out_n_host, n_frames, plane_size,
        rej_high, rej_low, max_iter, skip_zero_rgb, channels,
        "sigma_clip_fused_chunk_cuda kernel launch");
}
