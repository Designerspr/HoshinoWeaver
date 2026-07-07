#include "common/compat.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

struct SigmaClipCudaHostIoCache {
    void* stack = nullptr;
    void* mask = nullptr;
    double* out_sum = nullptr;
    double* out_sq = nullptr;
    double* out_n = nullptr;
    cudaStream_t stream = nullptr;
    size_t stack_capacity = 0;
    size_t mask_capacity = 0;
    size_t out_sum_capacity = 0;
    size_t out_sq_capacity = 0;
    size_t out_n_capacity = 0;
    int device = -1;

    ~SigmaClipCudaHostIoCache() {
        int current_device = -1;
        const cudaError_t get_device_error = cudaGetDevice(&current_device);
        if (get_device_error == cudaSuccess && device >= 0 && current_device != device) {
            cudaSetDevice(device);
        }
        cudaFree(stack);
        cudaFree(mask);
        cudaFree(out_sum);
        cudaFree(out_sq);
        cudaFree(out_n);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        if (get_device_error == cudaSuccess && device >= 0 && current_device != device) {
            cudaSetDevice(current_device);
        }
    }
};

thread_local SigmaClipCudaHostIoCache sigma_clip_host_io_cache;

void throw_if_cuda_failed(const cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(error));
    }
}

void ensure_device_buffer(void** ptr,
                          size_t* capacity,
                          const size_t required_bytes,
                          const char* context) {
    if (required_bytes <= *capacity) {
        return;
    }
    if (*ptr != nullptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
        *capacity = 0;
    }
    void* new_ptr = nullptr;
    throw_if_cuda_failed(cudaMalloc(&new_ptr, required_bytes), context);
    *ptr = new_ptr;
    *capacity = required_bytes;
}

void ensure_double_buffer(double** ptr,
                          size_t* capacity,
                          const size_t required_bytes,
                          const char* context) {
    void* raw_ptr = static_cast<void*>(*ptr);
    ensure_device_buffer(&raw_ptr, capacity, required_bytes, context);
    *ptr = static_cast<double*>(raw_ptr);
}

void clear_device_cache(SigmaClipCudaHostIoCache* cache) {
    cudaFree(cache->stack);
    cudaFree(cache->mask);
    cudaFree(cache->out_sum);
    cudaFree(cache->out_sq);
    cudaFree(cache->out_n);
    cache->stack = nullptr;
    cache->mask = nullptr;
    cache->out_sum = nullptr;
    cache->out_sq = nullptr;
    cache->out_n = nullptr;
    cache->stack_capacity = 0;
    cache->mask_capacity = 0;
    cache->out_sum_capacity = 0;
    cache->out_sq_capacity = 0;
    cache->out_n_capacity = 0;
}

void clear_cuda_host_io_cache(SigmaClipCudaHostIoCache* cache) {
    clear_device_cache(cache);
    if (cache->stream != nullptr) {
        cudaStreamDestroy(cache->stream);
        cache->stream = nullptr;
    }
    cache->device = -1;
}

void ensure_stream(SigmaClipCudaHostIoCache* cache) {
    if (cache->stream != nullptr) {
        return;
    }
    throw_if_cuda_failed(cudaStreamCreateWithFlags(&cache->stream, cudaStreamNonBlocking),
                         "sigma_clip_fused_chunk_cuda cudaStreamCreate");
}

void prepare_cuda_host_io_cache(SigmaClipCudaHostIoCache* cache) {
    int current_device = -1;
    throw_if_cuda_failed(cudaGetDevice(&current_device),
                         "sigma_clip_fused_chunk_cuda cudaGetDevice");
    if (cache->device == current_device) {
        ensure_stream(cache);
        return;
    }
    if (cache->device >= 0) {
        const int restore_device = current_device;
        throw_if_cuda_failed(cudaSetDevice(cache->device),
                             "sigma_clip_fused_chunk_cuda cudaSetDevice(old)");
        clear_cuda_host_io_cache(cache);
        throw_if_cuda_failed(cudaSetDevice(restore_device),
                             "sigma_clip_fused_chunk_cuda cudaSetDevice(restore)");
    }
    cache->device = current_device;
    ensure_stream(cache);
}

void reset_cuda_host_io_cache_after_error(SigmaClipCudaHostIoCache* cache) {
    if (cache->stream != nullptr) {
        cudaStreamSynchronize(cache->stream);
    }
    clear_cuda_host_io_cache(cache);
}

template <typename T>
__device__ inline double dtype_max_value() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return 255.0;
    }
    return 65535.0;
}

template <typename T>
__device__ inline bool is_zero_rgb_sample(
    const T* stack,
    const int64_t frame_offset,
    const int64_t idx,
    const int64_t channels) {
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
__device__ inline bool sample_is_valid(
    const T* stack,
    const uint8_t* mask,
    const int64_t frame,
    const int64_t plane_size,
    const int64_t idx,
    const bool skip_zero_rgb,
    const int64_t channels) {
    const int64_t frame_offset = frame * plane_size;
    if (mask != nullptr && !mask[frame_offset + idx]) {
        return false;
    }
    if (skip_zero_rgb && channels >= 3 &&
        is_zero_rgb_sample(stack, frame_offset, idx, channels)) {
        return false;
    }
    return true;
}

template <typename T>
__global__ void sigma_clip_fused_chunk_kernel(
    const T* stack,
    const uint8_t* mask,
    double* out_sum,
    double* out_sq,
    double* out_n,
    const int64_t n_frames,
    const int64_t plane_size,
    const double rej_high,
    const double rej_low,
    const int max_iter,
    const bool skip_zero_rgb,
    const int64_t channels) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane_size) {
        return;
    }

    double total_sum = 0.0;
    double total_sq = 0.0;
    double total_n = 0.0;
    for (int64_t frame = 0; frame < n_frames; ++frame) {
        if (!sample_is_valid(
                stack, mask, frame, plane_size, idx, skip_zero_rgb, channels)) {
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
            if (!sample_is_valid(
                    stack, mask, frame, plane_size, idx, skip_zero_rgb, channels)) {
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
void launch_sigma_clip_fused_chunk_cuda_impl(
    const T* stack_host,
    const uint8_t* mask_host,
    double* out_sum_host,
    double* out_sq_host,
    double* out_n_host,
    const int64_t n_frames,
    const int64_t plane_size,
    const double rej_high,
    const double rej_low,
    const int max_iter,
    const bool skip_zero_rgb,
    const int64_t channels,
    const char* op_name) {
    if (n_frames <= 0 || plane_size <= 0) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk_cuda: stack dimensions must be positive");
    }
    if (channels <= 0) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk_cuda: channels must be positive");
    }
    const size_t stack_count =
        static_cast<size_t>(n_frames) * static_cast<size_t>(plane_size);
    if (stack_count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::invalid_argument("sigma_clip_fused_chunk_cuda: stack is too large");
    }

    SigmaClipCudaHostIoCache* cache = &sigma_clip_host_io_cache;
    try {
        prepare_cuda_host_io_cache(cache);

        const size_t stack_bytes = stack_count * sizeof(T);
        const size_t mask_bytes = stack_count * sizeof(uint8_t);
        const size_t out_bytes = static_cast<size_t>(plane_size) * sizeof(double);
        ensure_device_buffer(
            &cache->stack,
            &cache->stack_capacity,
            stack_bytes,
            "sigma_clip_fused_chunk_cuda cudaMalloc(stack)");
        ensure_double_buffer(
            &cache->out_sum,
            &cache->out_sum_capacity,
            out_bytes,
            "sigma_clip_fused_chunk_cuda cudaMalloc(out_sum)");
        ensure_double_buffer(
            &cache->out_sq,
            &cache->out_sq_capacity,
            out_bytes,
            "sigma_clip_fused_chunk_cuda cudaMalloc(out_sq)");
        ensure_double_buffer(
            &cache->out_n,
            &cache->out_n_capacity,
            out_bytes,
            "sigma_clip_fused_chunk_cuda cudaMalloc(out_n)");

        uint8_t* mask_device = nullptr;
        if (mask_host != nullptr) {
            ensure_device_buffer(
                &cache->mask,
                &cache->mask_capacity,
                mask_bytes,
                "sigma_clip_fused_chunk_cuda cudaMalloc(mask)");
            mask_device = static_cast<uint8_t*>(cache->mask);
        }

        throw_if_cuda_failed(cudaMemcpyAsync(
                                 cache->stack,
                                 stack_host,
                                 stack_bytes,
                                 cudaMemcpyHostToDevice,
                                 cache->stream),
                             "sigma_clip_fused_chunk_cuda cudaMemcpy(stack)");
        if (mask_host != nullptr) {
            throw_if_cuda_failed(cudaMemcpyAsync(
                                     mask_device,
                                     mask_host,
                                     mask_bytes,
                                     cudaMemcpyHostToDevice,
                                     cache->stream),
                                 "sigma_clip_fused_chunk_cuda cudaMemcpy(mask)");
        }

        const int blocks =
            static_cast<int>((plane_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        sigma_clip_fused_chunk_kernel<T><<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->stack),
            mask_device,
            cache->out_sum,
            cache->out_sq,
            cache->out_n,
            n_frames,
            plane_size,
            rej_high,
            rej_low,
            max_iter,
            skip_zero_rgb,
            channels);
        throw_if_cuda_failed(cudaGetLastError(), op_name);

        throw_if_cuda_failed(cudaMemcpyAsync(
                                 out_sum_host,
                                 cache->out_sum,
                                 out_bytes,
                                 cudaMemcpyDeviceToHost,
                                 cache->stream),
                             "sigma_clip_fused_chunk_cuda cudaMemcpy(out_sum)");
        throw_if_cuda_failed(cudaMemcpyAsync(
                                 out_sq_host,
                                 cache->out_sq,
                                 out_bytes,
                                 cudaMemcpyDeviceToHost,
                                 cache->stream),
                             "sigma_clip_fused_chunk_cuda cudaMemcpy(out_sq)");
        throw_if_cuda_failed(cudaMemcpyAsync(
                                 out_n_host,
                                 cache->out_n,
                                 out_bytes,
                                 cudaMemcpyDeviceToHost,
                                 cache->stream),
                             "sigma_clip_fused_chunk_cuda cudaMemcpy(out_n)");
        throw_if_cuda_failed(cudaStreamSynchronize(cache->stream),
                             "sigma_clip_fused_chunk_cuda cudaStreamSynchronize");
    } catch (...) {
        reset_cuda_host_io_cache_after_error(cache);
        throw;
    }
}

}  // namespace

void launch_sigma_clip_fused_chunk_cuda_u8(
    const uint8_t* stack_host,
    const uint8_t* mask_host,
    double* out_sum_host,
    double* out_sq_host,
    double* out_n_host,
    const int64_t n_frames,
    const int64_t plane_size,
    const double rej_high,
    const double rej_low,
    const int max_iter,
    const bool skip_zero_rgb,
    const int64_t channels) {
    launch_sigma_clip_fused_chunk_cuda_impl<uint8_t>(
        stack_host,
        mask_host,
        out_sum_host,
        out_sq_host,
        out_n_host,
        n_frames,
        plane_size,
        rej_high,
        rej_low,
        max_iter,
        skip_zero_rgb,
        channels,
        "sigma_clip_fused_chunk_cuda kernel launch");
}

void launch_sigma_clip_fused_chunk_cuda_u16(
    const uint16_t* stack_host,
    const uint8_t* mask_host,
    double* out_sum_host,
    double* out_sq_host,
    double* out_n_host,
    const int64_t n_frames,
    const int64_t plane_size,
    const double rej_high,
    const double rej_low,
    const int max_iter,
    const bool skip_zero_rgb,
    const int64_t channels) {
    launch_sigma_clip_fused_chunk_cuda_impl<uint16_t>(
        stack_host,
        mask_host,
        out_sum_host,
        out_sq_host,
        out_n_host,
        n_frames,
        plane_size,
        rej_high,
        rej_low,
        max_iter,
        skip_zero_rgb,
        channels,
        "sigma_clip_fused_chunk_cuda kernel launch");
}
