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

struct HuberChunkCudaHostIoCache {
    void* stack = nullptr;
    double* ref_mean = nullptr;
    double* ref_std = nullptr;
    double* weights = nullptr;
    double* weighted_sum = nullptr;
    double* weight_total = nullptr;
    cudaStream_t stream = nullptr;
    size_t stack_capacity = 0;
    size_t ref_mean_capacity = 0;
    size_t ref_std_capacity = 0;
    size_t weights_capacity = 0;
    size_t weighted_sum_capacity = 0;
    size_t weight_total_capacity = 0;
    int device = -1;

    ~HuberChunkCudaHostIoCache() {
        int current_device = -1;
        const cudaError_t get_device_error = cudaGetDevice(&current_device);
        if (get_device_error == cudaSuccess && device >= 0 && current_device != device) {
            cudaSetDevice(device);
        }
        cudaFree(stack);
        cudaFree(ref_mean);
        cudaFree(ref_std);
        cudaFree(weights);
        cudaFree(weighted_sum);
        cudaFree(weight_total);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        if (get_device_error == cudaSuccess && device >= 0 && current_device != device) {
            cudaSetDevice(current_device);
        }
    }
};

thread_local HuberChunkCudaHostIoCache huber_chunk_host_io_cache;

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

void clear_device_cache(HuberChunkCudaHostIoCache* cache) {
    cudaFree(cache->stack);
    cudaFree(cache->ref_mean);
    cudaFree(cache->ref_std);
    cudaFree(cache->weights);
    cudaFree(cache->weighted_sum);
    cudaFree(cache->weight_total);
    cache->stack = nullptr;
    cache->ref_mean = nullptr;
    cache->ref_std = nullptr;
    cache->weights = nullptr;
    cache->weighted_sum = nullptr;
    cache->weight_total = nullptr;
    cache->stack_capacity = 0;
    cache->ref_mean_capacity = 0;
    cache->ref_std_capacity = 0;
    cache->weights_capacity = 0;
    cache->weighted_sum_capacity = 0;
    cache->weight_total_capacity = 0;
}

void clear_cuda_host_io_cache(HuberChunkCudaHostIoCache* cache) {
    clear_device_cache(cache);
    if (cache->stream != nullptr) {
        cudaStreamDestroy(cache->stream);
        cache->stream = nullptr;
    }
    cache->device = -1;
}

void ensure_stream(HuberChunkCudaHostIoCache* cache) {
    if (cache->stream != nullptr) {
        return;
    }
    throw_if_cuda_failed(cudaStreamCreateWithFlags(&cache->stream, cudaStreamNonBlocking),
                         "huber_weighted_chunk_cuda cudaStreamCreate");
}

void prepare_cuda_host_io_cache(HuberChunkCudaHostIoCache* cache) {
    int current_device = -1;
    throw_if_cuda_failed(cudaGetDevice(&current_device),
                         "huber_weighted_chunk_cuda cudaGetDevice");
    if (cache->device == current_device) {
        ensure_stream(cache);
        return;
    }
    if (cache->device >= 0) {
        const int restore_device = current_device;
        throw_if_cuda_failed(cudaSetDevice(cache->device),
                             "huber_weighted_chunk_cuda cudaSetDevice(old)");
        clear_cuda_host_io_cache(cache);
        throw_if_cuda_failed(cudaSetDevice(restore_device),
                             "huber_weighted_chunk_cuda cudaSetDevice(restore)");
    }
    cache->device = current_device;
    ensure_stream(cache);
}

void reset_cuda_host_io_cache_after_error(HuberChunkCudaHostIoCache* cache) {
    if (cache->stream != nullptr) {
        cudaStreamSynchronize(cache->stream);
    }
    clear_cuda_host_io_cache(cache);
}

template <typename T>
__global__ void huber_weighted_chunk_kernel(
    const T* stack,
    const double* ref_mean,
    const double* ref_std,
    const double* weights,
    double* weighted_sum,
    double* weight_total,
    const int64_t n_frames,
    const int64_t plane_size,
    const double huber_c) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
void launch_huber_weighted_chunk_cuda_impl(
    const T* stack_host,
    const double* ref_mean_host,
    const double* ref_std_host,
    const double* weights_host,
    double* weighted_sum_host,
    double* weight_total_host,
    const int64_t n_frames,
    const int64_t plane_size,
    const double huber_c,
    const char* op_name) {
    if (n_frames <= 0 || plane_size <= 0) {
        throw std::invalid_argument(
            "huber_weighted_chunk_cuda: stack dimensions must be positive");
    }
    const size_t stack_count =
        static_cast<size_t>(n_frames) * static_cast<size_t>(plane_size);
    if (stack_count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::invalid_argument("huber_weighted_chunk_cuda: stack is too large");
    }

    HuberChunkCudaHostIoCache* cache = &huber_chunk_host_io_cache;
    try {
        prepare_cuda_host_io_cache(cache);

        const size_t stack_bytes = stack_count * sizeof(T);
        const size_t plane_bytes = static_cast<size_t>(plane_size) * sizeof(double);
        const size_t weights_bytes = static_cast<size_t>(n_frames) * sizeof(double);

        ensure_device_buffer(
            &cache->stack,
            &cache->stack_capacity,
            stack_bytes,
            "huber_weighted_chunk_cuda cudaMalloc(stack)");
        ensure_double_buffer(
            &cache->ref_mean,
            &cache->ref_mean_capacity,
            plane_bytes,
            "huber_weighted_chunk_cuda cudaMalloc(ref_mean)");
        ensure_double_buffer(
            &cache->ref_std,
            &cache->ref_std_capacity,
            plane_bytes,
            "huber_weighted_chunk_cuda cudaMalloc(ref_std)");
        ensure_double_buffer(
            &cache->weighted_sum,
            &cache->weighted_sum_capacity,
            plane_bytes,
            "huber_weighted_chunk_cuda cudaMalloc(weighted_sum)");
        ensure_double_buffer(
            &cache->weight_total,
            &cache->weight_total_capacity,
            plane_bytes,
            "huber_weighted_chunk_cuda cudaMalloc(weight_total)");

        double* weights_device = nullptr;
        if (weights_host != nullptr) {
            ensure_double_buffer(
                &cache->weights,
                &cache->weights_capacity,
                weights_bytes,
                "huber_weighted_chunk_cuda cudaMalloc(weights)");
            weights_device = cache->weights;
        }

        throw_if_cuda_failed(cudaMemcpyAsync(
                                 cache->stack,
                                 stack_host,
                                 stack_bytes,
                                 cudaMemcpyHostToDevice,
                                 cache->stream),
                             "huber_weighted_chunk_cuda cudaMemcpy(stack)");
        throw_if_cuda_failed(cudaMemcpyAsync(
                                 cache->ref_mean,
                                 ref_mean_host,
                                 plane_bytes,
                                 cudaMemcpyHostToDevice,
                                 cache->stream),
                             "huber_weighted_chunk_cuda cudaMemcpy(ref_mean)");
        throw_if_cuda_failed(cudaMemcpyAsync(
                                 cache->ref_std,
                                 ref_std_host,
                                 plane_bytes,
                                 cudaMemcpyHostToDevice,
                                 cache->stream),
                             "huber_weighted_chunk_cuda cudaMemcpy(ref_std)");
        if (weights_host != nullptr) {
            throw_if_cuda_failed(cudaMemcpyAsync(
                                     weights_device,
                                     weights_host,
                                     weights_bytes,
                                     cudaMemcpyHostToDevice,
                                     cache->stream),
                                 "huber_weighted_chunk_cuda cudaMemcpy(weights)");
        }

        const int blocks =
            static_cast<int>((plane_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        huber_weighted_chunk_kernel<T><<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->stack),
            cache->ref_mean,
            cache->ref_std,
            weights_device,
            cache->weighted_sum,
            cache->weight_total,
            n_frames,
            plane_size,
            huber_c);
        throw_if_cuda_failed(cudaGetLastError(), op_name);

        throw_if_cuda_failed(cudaMemcpyAsync(
                                 weighted_sum_host,
                                 cache->weighted_sum,
                                 plane_bytes,
                                 cudaMemcpyDeviceToHost,
                                 cache->stream),
                             "huber_weighted_chunk_cuda cudaMemcpy(weighted_sum)");
        throw_if_cuda_failed(cudaMemcpyAsync(
                                 weight_total_host,
                                 cache->weight_total,
                                 plane_bytes,
                                 cudaMemcpyDeviceToHost,
                                 cache->stream),
                             "huber_weighted_chunk_cuda cudaMemcpy(weight_total)");
        throw_if_cuda_failed(cudaStreamSynchronize(cache->stream),
                             "huber_weighted_chunk_cuda cudaStreamSynchronize");
    } catch (...) {
        reset_cuda_host_io_cache_after_error(cache);
        throw;
    }
}

}  // namespace

void launch_huber_weighted_chunk_cuda_u8(
    const uint8_t* stack_host,
    const double* ref_mean_host,
    const double* ref_std_host,
    const double* weights_host,
    double* weighted_sum_host,
    double* weight_total_host,
    const int64_t n_frames,
    const int64_t plane_size,
    const double huber_c) {
    launch_huber_weighted_chunk_cuda_impl<uint8_t>(
        stack_host,
        ref_mean_host,
        ref_std_host,
        weights_host,
        weighted_sum_host,
        weight_total_host,
        n_frames,
        plane_size,
        huber_c,
        "huber_weighted_chunk_cuda kernel launch");
}

void launch_huber_weighted_chunk_cuda_u16(
    const uint16_t* stack_host,
    const double* ref_mean_host,
    const double* ref_std_host,
    const double* weights_host,
    double* weighted_sum_host,
    double* weight_total_host,
    const int64_t n_frames,
    const int64_t plane_size,
    const double huber_c) {
    launch_huber_weighted_chunk_cuda_impl<uint16_t>(
        stack_host,
        ref_mean_host,
        ref_std_host,
        weights_host,
        weighted_sum_host,
        weight_total_host,
        n_frames,
        plane_size,
        huber_c,
        "huber_weighted_chunk_cuda kernel launch");
}
