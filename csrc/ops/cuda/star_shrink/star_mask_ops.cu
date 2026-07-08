#include "common/compat.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

struct StarMaskDogCudaHostIoCache {
    void* image = nullptr;
    float* gray = nullptr;
    float* tmp = nullptr;
    float* blur_small = nullptr;
    float* blur_large = nullptr;
    float* dog = nullptr;
    float* kernel_small = nullptr;
    float* kernel_large = nullptr;
    uint8_t* mask = nullptr;
    uint8_t* scratch = nullptr;
    double* block_sum = nullptr;
    double* block_sq = nullptr;
    cudaStream_t stream = nullptr;
    size_t image_capacity = 0;
    size_t gray_capacity = 0;
    size_t tmp_capacity = 0;
    size_t blur_small_capacity = 0;
    size_t blur_large_capacity = 0;
    size_t dog_capacity = 0;
    size_t kernel_small_capacity = 0;
    size_t kernel_large_capacity = 0;
    size_t mask_capacity = 0;
    size_t scratch_capacity = 0;
    size_t block_sum_capacity = 0;
    size_t block_sq_capacity = 0;
    int device = -1;

    ~StarMaskDogCudaHostIoCache() {
        int current_device = -1;
        const cudaError_t get_device_error = cudaGetDevice(&current_device);
        if (get_device_error == cudaSuccess && device >= 0 && current_device != device) {
            cudaSetDevice(device);
        }
        cudaFree(image);
        cudaFree(gray);
        cudaFree(tmp);
        cudaFree(blur_small);
        cudaFree(blur_large);
        cudaFree(dog);
        cudaFree(kernel_small);
        cudaFree(kernel_large);
        cudaFree(mask);
        cudaFree(scratch);
        cudaFree(block_sum);
        cudaFree(block_sq);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        if (get_device_error == cudaSuccess && device >= 0 && current_device != device) {
            cudaSetDevice(current_device);
        }
    }
};

thread_local StarMaskDogCudaHostIoCache star_mask_dog_cache;

void throw_if_cuda_failed(const cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(error));
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

void ensure_float_buffer(float** ptr,
                         size_t* capacity,
                         const size_t required_bytes,
                         const char* context) {
    void* raw_ptr = static_cast<void*>(*ptr);
    ensure_device_buffer(&raw_ptr, capacity, required_bytes, context);
    *ptr = static_cast<float*>(raw_ptr);
}

void ensure_double_buffer(double** ptr,
                          size_t* capacity,
                          const size_t required_bytes,
                          const char* context) {
    void* raw_ptr = static_cast<void*>(*ptr);
    ensure_device_buffer(&raw_ptr, capacity, required_bytes, context);
    *ptr = static_cast<double*>(raw_ptr);
}

void clear_cuda_host_io_cache(StarMaskDogCudaHostIoCache* cache) {
    cudaFree(cache->image);
    cudaFree(cache->gray);
    cudaFree(cache->tmp);
    cudaFree(cache->blur_small);
    cudaFree(cache->blur_large);
    cudaFree(cache->dog);
    cudaFree(cache->kernel_small);
    cudaFree(cache->kernel_large);
    cudaFree(cache->mask);
    cudaFree(cache->scratch);
    cudaFree(cache->block_sum);
    cudaFree(cache->block_sq);
    cache->image = nullptr;
    cache->gray = nullptr;
    cache->tmp = nullptr;
    cache->blur_small = nullptr;
    cache->blur_large = nullptr;
    cache->dog = nullptr;
    cache->kernel_small = nullptr;
    cache->kernel_large = nullptr;
    cache->mask = nullptr;
    cache->scratch = nullptr;
    cache->block_sum = nullptr;
    cache->block_sq = nullptr;
    cache->image_capacity = 0;
    cache->gray_capacity = 0;
    cache->tmp_capacity = 0;
    cache->blur_small_capacity = 0;
    cache->blur_large_capacity = 0;
    cache->dog_capacity = 0;
    cache->kernel_small_capacity = 0;
    cache->kernel_large_capacity = 0;
    cache->mask_capacity = 0;
    cache->scratch_capacity = 0;
    cache->block_sum_capacity = 0;
    cache->block_sq_capacity = 0;
    if (cache->stream != nullptr) {
        cudaStreamDestroy(cache->stream);
        cache->stream = nullptr;
    }
    cache->device = -1;
}

void ensure_stream(StarMaskDogCudaHostIoCache* cache) {
    if (cache->stream != nullptr) {
        return;
    }
    throw_if_cuda_failed(cudaStreamCreateWithFlags(&cache->stream, cudaStreamNonBlocking),
                         "star_mask_dog_cuda cudaStreamCreate");
}

void prepare_cuda_host_io_cache(StarMaskDogCudaHostIoCache* cache) {
    int current_device = -1;
    throw_if_cuda_failed(cudaGetDevice(&current_device), "star_mask_dog_cuda cudaGetDevice");
    if (cache->device == current_device) {
        ensure_stream(cache);
        return;
    }
    if (cache->device >= 0) {
        const int restore_device = current_device;
        throw_if_cuda_failed(cudaSetDevice(cache->device),
                             "star_mask_dog_cuda cudaSetDevice(old)");
        clear_cuda_host_io_cache(cache);
        throw_if_cuda_failed(cudaSetDevice(restore_device),
                             "star_mask_dog_cuda cudaSetDevice(restore)");
    }
    cache->device = current_device;
    ensure_stream(cache);
}

void reset_cuda_host_io_cache_after_error(StarMaskDogCudaHostIoCache* cache) {
    if (cache->stream != nullptr) {
        cudaStreamSynchronize(cache->stream);
    }
    clear_cuda_host_io_cache(cache);
}

template <typename T>
__device__ inline float dtype_max_value() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return 255.0f;
    } else {
        return 65535.0f;
    }
}

template <typename T>
__device__ inline float normalized_sample(const T* ptr, const int64_t idx) {
    return static_cast<float>(ptr[idx]) / dtype_max_value<T>();
}

__device__ inline int reflect101(int idx, const int length) {
    if (length <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= length) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * length - idx - 2;
        }
    }
    return idx;
}

template <typename T>
__global__ void gray_kernel(const T* image,
                            float* gray,
                            const int64_t plane_size,
                            const int channels) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plane_size) {
        return;
    }
    if (channels == 1) {
        gray[idx] = normalized_sample(image, idx);
        return;
    }
    const int64_t base = idx * 3;
    const float b = normalized_sample(image, base);
    const float g = normalized_sample(image, base + 1);
    const float r = normalized_sample(image, base + 2);
    gray[idx] = 0.114f * b + 0.587f * g + 0.299f * r;
}

__global__ void gaussian_horizontal_kernel(const float* input,
                                           float* tmp,
                                           const float* kernel,
                                           const int radius,
                                           const int height,
                                           const int width) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t plane_size = static_cast<int64_t>(height) * width;
    if (idx >= plane_size) {
        return;
    }
    const int y = static_cast<int>(idx / width);
    const int x = static_cast<int>(idx - static_cast<int64_t>(y) * width);
    float sum = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx) {
        const int xx = reflect101(x + dx, width);
        sum += input[static_cast<int64_t>(y) * width + xx] * kernel[dx + radius];
    }
    tmp[idx] = sum;
}

__global__ void gaussian_vertical_kernel(const float* tmp,
                                         float* output,
                                         const float* kernel,
                                         const int radius,
                                         const int height,
                                         const int width) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t plane_size = static_cast<int64_t>(height) * width;
    if (idx >= plane_size) {
        return;
    }
    const int y = static_cast<int>(idx / width);
    const int x = static_cast<int>(idx - static_cast<int64_t>(y) * width);
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int yy = reflect101(y + dy, height);
        sum += tmp[static_cast<int64_t>(yy) * width + x] * kernel[dy + radius];
    }
    output[idx] = sum;
}

__global__ void dog_reduce_kernel(const float* blur_small,
                                  const float* blur_large,
                                  float* dog,
                                  double* block_sum,
                                  double* block_sq,
                                  const int64_t plane_size) {
    __shared__ double shared_sum[THREADS_PER_BLOCK];
    __shared__ double shared_sq[THREADS_PER_BLOCK];
    const int tid = threadIdx.x;
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (idx < plane_size) {
        const float diff = blur_small[idx] - blur_large[idx];
        dog[idx] = diff;
        value = static_cast<double>(diff);
    }
    shared_sum[tid] = value;
    shared_sq[tid] = value * value;
    __syncthreads();
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sq[tid] += shared_sq[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sum[blockIdx.x] = shared_sum[0];
        block_sq[blockIdx.x] = shared_sq[0];
    }
}

__global__ void threshold_mask_kernel(const float* dog,
                                      uint8_t* mask,
                                      const int64_t plane_size,
                                      const float threshold) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plane_size) {
        return;
    }
    mask[idx] = dog[idx] > threshold ? 1 : 0;
}

__device__ inline bool cross_active(const int ksize, const int ky, const int kx) {
    const int radius = ksize / 2;
    return (ky - radius) == 0 || (kx - radius) == 0;
}

__global__ void erode_cross_kernel(const uint8_t* input,
                                   uint8_t* output,
                                   const int height,
                                   const int width,
                                   const int ksize) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t plane_size = static_cast<int64_t>(height) * width;
    if (idx >= plane_size) {
        return;
    }
    const int y = static_cast<int>(idx / width);
    const int x = static_cast<int>(idx - static_cast<int64_t>(y) * width);
    const int radius = ksize / 2;
    bool keep = true;
    for (int ky = 0; ky < ksize && keep; ++ky) {
        const int yy = y + ky - radius;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int kx = 0; kx < ksize; ++kx) {
            if (!cross_active(ksize, ky, kx)) {
                continue;
            }
            const int xx = x + kx - radius;
            if (xx < 0 || xx >= width) {
                continue;
            }
            if (input[static_cast<int64_t>(yy) * width + xx] == 0) {
                keep = false;
                break;
            }
        }
    }
    output[idx] = keep ? 1 : 0;
}

__global__ void dilate_cross_kernel(const uint8_t* input,
                                    uint8_t* output,
                                    const int height,
                                    const int width,
                                    const int ksize) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t plane_size = static_cast<int64_t>(height) * width;
    if (idx >= plane_size) {
        return;
    }
    const int y = static_cast<int>(idx / width);
    const int x = static_cast<int>(idx - static_cast<int64_t>(y) * width);
    const int radius = ksize / 2;
    bool keep = false;
    for (int ky = 0; ky < ksize && !keep; ++ky) {
        const int yy = y + ky - radius;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int kx = 0; kx < ksize; ++kx) {
            if (!cross_active(ksize, ky, kx)) {
                continue;
            }
            const int xx = x + kx - radius;
            if (xx < 0 || xx >= width) {
                continue;
            }
            if (input[static_cast<int64_t>(yy) * width + xx] != 0) {
                keep = true;
                break;
            }
        }
    }
    output[idx] = keep ? 1 : 0;
}

void launch_morphology(StarMaskDogCudaHostIoCache* cache,
                       const int height,
                       const int width,
                       const int blocks,
                       const int open_ksize,
                       const int dilate_ksize) {
    if (open_ksize > 0) {
        erode_cross_kernel<<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->mask, cache->scratch, height, width, open_ksize);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda erode_open");
        dilate_cross_kernel<<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->scratch, cache->mask, height, width, open_ksize);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda dilate_open");
    }
    if (dilate_ksize > 0) {
        dilate_cross_kernel<<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->mask, cache->scratch, height, width, dilate_ksize);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda dilate");
        throw_if_cuda_failed(cudaMemcpyAsync(cache->mask,
                                             cache->scratch,
                                             static_cast<size_t>(height) * width,
                                             cudaMemcpyDeviceToDevice,
                                             cache->stream),
                             "star_mask_dog_cuda copy dilated mask");
    }
}

template <typename T>
void launch_star_mask_dog_cuda_impl(const T* image_host,
                                    uint8_t* mask_host,
                                    const int height,
                                    const int width,
                                    const int channels,
                                    const float* small_kernel_host,
                                    const int small_radius,
                                    const float* large_kernel_host,
                                    const int large_radius,
                                    const float threshold_ratio,
                                    const int open_ksize,
                                    const int dilate_ksize) {
    auto* cache = &star_mask_dog_cache;
    try {
        prepare_cuda_host_io_cache(cache);
        const int64_t plane_size = static_cast<int64_t>(height) * width;
        const int64_t total = plane_size * channels;
        const int blocks = static_cast<int>((plane_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        const size_t image_bytes = static_cast<size_t>(total) * sizeof(T);
        const size_t plane_float_bytes = static_cast<size_t>(plane_size) * sizeof(float);
        const size_t mask_bytes = static_cast<size_t>(plane_size) * sizeof(uint8_t);
        const size_t reduction_bytes = static_cast<size_t>(blocks) * sizeof(double);
        const size_t small_kernel_bytes = static_cast<size_t>(2 * small_radius + 1) * sizeof(float);
        const size_t large_kernel_bytes = static_cast<size_t>(2 * large_radius + 1) * sizeof(float);

        ensure_device_buffer(&cache->image, &cache->image_capacity, image_bytes, "star_mask_dog cudaMalloc image");
        ensure_float_buffer(&cache->gray, &cache->gray_capacity, plane_float_bytes, "star_mask_dog cudaMalloc gray");
        ensure_float_buffer(&cache->tmp, &cache->tmp_capacity, plane_float_bytes, "star_mask_dog cudaMalloc tmp");
        ensure_float_buffer(&cache->blur_small, &cache->blur_small_capacity, plane_float_bytes, "star_mask_dog cudaMalloc blur_small");
        ensure_float_buffer(&cache->blur_large, &cache->blur_large_capacity, plane_float_bytes, "star_mask_dog cudaMalloc blur_large");
        ensure_float_buffer(&cache->dog, &cache->dog_capacity, plane_float_bytes, "star_mask_dog cudaMalloc dog");
        ensure_float_buffer(&cache->kernel_small, &cache->kernel_small_capacity, small_kernel_bytes, "star_mask_dog cudaMalloc small_kernel");
        ensure_float_buffer(&cache->kernel_large, &cache->kernel_large_capacity, large_kernel_bytes, "star_mask_dog cudaMalloc large_kernel");
        void* mask_raw = static_cast<void*>(cache->mask);
        ensure_device_buffer(&mask_raw, &cache->mask_capacity, mask_bytes, "star_mask_dog cudaMalloc mask");
        cache->mask = static_cast<uint8_t*>(mask_raw);
        void* scratch_raw = static_cast<void*>(cache->scratch);
        ensure_device_buffer(&scratch_raw, &cache->scratch_capacity, mask_bytes, "star_mask_dog cudaMalloc scratch");
        cache->scratch = static_cast<uint8_t*>(scratch_raw);
        ensure_double_buffer(&cache->block_sum, &cache->block_sum_capacity, reduction_bytes, "star_mask_dog cudaMalloc block_sum");
        ensure_double_buffer(&cache->block_sq, &cache->block_sq_capacity, reduction_bytes, "star_mask_dog cudaMalloc block_sq");

        throw_if_cuda_failed(cudaMemcpyAsync(cache->image, image_host, image_bytes, cudaMemcpyHostToDevice, cache->stream),
                             "star_mask_dog_cuda copy image");
        throw_if_cuda_failed(cudaMemcpyAsync(cache->kernel_small, small_kernel_host, small_kernel_bytes, cudaMemcpyHostToDevice, cache->stream),
                             "star_mask_dog_cuda copy small kernel");
        throw_if_cuda_failed(cudaMemcpyAsync(cache->kernel_large, large_kernel_host, large_kernel_bytes, cudaMemcpyHostToDevice, cache->stream),
                             "star_mask_dog_cuda copy large kernel");

        gray_kernel<T><<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->image), cache->gray, plane_size, channels);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda gray");

        gaussian_horizontal_kernel<<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->gray, cache->tmp, cache->kernel_small, small_radius, height, width);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda small horizontal");
        gaussian_vertical_kernel<<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->tmp, cache->blur_small, cache->kernel_small, small_radius, height, width);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda small vertical");

        gaussian_horizontal_kernel<<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->gray, cache->tmp, cache->kernel_large, large_radius, height, width);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda large horizontal");
        gaussian_vertical_kernel<<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->tmp, cache->blur_large, cache->kernel_large, large_radius, height, width);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda large vertical");

        dog_reduce_kernel<<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->blur_small, cache->blur_large, cache->dog, cache->block_sum, cache->block_sq, plane_size);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda dog_reduce");

        std::vector<double> block_sum(static_cast<size_t>(blocks));
        std::vector<double> block_sq(static_cast<size_t>(blocks));
        throw_if_cuda_failed(cudaMemcpyAsync(block_sum.data(), cache->block_sum, reduction_bytes, cudaMemcpyDeviceToHost, cache->stream),
                             "star_mask_dog_cuda copy block_sum");
        throw_if_cuda_failed(cudaMemcpyAsync(block_sq.data(), cache->block_sq, reduction_bytes, cudaMemcpyDeviceToHost, cache->stream),
                             "star_mask_dog_cuda copy block_sq");
        throw_if_cuda_failed(cudaStreamSynchronize(cache->stream), "star_mask_dog_cuda sync reduce");

        double sum = 0.0;
        double sum_sq = 0.0;
        for (int i = 0; i < blocks; ++i) {
            sum += block_sum[static_cast<size_t>(i)];
            sum_sq += block_sq[static_cast<size_t>(i)];
        }
        const double mean = sum / static_cast<double>(plane_size);
        const double variance = fmax(0.0, sum_sq / static_cast<double>(plane_size) - mean * mean);
        const float threshold = static_cast<float>(sqrt(variance) * threshold_ratio);

        threshold_mask_kernel<<<blocks, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->dog, cache->mask, plane_size, threshold);
        throw_if_cuda_failed(cudaGetLastError(), "star_mask_dog_cuda threshold");
        launch_morphology(cache, height, width, blocks, open_ksize, dilate_ksize);

        throw_if_cuda_failed(cudaMemcpyAsync(mask_host, cache->mask, mask_bytes, cudaMemcpyDeviceToHost, cache->stream),
                             "star_mask_dog_cuda copy mask");
        throw_if_cuda_failed(cudaStreamSynchronize(cache->stream), "star_mask_dog_cuda synchronize");
    } catch (...) {
        reset_cuda_host_io_cache_after_error(cache);
        throw;
    }
}

}  // namespace

void launch_star_mask_dog_cuda_u8(const uint8_t* image_host,
                                  uint8_t* mask_host,
                                  const int height,
                                  const int width,
                                  const int channels,
                                  const float* small_kernel_host,
                                  const int small_radius,
                                  const float* large_kernel_host,
                                  const int large_radius,
                                  const float threshold_ratio,
                                  const int open_ksize,
                                  const int dilate_ksize) {
    launch_star_mask_dog_cuda_impl(image_host,
                                   mask_host,
                                   height,
                                   width,
                                   channels,
                                   small_kernel_host,
                                   small_radius,
                                   large_kernel_host,
                                   large_radius,
                                   threshold_ratio,
                                   open_ksize,
                                   dilate_ksize);
}

void launch_star_mask_dog_cuda_u16(const uint16_t* image_host,
                                   uint8_t* mask_host,
                                   const int height,
                                   const int width,
                                   const int channels,
                                   const float* small_kernel_host,
                                   const int small_radius,
                                   const float* large_kernel_host,
                                   const int large_radius,
                                   const float threshold_ratio,
                                   const int open_ksize,
                                   const int dilate_ksize) {
    launch_star_mask_dog_cuda_impl(image_host,
                                   mask_host,
                                   height,
                                   width,
                                   channels,
                                   small_kernel_host,
                                   small_radius,
                                   large_kernel_host,
                                   large_radius,
                                   threshold_ratio,
                                   open_ksize,
                                   dilate_ksize);
}
