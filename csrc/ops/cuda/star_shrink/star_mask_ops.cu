#include "common/compat.h"
#include "common/cuda_host_io_workspace.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

struct StarMaskDogBuffers {
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
    void* output = nullptr;
    float* luma = nullptr;
    float* luma_tmp = nullptr;
    float* lab_a = nullptr;
    float* lab_b = nullptr;
    float* shrunk = nullptr;
    float* box_tmp = nullptr;
    float* box_blurred = nullptr;
    double* block_sum = nullptr;
    double* block_sq = nullptr;
    cudaStream_t stream = nullptr;
};

void throw_if_cuda_failed(const cudaError_t error, const char* context) {
    hnw::cuda::throw_if_failed(error, context);
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

template <typename T>
__device__ inline T cast_output(const float value) {
    const float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
    const float scaled = nearbyintf(clamped * dtype_max_value<T>());
    return static_cast<T>(fminf(fmaxf(scaled, 0.0f), dtype_max_value<T>()));
}

__device__ inline float srgb_to_linear(const float value) {
    const float x = fminf(fmaxf(value, 0.0f), 1.0f);
    if (x <= 0.04045f) {
        return x / 12.92f;
    }
    return powf((x + 0.055f) / 1.055f, 2.4f);
}

__device__ inline float linear_to_srgb(const float value) {
    const float x = fminf(fmaxf(value, 0.0f), 1.0f);
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

__device__ inline float lab_f(const float t) {
    constexpr float delta = 6.0f / 29.0f;
    constexpr float delta3 = delta * delta * delta;
    if (t > delta3) {
        return cbrtf(t);
    }
    return t / (3.0f * delta * delta) + 4.0f / 29.0f;
}

__device__ inline float lab_f_inv(const float t) {
    constexpr float delta = 6.0f / 29.0f;
    if (t > delta) {
        return t * t * t;
    }
    return 3.0f * delta * delta * (t - 4.0f / 29.0f);
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
__global__ void bgr_to_lab_kernel(const T* image,
                                  float* luma,
                                  float* lab_a,
                                  float* lab_b,
                                  const int64_t plane_size,
                                  const int channels) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plane_size) {
        return;
    }
    if (channels == 1) {
        luma[idx] = normalized_sample(image, idx);
        return;
    }
    const int64_t base = idx * 3;
    const float b = normalized_sample(image, base);
    const float g = normalized_sample(image, base + 1);
    const float r = normalized_sample(image, base + 2);
    const float rl = srgb_to_linear(r);
    const float gl = srgb_to_linear(g);
    const float bl = srgb_to_linear(b);
    const float x = 0.4124564f * rl + 0.3575761f * gl + 0.1804375f * bl;
    const float y = 0.2126729f * rl + 0.7151522f * gl + 0.0721750f * bl;
    const float z = 0.0193339f * rl + 0.1191920f * gl + 0.9503041f * bl;
    const float fx = lab_f(x / 0.95047f);
    const float fy = lab_f(y);
    const float fz = lab_f(z / 1.08883f);
    luma[idx] = 116.0f * fy - 16.0f;
    lab_a[idx] = 500.0f * (fx - fy);
    lab_b[idx] = 200.0f * (fy - fz);
}

__device__ inline bool shrink_kernel_active(const int shape,
                                            const int ksize,
                                            const int ky,
                                            const int kx) {
    const int radius = ksize / 2;
    const int dy = ky - radius;
    const int dx = kx - radius;
    if (shape == 0) {
        return true;
    }
    if (shape == 1) {
        return dx == 0 || dy == 0;
    }
    return dx * dx + dy * dy <= radius * radius;
}

__global__ void erode_luma_kernel(const float* current,
                                  float* next,
                                  const int height,
                                  const int width,
                                  const int shrink_ksize,
                                  const int shrink_shape,
                                  const float shrink_ratio) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t plane_size = static_cast<int64_t>(height) * width;
    if (idx >= plane_size) {
        return;
    }
    const int y = static_cast<int>(idx / width);
    const int x = static_cast<int>(idx - static_cast<int64_t>(y) * width);
    const int radius = shrink_ksize / 2;
    float minimum = INFINITY;
    for (int ky = 0; ky < shrink_ksize; ++ky) {
        const int yy = y + ky - radius;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int kx = 0; kx < shrink_ksize; ++kx) {
            if (!shrink_kernel_active(shrink_shape, shrink_ksize, ky, kx)) {
                continue;
            }
            const int xx = x + kx - radius;
            if (xx < 0 || xx >= width) {
                continue;
            }
            minimum = fminf(minimum, current[static_cast<int64_t>(yy) * width + xx]);
        }
    }
    next[idx] = minimum * shrink_ratio + current[idx] * (1.0f - shrink_ratio);
}

__global__ void lab_to_bgr_kernel(const float* luma,
                                  const float* lab_a,
                                  const float* lab_b,
                                  float* shrunk,
                                  const int64_t plane_size,
                                  const int channels) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plane_size) {
        return;
    }
    if (channels == 1) {
        shrunk[idx] = luma[idx];
        return;
    }
    const float l = luma[idx];
    const float a = lab_a[idx];
    const float b_lab = lab_b[idx];
    const float fy = (l + 16.0f) / 116.0f;
    const float fx = fy + a / 500.0f;
    const float fz = fy - b_lab / 200.0f;
    const float x = 0.95047f * lab_f_inv(fx);
    const float y = lab_f_inv(fy);
    const float z = 1.08883f * lab_f_inv(fz);
    const float rl = 3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
    const float gl = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
    const float bl = 0.0556434f * x - 0.2040259f * y + 1.0572252f * z;
    const int64_t base = idx * 3;
    shrunk[base] = fminf(fmaxf(linear_to_srgb(bl), 0.0f), 1.0f);
    shrunk[base + 1] = fminf(fmaxf(linear_to_srgb(gl), 0.0f), 1.0f);
    shrunk[base + 2] = fminf(fmaxf(linear_to_srgb(rl), 0.0f), 1.0f);
}

template <typename T>
__global__ void box_horizontal_blur_kernel(const T* image,
                                           float* tmp,
                                           const int height,
                                           const int width,
                                           const int channels,
                                           const int ksize) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(height) * width * channels;
    if (idx >= total) {
        return;
    }
    const int c = static_cast<int>(idx % channels);
    const int pixel = static_cast<int>(idx / channels);
    const int y = pixel / width;
    const int x = pixel - y * width;
    const int radius = ksize / 2;
    float sum = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx) {
        const int xx = reflect101(x + dx, width);
        sum += normalized_sample(image, (static_cast<int64_t>(y) * width + xx) * channels + c);
    }
    tmp[idx] = sum / static_cast<float>(ksize);
}

__global__ void box_vertical_blur_kernel(const float* tmp,
                                         float* blurred,
                                         const int height,
                                         const int width,
                                         const int channels,
                                         const int ksize) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(height) * width * channels;
    if (idx >= total) {
        return;
    }
    const int c = static_cast<int>(idx % channels);
    const int pixel = static_cast<int>(idx / channels);
    const int y = pixel / width;
    const int x = pixel - y * width;
    const int radius = ksize / 2;
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int yy = reflect101(y + dy, height);
        sum += tmp[(static_cast<int64_t>(yy) * width + x) * channels + c];
    }
    blurred[idx] = sum / static_cast<float>(ksize);
}

template <typename T>
__global__ void final_mask_kernel(const T* image,
                                  const uint8_t* mask,
                                  const float* shrunk,
                                  const float* blurred,
                                  T* output,
                                  const int64_t plane_size,
                                  const int channels) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = plane_size * channels;
    if (idx >= total) {
        return;
    }
    const int64_t pixel = idx / channels;
    if (mask[pixel] == 0) {
        output[idx] = image[idx];
        return;
    }
    output[idx] = cast_output<T>(fmaxf(shrunk[idx], blurred[idx]));
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

void launch_morphology(StarMaskDogBuffers* cache,
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
    auto workspace = hnw::cuda::acquire_host_io_workspace(
        "star_mask_dog_cuda cudaGetDevice");
    StarMaskDogBuffers buffers;
    auto* cache = &buffers;
    try {
        cache->stream = workspace.stream();
        const int64_t plane_size = static_cast<int64_t>(height) * width;
        const int64_t total = plane_size * channels;
        const int blocks = static_cast<int>((plane_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        const size_t image_bytes = static_cast<size_t>(total) * sizeof(T);
        const size_t plane_float_bytes = static_cast<size_t>(plane_size) * sizeof(float);
        const size_t mask_bytes = static_cast<size_t>(plane_size) * sizeof(uint8_t);
        const size_t reduction_bytes = static_cast<size_t>(blocks) * sizeof(double);
        const size_t small_kernel_bytes = static_cast<size_t>(2 * small_radius + 1) * sizeof(float);
        const size_t large_kernel_bytes = static_cast<size_t>(2 * large_radius + 1) * sizeof(float);

        cache->image = workspace.device_buffer(
            image_bytes, "star_mask_dog cudaMalloc image");
        cache->gray = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_mask_dog cudaMalloc gray"));
        cache->tmp = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_mask_dog cudaMalloc tmp"));
        cache->blur_small = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_mask_dog cudaMalloc blur_small"));
        cache->blur_large = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_mask_dog cudaMalloc blur_large"));
        cache->dog = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_mask_dog cudaMalloc dog"));
        cache->kernel_small = static_cast<float*>(workspace.device_buffer(
            small_kernel_bytes, "star_mask_dog cudaMalloc small_kernel"));
        cache->kernel_large = static_cast<float*>(workspace.device_buffer(
            large_kernel_bytes, "star_mask_dog cudaMalloc large_kernel"));
        cache->mask = static_cast<uint8_t*>(workspace.device_buffer(
            mask_bytes, "star_mask_dog cudaMalloc mask"));
        cache->scratch = static_cast<uint8_t*>(workspace.device_buffer(
            mask_bytes, "star_mask_dog cudaMalloc scratch"));
        cache->block_sum = static_cast<double*>(workspace.device_buffer(
            reduction_bytes, "star_mask_dog cudaMalloc block_sum"));
        cache->block_sq = static_cast<double*>(workspace.device_buffer(
            reduction_bytes, "star_mask_dog cudaMalloc block_sq"));

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
        workspace.reset_after_error();
        throw;
    }
}

template <typename T>
void launch_star_shrink_dog_process_cuda_impl(const T* image_host,
                                              T* out_host,
                                              const int height,
                                              const int width,
                                              const int channels,
                                              const float* small_kernel_host,
                                              const int small_radius,
                                              const float* large_kernel_host,
                                              const int large_radius,
                                              const float threshold_ratio,
                                              const int open_ksize,
                                              const int dilate_ksize,
                                              const int shrink_ksize,
                                              const int shrink_shape,
                                              const int shrink_times,
                                              const float shrink_ratio,
                                              const int deringing_ksize) {
    auto workspace = hnw::cuda::acquire_host_io_workspace(
        "star_shrink_dog_cuda cudaGetDevice");
    StarMaskDogBuffers buffers;
    auto* cache = &buffers;
    try {
        cache->stream = workspace.stream();
        const int64_t plane_size = static_cast<int64_t>(height) * width;
        const int64_t total = plane_size * channels;
        const int blocks_plane =
            static_cast<int>((plane_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        const int blocks_total =
            static_cast<int>((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        const size_t image_bytes = static_cast<size_t>(total) * sizeof(T);
        const size_t plane_float_bytes = static_cast<size_t>(plane_size) * sizeof(float);
        const size_t total_float_bytes = static_cast<size_t>(total) * sizeof(float);
        const size_t mask_bytes = static_cast<size_t>(plane_size) * sizeof(uint8_t);
        const size_t reduction_bytes = static_cast<size_t>(blocks_plane) * sizeof(double);
        const size_t small_kernel_bytes = static_cast<size_t>(2 * small_radius + 1) * sizeof(float);
        const size_t large_kernel_bytes = static_cast<size_t>(2 * large_radius + 1) * sizeof(float);

        cache->image = workspace.device_buffer(
            image_bytes, "star_shrink_dog cudaMalloc image");
        cache->output = workspace.device_buffer(
            image_bytes, "star_shrink_dog cudaMalloc output");
        cache->gray = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_shrink_dog cudaMalloc gray"));
        cache->tmp = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_shrink_dog cudaMalloc tmp"));
        cache->blur_small = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_shrink_dog cudaMalloc blur_small"));
        cache->blur_large = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_shrink_dog cudaMalloc blur_large"));
        cache->dog = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_shrink_dog cudaMalloc dog"));
        cache->kernel_small = static_cast<float*>(workspace.device_buffer(
            small_kernel_bytes, "star_shrink_dog cudaMalloc small_kernel"));
        cache->kernel_large = static_cast<float*>(workspace.device_buffer(
            large_kernel_bytes, "star_shrink_dog cudaMalloc large_kernel"));
        cache->mask = static_cast<uint8_t*>(workspace.device_buffer(
            mask_bytes, "star_shrink_dog cudaMalloc mask"));
        cache->scratch = static_cast<uint8_t*>(workspace.device_buffer(
            mask_bytes, "star_shrink_dog cudaMalloc scratch"));
        cache->block_sum = static_cast<double*>(workspace.device_buffer(
            reduction_bytes, "star_shrink_dog cudaMalloc block_sum"));
        cache->block_sq = static_cast<double*>(workspace.device_buffer(
            reduction_bytes, "star_shrink_dog cudaMalloc block_sq"));
        cache->luma = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_shrink_dog cudaMalloc luma"));
        cache->luma_tmp = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_shrink_dog cudaMalloc luma_tmp"));
        cache->lab_a = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_shrink_dog cudaMalloc lab_a"));
        cache->lab_b = static_cast<float*>(workspace.device_buffer(
            plane_float_bytes, "star_shrink_dog cudaMalloc lab_b"));
        cache->shrunk = static_cast<float*>(workspace.device_buffer(
            total_float_bytes, "star_shrink_dog cudaMalloc shrunk"));
        cache->box_tmp = static_cast<float*>(workspace.device_buffer(
            total_float_bytes, "star_shrink_dog cudaMalloc box_tmp"));
        cache->box_blurred = static_cast<float*>(workspace.device_buffer(
            total_float_bytes, "star_shrink_dog cudaMalloc box_blurred"));

        throw_if_cuda_failed(cudaMemcpyAsync(cache->image, image_host, image_bytes, cudaMemcpyHostToDevice, cache->stream),
                             "star_shrink_dog_cuda copy image");
        throw_if_cuda_failed(cudaMemcpyAsync(cache->kernel_small, small_kernel_host, small_kernel_bytes, cudaMemcpyHostToDevice, cache->stream),
                             "star_shrink_dog_cuda copy small kernel");
        throw_if_cuda_failed(cudaMemcpyAsync(cache->kernel_large, large_kernel_host, large_kernel_bytes, cudaMemcpyHostToDevice, cache->stream),
                             "star_shrink_dog_cuda copy large kernel");

        gray_kernel<T><<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->image), cache->gray, plane_size, channels);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda gray");

        gaussian_horizontal_kernel<<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->gray, cache->tmp, cache->kernel_small, small_radius, height, width);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda small horizontal");
        gaussian_vertical_kernel<<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->tmp, cache->blur_small, cache->kernel_small, small_radius, height, width);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda small vertical");

        gaussian_horizontal_kernel<<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->gray, cache->tmp, cache->kernel_large, large_radius, height, width);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda large horizontal");
        gaussian_vertical_kernel<<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->tmp, cache->blur_large, cache->kernel_large, large_radius, height, width);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda large vertical");

        dog_reduce_kernel<<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->blur_small,
            cache->blur_large,
            cache->dog,
            cache->block_sum,
            cache->block_sq,
            plane_size);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda dog_reduce");

        std::vector<double> block_sum(static_cast<size_t>(blocks_plane));
        std::vector<double> block_sq(static_cast<size_t>(blocks_plane));
        throw_if_cuda_failed(cudaMemcpyAsync(block_sum.data(), cache->block_sum, reduction_bytes, cudaMemcpyDeviceToHost, cache->stream),
                             "star_shrink_dog_cuda copy block_sum");
        throw_if_cuda_failed(cudaMemcpyAsync(block_sq.data(), cache->block_sq, reduction_bytes, cudaMemcpyDeviceToHost, cache->stream),
                             "star_shrink_dog_cuda copy block_sq");
        throw_if_cuda_failed(cudaStreamSynchronize(cache->stream), "star_shrink_dog_cuda sync reduce");

        double sum = 0.0;
        double sum_sq = 0.0;
        for (int i = 0; i < blocks_plane; ++i) {
            sum += block_sum[static_cast<size_t>(i)];
            sum_sq += block_sq[static_cast<size_t>(i)];
        }
        const double mean = sum / static_cast<double>(plane_size);
        const double variance = fmax(0.0, sum_sq / static_cast<double>(plane_size) - mean * mean);
        const float threshold = static_cast<float>(sqrt(variance) * threshold_ratio);

        threshold_mask_kernel<<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->dog, cache->mask, plane_size, threshold);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda threshold");
        launch_morphology(cache, height, width, blocks_plane, open_ksize, dilate_ksize);

        bgr_to_lab_kernel<T><<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->image),
            cache->luma,
            cache->lab_a,
            cache->lab_b,
            plane_size,
            channels);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda bgr_to_lab");

        float* current = cache->luma;
        float* next = cache->luma_tmp;
        for (int iter = 0; iter < shrink_times; ++iter) {
            erode_luma_kernel<<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
                current,
                next,
                height,
                width,
                shrink_ksize,
                shrink_shape,
                shrink_ratio);
            throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda erode_luma");
            float* tmp_luma = current;
            current = next;
            next = tmp_luma;
        }

        lab_to_bgr_kernel<<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            current,
            cache->lab_a,
            cache->lab_b,
            cache->shrunk,
            plane_size,
            channels);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda lab_to_bgr");

        box_horizontal_blur_kernel<T><<<blocks_total, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->image),
            cache->box_tmp,
            height,
            width,
            channels,
            deringing_ksize);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda box_horizontal");
        box_vertical_blur_kernel<<<blocks_total, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->box_tmp,
            cache->box_blurred,
            height,
            width,
            channels,
            deringing_ksize);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda box_vertical");

        final_mask_kernel<T><<<blocks_total, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->image),
            cache->mask,
            cache->shrunk,
            cache->box_blurred,
            static_cast<T*>(cache->output),
            plane_size,
            channels);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_dog_cuda final_mask");

        throw_if_cuda_failed(cudaMemcpyAsync(out_host, cache->output, image_bytes, cudaMemcpyDeviceToHost, cache->stream),
                             "star_shrink_dog_cuda copy output");
        throw_if_cuda_failed(cudaStreamSynchronize(cache->stream), "star_shrink_dog_cuda synchronize");
    } catch (...) {
        workspace.reset_after_error();
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

void launch_star_shrink_dog_process_cuda_u8(const uint8_t* image_host,
                                            uint8_t* out_host,
                                            const int height,
                                            const int width,
                                            const int channels,
                                            const float* small_kernel_host,
                                            const int small_radius,
                                            const float* large_kernel_host,
                                            const int large_radius,
                                            const float threshold_ratio,
                                            const int open_ksize,
                                            const int dilate_ksize,
                                            const int shrink_ksize,
                                            const int shrink_shape,
                                            const int shrink_times,
                                            const float shrink_ratio,
                                            const int deringing_ksize) {
    launch_star_shrink_dog_process_cuda_impl(image_host,
                                             out_host,
                                             height,
                                             width,
                                             channels,
                                             small_kernel_host,
                                             small_radius,
                                             large_kernel_host,
                                             large_radius,
                                             threshold_ratio,
                                             open_ksize,
                                             dilate_ksize,
                                             shrink_ksize,
                                             shrink_shape,
                                             shrink_times,
                                             shrink_ratio,
                                             deringing_ksize);
}

void launch_star_shrink_dog_process_cuda_u16(const uint16_t* image_host,
                                             uint16_t* out_host,
                                             const int height,
                                             const int width,
                                             const int channels,
                                             const float* small_kernel_host,
                                             const int small_radius,
                                             const float* large_kernel_host,
                                             const int large_radius,
                                             const float threshold_ratio,
                                             const int open_ksize,
                                             const int dilate_ksize,
                                             const int shrink_ksize,
                                             const int shrink_shape,
                                             const int shrink_times,
                                             const float shrink_ratio,
                                             const int deringing_ksize) {
    launch_star_shrink_dog_process_cuda_impl(image_host,
                                             out_host,
                                             height,
                                             width,
                                             channels,
                                             small_kernel_host,
                                             small_radius,
                                             large_kernel_host,
                                             large_radius,
                                             threshold_ratio,
                                             open_ksize,
                                             dilate_ksize,
                                             shrink_ksize,
                                             shrink_shape,
                                             shrink_times,
                                             shrink_ratio,
                                             deringing_ksize);
}
