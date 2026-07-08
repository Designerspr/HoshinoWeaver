#include "common/compat.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

struct StarShrinkCudaHostIoCache {
    void* image = nullptr;
    uint8_t* mask = nullptr;
    void* output = nullptr;
    float* luma = nullptr;
    float* luma_tmp = nullptr;
    float* lab_a = nullptr;
    float* lab_b = nullptr;
    float* shrunk = nullptr;
    float* blur_tmp = nullptr;
    float* blurred = nullptr;
    cudaStream_t stream = nullptr;
    size_t image_capacity = 0;
    size_t mask_capacity = 0;
    size_t output_capacity = 0;
    size_t luma_capacity = 0;
    size_t luma_tmp_capacity = 0;
    size_t lab_a_capacity = 0;
    size_t lab_b_capacity = 0;
    size_t shrunk_capacity = 0;
    size_t blur_tmp_capacity = 0;
    size_t blurred_capacity = 0;
    int device = -1;

    ~StarShrinkCudaHostIoCache() {
        int current_device = -1;
        const cudaError_t get_device_error = cudaGetDevice(&current_device);
        if (get_device_error == cudaSuccess && device >= 0 && current_device != device) {
            cudaSetDevice(device);
        }
        cudaFree(image);
        cudaFree(mask);
        cudaFree(output);
        cudaFree(luma);
        cudaFree(luma_tmp);
        cudaFree(lab_a);
        cudaFree(lab_b);
        cudaFree(shrunk);
        cudaFree(blur_tmp);
        cudaFree(blurred);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        if (get_device_error == cudaSuccess && device >= 0 && current_device != device) {
            cudaSetDevice(current_device);
        }
    }
};

thread_local StarShrinkCudaHostIoCache star_shrink_cache;

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

void ensure_float_buffer(float** ptr,
                         size_t* capacity,
                         const size_t required_bytes,
                         const char* context) {
    void* raw_ptr = static_cast<void*>(*ptr);
    ensure_device_buffer(&raw_ptr, capacity, required_bytes, context);
    *ptr = static_cast<float*>(raw_ptr);
}

void clear_device_cache(StarShrinkCudaHostIoCache* cache) {
    cudaFree(cache->image);
    cudaFree(cache->mask);
    cudaFree(cache->output);
    cudaFree(cache->luma);
    cudaFree(cache->luma_tmp);
    cudaFree(cache->lab_a);
    cudaFree(cache->lab_b);
    cudaFree(cache->shrunk);
    cudaFree(cache->blur_tmp);
    cudaFree(cache->blurred);
    cache->image = nullptr;
    cache->mask = nullptr;
    cache->output = nullptr;
    cache->luma = nullptr;
    cache->luma_tmp = nullptr;
    cache->lab_a = nullptr;
    cache->lab_b = nullptr;
    cache->shrunk = nullptr;
    cache->blur_tmp = nullptr;
    cache->blurred = nullptr;
    cache->image_capacity = 0;
    cache->mask_capacity = 0;
    cache->output_capacity = 0;
    cache->luma_capacity = 0;
    cache->luma_tmp_capacity = 0;
    cache->lab_a_capacity = 0;
    cache->lab_b_capacity = 0;
    cache->shrunk_capacity = 0;
    cache->blur_tmp_capacity = 0;
    cache->blurred_capacity = 0;
}

void clear_cuda_host_io_cache(StarShrinkCudaHostIoCache* cache) {
    clear_device_cache(cache);
    if (cache->stream != nullptr) {
        cudaStreamDestroy(cache->stream);
        cache->stream = nullptr;
    }
    cache->device = -1;
}

void ensure_stream(StarShrinkCudaHostIoCache* cache) {
    if (cache->stream != nullptr) {
        return;
    }
    throw_if_cuda_failed(cudaStreamCreateWithFlags(&cache->stream, cudaStreamNonBlocking),
                         "star_shrink_process_cuda cudaStreamCreate");
}

void prepare_cuda_host_io_cache(StarShrinkCudaHostIoCache* cache) {
    int current_device = -1;
    throw_if_cuda_failed(cudaGetDevice(&current_device),
                         "star_shrink_process_cuda cudaGetDevice");
    if (cache->device == current_device) {
        ensure_stream(cache);
        return;
    }
    if (cache->device >= 0) {
        const int restore_device = current_device;
        throw_if_cuda_failed(cudaSetDevice(cache->device),
                             "star_shrink_process_cuda cudaSetDevice(old)");
        clear_cuda_host_io_cache(cache);
        throw_if_cuda_failed(cudaSetDevice(restore_device),
                             "star_shrink_process_cuda cudaSetDevice(restore)");
    }
    cache->device = current_device;
    ensure_stream(cache);
}

void reset_cuda_host_io_cache_after_error(StarShrinkCudaHostIoCache* cache) {
    if (cache->stream != nullptr) {
        cudaStreamSynchronize(cache->stream);
    }
    clear_cuda_host_io_cache(cache);
}

template <typename T>
__device__ inline float dtype_max_value() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return 255.0f;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return 65535.0f;
    } else {
        return 1.0f;
    }
}

template <typename T>
__device__ inline float normalized_sample(const T* ptr, const int64_t idx) {
    if constexpr (std::is_same_v<T, float>) {
        return fminf(fmaxf(ptr[idx], 0.0f), 1.0f);
    } else {
        return static_cast<float>(ptr[idx]) / dtype_max_value<T>();
    }
}

template <typename T>
__device__ inline T cast_output(const float value) {
    const float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
    if constexpr (std::is_same_v<T, float>) {
        return clamped;
    } else {
        const float scaled = nearbyintf(clamped * dtype_max_value<T>());
        return static_cast<T>(fminf(fmaxf(scaled, 0.0f), dtype_max_value<T>()));
    }
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

__device__ inline bool kernel_active(const int shape,
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
            if (!kernel_active(shrink_shape, shrink_ksize, ky, kx)) {
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

__device__ inline int reflect101(int idx, const int length) {
    if (length <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= length) {
        if (idx < 0) {
            // OpenCV BORDER_REFLECT_101 maps -1 -> 1; BORDER_REFLECT maps -1 -> 0.
            idx = -idx;
        } else {
            idx = 2 * length - idx - 2;
        }
    }
    return idx;
}

template <typename T>
__global__ void horizontal_blur_kernel(const T* image,
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

__global__ void vertical_blur_kernel(const float* tmp,
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
void launch_star_shrink_process_cuda_impl(const T* image_host,
                                          const uint8_t* mask_host,
                                          T* out_host,
                                          const int height,
                                          const int width,
                                          const int channels,
                                          const int shrink_ksize,
                                          const int shrink_shape,
                                          const int shrink_times,
                                          const float shrink_ratio,
                                          const int deringing_ksize) {
    auto* cache = &star_shrink_cache;
    try {
        prepare_cuda_host_io_cache(cache);
        const int64_t plane_size = static_cast<int64_t>(height) * width;
        const int64_t total = plane_size * channels;
        const size_t image_bytes = static_cast<size_t>(total) * sizeof(T);
        const size_t mask_bytes = static_cast<size_t>(plane_size) * sizeof(uint8_t);
        const size_t plane_float_bytes = static_cast<size_t>(plane_size) * sizeof(float);
        const size_t total_float_bytes = static_cast<size_t>(total) * sizeof(float);

        ensure_device_buffer(
            &cache->image, &cache->image_capacity, image_bytes, "star_shrink cudaMalloc image");
        ensure_device_buffer(
            &cache->output, &cache->output_capacity, image_bytes, "star_shrink cudaMalloc output");
        void* mask_raw = static_cast<void*>(cache->mask);
        ensure_device_buffer(
            &mask_raw, &cache->mask_capacity, mask_bytes, "star_shrink cudaMalloc mask");
        cache->mask = static_cast<uint8_t*>(mask_raw);
        ensure_float_buffer(
            &cache->luma, &cache->luma_capacity, plane_float_bytes, "star_shrink cudaMalloc luma");
        ensure_float_buffer(&cache->luma_tmp,
                            &cache->luma_tmp_capacity,
                            plane_float_bytes,
                            "star_shrink cudaMalloc luma_tmp");
        ensure_float_buffer(&cache->lab_a,
                            &cache->lab_a_capacity,
                            plane_float_bytes,
                            "star_shrink cudaMalloc lab_a");
        ensure_float_buffer(&cache->lab_b,
                            &cache->lab_b_capacity,
                            plane_float_bytes,
                            "star_shrink cudaMalloc lab_b");
        ensure_float_buffer(&cache->shrunk,
                            &cache->shrunk_capacity,
                            total_float_bytes,
                            "star_shrink cudaMalloc shrunk");
        ensure_float_buffer(&cache->blur_tmp,
                            &cache->blur_tmp_capacity,
                            total_float_bytes,
                            "star_shrink cudaMalloc blur_tmp");
        ensure_float_buffer(&cache->blurred,
                            &cache->blurred_capacity,
                            total_float_bytes,
                            "star_shrink cudaMalloc blurred");

        throw_if_cuda_failed(cudaMemcpyAsync(cache->image,
                                             image_host,
                                             image_bytes,
                                             cudaMemcpyHostToDevice,
                                             cache->stream),
                             "star_shrink_process_cuda copy image to device");
        throw_if_cuda_failed(cudaMemcpyAsync(cache->mask,
                                             mask_host,
                                             mask_bytes,
                                             cudaMemcpyHostToDevice,
                                             cache->stream),
                             "star_shrink_process_cuda copy mask to device");

        const int blocks_plane =
            static_cast<int>((plane_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        const int blocks_total =
            static_cast<int>((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

        bgr_to_lab_kernel<T><<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->image),
            cache->luma,
            cache->lab_a,
            cache->lab_b,
            plane_size,
            channels);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_process_cuda bgr_to_lab");

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
            throw_if_cuda_failed(cudaGetLastError(), "star_shrink_process_cuda erode_luma");
            float* tmp = current;
            current = next;
            next = tmp;
        }

        lab_to_bgr_kernel<<<blocks_plane, THREADS_PER_BLOCK, 0, cache->stream>>>(
            current,
            cache->lab_a,
            cache->lab_b,
            cache->shrunk,
            plane_size,
            channels);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_process_cuda lab_to_bgr");

        horizontal_blur_kernel<T><<<blocks_total, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->image),
            cache->blur_tmp,
            height,
            width,
            channels,
            deringing_ksize);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_process_cuda horizontal_blur");
        vertical_blur_kernel<<<blocks_total, THREADS_PER_BLOCK, 0, cache->stream>>>(
            cache->blur_tmp,
            cache->blurred,
            height,
            width,
            channels,
            deringing_ksize);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_process_cuda vertical_blur");

        final_mask_kernel<T><<<blocks_total, THREADS_PER_BLOCK, 0, cache->stream>>>(
            static_cast<const T*>(cache->image),
            cache->mask,
            cache->shrunk,
            cache->blurred,
            static_cast<T*>(cache->output),
            plane_size,
            channels);
        throw_if_cuda_failed(cudaGetLastError(), "star_shrink_process_cuda final_mask");

        throw_if_cuda_failed(cudaMemcpyAsync(out_host,
                                             cache->output,
                                             image_bytes,
                                             cudaMemcpyDeviceToHost,
                                             cache->stream),
                             "star_shrink_process_cuda copy output to host");
        throw_if_cuda_failed(cudaStreamSynchronize(cache->stream),
                             "star_shrink_process_cuda synchronize");
    } catch (...) {
        reset_cuda_host_io_cache_after_error(cache);
        throw;
    }
}

}  // namespace

void launch_star_shrink_process_cuda_u8(const uint8_t* image_host,
                                        const uint8_t* mask_host,
                                        uint8_t* out_host,
                                        const int height,
                                        const int width,
                                        const int channels,
                                        const int shrink_ksize,
                                        const int shrink_shape,
                                        const int shrink_times,
                                        const float shrink_ratio,
                                        const int deringing_ksize) {
    launch_star_shrink_process_cuda_impl(image_host,
                                         mask_host,
                                         out_host,
                                         height,
                                         width,
                                         channels,
                                         shrink_ksize,
                                         shrink_shape,
                                         shrink_times,
                                         shrink_ratio,
                                         deringing_ksize);
}

void launch_star_shrink_process_cuda_u16(const uint16_t* image_host,
                                         const uint8_t* mask_host,
                                         uint16_t* out_host,
                                         const int height,
                                         const int width,
                                         const int channels,
                                         const int shrink_ksize,
                                         const int shrink_shape,
                                         const int shrink_times,
                                         const float shrink_ratio,
                                         const int deringing_ksize) {
    launch_star_shrink_process_cuda_impl(image_host,
                                         mask_host,
                                         out_host,
                                         height,
                                         width,
                                         channels,
                                         shrink_ksize,
                                         shrink_shape,
                                         shrink_times,
                                         shrink_ratio,
                                         deringing_ksize);
}
