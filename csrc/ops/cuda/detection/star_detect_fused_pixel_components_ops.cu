#include "star_detect_fused_pixel_components_ops.h"

#include "../wavelet/wavelet_device.cuh"
#include "common/compat.h"
#include "common/cuda_error.h"
#include "common/cuda_host_io_workspace.cuh"
#include "common/star_detect_capacity.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Func>
decltype(auto) run_thrust_with_resource_translation(const char* context, Func&& func) {
    try {
        return func();
    } catch (const std::bad_alloc& exc) {
        throw hnw::CudaResourceExhaustedError(std::string(context) + ": " + exc.what());
    }
}

int next_power_of_two_int(const int value) {
    if (value <= 1) {
        return 1;
    }
    if (value > (std::numeric_limits<int>::max() / 2 + 1)) {
        throw std::runtime_error("star_detect_fused_pixel_components: hash table is too large");
    }
    unsigned int result = 1;
    const unsigned int target = static_cast<unsigned int>(value);
    while (result < target) {
        result <<= 1U;
    }
    return static_cast<int>(result);
}

__global__ void resize_linear_mask_kernel(const double* input, const uint8_t* mask, double* output,
                                          const int in_h, const int in_w, const int out_h,
                                          const int out_w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int y = idx / out_w;
    const int x = idx - y * out_w;
    const double src_y =
        (static_cast<double>(y) + 0.5) * static_cast<double>(in_h) / static_cast<double>(out_h) -
        0.5;
    const double src_x =
        (static_cast<double>(x) + 0.5) * static_cast<double>(in_w) / static_cast<double>(out_w) -
        0.5;
    const int y0_raw = static_cast<int>(floor(src_y));
    const int x0_raw = static_cast<int>(floor(src_x));
    const double wy = src_y - static_cast<double>(y0_raw);
    const double wx = src_x - static_cast<double>(x0_raw);
    const int y0 = min(max(y0_raw, 0), in_h - 1);
    const int x0 = min(max(x0_raw, 0), in_w - 1);
    const int y1 = min(y0_raw + 1, in_h - 1);
    const int x1 = min(x0_raw + 1, in_w - 1);
    const double v00 = input[y0 * in_w + x0];
    const double v01 = input[y0 * in_w + x1];
    const double v10 = input[y1 * in_w + x0];
    const double v11 = input[y1 * in_w + x1];
    const double top = v00 + (v01 - v00) * wx;
    const double bottom = v10 + (v11 - v10) * wx;
    const double value = top + (bottom - top) * wy;
    output[idx] = mask[idx] == 0 ? 0.0 : value;
}

__global__ void resize_linear_kernel(const double* input, double* output, const int in_h,
                                     const int in_w, const int out_h, const int out_w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int y = idx / out_w;
    const int x = idx - y * out_w;
    const double src_y =
        (static_cast<double>(y) + 0.5) * static_cast<double>(in_h) / static_cast<double>(out_h) -
        0.5;
    const double src_x =
        (static_cast<double>(x) + 0.5) * static_cast<double>(in_w) / static_cast<double>(out_w) -
        0.5;
    const int y0_raw = static_cast<int>(floor(src_y));
    const int x0_raw = static_cast<int>(floor(src_x));
    const double wy = src_y - static_cast<double>(y0_raw);
    const double wx = src_x - static_cast<double>(x0_raw);
    const int y0 = min(max(y0_raw, 0), in_h - 1);
    const int x0 = min(max(x0_raw, 0), in_w - 1);
    const int y1 = min(y0_raw + 1, in_h - 1);
    const int x1 = min(x0_raw + 1, in_w - 1);
    const double v00 = input[y0 * in_w + x0];
    const double v01 = input[y0 * in_w + x1];
    const double v10 = input[y1 * in_w + x0];
    const double v11 = input[y1 * in_w + x1];
    const double top = v00 + (v01 - v00) * wx;
    const double bottom = v10 + (v11 - v10) * wx;
    output[idx] = top + (bottom - top) * wy;
}

__device__ inline int reflect101_index(int idx, const int n) {
    if (n <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= n) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * n - idx - 2;
        }
    }
    return idx;
}

__global__ void gaussian_rows_kernel(const double* input, const double* kernel, double* output,
                                     const int height, const int width, const int ksize) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int y = idx / width;
    const int x = idx - y * width;
    const int radius = ksize / 2;
    double value = 0.0;
    for (int k = 0; k < ksize; ++k) {
        const int xx = reflect101_index(x + k - radius, width);
        value += input[y * width + xx] * kernel[k];
    }
    output[idx] = value;
}

__global__ void gaussian_cols_kernel(const double* input, const double* kernel, double* output,
                                     const int height, const int width, const int ksize) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int y = idx / width;
    const int x = idx - y * width;
    const int radius = ksize / 2;
    double value = 0.0;
    for (int k = 0; k < ksize; ++k) {
        const int yy = reflect101_index(y + k - radius, height);
        value += input[yy * width + x] * kernel[k];
    }
    output[idx] = value;
}

__global__ void normalize_kernel(const double* input, double* output, const double mean,
                                 const double range, const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    output[idx] = (input[idx] - mean) / range;
}

__global__ void compact_masked_values_kernel(const double* image, const uint8_t* mask,
                                             double* values, int* count, const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total || mask[idx] == 0) {
        return;
    }
    const int pos = atomicAdd(count, 1);
    values[pos] = image[idx];
}

__device__ inline uint8_t threshold_pixel(const double value, const uint8_t mask,
                                          const double threshold) {
    return (mask != 0 && value > threshold) ? 255 : 0;
}

__global__ void erode_threshold_kernel(const double* image, const uint8_t* mask, uint8_t* eroded,
                                       const int height, const int width, const double threshold) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int y = idx / width;
    const int x = idx - y * width;
    uint8_t value = 255;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= width) {
                continue;
            }
            const int offset = yy * width + xx;
            const uint8_t candidate = threshold_pixel(image[offset], mask[offset], threshold);
            value = min(value, candidate);
        }
    }
    eroded[idx] = value;
}

__global__ void dilate_kernel(const uint8_t* eroded, uint8_t* out, const int height,
                              const int width) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int y = idx / width;
    const int x = idx - y * width;
    uint8_t value = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= width) {
                continue;
            }
            value = max(value, eroded[yy * width + xx]);
        }
    }
    out[idx] = value;
}

__global__ void init_component_labels_kernel(const uint8_t* bw, int* labels, const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    labels[idx] = bw[idx] == 0 ? 0 : idx + 1;
}

__global__ void propagate_foreground_component_labels_kernel(const int* foreground_indices,
                                                             const int foreground_count,
                                                             const int* in_labels, int* out_labels,
                                                             int* changed, const int height,
                                                             const int width) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= foreground_count) {
        return;
    }
    const int idx = foreground_indices[item];
    const int current = in_labels[idx];
    const int y = idx / width;
    const int x = idx - y * width;
    int best = current;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= width) {
                continue;
            }
            const int candidate = in_labels[yy * width + xx];
            if (candidate != 0 && candidate < best) {
                best = candidate;
            }
        }
    }
    out_labels[idx] = best;
    if (best != current) {
        atomicAdd(changed, 1);
    }
}

__global__ void compact_foreground_indices_kernel(const uint8_t* bw, int* indices, int* count,
                                                  const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total || bw[idx] == 0) {
        return;
    }
    const int pos = atomicAdd(count, 1);
    indices[pos] = idx;
}

__device__ inline unsigned int component_hash(const unsigned int value) {
    unsigned int x = value;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ int find_component_slot(int* keys, int* overflow, const int capacity,
                                   const int root_label) {
    const int mask = capacity - 1;
    unsigned int pos =
        component_hash(static_cast<unsigned int>(root_label)) & static_cast<unsigned int>(mask);
    for (int probe = 0; probe < capacity; ++probe) {
        const int old = atomicCAS(&keys[pos], 0, root_label);
        if (old == 0 || old == root_label) {
            return static_cast<int>(pos);
        }
        pos = (pos + 1U) & static_cast<unsigned int>(mask);
    }
    atomicExch(overflow, 1);
    return -1;
}

__global__ void accumulate_component_stats_kernel(const int* foreground_indices,
                                                  const int foreground_count, const int* labels,
                                                  const double* image, int* keys, int* counts,
                                                  double* sum_x, double* sum_y,
                                                  double* sum_intensity, int* overflow,
                                                  const int width, const int hash_capacity) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= foreground_count) {
        return;
    }
    const int idx = foreground_indices[item];
    const int root = labels[idx];
    if (root == 0) {
        return;
    }
    const int slot = find_component_slot(keys, overflow, hash_capacity, root);
    if (slot < 0) {
        return;
    }

    const double x = static_cast<double>(idx % width);
    const double y = static_cast<double>(idx / width);
    atomicAdd(&counts[slot], 1);
    atomicAdd(&sum_x[slot], x);
    atomicAdd(&sum_y[slot], y);
    atomicAdd(&sum_intensity[slot], image[idx]);
}

__global__ void count_component_outputs_kernel(const int* keys, const int* counts, int* out_count,
                                               const int hash_capacity) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hash_capacity) {
        return;
    }
    if (keys[idx] != 0 && counts[idx] > 0) {
        atomicAdd(out_count, 1);
    }
}

__global__ void fill_component_outputs_kernel(const int* keys, const int* counts,
                                              const double* sum_x, const double* sum_y,
                                              const double* sum_intensity, double* positions,
                                              double* intensities, int* out_index,
                                              const int hash_capacity) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hash_capacity || keys[idx] == 0 || counts[idx] <= 0) {
        return;
    }

    const int out = atomicAdd(out_index, 1);
    const double count = static_cast<double>(counts[idx]);
    const double inv_count = 1.0 / count;
    const double cx = sum_x[idx] * inv_count;
    const double cy = sum_y[idx] * inv_count;

    positions[out * 2] = cx;
    positions[out * 2 + 1] = cy;
    intensities[out] = sum_intensity[idx] * inv_count;
}

double percentile_995_sorted(const double* values, const int count, const cudaStream_t stream,
                             double* host_values) {
    const double rank = 0.995 * static_cast<double>(count - 1);
    const int lower_idx = static_cast<int>(floor(rank));
    const int upper_idx = static_cast<int>(ceil(rank));
    const double weight = rank - static_cast<double>(lower_idx);
    throw_if_cuda_failed(cudaMemcpyAsync(host_values, values + lower_idx, sizeof(double),
                                         cudaMemcpyDeviceToHost, stream),
                         "star_detect percentile cudaMemcpy lower percentile");
    throw_if_cuda_failed(cudaMemcpyAsync(host_values + 1, values + upper_idx, sizeof(double),
                                         cudaMemcpyDeviceToHost, stream),
                         "star_detect percentile cudaMemcpy upper percentile");
    throw_if_cuda_failed(cudaStreamSynchronize(stream),
                         "star_detect percentile cudaStreamSynchronize");
    return host_values[0] + (host_values[1] - host_values[0]) * weight;
}

} // namespace

void launch_star_detect_fused_pixel_components(
    const double* image_host, const uint8_t* external_mask_host, const double* gaussian_kernel_host,
    std::vector<double>* positions_xy_host, std::vector<double>* intensities_host,
    uint8_t* binary_mask_host, const int height, const int width, const int small_height,
    const int small_width, const int level, const int gaussian_ksize) {
    const int threads = 256;
    const int total = height * width;
    const int small_total = small_height * small_width;
    const size_t plane_size = static_cast<size_t>(total);
    const size_t small_size = static_cast<size_t>(small_total);

    auto workspace =
        hnw::cuda::acquire_host_io_workspace("star_detect_fused_pixel_components cudaGetDevice");
    try {
        const size_t image_bytes = plane_size * sizeof(double);
        const size_t mask_bytes = plane_size * sizeof(uint8_t);
        const size_t gaussian_kernel_bytes = static_cast<size_t>(gaussian_ksize) * sizeof(double);
        const cudaStream_t stream = workspace.stream();
        auto* scalar_doubles = static_cast<double*>(workspace.pinned_buffer(
            2 * sizeof(double),
            "star_detect_fused_pixel_components cudaMallocHost scalar doubles"));
        auto* scalar_int = static_cast<int*>(workspace.pinned_buffer(
            sizeof(int), "star_detect_fused_pixel_components cudaMallocHost scalar int"));
        DeviceBuffer image;
        DeviceBuffer gaussian_kernel;
        DeviceBuffer blur_rows;
        DeviceBuffer blur;
        DeviceBuffer normalized;
        DeviceBuffer small_blur;
        DeviceBuffer img_rec;
        DeviceBuffer values;
        DeviceTypedBuffer<uint8_t> mask;
        DeviceTypedBuffer<uint8_t> eroded;
        DeviceTypedBuffer<uint8_t> bw;
        DeviceTypedBuffer<int> count;
        DeviceTypedBuffer<int> foreground_indices;
        DeviceTypedBuffer<int> labels_a;
        DeviceTypedBuffer<int> labels_b;
        DeviceTypedBuffer<int> changed;
        DeviceTypedBuffer<int> keys;
        DeviceTypedBuffer<int> counts;
        DeviceTypedBuffer<int> overflow;
        DeviceBuffer sum_x;
        DeviceBuffer sum_y;
        DeviceBuffer sum_intensity;
        DeviceBuffer out_positions;
        DeviceBuffer out_intensities;

        positions_xy_host->clear();
        intensities_host->clear();

        image.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc image",
                       &workspace);
        gaussian_kernel.allocate(static_cast<size_t>(gaussian_ksize),
                                 "star_detect_fused_pixel_components cudaMalloc gaussian kernel",
                                 &workspace);
        blur_rows.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc blur rows",
                           &workspace);
        blur.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc blur", &workspace);
        normalized.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc normalized",
                            &workspace);
        small_blur.allocate(small_size, "star_detect_fused_pixel_components cudaMalloc small blur",
                            &workspace);
        img_rec.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc img_rec",
                         &workspace);
        values.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc values",
                        &workspace);
        mask.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc mask", &workspace);
        eroded.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc eroded",
                        &workspace);
        bw.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc bw", &workspace);
        count.allocate(1, "star_detect_fused_pixel_components cudaMalloc count", &workspace);

        throw_if_cuda_failed(
            cudaMemcpyAsync(image.get(), image_host, image_bytes, cudaMemcpyHostToDevice, stream),
            "star_detect_fused_pixel_components cudaMemcpy image");
        throw_if_cuda_failed(cudaMemcpyAsync(gaussian_kernel.get(), gaussian_kernel_host,
                                             gaussian_kernel_bytes, cudaMemcpyHostToDevice, stream),
                             "star_detect_fused_pixel_components cudaMemcpy gaussian kernel");
        if (external_mask_host != nullptr) {
            throw_if_cuda_failed(cudaMemcpyAsync(mask.get(), external_mask_host, mask_bytes,
                                                 cudaMemcpyHostToDevice, stream),
                                 "star_detect_fused_pixel_components cudaMemcpy external mask");
        } else {
            throw_if_cuda_failed(cudaMemsetAsync(mask.get(), 1, mask_bytes, stream),
                                 "star_detect_fused_pixel_components cudaMemset full mask");
        }

        const int blocks = (total + threads - 1) / threads;
        const int small_blocks = (small_total + threads - 1) / threads;

        gaussian_rows_kernel<<<blocks, threads, 0, stream>>>(
            image.get(), gaussian_kernel.get(), blur_rows.get(), height, width, gaussian_ksize);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components gaussian rows launch");
        gaussian_cols_kernel<<<blocks, threads, 0, stream>>>(
            blur_rows.get(), gaussian_kernel.get(), blur.get(), height, width, gaussian_ksize);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components gaussian cols launch");
        blur_rows.reset();
        gaussian_kernel.reset();

        thrust::device_ptr<double> blur_ptr(blur.get());
        const double blur_sum =
            run_thrust_with_resource_translation("star_detect thrust reduce allocation", [&] {
                return thrust::reduce(thrust::cuda::par.on(stream), blur_ptr, blur_ptr + total,
                                      0.0);
            });
        const auto minmax =
            run_thrust_with_resource_translation("star_detect thrust minmax allocation", [&] {
                return thrust::minmax_element(thrust::cuda::par.on(stream), blur_ptr,
                                              blur_ptr + total);
            });
        throw_if_cuda_failed(cudaMemcpyAsync(scalar_doubles, thrust::raw_pointer_cast(minmax.first),
                                             sizeof(double), cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy blur min");
        throw_if_cuda_failed(cudaMemcpyAsync(scalar_doubles + 1,
                                             thrust::raw_pointer_cast(minmax.second),
                                             sizeof(double), cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy blur max");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components blur extrema sync");
        const double blur_min = scalar_doubles[0];
        const double blur_max = scalar_doubles[1];
        const double blur_range = blur_max - blur_min;
        if (!(blur_range > 0.0)) {
            throw std::runtime_error(
                "star_detect_fused_pixel_components: blurred image has zero range");
        }
        const double blur_mean = blur_sum / static_cast<double>(total);
        normalize_kernel<<<blocks, threads, 0, stream>>>(blur.get(), normalized.get(), blur_mean,
                                                         blur_range, total);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components normalize launch");
        blur.reset();
        image.reset();

        resize_linear_kernel<<<small_blocks, threads, 0, stream>>>(
            normalized.get(), small_blur.get(), height, width, small_height, small_width);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components blur resize launch");
        normalized.reset();

        DeviceImage rec_small = wavelet_dec_rec_device(
            std::move(small_blur), small_height, small_width, level, threads, &workspace, stream);

        resize_linear_mask_kernel<<<blocks, threads, 0, stream>>>(rec_small.data.get(), mask.get(),
                                                                  img_rec.get(), rec_small.h,
                                                                  rec_small.w, height, width);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components resize up launch");

        throw_if_cuda_failed(cudaMemsetAsync(count.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset count");
        compact_masked_values_kernel<<<blocks, threads, 0, stream>>>(
            img_rec.get(), mask.get(), values.get(), count.get(), total);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components compact launch");

        throw_if_cuda_failed(
            cudaMemcpyAsync(scalar_int, count.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
            "star_detect_fused_pixel_components cudaMemcpy count");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components compact sync");
        const int masked_count = *scalar_int;
        rec_small.data.reset();
        if (masked_count <= 0) {
            throw std::runtime_error("star_detect_fused_pixel_components: mask selects no pixels");
        }

        thrust::device_ptr<double> values_ptr(values.get());
        run_thrust_with_resource_translation("star_detect thrust sort allocation", [&] {
            thrust::sort(thrust::cuda::par.on(stream), values_ptr, values_ptr + masked_count);
        });
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components sort sync");
        const double threshold =
            percentile_995_sorted(values.get(), masked_count, stream, scalar_doubles);
        values.reset();

        erode_threshold_kernel<<<blocks, threads, 0, stream>>>(
            img_rec.get(), mask.get(), eroded.get(), height, width, threshold);
        throw_if_cuda_failed(cudaGetLastError(), "star_detect_fused_pixel_components erode launch");
        dilate_kernel<<<blocks, threads, 0, stream>>>(eroded.get(), bw.get(), height, width);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components dilate launch");
        eroded.reset();
        mask.reset();
        throw_if_cuda_failed(
            cudaMemcpyAsync(binary_mask_host, bw.get(), mask_bytes, cudaMemcpyDeviceToHost, stream),
            "star_detect_fused_pixel_components cudaMemcpy binary mask");

        foreground_indices.allocate(
            plane_size, "star_detect_fused_pixel_components cudaMalloc foreground indices",
            &workspace);
        throw_if_cuda_failed(cudaMemsetAsync(count.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset foreground count");
        compact_foreground_indices_kernel<<<blocks, threads, 0, stream>>>(
            bw.get(), foreground_indices.get(), count.get(), total);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components foreground compact launch");

        throw_if_cuda_failed(
            cudaMemcpyAsync(scalar_int, count.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
            "star_detect_fused_pixel_components cudaMemcpy foreground count");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components foreground sync");
        const int foreground_count = *scalar_int;
        if (foreground_count <= 0) {
            return;
        }
        if (foreground_count > total / hnw::star_detect::kMaxForegroundDivisor) {
            throw hnw::StarDetectCapacityError("star_detect_fused_pixel_components: "
                                               "foreground too dense for GPU CC");
        }

        labels_a.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc labels a",
                          &workspace);
        labels_b.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc labels b",
                          &workspace);
        changed.allocate(1, "star_detect_fused_pixel_components cudaMalloc changed", &workspace);
        init_component_labels_kernel<<<blocks, threads, 0, stream>>>(bw.get(), labels_a.get(),
                                                                     total);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components label init launch");
        throw_if_cuda_failed(cudaMemsetAsync(labels_b.get(), 0, plane_size * sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset labels b");
        bw.reset();

        bool converged = false;
        const int max_label_iterations =
            std::min(std::max(height, width), hnw::star_detect::kMaxLabelIterations);
        const int component_blocks = (foreground_count + threads - 1) / threads;
        for (int iter = 0; iter < max_label_iterations; ++iter) {
            throw_if_cuda_failed(cudaMemsetAsync(changed.get(), 0, sizeof(int), stream),
                                 "star_detect_fused_pixel_components cudaMemset changed");
            propagate_foreground_component_labels_kernel<<<component_blocks, threads, 0, stream>>>(
                foreground_indices.get(), foreground_count, labels_a.get(), labels_b.get(),
                changed.get(), height, width);
            throw_if_cuda_failed(cudaGetLastError(),
                                 "star_detect_fused_pixel_components label propagation launch");
            throw_if_cuda_failed(cudaMemcpyAsync(scalar_int, changed.get(), sizeof(int),
                                                 cudaMemcpyDeviceToHost, stream),
                                 "star_detect_fused_pixel_components cudaMemcpy changed");
            throw_if_cuda_failed(cudaStreamSynchronize(stream),
                                 "star_detect_fused_pixel_components label propagation sync");
            const int changed_count = *scalar_int;
            std::swap(labels_a, labels_b);
            if (changed_count == 0) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            throw hnw::StarDetectCapacityError(
                "star_detect_fused_pixel_components: GPU CC did not converge");
        }

        if (foreground_count > std::numeric_limits<int>::max() / 2) {
            throw hnw::StarDetectCapacityError(
                "star_detect_fused_pixel_components: too many foreground pixels");
        }
        const int hash_capacity = next_power_of_two_int(
            std::max(hnw::star_detect::kMinHashCapacity,
                     foreground_count * hnw::star_detect::kHashCapacityMultiplier));
        const int hash_blocks = (hash_capacity + threads - 1) / threads;
        keys.allocate(hash_capacity, "star_detect_fused_pixel_components cudaMalloc hash keys",
                      &workspace);
        counts.allocate(hash_capacity, "star_detect_fused_pixel_components cudaMalloc hash counts",
                        &workspace);
        overflow.allocate(1, "star_detect_fused_pixel_components cudaMalloc overflow", &workspace);
        sum_x.allocate(hash_capacity, "star_detect_fused_pixel_components cudaMalloc sum_x",
                       &workspace);
        sum_y.allocate(hash_capacity, "star_detect_fused_pixel_components cudaMalloc sum_y",
                       &workspace);
        sum_intensity.allocate(hash_capacity,
                               "star_detect_fused_pixel_components cudaMalloc sum_intensity",
                               &workspace);

        throw_if_cuda_failed(cudaMemsetAsync(keys.get(), 0, hash_capacity * sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset keys");
        throw_if_cuda_failed(cudaMemsetAsync(counts.get(), 0, hash_capacity * sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset counts");
        throw_if_cuda_failed(cudaMemsetAsync(overflow.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset overflow");
        throw_if_cuda_failed(
            cudaMemsetAsync(sum_x.get(), 0, hash_capacity * sizeof(double), stream),
            "star_detect_fused_pixel_components cudaMemset sum_x");
        throw_if_cuda_failed(
            cudaMemsetAsync(sum_y.get(), 0, hash_capacity * sizeof(double), stream),
            "star_detect_fused_pixel_components cudaMemset sum_y");
        throw_if_cuda_failed(
            cudaMemsetAsync(sum_intensity.get(), 0, hash_capacity * sizeof(double), stream),
            "star_detect_fused_pixel_components cudaMemset sum_intensity");

        const int foreground_blocks = (foreground_count + threads - 1) / threads;
        accumulate_component_stats_kernel<<<foreground_blocks, threads, 0, stream>>>(
            foreground_indices.get(), foreground_count, labels_a.get(), img_rec.get(), keys.get(),
            counts.get(), sum_x.get(), sum_y.get(), sum_intensity.get(), overflow.get(), width,
            hash_capacity);
        throw_if_cuda_failed(cudaGetLastError(), "star_detect_fused_pixel_components stats launch");
        foreground_indices.reset();
        labels_a.reset();
        labels_b.reset();
        img_rec.reset();

        throw_if_cuda_failed(cudaMemcpyAsync(scalar_int, overflow.get(), sizeof(int),
                                             cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy overflow");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components overflow sync");
        const int overflow_flag = *scalar_int;
        if (overflow_flag != 0) {
            throw hnw::StarDetectCapacityError(
                "star_detect_fused_pixel_components: component hash table overflow");
        }

        throw_if_cuda_failed(cudaMemsetAsync(count.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset output count");
        count_component_outputs_kernel<<<hash_blocks, threads, 0, stream>>>(
            keys.get(), counts.get(), count.get(), hash_capacity);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components output count launch");

        throw_if_cuda_failed(
            cudaMemcpyAsync(scalar_int, count.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
            "star_detect_fused_pixel_components cudaMemcpy output count");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components output count sync");
        const int output_count = *scalar_int;
        if (output_count <= 0) {
            return;
        }

        out_positions.allocate(static_cast<size_t>(output_count) * 2,
                               "star_detect_fused_pixel_components cudaMalloc output positions",
                               &workspace);
        out_intensities.allocate(static_cast<size_t>(output_count),
                                 "star_detect_fused_pixel_components cudaMalloc output intensities",
                                 &workspace);

        throw_if_cuda_failed(cudaMemsetAsync(count.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset output index");
        fill_component_outputs_kernel<<<hash_blocks, threads, 0, stream>>>(
            keys.get(), counts.get(), sum_x.get(), sum_y.get(), sum_intensity.get(),
            out_positions.get(), out_intensities.get(), count.get(), hash_capacity);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components output fill launch");

        positions_xy_host->resize(static_cast<size_t>(output_count) * 2);
        intensities_host->resize(static_cast<size_t>(output_count));
        const size_t positions_bytes = positions_xy_host->size() * sizeof(double);
        const size_t intensities_bytes = intensities_host->size() * sizeof(double);
        throw_if_cuda_failed(cudaMemcpyAsync(positions_xy_host->data(), out_positions.get(),
                                             positions_bytes, cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy positions");
        throw_if_cuda_failed(cudaMemcpyAsync(intensities_host->data(), out_intensities.get(),
                                             intensities_bytes, cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy intensities");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components final sync");
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}
