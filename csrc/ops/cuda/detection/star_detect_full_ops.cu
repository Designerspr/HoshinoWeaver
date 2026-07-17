#include "../wavelet/wavelet_device.cuh"
#include "common/compat.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int next_power_of_two_int(const int value) {
    if (value <= 1) {
        return 1;
    }
    if (value > (std::numeric_limits<int>::max() / 2 + 1)) {
        throw std::runtime_error("star_detect_connected_components: hash table is too large");
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

__global__ void threshold_small_dark_mask_kernel(const double* image, uint8_t* mask,
                                                 const int total, const double threshold) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    mask[idx] = image[idx] < threshold ? 255 : 0;
}

__global__ void dilate_mask_kernel(const uint8_t* input, const uint8_t* kernel, uint8_t* output,
                                   const int height, const int width, const int kernel_h,
                                   const int kernel_w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int y = idx / width;
    const int x = idx - y * width;
    const int anchor_y = kernel_h / 2;
    const int anchor_x = kernel_w / 2;
    uint8_t value = 0;
    for (int ky = 0; ky < kernel_h; ++ky) {
        const int yy = y + ky - anchor_y;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int kx = 0; kx < kernel_w; ++kx) {
            if (kernel[ky * kernel_w + kx] == 0) {
                continue;
            }
            const int xx = x + kx - anchor_x;
            if (xx < 0 || xx >= width) {
                continue;
            }
            value = max(value, input[yy * width + xx]);
        }
    }
    output[idx] = value;
}

__global__ void resize_allowed_mask_kernel(const uint8_t* dark_mask, const uint8_t* external_mask,
                                           uint8_t* output, int* allowed_count, const int in_h,
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
    const double v00 = static_cast<double>(dark_mask[y0 * in_w + x0]);
    const double v01 = static_cast<double>(dark_mask[y0 * in_w + x1]);
    const double v10 = static_cast<double>(dark_mask[y1 * in_w + x0]);
    const double v11 = static_cast<double>(dark_mask[y1 * in_w + x1]);
    const double top = v00 + (v01 - v00) * wx;
    const double bottom = v10 + (v11 - v10) * wx;
    const double resized_dark = top + (bottom - top) * wy;
    uint8_t allowed = resized_dark > 127.0 ? 0 : 1;
    if (external_mask != nullptr && external_mask[idx] == 0) {
        allowed = 0;
    }
    output[idx] = allowed;
    if (allowed != 0) {
        atomicAdd(allowed_count, 1);
    }
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

__global__ void propagate_component_labels_kernel(const int* in_labels, int* out_labels,
                                                  int* changed, const int height, const int width) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int current = in_labels[idx];
    if (current == 0) {
        out_labels[idx] = 0;
        return;
    }

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
                                                  double* sum_x, double* sum_y, double* sum_x2,
                                                  double* sum_y2, double* sum_xy,
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
    atomicAdd(&sum_x2[slot], x * x);
    atomicAdd(&sum_y2[slot], y * y);
    atomicAdd(&sum_xy[slot], x * y);
    atomicAdd(&sum_intensity[slot], image[idx]);
}

__global__ void count_component_outputs_kernel(const int* keys, const int* counts, int* out_count,
                                               const int hash_capacity) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hash_capacity) {
        return;
    }
    if (keys[idx] != 0 && counts[idx] > 5) {
        atomicAdd(out_count, 1);
    }
}

__global__ void fill_component_outputs_kernel(const int* keys, const int* counts,
                                              const double* sum_x, const double* sum_y,
                                              const double* sum_x2, const double* sum_y2,
                                              const double* sum_xy, const double* sum_intensity,
                                              double* positions, double* areas, double* intensities,
                                              double* eccentricities, int* out_index,
                                              const int hash_capacity) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hash_capacity || keys[idx] == 0 || counts[idx] <= 5) {
        return;
    }

    const int out = atomicAdd(out_index, 1);
    const double count = static_cast<double>(counts[idx]);
    const double inv_count = 1.0 / count;
    const double cx = sum_x[idx] * inv_count;
    const double cy = sum_y[idx] * inv_count;
    const double cov_xx = sum_x2[idx] * inv_count - cx * cx;
    const double cov_yy = sum_y2[idx] * inv_count - cy * cy;
    const double cov_xy = sum_xy[idx] * inv_count - cx * cy;
    const double trace = cov_xx + cov_yy;
    const double det_term =
        sqrt(fmax(0.0, (cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy));
    const double lambda_max = 0.5 * (trace + det_term);
    const double lambda_min = 0.5 * (trace - det_term);
    double eccentricity = 0.0;
    if (lambda_max > 1e-12) {
        eccentricity = sqrt(fmax(0.0, 1.0 - lambda_min / lambda_max));
    }

    positions[out * 2] = cx;
    positions[out * 2 + 1] = cy;
    areas[out] = count;
    intensities[out] = sum_intensity[idx] * inv_count;
    eccentricities[out] = eccentricity;
}

double percentile_995_sorted(const double* values, const int count) {
    const double rank = 0.995 * static_cast<double>(count - 1);
    const int lower_idx = static_cast<int>(floor(rank));
    const int upper_idx = static_cast<int>(ceil(rank));
    const double weight = rank - static_cast<double>(lower_idx);
    double lower = 0.0;
    double upper = 0.0;
    throw_if_cuda_failed(
        cudaMemcpy(&lower, values + lower_idx, sizeof(double), cudaMemcpyDeviceToHost),
        "star_detect percentile cudaMemcpy lower percentile");
    throw_if_cuda_failed(
        cudaMemcpy(&upper, values + upper_idx, sizeof(double), cudaMemcpyDeviceToHost),
        "star_detect percentile cudaMemcpy upper percentile");
    return lower + (upper - lower) * weight;
}

double percentile_sorted(const double* values, const int count, const double percentile,
                         const char* context) {
    const double rank = percentile * static_cast<double>(count - 1);
    const int lower_idx = static_cast<int>(floor(rank));
    const int upper_idx = static_cast<int>(ceil(rank));
    const double weight = rank - static_cast<double>(lower_idx);
    double lower = 0.0;
    double upper = 0.0;
    throw_if_cuda_failed(
        cudaMemcpy(&lower, values + lower_idx, sizeof(double), cudaMemcpyDeviceToHost), context);
    throw_if_cuda_failed(
        cudaMemcpy(&upper, values + upper_idx, sizeof(double), cudaMemcpyDeviceToHost), context);
    return lower + (upper - lower) * weight;
}

} // namespace

void launch_star_detect_full_connected_components(
    const double* image_host, const uint8_t* external_mask_host, const double* gaussian_kernel_host,
    const uint8_t* dilate_kernel_host, std::vector<double>* positions_xy_host,
    std::vector<double>* areas_host, std::vector<double>* intensities_host,
    std::vector<double>* eccentricities_host, const int height, const int width,
    const int small_height, const int small_width, const int level, const int gaussian_ksize,
    const int dilate_height, const int dilate_width) {
    const int threads = 256;
    const int total = height * width;
    const int small_total = small_height * small_width;
    const size_t plane_size = static_cast<size_t>(total);
    const size_t small_size = static_cast<size_t>(small_total);

    DeviceBuffer image;
    DeviceBuffer gaussian_kernel;
    DeviceBuffer blur_rows;
    DeviceBuffer blur;
    DeviceBuffer normalized;
    DeviceBuffer small_mask_values;
    DeviceBuffer small_mask_sorted;
    DeviceBuffer small_blur;
    DeviceBuffer img_rec;
    DeviceBuffer values;
    DeviceTypedBuffer<uint8_t> external_mask;
    DeviceTypedBuffer<uint8_t> dilate_structuring;
    DeviceTypedBuffer<uint8_t> small_dark_mask;
    DeviceTypedBuffer<uint8_t> small_dilated_mask;
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
    DeviceBuffer sum_x2;
    DeviceBuffer sum_y2;
    DeviceBuffer sum_xy;
    DeviceBuffer sum_intensity;
    DeviceBuffer out_positions;
    DeviceBuffer out_areas;
    DeviceBuffer out_intensities;
    DeviceBuffer out_eccentricities;

    positions_xy_host->clear();
    areas_host->clear();
    intensities_host->clear();
    eccentricities_host->clear();

    image.allocate(plane_size, "star_detect_full_connected_components cudaMalloc image");
    gaussian_kernel.allocate(static_cast<size_t>(gaussian_ksize),
                             "star_detect_full_connected_components cudaMalloc gaussian kernel");
    blur_rows.allocate(plane_size, "star_detect_full_connected_components cudaMalloc blur rows");
    blur.allocate(plane_size, "star_detect_full_connected_components cudaMalloc blur");
    normalized.allocate(plane_size, "star_detect_full_connected_components cudaMalloc normalized");
    small_mask_values.allocate(
        small_size, "star_detect_full_connected_components cudaMalloc small mask values");
    small_mask_sorted.allocate(
        small_size, "star_detect_full_connected_components cudaMalloc small mask sorted");
    small_blur.allocate(small_size, "star_detect_full_connected_components cudaMalloc small blur");
    img_rec.allocate(plane_size, "star_detect_full_connected_components cudaMalloc img_rec");
    values.allocate(plane_size, "star_detect_full_connected_components cudaMalloc values");
    dilate_structuring.allocate(static_cast<size_t>(dilate_height) *
                                    static_cast<size_t>(dilate_width),
                                "star_detect_full_connected_components cudaMalloc dilate kernel");
    small_dark_mask.allocate(small_size,
                             "star_detect_full_connected_components cudaMalloc small dark mask");
    small_dilated_mask.allocate(
        small_size, "star_detect_full_connected_components cudaMalloc small dilated mask");
    mask.allocate(plane_size, "star_detect_full_connected_components cudaMalloc mask");
    eroded.allocate(plane_size, "star_detect_full_connected_components cudaMalloc eroded");
    bw.allocate(plane_size, "star_detect_full_connected_components cudaMalloc bw");
    count.allocate(1, "star_detect_full_connected_components cudaMalloc count");
    if (external_mask_host != nullptr) {
        external_mask.allocate(plane_size,
                               "star_detect_full_connected_components cudaMalloc external mask");
    }

    throw_if_cuda_failed(
        cudaMemcpy(image.get(), image_host, plane_size * sizeof(double), cudaMemcpyHostToDevice),
        "star_detect_full_connected_components cudaMemcpy image");
    throw_if_cuda_failed(cudaMemcpy(gaussian_kernel.get(), gaussian_kernel_host,
                                    static_cast<size_t>(gaussian_ksize) * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "star_detect_full_connected_components cudaMemcpy gaussian kernel");
    throw_if_cuda_failed(cudaMemcpy(dilate_structuring.get(), dilate_kernel_host,
                                    static_cast<size_t>(dilate_height) *
                                        static_cast<size_t>(dilate_width) * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice),
                         "star_detect_full_connected_components cudaMemcpy dilate kernel");
    if (external_mask_host != nullptr) {
        throw_if_cuda_failed(cudaMemcpy(external_mask.get(), external_mask_host,
                                        plane_size * sizeof(uint8_t), cudaMemcpyHostToDevice),
                             "star_detect_full_connected_components cudaMemcpy external mask");
    }

    const int blocks = (total + threads - 1) / threads;
    const int small_blocks = (small_total + threads - 1) / threads;

    gaussian_rows_kernel<<<blocks, threads>>>(image.get(), gaussian_kernel.get(), blur_rows.get(),
                                              height, width, gaussian_ksize);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components gaussian rows launch");
    gaussian_cols_kernel<<<blocks, threads>>>(blur_rows.get(), gaussian_kernel.get(), blur.get(),
                                              height, width, gaussian_ksize);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components gaussian cols launch");
    blur_rows.reset();
    gaussian_kernel.reset();

    thrust::device_ptr<double> blur_ptr(blur.get());
    const double blur_sum = thrust::reduce(thrust::device, blur_ptr, blur_ptr + total, 0.0);
    const auto minmax = thrust::minmax_element(thrust::device, blur_ptr, blur_ptr + total);
    double blur_min = 0.0;
    double blur_max = 0.0;
    throw_if_cuda_failed(cudaMemcpy(&blur_min, thrust::raw_pointer_cast(minmax.first),
                                    sizeof(double), cudaMemcpyDeviceToHost),
                         "star_detect_full_connected_components cudaMemcpy blur min");
    throw_if_cuda_failed(cudaMemcpy(&blur_max, thrust::raw_pointer_cast(minmax.second),
                                    sizeof(double), cudaMemcpyDeviceToHost),
                         "star_detect_full_connected_components cudaMemcpy blur max");
    const double blur_range = blur_max - blur_min;
    if (!(blur_range > 0.0)) {
        throw std::runtime_error(
            "star_detect_full_connected_components: blurred image has zero range");
    }
    const double blur_mean = blur_sum / static_cast<double>(total);
    normalize_kernel<<<blocks, threads>>>(blur.get(), normalized.get(), blur_mean, blur_range,
                                          total);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components normalize launch");
    blur.reset();

    resize_linear_kernel<<<small_blocks, threads>>>(image.get(), small_mask_values.get(), height,
                                                    width, small_height, small_width);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components mask resize launch");
    image.reset();

    throw_if_cuda_failed(cudaMemcpy(small_mask_sorted.get(), small_mask_values.get(),
                                    small_size * sizeof(double), cudaMemcpyDeviceToDevice),
                         "star_detect_full_connected_components cudaMemcpy small mask sorted");
    thrust::device_ptr<double> small_values_ptr(small_mask_sorted.get());
    thrust::sort(thrust::device, small_values_ptr, small_values_ptr + small_total);
    throw_if_cuda_failed(cudaDeviceSynchronize(),
                         "star_detect_full_connected_components mask sort sync");
    const double mask_threshold = std::min(
        percentile_sorted(small_mask_sorted.get(), small_total, 0.10,
                          "star_detect_full_connected_components cudaMemcpy mask percentile"),
        0.15);
    small_mask_sorted.reset();
    threshold_small_dark_mask_kernel<<<small_blocks, threads>>>(
        small_mask_values.get(), small_dark_mask.get(), small_total, mask_threshold);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components mask threshold launch");
    small_mask_values.reset();

    dilate_mask_kernel<<<small_blocks, threads>>>(small_dark_mask.get(), dilate_structuring.get(),
                                                  small_dilated_mask.get(), small_height,
                                                  small_width, dilate_height, dilate_width);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components mask dilate launch");
    small_dark_mask.reset();
    dilate_structuring.reset();

    throw_if_cuda_failed(cudaMemset(count.get(), 0, sizeof(int)),
                         "star_detect_full_connected_components cudaMemset mask count");
    resize_allowed_mask_kernel<<<blocks, threads>>>(
        small_dilated_mask.get(), external_mask_host == nullptr ? nullptr : external_mask.get(),
        mask.get(), count.get(), small_height, small_width, height, width);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components mask resize up launch");
    small_dilated_mask.reset();
    external_mask.reset();

    int allowed_count = 0;
    throw_if_cuda_failed(
        cudaMemcpy(&allowed_count, count.get(), sizeof(int), cudaMemcpyDeviceToHost),
        "star_detect_full_connected_components cudaMemcpy mask count");
    if (static_cast<long long>(allowed_count) * 100LL < static_cast<long long>(total) * 50LL) {
        throw_if_cuda_failed(cudaMemset(mask.get(), 1, plane_size * sizeof(uint8_t)),
                             "star_detect_full_connected_components cudaMemset full mask");
    }

    resize_linear_kernel<<<small_blocks, threads>>>(normalized.get(), small_blur.get(), height,
                                                    width, small_height, small_width);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components blur resize launch");
    normalized.reset();

    DeviceImage rec_small =
        wavelet_dec_rec_device(std::move(small_blur), small_height, small_width, level, threads);

    resize_linear_mask_kernel<<<blocks, threads>>>(rec_small.data.get(), mask.get(), img_rec.get(),
                                                   rec_small.h, rec_small.w, height, width);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components resize up launch");

    throw_if_cuda_failed(cudaMemset(count.get(), 0, sizeof(int)),
                         "star_detect_full_connected_components cudaMemset count");
    compact_masked_values_kernel<<<blocks, threads>>>(img_rec.get(), mask.get(), values.get(),
                                                      count.get(), total);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components compact launch");

    int masked_count = 0;
    throw_if_cuda_failed(
        cudaMemcpy(&masked_count, count.get(), sizeof(int), cudaMemcpyDeviceToHost),
        "star_detect_full_connected_components cudaMemcpy count");
    throw_if_cuda_failed(cudaDeviceSynchronize(),
                         "star_detect_full_connected_components compact sync");
    if (masked_count <= 0) {
        throw std::runtime_error("star_detect_full_connected_components: mask selects no pixels");
    }

    thrust::device_ptr<double> values_ptr(values.get());
    thrust::sort(thrust::device, values_ptr, values_ptr + masked_count);
    throw_if_cuda_failed(cudaDeviceSynchronize(),
                         "star_detect_full_connected_components sort sync");
    const double threshold = percentile_995_sorted(values.get(), masked_count);
    values.reset();

    erode_threshold_kernel<<<blocks, threads>>>(img_rec.get(), mask.get(), eroded.get(), height,
                                                width, threshold);
    throw_if_cuda_failed(cudaGetLastError(), "star_detect_full_connected_components erode launch");
    dilate_kernel<<<blocks, threads>>>(eroded.get(), bw.get(), height, width);
    throw_if_cuda_failed(cudaGetLastError(), "star_detect_full_connected_components dilate launch");
    eroded.reset();
    mask.reset();

    foreground_indices.allocate(
        plane_size, "star_detect_full_connected_components cudaMalloc foreground indices");
    throw_if_cuda_failed(cudaMemset(count.get(), 0, sizeof(int)),
                         "star_detect_full_connected_components cudaMemset foreground count");
    compact_foreground_indices_kernel<<<blocks, threads>>>(bw.get(), foreground_indices.get(),
                                                           count.get(), total);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components foreground compact launch");

    int foreground_count = 0;
    throw_if_cuda_failed(
        cudaMemcpy(&foreground_count, count.get(), sizeof(int), cudaMemcpyDeviceToHost),
        "star_detect_full_connected_components cudaMemcpy foreground count");
    throw_if_cuda_failed(cudaDeviceSynchronize(),
                         "star_detect_full_connected_components foreground sync");
    if (foreground_count <= 0) {
        return;
    }
    if (foreground_count > total / 4) {
        throw std::runtime_error(
            "star_detect_full_connected_components: foreground too dense for GPU CC");
    }

    labels_a.allocate(plane_size, "star_detect_full_connected_components cudaMalloc labels a");
    labels_b.allocate(plane_size, "star_detect_full_connected_components cudaMalloc labels b");
    changed.allocate(1, "star_detect_full_connected_components cudaMalloc changed");
    init_component_labels_kernel<<<blocks, threads>>>(bw.get(), labels_a.get(), total);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components label init launch");
    bw.reset();

    bool converged = false;
    const int max_label_iterations = std::min(std::max(height, width), 1024);
    for (int iter = 0; iter < max_label_iterations; ++iter) {
        throw_if_cuda_failed(cudaMemset(changed.get(), 0, sizeof(int)),
                             "star_detect_full_connected_components cudaMemset changed");
        propagate_component_labels_kernel<<<blocks, threads>>>(labels_a.get(), labels_b.get(),
                                                               changed.get(), height, width);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_full_connected_components label propagation launch");
        int changed_count = 0;
        throw_if_cuda_failed(
            cudaMemcpy(&changed_count, changed.get(), sizeof(int), cudaMemcpyDeviceToHost),
            "star_detect_full_connected_components cudaMemcpy changed");
        std::swap(labels_a, labels_b);
        if (changed_count == 0) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        throw std::runtime_error("star_detect_full_connected_components: GPU CC did not converge");
    }

    if (foreground_count > std::numeric_limits<int>::max() / 2) {
        throw std::runtime_error(
            "star_detect_full_connected_components: too many foreground pixels");
    }
    const int hash_capacity = next_power_of_two_int(std::max(16, foreground_count * 2));
    const int hash_blocks = (hash_capacity + threads - 1) / threads;
    keys.allocate(hash_capacity, "star_detect_full_connected_components cudaMalloc hash keys");
    counts.allocate(hash_capacity, "star_detect_full_connected_components cudaMalloc hash counts");
    overflow.allocate(1, "star_detect_full_connected_components cudaMalloc overflow");
    sum_x.allocate(hash_capacity, "star_detect_full_connected_components cudaMalloc sum_x");
    sum_y.allocate(hash_capacity, "star_detect_full_connected_components cudaMalloc sum_y");
    sum_x2.allocate(hash_capacity, "star_detect_full_connected_components cudaMalloc sum_x2");
    sum_y2.allocate(hash_capacity, "star_detect_full_connected_components cudaMalloc sum_y2");
    sum_xy.allocate(hash_capacity, "star_detect_full_connected_components cudaMalloc sum_xy");
    sum_intensity.allocate(hash_capacity,
                           "star_detect_full_connected_components cudaMalloc sum_intensity");

    throw_if_cuda_failed(cudaMemset(keys.get(), 0, hash_capacity * sizeof(int)),
                         "star_detect_full_connected_components cudaMemset keys");
    throw_if_cuda_failed(cudaMemset(counts.get(), 0, hash_capacity * sizeof(int)),
                         "star_detect_full_connected_components cudaMemset counts");
    throw_if_cuda_failed(cudaMemset(overflow.get(), 0, sizeof(int)),
                         "star_detect_full_connected_components cudaMemset overflow");
    throw_if_cuda_failed(cudaMemset(sum_x.get(), 0, hash_capacity * sizeof(double)),
                         "star_detect_full_connected_components cudaMemset sum_x");
    throw_if_cuda_failed(cudaMemset(sum_y.get(), 0, hash_capacity * sizeof(double)),
                         "star_detect_full_connected_components cudaMemset sum_y");
    throw_if_cuda_failed(cudaMemset(sum_x2.get(), 0, hash_capacity * sizeof(double)),
                         "star_detect_full_connected_components cudaMemset sum_x2");
    throw_if_cuda_failed(cudaMemset(sum_y2.get(), 0, hash_capacity * sizeof(double)),
                         "star_detect_full_connected_components cudaMemset sum_y2");
    throw_if_cuda_failed(cudaMemset(sum_xy.get(), 0, hash_capacity * sizeof(double)),
                         "star_detect_full_connected_components cudaMemset sum_xy");
    throw_if_cuda_failed(cudaMemset(sum_intensity.get(), 0, hash_capacity * sizeof(double)),
                         "star_detect_full_connected_components cudaMemset sum_intensity");

    const int foreground_blocks = (foreground_count + threads - 1) / threads;
    accumulate_component_stats_kernel<<<foreground_blocks, threads>>>(
        foreground_indices.get(), foreground_count, labels_a.get(), img_rec.get(), keys.get(),
        counts.get(), sum_x.get(), sum_y.get(), sum_x2.get(), sum_y2.get(), sum_xy.get(),
        sum_intensity.get(), overflow.get(), width, hash_capacity);
    throw_if_cuda_failed(cudaGetLastError(), "star_detect_full_connected_components stats launch");
    foreground_indices.reset();
    labels_a.reset();
    labels_b.reset();
    img_rec.reset();

    int overflow_flag = 0;
    throw_if_cuda_failed(
        cudaMemcpy(&overflow_flag, overflow.get(), sizeof(int), cudaMemcpyDeviceToHost),
        "star_detect_full_connected_components cudaMemcpy overflow");
    if (overflow_flag != 0) {
        throw std::runtime_error(
            "star_detect_full_connected_components: component hash table overflow");
    }

    throw_if_cuda_failed(cudaMemset(count.get(), 0, sizeof(int)),
                         "star_detect_full_connected_components cudaMemset output count");
    count_component_outputs_kernel<<<hash_blocks, threads>>>(keys.get(), counts.get(), count.get(),
                                                             hash_capacity);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components output count launch");

    int output_count = 0;
    throw_if_cuda_failed(
        cudaMemcpy(&output_count, count.get(), sizeof(int), cudaMemcpyDeviceToHost),
        "star_detect_full_connected_components cudaMemcpy output count");
    if (output_count <= 0) {
        return;
    }

    out_positions.allocate(static_cast<size_t>(output_count) * 2,
                           "star_detect_full_connected_components cudaMalloc output positions");
    out_areas.allocate(static_cast<size_t>(output_count),
                       "star_detect_full_connected_components cudaMalloc output areas");
    out_intensities.allocate(static_cast<size_t>(output_count),
                             "star_detect_full_connected_components cudaMalloc output intensities");
    out_eccentricities.allocate(
        static_cast<size_t>(output_count),
        "star_detect_full_connected_components cudaMalloc output eccentricities");

    throw_if_cuda_failed(cudaMemset(count.get(), 0, sizeof(int)),
                         "star_detect_full_connected_components cudaMemset output index");
    fill_component_outputs_kernel<<<hash_blocks, threads>>>(
        keys.get(), counts.get(), sum_x.get(), sum_y.get(), sum_x2.get(), sum_y2.get(),
        sum_xy.get(), sum_intensity.get(), out_positions.get(), out_areas.get(),
        out_intensities.get(), out_eccentricities.get(), count.get(), hash_capacity);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_full_connected_components output fill launch");

    positions_xy_host->resize(static_cast<size_t>(output_count) * 2);
    areas_host->resize(static_cast<size_t>(output_count));
    intensities_host->resize(static_cast<size_t>(output_count));
    eccentricities_host->resize(static_cast<size_t>(output_count));
    throw_if_cuda_failed(cudaMemcpy(positions_xy_host->data(), out_positions.get(),
                                    positions_xy_host->size() * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "star_detect_full_connected_components cudaMemcpy positions");
    throw_if_cuda_failed(cudaMemcpy(areas_host->data(), out_areas.get(),
                                    areas_host->size() * sizeof(double), cudaMemcpyDeviceToHost),
                         "star_detect_full_connected_components cudaMemcpy areas");
    throw_if_cuda_failed(cudaMemcpy(intensities_host->data(), out_intensities.get(),
                                    intensities_host->size() * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "star_detect_full_connected_components cudaMemcpy intensities");
    throw_if_cuda_failed(cudaMemcpy(eccentricities_host->data(), out_eccentricities.get(),
                                    eccentricities_host->size() * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "star_detect_full_connected_components cudaMemcpy eccentricities");
    throw_if_cuda_failed(cudaDeviceSynchronize(),
                         "star_detect_full_connected_components final sync");
}
