#include "common/compat.h"
#include "common/cuda_host_io_workspace.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

template <typename T> __device__ inline T cast_output(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        value = fminf(fmaxf(value, 0.0f), 255.0f);
        return static_cast<unsigned char>(nearbyintf(value));
    } else {
        value = fminf(fmaxf(value, 0.0f), 65535.0f);
        return static_cast<unsigned short>(nearbyintf(value));
    }
}

__device__ inline size_t source_offset(const int y, const int x, const int src_width,
                                       const int channels, const int c) {
    return (static_cast<size_t>(y) * static_cast<size_t>(src_width) + static_cast<size_t>(x)) *
               static_cast<size_t>(channels) +
           static_cast<size_t>(c);
}

template <typename T>
__global__ void camera_model_remap_fused_kernel(
    const T* image, T* out, const int src_height, const int src_width, const int channels,
    const int out_height, const int out_width, const float fx_src, const float fy_src,
    const float cx_src, const float cy_src, const float fx_dst, const float fy_dst,
    const float cx_dst, const float cy_dst, const float r00, const float r01, const float r02,
    const float r10, const float r11, const float r12, const float r20, const float r21,
    const float r22, const bool src_has_dist, const float src_k1, const float src_k2,
    const float src_p1, const float src_p2, const float src_k3, const bool dst_has_dist,
    const float dst_k1, const float dst_k2, const float dst_p1, const float dst_p2,
    const float dst_k3) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_height * out_width;
    if (idx >= total) {
        return;
    }

    const int row = idx / out_width;
    const int col = idx - row * out_width;
    const float xd = (static_cast<float>(col) - cx_dst) / fx_dst;
    const float yd = (static_cast<float>(row) - cy_dst) / fy_dst;
    float x = xd;
    float y = yd;
    if (dst_has_dist) {
        // OpenCV undistortPoints uses iterative inversion for the Brown-Conrady model.
        // Fixed iteration keeps runtime predictable and is sufficient for camera remap.
        for (int iter = 0; iter < 5; ++iter) {
            const float r2 = x * x + y * y;
            const float r4 = r2 * r2;
            const float r6 = r4 * r2;
            const float radial = 1.0f + dst_k1 * r2 + dst_k2 * r4 + dst_k3 * r6;
            const float xy2 = 2.0f * x * y;
            const float delta_x = dst_p1 * xy2 + dst_p2 * (r2 + 2.0f * x * x);
            const float delta_y = dst_p1 * (r2 + 2.0f * y * y) + dst_p2 * xy2;
            x = (xd - delta_x) / radial;
            y = (yd - delta_y) / radial;
        }
    }

    const float proj_x = r00 * x + r01 * y + r02;
    const float proj_y = r10 * x + r11 * y + r12;
    const float proj_z = r20 * x + r21 * y + r22;
    const size_t out_base = static_cast<size_t>(idx) * static_cast<size_t>(channels);

    if (!(proj_z > 0.0f)) {
        for (int c = 0; c < channels; ++c) {
            out[out_base + c] = static_cast<T>(0);
        }
        return;
    }

    const float inv_z = 1.0f / proj_z;
    float src_x_norm = proj_x * inv_z;
    float src_y_norm = proj_y * inv_z;
    if (src_has_dist) {
        const float r2 = src_x_norm * src_x_norm + src_y_norm * src_y_norm;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        const float radial = 1.0f + src_k1 * r2 + src_k2 * r4 + src_k3 * r6;
        const float xy2 = 2.0f * src_x_norm * src_y_norm;
        const float x_dist =
            src_x_norm * radial + src_p1 * xy2 + src_p2 * (r2 + 2.0f * src_x_norm * src_x_norm);
        const float y_dist =
            src_y_norm * radial + src_p1 * (r2 + 2.0f * src_y_norm * src_y_norm) + src_p2 * xy2;
        src_x_norm = x_dist;
        src_y_norm = y_dist;
    }
    const float src_x = fx_src * src_x_norm + cx_src;
    const float src_y = fy_src * src_y_norm + cy_src;

    if (!isfinite(src_x) || !isfinite(src_y)) {
        for (int c = 0; c < channels; ++c) {
            out[out_base + c] = static_cast<T>(0);
        }
        return;
    }

    const int x0 = static_cast<int>(floorf(src_x));
    const int y0 = static_cast<int>(floorf(src_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float dx_raw = src_x - static_cast<float>(x0);
    const float dy_raw = src_y - static_cast<float>(y0);
    // Match OpenCV INTER_LINEAR's fixed-point interpolation table.
    const float dx = nearbyintf(dx_raw * 32.0f) * (1.0f / 32.0f);
    const float dy = nearbyintf(dy_raw * 32.0f) * (1.0f / 32.0f);
    const float w00 = (1.0f - dx) * (1.0f - dy);
    const float w01 = dx * (1.0f - dy);
    const float w10 = (1.0f - dx) * dy;
    const float w11 = dx * dy;

    for (int c = 0; c < channels; ++c) {
        float accum = 0.0f;
        if (x0 >= 0 && x0 < src_width && y0 >= 0 && y0 < src_height) {
            accum += w00 * static_cast<float>(image[source_offset(y0, x0, src_width, channels, c)]);
        }
        if (x1 >= 0 && x1 < src_width && y0 >= 0 && y0 < src_height) {
            accum += w01 * static_cast<float>(image[source_offset(y0, x1, src_width, channels, c)]);
        }
        if (x0 >= 0 && x0 < src_width && y1 >= 0 && y1 < src_height) {
            accum += w10 * static_cast<float>(image[source_offset(y1, x0, src_width, channels, c)]);
        }
        if (x1 >= 0 && x1 < src_width && y1 >= 0 && y1 < src_height) {
            accum += w11 * static_cast<float>(image[source_offset(y1, x1, src_width, channels, c)]);
        }
        out[out_base + c] = cast_output<T>(accum);
    }
}

template <typename T>
void launch_camera_model_remap_fused_impl(
    const T* image_host, T* out_host, const int src_height, const int src_width, const int channels,
    const int out_height, const int out_width, const float fx_src, const float fy_src,
    const float cx_src, const float cy_src, const float fx_dst, const float fy_dst,
    const float cx_dst, const float cy_dst, const float* rotation_dst_to_src,
    const bool src_has_dist, const float* src_dist_coeffs, const bool dst_has_dist,
    const float* dst_dist_coeffs) {
    const size_t src_total = static_cast<size_t>(src_height) * static_cast<size_t>(src_width) *
                             static_cast<size_t>(channels);
    const size_t out_total = static_cast<size_t>(out_height) * static_cast<size_t>(out_width) *
                             static_cast<size_t>(channels);
    const size_t src_bytes = src_total * sizeof(T);
    const size_t out_bytes = out_total * sizeof(T);

    auto workspace = hnw::cuda::acquire_host_io_workspace("camera_model_remap cudaGetDevice");
    try {
        T* image_device = static_cast<T*>(
            workspace.device_buffer(src_bytes, "camera_model_remap cudaMalloc(image)"));
        T* out_device = static_cast<T*>(
            workspace.device_buffer(out_bytes, "camera_model_remap cudaMalloc(out)"));
        cudaStream_t stream = workspace.stream();
        void* pinned_image =
            workspace.pinned_buffer(src_bytes, "camera_model_remap cudaMallocHost(image)");
        void* pinned_out =
            workspace.pinned_buffer(out_bytes, "camera_model_remap cudaMallocHost(out)");
        std::memcpy(pinned_image, image_host, src_bytes);
        const T* image_copy_src = static_cast<const T*>(pinned_image);
        T* out_copy_dst = static_cast<T*>(pinned_out);

        hnw::cuda::throw_if_failed(cudaMemcpyAsync(image_device, image_copy_src, src_bytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "camera_model_remap cudaMemcpyAsync(image)");

        constexpr int threads_per_block = 256;
        const int total_pixels = out_height * out_width;
        const int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

        camera_model_remap_fused_kernel<<<blocks, threads_per_block, 0, stream>>>(
            image_device, out_device, src_height, src_width, channels, out_height, out_width,
            fx_src, fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src[0],
            rotation_dst_to_src[1], rotation_dst_to_src[2], rotation_dst_to_src[3],
            rotation_dst_to_src[4], rotation_dst_to_src[5], rotation_dst_to_src[6],
            rotation_dst_to_src[7], rotation_dst_to_src[8], src_has_dist, src_dist_coeffs[0],
            src_dist_coeffs[1], src_dist_coeffs[2], src_dist_coeffs[3], src_dist_coeffs[4],
            dst_has_dist, dst_dist_coeffs[0], dst_dist_coeffs[1], dst_dist_coeffs[2],
            dst_dist_coeffs[3], dst_dist_coeffs[4]);
        hnw::cuda::throw_if_failed(cudaGetLastError(), "camera_model_remap kernel launch");
        hnw::cuda::throw_if_failed(
            cudaMemcpyAsync(out_copy_dst, out_device, out_bytes, cudaMemcpyDeviceToHost, stream),
            "camera_model_remap cudaMemcpyAsync(out)");
        hnw::cuda::throw_if_failed(cudaStreamSynchronize(stream),
                                   "camera_model_remap cudaStreamSynchronize");
        std::memcpy(out_host, pinned_out, out_bytes);
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}

} // namespace

void launch_camera_model_remap_fused_u8(const unsigned char* image_host, unsigned char* out_host,
                                        int src_height, int src_width, int channels, int out_height,
                                        int out_width, float fx_src, float fy_src, float cx_src,
                                        float cy_src, float fx_dst, float fy_dst, float cx_dst,
                                        float cy_dst, const float* rotation_dst_to_src,
                                        bool src_has_dist, const float* src_dist_coeffs,
                                        bool dst_has_dist, const float* dst_dist_coeffs) {
    launch_camera_model_remap_fused_impl<unsigned char>(
        image_host, out_host, src_height, src_width, channels, out_height, out_width, fx_src,
        fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_has_dist,
        src_dist_coeffs, dst_has_dist, dst_dist_coeffs);
}

void launch_camera_model_remap_fused_u16(const unsigned short* image_host, unsigned short* out_host,
                                         int src_height, int src_width, int channels,
                                         int out_height, int out_width, float fx_src, float fy_src,
                                         float cx_src, float cy_src, float fx_dst, float fy_dst,
                                         float cx_dst, float cy_dst,
                                         const float* rotation_dst_to_src, bool src_has_dist,
                                         const float* src_dist_coeffs, bool dst_has_dist,
                                         const float* dst_dist_coeffs) {
    launch_camera_model_remap_fused_impl<unsigned short>(
        image_host, out_host, src_height, src_width, channels, out_height, out_width, fx_src,
        fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_has_dist,
        src_dist_coeffs, dst_has_dist, dst_dist_coeffs);
}

void launch_camera_model_remap_fused_f32(const float* image_host, float* out_host, int src_height,
                                         int src_width, int channels, int out_height, int out_width,
                                         float fx_src, float fy_src, float cx_src, float cy_src,
                                         float fx_dst, float fy_dst, float cx_dst, float cy_dst,
                                         const float* rotation_dst_to_src, bool src_has_dist,
                                         const float* src_dist_coeffs, bool dst_has_dist,
                                         const float* dst_dist_coeffs) {
    launch_camera_model_remap_fused_impl<float>(
        image_host, out_host, src_height, src_width, channels, out_height, out_width, fx_src,
        fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_has_dist,
        src_dist_coeffs, dst_has_dist, dst_dist_coeffs);
}
