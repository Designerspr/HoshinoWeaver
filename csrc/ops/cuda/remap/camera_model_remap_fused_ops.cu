#include "common/camera_model_math.h"
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
        value = fminf(fmaxf(value, 0.0F), 255.0F);
        return static_cast<unsigned char>(nearbyintf(value));
    } else {
        value = fminf(fmaxf(value, 0.0F), 65535.0F);
        return static_cast<unsigned short>(nearbyintf(value));
    }
}

template <typename T>
__device__ inline T interpolate_opencv5(const float p00, const float p01, const float p10,
                                        const float p11, const float dx, const float dy,
                                        const bool exact_linear) {
    if (exact_linear) {
        return cast_output<T>(hnw::camera::bilinear_interpolate_exact(p00, p01, p10, p11, dx, dy));
    }

    const int frac_x =
        static_cast<int>(nearbyintf(dx * static_cast<float>(hnw::camera::REMAP_LINEAR_TABLE_SIZE)));
    const int frac_y =
        static_cast<int>(nearbyintf(dy * static_cast<float>(hnw::camera::REMAP_LINEAR_TABLE_SIZE)));
    if constexpr (std::is_same_v<T, unsigned char>) {
        const int wx0 = hnw::camera::REMAP_LINEAR_TABLE_SIZE - frac_x;
        const int wy0 = hnw::camera::REMAP_LINEAR_TABLE_SIZE - frac_y;
        const int accum =
            (static_cast<int>(p00) * wx0 * wy0 + static_cast<int>(p01) * frac_x * wy0 +
             static_cast<int>(p10) * wx0 * frac_y + static_cast<int>(p11) * frac_x * frac_y) *
            hnw::camera::REMAP_LINEAR_TABLE_WEIGHT_SCALE;
        constexpr int rounding_delta = 1 << (hnw::camera::REMAP_LINEAR_COEF_BITS - 1);
        return static_cast<unsigned char>((accum + rounding_delta) >>
                                          hnw::camera::REMAP_LINEAR_COEF_BITS);
    }
    return cast_output<T>(
        hnw::camera::bilinear_interpolate_table(p00, p01, p10, p11, frac_x, frac_y));
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
    const int out_height, const int out_width, const double fx_src, const double fy_src,
    const double cx_src, const double cy_src, const double fx_dst, const double fy_dst,
    const double cx_dst, const double cy_dst, const double r00, const double r01, const double r02,
    const double r10, const double r11, const double r12, const double r20, const double r21,
    const double r22, const int src_projection, const bool src_has_dist, const double src_k1,
    const double src_k2, const double src_p1, const double src_p2, const double src_k3,
    const int dst_projection, const bool dst_has_dist, const double dst_k1, const double dst_k2,
    const double dst_p1, const double dst_p2, const double dst_k3) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_height * out_width;
    if (idx >= total) {
        return;
    }

    const int row = idx / out_width;
    const int col = idx - row * out_width;
    const double xd = (static_cast<double>(col) - cx_dst) / fx_dst;
    const double yd = (static_cast<double>(row) - cy_dst) / fy_dst;
    const double src_dist_coeffs[5] = {src_k1, src_k2, src_p1, src_p2, src_k3};
    const double dst_dist_coeffs[5] = {dst_k1, dst_k2, dst_p1, dst_p2, dst_k3};
    double ray_x = 0.0;
    double ray_y = 0.0;
    double ray_z = 0.0;
    hnw::camera::unproject(xd, yd, dst_projection, dst_has_dist, dst_dist_coeffs, &ray_x, &ray_y,
                           &ray_z);

    const double proj_x = r00 * ray_x + r01 * ray_y + r02 * ray_z;
    const double proj_y = r10 * ray_x + r11 * ray_y + r12 * ray_z;
    const double proj_z = r20 * ray_x + r21 * ray_y + r22 * ray_z;
    const size_t out_base = static_cast<size_t>(idx) * static_cast<size_t>(channels);

    double src_x_norm = 0.0;
    double src_y_norm = 0.0;
    if (!hnw::camera::project(proj_x, proj_y, proj_z, src_projection, src_has_dist, src_dist_coeffs,
                              &src_x_norm, &src_y_norm)) {
        for (int c = 0; c < channels; ++c) {
            out[out_base + c] = static_cast<T>(0);
        }
        return;
    }
    const double src_x = fx_src * src_x_norm + cx_src;
    const double src_y = fy_src * src_y_norm + cy_src;

    if (!isfinite(src_x) || !isfinite(src_y)) {
        for (int c = 0; c < channels; ++c) {
            out[out_base + c] = static_cast<T>(0);
        }
        return;
    }

    // OpenCV 5 uses exact float interpolation for C1/C3/C4. Other channel counts retain the
    // legacy 1/32 lookup-table path.
    const float src_x_map = static_cast<float>(src_x);
    const float src_y_map = static_cast<float>(src_y);
    if (!(src_x_map > -1.0F && src_x_map < src_width && src_y_map > -1.0F &&
          src_y_map < src_height)) {
        for (int c = 0; c < channels; ++c) {
            out[out_base + c] = static_cast<T>(0);
        }
        return;
    }
    const int x0 = static_cast<int>(floorf(src_x_map));
    const int y0 = static_cast<int>(floorf(src_y_map));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float dx = src_x_map - static_cast<float>(x0);
    const float dy = src_y_map - static_cast<float>(y0);

    const bool exact_linear = hnw::camera::remap_uses_exact_linear(channels);
    for (int c = 0; c < channels; ++c) {
        float p00 = 0.0F;
        float p01 = 0.0F;
        float p10 = 0.0F;
        float p11 = 0.0F;
        if (x0 >= 0 && x0 < src_width && y0 >= 0 && y0 < src_height) {
            p00 = static_cast<float>(image[source_offset(y0, x0, src_width, channels, c)]);
        }
        if (x1 >= 0 && x1 < src_width && y0 >= 0 && y0 < src_height) {
            p01 = static_cast<float>(image[source_offset(y0, x1, src_width, channels, c)]);
        }
        if (x0 >= 0 && x0 < src_width && y1 >= 0 && y1 < src_height) {
            p10 = static_cast<float>(image[source_offset(y1, x0, src_width, channels, c)]);
        }
        if (x1 >= 0 && x1 < src_width && y1 >= 0 && y1 < src_height) {
            p11 = static_cast<float>(image[source_offset(y1, x1, src_width, channels, c)]);
        }
        out[out_base + c] = interpolate_opencv5<T>(p00, p01, p10, p11, dx, dy, exact_linear);
    }
}

template <typename T>
void launch_camera_model_remap_fused_impl(
    const T* image_host, T* out_host, const int src_height, const int src_width, const int channels,
    const int out_height, const int out_width, const double fx_src, const double fy_src,
    const double cx_src, const double cy_src, const double fx_dst, const double fy_dst,
    const double cx_dst, const double cy_dst, const double* rotation_dst_to_src,
    const bool src_has_dist, const double* src_dist_coeffs, const bool dst_has_dist,
    const double* dst_dist_coeffs, const int src_projection, const int dst_projection) {
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
            rotation_dst_to_src[7], rotation_dst_to_src[8], src_projection, src_has_dist,
            src_dist_coeffs[0], src_dist_coeffs[1], src_dist_coeffs[2], src_dist_coeffs[3],
            src_dist_coeffs[4], dst_projection, dst_has_dist, dst_dist_coeffs[0],
            dst_dist_coeffs[1], dst_dist_coeffs[2], dst_dist_coeffs[3], dst_dist_coeffs[4]);
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
                                        int out_width, double fx_src, double fy_src, double cx_src,
                                        double cy_src, double fx_dst, double fy_dst, double cx_dst,
                                        double cy_dst, const double* rotation_dst_to_src,
                                        bool src_has_dist, const double* src_dist_coeffs,
                                        bool dst_has_dist, const double* dst_dist_coeffs,
                                        int src_projection, int dst_projection) {
    launch_camera_model_remap_fused_impl<unsigned char>(
        image_host, out_host, src_height, src_width, channels, out_height, out_width, fx_src,
        fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_has_dist,
        src_dist_coeffs, dst_has_dist, dst_dist_coeffs, src_projection, dst_projection);
}

void launch_camera_model_remap_fused_u16(
    const unsigned short* image_host, unsigned short* out_host, int src_height, int src_width,
    int channels, int out_height, int out_width, double fx_src, double fy_src, double cx_src,
    double cy_src, double fx_dst, double fy_dst, double cx_dst, double cy_dst,
    const double* rotation_dst_to_src, bool src_has_dist, const double* src_dist_coeffs,
    bool dst_has_dist, const double* dst_dist_coeffs, int src_projection, int dst_projection) {
    launch_camera_model_remap_fused_impl<unsigned short>(
        image_host, out_host, src_height, src_width, channels, out_height, out_width, fx_src,
        fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_has_dist,
        src_dist_coeffs, dst_has_dist, dst_dist_coeffs, src_projection, dst_projection);
}

void launch_camera_model_remap_fused_f32(const float* image_host, float* out_host, int src_height,
                                         int src_width, int channels, int out_height, int out_width,
                                         double fx_src, double fy_src, double cx_src, double cy_src,
                                         double fx_dst, double fy_dst, double cx_dst, double cy_dst,
                                         const double* rotation_dst_to_src, bool src_has_dist,
                                         const double* src_dist_coeffs, bool dst_has_dist,
                                         const double* dst_dist_coeffs, int src_projection,
                                         int dst_projection) {
    launch_camera_model_remap_fused_impl<float>(
        image_host, out_host, src_height, src_width, channels, out_height, out_width, fx_src,
        fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_has_dist,
        src_dist_coeffs, dst_has_dist, dst_dist_coeffs, src_projection, dst_projection);
}
