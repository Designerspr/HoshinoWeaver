#include "remap_ops.h"

#include "common/compat.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <pybind11/numpy.h>

namespace {

#if defined(_MSC_VER)
#define HNW_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define HNW_RESTRICT __restrict__
#else
#define HNW_RESTRICT
#endif

std::array<float, 5> parse_dist_coeffs(const py::object& dist_obj,
                                       const char* name,
                                       bool* has_dist) {
    std::array<float, 5> coeffs = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    *has_dist = false;
    if (dist_obj.is_none()) {
        return coeffs;
    }

    auto dist = dist_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    if (dist.ndim() != 1 || dist.shape(0) != 5) {
        throw std::invalid_argument(
            std::string("camera_model_remap_cpu: ") + name +
            " must be None or a 5-element array [k1, k2, p1, p2, k3]");
    }

    const float* ptr = dist.data();
    for (ssize_t idx = 0; idx < 5; ++idx) {
        if (!std::isfinite(ptr[idx])) {
            throw std::invalid_argument(
                std::string("camera_model_remap_cpu: ") + name +
                " must contain only finite values");
        }
        coeffs[static_cast<size_t>(idx)] = ptr[idx];
        *has_dist = *has_dist || ptr[idx] != 0.0f;
    }
    return coeffs;
}

void validate_scalar_finite(const float value,
                            const char* name,
                            const bool non_zero = false) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::string("camera_model_remap_cpu: ") + name +
            " must contain only finite values");
    }
    if (non_zero && value == 0.0f) {
        throw std::invalid_argument(
            std::string("camera_model_remap_cpu: ") + name + " must be non-zero");
    }
}

void validate_rotation_finite(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& rotation) {
    const float* ptr = rotation.data();
    for (ssize_t idx = 0; idx < 9; ++idx) {
        if (!std::isfinite(ptr[idx])) {
            throw std::invalid_argument(
                "camera_model_remap_cpu: rotation_dst_to_src must contain only finite values");
        }
    }
}

void validate_int_bounds(const ssize_t value, const char* name) {
    if (value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("camera_model_remap_cpu: ") + name + " is too large");
    }
}

template <typename T>
inline T cast_output(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        value = std::min(std::max(value, 0.0f), 255.0f);
        return static_cast<unsigned char>(std::floor(value + 0.5f));
    } else {
        value = std::min(std::max(value, 0.0f), 65535.0f);
        return static_cast<unsigned short>(std::nearbyint(value));
    }
}

inline ssize_t source_offset(const int y,
                             const int x,
                             const int src_width,
                             const int channels,
                             const int c) {
    return (static_cast<ssize_t>(y) * static_cast<ssize_t>(src_width) +
            static_cast<ssize_t>(x)) *
               static_cast<ssize_t>(channels) +
           static_cast<ssize_t>(c);
}

template <typename T>
void camera_model_remap_cpu_kernel(
    const T* HNW_RESTRICT image,
    T* HNW_RESTRICT out,
    const int src_height,
    const int src_width,
    const int channels,
    const int out_height,
    const int out_width,
    const float fx_src,
    const float fy_src,
    const float cx_src,
    const float cy_src,
    const float fx_dst,
    const float fy_dst,
    const float cx_dst,
    const float cy_dst,
    const float* HNW_RESTRICT rotation_dst_to_src,
    const bool src_has_dist,
    const float* HNW_RESTRICT src_dist_coeffs,
    const bool dst_has_dist,
    const float* HNW_RESTRICT dst_dist_coeffs) {
    const ssize_t total =
        static_cast<ssize_t>(out_height) * static_cast<ssize_t>(out_width);
    const float r00 = rotation_dst_to_src[0];
    const float r01 = rotation_dst_to_src[1];
    const float r02 = rotation_dst_to_src[2];
    const float r10 = rotation_dst_to_src[3];
    const float r11 = rotation_dst_to_src[4];
    const float r12 = rotation_dst_to_src[5];
    const float r20 = rotation_dst_to_src[6];
    const float r21 = rotation_dst_to_src[7];
    const float r22 = rotation_dst_to_src[8];
    const float src_k1 = src_dist_coeffs[0];
    const float src_k2 = src_dist_coeffs[1];
    const float src_p1 = src_dist_coeffs[2];
    const float src_p2 = src_dist_coeffs[3];
    const float src_k3 = src_dist_coeffs[4];
    const float dst_k1 = dst_dist_coeffs[0];
    const float dst_k2 = dst_dist_coeffs[1];
    const float dst_p1 = dst_dist_coeffs[2];
    const float dst_p2 = dst_dist_coeffs[3];
    const float dst_k3 = dst_dist_coeffs[4];

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t idx = 0; idx < total; ++idx) {
        const int row = static_cast<int>(idx / out_width);
        const int col = static_cast<int>(idx - static_cast<ssize_t>(row) * out_width);
        const float xd = (static_cast<float>(col) - cx_dst) / fx_dst;
        const float yd = (static_cast<float>(row) - cy_dst) / fy_dst;
        float x = xd;
        float y = yd;
        if (dst_has_dist) {
            for (int iter = 0; iter < 5; ++iter) {
                const float r2 = x * x + y * y;
                const float r4 = r2 * r2;
                const float r6 = r4 * r2;
                const float radial = 1.0f + dst_k1 * r2 + dst_k2 * r4 + dst_k3 * r6;
                const float xy2 = 2.0f * x * y;
                const float delta_x =
                    dst_p1 * xy2 + dst_p2 * (r2 + 2.0f * x * x);
                const float delta_y =
                    dst_p1 * (r2 + 2.0f * y * y) + dst_p2 * xy2;
                x = (xd - delta_x) / radial;
                y = (yd - delta_y) / radial;
            }
        }

        const float proj_x = r00 * x + r01 * y + r02;
        const float proj_y = r10 * x + r11 * y + r12;
        const float proj_z = r20 * x + r21 * y + r22;
        const ssize_t out_base = idx * static_cast<ssize_t>(channels);
        if (!(proj_z > 0.0f)) {
            for (int c = 0; c < channels; ++c) {
                out[out_base + c] = static_cast<T>(0);
            }
            continue;
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
            const float x_dist = src_x_norm * radial +
                                 src_p1 * xy2 +
                                 src_p2 * (r2 + 2.0f * src_x_norm * src_x_norm);
            const float y_dist = src_y_norm * radial +
                                 src_p1 * (r2 + 2.0f * src_y_norm * src_y_norm) +
                                 src_p2 * xy2;
            src_x_norm = x_dist;
            src_y_norm = y_dist;
        }

        const float src_x = fx_src * src_x_norm + cx_src;
        const float src_y = fy_src * src_y_norm + cy_src;
        if (!std::isfinite(src_x) || !std::isfinite(src_y)) {
            for (int c = 0; c < channels; ++c) {
                out[out_base + c] = static_cast<T>(0);
            }
            continue;
        }

        const int x0 = static_cast<int>(std::floor(src_x));
        const int y0 = static_cast<int>(std::floor(src_y));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const float dx_raw = src_x - static_cast<float>(x0);
        const float dy_raw = src_y - static_cast<float>(y0);
        const float dx = std::nearbyint(dx_raw * 32.0f) * (1.0f / 32.0f);
        const float dy = std::nearbyint(dy_raw * 32.0f) * (1.0f / 32.0f);
        const float w00 = (1.0f - dx) * (1.0f - dy);
        const float w01 = dx * (1.0f - dy);
        const float w10 = (1.0f - dx) * dy;
        const float w11 = dx * dy;

        for (int c = 0; c < channels; ++c) {
            float accum = 0.0f;
            if (x0 >= 0 && x0 < src_width && y0 >= 0 && y0 < src_height) {
                accum += w00 * static_cast<float>(
                    image[source_offset(y0, x0, src_width, channels, c)]);
            }
            if (x1 >= 0 && x1 < src_width && y0 >= 0 && y0 < src_height) {
                accum += w01 * static_cast<float>(
                    image[source_offset(y0, x1, src_width, channels, c)]);
            }
            if (x0 >= 0 && x0 < src_width && y1 >= 0 && y1 < src_height) {
                accum += w10 * static_cast<float>(
                    image[source_offset(y1, x0, src_width, channels, c)]);
            }
            if (x1 >= 0 && x1 < src_width && y1 >= 0 && y1 < src_height) {
                accum += w11 * static_cast<float>(
                    image[source_offset(y1, x1, src_width, channels, c)]);
            }
            out[out_base + c] = cast_output<T>(accum);
        }
    }
}

template <typename T>
py::array_t<T> camera_model_remap_cpu_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
    const ssize_t out_height,
    const ssize_t out_width,
    const float fx_src,
    const float fy_src,
    const float cx_src,
    const float cy_src,
    const float fx_dst,
    const float fy_dst,
    const float cx_dst,
    const float cy_dst,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        rotation_dst_to_src,
    const py::object& src_dist_coeffs_obj,
    const py::object& dst_dist_coeffs_obj) {
    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument(
            "camera_model_remap_cpu: output height and width must be positive");
    }
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "camera_model_remap_cpu: image must have shape (H, W) or (H, W, C)");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument(
            "camera_model_remap_cpu: image height and width must be positive");
    }
    if (image.ndim() == 3 && image.shape(2) <= 0) {
        throw std::invalid_argument(
            "camera_model_remap_cpu: image channels must be positive");
    }
    validate_int_bounds(out_height, "out_height");
    validate_int_bounds(out_width, "out_width");
    validate_int_bounds(image.shape(0), "image height");
    validate_int_bounds(image.shape(1), "image width");
    if (image.ndim() == 3) {
        validate_int_bounds(image.shape(2), "image channels");
    }
    if (out_height > std::numeric_limits<int>::max() / out_width) {
        throw std::invalid_argument(
            "camera_model_remap_cpu: output image is too large");
    }
    if (rotation_dst_to_src.ndim() != 2 || rotation_dst_to_src.shape(0) != 3 ||
        rotation_dst_to_src.shape(1) != 3) {
        throw std::invalid_argument(
            "camera_model_remap_cpu: rotation_dst_to_src must have shape (3, 3)");
    }
    validate_scalar_finite(fx_src, "fx_src", true);
    validate_scalar_finite(fy_src, "fy_src", true);
    validate_scalar_finite(cx_src, "cx_src");
    validate_scalar_finite(cy_src, "cy_src");
    validate_scalar_finite(fx_dst, "fx_dst", true);
    validate_scalar_finite(fy_dst, "fy_dst", true);
    validate_scalar_finite(cx_dst, "cx_dst");
    validate_scalar_finite(cy_dst, "cy_dst");
    validate_rotation_finite(rotation_dst_to_src);

    auto image_info = image.request();
    auto rotation_info = rotation_dst_to_src.request();
    bool src_has_dist = false;
    bool dst_has_dist = false;
    const auto src_dist_coeffs =
        parse_dist_coeffs(src_dist_coeffs_obj, "src_dist_coeffs", &src_has_dist);
    const auto dst_dist_coeffs =
        parse_dist_coeffs(dst_dist_coeffs_obj, "dst_dist_coeffs", &dst_has_dist);
    const int src_height = static_cast<int>(image.shape(0));
    const int src_width = static_cast<int>(image.shape(1));
    const int channels = image.ndim() == 3 ? static_cast<int>(image.shape(2)) : 1;

    std::vector<py::ssize_t> out_shape = {out_height, out_width};
    if (image.ndim() == 3) {
        out_shape.push_back(static_cast<py::ssize_t>(channels));
    }
    py::array_t<T> out(out_shape);
    auto out_info = out.request();

    {
        py::gil_scoped_release release;
        camera_model_remap_cpu_kernel(
            static_cast<const T*>(image_info.ptr),
            static_cast<T*>(out_info.ptr),
            src_height,
            src_width,
            channels,
            static_cast<int>(out_height),
            static_cast<int>(out_width),
            fx_src,
            fy_src,
            cx_src,
            cy_src,
            fx_dst,
            fy_dst,
            cx_dst,
            cy_dst,
            static_cast<const float*>(rotation_info.ptr),
            src_has_dist,
            src_dist_coeffs.data(),
            dst_has_dist,
            dst_dist_coeffs.data());
    }
    return out;
}

py::array camera_model_remap_cpu_dispatch(
    const py::array& image,
    const ssize_t out_height,
    const ssize_t out_width,
    const float fx_src,
    const float fy_src,
    const float cx_src,
    const float cy_src,
    const float fx_dst,
    const float fy_dst,
    const float cx_dst,
    const float cy_dst,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        rotation_dst_to_src,
    const py::object& src_dist_coeffs,
    const py::object& dst_dist_coeffs) {
    if (py::isinstance<py::array_t<unsigned char>>(image)) {
        return camera_model_remap_cpu_impl<unsigned char>(
            image.cast<py::array_t<unsigned char>>(),
            out_height,
            out_width,
            fx_src,
            fy_src,
            cx_src,
            cy_src,
            fx_dst,
            fy_dst,
            cx_dst,
            cy_dst,
            rotation_dst_to_src,
            src_dist_coeffs,
            dst_dist_coeffs);
    }
    if (py::isinstance<py::array_t<unsigned short>>(image)) {
        return camera_model_remap_cpu_impl<unsigned short>(
            image.cast<py::array_t<unsigned short>>(),
            out_height,
            out_width,
            fx_src,
            fy_src,
            cx_src,
            cy_src,
            fx_dst,
            fy_dst,
            cx_dst,
            cy_dst,
            rotation_dst_to_src,
            src_dist_coeffs,
            dst_dist_coeffs);
    }
    if (py::isinstance<py::array_t<float>>(image)) {
        return camera_model_remap_cpu_impl<float>(
            image.cast<py::array_t<float>>(),
            out_height,
            out_width,
            fx_src,
            fy_src,
            cx_src,
            cy_src,
            fx_dst,
            fy_dst,
            cx_dst,
            cy_dst,
            rotation_dst_to_src,
            src_dist_coeffs,
            dst_dist_coeffs);
    }
    throw std::invalid_argument(
        "camera_model_remap_cpu: unsupported image dtype; expected uint8/uint16/float32");
}

}  // namespace

void bind_remap_ops(py::module_& m) {
    m.def("camera_model_remap_cpu",
          &camera_model_remap_cpu_dispatch,
          py::arg("image"),
          py::arg("out_height"),
          py::arg("out_width"),
          py::arg("fx_src"),
          py::arg("fy_src"),
          py::arg("cx_src"),
          py::arg("cy_src"),
          py::arg("fx_dst"),
          py::arg("fy_dst"),
          py::arg("cx_dst"),
          py::arg("cy_dst"),
          py::arg("rotation_dst_to_src"),
          py::arg("src_dist_coeffs") = py::none(),
          py::arg("dst_dist_coeffs") = py::none(),
          "Apply a camera-model remap with fused grid generation and bilinear sampling using OpenMP.");
}
