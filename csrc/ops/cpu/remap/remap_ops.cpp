#include "remap_ops.h"

#include "common/camera_model_math.h"
#include "common/compat.h"
#include "common/cpu_compat.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

std::array<double, 5> parse_dist_coeffs(const py::object& dist_obj, const char* name,
                                        bool* has_dist) {
    std::array<double, 5> coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    *has_dist = false;
    if (dist_obj.is_none()) {
        return coeffs;
    }

    auto dist = dist_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (dist.ndim() != 1 || dist.shape(0) != 5) {
        throw std::invalid_argument(std::string("camera_model_remap_cpu: ") + name +
                                    " must be None or a 5-element projection coefficient array");
    }

    const double* ptr = dist.data();
    for (ssize_t idx = 0; idx < 5; ++idx) {
        if (!std::isfinite(ptr[idx])) {
            throw std::invalid_argument(std::string("camera_model_remap_cpu: ") + name +
                                        " must contain only finite values");
        }
        coeffs[static_cast<size_t>(idx)] = ptr[idx];
        *has_dist = *has_dist || ptr[idx] != 0.0;
    }
    return coeffs;
}

void validate_scalar_finite(const double value, const char* name, const bool non_zero = false) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string("camera_model_remap_cpu: ") + name +
                                    " must contain only finite values");
    }
    if (non_zero && value == 0.0) {
        throw std::invalid_argument(std::string("camera_model_remap_cpu: ") + name +
                                    " must be non-zero");
    }
}

void validate_rotation_finite(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rotation) {
    const double* ptr = rotation.data();
    for (ssize_t idx = 0; idx < 9; ++idx) {
        if (!std::isfinite(ptr[idx])) {
            throw std::invalid_argument("camera_model_remap_cpu: rotation_dst_to_src "
                                        "must contain only finite values");
        }
    }
}

void validate_int_bounds(const ssize_t value, const char* name) {
    if (value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("camera_model_remap_cpu: ") + name +
                                    " is too large");
    }
}

template <typename T> inline T cast_output(double value) {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<float>(value);
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        value = std::min(std::max(value, 0.0), 255.0);
        // Match the CUDA remap kernel's integer tie handling.
        return static_cast<unsigned char>(std::nearbyint(value));
    } else {
        value = std::min(std::max(value, 0.0), 65535.0);
        return static_cast<unsigned short>(std::nearbyint(value));
    }
}

inline ssize_t source_offset(const int y, const int x, const int src_width, const int channels,
                             const int c) {
    return (static_cast<ssize_t>(y) * static_cast<ssize_t>(src_width) + static_cast<ssize_t>(x)) *
               static_cast<ssize_t>(channels) +
           static_cast<ssize_t>(c);
}

template <typename T>
void camera_model_remap_cpu_kernel(
    const T* HNW_RESTRICT image, T* HNW_RESTRICT out, const int src_height, const int src_width,
    const int channels, const int out_height, const int out_width, const double fx_src,
    const double fy_src, const double cx_src, const double cy_src, const double fx_dst,
    const double fy_dst, const double cx_dst, const double cy_dst,
    const double* HNW_RESTRICT rotation_dst_to_src, const int src_projection,
    const bool src_has_dist, const double* HNW_RESTRICT src_dist_coeffs, const int dst_projection,
    const bool dst_has_dist, const double* HNW_RESTRICT dst_dist_coeffs) {
    const double r00 = rotation_dst_to_src[0];
    const double r01 = rotation_dst_to_src[1];
    const double r02 = rotation_dst_to_src[2];
    const double r10 = rotation_dst_to_src[3];
    const double r11 = rotation_dst_to_src[4];
    const double r12 = rotation_dst_to_src[5];
    const double r20 = rotation_dst_to_src[6];
    const double r21 = rotation_dst_to_src[7];
    const double r22 = rotation_dst_to_src[8];
    const double inv_fx_dst = 1.0 / fx_dst;
    const double inv_fy_dst = 1.0 / fy_dst;
    const ssize_t src_row_stride = static_cast<ssize_t>(src_width) * static_cast<ssize_t>(channels);
    const ssize_t out_row_stride = static_cast<ssize_t>(out_width) * static_cast<ssize_t>(channels);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < out_height; ++row) {
        const double yd = (static_cast<double>(row) - cy_dst) * inv_fy_dst;
        const ssize_t out_row_base = static_cast<ssize_t>(row) * out_row_stride;
        for (int col = 0; col < out_width; ++col) {
            const double xd = (static_cast<double>(col) - cx_dst) * inv_fx_dst;
            double ray_x = 0.0;
            double ray_y = 0.0;
            double ray_z = 0.0;
            hnw::camera::unproject(xd, yd, dst_projection, dst_has_dist, dst_dist_coeffs, &ray_x,
                                   &ray_y, &ray_z);

            const double proj_x = r00 * ray_x + r01 * ray_y + r02 * ray_z;
            const double proj_y = r10 * ray_x + r11 * ray_y + r12 * ray_z;
            const double proj_z = r20 * ray_x + r21 * ray_y + r22 * ray_z;
            const ssize_t out_base =
                out_row_base + static_cast<ssize_t>(col) * static_cast<ssize_t>(channels);
            double src_x_norm = 0.0;
            double src_y_norm = 0.0;
            if (!hnw::camera::project(proj_x, proj_y, proj_z, src_projection, src_has_dist,
                                      src_dist_coeffs, &src_x_norm, &src_y_norm)) {
                for (int c = 0; c < channels; ++c) {
                    out[out_base + c] = static_cast<T>(0);
                }
                continue;
            }

            const double src_x = fx_src * src_x_norm + cx_src;
            const double src_y = fy_src * src_y_norm + cy_src;
            if (!std::isfinite(src_x) || !std::isfinite(src_y)) {
                for (int c = 0; c < channels; ++c) {
                    out[out_base + c] = static_cast<T>(0);
                }
                continue;
            }

            // cv2.remap consumes float32 maps before indexing its 1/32 table.
            const float src_x_map = static_cast<float>(src_x);
            const float src_y_map = static_cast<float>(src_y);
            if (!(src_x_map > -1.0F && src_x_map < src_width && src_y_map > -1.0F &&
                  src_y_map < src_height)) {
                for (int c = 0; c < channels; ++c) {
                    out[out_base + c] = static_cast<T>(0);
                }
                continue;
            }
            const int x0 = static_cast<int>(std::floor(src_x_map));
            const int y0 = static_cast<int>(std::floor(src_y_map));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const double dx_raw = static_cast<double>(src_x_map) - x0;
            const double dy_raw = static_cast<double>(src_y_map) - y0;
            const double dx = std::nearbyint(dx_raw * 32.0) * (1.0 / 32.0);
            const double dy = std::nearbyint(dy_raw * 32.0) * (1.0 / 32.0);
            const double w00 = (1.0 - dx) * (1.0 - dy);
            const double w01 = dx * (1.0 - dy);
            const double w10 = (1.0 - dx) * dy;
            const double w11 = dx * dy;

            if (x0 >= 0 && x1 < src_width && y0 >= 0 && y1 < src_height) {
                const ssize_t base00 = static_cast<ssize_t>(y0) * src_row_stride +
                                       static_cast<ssize_t>(x0) * static_cast<ssize_t>(channels);
                const ssize_t base01 = base00 + static_cast<ssize_t>(channels);
                const ssize_t base10 = base00 + src_row_stride;
                const ssize_t base11 = base10 + static_cast<ssize_t>(channels);
                for (int c = 0; c < channels; ++c) {
                    const double accum = w00 * static_cast<double>(image[base00 + c]) +
                                         w01 * static_cast<double>(image[base01 + c]) +
                                         w10 * static_cast<double>(image[base10 + c]) +
                                         w11 * static_cast<double>(image[base11 + c]);
                    out[out_base + c] = cast_output<T>(accum);
                }
                continue;
            }

            for (int c = 0; c < channels; ++c) {
                double accum = 0.0;
                if (x0 >= 0 && x0 < src_width && y0 >= 0 && y0 < src_height) {
                    accum += w00 * static_cast<double>(
                                       image[source_offset(y0, x0, src_width, channels, c)]);
                }
                if (x1 >= 0 && x1 < src_width && y0 >= 0 && y0 < src_height) {
                    accum += w01 * static_cast<double>(
                                       image[source_offset(y0, x1, src_width, channels, c)]);
                }
                if (x0 >= 0 && x0 < src_width && y1 >= 0 && y1 < src_height) {
                    accum += w10 * static_cast<double>(
                                       image[source_offset(y1, x0, src_width, channels, c)]);
                }
                if (x1 >= 0 && x1 < src_width && y1 >= 0 && y1 < src_height) {
                    accum += w11 * static_cast<double>(
                                       image[source_offset(y1, x1, src_width, channels, c)]);
                }
                out[out_base + c] = cast_output<T>(accum);
            }
        }
    }
}

template <typename T>
py::array_t<T> camera_model_remap_cpu_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
    const ssize_t out_height, const ssize_t out_width, const double fx_src, const double fy_src,
    const double cx_src, const double cy_src, const double fx_dst, const double fy_dst,
    const double cx_dst, const double cy_dst,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rotation_dst_to_src,
    const py::object& src_dist_coeffs_obj, const py::object& dst_dist_coeffs_obj,
    const int src_projection, const int dst_projection) {
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
        throw std::invalid_argument("camera_model_remap_cpu: image channels must be positive");
    }
    validate_int_bounds(out_height, "out_height");
    validate_int_bounds(out_width, "out_width");
    validate_int_bounds(image.shape(0), "image height");
    validate_int_bounds(image.shape(1), "image width");
    if (image.ndim() == 3) {
        validate_int_bounds(image.shape(2), "image channels");
    }
    if (out_height > std::numeric_limits<int>::max() / out_width) {
        throw std::invalid_argument("camera_model_remap_cpu: output image is too large");
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
    if (!hnw::camera::valid_projection(src_projection) ||
        !hnw::camera::valid_projection(dst_projection)) {
        throw std::invalid_argument("camera_model_remap_cpu: projection must be 0 "
                                    "(perspective) or 1 (fisheye)");
    }

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
            static_cast<const T*>(image_info.ptr), static_cast<T*>(out_info.ptr), src_height,
            src_width, channels, static_cast<int>(out_height), static_cast<int>(out_width), fx_src,
            fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst,
            static_cast<const double*>(rotation_info.ptr), src_projection, src_has_dist,
            src_dist_coeffs.data(), dst_projection, dst_has_dist, dst_dist_coeffs.data());
    }
    return out;
}

py::array camera_model_remap_cpu_dispatch(
    const py::array& image, const ssize_t out_height, const ssize_t out_width, const double fx_src,
    const double fy_src, const double cx_src, const double cy_src, const double fx_dst,
    const double fy_dst, const double cx_dst, const double cy_dst,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rotation_dst_to_src,
    const py::object& src_dist_coeffs, const py::object& dst_dist_coeffs, const int src_projection,
    const int dst_projection) {
    if (py::isinstance<py::array_t<unsigned char>>(image)) {
        return camera_model_remap_cpu_impl<unsigned char>(
            image.cast<py::array_t<unsigned char>>(), out_height, out_width, fx_src, fy_src, cx_src,
            cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_dist_coeffs,
            dst_dist_coeffs, src_projection, dst_projection);
    }
    if (py::isinstance<py::array_t<unsigned short>>(image)) {
        return camera_model_remap_cpu_impl<unsigned short>(
            image.cast<py::array_t<unsigned short>>(), out_height, out_width, fx_src, fy_src,
            cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_dist_coeffs,
            dst_dist_coeffs, src_projection, dst_projection);
    }
    if (py::isinstance<py::array_t<float>>(image)) {
        return camera_model_remap_cpu_impl<float>(
            image.cast<py::array_t<float>>(), out_height, out_width, fx_src, fy_src, cx_src, cy_src,
            fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_dist_coeffs, dst_dist_coeffs,
            src_projection, dst_projection);
    }
    throw std::invalid_argument("camera_model_remap_cpu: unsupported image "
                                "dtype; expected uint8/uint16/float32");
}

} // namespace

void bind_remap_ops(py::module_& m) {
    m.def("camera_model_remap_cpu", &camera_model_remap_cpu_dispatch, py::arg("image"),
          py::arg("out_height"), py::arg("out_width"), py::arg("fx_src"), py::arg("fy_src"),
          py::arg("cx_src"), py::arg("cy_src"), py::arg("fx_dst"), py::arg("fy_dst"),
          py::arg("cx_dst"), py::arg("cy_dst"), py::arg("rotation_dst_to_src"),
          py::arg("src_dist_coeffs") = py::none(), py::arg("dst_dist_coeffs") = py::none(),
          py::arg("src_projection") = hnw::camera::PROJECTION_PERSPECTIVE,
          py::arg("dst_projection") = hnw::camera::PROJECTION_PERSPECTIVE,
          "Apply a camera-model remap with fused grid generation and bilinear "
          "sampling using OpenMP.");
}
