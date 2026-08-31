#include "camera_model_remap_fused_ops.h"

#include "common/camera_model_math.h"
#include "common/compat.h"

#include <pybind11/numpy.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

void launch_camera_model_remap_fused_u8(const unsigned char* image_host, unsigned char* out_host,
                                        int src_height, int src_width, int channels, int out_height,
                                        int out_width, double fx_src, double fy_src, double cx_src,
                                        double cy_src, double fx_dst, double fy_dst, double cx_dst,
                                        double cy_dst, const double* rotation_dst_to_src,
                                        bool src_has_dist, const double* src_dist_coeffs,
                                        bool dst_has_dist, const double* dst_dist_coeffs,
                                        int src_projection, int dst_projection);

void launch_camera_model_remap_fused_u16(
    const unsigned short* image_host, unsigned short* out_host, int src_height, int src_width,
    int channels, int out_height, int out_width, double fx_src, double fy_src, double cx_src,
    double cy_src, double fx_dst, double fy_dst, double cx_dst, double cy_dst,
    const double* rotation_dst_to_src, bool src_has_dist, const double* src_dist_coeffs,
    bool dst_has_dist, const double* dst_dist_coeffs, int src_projection, int dst_projection);

void launch_camera_model_remap_fused_f32(const float* image_host, float* out_host, int src_height,
                                         int src_width, int channels, int out_height, int out_width,
                                         double fx_src, double fy_src, double cx_src, double cy_src,
                                         double fx_dst, double fy_dst, double cx_dst, double cy_dst,
                                         const double* rotation_dst_to_src, bool src_has_dist,
                                         const double* src_dist_coeffs, bool dst_has_dist,
                                         const double* dst_dist_coeffs, int src_projection,
                                         int dst_projection);

namespace {

template <typename T>
using launch_fn_t = void (*)(const T* image_host, T* out_host, int src_height, int src_width,
                             int channels, int out_height, int out_width, double fx_src,
                             double fy_src, double cx_src, double cy_src, double fx_dst,
                             double fy_dst, double cx_dst, double cy_dst,
                             const double* rotation_dst_to_src, bool src_has_dist,
                             const double* src_dist_coeffs, bool dst_has_dist,
                             const double* dst_dist_coeffs, int src_projection, int dst_projection);

std::array<double, 5> parse_dist_coeffs(const py::object& dist_obj, const char* name,
                                        bool* has_dist) {
    std::array<double, 5> coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    *has_dist = false;
    if (dist_obj.is_none()) {
        return coeffs;
    }

    auto dist = dist_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (dist.ndim() != 1 || dist.shape(0) != 5) {
        throw std::invalid_argument(std::string("camera_model_remap: ") + name +
                                    " must be None or a 5-element projection coefficient array");
    }

    const double* ptr = dist.data();
    for (ssize_t idx = 0; idx < 5; ++idx) {
        if (!std::isfinite(ptr[idx])) {
            throw std::invalid_argument(std::string("camera_model_remap: ") + name +
                                        " must contain only finite values");
        }
        coeffs[static_cast<size_t>(idx)] = ptr[idx];
        *has_dist = *has_dist || ptr[idx] != 0.0;
    }
    return coeffs;
}

void validate_scalar_finite(const double value, const char* op_name, const char* name,
                            const bool non_zero = false) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(op_name) + ": " + name +
                                    " must contain only finite values");
    }
    if (non_zero && value == 0.0) {
        throw std::invalid_argument(std::string(op_name) + ": " + name + " must be non-zero");
    }
}

void validate_rotation_finite(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rotation,
    const char* op_name) {
    const double* ptr = rotation.data();
    for (ssize_t idx = 0; idx < 9; ++idx) {
        if (!std::isfinite(ptr[idx])) {
            throw std::invalid_argument(std::string(op_name) +
                                        ": rotation_dst_to_src must contain only finite values");
        }
    }
}

void validate_int_bounds(const ssize_t value, const char* op_name, const char* name) {
    if (value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(op_name) + ": " + name + " is too large");
    }
}

template <typename T>
py::array_t<T> camera_model_remap_fused_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
    const ssize_t out_height, const ssize_t out_width, const double fx_src, const double fy_src,
    const double cx_src, const double cy_src, const double fx_dst, const double fy_dst,
    const double cx_dst, const double cy_dst,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rotation_dst_to_src,
    const py::object& src_dist_coeffs_obj, const py::object& dst_dist_coeffs_obj,
    const int src_projection, const int dst_projection, launch_fn_t<T> launcher) {
    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("camera_model_remap: output height and width must be positive");
    }
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "camera_model_remap: image must have shape (H, W) or (H, W, C)");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument("camera_model_remap: image height and width must be positive");
    }
    if (image.ndim() == 3 && image.shape(2) <= 0) {
        throw std::invalid_argument("camera_model_remap: image channels must be positive");
    }
    validate_int_bounds(out_height, "camera_model_remap", "out_height");
    validate_int_bounds(out_width, "camera_model_remap", "out_width");
    validate_int_bounds(image.shape(0), "camera_model_remap", "image height");
    validate_int_bounds(image.shape(1), "camera_model_remap", "image width");
    if (image.ndim() == 3) {
        validate_int_bounds(image.shape(2), "camera_model_remap", "image channels");
    }
    if (out_height > std::numeric_limits<int>::max() / out_width) {
        throw std::invalid_argument("camera_model_remap: output image is too large");
    }
    if (rotation_dst_to_src.ndim() != 2 || rotation_dst_to_src.shape(0) != 3 ||
        rotation_dst_to_src.shape(1) != 3) {
        throw std::invalid_argument(
            "camera_model_remap: rotation_dst_to_src must have shape (3, 3)");
    }
    validate_scalar_finite(fx_src, "camera_model_remap", "fx_src", true);
    validate_scalar_finite(fy_src, "camera_model_remap", "fy_src", true);
    validate_scalar_finite(cx_src, "camera_model_remap", "cx_src");
    validate_scalar_finite(cy_src, "camera_model_remap", "cy_src");
    validate_scalar_finite(fx_dst, "camera_model_remap", "fx_dst", true);
    validate_scalar_finite(fy_dst, "camera_model_remap", "fy_dst", true);
    validate_scalar_finite(cx_dst, "camera_model_remap", "cx_dst");
    validate_scalar_finite(cy_dst, "camera_model_remap", "cy_dst");
    validate_rotation_finite(rotation_dst_to_src, "camera_model_remap");
    if (!hnw::camera::valid_projection(src_projection) ||
        !hnw::camera::valid_projection(dst_projection)) {
        throw std::invalid_argument("camera_model_remap: projection must be 0 "
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
        launcher(static_cast<const T*>(image_info.ptr), static_cast<T*>(out_info.ptr), src_height,
                 src_width, channels, static_cast<int>(out_height), static_cast<int>(out_width),
                 fx_src, fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst,
                 static_cast<const double*>(rotation_info.ptr), src_has_dist,
                 src_dist_coeffs.data(), dst_has_dist, dst_dist_coeffs.data(), src_projection,
                 dst_projection);
    }
    return out;
}

py::array camera_model_remap_fused_dispatch(
    const py::array& image, const ssize_t out_height, const ssize_t out_width, const double fx_src,
    const double fy_src, const double cx_src, const double cy_src, const double fx_dst,
    const double fy_dst, const double cx_dst, const double cy_dst,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rotation_dst_to_src,
    const py::object& src_dist_coeffs, const py::object& dst_dist_coeffs, const int src_projection,
    const int dst_projection) {
    if (py::isinstance<py::array_t<unsigned char>>(image)) {
        return camera_model_remap_fused_impl<unsigned char>(
            image.cast<py::array_t<unsigned char>>(), out_height, out_width, fx_src, fy_src, cx_src,
            cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_dist_coeffs,
            dst_dist_coeffs, src_projection, dst_projection, launch_camera_model_remap_fused_u8);
    }
    if (py::isinstance<py::array_t<unsigned short>>(image)) {
        return camera_model_remap_fused_impl<unsigned short>(
            image.cast<py::array_t<unsigned short>>(), out_height, out_width, fx_src, fy_src,
            cx_src, cy_src, fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_dist_coeffs,
            dst_dist_coeffs, src_projection, dst_projection, launch_camera_model_remap_fused_u16);
    }
    if (py::isinstance<py::array_t<float>>(image)) {
        return camera_model_remap_fused_impl<float>(
            image.cast<py::array_t<float>>(), out_height, out_width, fx_src, fy_src, cx_src, cy_src,
            fx_dst, fy_dst, cx_dst, cy_dst, rotation_dst_to_src, src_dist_coeffs, dst_dist_coeffs,
            src_projection, dst_projection, launch_camera_model_remap_fused_f32);
    }
    throw std::invalid_argument("camera_model_remap: unsupported image dtype; "
                                "expected uint8/uint16/float32");
}

} // namespace

void bind_camera_model_remap_fused_ops(py::module_& m) {
    m.def("camera_model_remap", &camera_model_remap_fused_dispatch, py::arg("image"),
          py::arg("out_height"), py::arg("out_width"), py::arg("fx_src"), py::arg("fy_src"),
          py::arg("cx_src"), py::arg("cy_src"), py::arg("fx_dst"), py::arg("fy_dst"),
          py::arg("cx_dst"), py::arg("cy_dst"), py::arg("rotation_dst_to_src"),
          py::arg("src_dist_coeffs") = py::none(), py::arg("dst_dist_coeffs") = py::none(),
          py::arg("src_projection") = hnw::camera::PROJECTION_PERSPECTIVE,
          py::arg("dst_projection") = hnw::camera::PROJECTION_PERSPECTIVE,
          "Apply a camera-model remap with fused grid generation and bilinear "
          "sampling using CUDA.");
}
