#include "star_mask_ops.h"

#include "common/compat.h"
#include "common/gaussian_kernel.h"

#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

void launch_star_mask_dog_cuda_u8(const uint8_t* image_host, uint8_t* mask_host, int height,
                                  int width, int channels, const float* small_kernel_host,
                                  int small_radius, const float* large_kernel_host,
                                  int large_radius, float threshold_ratio, int open_ksize,
                                  int dilate_ksize);

void launch_star_mask_dog_cuda_u16(const uint16_t* image_host, uint8_t* mask_host, int height,
                                   int width, int channels, const float* small_kernel_host,
                                   int small_radius, const float* large_kernel_host,
                                   int large_radius, float threshold_ratio, int open_ksize,
                                   int dilate_ksize);

void launch_star_shrink_dog_process_cuda_u8(const uint8_t* image_host, uint8_t* out_host,
                                            int height, int width, int channels,
                                            const float* small_kernel_host, int small_radius,
                                            const float* large_kernel_host, int large_radius,
                                            float threshold_ratio, int open_ksize, int dilate_ksize,
                                            int shrink_ksize, int shrink_shape, int shrink_times,
                                            float shrink_ratio, int deringing_ksize);

void launch_star_shrink_dog_process_cuda_u16(
    const uint16_t* image_host, uint16_t* out_host, int height, int width, int channels,
    const float* small_kernel_host, int small_radius, const float* large_kernel_host,
    int large_radius, float threshold_ratio, int open_ksize, int dilate_ksize, int shrink_ksize,
    int shrink_shape, int shrink_times, float shrink_ratio, int deringing_ksize);

namespace {

template <typename T>
using launch_fn_t = void (*)(const T* image_host, uint8_t* mask_host, int height, int width,
                             int channels, const float* small_kernel_host, int small_radius,
                             const float* large_kernel_host, int large_radius,
                             float threshold_ratio, int open_ksize, int dilate_ksize);

template <typename T>
using process_launch_fn_t = void (*)(const T* image_host, T* out_host, int height, int width,
                                     int channels, const float* small_kernel_host, int small_radius,
                                     const float* large_kernel_host, int large_radius,
                                     float threshold_ratio, int open_ksize, int dilate_ksize,
                                     int shrink_ksize, int shrink_shape, int shrink_times,
                                     float shrink_ratio, int deringing_ksize);

void validate_common(const py::array& image, const int open_ksize, const int dilate_ksize) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "star_mask_dog_cuda: image must have shape (H, W) or (H, W, C)");
    }
    if (image.ndim() == 3 && image.shape(2) != 3) {
        throw std::invalid_argument("star_mask_dog_cuda: 3D image must have exactly 3 channels");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument("star_mask_dog_cuda: image height and width must be positive");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() ||
        image.shape(1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("star_mask_dog_cuda: image is too large");
    }
    if ((open_ksize < 0) || (open_ksize > 0 && open_ksize % 2 == 0) || (dilate_ksize < 0) ||
        (dilate_ksize > 0 && dilate_ksize % 2 == 0)) {
        throw std::invalid_argument(
            "star_mask_dog_cuda: morphology kernel sizes must be zero or positive odd values");
    }
}

int parse_shape(const std::string& shape) {
    if (shape == "RECT") {
        return 0;
    }
    if (shape == "CROSS") {
        return 1;
    }
    if (shape == "CIRCLE") {
        return 2;
    }
    throw std::invalid_argument("star_shrink_dog_process_cuda: unknown shrink_shape");
}

void validate_process_params(const int shrink_ksize, const int shrink_times,
                             const float shrink_ratio, const int deringing_ksize) {
    if (shrink_ksize <= 0 || shrink_ksize % 2 == 0) {
        throw std::invalid_argument(
            "star_shrink_dog_process_cuda: shrink_ksize must be a positive odd value");
    }
    if (shrink_times <= 0) {
        throw std::invalid_argument("star_shrink_dog_process_cuda: shrink_times must be positive");
    }
    if (!(shrink_ratio > 0.0f && shrink_ratio <= 1.0f)) {
        throw std::invalid_argument("star_shrink_dog_process_cuda: shrink_ratio must be in (0, 1]");
    }
    if (deringing_ksize <= 0 || deringing_ksize % 2 == 0) {
        throw std::invalid_argument(
            "star_shrink_dog_process_cuda: deringing_ksize must be a positive odd value");
    }
}

template <typename T>
py::array_t<uint8_t>
star_mask_dog_cuda_impl(const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
                        const float sigma_small, const float sigma_large,
                        const float threshold_ratio, const int open_ksize, const int dilate_ksize,
                        launch_fn_t<T> launcher) {
    validate_common(image, open_ksize, dilate_ksize);
    const int height = static_cast<int>(image.shape(0));
    const int width = static_cast<int>(image.shape(1));
    const int channels = image.ndim() == 3 ? 3 : 1;
    std::vector<float> small_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_small, "star_mask_dog_cuda");
    std::vector<float> large_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_large, "star_mask_dog_cuda");
    const int small_radius = static_cast<int>(small_kernel.size() / 2);
    const int large_radius = static_cast<int>(large_kernel.size() / 2);
    py::array_t<uint8_t> output({height, width});
    {
        py::gil_scoped_release release;
        launcher(image.data(), output.mutable_data(), height, width, channels, small_kernel.data(),
                 small_radius, large_kernel.data(), large_radius, threshold_ratio, open_ksize,
                 dilate_ksize);
    }
    return output;
}

py::array star_mask_dog_cuda_dispatch(const py::array& image, const float sigma_small,
                                      const float sigma_large, const float threshold_ratio,
                                      const int open_ksize, const int dilate_ksize) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_mask_dog_cuda_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_ksize, dilate_ksize,
            launch_star_mask_dog_cuda_u8);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_mask_dog_cuda_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_ksize, dilate_ksize,
            launch_star_mask_dog_cuda_u16);
    }
    throw std::invalid_argument("star_mask_dog_cuda: unsupported dtype; expected uint8 or uint16");
}

template <typename T>
py::array_t<T> star_shrink_dog_process_cuda_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image, const float sigma_small,
    const float sigma_large, const float threshold_ratio, const int open_ksize,
    const int dilate_ksize, const int shrink_ksize, const std::string& shrink_shape,
    const int shrink_times, const float shrink_ratio, const int deringing_ksize,
    process_launch_fn_t<T> launcher) {
    validate_common(image, open_ksize, dilate_ksize);
    validate_process_params(shrink_ksize, shrink_times, shrink_ratio, deringing_ksize);
    const int height = static_cast<int>(image.shape(0));
    const int width = static_cast<int>(image.shape(1));
    const int channels = image.ndim() == 3 ? 3 : 1;
    const int shape = parse_shape(shrink_shape);
    std::vector<float> small_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_small, "star_mask_dog_cuda");
    std::vector<float> large_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_large, "star_mask_dog_cuda");
    const int small_radius = static_cast<int>(small_kernel.size() / 2);
    const int large_radius = static_cast<int>(large_kernel.size() / 2);
    py::array_t<T> output(image.request().shape);
    {
        py::gil_scoped_release release;
        launcher(image.data(), output.mutable_data(), height, width, channels, small_kernel.data(),
                 small_radius, large_kernel.data(), large_radius, threshold_ratio, open_ksize,
                 dilate_ksize, shrink_ksize, shape, shrink_times, shrink_ratio, deringing_ksize);
    }
    return output;
}

py::array star_shrink_dog_process_cuda_dispatch(const py::array& image, const float sigma_small,
                                                const float sigma_large,
                                                const float threshold_ratio, const int open_ksize,
                                                const int dilate_ksize, const int shrink_ksize,
                                                const std::string& shrink_shape,
                                                const int shrink_times, const float shrink_ratio,
                                                const int deringing_ksize) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_shrink_dog_process_cuda_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_ksize, dilate_ksize, shrink_ksize,
            shrink_shape, shrink_times, shrink_ratio, deringing_ksize,
            launch_star_shrink_dog_process_cuda_u8);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_shrink_dog_process_cuda_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_ksize, dilate_ksize, shrink_ksize,
            shrink_shape, shrink_times, shrink_ratio, deringing_ksize,
            launch_star_shrink_dog_process_cuda_u16);
    }
    throw std::invalid_argument(
        "star_shrink_dog_process_cuda: unsupported dtype; expected uint8 or uint16");
}

} // namespace

void bind_star_mask_cuda_ops(py::module_& m) {
    m.def("star_mask_dog_cuda", &star_mask_dog_cuda_dispatch, py::arg("image"),
          py::arg("sigma_small"), py::arg("sigma_large"), py::arg("threshold_ratio"),
          py::arg("open_ksize"), py::arg("dilate_ksize"),
          "CUDA host-in/out DoG star mask detector.");
    m.def("star_shrink_dog_process_cuda", &star_shrink_dog_process_cuda_dispatch, py::arg("image"),
          py::arg("sigma_small"), py::arg("sigma_large"), py::arg("threshold_ratio"),
          py::arg("open_ksize"), py::arg("dilate_ksize"), py::arg("shrink_ksize"),
          py::arg("shrink_shape"), py::arg("shrink_times"), py::arg("shrink_ratio"),
          py::arg("deringing_ksize"),
          "CUDA host-in/out fused DoG mask detection and star shrink processing.");
}
