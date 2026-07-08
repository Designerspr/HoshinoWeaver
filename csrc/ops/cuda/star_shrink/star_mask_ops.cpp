#include "common/compat.h"
#include "star_mask_ops.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>

void launch_star_mask_dog_cuda_u8(const uint8_t* image_host,
                                  uint8_t* mask_host,
                                  int height,
                                  int width,
                                  int channels,
                                  const float* small_kernel_host,
                                  int small_radius,
                                  const float* large_kernel_host,
                                  int large_radius,
                                  float threshold_ratio,
                                  int open_ksize,
                                  int dilate_ksize);

void launch_star_mask_dog_cuda_u16(const uint16_t* image_host,
                                   uint8_t* mask_host,
                                   int height,
                                   int width,
                                   int channels,
                                   const float* small_kernel_host,
                                   int small_radius,
                                   const float* large_kernel_host,
                                   int large_radius,
                                   float threshold_ratio,
                                   int open_ksize,
                                   int dilate_ksize);

namespace {

template <typename T>
using launch_fn_t = void (*)(const T* image_host,
                             uint8_t* mask_host,
                             int height,
                             int width,
                             int channels,
                             const float* small_kernel_host,
                             int small_radius,
                             const float* large_kernel_host,
                             int large_radius,
                             float threshold_ratio,
                             int open_ksize,
                             int dilate_ksize);

std::vector<float> make_gaussian_kernel(const float sigma) {
    if (!(sigma > 0.0f)) {
        throw std::invalid_argument("star_mask_dog_cuda: sigma values must be positive");
    }
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    const float denom = 2.0f * sigma * sigma;
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const float value = std::exp(-(static_cast<float>(i * i)) / denom);
        kernel[static_cast<size_t>(i + radius)] = value;
        sum += value;
    }
    for (float& value : kernel) {
        value = static_cast<float>(static_cast<double>(value) / sum);
    }
    return kernel;
}

void validate_common(const py::array& image,
                     const int open_ksize,
                     const int dilate_ksize) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "star_mask_dog_cuda: image must have shape (H, W) or (H, W, C)");
    }
    if (image.ndim() == 3 && image.shape(2) != 3) {
        throw std::invalid_argument(
            "star_mask_dog_cuda: 3D image must have exactly 3 channels");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument(
            "star_mask_dog_cuda: image height and width must be positive");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() ||
        image.shape(1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("star_mask_dog_cuda: image is too large");
    }
    if ((open_ksize < 0) || (open_ksize > 0 && open_ksize % 2 == 0) ||
        (dilate_ksize < 0) || (dilate_ksize > 0 && dilate_ksize % 2 == 0)) {
        throw std::invalid_argument(
            "star_mask_dog_cuda: morphology kernel sizes must be zero or positive odd values");
    }
}

template <typename T>
py::array_t<uint8_t> star_mask_dog_cuda_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
    const float sigma_small,
    const float sigma_large,
    const float threshold_ratio,
    const int open_ksize,
    const int dilate_ksize,
    launch_fn_t<T> launcher) {
    validate_common(image, open_ksize, dilate_ksize);
    const int height = static_cast<int>(image.shape(0));
    const int width = static_cast<int>(image.shape(1));
    const int channels = image.ndim() == 3 ? 3 : 1;
    std::vector<float> small_kernel = make_gaussian_kernel(sigma_small);
    std::vector<float> large_kernel = make_gaussian_kernel(sigma_large);
    const int small_radius = static_cast<int>(small_kernel.size() / 2);
    const int large_radius = static_cast<int>(large_kernel.size() / 2);
    py::array_t<uint8_t> output({height, width});
    {
        py::gil_scoped_release release;
        launcher(image.data(),
                 output.mutable_data(),
                 height,
                 width,
                 channels,
                 small_kernel.data(),
                 small_radius,
                 large_kernel.data(),
                 large_radius,
                 threshold_ratio,
                 open_ksize,
                 dilate_ksize);
    }
    return output;
}

py::array star_mask_dog_cuda_dispatch(const py::array& image,
                                      const float sigma_small,
                                      const float sigma_large,
                                      const float threshold_ratio,
                                      const int open_ksize,
                                      const int dilate_ksize) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_mask_dog_cuda_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small,
            sigma_large,
            threshold_ratio,
            open_ksize,
            dilate_ksize,
            launch_star_mask_dog_cuda_u8);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_mask_dog_cuda_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small,
            sigma_large,
            threshold_ratio,
            open_ksize,
            dilate_ksize,
            launch_star_mask_dog_cuda_u16);
    }
    throw std::invalid_argument(
        "star_mask_dog_cuda: unsupported dtype; expected uint8 or uint16");
}

}  // namespace

void bind_star_mask_cuda_ops(py::module_& m) {
    m.def("star_mask_dog_cuda",
          &star_mask_dog_cuda_dispatch,
          py::arg("image"),
          py::arg("sigma_small"),
          py::arg("sigma_large"),
          py::arg("threshold_ratio"),
          py::arg("open_ksize"),
          py::arg("dilate_ksize"),
          "CUDA host-in/out DoG star mask detector.");
}
