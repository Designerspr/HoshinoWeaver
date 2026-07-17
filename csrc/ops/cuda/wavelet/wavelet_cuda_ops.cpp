#include "wavelet_cuda_ops.h"

#include "common/compat.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

void launch_wavelet_dec_rec_cuda_core(const double* image_host, double* out_host, int height,
                                      int width, int level);

namespace {

constexpr ssize_t DB8_FILTER_LEN = 16;

ssize_t dwt_len(const ssize_t n) {
    return (n + DB8_FILTER_LEN - 1) / 2;
}

ssize_t idwt_len(const ssize_t n) {
    return 2 * n - DB8_FILTER_LEN + 2;
}

std::pair<ssize_t, ssize_t> wavelet_output_shape(const ssize_t height, const ssize_t width,
                                                 const ssize_t level) {
    std::vector<std::pair<ssize_t, ssize_t>> details;
    details.reserve(static_cast<size_t>(level));
    ssize_t current_h = height;
    ssize_t current_w = width;
    for (ssize_t idx = 0; idx < level; ++idx) {
        current_h = dwt_len(current_h);
        current_w = dwt_len(current_w);
        details.emplace_back(current_h, current_w);
    }
    for (ssize_t idx = level - 1; idx >= 0; --idx) {
        current_h = idwt_len(details[static_cast<size_t>(idx)].first);
        current_w = idwt_len(details[static_cast<size_t>(idx)].second);
    }
    return {current_h, current_w};
}

py::array_t<double> wavelet_dec_rec_cuda_core_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& image,
    const ssize_t level) {
    if (image.ndim() != 2) {
        throw std::invalid_argument("wavelet_dec_rec_cuda_core: image must be 2D");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument(
            "wavelet_dec_rec_cuda_core: image height and width must be positive");
    }
    if (level <= 0) {
        throw std::invalid_argument("wavelet_dec_rec_cuda_core: invalid wavelet level");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() ||
        image.shape(1) > std::numeric_limits<int>::max() ||
        level > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("wavelet_dec_rec_cuda_core: input is too large");
    }
    const ssize_t input_size = image.shape(0) * image.shape(1);
    if (input_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("wavelet_dec_rec_cuda_core: input is too large");
    }
    const auto [out_h, out_w] = wavelet_output_shape(image.shape(0), image.shape(1), level);
    const ssize_t output_size = out_h * out_w;
    if (out_h <= 0 || out_w <= 0 || out_h > std::numeric_limits<int>::max() ||
        out_w > std::numeric_limits<int>::max() || output_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("wavelet_dec_rec_cuda_core: output shape is invalid");
    }

    py::array_t<double> output({out_h, out_w});
    launch_wavelet_dec_rec_cuda_core(image.data(), output.mutable_data(),
                                     static_cast<int>(image.shape(0)),
                                     static_cast<int>(image.shape(1)), static_cast<int>(level));
    return output;
}

} // namespace

void bind_wavelet_cuda_ops(py::module_& m) {
    m.def("wavelet_dec_rec_cuda_core", &wavelet_dec_rec_cuda_core_impl, py::arg("image"),
          py::arg("level"));
}
