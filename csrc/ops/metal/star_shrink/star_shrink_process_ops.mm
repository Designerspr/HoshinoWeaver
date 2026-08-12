#include "star_shrink_process_ops.h"

#include "common/compat.h"
#include "common/metal_dispatch.h"
#include "common/metal_error.h"
#include "common/metal_host_io_workspace.h"

#include <pybind11/numpy.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr const char* kContext = "star_shrink_process_metal";

struct StarShrinkParams {
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t shrink_ksize;
    uint32_t shrink_shape;
    uint32_t deringing_ksize;
    float shrink_ratio;
};

static_assert(sizeof(StarShrinkParams) == 28,
              "Metal host parameters must match the MSL struct layout");

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
    throw std::invalid_argument("star_shrink_process_metal: unknown shrink_shape");
}

void validate_common(const py::array& image, const py::array& mask, const int shrink_ksize,
                     const int shrink_times, const float shrink_ratio, const int deringing_ksize) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "star_shrink_process_metal: image must have shape (H, W) or (H, W, C)");
    }
    if (image.ndim() == 3 && image.shape(2) != 3) {
        throw std::invalid_argument(
            "star_shrink_process_metal: 3D image must have exactly 3 channels");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument(
            "star_shrink_process_metal: image height and width must be positive");
    }
    if (mask.ndim() != 2 || mask.shape(0) != image.shape(0) || mask.shape(1) != image.shape(1)) {
        throw std::invalid_argument("star_shrink_process_metal: star_mask must have shape (H, W)");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() ||
        image.shape(1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("star_shrink_process_metal: image is too large");
    }
    const uint64_t channels = image.ndim() == 3 ? 3 : 1;
    const uint64_t total = static_cast<uint64_t>(image.shape(0)) * image.shape(1) * channels;
    if (total > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("star_shrink_process_metal: image is too large");
    }
    if (shrink_ksize <= 0 || shrink_ksize % 2 == 0) {
        throw std::invalid_argument(
            "star_shrink_process_metal: shrink_ksize must be a positive odd value");
    }
    if (shrink_times <= 0) {
        throw std::invalid_argument("star_shrink_process_metal: shrink_times must be positive");
    }
    if (!(shrink_ratio > 0.0f && shrink_ratio <= 1.0f)) {
        throw std::invalid_argument("star_shrink_process_metal: shrink_ratio must be in (0, 1]");
    }
    if (deringing_ksize <= 0 || deringing_ksize % 2 == 0) {
        throw std::invalid_argument(
            "star_shrink_process_metal: deringing_ksize must be a positive odd value");
    }
}

template <typename T>
void launch_star_shrink_process_metal(const T* image_host, const uint8_t* mask_host, T* out_host,
                                      const int height, const int width, const int channels,
                                      const int shrink_ksize, const int shrink_shape,
                                      const int shrink_times, const float shrink_ratio,
                                      const int deringing_ksize) {
    @autoreleasepool {
        auto& workspace = hnw::metal::HostIOWorkspace::current();
        workspace.begin_operation("star_shrink_process");
        try {
            const uint32_t plane_size =
                static_cast<uint32_t>(static_cast<uint64_t>(height) * static_cast<uint64_t>(width));
            const uint32_t total = static_cast<uint32_t>(static_cast<uint64_t>(plane_size) *
                                                         static_cast<uint64_t>(channels));
            const size_t image_bytes = static_cast<size_t>(total) * sizeof(T);
            const size_t mask_bytes = static_cast<size_t>(plane_size);
            const size_t plane_float_bytes = static_cast<size_t>(plane_size) * sizeof(float);
            const size_t total_float_bytes = static_cast<size_t>(total) * sizeof(float);

            id<MTLBuffer> image = workspace.buffer(image_bytes, "star_shrink_process_metal image");
            id<MTLBuffer> output =
                workspace.buffer(image_bytes, "star_shrink_process_metal output");
            id<MTLBuffer> mask = workspace.buffer(mask_bytes, "star_shrink_process_metal mask");
            id<MTLBuffer> luma =
                workspace.buffer(plane_float_bytes, "star_shrink_process_metal luma");
            id<MTLBuffer> luma_tmp =
                workspace.buffer(plane_float_bytes, "star_shrink_process_metal luma_tmp");
            id<MTLBuffer> lab_a =
                workspace.buffer(plane_float_bytes, "star_shrink_process_metal lab_a");
            id<MTLBuffer> lab_b =
                workspace.buffer(plane_float_bytes, "star_shrink_process_metal lab_b");
            id<MTLBuffer> shrunk =
                workspace.buffer(total_float_bytes, "star_shrink_process_metal shrunk");
            id<MTLBuffer> blur_tmp =
                workspace.buffer(total_float_bytes, "star_shrink_process_metal blur_tmp");
            id<MTLBuffer> blurred =
                workspace.buffer(total_float_bytes, "star_shrink_process_metal blurred");

            std::memcpy(image.contents, image_host, image_bytes);
            std::memcpy(mask.contents, mask_host, mask_bytes);

            const StarShrinkParams params{
                static_cast<uint32_t>(height),
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(channels),
                static_cast<uint32_t>(shrink_ksize),
                static_cast<uint32_t>(shrink_shape),
                static_cast<uint32_t>(deringing_ksize),
                shrink_ratio,
            };
            const char* suffix = std::is_same_v<T, uint8_t> ? "u8" : "u16";

            id<MTLCommandBuffer> command_buffer =
                hnw::metal::new_command_buffer(workspace.command_queue(), kContext);

            {
                const std::string name = std::string("star_shrink_bgr_to_lab_") + suffix;
                id<MTLComputePipelineState> pipeline = workspace.pipeline(name.c_str());
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(command_buffer, pipeline, kContext);
                [encoder setBuffer:image offset:0 atIndex:0];
                [encoder setBuffer:luma offset:0 atIndex:1];
                [encoder setBuffer:lab_a offset:0 atIndex:2];
                [encoder setBuffer:lab_b offset:0 atIndex:3];
                [encoder setBytes:&params length:sizeof(params) atIndex:4];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }

            id<MTLBuffer> current = luma;
            id<MTLBuffer> next = luma_tmp;
            id<MTLComputePipelineState> erode_pipeline =
                workspace.pipeline("star_shrink_erode_luma");
            for (int iteration = 0; iteration < shrink_times; ++iteration) {
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(command_buffer, erode_pipeline, kContext);
                [encoder setBuffer:current offset:0 atIndex:0];
                [encoder setBuffer:next offset:0 atIndex:1];
                [encoder setBytes:&params length:sizeof(params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, erode_pipeline, plane_size);
                [encoder endEncoding];
                std::swap(current, next);
            }

            {
                id<MTLComputePipelineState> pipeline = workspace.pipeline("star_shrink_lab_to_bgr");
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(command_buffer, pipeline, kContext);
                [encoder setBuffer:current offset:0 atIndex:0];
                [encoder setBuffer:lab_a offset:0 atIndex:1];
                [encoder setBuffer:lab_b offset:0 atIndex:2];
                [encoder setBuffer:shrunk offset:0 atIndex:3];
                [encoder setBytes:&params length:sizeof(params) atIndex:4];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }

            {
                const std::string name = std::string("star_shrink_horizontal_blur_") + suffix;
                id<MTLComputePipelineState> pipeline = workspace.pipeline(name.c_str());
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(command_buffer, pipeline, kContext);
                [encoder setBuffer:image offset:0 atIndex:0];
                [encoder setBuffer:blur_tmp offset:0 atIndex:1];
                [encoder setBytes:&params length:sizeof(params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, pipeline, total);
                [encoder endEncoding];
            }

            {
                id<MTLComputePipelineState> pipeline =
                    workspace.pipeline("star_shrink_vertical_blur");
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(command_buffer, pipeline, kContext);
                [encoder setBuffer:blur_tmp offset:0 atIndex:0];
                [encoder setBuffer:blurred offset:0 atIndex:1];
                [encoder setBytes:&params length:sizeof(params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, pipeline, total);
                [encoder endEncoding];
            }

            {
                const std::string name = std::string("star_shrink_final_mask_") + suffix;
                id<MTLComputePipelineState> pipeline = workspace.pipeline(name.c_str());
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(command_buffer, pipeline, kContext);
                [encoder setBuffer:image offset:0 atIndex:0];
                [encoder setBuffer:mask offset:0 atIndex:1];
                [encoder setBuffer:shrunk offset:0 atIndex:2];
                [encoder setBuffer:blurred offset:0 atIndex:3];
                [encoder setBuffer:output offset:0 atIndex:4];
                [encoder setBytes:&params length:sizeof(params) atIndex:5];
                hnw::metal::dispatch_1d(encoder, pipeline, total);
                [encoder endEncoding];
            }

            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            hnw::metal::throw_if_command_failed(command_buffer, kContext);
            std::memcpy(out_host, output.contents, image_bytes);
            workspace.finish_operation();
        } catch (...) {
            workspace.reset_after_error();
            throw;
        }
    }
}

template <typename T>
py::array_t<T> star_shrink_process_metal_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    const int shrink_ksize, const std::string& shrink_shape, const int shrink_times,
    const float shrink_ratio, const int deringing_ksize) {
    validate_common(image, mask, shrink_ksize, shrink_times, shrink_ratio, deringing_ksize);
    const int height = static_cast<int>(image.shape(0));
    const int width = static_cast<int>(image.shape(1));
    const int channels = image.ndim() == 3 ? 3 : 1;
    const int shape = parse_shape(shrink_shape);
    py::array_t<T> output(image.request().shape);
    {
        py::gil_scoped_release release;
        launch_star_shrink_process_metal(image.data(), mask.data(), output.mutable_data(), height,
                                         width, channels, shrink_ksize, shape, shrink_times,
                                         shrink_ratio, deringing_ksize);
    }
    return output;
}

py::array star_shrink_process_metal_dispatch(const py::array& image, const py::array& mask,
                                             const int shrink_ksize,
                                             const std::string& shrink_shape,
                                             const int shrink_times, const float shrink_ratio,
                                             const int deringing_ksize) {
    const auto mask_array =
        mask.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_shrink_process_metal_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            mask_array, shrink_ksize, shrink_shape, shrink_times, shrink_ratio, deringing_ksize);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_shrink_process_metal_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            mask_array, shrink_ksize, shrink_shape, shrink_times, shrink_ratio, deringing_ksize);
    }
    throw std::invalid_argument(
        "star_shrink_process_metal: unsupported dtype; expected uint8 or uint16");
}

} // namespace

void bind_star_shrink_process_metal_ops(py::module_& m) {
    m.def("star_shrink_process_metal", &star_shrink_process_metal_dispatch, py::arg("image"),
          py::arg("star_mask"), py::arg("shrink_ksize"), py::arg("shrink_shape"),
          py::arg("shrink_times"), py::arg("shrink_ratio"), py::arg("deringing_ksize"),
          "Run star-shrink processing with the Metal host-I/O backend.");
}
