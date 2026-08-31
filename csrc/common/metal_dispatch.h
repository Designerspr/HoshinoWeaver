#pragma once

#include "common/metal_error.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace hnw::metal {

// Shared command-buffer plumbing for the host-I/O Metal ops. `context` is the
// wrapper name used in error messages, e.g. "star_mask_dog_metal".

inline void dispatch_1d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                        const uint32_t count) {
    const NSUInteger width = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

inline id<MTLComputeCommandEncoder> begin_encoder(id<MTLCommandBuffer> command_buffer,
                                                  id<MTLComputePipelineState> pipeline,
                                                  const char* context) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        throw std::runtime_error(std::string(context) + ": failed to create command encoder");
    }
    [encoder setComputePipelineState:pipeline];
    return encoder;
}

inline id<MTLCommandBuffer> new_command_buffer(id<MTLCommandQueue> queue, const char* context) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (command_buffer == nil) {
        throw MetalRuntimeUnavailableError(std::string(context) +
                                           ": failed to create command buffer");
    }
    return command_buffer;
}

inline void throw_if_command_failed(id<MTLCommandBuffer> command_buffer, const char* context) {
    if (command_buffer.status != MTLCommandBufferStatusError) {
        return;
    }
    NSError* error = command_buffer.error;
    const std::string message = error == nil || error.localizedDescription.UTF8String == nullptr
                                    ? "unknown Metal command-buffer error"
                                    : std::string(error.localizedDescription.UTF8String);
    if ([error.domain isEqualToString:MTLCommandBufferErrorDomain]) {
        if (error.code == MTLCommandBufferErrorOutOfMemory) {
            throw MetalResourceExhaustedError(std::string(context) + ": " + message);
        }
        if (error.code == MTLCommandBufferErrorDeviceRemoved) {
            throw MetalRuntimeUnavailableError(std::string(context) + ": " + message);
        }
    }
    throw std::runtime_error(std::string(context) + " command failed: " + message);
}

} // namespace hnw::metal
