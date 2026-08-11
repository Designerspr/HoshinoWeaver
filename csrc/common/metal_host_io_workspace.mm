#include "metal_host_io_workspace.h"

#include "common/compat.h"
#include "common/metal_error.h"

#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace hnw::metal {
namespace {

void metal_extension_anchor() {}

std::filesystem::path shader_library_path() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&metal_extension_anchor), &info) == 0 ||
        info.dli_fname == nullptr) {
        throw std::runtime_error("Metal backend failed to locate its extension module");
    }
    return std::filesystem::path(info.dli_fname).parent_path() / "_metal_kernels.metallib";
}

std::string ns_error_message(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const char* text = error.localizedDescription.UTF8String;
    return text == nullptr ? "unknown Metal error" : std::string(text);
}

} // namespace

HostIOWorkspace& HostIOWorkspace::current() {
    static thread_local HostIOWorkspace workspace;
    return workspace;
}

HostIOWorkspace::HostIOWorkspace()
    : device_(MTLCreateSystemDefaultDevice()), command_queue_(nil), library_(nil),
      buffers_([[NSMutableArray alloc] init]), pipelines_([[NSMutableDictionary alloc] init]) {
    if (device_ == nil) {
        throw hnw::MetalRuntimeUnavailableError("no Metal device is available");
    }
    command_queue_ = [device_ newCommandQueue];
    if (command_queue_ == nil) {
        throw hnw::MetalRuntimeUnavailableError("failed to create a Metal command queue");
    }
}

id<MTLDevice> HostIOWorkspace::device() const {
    return device_;
}

id<MTLCommandQueue> HostIOWorkspace::command_queue() const {
    return command_queue_;
}

id<MTLBuffer> HostIOWorkspace::buffer(const size_t bytes, const char* context) {
    if (bytes == 0) {
        throw std::invalid_argument("Metal workspace buffer size must be positive");
    }
    logical_bytes_ += bytes;
    const size_t slot = next_buffer_++;
    if (slot < buffer_sizes_.size() && buffer_sizes_[slot] >= bytes) {
        return [buffers_ objectAtIndex:slot];
    }

    id<MTLBuffer> allocated = [device_ newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    if (allocated == nil) {
        throw hnw::MetalResourceExhaustedError(std::string(context) +
                                               ": Metal buffer allocation failed");
    }
    if (slot < buffer_sizes_.size()) {
        [buffers_ replaceObjectAtIndex:slot withObject:allocated];
        buffer_sizes_[slot] = bytes;
    } else {
        [buffers_ addObject:allocated];
        buffer_sizes_.push_back(bytes);
    }
    return allocated;
}

id<MTLLibrary> HostIOWorkspace::load_library() {
    if (library_ != nil) {
        return library_;
    }
    const std::filesystem::path path = shader_library_path();
    NSString* path_string = [NSString stringWithUTF8String:path.string().c_str()];
    NSError* error = nil;
    library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:path_string] error:&error];
    if (library_ == nil) {
        throw std::runtime_error("failed to load packaged Metal shader library " + path.string() +
                                 ": " + ns_error_message(error));
    }
    return library_;
}

id<MTLComputePipelineState> HostIOWorkspace::pipeline(const char* function_name) {
    NSString* name = [NSString stringWithUTF8String:function_name];
    id<MTLComputePipelineState> cached = [pipelines_ objectForKey:name];
    if (cached != nil) {
        return cached;
    }

    id<MTLFunction> function = [load_library() newFunctionWithName:name];
    if (function == nil) {
        throw std::runtime_error(std::string("packaged Metal library is missing function: ") +
                                 function_name);
    }
    NSError* error = nil;
    id<MTLComputePipelineState> created = [device_ newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    if (created == nil) {
        throw std::runtime_error(std::string("failed to create Metal pipeline ") + function_name +
                                 ": " + ns_error_message(error));
    }
    [pipelines_ setObject:created forKey:name];
    return created;
}

void HostIOWorkspace::begin_operation(const char* logical_op) {
    next_buffer_ = 0;
    logical_bytes_ = 0;
    logical_op_ = logical_op;
}

void HostIOWorkspace::finish_operation() {
    last_logical_peak_bytes_ = logical_bytes_;
    last_logical_op_ = logical_op_;
    logical_op_.clear();
    next_buffer_ = 0;
    logical_bytes_ = 0;
}

void HostIOWorkspace::reset_after_error() {
    logical_op_.clear();
    next_buffer_ = 0;
    logical_bytes_ = 0;
}

bool HostIOWorkspace::clear() {
    const bool changed = buffers_.count != 0 || pipelines_.count != 0 || library_ != nil;
    [buffers_ removeAllObjects];
    [pipelines_ removeAllObjects];
    buffer_sizes_.clear();
    library_ = nil;
    next_buffer_ = 0;
    logical_bytes_ = 0;
    last_logical_peak_bytes_ = 0;
    logical_op_.clear();
    last_logical_op_.clear();
    return changed;
}

size_t HostIOWorkspace::retained_bytes() const {
    size_t total = 0;
    for (const size_t bytes : buffer_sizes_) {
        total += bytes;
    }
    return total;
}

size_t HostIOWorkspace::last_logical_peak_bytes() const {
    return last_logical_peak_bytes_;
}

const std::string& HostIOWorkspace::last_logical_op() const {
    return last_logical_op_;
}

} // namespace hnw::metal
