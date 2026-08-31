#include "metal_runtime.h"

#include "common/compat.h"
#include "common/metal_error.h"
#include "common/metal_host_io_workspace.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <string>

namespace {

py::dict metal_build_info() {
    py::dict info;
#if defined(__aarch64__) || defined(__arm64__)
    info["arch"] = "arm64";
#elif defined(__x86_64__)
    info["arch"] = "x86_64";
#else
    info["arch"] = "unknown";
#endif
    info["platform"] = "macos";
    info["compiler"] = "clang";
    info["metal"] = true;
    return info;
}

py::dict metal_device_info() {
    @autoreleasepool {
        py::dict info;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            info["available"] = false;
            info["status"] = "explicitly_unavailable";
            info["reason_code"] = "metal_device_unavailable";
            info["category"] = "runtime";
            info["reason"] = "no Metal device is available";
            return info;
        }

        if (!device.hasUnifiedMemory) {
            info["available"] = false;
            info["status"] = "explicitly_unavailable";
            info["reason_code"] = "metal_unified_memory_required";
            info["category"] = "device";
            info["reason"] = "the Metal backend currently requires unified memory";
            return info;
        }

        info["available"] = true;
        info["status"] = "available";
        info["reason_code"] = "metal_available";
        info["category"] = "available";
        info["name"] = std::string(device.name.UTF8String);
        info["registry_id"] = py::int_(device.registryID);
        info["recommended_max_working_set_bytes"] = py::int_(device.recommendedMaxWorkingSetSize);
        info["current_allocated_bytes"] = py::int_(device.currentAllocatedSize);
        info["has_unified_memory"] = device.hasUnifiedMemory;
        return info;
    }
}

py::dict metal_host_io_cache_info() {
    py::dict info;
    try {
        auto& workspace = hnw::metal::HostIOWorkspace::current();
        info["available"] = true;
        info["retained_bytes"] = py::int_(workspace.retained_bytes());
        info["last_logical_peak_bytes"] = py::int_(workspace.last_logical_peak_bytes());
        info["last_logical_op"] = workspace.last_logical_op();
    } catch (const hnw::MetalRuntimeUnavailableError& exc) {
        info["available"] = false;
        info["reason"] = exc.what();
    }
    return info;
}

bool clear_metal_host_io_cache() {
    try {
        return hnw::metal::HostIOWorkspace::current().clear();
    } catch (const hnw::MetalRuntimeUnavailableError&) {
        return false;
    }
}

} // namespace

void bind_metal_runtime(py::module_& m) {
    m.def("build_info", &metal_build_info, "Return Metal backend build metadata.");
    m.def("metal_device_info", &metal_device_info,
          "Return Metal device and unified-memory metadata.");
    m.def("metal_host_io_cache_info", &metal_host_io_cache_info,
          "Return Metal host-I/O workspace metadata for the current worker thread.");
    m.def("clear_metal_host_io_cache", &clear_metal_host_io_cache,
          "Clear the Metal host-I/O cache owned by the current worker thread.");
}
