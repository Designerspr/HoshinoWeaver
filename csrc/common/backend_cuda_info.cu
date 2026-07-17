#include "common/backend_info.h"
#include "common/compat.h"
#include "common/cuda_host_io_workspace.cuh"
#include "common/cuda_runtime_utils.cuh"

#include <cuda_runtime.h>

namespace {

py::dict cuda_unavailable_info(const cudaError_t error) {
    py::dict info;
    info["available"] = false;
    info["status"] = "explicitly_unavailable";
    info["reason_code"] = "cuda_runtime_unavailable";
    info["category"] = "availability";
    info["error_code"] = static_cast<int>(error);
    info["reason"] = cudaGetErrorString(error);
    return info;
}

py::dict cuda_error_info(const cudaError_t error) {
    py::dict info;
    info["available"] = false;
    info["status"] = "error";
    info["reason_code"] = "cuda_runtime_error";
    info["category"] = error == cudaErrorMemoryAllocation ? "resource" : "runtime";
    info["error_code"] = static_cast<int>(error);
    info["reason"] = cudaGetErrorString(error);
    return info;
}

} // namespace

py::dict cuda_memory_info_cuda_dict() {
    py::dict info;
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        if (hnw::cuda::runtime_unavailable(error)) {
            return cuda_unavailable_info(error);
        }
        return cuda_error_info(error);
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        if (hnw::cuda::runtime_unavailable(error)) {
            return cuda_unavailable_info(error);
        }
        return cuda_error_info(error);
    }

    info["available"] = true;
    info["status"] = "available";
    info["reason_code"] = "cuda_available";
    info["category"] = "available";
    info["device"] = device;
    info["free_bytes"] = static_cast<unsigned long long>(free_bytes);
    info["total_bytes"] = static_cast<unsigned long long>(total_bytes);
    return info;
}

py::dict cuda_host_io_cache_info_cuda_dict() {
    py::dict info;
    info["available"] = true;
    info["current_thread_device_bytes"] =
        static_cast<unsigned long long>(hnw::cuda::host_io_workspace.retained_device_bytes());
    info["current_thread_pinned_bytes"] =
        static_cast<unsigned long long>(hnw::cuda::host_io_workspace.retained_pinned_bytes());
    info["process_device_bytes"] = static_cast<unsigned long long>(
        hnw::cuda::process_device_cache_bytes.load(std::memory_order_relaxed));
    info["process_pinned_bytes"] = static_cast<unsigned long long>(
        hnw::cuda::process_pinned_cache_bytes.load(std::memory_order_relaxed));
    info["per_thread_limit_bytes"] =
        static_cast<unsigned long long>(hnw::cuda::host_io_cache_per_thread_limit_bytes());
    info["process_limit_bytes"] =
        static_cast<unsigned long long>(hnw::cuda::host_io_cache_process_limit_bytes());
    info["measurement_active"] = hnw::cuda::memory_measurement_active;
    info["current_operation"] = hnw::cuda::current_memory_measurement.operation;
    info["current_device_bytes"] =
        static_cast<unsigned long long>(hnw::cuda::current_memory_measurement.current_device_bytes);
    info["current_device_peak_bytes"] =
        static_cast<unsigned long long>(hnw::cuda::current_memory_measurement.peak_device_bytes);
    info["last_operation"] = hnw::cuda::last_memory_measurement.operation;
    info["last_device_peak_bytes"] =
        static_cast<unsigned long long>(hnw::cuda::last_memory_measurement.peak_device_bytes);
    info["last_pinned_peak_bytes"] =
        static_cast<unsigned long long>(hnw::cuda::last_memory_measurement.peak_pinned_bytes);
    return info;
}

bool clear_cuda_host_io_cache_cuda() {
    hnw::cuda::clear_current_thread_host_io_workspace();
    return true;
}
