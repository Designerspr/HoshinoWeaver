#include "common/backend_info.h"

#include "common/compat.h"
#include "common/cuda_host_io_workspace.cuh"

#include <cuda_runtime.h>

py::dict cuda_memory_info_cuda_dict() {
    py::dict info;
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        info["available"] = false;
        info["reason"] = cudaGetErrorString(error);
        return info;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        info["available"] = false;
        info["reason"] = cudaGetErrorString(error);
        return info;
    }

    info["available"] = true;
    info["device"] = device;
    info["free_bytes"] = static_cast<unsigned long long>(free_bytes);
    info["total_bytes"] = static_cast<unsigned long long>(total_bytes);
    return info;
}

py::dict cuda_host_io_cache_info_cuda_dict() {
    py::dict info;
    info["available"] = true;
    info["current_thread_device_bytes"] = static_cast<unsigned long long>(
        hnw::cuda::host_io_workspace.retained_device_bytes());
    info["current_thread_pinned_bytes"] = static_cast<unsigned long long>(
        hnw::cuda::host_io_workspace.retained_pinned_bytes());
    info["process_device_bytes"] = static_cast<unsigned long long>(
        hnw::cuda::process_device_cache_bytes.load(std::memory_order_relaxed));
    info["process_pinned_bytes"] = static_cast<unsigned long long>(
        hnw::cuda::process_pinned_cache_bytes.load(std::memory_order_relaxed));
    info["per_thread_limit_bytes"] = static_cast<unsigned long long>(
        hnw::cuda::host_io_cache_per_thread_limit_bytes());
    info["process_limit_bytes"] = static_cast<unsigned long long>(
        hnw::cuda::host_io_cache_process_limit_bytes());
    return info;
}

bool clear_cuda_host_io_cache_cuda() {
    hnw::cuda::clear_current_thread_host_io_workspace();
    return true;
}
