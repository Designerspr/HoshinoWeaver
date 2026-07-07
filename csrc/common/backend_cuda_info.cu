#include "common/backend_info.h"

#include "common/compat.h"

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
