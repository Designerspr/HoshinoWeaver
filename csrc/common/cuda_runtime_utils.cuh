#pragma once

#include "common/compat.h"
#include "common/cuda_error.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hnw::cuda {

inline bool runtime_unavailable(const cudaError_t error) {
    return error == cudaErrorNoDevice ||
           error == cudaErrorInsufficientDriver ||
           error == cudaErrorInitializationError ||
           error == cudaErrorDevicesUnavailable;
}

inline void throw_if_failed(const cudaError_t error, const char* context) {
    if (error == cudaSuccess) {
        return;
    }
    const std::string message =
        std::string(context) + ": " + cudaGetErrorString(error);
    if (runtime_unavailable(error)) {
        throw hnw::CudaRuntimeUnavailableError(message);
    }
    throw std::runtime_error(message);
}

}  // namespace hnw::cuda
