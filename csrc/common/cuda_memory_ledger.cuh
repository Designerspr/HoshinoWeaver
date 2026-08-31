#pragma once

#include "common/compat.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace hnw::cuda {

struct CudaMemoryMeasurement {
    std::string operation;
    size_t current_device_bytes = 0;
    size_t peak_device_bytes = 0;
    size_t current_pinned_bytes = 0;
    size_t peak_pinned_bytes = 0;
};

inline thread_local bool memory_measurement_active = false;
inline thread_local CudaMemoryMeasurement current_memory_measurement;
inline thread_local CudaMemoryMeasurement last_memory_measurement;

inline void begin_cuda_memory_measurement(const char* operation) {
    if (memory_measurement_active) {
        throw std::runtime_error("nested CUDA memory measurement is unsupported");
    }
    current_memory_measurement = CudaMemoryMeasurement{};
    current_memory_measurement.operation = operation;
    memory_measurement_active = true;
}

inline void finish_cuda_memory_measurement() noexcept {
    if (!memory_measurement_active) {
        return;
    }
    last_memory_measurement = current_memory_measurement;
    current_memory_measurement = CudaMemoryMeasurement{};
    memory_measurement_active = false;
}

inline void record_device_memory_acquire(const size_t bytes) {
    if (!memory_measurement_active || bytes == 0) {
        return;
    }
    if (bytes >
        std::numeric_limits<size_t>::max() - current_memory_measurement.current_device_bytes) {
        throw std::overflow_error("CUDA device memory measurement overflow");
    }
    current_memory_measurement.current_device_bytes += bytes;
    current_memory_measurement.peak_device_bytes =
        std::max(current_memory_measurement.peak_device_bytes,
                 current_memory_measurement.current_device_bytes);
}

inline void record_device_memory_release(const size_t bytes) noexcept {
    if (!memory_measurement_active || bytes == 0) {
        return;
    }
    current_memory_measurement.current_device_bytes =
        bytes >= current_memory_measurement.current_device_bytes
            ? 0
            : current_memory_measurement.current_device_bytes - bytes;
}

inline void record_pinned_memory_acquire(const size_t bytes) {
    if (!memory_measurement_active || bytes == 0) {
        return;
    }
    if (bytes >
        std::numeric_limits<size_t>::max() - current_memory_measurement.current_pinned_bytes) {
        throw std::overflow_error("CUDA pinned memory measurement overflow");
    }
    current_memory_measurement.current_pinned_bytes += bytes;
    current_memory_measurement.peak_pinned_bytes =
        std::max(current_memory_measurement.peak_pinned_bytes,
                 current_memory_measurement.current_pinned_bytes);
}

inline void record_pinned_memory_release(const size_t bytes) noexcept {
    if (!memory_measurement_active || bytes == 0) {
        return;
    }
    current_memory_measurement.current_pinned_bytes =
        bytes >= current_memory_measurement.current_pinned_bytes
            ? 0
            : current_memory_measurement.current_pinned_bytes - bytes;
}

} // namespace hnw::cuda
