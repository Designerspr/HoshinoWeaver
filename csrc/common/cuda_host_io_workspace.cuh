#pragma once

#include "common/compat.h"
#include "common/cuda_runtime_utils.cuh"

#include <cuda_runtime.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hnw::cuda {

constexpr size_t DEFAULT_HOST_IO_CACHE_PER_THREAD_LIMIT_BYTES =
    512ULL * 1024ULL * 1024ULL;
constexpr size_t DEFAULT_HOST_IO_CACHE_PROCESS_LIMIT_BYTES =
    1024ULL * 1024ULL * 1024ULL;

inline std::atomic<size_t> process_device_cache_bytes{0};
inline std::atomic<size_t> process_pinned_cache_bytes{0};

inline size_t parse_cache_limit_bytes(const char* environment_name,
                                      const size_t default_bytes) {
    const char* raw = std::getenv(environment_name);
    if (raw == nullptr || *raw == '\0' || *raw == '-') {
        return default_bytes;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0') {
        return default_bytes;
    }
    constexpr unsigned long long bytes_per_mb = 1024ULL * 1024ULL;
    if (value > std::numeric_limits<size_t>::max() / bytes_per_mb) {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(value * bytes_per_mb);
}

inline size_t host_io_cache_per_thread_limit_bytes() {
    static const size_t limit = parse_cache_limit_bytes(
        "HNW_CUDA_HOST_IO_CACHE_PER_THREAD_MB",
        DEFAULT_HOST_IO_CACHE_PER_THREAD_LIMIT_BYTES);
    return limit;
}

inline size_t host_io_cache_process_limit_bytes() {
    static const size_t limit = parse_cache_limit_bytes(
        "HNW_CUDA_HOST_IO_CACHE_MB",
        DEFAULT_HOST_IO_CACHE_PROCESS_LIMIT_BYTES);
    return limit;
}

inline size_t process_host_io_cache_bytes() {
    const size_t device_bytes =
        process_device_cache_bytes.load(std::memory_order_relaxed);
    const size_t pinned_bytes =
        process_pinned_cache_bytes.load(std::memory_order_relaxed);
    if (device_bytes > std::numeric_limits<size_t>::max() - pinned_bytes) {
        return std::numeric_limits<size_t>::max();
    }
    return device_bytes + pinned_bytes;
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), capacity_(other.capacity_), device_(other.device_) {
        other.ptr_ = nullptr;
        other.capacity_ = 0;
        other.device_ = -1;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            capacity_ = other.capacity_;
            device_ = other.device_;
            other.ptr_ = nullptr;
            other.capacity_ = 0;
            other.device_ = -1;
        }
        return *this;
    }

    ~DeviceBuffer() {
        reset();
    }

    void* ensure(const size_t required_bytes, const char* context) {
        int current_device = -1;
        throw_if_failed(cudaGetDevice(&current_device), context);
        if (device_ != current_device) {
            reset();
        }
        if (required_bytes <= capacity_) {
            return ptr_;
        }
        reset();
        void* new_ptr = nullptr;
        throw_if_failed(cudaMalloc(&new_ptr, required_bytes), context);
        ptr_ = new_ptr;
        capacity_ = required_bytes;
        device_ = current_device;
        process_device_cache_bytes.fetch_add(capacity_, std::memory_order_relaxed);
        return ptr_;
    }

    void reset() noexcept {
        if (ptr_ == nullptr) {
            return;
        }
        int current_device = -1;
        const cudaError_t get_device_error = cudaGetDevice(&current_device);
        if (get_device_error == cudaSuccess && device_ >= 0 && current_device != device_) {
            cudaSetDevice(device_);
        }
        cudaFree(ptr_);
        if (get_device_error == cudaSuccess && device_ >= 0 && current_device != device_) {
            cudaSetDevice(current_device);
        }
        process_device_cache_bytes.fetch_sub(capacity_, std::memory_order_relaxed);
        ptr_ = nullptr;
        capacity_ = 0;
        device_ = -1;
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    void* ptr_ = nullptr;
    size_t capacity_ = 0;
    int device_ = -1;
};

class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : ptr_(other.ptr_), capacity_(other.capacity_) {
        other.ptr_ = nullptr;
        other.capacity_ = 0;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            capacity_ = other.capacity_;
            other.ptr_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }

    ~PinnedHostBuffer() {
        reset();
    }

    void* ensure(const size_t required_bytes, const char* context) {
        if (required_bytes <= capacity_) {
            return ptr_;
        }
        reset();
        void* new_ptr = nullptr;
        throw_if_failed(cudaMallocHost(&new_ptr, required_bytes), context);
        ptr_ = new_ptr;
        capacity_ = required_bytes;
        process_pinned_cache_bytes.fetch_add(capacity_, std::memory_order_relaxed);
        return ptr_;
    }

    void reset() noexcept {
        if (ptr_ == nullptr) {
            return;
        }
        cudaFreeHost(ptr_);
        process_pinned_cache_bytes.fetch_sub(capacity_, std::memory_order_relaxed);
        ptr_ = nullptr;
        capacity_ = 0;
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    void* ptr_ = nullptr;
    size_t capacity_ = 0;
};

class CudaStream {
public:
    CudaStream() = default;
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    ~CudaStream() {
        reset();
    }

    cudaStream_t ensure(const char* context) {
        int current_device = -1;
        throw_if_failed(cudaGetDevice(&current_device), context);
        if (device_ != current_device) {
            reset();
        }
        if (stream_ == nullptr) {
            throw_if_failed(
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), context);
            device_ = current_device;
        }
        return stream_;
    }

    void synchronize_noexcept() noexcept {
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }
    }

    void reset() noexcept {
        if (stream_ == nullptr) {
            return;
        }
        int current_device = -1;
        const cudaError_t get_device_error = cudaGetDevice(&current_device);
        if (get_device_error == cudaSuccess && device_ >= 0 && current_device != device_) {
            cudaSetDevice(device_);
        }
        cudaStreamDestroy(stream_);
        if (get_device_error == cudaSuccess && device_ >= 0 && current_device != device_) {
            cudaSetDevice(current_device);
        }
        stream_ = nullptr;
        device_ = -1;
    }

private:
    cudaStream_t stream_ = nullptr;
    int device_ = -1;
};

class HostIoWorkspace;

class HostIoWorkspaceSession {
public:
    HostIoWorkspaceSession(const HostIoWorkspaceSession&) = delete;
    HostIoWorkspaceSession& operator=(const HostIoWorkspaceSession&) = delete;
    HostIoWorkspaceSession(HostIoWorkspaceSession&& other) noexcept
        : workspace_(other.workspace_),
          next_device_slot_(other.next_device_slot_),
          next_pinned_slot_(other.next_pinned_slot_) {
        other.workspace_ = nullptr;
    }
    ~HostIoWorkspaceSession();

    void* device_buffer(size_t required_bytes, const char* context);
    void* pinned_buffer(size_t required_bytes, const char* context);
    cudaStream_t stream() const;
    void reset_after_error() noexcept;

private:
    friend class HostIoWorkspace;
    explicit HostIoWorkspaceSession(HostIoWorkspace* workspace)
        : workspace_(workspace) {}

    HostIoWorkspace* workspace_ = nullptr;
    size_t next_device_slot_ = 0;
    size_t next_pinned_slot_ = 0;
};

class HostIoWorkspace {
public:
    HostIoWorkspaceSession begin(const char* context) {
        if (active_) {
            throw std::runtime_error("nested CUDA host-I/O workspace use is unsupported");
        }
        int current_device = -1;
        throw_if_failed(cudaGetDevice(&current_device), context);
        if (device_ != current_device) {
            clear();
            device_ = current_device;
        }
        stream_.ensure(context);
        active_ = true;
        return HostIoWorkspaceSession(this);
    }

    void clear() noexcept {
        active_ = false;
        stream_.synchronize_noexcept();
        for (auto& buffer : device_buffers_) {
            buffer.reset();
        }
        for (auto& buffer : pinned_buffers_) {
            buffer.reset();
        }
        stream_.reset();
        device_buffers_.clear();
        pinned_buffers_.clear();
        device_ = -1;
    }

    size_t retained_device_bytes() const {
        size_t total = 0;
        for (const auto& buffer : device_buffers_) {
            total += buffer.capacity();
        }
        return total;
    }

    size_t retained_pinned_bytes() const {
        size_t total = 0;
        for (const auto& buffer : pinned_buffers_) {
            total += buffer.capacity();
        }
        return total;
    }

private:
    friend class HostIoWorkspaceSession;

    void* device_buffer(const size_t slot,
                        const size_t required_bytes,
                        const char* context) {
        if (slot >= device_buffers_.size()) {
            device_buffers_.resize(slot + 1);
        }
        return device_buffers_[slot].ensure(required_bytes, context);
    }

    void* pinned_buffer(const size_t slot,
                        const size_t required_bytes,
                        const char* context) {
        if (slot >= pinned_buffers_.size()) {
            pinned_buffers_.resize(slot + 1);
        }
        return pinned_buffers_[slot].ensure(required_bytes, context);
    }

    void finish() noexcept {
        active_ = false;
        const size_t retained = retained_device_bytes() + retained_pinned_bytes();
        if (retained > host_io_cache_per_thread_limit_bytes() ||
            process_host_io_cache_bytes() > host_io_cache_process_limit_bytes()) {
            clear();
        }
    }

    void reset_after_error() noexcept {
        active_ = false;
        clear();
    }

    CudaStream stream_;
    std::vector<DeviceBuffer> device_buffers_;
    std::vector<PinnedHostBuffer> pinned_buffers_;
    int device_ = -1;
    bool active_ = false;
};

inline HostIoWorkspaceSession::~HostIoWorkspaceSession() {
    if (workspace_ != nullptr) {
        workspace_->finish();
    }
}

inline void* HostIoWorkspaceSession::device_buffer(
    const size_t required_bytes,
    const char* context) {
    return workspace_->device_buffer(next_device_slot_++, required_bytes, context);
}

inline void* HostIoWorkspaceSession::pinned_buffer(
    const size_t required_bytes,
    const char* context) {
    return workspace_->pinned_buffer(next_pinned_slot_++, required_bytes, context);
}

inline cudaStream_t HostIoWorkspaceSession::stream() const {
    return workspace_->stream_.ensure("CUDA host-I/O workspace stream");
}

inline void HostIoWorkspaceSession::reset_after_error() noexcept {
    if (workspace_ != nullptr) {
        workspace_->reset_after_error();
        workspace_ = nullptr;
    }
}

inline thread_local HostIoWorkspace host_io_workspace;

inline HostIoWorkspaceSession acquire_host_io_workspace(const char* context) {
    return host_io_workspace.begin(context);
}

inline void clear_current_thread_host_io_workspace() noexcept {
    host_io_workspace.clear();
}

}  // namespace hnw::cuda
