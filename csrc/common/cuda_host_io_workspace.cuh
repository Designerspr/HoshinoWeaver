#pragma once

#include "common/compat.h"
#include "common/cuda_memory_ledger.cuh"
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

constexpr size_t DEFAULT_HOST_IO_CACHE_PER_THREAD_LIMIT_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr size_t DEFAULT_HOST_IO_CACHE_PROCESS_LIMIT_BYTES = 1024ULL * 1024ULL * 1024ULL;

inline std::atomic<size_t> process_device_cache_bytes{0};
inline std::atomic<size_t> process_pinned_cache_bytes{0};

inline size_t parse_cache_limit_bytes(const char* environment_name, const size_t default_bytes) {
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
        "HNW_CUDA_HOST_IO_CACHE_PER_THREAD_MB", DEFAULT_HOST_IO_CACHE_PER_THREAD_LIMIT_BYTES);
    return limit;
}

inline size_t host_io_cache_process_limit_bytes() {
    static const size_t limit = parse_cache_limit_bytes("HNW_CUDA_HOST_IO_CACHE_MB",
                                                        DEFAULT_HOST_IO_CACHE_PROCESS_LIMIT_BYTES);
    return limit;
}

inline size_t process_host_io_cache_bytes() {
    const size_t device_bytes = process_device_cache_bytes.load(std::memory_order_relaxed);
    const size_t pinned_bytes = process_pinned_cache_bytes.load(std::memory_order_relaxed);
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

    ~DeviceBuffer() { reset(); }

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

    size_t capacity() const { return capacity_; }

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

    ~PinnedHostBuffer() { reset(); }

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

    size_t capacity() const { return capacity_; }

private:
    void* ptr_ = nullptr;
    size_t capacity_ = 0;
};

class CudaStream {
public:
    CudaStream() = default;
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    ~CudaStream() { reset(); }

    cudaStream_t ensure(const char* context) {
        int current_device = -1;
        throw_if_failed(cudaGetDevice(&current_device), context);
        if (device_ != current_device) {
            reset();
        }
        if (stream_ == nullptr) {
            throw_if_failed(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), context);
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
class HostIoDeviceLease;

class HostIoWorkspaceSession {
public:
    HostIoWorkspaceSession(const HostIoWorkspaceSession&) = delete;
    HostIoWorkspaceSession& operator=(const HostIoWorkspaceSession&) = delete;
    HostIoWorkspaceSession(HostIoWorkspaceSession&& other) noexcept
        : workspace_(other.workspace_), next_device_slot_(other.next_device_slot_),
          next_pinned_slot_(other.next_pinned_slot_),
          device_request_bytes_(std::move(other.device_request_bytes_)),
          pinned_request_bytes_(std::move(other.pinned_request_bytes_)) {
        other.workspace_ = nullptr;
    }
    ~HostIoWorkspaceSession();

    void* device_buffer(size_t required_bytes, const char* context);
    void* pinned_buffer(size_t required_bytes, const char* context);
    HostIoDeviceLease device_lease(size_t required_bytes, const char* context);
    size_t device_mark() const;
    size_t pinned_mark() const;
    void rewind_device_buffers(size_t mark);
    void rewind_pinned_buffers(size_t mark);
    cudaStream_t stream() const;
    void reset_after_error() noexcept;

private:
    friend class HostIoWorkspace;
    explicit HostIoWorkspaceSession(HostIoWorkspace* workspace) : workspace_(workspace) {}

    HostIoWorkspace* workspace_ = nullptr;
    size_t next_device_slot_ = 0;
    size_t next_pinned_slot_ = 0;
    std::vector<size_t> device_request_bytes_;
    std::vector<size_t> pinned_request_bytes_;
};

class HostIoDeviceLease {
public:
    HostIoDeviceLease() = default;
    HostIoDeviceLease(const HostIoDeviceLease&) = delete;
    HostIoDeviceLease& operator=(const HostIoDeviceLease&) = delete;
    HostIoDeviceLease(HostIoDeviceLease&& other) noexcept
        : workspace_(other.workspace_), slot_(other.slot_), ptr_(other.ptr_),
          requested_bytes_(other.requested_bytes_) {
        other.workspace_ = nullptr;
        other.ptr_ = nullptr;
        other.requested_bytes_ = 0;
    }
    HostIoDeviceLease& operator=(HostIoDeviceLease&& other) noexcept {
        if (this != &other) {
            reset();
            workspace_ = other.workspace_;
            slot_ = other.slot_;
            ptr_ = other.ptr_;
            requested_bytes_ = other.requested_bytes_;
            other.workspace_ = nullptr;
            other.ptr_ = nullptr;
            other.requested_bytes_ = 0;
        }
        return *this;
    }
    ~HostIoDeviceLease() { reset(); }

    void* get() const { return ptr_; }
    void reset() noexcept;

private:
    friend class HostIoWorkspace;
    HostIoDeviceLease(HostIoWorkspace* workspace, size_t slot, void* ptr, size_t requested_bytes)
        : workspace_(workspace), slot_(slot), ptr_(ptr), requested_bytes_(requested_bytes) {}

    HostIoWorkspace* workspace_ = nullptr;
    size_t slot_ = 0;
    void* ptr_ = nullptr;
    size_t requested_bytes_ = 0;
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
        begin_cuda_memory_measurement(context);
        active_ = true;
        return HostIoWorkspaceSession(this);
    }

    void clear() noexcept {
        active_ = false;
        stream_.synchronize_noexcept();
        for (auto& buffer : device_buffers_) {
            buffer.reset();
        }
        for (auto& slot : scratch_device_buffers_) {
            slot.buffer.reset();
        }
        for (auto& buffer : pinned_buffers_) {
            buffer.reset();
        }
        stream_.reset();
        device_buffers_.clear();
        scratch_device_buffers_.clear();
        pinned_buffers_.clear();
        device_ = -1;
    }

    size_t retained_device_bytes() const {
        size_t total = 0;
        for (const auto& buffer : device_buffers_) {
            total += buffer.capacity();
        }
        for (const auto& slot : scratch_device_buffers_) {
            total += slot.buffer.capacity();
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
    friend class HostIoDeviceLease;

    struct ScratchDeviceSlot {
        DeviceBuffer buffer;
        bool in_use = false;
    };

    void* device_buffer(const size_t slot, const size_t required_bytes, const char* context) {
        if (slot >= device_buffers_.size()) {
            device_buffers_.resize(slot + 1);
        }
        return device_buffers_[slot].ensure(required_bytes, context);
    }

    void* pinned_buffer(const size_t slot, const size_t required_bytes, const char* context) {
        if (slot >= pinned_buffers_.size()) {
            pinned_buffers_.resize(slot + 1);
        }
        return pinned_buffers_[slot].ensure(required_bytes, context);
    }

    HostIoDeviceLease device_lease(const size_t required_bytes, const char* context) {
        size_t selected = scratch_device_buffers_.size();
        size_t selected_capacity = std::numeric_limits<size_t>::max();
        size_t grow_candidate = scratch_device_buffers_.size();
        size_t grow_candidate_capacity = 0;
        for (size_t slot = 0; slot < scratch_device_buffers_.size(); ++slot) {
            const auto& candidate = scratch_device_buffers_[slot];
            if (candidate.in_use) {
                continue;
            }
            if (candidate.buffer.capacity() >= required_bytes &&
                candidate.buffer.capacity() < selected_capacity) {
                selected = slot;
                selected_capacity = candidate.buffer.capacity();
            } else if (candidate.buffer.capacity() < required_bytes &&
                       candidate.buffer.capacity() >= grow_candidate_capacity) {
                grow_candidate = slot;
                grow_candidate_capacity = candidate.buffer.capacity();
            }
        }
        if (selected == scratch_device_buffers_.size() &&
            grow_candidate != scratch_device_buffers_.size()) {
            selected = grow_candidate;
        }
        if (selected == scratch_device_buffers_.size()) {
            scratch_device_buffers_.resize(selected + 1);
        }
        auto& slot = scratch_device_buffers_[selected];
        void* ptr = slot.buffer.ensure(required_bytes, context);
        slot.in_use = true;
        record_device_memory_acquire(required_bytes);
        return HostIoDeviceLease(this, selected, ptr, required_bytes);
    }

    void release_device_lease(const size_t slot) noexcept {
        if (slot < scratch_device_buffers_.size()) {
            scratch_device_buffers_[slot].in_use = false;
        }
    }

    void finish() noexcept {
        active_ = false;
        finish_cuda_memory_measurement();
        const size_t retained = retained_device_bytes() + retained_pinned_bytes();
        if (retained > host_io_cache_per_thread_limit_bytes() ||
            process_host_io_cache_bytes() > host_io_cache_process_limit_bytes()) {
            clear();
        }
    }

    void reset_after_error() noexcept {
        active_ = false;
        finish_cuda_memory_measurement();
        clear();
    }

    CudaStream stream_;
    std::vector<DeviceBuffer> device_buffers_;
    std::vector<ScratchDeviceSlot> scratch_device_buffers_;
    std::vector<PinnedHostBuffer> pinned_buffers_;
    int device_ = -1;
    bool active_ = false;
};

inline HostIoWorkspaceSession::~HostIoWorkspaceSession() {
    if (workspace_ != nullptr) {
        rewind_device_buffers(0);
        rewind_pinned_buffers(0);
        workspace_->finish();
    }
}

inline void* HostIoWorkspaceSession::device_buffer(const size_t required_bytes,
                                                   const char* context) {
    void* ptr = workspace_->device_buffer(next_device_slot_, required_bytes, context);
    if (next_device_slot_ >= device_request_bytes_.size()) {
        device_request_bytes_.resize(next_device_slot_ + 1, 0);
    }
    device_request_bytes_[next_device_slot_] = required_bytes;
    ++next_device_slot_;
    record_device_memory_acquire(required_bytes);
    return ptr;
}

inline void* HostIoWorkspaceSession::pinned_buffer(const size_t required_bytes,
                                                   const char* context) {
    void* ptr = workspace_->pinned_buffer(next_pinned_slot_, required_bytes, context);
    if (next_pinned_slot_ >= pinned_request_bytes_.size()) {
        pinned_request_bytes_.resize(next_pinned_slot_ + 1, 0);
    }
    pinned_request_bytes_[next_pinned_slot_] = required_bytes;
    ++next_pinned_slot_;
    record_pinned_memory_acquire(required_bytes);
    return ptr;
}

inline HostIoDeviceLease HostIoWorkspaceSession::device_lease(const size_t required_bytes,
                                                              const char* context) {
    return workspace_->device_lease(required_bytes, context);
}

inline size_t HostIoWorkspaceSession::device_mark() const {
    return next_device_slot_;
}

inline size_t HostIoWorkspaceSession::pinned_mark() const {
    return next_pinned_slot_;
}

inline void HostIoWorkspaceSession::rewind_device_buffers(const size_t mark) {
    if (mark > next_device_slot_) {
        throw std::invalid_argument("CUDA workspace device rewind mark is invalid");
    }
    for (size_t slot = mark; slot < next_device_slot_; ++slot) {
        record_device_memory_release(device_request_bytes_[slot]);
        device_request_bytes_[slot] = 0;
    }
    next_device_slot_ = mark;
}

inline void HostIoWorkspaceSession::rewind_pinned_buffers(const size_t mark) {
    if (mark > next_pinned_slot_) {
        throw std::invalid_argument("CUDA workspace pinned rewind mark is invalid");
    }
    for (size_t slot = mark; slot < next_pinned_slot_; ++slot) {
        record_pinned_memory_release(pinned_request_bytes_[slot]);
        pinned_request_bytes_[slot] = 0;
    }
    next_pinned_slot_ = mark;
}

inline cudaStream_t HostIoWorkspaceSession::stream() const {
    return workspace_->stream_.ensure("CUDA host-I/O workspace stream");
}

inline void HostIoWorkspaceSession::reset_after_error() noexcept {
    if (workspace_ != nullptr) {
        for (size_t slot = 0; slot < next_device_slot_; ++slot) {
            record_device_memory_release(device_request_bytes_[slot]);
        }
        for (size_t slot = 0; slot < next_pinned_slot_; ++slot) {
            record_pinned_memory_release(pinned_request_bytes_[slot]);
        }
        next_device_slot_ = 0;
        next_pinned_slot_ = 0;
        workspace_->reset_after_error();
        workspace_ = nullptr;
    }
}

inline void HostIoDeviceLease::reset() noexcept {
    if (workspace_ == nullptr) {
        return;
    }
    record_device_memory_release(requested_bytes_);
    workspace_->release_device_lease(slot_);
    workspace_ = nullptr;
    ptr_ = nullptr;
    requested_bytes_ = 0;
}

inline thread_local HostIoWorkspace host_io_workspace;

inline HostIoWorkspaceSession acquire_host_io_workspace(const char* context) {
    return host_io_workspace.begin(context);
}

inline void clear_current_thread_host_io_workspace() noexcept {
    host_io_workspace.clear();
}

} // namespace hnw::cuda
