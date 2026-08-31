#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <string>
#include <vector>

namespace hnw::metal {

class HostIOWorkspace {
public:
    static HostIOWorkspace& current();

    id<MTLDevice> device() const;
    id<MTLCommandQueue> command_queue() const;
    id<MTLBuffer> buffer(size_t bytes, const char* context);
    id<MTLComputePipelineState> pipeline(const char* function_name);

    void begin_operation(const char* logical_op);
    void finish_operation();
    void reset_after_error();
    bool clear();

    size_t retained_bytes() const;
    size_t last_logical_peak_bytes() const;
    const std::string& last_logical_op() const;

private:
    HostIOWorkspace();
    id<MTLLibrary> load_library();

    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> library_;
    NSMutableArray<id<MTLBuffer>>* buffers_;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines_;
    std::vector<size_t> buffer_sizes_;
    size_t next_buffer_ = 0;
    size_t logical_bytes_ = 0;
    size_t last_logical_peak_bytes_ = 0;
    std::string logical_op_;
    std::string last_logical_op_;
};

} // namespace hnw::metal
