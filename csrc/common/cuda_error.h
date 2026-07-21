#pragma once

#include <stdexcept>
#include <string>

namespace hnw {

class CudaRuntimeUnavailableError : public std::runtime_error {
public:
    explicit CudaRuntimeUnavailableError(const std::string& message)
        : std::runtime_error(message) {}
};

class CudaResourceExhaustedError : public std::runtime_error {
public:
    explicit CudaResourceExhaustedError(const std::string& message) : std::runtime_error(message) {}
};

} // namespace hnw
