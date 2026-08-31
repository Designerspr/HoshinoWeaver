#pragma once

#include <stdexcept>
#include <string>

namespace hnw {

class MetalRuntimeUnavailableError : public std::runtime_error {
public:
    explicit MetalRuntimeUnavailableError(const std::string& message)
        : std::runtime_error(message) {}
};

class MetalResourceExhaustedError : public std::runtime_error {
public:
    explicit MetalResourceExhaustedError(const std::string& message)
        : std::runtime_error(message) {}
};

} // namespace hnw
