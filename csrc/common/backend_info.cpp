#include "backend_info.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

const char* detect_arch() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__)
    return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

const char* detect_platform() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

const char* detect_compiler() {
#if defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

} // namespace

py::dict build_info_dict() {
    py::dict info;
    info["arch"] = detect_arch();
    info["platform"] = detect_platform();
    info["compiler"] = detect_compiler();
#ifdef _OPENMP
    info["openmp"] = true;
#else
    info["openmp"] = false;
#endif
#if HNW_ENABLE_OMP_SIMD
    info["omp_simd"] = true;
#else
    info["omp_simd"] = false;
#endif
#ifdef NDEBUG
    info["ndebug"] = true;
#else
    info["ndebug"] = false;
#endif
#if HNW_ENABLE_CUDA
    info["cuda"] = true;
#else
    info["cuda"] = false;
#endif
    info["openmp_max_threads"] = get_openmp_max_threads();
    return info;
}

py::dict cuda_memory_info_dict() {
#if HNW_ENABLE_CUDA
    return cuda_memory_info_cuda_dict();
#else
    py::dict info;
    info["available"] = false;
    info["status"] = "unavailable";
    info["reason_code"] = "cuda_not_built";
    info["category"] = "build";
    info["reason"] = "CUDA backend is not built";
    return info;
#endif
}

py::dict cuda_host_io_cache_info_dict() {
#if HNW_ENABLE_CUDA
    return cuda_host_io_cache_info_cuda_dict();
#else
    py::dict info;
    info["available"] = false;
    info["reason"] = "CUDA backend is not built";
    return info;
#endif
}

bool clear_cuda_host_io_cache_dict() {
#if HNW_ENABLE_CUDA
    return clear_cuda_host_io_cache_cuda();
#else
    return false;
#endif
}

int get_openmp_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

bool set_openmp_threads(int num_threads) {
#ifdef _OPENMP
    if (num_threads <= 0) {
        return false;
    }
    omp_set_num_threads(num_threads);
    return true;
#else
    (void)num_threads;
    return false;
#endif
}

void bind_backend_info(py::module_& m) {
    m.def("build_info", &build_info_dict, "Return compiled op backend metadata.");
    m.def("cuda_memory_info", &cuda_memory_info_dict,
          "Return CUDA device memory metadata when CUDA runtime is available.");
    m.def("cuda_host_io_cache_info", &cuda_host_io_cache_info_dict,
          "Return bounded CUDA host-I/O cache metadata for the current worker thread.");
    m.def("clear_cuda_host_io_cache", &clear_cuda_host_io_cache_dict,
          "Clear the CUDA host-I/O cache owned by the current worker thread.");
    m.def("get_openmp_max_threads", &get_openmp_max_threads,
          "Return the current OpenMP max thread count.");
    m.def("set_openmp_threads", &set_openmp_threads, py::arg("num_threads"),
          "Set the current OpenMP max thread count.");
}
