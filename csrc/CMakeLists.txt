cmake_minimum_required(VERSION 3.23)

project(hnw_custom_ops LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(HnwOptions)
include(HnwCompiler)
include(HnwOpenMP)
include(HnwPython)
include(HnwCuda)

hnw_enable_cuda_language()
hnw_require_python_and_pybind11()

set(HNW_PACKAGE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hoshicore/_custom_op")

add_library(hnw_ops_common STATIC
    common/backend_info.cpp
)

add_library(hnw_ops_cpu_calibration STATIC
    ops/cpu/calibration/calibration_ops.cpp
)

add_library(hnw_ops_cpu_alignment STATIC
    ops/cpu/alignment/alignment_ops.cpp
    ops/cpu/alignment/matching_bidirectional_nearest_ops.cpp
)

add_library(hnw_ops_cpu_fgp STATIC
    ops/cpu/fgp/fgp_accumulate_ops.cpp
    ops/cpu/fgp/fgp_merge_ops.cpp
    ops/cpu/fgp/fgp_ops.cpp
    ops/cpu/fgp/huber_accumulate_ops.cpp
    ops/cpu/fgp/sigma_clip_merge_ops.cpp
)

add_library(hnw_ops_cpu_filter STATIC
    ops/cpu/filter/filter_ops.cpp
)

add_library(hnw_ops_cpu_max STATIC
    ops/cpu/max/max_ops.cpp
)

add_library(hnw_ops_cpu_median STATIC
    ops/cpu/median/median_ops.cpp
)

add_library(hnw_ops_cpu_noise STATIC
    ops/cpu/noise/noise_ops.cpp
)

add_library(hnw_ops_cpu_remap STATIC
    ops/cpu/remap/remap_ops.cpp
)

add_library(hnw_ops_cpu_sigma_clip STATIC
    ops/cpu/sigma_clip/sigma_clip_chunk_ops.cpp
)

add_library(hnw_ops_cpu_star_shrink STATIC
    ops/cpu/star_shrink/star_shrink_ops.cpp
)

add_library(hnw_ops_cpu_wavelet STATIC
    ops/cpu/wavelet/wavelet_ops.cpp
)

set(hnw_cpu_targets
    hnw_ops_common
    hnw_ops_cpu_calibration
    hnw_ops_cpu_alignment
    hnw_ops_cpu_fgp
    hnw_ops_cpu_filter
    hnw_ops_cpu_max
    hnw_ops_cpu_median
    hnw_ops_cpu_noise
    hnw_ops_cpu_remap
    hnw_ops_cpu_sigma_clip
    hnw_ops_cpu_star_shrink
    hnw_ops_cpu_wavelet
)

foreach(target IN LISTS hnw_cpu_targets)
    target_include_directories(
        "${target}"
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            ${HNW_PYBIND11_INCLUDE_DIRS}
    )
    target_link_libraries("${target}" PRIVATE pybind11::headers)
    hnw_apply_common_compile_options("${target}")
    hnw_link_openmp("${target}")
    hnw_enable_lto("${target}")
endforeach()

pybind11_add_module(_C MODULE
    NO_EXTRAS
    module.cpp
)

if(HNW_ENABLE_CUDA)
    target_sources(_C PRIVATE
        common/backend_cuda_info.cu
        # Alignment matching distance and bidirectional nearest-neighbor reduction.
        ops/cuda/alignment/matching_bidirectional_nearest_ops.cpp
        ops/cuda/alignment/matching_bidirectional_nearest_ops.cu
        # Fused camera-model remap.
        ops/cuda/remap/camera_model_remap_fused_ops.cpp
        ops/cuda/remap/camera_model_remap_fused_ops.cu
        # Wavelet CUDA backend and fused star-detection pixel/component stage.
        ops/cuda/wavelet/wavelet_cuda_ops.cpp
        ops/cuda/wavelet/wavelet_cuda_ops.cu
        ops/cuda/detection/star_detect_fused_pixel_components_ops.cpp
        ops/cuda/detection/star_detect_fused_pixel_components_ops.cu
        # Chunk stack kernels.
        ops/cuda/fgp/huber_weighted_chunk_ops.cpp
        ops/cuda/fgp/huber_weighted_chunk_ops.cu
        ops/cuda/sigma_clip/sigma_clip_fused_chunk_ops.cpp
        ops/cuda/sigma_clip/sigma_clip_fused_chunk_ops.cu
        # Star shrink fused processing.
        ops/cuda/star_shrink/star_mask_ops.cpp
        ops/cuda/star_shrink/star_mask_ops.cu
        ops/cuda/star_shrink/star_shrink_process_ops.cpp
        ops/cuda/star_shrink/star_shrink_process_ops.cu
    )
endif()

target_include_directories(
    _C
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        ${HNW_PYBIND11_INCLUDE_DIRS}
)
target_link_libraries(_C PRIVATE
    hnw_ops_common
    hnw_ops_cpu_calibration
    hnw_ops_cpu_alignment
    hnw_ops_cpu_fgp
    hnw_ops_cpu_filter
    hnw_ops_cpu_max
    hnw_ops_cpu_median
    hnw_ops_cpu_noise
    hnw_ops_cpu_remap
    hnw_ops_cpu_sigma_clip
    hnw_ops_cpu_star_shrink
    hnw_ops_cpu_wavelet
)
hnw_apply_common_compile_options(_C)
hnw_link_openmp(_C)
hnw_link_cuda_runtime(_C)
hnw_enable_lto(_C)

set_target_properties(_C PROPERTIES
    OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY "${HNW_PACKAGE_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${HNW_PACKAGE_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/archive"
)
