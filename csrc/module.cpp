#include "common/backend_info.h"
#include "common/cuda_error.h"
#include "common/star_detect_capacity.h"
#include "ops/cpu/alignment/alignment_ops.h"
#include "ops/cpu/calibration/calibration_ops.h"
#include "ops/cpu/detection/detection_ops.h"
#include "ops/cpu/fgp/fgp_ops.h"
#include "ops/cpu/filter/filter_ops.h"
#include "ops/cpu/max/max_ops.h"
#include "ops/cpu/median/median_ops.h"
#include "ops/cpu/noise/noise_ops.h"
#include "ops/cpu/remap/remap_ops.h"
#include "ops/cpu/sigma_clip/sigma_clip_chunk_ops.h"
#include "ops/cpu/star_shrink/star_shrink_ops.h"
#include "ops/cpu/wavelet/wavelet_ops.h"

#include <pybind11/pybind11.h>
#if HNW_ENABLE_CUDA
#include "ops/cuda/detection/star_detect_full_ops.h"
#include "ops/cuda/fgp/huber_weighted_chunk_ops.h"
#include "ops/cuda/remap/camera_model_remap_fused_ops.h"
#include "ops/cuda/sigma_clip/sigma_clip_fused_chunk_ops.h"
#include "ops/cuda/star_shrink/star_mask_ops.h"
#include "ops/cuda/star_shrink/star_shrink_process_ops.h"
#include "ops/cuda/wavelet/wavelet_cuda_ops.h"
#endif

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "Optional C++ ops for HoshinoWeaver";

    py::register_exception<hnw::CudaRuntimeUnavailableError>(m, "CudaRuntimeUnavailableError",
                                                             PyExc_RuntimeError);
    py::register_exception<hnw::CudaResourceExhaustedError>(m, "CudaResourceExhaustedError",
                                                            PyExc_RuntimeError);
#if HNW_ENABLE_CUDA
    py::register_exception<hnw::StarDetectCapacityError>(m, "StarDetectCapacityError",
                                                         PyExc_RuntimeError);
#endif

    bind_backend_info(m);
    bind_alignment_ops(m);
    bind_calibration_ops(m);
    bind_detection_ops(m);
    bind_filter_ops(m);
    bind_fgp_ops(m);
    bind_max_ops(m);
    bind_median_ops(m);
    bind_noise_ops(m);
    bind_remap_ops(m);
    bind_sigma_clip_chunk_ops(m);
    bind_star_shrink_ops(m);
    bind_wavelet_ops(m);
#if HNW_ENABLE_CUDA
    bind_camera_model_remap_fused_ops(m);
    bind_huber_weighted_chunk_cuda_ops(m);
    bind_sigma_clip_fused_chunk_cuda_ops(m);
    bind_star_mask_cuda_ops(m);
    bind_star_shrink_process_cuda_ops(m);
    bind_star_detect_full_ops(m);
    bind_wavelet_cuda_ops(m);
#endif
}
