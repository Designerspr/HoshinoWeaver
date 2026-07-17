"""Public facade for custom-op APIs."""

from hoshicore._custom_op.ops.alignment import (
    extract_point_features,
    matching_cosine_bidirectional_nearest,
)
from hoshicore._custom_op.ops.calibration import (
    calibration_divide,
    calibration_subtract,
)
from hoshicore._custom_op.ops.detection import star_detect_fused_pixel_components
from hoshicore._custom_op.ops.fgp import (
    fgp_accumulate,
    fgp_masked_mean_merge,
    huber_weighted_chunk,
    huber_weighted_accumulate,
    sigma_clip_fused_masked_merge,
    sigma_clip_fused_merge,
)
from hoshicore._custom_op.ops.filter import median_filter_2d
from hoshicore._custom_op.ops.max import (
    build_info,
    custom_ops_available,
    max_combine,
    threshold_max_merge,
)
from hoshicore._custom_op.ops.median import median_reduce_chunk
from hoshicore._custom_op.ops.noise import (
    equalize_noise_correct,
    noise_equalization_params,
    noise_fill_local_mean,
)
from hoshicore._custom_op.ops.remap import camera_model_remap
from hoshicore._custom_op.ops.sigma_clip import (
    sigma_clip_iterative_chunk,
    sigma_clip_fused_chunk,
)
from hoshicore._custom_op.ops.star_shrink import (
    star_mask_dog,
    star_shrink_detect_mask,
    star_shrink_dog_process,
    star_shrink_process,
)
from hoshicore._custom_op.ops.wavelet import wavelet_dec_rec

__all__ = [
    "build_info",
    "calibration_divide",
    "calibration_subtract",
    "camera_model_remap",
    "custom_ops_available",
    "equalize_noise_correct",
    "extract_point_features",
    "fgp_accumulate",
    "fgp_masked_mean_merge",
    "huber_weighted_chunk",
    "huber_weighted_accumulate",
    "max_combine",
    "matching_cosine_bidirectional_nearest",
    "median_filter_2d",
    "median_reduce_chunk",
    "noise_equalization_params",
    "noise_fill_local_mean",
    "sigma_clip_fused_chunk",
    "sigma_clip_fused_masked_merge",
    "sigma_clip_fused_merge",
    "sigma_clip_iterative_chunk",
    "star_detect_fused_pixel_components",
    "star_mask_dog",
    "star_shrink_detect_mask",
    "star_shrink_dog_process",
    "star_shrink_process",
    "threshold_max_merge",
    "wavelet_dec_rec",
]
