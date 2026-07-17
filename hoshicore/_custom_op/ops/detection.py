"""Star detection pixel-stage custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import CustomOpUnavailableError
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import is_cuda_resource_exhausted_error
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available
from hoshicore._custom_op.cuda_memory import cuda_memory_admission
from hoshicore._custom_op.cuda_memory import cuda_memory_estimate
from hoshicore._custom_op.ops.wavelet import _wavelet_level


_debug_log = partial(debug_log, "detection")


class StarDetectCapacityError(RuntimeError):
    """Raised when the CUDA connected-component algorithm exceeds its limits."""


def _is_native_star_detect_capacity_error(exc: RuntimeError) -> bool:
    if isinstance(exc, StarDetectCapacityError):
        return True
    module, _ = _load_compiled_module_result()
    capacity_type = (
        getattr(module, "StarDetectCapacityError", None)
        if module is not None
        else None
    )
    return capacity_type is not None and isinstance(exc, capacity_type)


def _validate_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    image_arr = np.asarray(image, dtype=np.float64)
    mask_arr = np.asarray(mask)
    if image_arr.ndim != 2:
        raise ValueError("star_detect_threshold_morph: image must be 2D")
    if mask_arr.ndim != 2:
        raise ValueError("star_detect_threshold_morph: mask must be 2D")
    if image_arr.shape != mask_arr.shape:
        raise ValueError(
            "star_detect_threshold_morph: image and mask shapes must match")
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError(
            "star_detect_threshold_morph: image height and width must be positive")
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    mask_u8 = (mask_arr > 0).astype(np.uint8, copy=False)
    if not mask_u8.flags.c_contiguous:
        mask_u8 = np.ascontiguousarray(mask_u8)
    return image_arr, mask_u8


def star_detect_threshold_morph_numpy(
    image: np.ndarray,
    mask: np.ndarray,
) -> NDArray[np.uint8]:
    """Return the NumPy/OpenCV pixel-stage baseline used by tests and benchmarks."""
    image_arr, mask_u8 = _validate_image_mask(image, mask)
    mask_bool = mask_u8 > 0
    threshold = np.percentile(image_arr[mask_bool], 99.5)
    bw = ((image_arr > threshold) * mask_bool).astype(np.uint8) * 255
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _validate_fused_pixel_component_inputs(
    image: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None]:
    image_arr = np.asarray(image, dtype=np.float64)
    if image_arr.ndim != 2:
        raise ValueError("star_detect_fused_pixel_components: image must be 2D")
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError(
            "star_detect_fused_pixel_components: image height and width must be positive"
        )
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)

    if mask is None:
        return image_arr, None
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2 or mask_arr.shape != image_arr.shape:
        raise ValueError(
            "star_detect_fused_pixel_components: mask shape must match image"
        )
    mask_u8 = (mask_arr > 0).astype(np.uint8, copy=False)
    if not mask_u8.flags.c_contiguous:
        mask_u8 = np.ascontiguousarray(mask_u8)
    return image_arr, mask_u8


def _fused_pixel_component_kernel_params(
    shape: tuple[int, int],
    resize_factor: float,
    gaussian_ksize: int,
    sigma: float,
) -> tuple[int, int, int, NDArray[np.float64]]:
    if resize_factor <= 0:
        raise ValueError(
            "star_detect_fused_pixel_components: resize_factor must be positive")
    if gaussian_ksize <= 0 or gaussian_ksize % 2 == 0:
        raise ValueError(
            "star_detect_fused_pixel_components: gaussian_ksize must be positive and odd"
        )
    height, width = shape
    small_height = max(1, int(round(height * resize_factor)))
    small_width = max(1, int(round(width * resize_factor)))
    level = _wavelet_level(float(resize_factor))
    gaussian_kernel = cv2.getGaussianKernel(
        int(gaussian_ksize), float(sigma), ktype=cv2.CV_64F
    ).reshape(-1)
    if not gaussian_kernel.flags.c_contiguous:
        gaussian_kernel = np.ascontiguousarray(gaussian_kernel)
    return small_height, small_width, level, gaussian_kernel


def _star_detect_fused_pixel_components_compiled_validated(
    image_arr: NDArray[np.float64],
    mask_u8: NDArray[np.uint8] | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint8],
]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(
            module, "star_detect_fused_pixel_components_cuda"):
        raise RuntimeError("compiled CUDA custom op backend is unavailable")
    small_height, small_width, level, gaussian_kernel = (
        _fused_pixel_component_kernel_params(
            image_arr.shape, resize_factor, gaussian_ksize, sigma)
    )
    estimate = cuda_memory_estimate(
        "star_detect_fused_pixel_components",
        height=image_arr.shape[0],
        width=image_arr.shape[1],
        small_height=small_height,
        small_width=small_width,
        level=level,
        gaussian_ksize=gaussian_kernel.size,
    )
    with cuda_memory_admission(estimate) as admission:
        if not admission.granted:
            raise CustomOpResourceExhaustedError(
                "star_detect_fused_pixel_components skipped CUDA because "
                f"estimated peak {admission.estimated_peak_bytes} bytes exceeds "
                f"usable VRAM (free={admission.free_bytes}, "
                f"reserved={admission.reserved_bytes}, "
                f"headroom={admission.headroom_bytes})"
            )
        return module.star_detect_fused_pixel_components_cuda(
            image_arr,
            mask_u8,
            small_height,
            small_width,
            level,
            gaussian_kernel,
        )


def star_detect_fused_pixel_components_compiled(
    image: np.ndarray,
    mask: np.ndarray | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint8],
]:
    image_arr, mask_u8 = _validate_fused_pixel_component_inputs(image, mask)
    return _star_detect_fused_pixel_components_compiled_validated(
        image_arr,
        mask_u8,
        resize_factor,
        gaussian_ksize,
        sigma,
    )


@lru_cache(maxsize=2)
def _select_star_detect_fused_pixel_components_backend(
    preference: str,
) -> tuple[
    str,
    Callable[[np.ndarray, np.ndarray | None, float, int, float],
             tuple[
                 NDArray[np.float64],
                 NDArray[np.float64],
                 NDArray[np.uint8],
             ]],
]:
    available, compiled_error = _native_backend_available(
        "star_detect_fused_pixel_components",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", _star_detect_fused_pixel_components_compiled_validated

    reason = compiled_error or "compiled CUDA pixel-component backend unavailable"
    _debug_log(f"compiled CUDA pixel-component backend unavailable, reason: {reason}")
    raise CustomOpUnavailableError(
        "star_detect_fused_pixel_components requires the compiled CUDA "
        f"backend; use the contour detector for production fallback. {reason}"
    )


def star_detect_fused_pixel_components(
    image: np.ndarray,
    mask: np.ndarray | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint8],
]:
    image_arr, mask_u8 = _validate_fused_pixel_component_inputs(image, mask)
    _, backend = _select_star_detect_fused_pixel_components_backend(
        _fallback_preference())
    try:
        return backend(
            image_arr, mask_u8, resize_factor, int(gaussian_ksize), float(sigma))
    except RuntimeError as exc:
        if _is_native_star_detect_capacity_error(exc):
            raise StarDetectCapacityError(str(exc)) from exc
        if is_cuda_resource_exhausted_error(exc):
            raise CustomOpResourceExhaustedError(
                "star_detect_fused_pixel_components CUDA resources exhausted: "
                f"{exc}"
            ) from exc
        if not is_cuda_runtime_unavailable_error(exc):
            raise
        raise CustomOpUnavailableError(
            "star_detect_fused_pixel_components CUDA backend unavailable "
            f"at runtime: {exc}"
        ) from exc
