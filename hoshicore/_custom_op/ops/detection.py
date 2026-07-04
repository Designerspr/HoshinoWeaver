"""Star detection pixel-stage custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available
from hoshicore._custom_op.ops.wavelet import _wavelet_level
from hoshicore._custom_op.ops.wavelet import wavelet_dec_rec_core_numpy


_debug_log = partial(debug_log, "detection")


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
    image_arr, mask_u8 = _validate_image_mask(image, mask)
    mask_bool = mask_u8 > 0
    threshold = np.percentile(image_arr[mask_bool], 99.5)
    bw = ((image_arr > threshold) * mask_bool).astype(np.uint8) * 255
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _validate_component_inputs(
    image: np.ndarray,
    bw: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    image_arr = np.asarray(image, dtype=np.float64)
    bw_arr = np.asarray(bw)
    if image_arr.ndim != 2:
        raise ValueError(
            "star_detect_connected_components_candidates: image must be 2D")
    if bw_arr.ndim != 2:
        raise ValueError(
            "star_detect_connected_components_candidates: bw must be 2D")
    if image_arr.shape != bw_arr.shape:
        raise ValueError(
            "star_detect_connected_components_candidates: image and bw shapes must match"
        )
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError(
            "star_detect_connected_components_candidates: image height and width must be positive"
        )
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    bw_u8 = (bw_arr > 0).astype(np.uint8, copy=False)
    if not bw_u8.flags.c_contiguous:
        bw_u8 = np.ascontiguousarray(bw_u8)
    return image_arr, bw_u8


def star_detect_connected_components_candidates_compiled(
    image: np.ndarray,
    bw: np.ndarray,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(
            module, "star_detect_connected_components_candidates"):
        raise RuntimeError("compiled custom op backend is unavailable")
    image_arr, bw_u8 = _validate_component_inputs(image, bw)
    return module.star_detect_connected_components_candidates(image_arr, bw_u8)


@lru_cache(maxsize=2)
def _select_star_detect_connected_components_backend(
    preference: str,
) -> tuple[
    str,
    Callable[[np.ndarray, np.ndarray],
             tuple[
                 NDArray[np.float64],
                 NDArray[np.float64],
                 NDArray[np.float64],
                 NDArray[np.float64],
             ]],
]:
    available, compiled_error = _native_backend_available(
        "star_detect_connected_components_candidates",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", star_detect_connected_components_candidates_compiled

    reason = compiled_error or "compiled CC backend unavailable"
    _debug_log(f"compiled CC backend unavailable, reason: {reason}")
    raise RuntimeError(
        "star_detect_connected_components_candidates requires the compiled "
        f"experimental CC backend; use detect_star_points() for production fallback. {reason}"
    )


def star_detect_connected_components_candidates(
    image: np.ndarray,
    bw: np.ndarray,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    _, backend = _select_star_detect_connected_components_backend(
        _fallback_preference())
    return backend(image, bw)


def star_detect_bandpass_threshold_morph_numpy(
    image: np.ndarray,
    mask: np.ndarray,
    resize_factor: float,
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    image_arr, mask_u8 = _validate_image_mask(image, mask)
    level = _wavelet_level(float(resize_factor))
    small = cv2.resize(image_arr, None, fx=resize_factor, fy=resize_factor)
    reconstructed = wavelet_dec_rec_core_numpy(small, level)
    img_rec = cv2.resize(
        reconstructed, (image_arr.shape[1], image_arr.shape[0]))
    img_rec = img_rec * (mask_u8 > 0)
    bw = star_detect_threshold_morph_numpy(img_rec, mask_u8)
    return img_rec, bw


def _validate_full_detection_inputs(
    image: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None]:
    image_arr = np.asarray(image, dtype=np.float64)
    if image_arr.ndim != 2:
        raise ValueError("star_detect_full_connected_components: image must be 2D")
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError(
            "star_detect_full_connected_components: image height and width must be positive"
        )
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)

    if mask is None:
        return image_arr, None
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2 or mask_arr.shape != image_arr.shape:
        raise ValueError(
            "star_detect_full_connected_components: mask shape must match image"
        )
    mask_u8 = (mask_arr > 0).astype(np.uint8, copy=False)
    if not mask_u8.flags.c_contiguous:
        mask_u8 = np.ascontiguousarray(mask_u8)
    return image_arr, mask_u8


def _full_detection_kernel_params(
    shape: tuple[int, int],
    resize_factor: float,
    gaussian_ksize: int,
    sigma: float,
) -> tuple[int, int, int, NDArray[np.float64], NDArray[np.uint8]]:
    if resize_factor <= 0:
        raise ValueError(
            "star_detect_full_connected_components: resize_factor must be positive")
    if gaussian_ksize <= 0 or gaussian_ksize % 2 == 0:
        raise ValueError(
            "star_detect_full_connected_components: gaussian_ksize must be positive and odd"
        )
    height, width = shape
    small_height = max(1, int(round(height * resize_factor)))
    small_width = max(1, int(round(width * resize_factor)))
    level = _wavelet_level(float(resize_factor))
    gaussian_kernel = cv2.getGaussianKernel(
        int(gaussian_ksize), float(sigma), ktype=cv2.CV_64F
    ).reshape(-1)
    dilate_size = int(max(shape) * 0.003 * resize_factor)
    if dilate_size <= 0:
        raise RuntimeError(
            "star_detect_full_connected_components: image is too small for GPU mask dilation"
        )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
    ).astype(np.uint8, copy=False)
    if not gaussian_kernel.flags.c_contiguous:
        gaussian_kernel = np.ascontiguousarray(gaussian_kernel)
    if not dilate_kernel.flags.c_contiguous:
        dilate_kernel = np.ascontiguousarray(dilate_kernel)
    return small_height, small_width, level, gaussian_kernel, dilate_kernel


def star_detect_full_connected_components_compiled(
    image: np.ndarray,
    mask: np.ndarray | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(
            module, "star_detect_full_connected_components_core"):
        raise RuntimeError("compiled CUDA custom op backend is unavailable")
    image_arr, mask_u8 = _validate_full_detection_inputs(image, mask)
    small_height, small_width, level, gaussian_kernel, dilate_kernel = (
        _full_detection_kernel_params(
            image_arr.shape, resize_factor, gaussian_ksize, sigma)
    )
    return module.star_detect_full_connected_components_core(
        image_arr,
        mask_u8,
        small_height,
        small_width,
        level,
        gaussian_kernel,
        dilate_kernel,
    )


@lru_cache(maxsize=2)
def _select_star_detect_full_connected_components_backend(
    preference: str,
) -> tuple[
    str,
    Callable[[np.ndarray, np.ndarray | None, float, int, float],
             tuple[
                 NDArray[np.float64],
                 NDArray[np.float64],
                 NDArray[np.float64],
                 NDArray[np.float64],
             ]],
]:
    available, compiled_error = _native_backend_available(
        "star_detect_full_connected_components",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", star_detect_full_connected_components_compiled

    reason = compiled_error or "compiled full CUDA detector unavailable"
    _debug_log(f"compiled full detector unavailable, reason: {reason}")
    raise RuntimeError(
        "star_detect_full_connected_components requires the compiled CUDA "
        f"backend; use the original contour detector for production fallback. {reason}"
    )


def star_detect_full_connected_components(
    image: np.ndarray,
    mask: np.ndarray | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    image_arr, mask_u8 = _validate_full_detection_inputs(image, mask)
    _, backend = _select_star_detect_full_connected_components_backend(
        _fallback_preference())
    return backend(
        image_arr, mask_u8, resize_factor, int(gaussian_ksize), float(sigma))
