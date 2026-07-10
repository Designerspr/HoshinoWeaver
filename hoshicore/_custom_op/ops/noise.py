"""Noise equalization custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Any, Callable

import cv2
import numpy as np

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import compiled_build_info as _compiled_build_info
from hoshicore._custom_op._dispatch import debug_enabled as _debug_enabled
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available


_debug_log = partial(debug_log, "noise")


def _validate_fill_local_mean_inputs(
    img: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    img_arr = np.asarray(img)
    mask_arr = np.asarray(mask)
    if img_arr.shape != mask_arr.shape:
        raise ValueError("noise_fill_local_mean: shape mismatch")
    if img_arr.ndim not in (2, 3):
        raise ValueError("noise_fill_local_mean: expected 2D or 3D array")
    if not np.issubdtype(img_arr.dtype, np.floating):
        raise ValueError("noise_fill_local_mean: floating-point array required")
    if mask_arr.dtype != np.bool_:
        mask_arr = mask_arr.astype(np.bool_, copy=False)
    if not img_arr.flags.c_contiguous:
        img_arr = np.ascontiguousarray(img_arr)
    if not mask_arr.flags.c_contiguous:
        mask_arr = np.ascontiguousarray(mask_arr)
    return img_arr, mask_arr


def _validate_equalization_param_inputs(
    max_img: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    n_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_arr = np.asarray(max_img)
    mean_arr = np.asarray(mean_img)
    std_arr = np.asarray(std_img)
    n_arr = np.asarray(n_img)
    if max_arr.shape != mean_arr.shape or max_arr.shape != std_arr.shape:
        raise ValueError("noise_equalization_params: shape mismatch")
    if max_arr.dtype != mean_arr.dtype or max_arr.dtype != std_arr.dtype:
        raise ValueError("noise_equalization_params: dtype mismatch")
    if max_arr.ndim not in (2, 3):
        raise ValueError("noise_equalization_params: expected 2D or 3D array")
    n_shape_matches = n_arr.shape == max_arr.shape
    n_shape_matches_pixels = (
        max_arr.ndim == 3
        and n_arr.ndim == 2
        and n_arr.shape == max_arr.shape[:2]
    )
    if not n_shape_matches and not n_shape_matches_pixels:
        raise ValueError("noise_equalization_params: n_img shape mismatch")
    if not np.issubdtype(max_arr.dtype, np.floating):
        raise ValueError("noise_equalization_params: floating-point arrays required")
    if not max_arr.flags.c_contiguous:
        max_arr = np.ascontiguousarray(max_arr)
    if not mean_arr.flags.c_contiguous:
        mean_arr = np.ascontiguousarray(mean_arr)
    if not std_arr.flags.c_contiguous:
        std_arr = np.ascontiguousarray(std_arr)
    if not n_arr.flags.c_contiguous:
        n_arr = np.ascontiguousarray(n_arr)
    return max_arr, mean_arr, std_arr, n_arr


def _validate_kernel_size(kernel_size: int) -> int:
    value = int(kernel_size)
    if value <= 0:
        raise ValueError("noise_fill_local_mean: kernel_size must be positive")
    return value


def _validate_equalize_noise_inputs(
    max_img: np.ndarray,
    filled_std_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_arr = np.asarray(max_img)
    filled_std_arr = np.asarray(filled_std_img)
    if max_arr.shape != filled_std_arr.shape:
        raise ValueError("equalize_noise_correct: shape mismatch")
    if max_arr.dtype != filled_std_arr.dtype:
        raise ValueError("equalize_noise_correct: dtype mismatch")
    if not np.issubdtype(max_arr.dtype, np.floating):
        raise ValueError("equalize_noise_correct: floating-point arrays required")
    if not max_arr.flags.c_contiguous:
        max_arr = np.ascontiguousarray(max_arr)
    if not filled_std_arr.flags.c_contiguous:
        filled_std_arr = np.ascontiguousarray(filled_std_arr)
    return max_arr, filled_std_arr


def noise_fill_local_mean_numpy(
    img: np.ndarray,
    mask: np.ndarray,
    kernel_size: int = 21,
) -> np.ndarray:
    img_arr, mask_arr = _validate_fill_local_mean_inputs(img, mask)
    size = _validate_kernel_size(kernel_size)
    valid = (~mask_arr).astype(np.float32)
    kernel = np.ones((size, size), dtype=np.float64)
    sum_valid = cv2.filter2D(
        img_arr * valid,
        -1,
        kernel,
        borderType=cv2.BORDER_REFLECT,
    )
    count_valid = cv2.filter2D(
        valid,
        -1,
        kernel,
        borderType=cv2.BORDER_REFLECT,
    )
    out = img_arr.copy()
    out[mask_arr] = (sum_valid / np.maximum(count_valid, 1e-8))[mask_arr]
    return out


def noise_equalization_params_numpy(
    max_img: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    n_img: np.ndarray,
    top_fraction: float = 0.02,
    sigma_reject: float = 3.0,
    minus_only: bool = False,
) -> tuple[float, float, np.ndarray] | None:
    max_arr, mean_arr, std_arr, n_arr = _validate_equalization_param_inputs(
        max_img,
        mean_img,
        std_img,
        n_img,
    )
    if not (0.0 <= float(top_fraction) <= 1.0):
        raise ValueError("noise_equalization_params: top_fraction must be in [0, 1]")
    if float(sigma_reject) < 0.0:
        raise ValueError("noise_equalization_params: sigma_reject must be non-negative")
    threshold = np.quantile(n_arr, 1.0 - float(top_fraction))
    bg_mask = n_arr >= threshold
    if not np.any(bg_mask):
        return None

    residual = (max_arr - mean_arr)[bg_mask]
    sigma_bg = std_arr[bg_mask]
    valid = sigma_bg > 0
    if not np.any(valid):
        return None

    r_valid = residual[valid]
    s_valid = sigma_bg[valid]
    c_n_eff = float(np.median(r_valid / s_valid))
    sigma_ref = 0.0 if minus_only else float(np.median(s_valid))

    channels = max_arr.shape[-1] if max_arr.ndim == 3 else 1
    squeeze_std = std_arr.reshape((-1, channels)).astype(np.float64, copy=False)
    mean_std = np.mean(squeeze_std, axis=0)
    std_std = np.std(squeeze_std, axis=0)
    std_mask = std_arr > (mean_std + float(sigma_reject) * std_std).reshape(
        (1,) * (std_arr.ndim - 1) + (channels,)
    )
    return sigma_ref, c_n_eff, std_mask


def noise_equalization_params_compiled(
    max_img: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    n_img: np.ndarray,
    top_fraction: float = 0.02,
    sigma_reject: float = 3.0,
    minus_only: bool = False,
) -> tuple[float, float, np.ndarray] | None:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "noise_equalization_params"):
        raise RuntimeError("compiled custom op backend is unavailable")
    max_arr, mean_arr, std_arr, n_arr = _validate_equalization_param_inputs(
        max_img,
        mean_img,
        std_img,
        n_img,
    )
    _apply_compiled_threads("noise_equalization_params", max_arr)
    result = module.noise_equalization_params(
        max_arr,
        mean_arr,
        std_arr,
        n_arr,
        float(top_fraction),
        float(sigma_reject),
        bool(minus_only),
    )
    if result is None:
        return None
    sigma_ref, c_n_eff, std_mask = result
    return float(sigma_ref), float(c_n_eff), np.asarray(std_mask, dtype=np.bool_)


def noise_fill_local_mean_compiled(
    img: np.ndarray,
    mask: np.ndarray,
    kernel_size: int = 21,
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "noise_fill_local_mean"):
        raise RuntimeError("compiled custom op backend is unavailable")
    img_arr, mask_arr = _validate_fill_local_mean_inputs(img, mask)
    size = _validate_kernel_size(kernel_size)
    _apply_compiled_threads("noise_fill_local_mean", img_arr)
    return module.noise_fill_local_mean(img_arr, mask_arr, size)


def _validate_highlight_preserve(highlight_preserve: float) -> float:
    value = float(highlight_preserve)
    if not (0.0 <= value < 1.0):
        raise ValueError("equalize_noise_correct: highlight_preserve must be in [0, 1)")
    return value


def equalize_noise_correct_numpy(
    max_img: np.ndarray,
    filled_std_img: np.ndarray,
    sigma_ref: float,
    c_n_eff: float,
    max_value: float,
    highlight_preserve: float,
) -> np.ndarray:
    max_arr, filled_std_arr = _validate_equalize_noise_inputs(max_img, filled_std_img)
    highlight_value = _validate_highlight_preserve(highlight_preserve)
    max_value_float = float(max_value)
    fix_strength = (
        (max_value_float * highlight_value - max_arr).clip(max=0)
        / (max_value_float * (1.0 - highlight_value))
        + 1.0
    )
    fixed_std_img = fix_strength * filled_std_arr
    corrected = max_arr - (fixed_std_img - float(sigma_ref)) * float(c_n_eff)
    return np.clip(corrected, a_min=0.0, a_max=max_value_float)


def equalize_noise_correct_compiled(
    max_img: np.ndarray,
    filled_std_img: np.ndarray,
    sigma_ref: float,
    c_n_eff: float,
    max_value: float,
    highlight_preserve: float,
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "equalize_noise_correct"):
        raise RuntimeError("compiled custom op backend is unavailable")
    max_arr, filled_std_arr = _validate_equalize_noise_inputs(max_img, filled_std_img)
    highlight_value = _validate_highlight_preserve(highlight_preserve)
    _apply_compiled_threads("equalize_noise_correct", max_arr)
    return module.equalize_noise_correct(
        max_arr,
        filled_std_arr,
        float(sigma_ref),
        float(c_n_eff),
        float(max_value),
        highlight_value,
    )


@lru_cache(maxsize=2)
def _select_equalize_noise_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray, np.ndarray, float, float, float, float], np.ndarray]]:
    available, compiled_error = _native_backend_available(
        "equalize_noise_correct",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", equalize_noise_correct_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", equalize_noise_correct_numpy


@lru_cache(maxsize=2)
def _select_fill_local_mean_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]]:
    available, compiled_error = _native_backend_available(
        "noise_fill_local_mean",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", noise_fill_local_mean_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", noise_fill_local_mean_numpy


@lru_cache(maxsize=2)
def _select_equalization_params_backend(
    preference: str,
) -> tuple[
    str,
    Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, bool],
        tuple[float, float, np.ndarray] | None,
    ],
]:
    available, compiled_error = _native_backend_available(
        "noise_equalization_params",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", noise_equalization_params_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", noise_equalization_params_numpy


def equalize_noise_correct(
    max_img: np.ndarray,
    filled_std_img: np.ndarray,
    sigma_ref: float,
    c_n_eff: float,
    max_value: float,
    highlight_preserve: float,
) -> np.ndarray:
    _, backend = _select_equalize_noise_backend(_fallback_preference())
    return backend(
        max_img,
        filled_std_img,
        sigma_ref,
        c_n_eff,
        max_value,
        highlight_preserve,
    )


def noise_fill_local_mean(
    img: np.ndarray,
    mask: np.ndarray,
    kernel_size: int = 21,
) -> np.ndarray:
    _, backend = _select_fill_local_mean_backend(_fallback_preference())
    return backend(img, mask, kernel_size)


def noise_equalization_params(
    max_img: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    n_img: np.ndarray,
    top_fraction: float = 0.02,
    sigma_reject: float = 3.0,
    minus_only: bool = False,
    estimate_method: str = "median",
) -> tuple[float, float, np.ndarray] | None:
    if estimate_method != "median":
        raise ValueError("noise_equalization_params: only median estimate_method is supported")
    _, backend = _select_equalization_params_backend(_fallback_preference())
    return backend(
        max_img,
        mean_img,
        std_img,
        n_img,
        top_fraction,
        sigma_reject,
        minus_only,
    )
