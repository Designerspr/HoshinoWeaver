"""Wavelet bandpass custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from math import log
from typing import Callable

import cv2
import numpy as np
import pywt
from numpy.typing import NDArray

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available
from hoshicore._custom_op.backend_registry import resolve_after_cuda_failure
from hoshicore._custom_op.backend_registry import select_backend as _select_backend
from hoshicore._custom_op.cuda_memory import cuda_memory_estimate
from hoshicore._custom_op.cuda_memory import run_admitted_cuda as _run_admitted_cuda


_debug_log = partial(debug_log, "wavelet")
# 小图上 PyWavelets 更快；只在大图热点路径启用 compiled core。
MIN_COMPILED_WAVELET_PIXELS = 4_000_000
# CUDA host-IO core 在中等 resize 后图像上已快于 PyWavelets；更小图保留 numpy。
MIN_CUDA_WAVELET_PIXELS = 1_000_000


def _wavelet_level(resize_factor: float) -> int:
    if resize_factor <= 0:
        raise ValueError("wavelet_dec_rec: resize_factor must be positive")
    level = int(6 - log(1 / resize_factor, 2))
    if level <= 0:
        raise ValueError("wavelet_dec_rec: computed wavelet level must be positive")
    return level


def _as_float64_2d(image: np.ndarray) -> NDArray[np.float64]:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("wavelet_dec_rec: image must be 2D")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError("wavelet_dec_rec: image height and width must be positive")
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def wavelet_dec_rec_core_numpy(
    image: np.ndarray,
    level: int,
) -> NDArray[np.float64]:
    image_arr = _as_float64_2d(image)
    coeffs = pywt.wavedec2(image_arr, "db8", level=level)
    coeffs[0].fill(0)
    coeffs[-1][0].fill(0)
    coeffs[-1][1].fill(0)
    coeffs[-1][2].fill(0)
    return np.asarray(pywt.waverec2(coeffs, "db8"), dtype=np.float64)


def wavelet_dec_rec_core_compiled(
    image: np.ndarray,
    level: int,
) -> NDArray[np.float64]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "wavelet_dec_rec_cpu"):
        raise RuntimeError("compiled custom op backend is unavailable")
    image_arr = _as_float64_2d(image)
    _apply_compiled_threads("wavelet_dec_rec", image_arr)
    return module.wavelet_dec_rec_cpu(image_arr, int(level))


def wavelet_dec_rec_core_cuda(
    image: np.ndarray,
    level: int,
    *,
    memory_logical_op: str = "wavelet_dec_rec_cuda_core",
) -> NDArray[np.float64]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "wavelet_dec_rec_cuda_core"):
        raise RuntimeError("compiled CUDA custom op backend is unavailable")
    image_arr = _as_float64_2d(image)
    estimate = cuda_memory_estimate(
        memory_logical_op,
        height=image_arr.shape[0],
        width=image_arr.shape[1],
        level=int(level),
    )
    return _run_admitted_cuda(
        estimate, module.wavelet_dec_rec_cuda_core, image_arr, int(level))


@lru_cache(maxsize=2)
def _select_wavelet_dec_rec_cuda_core_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray, int], NDArray[np.float64]]]:
    available, compiled_error = _native_backend_available(
        "wavelet_dec_rec_cuda_core",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", wavelet_dec_rec_core_cuda

    if compiled_error:
        _debug_log(f"CUDA backend unavailable, reason: {compiled_error}")

    return "numpy", wavelet_dec_rec_core_numpy


def wavelet_dec_rec_core_cuda_or_numpy(
    image: np.ndarray,
    level: int,
) -> NDArray[np.float64]:
    backend_name, backend = _select_wavelet_dec_rec_cuda_core_backend(
        _fallback_preference())
    if backend_name != "compiled":
        return backend(image, level)
    try:
        return backend(image, level)
    except RuntimeError as exc:
        fallback_selection = resolve_after_cuda_failure(
            "wavelet_dec_rec_cuda_core",
            exc,
            load_module=_load_compiled_module_result,
            log=_debug_log,
        )
        if fallback_selection.native:
            raise RuntimeError(
                "wavelet_dec_rec_cuda_core selected an unsupported native fallback"
            )
        return wavelet_dec_rec_core_numpy(image, level)


@lru_cache(maxsize=2)
def _select_wavelet_dec_rec_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray, int], NDArray[np.float64]]]:
    selection = _select_backend(
        "wavelet_dec_rec",
        preference,
        load_module=_load_compiled_module_result,
    )
    if selection.reason:
        _debug_log(f"compiled backend unavailable, reason: {selection.reason}")
    return _wavelet_dec_rec_backend(selection)


def _wavelet_dec_rec_backend(
    selection: BackendSelection,
) -> tuple[str, Callable[[np.ndarray, int], NDArray[np.float64]]]:
    if not selection.native or selection.candidate is None:
        return "numpy", wavelet_dec_rec_core_numpy
    if selection.candidate.kernel_name == "wavelet_dec_rec_cuda_core":
        return "cuda", partial(
            wavelet_dec_rec_core_cuda,
            memory_logical_op=selection.candidate.logical_op,
        )
    if selection.candidate.kernel_name == "wavelet_dec_rec_cpu":
        return "compiled", wavelet_dec_rec_core_compiled
    raise RuntimeError(
        f"unknown wavelet_dec_rec backend candidate: {selection.candidate}"
    )


def wavelet_dec_rec_core(
    image: np.ndarray,
    level: int,
) -> NDArray[np.float64]:
    backend_name, backend = _select_wavelet_dec_rec_backend(_fallback_preference())
    if backend_name != "cuda":
        return backend(image, level)
    try:
        return backend(image, level)
    except RuntimeError as exc:
        fallback_selection = resolve_after_cuda_failure(
            "wavelet_dec_rec",
            exc,
            load_module=_load_compiled_module_result,
            log=_debug_log,
        )
    fallback_name, fallback_backend = _wavelet_dec_rec_backend(
        fallback_selection
    )
    if fallback_name == "cuda":
        raise RuntimeError("CUDA backend remained selected after runtime exclusion")
    return fallback_backend(image, level)


def wavelet_dec_rec(
    image: np.ndarray,
    resize_factor: float = 0.25,
) -> NDArray[np.float64]:
    image_arr = _as_float64_2d(image)
    level = _wavelet_level(resize_factor)
    small = cv2.resize(image_arr, None, fx=resize_factor, fy=resize_factor)
    if not small.flags.c_contiguous:
        small = np.ascontiguousarray(small)
    if small.size < MIN_CUDA_WAVELET_PIXELS:
        reconstructed = wavelet_dec_rec_core_numpy(small, level)
        return cv2.resize(reconstructed, (image_arr.shape[1], image_arr.shape[0]))
    # Mid-size tier: the standalone ``wavelet_dec_rec_cuda_core`` logical op runs
    # CUDA-or-NumPy and deliberately skips the OpenMP CPU kernel, whose launch
    # overhead does not pay off at this size. This branch is that op's only
    # consumer, so the op is reachable-but-internal here, not dead code.
    if small.size < MIN_COMPILED_WAVELET_PIXELS:
        reconstructed = wavelet_dec_rec_core_cuda_or_numpy(small, level)
    else:
        reconstructed = wavelet_dec_rec_core(small, level)
    return cv2.resize(reconstructed, (image_arr.shape[1], image_arr.shape[0]))
