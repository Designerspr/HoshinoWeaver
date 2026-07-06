"""Calibration custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import numpy as np

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available
from hoshicore.component.data_container import DTYPE_MAX_VALUE
from hoshicore.component.data_container import align_dtype_pair


_debug_log = partial(debug_log, "calibration")
_COMPILED_DTYPES = {
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("float32"),
    np.dtype("float64"),
}


def _prepare_aligned_inputs(
    frame: np.ndarray,
    reference: np.ndarray,
    frame_dtype: np.dtype,
    ref_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    frame_arr = np.asarray(frame)
    ref_arr = np.asarray(reference)
    frame_aligned, ref_aligned, output_dtype = align_dtype_pair(
        frame_arr,
        np.dtype(frame_dtype),
        ref_arr,
        np.dtype(ref_dtype),
    )
    return frame_aligned, ref_aligned, np.dtype(output_dtype)


def _as_compiled_inputs(
    frame: np.ndarray,
    reference: np.ndarray,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray] | None:
    if output_dtype not in _COMPILED_DTYPES:
        return None
    if np.shape(frame) != np.shape(reference):
        return None
    frame_arr = np.asarray(frame, dtype=output_dtype)
    ref_arr = np.asarray(reference, dtype=output_dtype)
    if not frame_arr.flags.c_contiguous:
        frame_arr = np.ascontiguousarray(frame_arr)
    if not ref_arr.flags.c_contiguous:
        ref_arr = np.ascontiguousarray(ref_arr)
    return frame_arr, ref_arr


def calibration_subtract_numpy(
    frame: np.ndarray,
    reference: np.ndarray,
    frame_dtype: np.dtype,
    ref_dtype: np.dtype,
) -> tuple[np.ndarray, np.dtype]:
    frame_aligned, ref_aligned, output_dtype = _prepare_aligned_inputs(
        frame,
        reference,
        frame_dtype,
        ref_dtype,
    )
    if output_dtype in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32")):
        calc_dtype = {
            np.dtype("uint8"): np.dtype("int16"),
            np.dtype("uint16"): np.dtype("int32"),
            np.dtype("uint32"): np.dtype("int64"),
        }[output_dtype]
        result = frame_aligned.astype(calc_dtype) - ref_aligned.astype(calc_dtype)
        np.clip(result, 0, DTYPE_MAX_VALUE[output_dtype], out=result)
        return result.astype(output_dtype), output_dtype

    result = frame_aligned - ref_aligned
    np.maximum(result, 0, out=result)
    return result.astype(output_dtype, copy=False), output_dtype


def calibration_divide_numpy(
    frame: np.ndarray,
    reference: np.ndarray,
    frame_dtype: np.dtype,
    ref_dtype: np.dtype,
) -> tuple[np.ndarray, np.dtype]:
    frame_aligned, ref_aligned, output_dtype = _prepare_aligned_inputs(
        frame,
        reference,
        frame_dtype,
        ref_dtype,
    )
    ref_f = ref_aligned.astype(np.float64)
    ref_mean = np.mean(ref_f)
    ref_safe = np.where(ref_f > 0, ref_f, 1.0)
    result = frame_aligned.astype(np.float64) / ref_safe * ref_mean
    if output_dtype in DTYPE_MAX_VALUE:
        np.clip(result, 0, DTYPE_MAX_VALUE[output_dtype], out=result)
    return result.astype(output_dtype), output_dtype


def calibration_subtract_compiled(
    frame: np.ndarray,
    reference: np.ndarray,
    frame_dtype: np.dtype,
    ref_dtype: np.dtype,
) -> tuple[np.ndarray, np.dtype]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "calibration_subtract"):
        raise RuntimeError("compiled custom op backend is unavailable")
    frame_aligned, ref_aligned, output_dtype = _prepare_aligned_inputs(
        frame,
        reference,
        frame_dtype,
        ref_dtype,
    )
    compiled_inputs = _as_compiled_inputs(frame_aligned, ref_aligned, output_dtype)
    if compiled_inputs is None:
        return calibration_subtract_numpy(frame, reference, frame_dtype, ref_dtype)
    frame_arr, ref_arr = compiled_inputs
    _apply_compiled_threads("calibration_subtract", frame_arr)
    return module.calibration_subtract(frame_arr, ref_arr), output_dtype


def calibration_divide_compiled(
    frame: np.ndarray,
    reference: np.ndarray,
    frame_dtype: np.dtype,
    ref_dtype: np.dtype,
) -> tuple[np.ndarray, np.dtype]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "calibration_divide"):
        raise RuntimeError("compiled custom op backend is unavailable")
    frame_aligned, ref_aligned, output_dtype = _prepare_aligned_inputs(
        frame,
        reference,
        frame_dtype,
        ref_dtype,
    )
    compiled_inputs = _as_compiled_inputs(frame_aligned, ref_aligned, output_dtype)
    if compiled_inputs is None:
        return calibration_divide_numpy(frame, reference, frame_dtype, ref_dtype)
    frame_arr, ref_arr = compiled_inputs
    _apply_compiled_threads("calibration_divide", frame_arr)
    return module.calibration_divide(frame_arr, ref_arr), output_dtype


@lru_cache(maxsize=2)
def _select_calibration_subtract_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray, np.ndarray, np.dtype, np.dtype], tuple[np.ndarray, np.dtype]]]:
    available, compiled_error = _native_backend_available(
        "calibration_subtract",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", calibration_subtract_compiled
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    return "numpy", calibration_subtract_numpy


@lru_cache(maxsize=2)
def _select_calibration_divide_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray, np.ndarray, np.dtype, np.dtype], tuple[np.ndarray, np.dtype]]]:
    available, compiled_error = _native_backend_available(
        "calibration_divide",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", calibration_divide_compiled
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    return "numpy", calibration_divide_numpy


def calibration_subtract(
    frame: np.ndarray,
    reference: np.ndarray,
    frame_dtype: np.dtype,
    ref_dtype: np.dtype,
) -> tuple[np.ndarray, np.dtype]:
    _, backend = _select_calibration_subtract_backend(_fallback_preference())
    return backend(frame, reference, np.dtype(frame_dtype), np.dtype(ref_dtype))


def calibration_divide(
    frame: np.ndarray,
    reference: np.ndarray,
    frame_dtype: np.dtype,
    ref_dtype: np.dtype,
) -> tuple[np.ndarray, np.dtype]:
    _, backend = _select_calibration_divide_backend(_fallback_preference())
    return backend(frame, reference, np.dtype(frame_dtype), np.dtype(ref_dtype))
