"""FastGaussianParam custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Any, Callable

import numpy as np

from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import is_cuda_resource_exhausted_error
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available
from hoshicore._custom_op.backend_registry import resolve_after_resource_exhausted
from hoshicore._custom_op.backend_registry import resolve_after_runtime_unavailable
from hoshicore._custom_op.backend_registry import resolve_backend as _resolve_backend
from hoshicore._custom_op.cuda_memory import cuda_chunk_memory_model
from hoshicore._custom_op.cuda_memory import cuda_memory_admission


_debug_log = partial(debug_log, "fgp")


def _compiled_backend_available(logical_op: str, preference: str) -> tuple[bool, str | None]:
    return _native_backend_available(
        logical_op,
        preference,
        load_module=_load_compiled_module_result,
    )


def _validate_target(base: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_mu = np.asarray(base.sum_mu)
    square_sum = np.asarray(base.square_sum)
    count = np.asarray(base.n)
    if sum_mu.shape != square_sum.shape or sum_mu.shape != count.shape:
        raise ValueError("fgp_accumulate: base buffers shape mismatch")
    if not sum_mu.flags.c_contiguous or not square_sum.flags.c_contiguous or not count.flags.c_contiguous:
        raise ValueError("fgp_accumulate: base buffers must be C-contiguous")
    if not sum_mu.flags.writeable or not square_sum.flags.writeable or not count.flags.writeable:
        raise ValueError("fgp_accumulate: base buffers must be writeable")
    return sum_mu, square_sum, count


def _validate_buffers(
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    count: np.ndarray,
    *,
    op_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_arr = np.asarray(sum_mu)
    square_arr = np.asarray(square_sum)
    count_arr = np.asarray(count)
    if sum_arr.shape != square_arr.shape or sum_arr.shape != count_arr.shape:
        raise ValueError(f"{op_name}: accumulator shape mismatch")
    if not sum_arr.flags.c_contiguous or not square_arr.flags.c_contiguous or not count_arr.flags.c_contiguous:
        raise ValueError(f"{op_name}: accumulators must be C-contiguous")
    if not sum_arr.flags.writeable or not square_arr.flags.writeable or not count_arr.flags.writeable:
        raise ValueError(f"{op_name}: accumulators must be writeable")
    return sum_arr, square_arr, count_arr


def _validate_fresh(sum_mu: np.ndarray, fresh: np.ndarray, *, op_name: str = "fgp_accumulate") -> np.ndarray:
    fresh_arr = np.asarray(fresh)
    if fresh_arr.shape != sum_mu.shape:
        raise ValueError(f"{op_name}: shape mismatch")
    if not fresh_arr.flags.c_contiguous:
        fresh_arr = np.ascontiguousarray(fresh_arr)
    return fresh_arr


def _validate_integer_weight(weight: Any) -> int | None:
    if weight is None:
        return None
    if isinstance(weight, (int, np.integer)):
        value = int(weight)
        if value <= 0:
            raise ValueError("fgp_accumulate: weight must be positive")
        return value
    return None


def _validate_scalar_weight(weight: Any, *, op_name: str) -> float | None:
    if weight is None:
        return None
    if isinstance(weight, np.ndarray):
        if weight.ndim == 0:
            return float(weight.item())
        return None
    if np.isscalar(weight):
        return float(weight)
    raise TypeError(f"{op_name}: unsupported weight type")


def _python_fallback(base: Any, fresh: np.ndarray, weight: Any) -> Any:
    from hoshicore.component.data_container import FastGaussianParam

    patch = FastGaussianParam(fresh, source_dtype=fresh.dtype)
    if weight is not None:
        patch = patch * weight
    return base + patch


def _maybe_prepare_target(base: Any, weight: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_mu, square_sum, count = _validate_target(base)
    fresh_weight = _validate_integer_weight(weight)
    delta = 1 if fresh_weight is None else fresh_weight
    if getattr(base, "max_n", None) is not None:
        # 与原有 FastGaussianParam 路径保持一致：在真正写入前先处理计数和精度扩容。
        next_max_n = int(base.max_n) + int(delta)
        if next_max_n > base._safe_add_count():
            base.upscale()
            sum_mu, square_sum, count = _validate_target(base)
        from hoshicore.component.data_container import DTYPE_MAX_VALUE, DTYPE_UPSCALE_MAP

        if count.dtype in DTYPE_MAX_VALUE and next_max_n > DTYPE_MAX_VALUE[count.dtype]:
            if count.dtype in DTYPE_UPSCALE_MAP:
                base.n = count.astype(DTYPE_UPSCALE_MAP[count.dtype])
                sum_mu, square_sum, count = _validate_target(base)
        base.max_n = next_max_n
    return sum_mu, square_sum, count


def _validate_spatial_mask(fresh: np.ndarray, mask: np.ndarray, *, op_name: str) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if fresh.ndim == mask_arr.ndim + 1:
        if fresh.shape[:-1] != mask_arr.shape:
            raise ValueError(f"{op_name}: mask shape mismatch")
    elif fresh.ndim == mask_arr.ndim:
        if fresh.shape != mask_arr.shape:
            raise ValueError(f"{op_name}: mask shape mismatch")
    else:
        raise ValueError(f"{op_name}: mask ndim mismatch")
    if not mask_arr.flags.c_contiguous:
        mask_arr = np.ascontiguousarray(mask_arr)
    return mask_arr


def _validate_rejection_images(
    fresh: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    *,
    op_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    rej_high_arr = np.asarray(rej_high_img)
    rej_low_arr = np.asarray(rej_low_img)
    if rej_high_arr.shape != fresh.shape or rej_low_arr.shape != fresh.shape:
        raise ValueError(f"{op_name}: rejection image shape mismatch")
    if rej_high_arr.dtype != fresh.dtype or rej_low_arr.dtype != fresh.dtype:
        raise ValueError(f"{op_name}: rejection image dtype mismatch")
    if not rej_high_arr.flags.c_contiguous:
        rej_high_arr = np.ascontiguousarray(rej_high_arr)
    if not rej_low_arr.flags.c_contiguous:
        rej_low_arr = np.ascontiguousarray(rej_low_arr)
    return rej_high_arr, rej_low_arr


def _broadcast_mask(mask: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    if mask.ndim == fresh.ndim:
        return mask
    return mask[..., None]


def _validate_huber_target(base: Any) -> tuple[np.ndarray, np.ndarray]:
    weighted_sum = np.asarray(base.weighted_sum)
    weight_total = np.asarray(base.weight_total)
    if weighted_sum.shape != weight_total.shape:
        raise ValueError("huber_weighted_accumulate: accumulator shape mismatch")
    if not weighted_sum.flags.c_contiguous or not weight_total.flags.c_contiguous:
        raise ValueError("huber_weighted_accumulate: accumulators must be C-contiguous")
    if not weighted_sum.flags.writeable or not weight_total.flags.writeable:
        raise ValueError("huber_weighted_accumulate: accumulators must be writeable")
    if weighted_sum.dtype != np.float64 or weight_total.dtype != np.float64:
        raise ValueError("huber_weighted_accumulate: accumulators must be float64")
    return weighted_sum, weight_total


def _validate_ref_stats(
    fresh: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    *,
    op_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    ref_mean_arr = np.asarray(ref_mean, dtype=np.float32)
    ref_std_arr = np.asarray(ref_std, dtype=np.float32)
    if ref_mean_arr.shape != fresh.shape or ref_std_arr.shape != fresh.shape:
        raise ValueError(f"{op_name}: reference stats shape mismatch")
    if not ref_mean_arr.flags.c_contiguous:
        ref_mean_arr = np.ascontiguousarray(ref_mean_arr)
    if not ref_std_arr.flags.c_contiguous:
        ref_std_arr = np.ascontiguousarray(ref_std_arr)
    return ref_mean_arr, ref_std_arr


def fgp_accumulate_numpy(base: Any, fresh: np.ndarray, weight: Any = None,
                         skip_zero_rgb: bool = False) -> Any:
    sum_mu, square_sum, count = _maybe_prepare_target(base, weight)
    fresh_arr = _validate_fresh(sum_mu, fresh)
    int_weight = _validate_integer_weight(weight)

    if skip_zero_rgb and fresh_arr.ndim >= 3 and fresh_arr.shape[-1] >= 3:
        zero_mask = np.all(fresh_arr[..., :3] == 0, axis=-1, keepdims=True)
        active = ~np.broadcast_to(zero_mask, fresh_arr.shape)
    else:
        active = None

    if int_weight is None:
        if active is not None:
            sum_mu += np.where(active, fresh_arr, 0)
            square_sum += np.where(active, np.square(fresh_arr, dtype=square_sum.dtype), 0)
            count += np.where(active, 1, 0).astype(count.dtype)
        else:
            sum_mu += fresh_arr
            square_sum += np.square(fresh_arr, dtype=square_sum.dtype)
            count += 1
        return base

    if active is not None:
        sum_mu += np.where(active, np.multiply(fresh_arr, int_weight, dtype=sum_mu.dtype), 0)
        square_sum += np.where(active, np.multiply(
            np.square(fresh_arr, dtype=square_sum.dtype), int_weight, dtype=square_sum.dtype), 0)
        count += np.where(active, int_weight, 0).astype(count.dtype)
    else:
        sum_mu += np.multiply(fresh_arr, int_weight, dtype=sum_mu.dtype)
        square_sum += np.multiply(
            np.square(fresh_arr, dtype=square_sum.dtype),
            int_weight,
            dtype=square_sum.dtype,
        )
        count += int_weight
    return base


def fgp_accumulate_compiled(base: Any, fresh: np.ndarray, weight: Any = None,
                            skip_zero_rgb: bool = False) -> Any:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    sum_mu, square_sum, count = _maybe_prepare_target(base, weight)
    fresh_arr = _validate_fresh(sum_mu, fresh)
    int_weight = _validate_integer_weight(weight)
    _apply_compiled_threads("fgp_accumulate", fresh_arr)
    module.fgp_accumulate(sum_mu, square_sum, count, fresh_arr, int_weight,
                          skip_zero_rgb)
    return base


@lru_cache(maxsize=2)
def _select_fgp_backend(preference: str) -> tuple[str, Callable[[Any, np.ndarray, Any], Any]]:
    available, compiled_error = _compiled_backend_available("fgp_accumulate", preference)
    if available:
        return "compiled", fgp_accumulate_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", fgp_accumulate_numpy


def fgp_accumulate(base: Any, fresh: np.ndarray, weight: Any = None,
                   skip_zero_rgb: bool = False) -> Any:
    int_weight = _validate_integer_weight(weight)
    if weight is not None and int_weight is None:
        return _python_fallback(base, np.asarray(fresh), weight)
    _, backend = _select_fgp_backend(_fallback_preference())
    return backend(base, fresh, int_weight, skip_zero_rgb=skip_zero_rgb)


def huber_weighted_accumulate_numpy(
    base: Any,
    fresh: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weight: Any = None,
) -> Any:
    weighted_sum, weight_total = _validate_huber_target(base)
    fresh_arr = _validate_fresh(
        weighted_sum,
        fresh,
        op_name="huber_weighted_accumulate",
    )
    ref_mean_arr, ref_std_arr = _validate_ref_stats(
        fresh_arr,
        ref_mean,
        ref_std,
        op_name="huber_weighted_accumulate",
    )
    scalar_weight = _validate_scalar_weight(
        weight,
        op_name="huber_weighted_accumulate",
    )
    residual = (
        fresh_arr.astype(np.float32) - ref_mean_arr
    ) / (ref_std_arr + np.float32(1e-10))
    abs_residual = np.abs(residual)
    huber_weight = np.where(
        abs_residual <= huber_c,
        np.ones_like(abs_residual, dtype=np.float32),
        (huber_c / (abs_residual + np.float32(1e-10))).astype(np.float32),
    )
    if weight is not None:
        if scalar_weight is not None:
            huber_weight = huber_weight * scalar_weight
        else:
            huber_weight = huber_weight * np.asarray(weight)
    weighted_sum += np.multiply(fresh_arr, huber_weight, dtype=weighted_sum.dtype)
    weight_total += huber_weight.astype(weight_total.dtype, copy=False)
    return base


def huber_weighted_accumulate_compiled(
    base: Any,
    fresh: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weight: Any = None,
) -> Any:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    scalar_weight = _validate_scalar_weight(
        weight,
        op_name="huber_weighted_accumulate",
    )
    if weight is not None and scalar_weight is None:
        raise ValueError("huber_weighted_accumulate: compiled backend only supports scalar weight")
    weighted_sum, weight_total = _validate_huber_target(base)
    fresh_arr = _validate_fresh(
        weighted_sum,
        fresh,
        op_name="huber_weighted_accumulate",
    )
    ref_mean_arr, ref_std_arr = _validate_ref_stats(
        fresh_arr,
        ref_mean,
        ref_std,
        op_name="huber_weighted_accumulate",
    )
    _apply_compiled_threads("huber_weighted_accumulate", fresh_arr)
    module.huber_weighted_accumulate(
        weighted_sum,
        weight_total,
        fresh_arr,
        ref_mean_arr,
        ref_std_arr,
        float(huber_c),
        scalar_weight,
    )
    return base


@lru_cache(maxsize=2)
def _select_huber_backend(
    preference: str,
) -> tuple[str, Callable[[Any, np.ndarray, np.ndarray, np.ndarray, float, Any], Any]]:
    available, compiled_error = _compiled_backend_available(
        "huber_weighted_accumulate",
        preference,
    )
    if available:
        return "compiled", huber_weighted_accumulate_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", huber_weighted_accumulate_numpy


def huber_weighted_accumulate(
    base: Any,
    fresh: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weight: Any = None,
) -> Any:
    scalar_weight = _validate_scalar_weight(
        weight,
        op_name="huber_weighted_accumulate",
    )
    if weight is not None and scalar_weight is None:
        return huber_weighted_accumulate_numpy(
            base,
            fresh,
            ref_mean,
            ref_std,
            huber_c,
            weight,
        )
    _, backend = _select_huber_backend(_fallback_preference())
    return backend(base, fresh, ref_mean, ref_std, huber_c, scalar_weight)


def _validate_huber_chunk_inputs(
    stack: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    stack_arr = np.asarray(stack)
    if stack_arr.ndim != 2:
        raise ValueError("huber_weighted_chunk: stack must be 2D (n_frames, plane_size)")
    if stack_arr.shape[0] <= 0 or stack_arr.shape[1] <= 0:
        raise ValueError("huber_weighted_chunk: stack dimensions must be positive")
    if stack_arr.dtype not in (np.uint8, np.uint16):
        raise ValueError("huber_weighted_chunk: unsupported stack dtype; expected uint8/uint16")
    if not stack_arr.flags.c_contiguous:
        stack_arr = np.ascontiguousarray(stack_arr)

    ref_mean_arr = np.asarray(ref_mean, dtype=np.float64)
    ref_std_arr = np.asarray(ref_std, dtype=np.float64)
    if ref_mean_arr.shape != (stack_arr.shape[1],) or ref_std_arr.shape != (stack_arr.shape[1],):
        raise ValueError("huber_weighted_chunk: ref_mean/ref_std must match plane_size")
    if not ref_mean_arr.flags.c_contiguous:
        ref_mean_arr = np.ascontiguousarray(ref_mean_arr)
    if not ref_std_arr.flags.c_contiguous:
        ref_std_arr = np.ascontiguousarray(ref_std_arr)

    weights_arr = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape != (stack_arr.shape[0],):
            raise ValueError("huber_weighted_chunk: weights must have shape (n_frames,)")
        if not weights_arr.flags.c_contiguous:
            weights_arr = np.ascontiguousarray(weights_arr)
    return stack_arr, ref_mean_arr, ref_std_arr, weights_arr


def huber_weighted_chunk_numpy(
    stack: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stack_arr, ref_mean_arr, ref_std_arr, weights_arr = _validate_huber_chunk_inputs(
        stack, ref_mean, ref_std, weights)
    values = stack_arr.astype(np.float32, copy=False)
    residual = (values - ref_mean_arr.astype(np.float32)) / (
        ref_std_arr.astype(np.float32) + np.float32(1e-10))
    abs_residual = np.abs(residual)
    huber_weight = np.where(
        abs_residual <= np.float32(huber_c),
        np.ones_like(abs_residual, dtype=np.float32),
        (np.float32(huber_c) / (abs_residual + np.float32(1e-10))).astype(np.float32),
    )
    if weights_arr is not None:
        final_weight = np.multiply(
            huber_weight,
            weights_arr[:, np.newaxis],
            dtype=np.float64,
        )
    else:
        final_weight = huber_weight
    weighted_sum = np.sum(
        np.multiply(stack_arr, final_weight, dtype=np.float64),
        axis=0,
        dtype=np.float64,
    )
    weight_total = np.sum(final_weight, axis=0, dtype=np.float64)
    return weighted_sum, weight_total


def huber_weighted_chunk_compiled_cuda(
    stack: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "huber_weighted_chunk_cuda"):
        raise RuntimeError("compiled CUDA custom op backend is unavailable")
    stack_arr, ref_mean_arr, ref_std_arr, weights_arr = _validate_huber_chunk_inputs(
        stack, ref_mean, ref_std, weights)
    model = cuda_chunk_memory_model(
        "huber_weighted_chunk",
        n_frames=stack_arr.shape[0],
        row_bytes=stack_arr.shape[1] * stack_arr.dtype.itemsize,
        dtype_bytes=stack_arr.dtype.itemsize,
        include_weights=weights_arr is not None,
    )
    # The flattened plane is one modeled row, so estimate(1) covers this chunk.
    with cuda_memory_admission(model.estimate(1)) as admission:
        if not admission.granted:
            raise CustomOpResourceExhaustedError(
                "huber_weighted_chunk skipped CUDA because estimated peak "
                f"{admission.estimated_peak_bytes} bytes exceeds usable VRAM"
            )
        return module.huber_weighted_chunk_cuda(
            stack_arr,
            ref_mean_arr,
            ref_std_arr,
            float(huber_c),
            weights_arr,
        )


def _select_huber_chunk_backend(
    preference: str,
) -> BackendSelection:
    selection = _resolve_backend(
        "huber_weighted_chunk",
        preference,
        load_module=_load_compiled_module_result,
    )
    if selection.reason:
        _debug_log(f"compiled backend unavailable, reason: {selection.reason}")
    return selection


def _huber_chunk_backend(
    selection: BackendSelection,
) -> tuple[str, Callable[..., tuple[np.ndarray, np.ndarray]]]:
    if not selection.native or selection.candidate is None:
        return "numpy", huber_weighted_chunk_numpy
    if selection.candidate.kernel_name == "huber_weighted_chunk_cuda":
        return "cuda", huber_weighted_chunk_compiled_cuda
    raise RuntimeError(
        f"unknown huber_weighted_chunk backend candidate: {selection.candidate}"
    )


def huber_weighted_chunk(
    stack: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selection = _select_huber_chunk_backend(_fallback_preference())
    backend_name, backend = _huber_chunk_backend(selection)
    if backend_name != "cuda":
        return backend(stack, ref_mean, ref_std, huber_c, weights)
    try:
        return backend(stack, ref_mean, ref_std, huber_c, weights)
    except RuntimeError as exc:
        if is_cuda_resource_exhausted_error(exc):
            resolve_after_resource_exhausted(
                "huber_weighted_chunk",
                "cuda_host_io",
                exc,
                load_module=_load_compiled_module_result,
            )
            _debug_log(
                "compiled CUDA backend exhausted resources, falling back to "
                f"numpy: {exc}"
            )
        else:
            resolve_after_runtime_unavailable(
                "huber_weighted_chunk",
                "cuda_host_io",
                exc,
                load_module=_load_compiled_module_result,
            )
            _debug_log(
                "compiled CUDA backend unavailable at runtime, falling back "
                f"to numpy: {exc}"
            )
    return huber_weighted_chunk_numpy(stack, ref_mean, ref_std, huber_c, weights)


def try_huber_weighted_chunk_native(
    stack: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    selection = _select_huber_chunk_backend(_fallback_preference())
    backend_name, backend = _huber_chunk_backend(selection)
    if backend_name != "cuda":
        return None
    try:
        return backend(stack, ref_mean, ref_std, huber_c, weights)
    except RuntimeError as exc:
        if is_cuda_resource_exhausted_error(exc):
            resolve_after_resource_exhausted(
                "huber_weighted_chunk",
                "cuda_host_io",
                exc,
                load_module=_load_compiled_module_result,
            )
            _debug_log(f"compiled CUDA backend exhausted resources: {exc}")
        else:
            resolve_after_runtime_unavailable(
                "huber_weighted_chunk",
                "cuda_host_io",
                exc,
                load_module=_load_compiled_module_result,
            )
            _debug_log(f"compiled CUDA backend unavailable at runtime: {exc}")
        return None


def huber_weighted_chunk_native_available() -> bool:
    selection = _select_huber_chunk_backend(_fallback_preference())
    backend_name, _ = _huber_chunk_backend(selection)
    return backend_name == "cuda"


def fgp_masked_mean_merge_numpy(
    fresh: np.ndarray,
    mask: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="fgp_masked_mean_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="fgp_masked_mean_merge")
    mask_arr = _validate_spatial_mask(fresh_arr, mask, op_name="fgp_masked_mean_merge")
    active = _broadcast_mask(mask_arr, fresh_arr)
    if skip_zero_rgb and fresh_arr.ndim >= 3 and fresh_arr.shape[-1] >= 3:
        zero_mask = np.all(fresh_arr[..., :3] == 0, axis=-1, keepdims=True)
        active = active & ~np.broadcast_to(zero_mask, fresh_arr.shape)
    sum_arr += np.multiply(fresh_arr, active, dtype=sum_arr.dtype)
    square_arr += np.multiply(
        np.square(fresh_arr, dtype=square_arr.dtype),
        active,
        dtype=square_arr.dtype,
    )
    count_arr += active.astype(count_arr.dtype, copy=False)


def fgp_masked_mean_merge_compiled(
    fresh: np.ndarray,
    mask: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="fgp_masked_mean_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="fgp_masked_mean_merge")
    mask_arr = _validate_spatial_mask(fresh_arr, mask, op_name="fgp_masked_mean_merge")
    _apply_compiled_threads("fgp_masked_mean_merge", fresh_arr)
    module.fgp_masked_mean_merge(sum_arr, square_arr, count_arr, fresh_arr, mask_arr,
                                 skip_zero_rgb)


def fgp_masked_mean_merge(
    fresh: np.ndarray,
    mask: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    available, compiled_error = _compiled_backend_available(
        "fgp_masked_mean_merge",
        _fallback_preference(),
    )
    if available:
        fgp_masked_mean_merge_compiled(fresh, mask, sum_mu, square_sum, n,
                                       skip_zero_rgb)
        return
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    fgp_masked_mean_merge_numpy(fresh, mask, sum_mu, square_sum, n,
                                skip_zero_rgb)


def sigma_clip_fused_merge_numpy(
    fresh: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="sigma_clip_fused_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="sigma_clip_fused_merge")
    rej_high_arr, rej_low_arr = _validate_rejection_images(
        fresh_arr, rej_high_img, rej_low_img, op_name="sigma_clip_fused_merge")
    rejected = (fresh_arr < rej_low_arr) | (fresh_arr > rej_high_arr)
    if skip_zero_rgb and fresh_arr.ndim >= 3 and fresh_arr.shape[-1] >= 3:
        zero_mask = np.all(fresh_arr[..., :3] == 0, axis=-1, keepdims=True)
        rejected = rejected & ~np.broadcast_to(zero_mask, fresh_arr.shape)
    sum_arr += np.multiply(fresh_arr, rejected, dtype=sum_arr.dtype)
    square_arr += np.multiply(
        np.square(fresh_arr, dtype=square_arr.dtype),
        rejected,
        dtype=square_arr.dtype,
    )
    count_arr += rejected.astype(count_arr.dtype, copy=False)


def sigma_clip_fused_merge_compiled(
    fresh: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="sigma_clip_fused_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="sigma_clip_fused_merge")
    rej_high_arr, rej_low_arr = _validate_rejection_images(
        fresh_arr, rej_high_img, rej_low_img, op_name="sigma_clip_fused_merge")
    _apply_compiled_threads("sigma_clip_fused_merge", fresh_arr)
    module.sigma_clip_fused_merge(
        sum_arr,
        square_arr,
        count_arr,
        fresh_arr,
        rej_high_arr,
        rej_low_arr,
        skip_zero_rgb,
    )


def sigma_clip_fused_merge(
    fresh: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    available, compiled_error = _compiled_backend_available(
        "sigma_clip_fused_merge",
        _fallback_preference(),
    )
    if available:
        sigma_clip_fused_merge_compiled(
            fresh, rej_high_img, rej_low_img, sum_mu, square_sum, n,
            skip_zero_rgb)
        return
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    sigma_clip_fused_merge_numpy(
        fresh, rej_high_img, rej_low_img, sum_mu, square_sum, n,
        skip_zero_rgb)


def sigma_clip_fused_masked_merge_numpy(
    fresh: np.ndarray,
    mask: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="sigma_clip_fused_masked_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="sigma_clip_fused_masked_merge")
    mask_arr = _validate_spatial_mask(
        fresh_arr, mask, op_name="sigma_clip_fused_masked_merge")
    rej_high_arr, rej_low_arr = _validate_rejection_images(
        fresh_arr,
        rej_high_img,
        rej_low_img,
        op_name="sigma_clip_fused_masked_merge",
    )
    active = _broadcast_mask(mask_arr, fresh_arr)
    if skip_zero_rgb and fresh_arr.ndim >= 3 and fresh_arr.shape[-1] >= 3:
        zero_mask = np.all(fresh_arr[..., :3] == 0, axis=-1, keepdims=True)
        active = active & ~np.broadcast_to(zero_mask, fresh_arr.shape)
    rejected = active & ((fresh_arr < rej_low_arr) | (fresh_arr > rej_high_arr))
    sum_arr += np.multiply(fresh_arr, rejected, dtype=sum_arr.dtype)
    square_arr += np.multiply(
        np.square(fresh_arr, dtype=square_arr.dtype),
        rejected,
        dtype=square_arr.dtype,
    )
    count_arr += rejected.astype(count_arr.dtype, copy=False)


def sigma_clip_fused_masked_merge_compiled(
    fresh: np.ndarray,
    mask: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="sigma_clip_fused_masked_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="sigma_clip_fused_masked_merge")
    mask_arr = _validate_spatial_mask(
        fresh_arr, mask, op_name="sigma_clip_fused_masked_merge")
    rej_high_arr, rej_low_arr = _validate_rejection_images(
        fresh_arr,
        rej_high_img,
        rej_low_img,
        op_name="sigma_clip_fused_masked_merge",
    )
    _apply_compiled_threads("sigma_clip_fused_masked_merge", fresh_arr)
    module.sigma_clip_fused_masked_merge(
        sum_arr,
        square_arr,
        count_arr,
        fresh_arr,
        rej_high_arr,
        rej_low_arr,
        mask_arr,
        skip_zero_rgb,
    )


def sigma_clip_fused_masked_merge(
    fresh: np.ndarray,
    mask: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    available, compiled_error = _compiled_backend_available(
        "sigma_clip_fused_masked_merge",
        _fallback_preference(),
    )
    if available:
        sigma_clip_fused_masked_merge_compiled(
            fresh, mask, rej_high_img, rej_low_img, sum_mu, square_sum, n,
            skip_zero_rgb)
        return
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    sigma_clip_fused_masked_merge_numpy(
        fresh, mask, rej_high_img, rej_low_img, sum_mu, square_sum, n,
        skip_zero_rgb)
