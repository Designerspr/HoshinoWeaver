"""Star shrink custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import numpy as np

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import is_cuda_resource_exhausted_error
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import resolve_after_resource_exhausted
from hoshicore._custom_op.backend_registry import resolve_after_runtime_unavailable
from hoshicore._custom_op.backend_registry import select_backend as _select_backend
from hoshicore._custom_op.cuda_memory import CudaMemoryEstimate
from hoshicore._custom_op.cuda_memory import cuda_memory_admission
from hoshicore._custom_op.cuda_memory import cuda_memory_estimate
from hoshicore.component.star_shrink import apply_mask, deringing, morph_shrink_luma


_debug_log = partial(debug_log, "star_shrink")
_SUPPORTED_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32))
_COMPILED_SUPPORTED_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16))


def _run_admitted_cuda(
    estimate: CudaMemoryEstimate,
    kernel: Callable[..., np.ndarray],
    *args: object,
) -> np.ndarray:
    with cuda_memory_admission(estimate) as admission:
        if not admission.granted:
            raise CustomOpResourceExhaustedError(
                f"{estimate.logical_op} skipped CUDA because estimated peak "
                f"{admission.estimated_peak_bytes} bytes exceeds usable VRAM"
            )
        return kernel(*args)


def _validate_dog_image(image: np.ndarray, logical_op: str) -> np.ndarray:
    image_arr = np.asarray(image)
    if image_arr.ndim not in (2, 3):
        raise ValueError(f"{logical_op}: image must have shape (H, W) or (H, W, C)")
    if image_arr.ndim == 3 and image_arr.shape[2] != 3:
        raise ValueError(f"{logical_op}: 3D image must have exactly 3 channels")
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError(f"{logical_op}: image height and width must be positive")
    if np.dtype(image_arr.dtype) not in _COMPILED_SUPPORTED_DTYPES:
        raise ValueError(f"{logical_op}: CUDA backend supports uint8/uint16 only")
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    return image_arr


def _dog_kernel_size(sigma: float, logical_op: str) -> int:
    sigma32 = np.float32(sigma)
    if not np.isfinite(sigma32) or sigma32 <= 0:
        raise ValueError(f"{logical_op}: sigma values must be positive and finite")
    radius = max(1, int(np.ceil(np.float32(3.0) * sigma32)))
    return 2 * radius + 1


def _star_shrink_estimate_args(image: np.ndarray) -> dict[str, int]:
    return {
        "height": image.shape[0],
        "width": image.shape[1],
        "channels": image.shape[2] if image.ndim == 3 else 1,
        "dtype_bytes": image.dtype.itemsize,
    }


def _validate_process_params(
    shrink_ksize: int,
    shrink_shape: str,
    shrink_times: int,
    shrink_ratio: float | None,
    deringing_ksize: int,
) -> tuple[int, str, int, float, int]:
    shrink_size = int(shrink_ksize)
    dering_size = int(deringing_ksize)
    if shrink_size <= 0 or shrink_size % 2 == 0:
        raise ValueError("star_shrink_process: shrink_ksize must be a positive odd value")
    if dering_size <= 0 or dering_size % 2 == 0:
        raise ValueError("star_shrink_process: deringing_ksize must be a positive odd value")
    times = int(shrink_times)
    if times <= 0:
        raise ValueError("star_shrink_process: shrink_times must be positive")
    shape = str(shrink_shape)
    if shape not in {"RECT", "CROSS", "CIRCLE"}:
        raise ValueError("star_shrink_process: unknown shrink_shape")
    ratio = 1.0 / times if shrink_ratio is None else float(shrink_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError("star_shrink_process: shrink_ratio must be in (0, 1]")
    return shrink_size, shape, times, ratio, dering_size


def _validate_inputs(
    image: np.ndarray,
    star_mask: np.ndarray,
    shrink_ksize: int,
    shrink_shape: str,
    shrink_times: int,
    shrink_ratio: float | None,
    deringing_ksize: int,
) -> tuple[np.ndarray, np.ndarray, int, str, int, float, int]:
    image_arr = np.asarray(image)
    mask_arr = np.asarray(star_mask)
    if image_arr.ndim not in (2, 3):
        raise ValueError("star_shrink_process: image must have shape (H, W) or (H, W, C)")
    if image_arr.ndim == 3 and image_arr.shape[2] != 3:
        raise ValueError("star_shrink_process: 3D image must have exactly 3 channels")
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError("star_shrink_process: image height and width must be positive")
    if mask_arr.shape != image_arr.shape[:2]:
        raise ValueError("star_shrink_process: star_mask must have shape (H, W)")
    if np.dtype(image_arr.dtype) not in _SUPPORTED_DTYPES:
        raise ValueError("star_shrink_process: image dtype must be uint8, uint16, or float32")

    shrink_size, shape, times, ratio, dering_size = _validate_process_params(
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )

    if mask_arr.dtype != np.uint8:
        mask_arr = mask_arr.astype(np.uint8, copy=False)
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    if not mask_arr.flags.c_contiguous:
        mask_arr = np.ascontiguousarray(mask_arr)
    return image_arr, mask_arr, shrink_size, shape, times, ratio, dering_size


def star_shrink_process_numpy(
    image: np.ndarray,
    star_mask: np.ndarray,
    shrink_ksize: int,
    shrink_shape: str,
    shrink_times: int,
    shrink_ratio: float | None,
    deringing_ksize: int,
) -> np.ndarray:
    image_arr, mask_arr, shrink_size, shape, times, ratio, dering_size = _validate_inputs(
        image,
        star_mask,
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )
    shrunk = morph_shrink_luma(
        image_arr,
        ksize=shrink_size,
        shape=shape,
        times=times,
        ratio=ratio,
    )
    processed = deringing(image_arr, shrunk, algo="mean", ksize=dering_size)
    return apply_mask(image_arr, processed, mask_arr)


def _star_shrink_process_compiled_kernel(
    kernel_name: str,
    image: np.ndarray,
    star_mask: np.ndarray,
    shrink_ksize: int,
    shrink_shape: str,
    shrink_times: int,
    shrink_ratio: float | None,
    deringing_ksize: int,
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, kernel_name):
        raise RuntimeError("compiled custom op backend is unavailable")
    image_arr, mask_arr, shrink_size, shape, times, ratio, dering_size = _validate_inputs(
        image,
        star_mask,
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )
    if np.dtype(image_arr.dtype) not in _COMPILED_SUPPORTED_DTYPES:
        raise ValueError("star_shrink_process: compiled backend supports uint8/uint16 only")
    _apply_compiled_threads("star_shrink_process", image_arr)
    kernel = getattr(module, kernel_name)
    kernel_args = (
        image_arr,
        mask_arr,
        shrink_size,
        shape,
        times,
        ratio,
        dering_size,
    )
    if kernel_name != "star_shrink_process_cuda":
        return kernel(*kernel_args)

    estimate = cuda_memory_estimate(
        "star_shrink_process",
        **_star_shrink_estimate_args(image_arr),
    )
    return _run_admitted_cuda(estimate, kernel, *kernel_args)


def star_shrink_process_compiled(
    image: np.ndarray,
    star_mask: np.ndarray,
    shrink_ksize: int,
    shrink_shape: str,
    shrink_times: int,
    shrink_ratio: float | None,
    deringing_ksize: int,
) -> np.ndarray:
    return _star_shrink_process_compiled_kernel(
        "star_shrink_process",
        image,
        star_mask,
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )


def star_shrink_process_compiled_cuda(
    image: np.ndarray,
    star_mask: np.ndarray,
    shrink_ksize: int,
    shrink_shape: str,
    shrink_times: int,
    shrink_ratio: float | None,
    deringing_ksize: int,
) -> np.ndarray:
    return _star_shrink_process_compiled_kernel(
        "star_shrink_process_cuda",
        image,
        star_mask,
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )


@lru_cache(maxsize=2)
def _select_star_shrink_process_backend(preference: str) -> BackendSelection:
    return _select_backend(
        "star_shrink_process",
        preference,
        load_module=_load_compiled_module_result,
    )


def _star_shrink_process_backend(
    selection: BackendSelection,
) -> tuple[str, Callable[..., np.ndarray]]:
    if not selection.native or selection.candidate is None:
        return "numpy", star_shrink_process_numpy
    if selection.candidate.kernel_name == "star_shrink_process_cuda":
        return "cuda", star_shrink_process_compiled_cuda
    if selection.candidate.kernel_name == "star_shrink_process":
        return "cpu", star_shrink_process_compiled
    raise RuntimeError(
        f"unknown star_shrink_process backend candidate: {selection.candidate}"
    )


def star_shrink_process(
    image: np.ndarray,
    star_mask: np.ndarray,
    shrink_ksize: int,
    shrink_shape: str,
    shrink_times: int,
    shrink_ratio: float | None,
    deringing_ksize: int,
) -> np.ndarray:
    image_arr = np.asarray(image)
    if np.dtype(image_arr.dtype) not in _COMPILED_SUPPORTED_DTYPES:
        return star_shrink_process_numpy(
            image,
            star_mask,
            shrink_ksize,
            shrink_shape,
            shrink_times,
            shrink_ratio,
            deringing_ksize,
        )

    selection = _select_star_shrink_process_backend(_fallback_preference())
    if selection.reason:
        _debug_log(f"compiled backend unavailable, reason: {selection.reason}")
    backend_name, backend = _star_shrink_process_backend(selection)
    kernel_args = (
        image,
        star_mask,
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )
    if backend_name != "cuda":
        return backend(*kernel_args)

    try:
        return backend(*kernel_args)
    except RuntimeError as exc:
        if is_cuda_resource_exhausted_error(exc):
            fallback_selection = resolve_after_resource_exhausted(
                "star_shrink_process",
                "cuda_host_io",
                exc,
                load_module=_load_compiled_module_result,
            )
            _debug_log(
                "CUDA backend exhausted resources, falling back to the next "
                f"backend: {exc}"
            )
        else:
            fallback_selection = resolve_after_runtime_unavailable(
                "star_shrink_process",
                "cuda_host_io",
                exc,
                load_module=_load_compiled_module_result,
            )
            _debug_log(
                "CUDA backend unavailable at runtime, falling back to the next "
                f"backend: {exc}"
            )

    fallback_name, fallback_backend = _star_shrink_process_backend(
        fallback_selection
    )
    if fallback_name == "cuda":
        raise RuntimeError("CUDA backend remained selected after runtime exclusion")
    return fallback_backend(*kernel_args)


StarShrinkProcessCallable = Callable[
    [np.ndarray, np.ndarray, int, str, int, float | None, int],
    np.ndarray,
]


def star_shrink_detect_mask_numpy(
    image: np.ndarray,
    ksize: int = 13,
    med_algo: str = "median",
    threshold_ratio: int | float = 5,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
) -> np.ndarray:
    from hoshicore.component.star_detect import detect_starmask_by_threshold

    return detect_starmask_by_threshold(
        image,
        ksize=ksize,
        med_algo=med_algo,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        dilate_ksize=dilate_ksize,
    )


def star_shrink_detect_mask_compiled(
    image: np.ndarray,
    ksize: int = 13,
    med_algo: str = "median",
    threshold_ratio: int | float = 5,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
) -> np.ndarray:
    if med_algo != "median":
        raise ValueError("star_shrink_detect_mask: compiled backend supports med_algo='median' only")
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "star_shrink_detect_mask"):
        raise RuntimeError("compiled custom op backend is unavailable")
    image_arr = np.asarray(image)
    if image_arr.ndim not in (2, 3):
        raise ValueError("star_shrink_detect_mask: image must have shape (H, W) or (H, W, C)")
    if image_arr.ndim == 3 and image_arr.shape[2] != 3:
        raise ValueError("star_shrink_detect_mask: 3D image must have exactly 3 channels")
    if np.dtype(image_arr.dtype) not in _COMPILED_SUPPORTED_DTYPES:
        raise ValueError("star_shrink_detect_mask: compiled backend supports uint8/uint16 only")
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    _apply_compiled_threads("star_shrink_detect_mask", image_arr)
    return module.star_shrink_detect_mask(
        image_arr,
        int(ksize),
        float(threshold_ratio),
        int(open_ksize),
        int(dilate_ksize),
    )


@lru_cache(maxsize=2)
def _select_star_shrink_detect_mask_backend(preference: str) -> BackendSelection:
    return _select_backend(
        "star_shrink_detect_mask",
        preference,
        load_module=_load_compiled_module_result,
    )


def star_shrink_detect_mask(
    image: np.ndarray,
    ksize: int = 13,
    med_algo: str = "median",
    threshold_ratio: int | float = 5,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
) -> np.ndarray:
    image_arr = np.asarray(image)
    if med_algo != "median" or np.dtype(image_arr.dtype) not in _COMPILED_SUPPORTED_DTYPES:
        return star_shrink_detect_mask_numpy(
            image,
            ksize=ksize,
            med_algo=med_algo,
            threshold_ratio=threshold_ratio,
            open_ksize=open_ksize,
            dilate_ksize=dilate_ksize,
        )

    selection = _select_star_shrink_detect_mask_backend(_fallback_preference())
    if not selection.native or selection.candidate is None:
        if selection.reason:
            _debug_log(f"detect mask compiled backend unavailable, reason: {selection.reason}")
        return star_shrink_detect_mask_numpy(
            image,
            ksize=ksize,
            med_algo=med_algo,
            threshold_ratio=threshold_ratio,
            open_ksize=open_ksize,
            dilate_ksize=dilate_ksize,
        )

    if selection.candidate.backend == "openmp_cpu":
        return star_shrink_detect_mask_compiled(
            image,
            ksize=ksize,
            med_algo=med_algo,
            threshold_ratio=threshold_ratio,
            open_ksize=open_ksize,
            dilate_ksize=dilate_ksize,
        )

    return star_shrink_detect_mask_numpy(
        image,
        ksize=ksize,
        med_algo=med_algo,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        dilate_ksize=dilate_ksize,
    )


def star_mask_dog_numpy(
    image: np.ndarray,
    sigma_small: float = 1.5,
    sigma_large: float = 12.0,
    threshold_ratio: int | float = 3,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
) -> np.ndarray:
    from hoshicore.component.star_detect import detect_starmask_by_dog

    return detect_starmask_by_dog(
        image,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        dilate_ksize=dilate_ksize,
    )


def star_mask_dog_compiled_cuda(
    image: np.ndarray,
    sigma_small: float = 1.5,
    sigma_large: float = 12.0,
    threshold_ratio: int | float = 3,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "star_mask_dog_cuda"):
        raise RuntimeError("compiled custom op backend is unavailable")
    image_arr = _validate_dog_image(image, "star_mask_dog")
    small_kernel_size = _dog_kernel_size(sigma_small, "star_mask_dog")
    large_kernel_size = _dog_kernel_size(sigma_large, "star_mask_dog")
    estimate = cuda_memory_estimate(
        "star_mask_dog",
        **_star_shrink_estimate_args(image_arr),
        small_kernel_size=small_kernel_size,
        large_kernel_size=large_kernel_size,
    )
    kernel_args = (
        image_arr,
        float(sigma_small),
        float(sigma_large),
        float(threshold_ratio),
        int(open_ksize),
        int(dilate_ksize),
    )
    return _run_admitted_cuda(
        estimate,
        module.star_mask_dog_cuda,
        *kernel_args,
    )


@lru_cache(maxsize=2)
def _select_star_mask_dog_backend(preference: str) -> BackendSelection:
    return _select_backend(
        "star_mask_dog",
        preference,
        load_module=_load_compiled_module_result,
    )


def star_mask_dog(
    image: np.ndarray,
    sigma_small: float = 1.5,
    sigma_large: float = 12.0,
    threshold_ratio: int | float = 3,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
) -> np.ndarray:
    image_arr = np.asarray(image)
    if np.dtype(image_arr.dtype) not in _COMPILED_SUPPORTED_DTYPES:
        return star_mask_dog_numpy(
            image,
            sigma_small=sigma_small,
            sigma_large=sigma_large,
            threshold_ratio=threshold_ratio,
            open_ksize=open_ksize,
            dilate_ksize=dilate_ksize,
        )

    selection = _select_star_mask_dog_backend(_fallback_preference())
    if not selection.native or selection.candidate is None:
        if selection.reason:
            _debug_log(f"DoG mask compiled backend unavailable, reason: {selection.reason}")
        return star_mask_dog_numpy(
            image,
            sigma_small=sigma_small,
            sigma_large=sigma_large,
            threshold_ratio=threshold_ratio,
            open_ksize=open_ksize,
            dilate_ksize=dilate_ksize,
        )

    if selection.candidate.backend == "cuda_host_io":
        try:
            return star_mask_dog_compiled_cuda(
                image,
                sigma_small=sigma_small,
                sigma_large=sigma_large,
                threshold_ratio=threshold_ratio,
                open_ksize=open_ksize,
                dilate_ksize=dilate_ksize,
            )
        except RuntimeError as exc:
            if is_cuda_resource_exhausted_error(exc):
                fallback_selection = resolve_after_resource_exhausted(
                    "star_mask_dog",
                    "cuda_host_io",
                    exc,
                    load_module=_load_compiled_module_result,
                )
                _debug_log(
                    "CUDA DoG mask backend exhausted resources, falling back "
                    f"to NumPy: {exc}"
                )
            else:
                fallback_selection = resolve_after_runtime_unavailable(
                    "star_mask_dog",
                    "cuda_host_io",
                    exc,
                    load_module=_load_compiled_module_result,
                )
                _debug_log(
                    "CUDA DoG mask backend unavailable at runtime, falling "
                    f"back to NumPy: {exc}"
                )
            if fallback_selection.native:
                raise RuntimeError(
                    "star_mask_dog selected an unsupported native fallback"
                )

    return star_mask_dog_numpy(
        image,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        dilate_ksize=dilate_ksize,
    )


def star_shrink_dog_process_numpy(
    image: np.ndarray,
    sigma_small: float = 1.5,
    sigma_large: float = 12.0,
    threshold_ratio: int | float = 3,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
    shrink_ksize: int = 3,
    shrink_shape: str = "CIRCLE",
    shrink_times: int = 1,
    shrink_ratio: float | None = 1.0,
    deringing_ksize: int = 51,
) -> np.ndarray:
    image_arr = np.asarray(image)
    mask = star_mask_dog_numpy(
        image_arr,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        dilate_ksize=dilate_ksize,
    )
    return star_shrink_process_numpy(
        image_arr,
        mask,
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )


def star_shrink_dog_process_compiled_cuda(
    image: np.ndarray,
    sigma_small: float = 1.5,
    sigma_large: float = 12.0,
    threshold_ratio: int | float = 3,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
    shrink_ksize: int = 3,
    shrink_shape: str = "CIRCLE",
    shrink_times: int = 1,
    shrink_ratio: float | None = 1.0,
    deringing_ksize: int = 51,
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "star_shrink_dog_process_cuda"):
        raise RuntimeError("compiled custom op backend is unavailable")
    image_arr = _validate_dog_image(image, "star_shrink_dog_process")
    small_kernel_size = _dog_kernel_size(
        sigma_small, "star_shrink_dog_process"
    )
    large_kernel_size = _dog_kernel_size(
        sigma_large, "star_shrink_dog_process"
    )
    shrink_size, shape, times, ratio, dering_size = _validate_process_params(
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )
    estimate = cuda_memory_estimate(
        "star_shrink_dog_process",
        **_star_shrink_estimate_args(image_arr),
        small_kernel_size=small_kernel_size,
        large_kernel_size=large_kernel_size,
    )
    kernel_args = (
        image_arr,
        float(sigma_small),
        float(sigma_large),
        float(threshold_ratio),
        int(open_ksize),
        int(dilate_ksize),
        shrink_size,
        shape,
        times,
        ratio,
        dering_size,
    )
    return _run_admitted_cuda(
        estimate,
        module.star_shrink_dog_process_cuda,
        *kernel_args,
    )


@lru_cache(maxsize=2)
def _select_star_shrink_dog_process_backend(preference: str) -> BackendSelection:
    return _select_backend(
        "star_shrink_dog_process",
        preference,
        load_module=_load_compiled_module_result,
    )


def star_shrink_dog_process(
    image: np.ndarray,
    sigma_small: float = 1.5,
    sigma_large: float = 12.0,
    threshold_ratio: int | float = 3,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
    shrink_ksize: int = 3,
    shrink_shape: str = "CIRCLE",
    shrink_times: int = 1,
    shrink_ratio: float | None = 1.0,
    deringing_ksize: int = 51,
) -> np.ndarray:
    image_arr = np.asarray(image)
    if np.dtype(image_arr.dtype) not in _COMPILED_SUPPORTED_DTYPES:
        return star_shrink_dog_process_numpy(
            image,
            sigma_small=sigma_small,
            sigma_large=sigma_large,
            threshold_ratio=threshold_ratio,
            open_ksize=open_ksize,
            dilate_ksize=dilate_ksize,
            shrink_ksize=shrink_ksize,
            shrink_shape=shrink_shape,
            shrink_times=shrink_times,
            shrink_ratio=shrink_ratio,
            deringing_ksize=deringing_ksize,
        )

    selection = _select_star_shrink_dog_process_backend(_fallback_preference())
    if selection.native and selection.candidate is not None:
        if selection.candidate.backend == "cuda_host_io":
            try:
                return star_shrink_dog_process_compiled_cuda(
                    image,
                    sigma_small=sigma_small,
                    sigma_large=sigma_large,
                    threshold_ratio=threshold_ratio,
                    open_ksize=open_ksize,
                    dilate_ksize=dilate_ksize,
                    shrink_ksize=shrink_ksize,
                    shrink_shape=shrink_shape,
                    shrink_times=shrink_times,
                    shrink_ratio=shrink_ratio,
                    deringing_ksize=deringing_ksize,
                )
            except RuntimeError as exc:
                if is_cuda_resource_exhausted_error(exc):
                    fallback_selection = resolve_after_resource_exhausted(
                        "star_shrink_dog_process",
                        "cuda_host_io",
                        exc,
                        load_module=_load_compiled_module_result,
                    )
                    _debug_log(
                        "CUDA DoG shrink backend exhausted resources, using "
                        f"the composed path: {exc}"
                    )
                else:
                    fallback_selection = resolve_after_runtime_unavailable(
                        "star_shrink_dog_process",
                        "cuda_host_io",
                        exc,
                        load_module=_load_compiled_module_result,
                    )
                    _debug_log(
                        "CUDA DoG shrink backend unavailable at runtime, using "
                        f"the composed path: {exc}"
                    )
                if fallback_selection.native:
                    raise RuntimeError(
                        "star_shrink_dog_process selected an unsupported native fallback"
                    )
    elif selection.reason:
        _debug_log(f"DoG shrink fused backend unavailable, reason: {selection.reason}")

    mask = star_mask_dog(
        image,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        dilate_ksize=dilate_ksize,
    )
    return star_shrink_process(
        image,
        mask,
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )
