"""Star shrink custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import numpy as np

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import select_backend as _select_backend
from hoshicore.component.star_shrink import apply_mask, deringing, morph_shrink_luma


_debug_log = partial(debug_log, "star_shrink")
_SUPPORTED_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32))
_COMPILED_SUPPORTED_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16))


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
    return getattr(module, kernel_name)(
        image_arr,
        mask_arr,
        shrink_size,
        shape,
        times,
        ratio,
        dering_size,
    )


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


def _compiled_cpu_available() -> bool:
    module, _ = _load_compiled_module_result()
    return module is not None and hasattr(module, "star_shrink_process")


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
    if not selection.native or selection.candidate is None:
        if selection.reason:
            _debug_log(f"compiled backend unavailable, reason: {selection.reason}")
        return star_shrink_process_numpy(
            image,
            star_mask,
            shrink_ksize,
            shrink_shape,
            shrink_times,
            shrink_ratio,
            deringing_ksize,
        )

    if selection.candidate.backend == "cuda_host_io":
        try:
            return star_shrink_process_compiled_cuda(
                image,
                star_mask,
                shrink_ksize,
                shrink_shape,
                shrink_times,
                shrink_ratio,
                deringing_ksize,
            )
        except RuntimeError as exc:
            if not is_cuda_runtime_unavailable_error(exc):
                raise
            _debug_log(f"CUDA backend unavailable at runtime, reason: {exc}")
            if _compiled_cpu_available():
                return star_shrink_process_compiled(
                    image,
                    star_mask,
                    shrink_ksize,
                    shrink_shape,
                    shrink_times,
                    shrink_ratio,
                    deringing_ksize,
                )
            return star_shrink_process_numpy(
                image,
                star_mask,
                shrink_ksize,
                shrink_shape,
                shrink_times,
                shrink_ratio,
                deringing_ksize,
            )

    if selection.candidate.backend == "openmp_cpu":
        return star_shrink_process_compiled(
            image,
            star_mask,
            shrink_ksize,
            shrink_shape,
            shrink_times,
            shrink_ratio,
            deringing_ksize,
        )

    return star_shrink_process_numpy(
        image,
        star_mask,
        shrink_ksize,
        shrink_shape,
        shrink_times,
        shrink_ratio,
        deringing_ksize,
    )


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
    image_arr = np.asarray(image)
    if image_arr.ndim not in (2, 3):
        raise ValueError("star_mask_dog: image must have shape (H, W) or (H, W, C)")
    if image_arr.ndim == 3 and image_arr.shape[2] != 3:
        raise ValueError("star_mask_dog: 3D image must have exactly 3 channels")
    if np.dtype(image_arr.dtype) not in _COMPILED_SUPPORTED_DTYPES:
        raise ValueError("star_mask_dog: CUDA backend supports uint8/uint16 only")
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    return module.star_mask_dog_cuda(
        image_arr,
        float(sigma_small),
        float(sigma_large),
        float(threshold_ratio),
        int(open_ksize),
        int(dilate_ksize),
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
            if not is_cuda_runtime_unavailable_error(exc):
                raise
            _debug_log(f"CUDA DoG mask backend unavailable at runtime, reason: {exc}")

    return star_mask_dog_numpy(
        image,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        dilate_ksize=dilate_ksize,
    )
