"""Camera-model remap custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from hoshicore._custom_op._dispatch import debug_enabled as _debug_enabled
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import select_backend as _select_backend


_debug_log = partial(debug_log, "remap")
_is_cuda_runtime_unavailable_error = is_cuda_runtime_unavailable_error
_COMPILED_SUPPORTED_DTYPES = (
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.float32),
)


def _compiled_supports_dtype(dtype: np.dtype) -> bool:
    return np.dtype(dtype) in _COMPILED_SUPPORTED_DTYPES


def _validate_rotation(rotation_dst_to_src: np.ndarray) -> np.ndarray:
    rotation_arr = np.asarray(rotation_dst_to_src, dtype=np.float32)
    if rotation_arr.shape != (3, 3):
        raise ValueError(
            "camera_model_remap: rotation_dst_to_src must have shape (3, 3)")
    if not np.all(np.isfinite(rotation_arr)):
        raise ValueError(
            "camera_model_remap: rotation_dst_to_src must contain only finite values")
    if not rotation_arr.flags.c_contiguous:
        rotation_arr = np.ascontiguousarray(rotation_arr)
    return rotation_arr


def _validate_scalar(value: float, name: str, *, non_zero: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"camera_model_remap: {name} must contain only finite values")
    if non_zero and scalar == 0.0:
        raise ValueError(f"camera_model_remap: {name} must be non-zero")
    return scalar


def _validate_camera_scalars(
    *,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
) -> dict[str, float]:
    return {
        "fx_src": _validate_scalar(fx_src, "fx_src", non_zero=True),
        "fy_src": _validate_scalar(fy_src, "fy_src", non_zero=True),
        "cx_src": _validate_scalar(cx_src, "cx_src"),
        "cy_src": _validate_scalar(cy_src, "cy_src"),
        "fx_dst": _validate_scalar(fx_dst, "fx_dst", non_zero=True),
        "fy_dst": _validate_scalar(fy_dst, "fy_dst", non_zero=True),
        "cx_dst": _validate_scalar(cx_dst, "cx_dst"),
        "cy_dst": _validate_scalar(cy_dst, "cy_dst"),
    }


def _validate_image(image: np.ndarray) -> np.ndarray:
    image_arr = np.asarray(image)
    if image_arr.ndim not in {2, 3}:
        raise ValueError("camera_model_remap: image must have shape (H, W) or (H, W, C)")
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError("camera_model_remap: image height and width must be positive")
    if image_arr.ndim == 3 and image_arr.shape[2] <= 0:
        raise ValueError("camera_model_remap: image channels must be positive")
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    return image_arr


def _validate_dist_coeffs(dist_coeffs: np.ndarray | None,
                          name: str) -> np.ndarray | None:
    if dist_coeffs is None:
        return None
    dist_arr = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1)
    if dist_arr.size == 0 or np.all(dist_arr == 0):
        return None
    if dist_arr.size not in {2, 4, 5}:
        raise ValueError(
            f"camera_model_remap: {name} must have 2, 4, or 5 coefficients")
    if not np.all(np.isfinite(dist_arr)):
        raise ValueError(f"camera_model_remap: {name} must contain only finite values")
    if dist_arr.size < 5:
        dist_arr = np.pad(dist_arr, (0, 5 - dist_arr.size))
    if not dist_arr.flags.c_contiguous:
        dist_arr = np.ascontiguousarray(dist_arr)
    return dist_arr


def _make_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _build_remap_maps(
    *,
    out_height: int,
    out_width: int,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    rotation_dst_to_src: np.ndarray,
    src_dist_coeffs: np.ndarray | None,
    dst_dist_coeffs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if src_dist_coeffs is None and dst_dist_coeffs is None:
        xs = np.arange(out_width, dtype=np.float32)
        ys = np.arange(out_height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        x = (grid_x - np.float32(cx_dst)) / np.float32(fx_dst)
        y = (grid_y - np.float32(cy_dst)) / np.float32(fy_dst)
        rotation = rotation_dst_to_src.astype(np.float32, copy=False)
        proj_x = rotation[0, 0] * x + rotation[0, 1] * y + rotation[0, 2]
        proj_y = rotation[1, 0] * x + rotation[1, 1] * y + rotation[1, 2]
        proj_z = rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2]

        map_x = np.full((out_height, out_width), np.nan, dtype=np.float32)
        map_y = np.full((out_height, out_width), np.nan, dtype=np.float32)
        valid = proj_z > 0.0
        if np.any(valid):
            inv_z = (1.0 / proj_z[valid]).astype(np.float32, copy=False)
            map_x[valid] = np.float32(fx_src) * proj_x[valid] * inv_z + np.float32(cx_src)
            map_y[valid] = np.float32(fy_src) * proj_y[valid] * inv_z + np.float32(cy_src)
        return map_x, map_y

    xs = np.arange(out_width, dtype=np.float64)
    ys = np.arange(out_height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")

    dst_pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    if dst_dist_coeffs is not None:
        k_dst = _make_camera_matrix(fx_dst, fy_dst, cx_dst, cy_dst)
        dst_norm = cv2.undistortPoints(
            dst_pixels[:, None, :],
            k_dst,
            dst_dist_coeffs.astype(np.float64, copy=False),
            P=None,
        )[:, 0, :]
    else:
        dst_norm = np.empty_like(dst_pixels)
        dst_norm[:, 0] = (dst_pixels[:, 0] - cx_dst) / fx_dst
        dst_norm[:, 1] = (dst_pixels[:, 1] - cy_dst) / fy_dst

    dst_rays = np.concatenate(
        [dst_norm, np.ones((dst_norm.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    src_rays = (rotation_dst_to_src.astype(np.float64, copy=False) @ dst_rays.T).T
    valid = src_rays[:, 2] > 0.0
    src_pixels = np.full((src_rays.shape[0], 2), np.nan, dtype=np.float64)

    if np.any(valid):
        valid_rays = src_rays[valid]
        if src_dist_coeffs is not None:
            k_src = _make_camera_matrix(fx_src, fy_src, cx_src, cy_src)
            projected, _ = cv2.projectPoints(
                valid_rays.reshape(-1, 1, 3),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                k_src,
                src_dist_coeffs.astype(np.float64, copy=False),
            )
            src_pixels[valid] = projected[:, 0, :]
        else:
            normalized = valid_rays[:, :2] / valid_rays[:, 2:3]
            src_pixels[valid, 0] = fx_src * normalized[:, 0] + cx_src
            src_pixels[valid, 1] = fy_src * normalized[:, 1] + cy_src

    return (
        src_pixels[:, 0].reshape(out_height, out_width).astype(np.float32),
        src_pixels[:, 1].reshape(out_height, out_width).astype(np.float32),
    )


def camera_model_remap_numpy(
    *,
    image: np.ndarray,
    out_height: int,
    out_width: int,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    rotation_dst_to_src: np.ndarray,
    src_dist_coeffs: np.ndarray | None = None,
    dst_dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    image_arr = _validate_image(image)
    if out_height <= 0 or out_width <= 0:
        raise ValueError("camera_model_remap: output height and width must be positive")
    scalars = _validate_camera_scalars(
        fx_src=fx_src,
        fy_src=fy_src,
        cx_src=cx_src,
        cy_src=cy_src,
        fx_dst=fx_dst,
        fy_dst=fy_dst,
        cx_dst=cx_dst,
        cy_dst=cy_dst,
    )
    rotation_arr = _validate_rotation(rotation_dst_to_src)
    src_dist_arr = _validate_dist_coeffs(src_dist_coeffs, "src_dist_coeffs")
    dst_dist_arr = _validate_dist_coeffs(dst_dist_coeffs, "dst_dist_coeffs")
    map_x, map_y = _build_remap_maps(
        out_height=out_height,
        out_width=out_width,
        **scalars,
        rotation_dst_to_src=rotation_arr,
        src_dist_coeffs=src_dist_arr,
        dst_dist_coeffs=dst_dist_arr,
    )
    return cv2.remap(
        image_arr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0 if image_arr.ndim == 2 else (0, 0, 0),
    )


def _camera_model_remap_compiled_kernel(
    kernel_name: str,
    *,
    image: np.ndarray,
    out_height: int,
    out_width: int,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    rotation_dst_to_src: np.ndarray,
    src_dist_coeffs: np.ndarray | None = None,
    dst_dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, kernel_name):
        raise RuntimeError("compiled custom op backend is unavailable")
    image_arr = _validate_image(image)
    scalars = _validate_camera_scalars(
        fx_src=fx_src,
        fy_src=fy_src,
        cx_src=cx_src,
        cy_src=cy_src,
        fx_dst=fx_dst,
        fy_dst=fy_dst,
        cx_dst=cx_dst,
        cy_dst=cy_dst,
    )
    rotation_arr = _validate_rotation(rotation_dst_to_src)
    src_dist_arr = _validate_dist_coeffs(src_dist_coeffs, "src_dist_coeffs")
    dst_dist_arr = _validate_dist_coeffs(dst_dist_coeffs, "dst_dist_coeffs")
    if kernel_name == "camera_model_remap_cpu":
        _apply_compiled_threads("camera_model_remap", image_arr)
    kernel = getattr(module, kernel_name)
    return kernel(
        image_arr,
        int(out_height),
        int(out_width),
        scalars["fx_src"],
        scalars["fy_src"],
        scalars["cx_src"],
        scalars["cy_src"],
        scalars["fx_dst"],
        scalars["fy_dst"],
        scalars["cx_dst"],
        scalars["cy_dst"],
        rotation_arr,
        src_dist_arr,
        dst_dist_arr,
    )


def camera_model_remap_compiled(
    *,
    image: np.ndarray,
    out_height: int,
    out_width: int,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    rotation_dst_to_src: np.ndarray,
    src_dist_coeffs: np.ndarray | None = None,
    dst_dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    return _camera_model_remap_compiled_kernel(
        "camera_model_remap",
        image=image,
        out_height=out_height,
        out_width=out_width,
        fx_src=fx_src,
        fy_src=fy_src,
        cx_src=cx_src,
        cy_src=cy_src,
        fx_dst=fx_dst,
        fy_dst=fy_dst,
        cx_dst=cx_dst,
        cy_dst=cy_dst,
        rotation_dst_to_src=rotation_dst_to_src,
        src_dist_coeffs=src_dist_coeffs,
        dst_dist_coeffs=dst_dist_coeffs,
    )


def camera_model_remap_cpu_compiled(
    *,
    image: np.ndarray,
    out_height: int,
    out_width: int,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    rotation_dst_to_src: np.ndarray,
    src_dist_coeffs: np.ndarray | None = None,
    dst_dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    return _camera_model_remap_compiled_kernel(
        "camera_model_remap_cpu",
        image=image,
        out_height=out_height,
        out_width=out_width,
        fx_src=fx_src,
        fy_src=fy_src,
        cx_src=cx_src,
        cy_src=cy_src,
        fx_dst=fx_dst,
        fy_dst=fy_dst,
        cx_dst=cx_dst,
        cy_dst=cy_dst,
        rotation_dst_to_src=rotation_dst_to_src,
        src_dist_coeffs=src_dist_coeffs,
        dst_dist_coeffs=dst_dist_coeffs,
    )


@lru_cache(maxsize=2)
def _select_camera_model_remap_backend(
    preference: str,
) -> tuple[str, Callable[..., np.ndarray]]:
    selection = _select_backend(
        "camera_model_remap",
        preference,
        load_module=_load_compiled_module_result,
    )
    if selection.native:
        if selection.candidate and selection.candidate.kernel_name == "camera_model_remap":
            return "cuda", camera_model_remap_compiled
        if selection.candidate and selection.candidate.kernel_name == "camera_model_remap_cpu":
            return "cpu", camera_model_remap_cpu_compiled
        raise RuntimeError(
            f"unknown camera_model_remap backend candidate: {selection.candidate}"
        )

    if selection.reason:
        _debug_log(f"compiled backend unavailable, reason: {selection.reason}")

    return "numpy", camera_model_remap_numpy


def _camera_model_remap_cpu_fallback_available() -> bool:
    selection = _select_backend(
        "camera_model_remap",
        "auto",
        load_module=_load_compiled_module_result,
        exclude_backends={"cuda_host_io"},
    )
    if (
        selection.native
        and selection.candidate is not None
        and selection.candidate.kernel_name == "camera_model_remap_cpu"
    ):
        return True
    if selection.reason:
        _debug_log(f"compiled CPU backend unavailable after CUDA fallback, reason: {selection.reason}")
    return False


def camera_model_remap(
    *,
    image: np.ndarray,
    out_height: int,
    out_width: int,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    rotation_dst_to_src: np.ndarray,
    src_dist_coeffs: np.ndarray | None = None,
    dst_dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    image_arr = _validate_image(image)
    backend_name, backend = _select_camera_model_remap_backend(
        _fallback_preference())
    kwargs = {
        "image": image_arr,
        "out_height": out_height,
        "out_width": out_width,
        "fx_src": fx_src,
        "fy_src": fy_src,
        "cx_src": cx_src,
        "cy_src": cy_src,
        "fx_dst": fx_dst,
        "fy_dst": fy_dst,
        "cx_dst": cx_dst,
        "cy_dst": cy_dst,
        "rotation_dst_to_src": rotation_dst_to_src,
        "src_dist_coeffs": src_dist_coeffs,
        "dst_dist_coeffs": dst_dist_coeffs,
    }
    original_kwargs = kwargs
    restore_float64 = False
    if backend_name != "numpy" and image_arr.dtype == np.float64:
        kwargs = {**kwargs, "image": image_arr.astype(np.float32)}
        restore_float64 = True
    kernel_dtype = kwargs["image"].dtype
    logger.debug(
        "camera_model_remap: backend={} input_dtype={} kernel_dtype={} "
        "shape={} output={}x{}",
        backend_name, image_arr.dtype, kernel_dtype, image_arr.shape,
        out_width, out_height)

    def _restore_dtype(result: np.ndarray) -> np.ndarray:
        if restore_float64:
            return result.astype(np.float64)
        return result

    if backend_name != "numpy" and not _compiled_supports_dtype(kernel_dtype):
        _debug_log(
            f"compiled backend does not support dtype {image_arr.dtype}, falling back to numpy"
        )
        return camera_model_remap_numpy(**original_kwargs)
    if backend_name == "numpy":
        return backend(**kwargs)

    try:
        return _restore_dtype(backend(**kwargs))
    except RuntimeError as exc:
        if backend_name != "cuda" or not is_cuda_runtime_unavailable_error(exc):
            raise
        if _camera_model_remap_cpu_fallback_available():
            return _restore_dtype(camera_model_remap_cpu_compiled(**kwargs))
        _debug_log(
            f"compiled CUDA backend unavailable at runtime, falling back to numpy: {exc}"
        )
        return camera_model_remap_numpy(**original_kwargs)
