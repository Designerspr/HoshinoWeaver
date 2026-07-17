"""Shared runtime dispatch helpers for custom-op wrappers."""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from typing import Any

import numpy as np
from loguru import logger

from hoshicore._custom_op import thread_tuning


class CustomOpUnavailableError(RuntimeError):
    """Raised when a native backend is unavailable and production may fallback."""


class CustomOpResourceExhaustedError(RuntimeError):
    """Raised when a native backend lacks resources for the requested input."""


class CudaProbeError(RuntimeError):
    """Raised when a structured CUDA runtime probe reports a real error."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        error_code: int | None,
        category: str,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.error_code = error_code
        self.category = category


def debug_enabled() -> bool:
    return os.environ.get("HNW_CUSTOM_OPS_DEBUG", "0") not in {"", "0", "false", "False"}


def debug_log(module_name: str, message: str) -> None:
    text = f"[hoshicore._custom_op.{module_name}] {message}"
    logger.debug(text)
    if debug_enabled():
        print(text, file=sys.stderr)


def fallback_preference() -> str:
    raw = os.environ.get("HNW_CUSTOM_OPS_FALLBACK", "auto").strip().lower()
    if raw in {"auto", "numpy"}:
        return raw
    return "auto"


def is_cuda_runtime_unavailable_error(exc: RuntimeError) -> bool:
    module, _ = load_compiled_module()
    unavailable_type = (
        getattr(module, "CudaRuntimeUnavailableError", None)
        if module is not None
        else None
    )
    if unavailable_type is not None and isinstance(exc, unavailable_type):
        return True
    message = str(exc).lower()
    return (
        "no cuda-capable device is detected" in message
        or "cuda driver version is insufficient" in message
        or "cuda initialization error" in message
        or "device is busy" in message
        or "device unavailable" in message
    )


def is_cuda_resource_exhausted_error(exc: RuntimeError) -> bool:
    if isinstance(exc, CustomOpResourceExhaustedError):
        return True
    module, _ = load_compiled_module()
    exhausted_type = (
        getattr(module, "CudaResourceExhaustedError", None)
        if module is not None
        else None
    )
    return exhausted_type is not None and isinstance(exc, exhausted_type)


@lru_cache(maxsize=1)
def load_compiled_module() -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module("hoshicore._custom_op._C"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=1)
def compiled_build_info() -> dict[str, Any]:
    module, _ = load_compiled_module()
    if module is None or not hasattr(module, "build_info"):
        return {}
    payload = module.build_info()
    return payload if isinstance(payload, dict) else {}


def cuda_memory_info() -> dict[str, Any]:
    module, error = load_compiled_module()
    if module is None:
        return {
            "available": False,
            "status": "unavailable",
            "reason_code": "module_unavailable",
            "category": "build",
            "reason": error or "compiled backend unavailable",
        }
    if not hasattr(module, "cuda_memory_info"):
        return {
            "available": False,
            "status": "unavailable",
            "reason_code": "probe_unavailable",
            "category": "build",
            "reason": "compiled backend does not expose CUDA memory info",
        }
    payload = module.cuda_memory_info()
    if not isinstance(payload, dict) or not isinstance(payload.get("available"), bool):
        raise RuntimeError("invalid CUDA memory info payload")
    status = payload.get("status")
    if status not in {"available", "unavailable", "explicitly_unavailable", "error"}:
        raise RuntimeError("invalid CUDA memory info status")
    if payload["available"] != (status == "available"):
        raise RuntimeError("inconsistent CUDA memory info availability status")
    if status == "error":
        error_code = payload.get("error_code")
        if error_code is not None and not isinstance(error_code, int):
            raise RuntimeError("invalid CUDA memory info error code")
        raise CudaProbeError(
            str(payload.get("reason") or "CUDA runtime probe failed"),
            reason_code=str(payload.get("reason_code") or "cuda_runtime_error"),
            error_code=error_code,
            category=str(payload.get("category") or "runtime"),
        )
    if payload["available"] and not all(
        isinstance(payload.get(key), int) for key in ("free_bytes", "total_bytes")
    ):
        raise RuntimeError("available CUDA memory info is missing byte counts")
    return payload


def apply_compiled_threads(op_name: str, sample: np.ndarray) -> None:
    module, _ = load_compiled_module()
    if module is None:
        return
    build = compiled_build_info()
    if not build.get("openmp"):
        return
    if not hasattr(module, "set_openmp_threads"):
        return
    threads = thread_tuning.resolve_runtime_threads(
        op_name=op_name,
        shape=sample.shape,
        dtype=sample.dtype,
        build_info=build,
    )
    module.set_openmp_threads(int(threads))
