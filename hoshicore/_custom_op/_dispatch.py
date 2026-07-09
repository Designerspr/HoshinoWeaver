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
    message = str(exc).lower()
    return (
        "no cuda-capable device is detected" in message
        or "cuda driver version is insufficient" in message
        or "cuda initialization error" in message
        or "cudaunknown" in message
        or "cuda unknown" in message
        or "device is busy" in message
        or "device unavailable" in message
        or "no kernel image is available" in message
        or "no binary for gpu" in message
    )


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
        return {"available": False, "reason": error or "compiled backend unavailable"}
    if not hasattr(module, "cuda_memory_info"):
        return {
            "available": False,
            "reason": "compiled backend does not expose CUDA memory info",
        }
    try:
        payload = module.cuda_memory_info()
    except RuntimeError as exc:
        return {"available": False, "reason": str(exc)}
    return payload if isinstance(payload, dict) else {
        "available": False,
        "reason": "invalid CUDA memory info payload",
    }


_LAST_APPLIED_COMPILED_THREADS: int | None = None


def apply_compiled_threads(op_name: str, sample: np.ndarray) -> None:
    global _LAST_APPLIED_COMPILED_THREADS
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
    if threads == _LAST_APPLIED_COMPILED_THREADS:
        return
    if module.set_openmp_threads(int(threads)):
        _LAST_APPLIED_COMPILED_THREADS = int(threads)
