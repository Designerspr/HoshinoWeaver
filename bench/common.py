"""Benchmark 公共工具。

用途：
- 统一 benchmark 的计时和结果输出格式
- 生成合成 numpy 帧序列
- 加载 raw cache 或图片目录
- 为 OpenMP 线程数提供平台相关的自动选择
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DATA_DIR = PROJECT_ROOT / "bench" / "data"
BENCH_CACHE_DIR = BENCH_DATA_DIR / "cache"
DEFAULT_INPUT_DIRS = [
    BENCH_CACHE_DIR,
    BENCH_DATA_DIR / "input",
    BENCH_DATA_DIR / "generated",
]
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_OMP_SETTER: Any | None = None


def collect_env_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cwd": str(PROJECT_ROOT),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_seconds(value: float) -> str:
    return f"{value:.6f}s"


def _case_order(report: dict[str, Any]) -> list[str]:
    terminal_cases = report.get("terminal_cases", report.get("display_cases"))
    config = report.get("config", {})
    configured = config.get("cases")
    results = report.get("results", {})
    if isinstance(terminal_cases, list):
        return [str(case) for case in terminal_cases if case in results]
    if isinstance(configured, list):
        return [str(case) for case in configured if case in results]
    return list(results.keys())


def _plural_unit(unit: str) -> str:
    if unit.endswith("pass"):
        return f"{unit}es"
    if unit.endswith("s"):
        return unit
    return f"{unit}s"


def annotate_case_units(
    results: dict[str, dict[str, Any]],
    case_units: dict[str, dict[str, Any]],
) -> None:
    for case_name, unit_info in case_units.items():
        payload = results.get(case_name)
        if not isinstance(payload, dict):
            continue
        unit = str(unit_info.get("unit", "item"))
        count = unit_info.get("count")
        if not isinstance(count, (int, float)) or count <= 0:
            continue
        unit_count = int(count)
        payload["unit"] = unit
        payload["unit_count"] = unit_count
        for field_name in ("min_sec", "max_sec", "mean_sec", "median_sec"):
            value = payload.get(field_name)
            if isinstance(value, (int, float)):
                metric_name = field_name.replace("_sec", "_per_unit_sec")
                payload[metric_name] = float(value) / unit_count


def render_terminal_summary(report: dict[str, Any], output_json: str | None) -> str:
    lines: list[str] = []
    suite = report.get("suite", "benchmark")
    input_source = report.get("input_source", {})
    custom_ops = report.get("custom_ops", {})
    terminal_mode = report.get("terminal_mode")
    totals_only = terminal_mode == "pipeline_totals"
    terminal_case_labels = report.get("terminal_case_labels", {})

    lines.append(f"[{suite}]")

    if isinstance(input_source, dict) and input_source:
        mode = input_source.get("mode")
        frames = input_source.get("resolved_frames")
        shape = input_source.get("resolved_shape")
        dtype = input_source.get("resolved_dtype")
        parts = []
        if mode:
            parts.append(f"input={mode}")
        if frames is not None:
            parts.append(f"frames={frames}")
        if shape:
            parts.append(f"shape={shape}")
        if dtype:
            parts.append(f"dtype={dtype}")
        if parts:
            lines.append(" ".join(parts))
    elif isinstance(report.get("input_sources"), dict):
        input_parts = []
        for suite_id, source in report["input_sources"].items():
            if not isinstance(source, dict):
                continue
            mode = source.get("mode")
            frames = source.get("resolved_frames")
            if mode is None:
                continue
            suffix = f":{mode}"
            if frames is not None:
                suffix += f"({frames})"
            input_parts.append(f"{suite_id}{suffix}")
        if input_parts:
            lines.append("inputs=" + ", ".join(input_parts))

    if not totals_only and isinstance(custom_ops, dict) and custom_ops.get("available"):
        compiler = custom_ops.get("compiler")
        openmp = custom_ops.get("openmp")
        omp_simd = custom_ops.get("omp_simd")
        ndebug = custom_ops.get("ndebug")
        parts = []
        if compiler:
            parts.append(f"compiler={compiler}")
        if openmp is not None:
            parts.append(f"openmp={openmp}")
        if omp_simd is not None:
            parts.append(f"omp_simd={omp_simd}")
        if ndebug is not None:
            parts.append(f"ndebug={ndebug}")
        if parts:
            lines.append(" ".join(parts))

    pipeline = report.get("pipeline")
    if not totals_only and isinstance(pipeline, dict):
        parts = []
        for key in ("aligned_frames", "failed_frames", "output_shape", "output_dtype"):
            value = pipeline.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        if parts:
            lines.append("pipeline=" + " ".join(parts))
    pipelines = report.get("pipelines")
    if not totals_only and isinstance(pipelines, dict):
        for pipeline_name, pipeline_payload in pipelines.items():
            if not isinstance(pipeline_payload, dict):
                continue
            parts = []
            for key in ("aligned_frames", "failed_frames", "output_shape", "output_dtype"):
                value = pipeline_payload.get(key)
                if value is not None:
                    parts.append(f"{key}={value}")
            if parts:
                lines.append(f"pipeline.{pipeline_name}=" + " ".join(parts))

    backend_diagnostics = report.get("backend_diagnostics")
    if not totals_only and isinstance(backend_diagnostics, dict):
        parts = []
        for logical_op, payload in backend_diagnostics.items():
            if (
                logical_op in {
                    "preference",
                    "runtime_probes",
                    "registry_selection_only",
                    "note",
                }
                or not isinstance(payload, dict)
            ):
                continue
            backend = payload.get("candidate_backend")
            if backend is not None:
                parts.append(f"{logical_op}:{backend}")
        if parts:
            preference = backend_diagnostics.get("preference")
            prefix = f"backend_preference={preference} " if preference else ""
            lines.append(prefix + "registry_selection=" + ", ".join(parts))
        runtime_probes = backend_diagnostics.get("runtime_probes")
        if isinstance(runtime_probes, dict):
            probe_parts = []
            for name, payload in runtime_probes.items():
                if not isinstance(payload, dict):
                    continue
                status = payload.get("status")
                if status is not None:
                    probe_parts.append(f"{name}:{status}")
            if probe_parts:
                lines.append("cuda_runtime_probe=" + ", ".join(probe_parts))

    results = report.get("results", {})
    for case_name in _case_order(report):
        payload = results.get(case_name, {})
        if not isinstance(payload, dict):
            continue
        if payload.get("skipped"):
            reason = payload.get("reason")
            summary = f"{case_name}: skipped"
            if isinstance(reason, str) and reason:
                summary += f" ({reason})"
            lines.append(summary)
            continue
        mean_sec = payload.get("mean_sec")
        min_sec = payload.get("min_sec")
        max_sec = payload.get("max_sec")
        if isinstance(mean_sec, (int, float)):
            label = case_name
            if isinstance(terminal_case_labels, dict):
                label = str(terminal_case_labels.get(case_name, case_name))
            summary = f"{label}: mean={_format_seconds(float(mean_sec))}"
            if isinstance(min_sec, (int, float)) and isinstance(max_sec, (int, float)):
                summary += f" min={_format_seconds(float(min_sec))} max={_format_seconds(float(max_sec))}"
            per_unit_mean = payload.get("mean_per_unit_sec")
            unit = payload.get("unit")
            unit_count = payload.get("unit_count")
            if (not totals_only
                    and isinstance(per_unit_mean, (int, float))
                    and isinstance(unit, str)
                    and isinstance(unit_count, int)):
                summary += (
                    f" per_{unit}_mean={_format_seconds(float(per_unit_mean))}"
                    f" {_plural_unit(unit)}={unit_count}"
                )
            lines.append(summary)

    custom_backends = report.get("custom_backends", {})
    if isinstance(custom_backends, dict):
        for case_name, backend_name in custom_backends.items():
            lines.append(f"{case_name}_backend: {backend_name}")

    accuracy_by_case = report.get("accuracy_by_case")
    if not isinstance(accuracy_by_case, dict):
        accuracy = report.get("accuracy", {})
        if (
            isinstance(accuracy, dict)
            and isinstance(accuracy.get("max_abs_err"), (int, float))
            and isinstance(accuracy.get("mean_abs_err"), (int, float))
        ):
            accuracy_by_case = {"accuracy": accuracy}
        elif isinstance(accuracy, dict):
            accuracy_by_case = accuracy
        else:
            accuracy_by_case = {}
    if isinstance(accuracy_by_case, dict):
        for case_name, payload in accuracy_by_case.items():
            if not isinstance(payload, dict):
                continue
            max_abs_err = payload.get("max_abs_err")
            mean_abs_err = payload.get("mean_abs_err")
            if isinstance(max_abs_err, (int, float)) and isinstance(
                    mean_abs_err, (int, float)):
                label = "accuracy" if case_name == "accuracy" else f"{case_name}_accuracy"
                lines.append(
                    f"{label}: "
                    f"max_abs_err={max_abs_err:.6e} "
                    f"mean_abs_err={mean_abs_err:.6e}"
                )

    return "\n".join(lines)


def print_or_save_report(report: dict[str, Any], output_json: str | None) -> None:
    payload = json.dumps(to_jsonable(report), indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(render_terminal_summary(report, output_json))


def summarize_samples(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_sec": samples,
        "min_sec": min(samples),
        "max_sec": max(samples),
        "mean_sec": mean(samples),
        "median_sec": median(samples),
    }


def run_benchmark(
    func: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        func()

    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        samples.append(time.perf_counter() - t0)
    return summarize_samples(samples)


def available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _discover_omp_library() -> str | None:
    candidates = [
        ctypes.util.find_library("gomp"),
        ctypes.util.find_library("omp"),
        ctypes.util.find_library("vcomp140"),
        "libgomp.so.1",
        "libomp.so",
        "libomp.dylib",
        "vcomp140.dll",
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def set_omp_threads(num_threads: int) -> bool:
    global _OMP_SETTER
    if num_threads <= 0:
        return False
    if _OMP_SETTER is None:
        lib_name = _discover_omp_library()
        if lib_name is None:
            _OMP_SETTER = False
        else:
            try:
                runtime = ctypes.CDLL(lib_name)
                runtime.omp_set_num_threads.argtypes = [ctypes.c_int]
                runtime.omp_set_num_threads.restype = None
                _OMP_SETTER = runtime.omp_set_num_threads
            except Exception:
                _OMP_SETTER = False
    if _OMP_SETTER is False:
        return False
    _OMP_SETTER(num_threads)
    return True


def resolve_openmp_threads(
    raw_value: str,
    *,
    workers: int = 1,
) -> int:
    if raw_value != "auto":
        value = int(raw_value)
        if value <= 0:
            raise ValueError("openmp threads must be positive or 'auto'")
        return value
    return max(1, available_cpu_count() // max(1, workers))


def make_frames(
    *,
    frames: int,
    height: int,
    width: int,
    dtype: np.dtype,
    channels: int = 3,
    seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    shape = (height, width, channels) if channels > 1 else (height, width)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        data = rng.integers(
            low=0,
            high=info.max + 1,
            size=(frames, *shape),
            dtype=dtype,
        )
    else:
        data = rng.random((frames, *shape), dtype=np.float32).astype(dtype)

    return [data[i].copy() for i in range(frames)]


def make_weights(frames: int) -> list[float]:
    if frames <= 1:
        return [1.0]
    return np.linspace(0.25, 1.0, frames, dtype=np.float32).tolist()


def _iter_dataset_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return []
    return [root] + sorted(path for path in root.rglob("*") if path.is_dir())


def _cache_meta_path(root: Path) -> Path:
    return root / "meta.json"


def _cache_data_path(root: Path) -> Path:
    return root / "frames.dat"


def _is_cache_dir(root: Path) -> bool:
    return _cache_meta_path(root).is_file() and _cache_data_path(root).is_file()


def _load_cache_meta(root: Path) -> dict[str, Any]:
    meta = json.loads(_cache_meta_path(root).read_text(encoding="utf-8"))
    required = {"frames", "shape", "dtype"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"benchmark cache metadata missing keys: {sorted(missing)}")
    return meta


def discover_cache_dataset(
    *,
    input_dir: str | None,
    frames: int,
) -> tuple[Path | None, dict[str, Any] | None]:
    roots: list[Path] = []
    if input_dir is not None:
        roots.append(Path(input_dir))
    else:
        roots.extend(DEFAULT_INPUT_DIRS)

    for root in roots:
        if root.is_file():
            continue
        for dataset_dir in _iter_dataset_dirs(root):
            if not _is_cache_dir(dataset_dir):
                continue
            meta = _load_cache_meta(dataset_dir)
            if int(meta["frames"]) >= frames:
                return dataset_dir, meta
    return None, None


def open_cache_batch(
    root: Path,
    meta: dict[str, Any],
    *,
    frames: int,
) -> np.memmap:
    shape = tuple(int(v) for v in meta["shape"])
    if len(shape) < 2:
        raise ValueError(f"benchmark cache shape is invalid: {shape}")
    if int(meta["frames"]) < frames:
        raise ValueError(
            f"benchmark cache only has {meta['frames']} frames, requested {frames}"
        )
    full_shape = (int(meta["frames"]), *shape)
    batch = np.memmap(
        _cache_data_path(root),
        dtype=np.dtype(meta["dtype"]),
        mode="r",
        shape=full_shape,
    )
    return batch[:frames]


def discover_image_paths(
    *,
    input_dir: str | None,
    frames: int,
) -> tuple[Path | None, list[Path]]:
    roots: list[Path] = []
    if input_dir is not None:
        roots.append(Path(input_dir))
    else:
        roots.extend(DEFAULT_INPUT_DIRS)

    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root] if root.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES else []
            if len(candidates) >= frames:
                return root.parent, candidates[:frames]
            continue

        search_dirs = _iter_dataset_dirs(root)
        for dataset_dir in search_dirs:
            candidates = sorted(
                p for p in dataset_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            if len(candidates) >= frames:
                return dataset_dir, candidates[:frames]
    return None, []


def load_benchmark_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to decode benchmark image: {path}")
    return image


def load_frames_from_paths(paths: list[Path]) -> list[np.ndarray]:
    frames = [load_benchmark_image(path) for path in paths]
    first = frames[0]
    for idx, frame in enumerate(frames[1:], start=1):
        if frame.shape != first.shape:
            raise ValueError(
                f"benchmark input shape mismatch: frame 0 {first.shape} vs frame {idx} {frame.shape}"
            )
        if frame.dtype != first.dtype:
            raise ValueError(
                f"benchmark input dtype mismatch: frame 0 {first.dtype} vs frame {idx} {frame.dtype}"
            )
    return frames


def resolve_existing_frames(
    *,
    frames: int,
    input_dir: str | None = None,
    input_mode: str = "auto",
) -> tuple[list[np.ndarray], dict[str, Any]] | None:
    if input_mode not in {"auto", "cache", "images"}:
        raise ValueError(f"unsupported input_mode: {input_mode}")

    if input_mode in {"auto", "cache"}:
        cache_root, cache_meta = discover_cache_dataset(input_dir=input_dir, frames=frames)
        if cache_root is not None and cache_meta is not None:
            batch = open_cache_batch(cache_root, cache_meta, frames=frames)
            source = {
                "mode": "raw_cache",
                "input_dir": str(cache_root),
                "requested_input_mode": input_mode,
                "resolved_frames": int(batch.shape[0]),
                "resolved_shape": list(batch.shape[1:]),
                "resolved_dtype": str(batch.dtype),
                "cache_files": [
                    str(_cache_meta_path(cache_root)),
                    str(_cache_data_path(cache_root)),
                ],
            }
            return [batch[idx] for idx in range(batch.shape[0])], source
        if input_mode == "cache":
            raise FileNotFoundError(f"no raw cache dataset found for frames={frames} under: {input_dir or DEFAULT_INPUT_DIRS}")

    if input_mode in {"auto", "images"}:
        root, image_paths = discover_image_paths(input_dir=input_dir, frames=frames)
        if image_paths:
            loaded = load_frames_from_paths(image_paths)
            first = loaded[0]
            source = {
                "mode": "images",
                "input_dir": str(root),
                "requested_input_mode": input_mode,
                "resolved_frames": len(loaded),
                "resolved_shape": list(first.shape),
                "resolved_dtype": str(first.dtype),
                "sample_paths": [str(path) for path in image_paths[:3]],
            }
            return loaded, source
        if input_mode == "images":
            raise FileNotFoundError(f"no image dataset found for frames={frames} under: {input_dir or DEFAULT_INPUT_DIRS}")

    if input_dir is not None and input_mode == "auto":
        raise FileNotFoundError(
            f"no raw cache or image dataset found for frames={frames} under: {input_dir}"
        )

    return None


def prepare_frames(
    *,
    frames: int,
    height: int,
    width: int,
    dtype: np.dtype,
    channels: int = 3,
    seed: int = 0,
    input_dir: str | None = None,
    input_mode: str = "auto",
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if input_mode not in {"auto", "cache", "images", "synthetic"}:
        raise ValueError(f"unsupported input_mode: {input_mode}")

    if input_mode != "synthetic":
        existing = resolve_existing_frames(
            frames=frames,
            input_dir=input_dir,
            input_mode=input_mode,
        )
        if existing is not None:
            return existing

    synthetic = make_frames(
        frames=frames,
        height=height,
        width=width,
        dtype=dtype,
        channels=channels,
        seed=seed,
    )
    source = {
        "mode": "synthetic",
        "input_dir": None,
        "requested_input_mode": input_mode,
        "resolved_frames": len(synthetic),
        "resolved_shape": list(synthetic[0].shape),
        "resolved_dtype": str(synthetic[0].dtype),
        "seed": seed,
    }
    return synthetic, source


def prepare_batch(
    *,
    frames: int,
    height: int,
    width: int,
    dtype: np.dtype,
    channels: int = 3,
    seed: int = 0,
    input_dir: str | None = None,
    input_mode: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    frame_list, source = prepare_frames(
        frames=frames,
        height=height,
        width=width,
        dtype=dtype,
        channels=channels,
        seed=seed,
        input_dir=input_dir,
        input_mode=input_mode,
    )
    batch = np.stack(frame_list, axis=0)
    return batch, source
