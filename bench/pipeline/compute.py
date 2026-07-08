"""Application-level production-path overview benchmark.

This suite runs representative production paths independently and prints only
path totals. Per-stage details stay in the JSON report when requested.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import cv2
import numpy as np
from loguru import logger

from bench.common import (
    annotate_case_units,
    collect_env_info,
    print_or_save_report,
    summarize_samples,
)
from bench.cpu.alignment import prepare_alignment_frames
from bench.pipeline import alignment as alignment_bench
from hoshicore._custom_op import build_info as custom_ops_build_info
from hoshicore._custom_op.ops.median import median_reduce_chunk
from hoshicore._custom_op.ops.sigma_clip import sigma_clip_fused_chunk
from hoshicore.component.data_container import FastGaussianParam, FloatImage
from hoshicore.component.merger import (
    HuberWeightedMerger,
    MaxMerger,
    MeanMerger,
    MinMerger,
)
from hoshicore._custom_op import (
    equalize_noise_correct,
    noise_equalization_params,
    star_mask_dog,
    star_shrink_detect_mask,
    star_shrink_process,
)
from hoshicore.component.noise_equalization import equalize_noise, fill_local_mean
from hoshicore.component.calibration import calibration_divide
from hoshicore.component.calibration import calibration_subtract
from hoshicore.ops.satellite_clean_op import SatelliteCleanOp, _FrameSlot
from hoshicore.ops.star_ops import SHRINK_MODE_PRESETS, _star_shrink_pipeline


SUITE_ID = "pipeline.compute"
CASE_NAMES = [
    "alignment_homography",
    "alignment_camera_model",
    "stack_max",
    "stack_min",
    "stack_mean",
    "stack_mean_weighted",
    "stack_mean_masked",
    "stack_median",
    "stack_sigma_clip_fused",
    "stack_huber",
    "calibration_full",
    "noise_equalization",
    "noise_equalization_stages",
    "star_mask_threshold",
    "star_mask_dog",
    "star_shrink",
    "star_shrink_stages",
    "satellite_clean_window",
    "warp_homography",
    "remap_camera_model",
]
DEFAULT_CASES = [
    case_name
    for case_name in CASE_NAMES
    if case_name not in {"noise_equalization_stages", "star_shrink_stages"}
]


def parse_cases(raw: str) -> list[str]:
    if raw == "all":
        return list(DEFAULT_CASES)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _clone_first_frame(frame: np.ndarray) -> np.ndarray:
    return np.array(frame, copy=True)


@contextmanager
def _custom_op_backend(preference: str):
    old_value = os.environ.get("HNW_CUSTOM_OPS_FALLBACK")
    if preference == "auto":
        os.environ.pop("HNW_CUSTOM_OPS_FALLBACK", None)
    else:
        os.environ["HNW_CUSTOM_OPS_FALLBACK"] = preference
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("HNW_CUSTOM_OPS_FALLBACK", None)
        else:
            os.environ["HNW_CUSTOM_OPS_FALLBACK"] = old_value


def _ensure_three_channels(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return frame
    if frame.ndim == 2:
        return np.repeat(frame[..., np.newaxis], 3, axis=2)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return np.repeat(frame, 3, axis=2)
    return frame[..., :3]


def _output_payload(output: np.ndarray | FloatImage) -> dict[str, Any]:
    if isinstance(output, FloatImage):
        data = output.int_transform()
        return {"output_shape": list(data.shape), "output_dtype": str(data.dtype)}
    return {"output_shape": list(output.shape), "output_dtype": str(output.dtype)}


def _require_float_image(value: FloatImage | None, name: str) -> FloatImage:
    if not isinstance(value, FloatImage):
        raise RuntimeError(f"{name} did not produce a FloatImage result")
    return value


def _make_center_mask(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    y0, y1 = h // 4, h - h // 4
    x0, x1 = w // 4, w - w // 4
    mask[y0:y1, x0:x1] = True
    return mask


def _run_simple_merger_once(
    frames: list[np.ndarray],
    merger_cls: type[MaxMerger] | type[MinMerger],
) -> dict[str, Any]:
    merger = merger_cls()
    for idx, frame in enumerate(frames):
        merger.merge(_clone_first_frame(frame) if idx == 0 else frame)
    merged = merger.merged_image
    if not isinstance(merged, np.ndarray):
        raise RuntimeError(f"{merger_cls.__name__} did not produce an ndarray result")
    return {
        "output_shape": list(merged.shape),
        "output_dtype": str(merged.dtype),
    }


def _run_max_stack_once(frames: list[np.ndarray]) -> dict[str, Any]:
    return _run_simple_merger_once(frames, MaxMerger)


def _run_min_stack_once(frames: list[np.ndarray]) -> dict[str, Any]:
    return _run_simple_merger_once(frames, MinMerger)


def _run_mean_stack_once(frames: list[np.ndarray]) -> dict[str, Any]:
    merger = MeanMerger()
    for idx, frame in enumerate(frames):
        merger.merge(_clone_first_frame(frame) if idx == 0 else frame)
    return _output_payload(_require_float_image(merger.merged_image, "MeanMerger"))


def _run_mean_weighted_stack_once(frames: list[np.ndarray]) -> dict[str, Any]:
    merger = MeanMerger(int_weight=True)
    weights = np.linspace(0.5, 1.0, len(frames), dtype=np.float32)
    for idx, (frame, weight) in enumerate(zip(frames, weights)):
        merger.merge(
            _clone_first_frame(frame) if idx == 0 else frame,
            weight=float(weight),
        )
    return _output_payload(_require_float_image(merger.merged_image, "MeanMerger"))


def _run_mean_masked_stack_once(frames: list[np.ndarray]) -> dict[str, Any]:
    merger = MeanMerger()
    mask = _make_center_mask(frames[0])
    for frame in frames:
        merger.merge(
            frame,
            spatial_mask=mask,
            skip_zero_rgb=frame.ndim == 3 and frame.shape[-1] >= 3,
        )
    return _output_payload(_require_float_image(merger.merged_image, "MeanMerger"))


def _run_median_stack_once(frames: list[np.ndarray]) -> dict[str, Any]:
    stack = np.stack(frames, axis=0)
    result = median_reduce_chunk(stack)
    return _output_payload(FloatImage(result, dtype=frames[0].dtype))


def _run_sigma_clip_fused_stack_once(frames: list[np.ndarray]) -> dict[str, Any]:
    stats = _compute_sigma_clip_fused_stats(frames)
    safe_n = np.where(stats.n > 0, stats.n, 1)
    result = np.round(stats.sum_mu / safe_n).astype(frames[0].dtype)
    return _output_payload(result)


def _compute_sigma_clip_fused_stats(frames: list[np.ndarray]) -> FastGaussianParam:
    stack = np.stack(frames, axis=0)
    first = stack[0]
    channels = first.shape[2] if first.ndim == 3 else 1
    stack_2d = np.ascontiguousarray(stack.reshape(stack.shape[0], -1))
    acc_sum, acc_sq, acc_n = sigma_clip_fused_chunk(
        stack_2d,
        rej_high=3.0,
        rej_low=3.0,
        max_iter=5,
        mask=None,
        skip_zero_rgb=channels >= 3,
        channels=channels,
    )
    return FastGaussianParam(
        sum_mu=acc_sum.reshape(first.shape),
        square_sum=acc_sq.reshape(first.shape),
        n=acc_n.reshape(first.shape),
        source_dtype=first.dtype,
        inplace_calc=False,
    )


def _compute_mean_stats(frames: list[np.ndarray]):
    merger = MeanMerger()
    for idx, frame in enumerate(frames):
        merger.merge(_clone_first_frame(frame) if idx == 0 else frame)
    if merger.result is None:
        raise RuntimeError("MeanMerger did not produce statistics")
    merger.result.inplace_calc = False
    return merger.result


def _run_huber_stack_once(frames: list[np.ndarray]) -> dict[str, Any]:
    ref_stats = _compute_mean_stats(frames)
    merger = HuberWeightedMerger(ref_stats=ref_stats, huber_c=1.345)
    for idx, frame in enumerate(frames):
        merger.merge(_clone_first_frame(frame) if idx == 0 else frame)
    return _output_payload(
        _require_float_image(merger.merged_image, "HuberWeightedMerger")
    )


def _run_calibration_full_once(frames: list[np.ndarray]) -> dict[str, Any]:
    source_dtype = frames[0].dtype
    bias_ref = (frames[0] // 16).astype(source_dtype, copy=False)
    dark_ref = (frames[min(1, len(frames) - 1)] // 32).astype(source_dtype, copy=False)
    flat_ref = np.maximum(frames[-1], 1).astype(source_dtype, copy=False)
    last = None
    for frame in frames:
        bias_corrected, bias_dtype = calibration_subtract(
            frame,
            bias_ref,
            source_dtype,
            source_dtype,
        )
        dark_corrected, dark_dtype = calibration_subtract(
            bias_corrected,
            dark_ref,
            bias_dtype,
            source_dtype,
        )
        last, _ = calibration_divide(
            dark_corrected,
            flat_ref,
            dark_dtype,
            source_dtype,
        )
    if last is None:
        raise RuntimeError("calibration full path did not process any frame")
    return _output_payload(last)


def _run_noise_equalization_once(frames: list[np.ndarray]) -> dict[str, Any]:
    work_frames = [_ensure_three_channels(frame) for frame in frames]
    max_merger = MaxMerger()
    for idx, frame in enumerate(work_frames):
        max_merger.merge(_clone_first_frame(frame) if idx == 0 else frame)
    max_img = max_merger.merged_image
    if not isinstance(max_img, np.ndarray):
        raise RuntimeError("MaxMerger did not produce an ndarray result")
    stats = _compute_sigma_clip_fused_stats(work_frames)
    result = equalize_noise(
        max_img.astype(np.float64),
        stats.mu.astype(np.float64),
        np.sqrt(np.maximum(stats.var, 0)).astype(np.float64),
        stats.n,
    )
    return _output_payload(np.round(result).astype(max_img.dtype))


def _run_noise_equalization_stages_once(frames: list[np.ndarray]) -> dict[str, Any]:
    work_frames = [_ensure_three_channels(frame) for frame in frames]
    stages: dict[str, float] = {}

    t0 = time.perf_counter()
    max_merger = MaxMerger()
    for idx, frame in enumerate(work_frames):
        max_merger.merge(_clone_first_frame(frame) if idx == 0 else frame)
    max_img = max_merger.merged_image
    if not isinstance(max_img, np.ndarray):
        raise RuntimeError("MaxMerger did not produce an ndarray result")
    stages["max_stack_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stats = _compute_sigma_clip_fused_stats(work_frames)
    stages["sigma_clip_fused_stats_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    max_f64 = max_img.astype(np.float64)
    mean_f64 = stats.mu.astype(np.float64)
    std_f64 = np.sqrt(np.maximum(stats.var, 0)).astype(np.float64)
    n_img = stats.n
    params = noise_equalization_params(max_f64, mean_f64, std_f64, n_img)
    if params is None:
        raise RuntimeError("noise equalization stage benchmark found no valid background")
    sigma_ref, c_n_eff, mask = params
    stages["scalar_setup_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    filled_std = fill_local_mean(std_f64, mask, kernel_size=21)
    stages["fill_local_mean_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    corrected = equalize_noise_correct(
        max_f64,
        filled_std,
        sigma_ref,
        c_n_eff,
        float(np.max(max_f64)),
        0.9,
    )
    stages["equalize_noise_correct_sec"] = time.perf_counter() - t0

    payload = _output_payload(np.round(corrected).astype(max_img.dtype))
    payload["stages_sec"] = stages
    return payload


def _run_star_mask_threshold_once(frames: list[np.ndarray]) -> dict[str, Any]:
    total_nonzero = 0
    last_mask = None
    for frame in frames:
        last_mask = star_shrink_detect_mask(frame, ksize=13)
        total_nonzero += int(np.count_nonzero(last_mask))
    if last_mask is None:
        raise RuntimeError("star mask threshold did not process any frame")
    return {
        "output_shape": list(last_mask.shape),
        "output_dtype": str(last_mask.dtype),
        "mask_nonzero_total": total_nonzero,
    }


def _run_star_mask_dog_once(frames: list[np.ndarray]) -> dict[str, Any]:
    total_nonzero = 0
    last_mask = None
    for frame in frames:
        last_mask = star_mask_dog(frame)
        total_nonzero += int(np.count_nonzero(last_mask))
    if last_mask is None:
        raise RuntimeError("star mask dog did not process any frame")
    return {
        "output_shape": list(last_mask.shape),
        "output_dtype": str(last_mask.dtype),
        "mask_nonzero_total": total_nonzero,
    }


def _run_star_shrink_once(frames: list[np.ndarray]) -> dict[str, Any]:
    configs = {
        "mode": "moderate",
        "detect_method": "threshold",
        "detect_ksize": 13,
        "detect_threshold": 1.0,
        "detect_open": 3,
        "detect_dilate": 0,
        "dog_sigma_small": 1.5,
        "dog_sigma_large": 12.0,
    }
    last = None
    for frame in frames:
        last = _star_shrink_pipeline(_ensure_three_channels(frame), configs)
    if last is None:
        raise RuntimeError("star shrink did not process any frame")
    return _output_payload(last)


def _run_star_shrink_stages_once(frames: list[np.ndarray]) -> dict[str, Any]:
    configs = {
        "mode": "moderate",
        "detect_method": "threshold",
        "detect_ksize": 13,
        "detect_threshold": 1.0,
        "detect_open": 3,
        "detect_dilate": 0,
        "dog_sigma_small": 1.5,
        "dog_sigma_large": 12.0,
    }
    p = SHRINK_MODE_PRESETS[configs["mode"]]
    stages = {
        "ensure_three_channels_sec": 0.0,
        "detect_sec": 0.0,
        "fused_process_sec": 0.0,
    }
    mask_nonzero_total = 0
    last = None
    for frame in frames:
        t0 = time.perf_counter()
        img = _ensure_three_channels(frame)
        stages["ensure_three_channels_sec"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        star_mask = star_shrink_detect_mask(
            img,
            ksize=configs["detect_ksize"],
            threshold_ratio=configs["detect_threshold"],
            open_ksize=configs["detect_open"],
            dilate_ksize=configs["detect_dilate"],
        )
        stages["detect_sec"] += time.perf_counter() - t0
        mask_nonzero_total += int(np.count_nonzero(star_mask))

        t0 = time.perf_counter()
        last = star_shrink_process(
            img,
            star_mask,
            p["shrink_ksize"],
            p.get("shrink_shape", "CIRCLE"),
            p["shrink_times"],
            None if p.get("shrink_ratio", 0.0) == 0.0 else p.get("shrink_ratio", 0.0),
            p["deringing_ksize"],
        )
        stages["fused_process_sec"] += time.perf_counter() - t0

    if last is None:
        raise RuntimeError("star shrink stage benchmark did not process any frame")
    payload = _output_payload(last)
    payload["stages_sec"] = stages
    payload["mask_nonzero_total"] = mask_nonzero_total
    return payload


def _run_satellite_clean_window_once(frames: list[np.ndarray]) -> dict[str, Any]:
    window_size = min(len(frames), 3)
    if window_size % 2 == 0:
        window_size -= 1
    selected = frames[:max(1, window_size)]
    slots: deque[_FrameSlot] = deque(
        _FrameSlot(original=frame, geo=None, H_to_next=np.eye(3, dtype=np.float64))
        for frame in selected
    )
    result = SatelliteCleanOp._process_center(slots, len(slots) // 2, None)
    return _output_payload(result)


def _run_warp_homography_once(frames: list[np.ndarray]) -> dict[str, Any]:
    first = frames[0]
    h, w = first.shape[:2]
    h_src_to_dst = np.array(
        [[1.0, -0.002, 12.0], [0.002, 1.0, -8.0], [1e-6, -8e-7, 1.0]],
        dtype=np.float32,
    )
    last = None
    for frame in frames:
        last = cv2.warpPerspective(frame, h_src_to_dst, (w, h))
    if last is None:
        raise RuntimeError("homography warp did not process any frame")
    return _output_payload(last)


def _run_remap_camera_model_once(
    args: argparse.Namespace,
    frames: list[np.ndarray],
) -> dict[str, Any]:
    first = frames[0]
    src_camera = alignment_bench._make_camera(
        first,
        focal_length_mm=args.focal_length_mm,
        sensor_width_mm=args.sensor_width_mm,
        sensor_height_mm=args.sensor_height_mm,
        distortion_scale=args.distortion_scale,
    )
    dst_camera = src_camera.with_focal_length(args.focal_length_mm * 1.01)
    output_size = (first.shape[1], first.shape[0])
    last = None
    for frame in frames:
        last = alignment_bench.warp_image_by_remap(
            frame,
            src_camera,
            dst_camera,
            output_size,
        )
    if last is None:
        raise RuntimeError("camera model remap did not process any frame")
    return _output_payload(last)


def _time_repeated(
    func: Callable[[], dict[str, Any]],
    *,
    warmup: int,
    repeat: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for _ in range(warmup):
        func()

    samples: list[float] = []
    last_payload: dict[str, Any] = {}
    for _ in range(repeat):
        t0 = time.perf_counter()
        last_payload = func()
        samples.append(time.perf_counter() - t0)
    return summarize_samples(samples), last_payload


def _run_alignment_case(
    args: argparse.Namespace,
    frames: list[np.ndarray],
    *,
    method: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    method_args = argparse.Namespace(**vars(args))
    method_args.method = method
    return alignment_bench._run_repeated(method_args, frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--stars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shift", type=float, default=12.0)
    parser.add_argument("--max-rotation-deg", type=float, default=0.8)
    parser.add_argument("--noise-sigma", type=float, default=1.5)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument(
        "--input-mode",
        choices=["auto", "cache", "images", "synthetic"],
        default="auto",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help="Comma-separated cases or 'all'.",
    )
    parser.add_argument("--same-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--focal-length-mm", type=float, default=35.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--sensor-height-mm", type=float, default=24.0)
    parser.add_argument("--distortion-scale", type=float, default=0.0)
    parser.add_argument(
        "--backend",
        dest="backend",
        choices=["auto", "numpy"],
        default="auto",
        help="Use production backend selection or force numpy custom-op fallbacks.",
    )
    parser.add_argument(
        "--custom-op-backend",
        dest="backend",
        choices=["auto", "numpy"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--log-level", type=str, default="WARNING")
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    requested_cases = parse_cases(args.cases)
    unknown = sorted(set(requested_cases) - set(CASE_NAMES))
    if unknown:
        raise ValueError(f"Unknown pipeline case(s): {unknown}. Available: {CASE_NAMES}")

    frames, input_source = prepare_alignment_frames(
        frames=args.frames,
        height=args.height,
        width=args.width,
        stars=args.stars,
        seed=args.seed,
        channels=args.channels,
        max_shift=args.max_shift,
        max_rotation_deg=args.max_rotation_deg,
        noise_sigma=args.noise_sigma,
        input_dir=args.input_dir,
        input_mode=args.input_mode,
    )

    results: dict[str, dict[str, Any]] = {}
    pipeline_summaries: dict[str, Any] = {}

    with _custom_op_backend(args.backend):
        if "alignment_homography" in requested_cases:
            alignment_results, summary = _run_alignment_case(args, frames, method="homography")
            results["alignment_homography"] = alignment_results["alignment_pipeline"]
            pipeline_summaries["alignment_homography"] = summary

        if "alignment_camera_model" in requested_cases:
            alignment_results, summary = _run_alignment_case(args, frames, method="camera_model")
            results["alignment_camera_model"] = alignment_results["alignment_pipeline"]
            pipeline_summaries["alignment_camera_model"] = summary

        runners: dict[str, Callable[[], dict[str, Any]]] = {
            "stack_max": lambda: _run_max_stack_once(frames),
            "stack_min": lambda: _run_min_stack_once(frames),
            "stack_mean": lambda: _run_mean_stack_once(frames),
            "stack_mean_weighted": lambda: _run_mean_weighted_stack_once(frames),
            "stack_mean_masked": lambda: _run_mean_masked_stack_once(frames),
            "stack_median": lambda: _run_median_stack_once(frames),
            "stack_sigma_clip_fused": lambda: _run_sigma_clip_fused_stack_once(frames),
            "stack_huber": lambda: _run_huber_stack_once(frames),
            "calibration_full": lambda: _run_calibration_full_once(frames),
            "noise_equalization": lambda: _run_noise_equalization_once(frames),
            "noise_equalization_stages": lambda: _run_noise_equalization_stages_once(frames),
            "star_mask_threshold": lambda: _run_star_mask_threshold_once(frames),
            "star_mask_dog": lambda: _run_star_mask_dog_once(frames),
            "star_shrink": lambda: _run_star_shrink_once(frames),
            "star_shrink_stages": lambda: _run_star_shrink_stages_once(frames),
            "satellite_clean_window": lambda: _run_satellite_clean_window_once(frames),
            "warp_homography": lambda: _run_warp_homography_once(frames),
            "remap_camera_model": lambda: _run_remap_camera_model_once(args, frames),
        }
        for case_name in requested_cases:
            if case_name not in runners:
                continue
            results[case_name], pipeline_summaries[case_name] = _time_repeated(
                runners[case_name],
                warmup=args.warmup,
                repeat=args.repeat,
            )

    annotate_case_units(
        results,
        {
            case_name: {"unit": "input_frame", "count": len(frames)}
            for case_name in requested_cases
        },
    )

    return {
        "suite": SUITE_ID,
        "env": collect_env_info(),
        "custom_ops": custom_ops_build_info(),
        "config": {
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "channels": args.channels,
            "stars": args.stars,
            "seed": args.seed,
            "max_shift": args.max_shift,
            "max_rotation_deg": args.max_rotation_deg,
            "noise_sigma": args.noise_sigma,
            "input_dir": args.input_dir,
            "input_mode": args.input_mode,
            "cases": requested_cases,
            "same_camera": args.same_camera,
            "focal_length_mm": args.focal_length_mm,
            "sensor_width_mm": args.sensor_width_mm,
            "sensor_height_mm": args.sensor_height_mm,
            "distortion_scale": args.distortion_scale,
            "backend": args.backend,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "log_level": args.log_level,
        },
        "input_source": input_source,
        "results": results,
        "pipelines": pipeline_summaries,
        "terminal_cases": requested_cases,
        "terminal_mode": "pipeline_totals",
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run(args)
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
