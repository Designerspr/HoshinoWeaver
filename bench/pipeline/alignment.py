"""End-to-end alignment pipeline benchmark.

This suite measures the production alignment path as a whole. It does not
compare numpy/compiled implementations directly; backend selection is left to
the production wrappers used by detect/remap.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
from hoshicore._custom_op import build_info as custom_ops_build_info
from hoshicore._custom_op._dispatch import fallback_preference
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
from hoshicore._custom_op.backend_registry import select_backend
from hoshicore._custom_op.ops.detection import (
    star_detect_fused_pixel_components_compiled,
)
from hoshicore._custom_op.ops.remap import camera_model_remap_compiled
from hoshicore.component.norma.alignment import (match_star_pairs,
                                                  optimize_alignment)
from hoshicore.component.norma.frame_align import AlignmentError, _check_star_count
from hoshicore.component.norma.geometry_view import make_geometry
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics


SUITE_ID = "pipeline.alignment"
CASE_NAMES = [
    "alignment_pipeline",
]
DEFAULT_CASES = CASE_NAMES
PIPELINE_METHODS = ("homography", "camera_model")


@dataclass
class PipelineRunResult:
    stage_times: dict[str, float] = field(default_factory=dict)
    aligned_frames: int = 0
    failed_frames: int = 0
    output_shape: list[int] | None = None
    output_dtype: str | None = None


def _add_stage_time(stage_times: dict[str, float], name: str, elapsed: float) -> None:
    stage_times[name] = stage_times.get(name, 0.0) + elapsed


def _time_stage(
    stage_times: dict[str, float],
    name: str,
    func: Callable[[], Any],
) -> Any:
    t0 = time.perf_counter()
    try:
        return func()
    finally:
        _add_stage_time(stage_times, name, time.perf_counter() - t0)


def _make_geometry_eager(frame: np.ndarray, camera: CameraModel):
    geo = make_geometry(frame, camera=camera)
    _ = len(geo.positions)
    _ = len(geo.volumes)
    return geo


def _make_camera(
    frame: np.ndarray,
    *,
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    distortion_scale: float,
) -> CameraModel:
    height, width = frame.shape[:2]
    distortion = Distortion()
    if distortion_scale != 0.0:
        distortion = Distortion.from_cv2(
            np.array(
                [
                    0.01 * distortion_scale,
                    -0.0015 * distortion_scale,
                    0.0008 * distortion_scale,
                    -0.0004 * distortion_scale,
                    0.0001 * distortion_scale,
                ],
                dtype=np.float64,
            )
        )
    return CameraModel(
        intrinsics=Intrinsics(
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            image_width_px=width,
            image_height_px=height,
        ),
        distortion=distortion,
    )


def _run_alignment_pipeline_once(
    frames: list[np.ndarray],
    *,
    method: str,
    same_camera: bool,
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    distortion_scale: float,
) -> PipelineRunResult:
    if not frames:
        raise ValueError("alignment pipeline benchmark requires at least one frame")

    stage_times: dict[str, float] = {}
    reference = frames[0]
    output_size = (reference.shape[1], reference.shape[0])
    ref_camera = _make_camera(
        reference,
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        distortion_scale=distortion_scale if method == "camera_model" else 0.0,
    )
    ref_geo = _time_stage(
        stage_times,
        "reference_geometry",
        lambda: _make_geometry_eager(reference, ref_camera),
    )

    result = PipelineRunResult(
        stage_times=stage_times,
        aligned_frames=1,
        output_shape=list(reference.shape),
        output_dtype=str(reference.dtype),
    )

    for frame in frames[1:]:
        try:
            src_camera = _make_camera(
                frame,
                focal_length_mm=focal_length_mm,
                sensor_width_mm=sensor_width_mm,
                sensor_height_mm=sensor_height_mm,
                distortion_scale=(distortion_scale
                                  if method == "camera_model" else 0.0),
            )
            src_geo = _time_stage(
                stage_times,
                "source_geometry",
                lambda: _make_geometry_eager(frame, src_camera),
            )
            _time_stage(stage_times, "star_count_check", lambda: _check_star_count(ref_geo, src_geo))
            match = _time_stage(stage_times, "match", lambda: match_star_pairs(ref_geo, src_geo))

            if method == "homography":
                def warp_homography() -> np.ndarray:
                    if match.homography is None:
                        raise AlignmentError(
                            "Homography is unavailable for this camera pair")
                    h_src_to_ref = np.linalg.inv(match.homography)
                    return cv2.warpPerspective(frame, h_src_to_ref, output_size)

                aligned = _time_stage(stage_times, "warp", warp_homography)
            else:
                def optimize() -> Any:
                    try:
                        return optimize_alignment(
                            match,
                            ref_camera,
                            src_camera,
                            same_camera=same_camera,
                        )
                    except Exception as exc:
                        raise AlignmentError(f"Optimization failed: {exc}") from exc

                optimized = _time_stage(stage_times, "optimize", optimize)
                aligned = _time_stage(
                    stage_times,
                    "remap",
                    lambda: optimized.ref_camera.project_image_from_camera(
                        optimized.src_camera,
                        frame,
                        output_size,
                        rotation_dst_to_src=optimized.rotation_ref_to_src,
                    ),
                )

            result.aligned_frames += 1
            result.output_shape = list(aligned.shape)
            result.output_dtype = str(aligned.dtype)
        except AlignmentError:
            result.failed_frames += 1

    return result


def _summarize_stage_samples(
    samples_by_case: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    return {
        case_name: summarize_samples(samples)
        for case_name, samples in samples_by_case.items()
        if samples
    }


def _run_repeated(
    args: argparse.Namespace,
    frames: list[np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    samples_by_case: dict[str, list[float]] = {"alignment_pipeline": []}
    last_result: PipelineRunResult | None = None

    for _ in range(args.warmup):
        _run_alignment_pipeline_once(
            frames,
            method=args.method,
            same_camera=args.same_camera,
            focal_length_mm=args.focal_length_mm,
            sensor_width_mm=args.sensor_width_mm,
            sensor_height_mm=args.sensor_height_mm,
            distortion_scale=args.distortion_scale,
        )

    for _ in range(args.repeat):
        t0 = time.perf_counter()
        result = _run_alignment_pipeline_once(
            frames,
            method=args.method,
            same_camera=args.same_camera,
            focal_length_mm=args.focal_length_mm,
            sensor_width_mm=args.sensor_width_mm,
            sensor_height_mm=args.sensor_height_mm,
            distortion_scale=args.distortion_scale,
        )
        samples_by_case["alignment_pipeline"].append(time.perf_counter() - t0)
        for stage_name, elapsed in result.stage_times.items():
            samples_by_case[f"stage.{stage_name}"] = samples_by_case.get(
                f"stage.{stage_name}", [])
            samples_by_case[f"stage.{stage_name}"].append(elapsed)
        last_result = result

    results = _summarize_stage_samples(samples_by_case)
    annotate_case_units(
        results,
        {
            "alignment_pipeline": {
                "unit": "input_frame",
                "count": len(frames),
            },
            **{
                case_name: {
                    "unit": "aligned_frame",
                    "count": max(1, len(frames) - 1),
                }
                for case_name in results
                if case_name.startswith("stage.") and case_name != "stage.reference_geometry"
            },
        },
    )
    if last_result is None:
        raise ValueError("--repeat must be at least 1")

    summary = {
        "aligned_frames": last_result.aligned_frames,
        "failed_frames": last_result.failed_frames,
        "output_shape": last_result.output_shape,
        "output_dtype": last_result.output_dtype,
    }
    return results, summary


def _backend_diagnostics() -> dict[str, Any]:
    preference = fallback_preference()
    diagnostics: dict[str, Any] = {
        "preference": preference,
        "registry_selection_only": True,
        "note": (
            "Registry selection describes preferred native candidates. "
            "Production wrappers may still fall back at runtime; see runtime_probes."
        ),
    }
    for logical_op in (
        "star_detect_fused_pixel_components",
        "camera_model_remap",
    ):
        selection = select_backend(logical_op, preference)
        diagnostics[logical_op] = {
            "candidate_backend": selection.backend,
            "native_candidate_available": selection.native,
            "kernel_name": (
                selection.candidate.kernel_name
                if selection.candidate is not None
                else None
            ),
            "reason": selection.reason,
        }
    return diagnostics


def _probe_cuda_runtime() -> dict[str, Any]:
    probes: dict[str, Any] = {}

    def record(name: str, func: Callable[[], Any]) -> None:
        try:
            func()
            probes[name] = {"status": "ok"}
        except Exception as exc:
            unavailable = (
                isinstance(exc, RuntimeError)
                and (
                    is_cuda_runtime_unavailable_error(exc)
                    or "compiled cuda custom op backend is unavailable"
                    in str(exc).lower()
                )
            )
            probes[name] = {
                "status": "unavailable" if unavailable else "error",
                "error": str(exc),
            }

    def probe_detection() -> None:
        image = np.arange(512 * 512, dtype=np.float64).reshape(512, 512)
        star_detect_fused_pixel_components_compiled(
            image,
            None,
            1.0,
            gaussian_ksize=3,
            sigma=1.0,
        )

    def probe_remap() -> None:
        image = np.zeros((8, 8, 1), dtype=np.float32)
        camera_model_remap_compiled(
            image=image,
            out_height=8,
            out_width=8,
            fx_src=8.0,
            fy_src=8.0,
            cx_src=3.5,
            cy_src=3.5,
            fx_dst=8.0,
            fy_dst=8.0,
            cx_dst=3.5,
            cy_dst=3.5,
            rotation_dst_to_src=np.eye(3, dtype=np.float32),
            src_dist_coeffs=None,
            dst_dist_coeffs=None,
        )

    record("star_detect_fused_pixel_components_cuda", probe_detection)
    record("camera_model_remap_cuda", probe_remap)
    return probes


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
        "--method",
        choices=["homography", "camera_model", "all"],
        default="camera_model",
    )
    parser.add_argument("--same-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--focal-length-mm", type=float, default=35.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--sensor-height-mm", type=float, default=24.0)
    parser.add_argument("--distortion-scale", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="WARNING")
    parser.add_argument("--probe-backends", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def _prefix_results(
    results: dict[str, dict[str, Any]],
    method: str,
) -> dict[str, dict[str, Any]]:
    return {
        f"{method}.{case_name}": payload
        for case_name, payload in results.items()
    }


def _case_order_for_results(results: dict[str, dict[str, Any]]) -> list[str]:
    return [
        "alignment_pipeline",
        *[case_name for case_name in results if case_name.startswith("stage.")],
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

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

    if args.method == "all":
        results: dict[str, dict[str, Any]] = {}
        pipeline_summaries: dict[str, Any] = {}
        terminal_cases = []
        for method in PIPELINE_METHODS:
            method_args = argparse.Namespace(**vars(args))
            method_args.method = method
            method_results, method_summary = _run_repeated(method_args, frames)
            results.update(_prefix_results(method_results, method))
            pipeline_summaries[method] = method_summary
            terminal_cases.append(f"{method}.alignment_pipeline")
        pipeline_summary: dict[str, Any] | None = None
        cases = [
            f"{method}.{case_name}"
            for method in PIPELINE_METHODS
            for case_name in _case_order_for_results({
                key.removeprefix(f"{method}."): value
                for key, value in results.items()
                if key.startswith(f"{method}.")
            })
        ]
    else:
        results, single_summary = _run_repeated(args, frames)
        pipeline_summary = single_summary
        pipeline_summaries = {}
        terminal_cases = None
        cases = _case_order_for_results(results)

    backend_diagnostics = _backend_diagnostics()
    if args.probe_backends:
        backend_diagnostics["runtime_probes"] = _probe_cuda_runtime()

    report: dict[str, object] = {
        "suite": "pipeline.alignment",
        "env": collect_env_info(),
        "custom_ops": custom_ops_build_info(),
        "backend_diagnostics": backend_diagnostics,
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
            "method": args.method,
            "same_camera": args.same_camera,
            "focal_length_mm": args.focal_length_mm,
            "sensor_width_mm": args.sensor_width_mm,
            "sensor_height_mm": args.sensor_height_mm,
            "distortion_scale": args.distortion_scale,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "log_level": args.log_level,
            "probe_backends": args.probe_backends,
            "cases": cases,
        },
        "input_source": input_source,
        "results": results,
    }
    if pipeline_summary is not None:
        report["pipeline"] = pipeline_summary
    if pipeline_summaries:
        report["pipelines"] = pipeline_summaries
    if terminal_cases is not None:
        report["terminal_cases"] = terminal_cases
        report["terminal_case_labels"] = {
            f"{method}.alignment_pipeline": method
            for method in PIPELINE_METHODS
        }
        report["terminal_mode"] = "pipeline_totals"
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run(args)
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
