"""Compare deterministic per-edge BA sampling caps on one local sequence."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from tools.debug.debug_norma_ba import _camera_payload, _read_exif
from hoshicore.component.dataloader import load_img
from hoshicore.component.norma.bundle import (
    BundleAdjustmentError,
    BundleFrame,
    _build_edges,
    _connected_component,
    _edge_residuals,
    _sample_edge_pairs,
    _solve_bundle_parameters,
)
from hoshicore.component.norma.detection import detect_star_points
from hoshicore.component.norma.frame_align import (
    CameraInitializationPolicy,
    build_camera_candidate,
)
from hoshicore.component.norma.geometry_view import to_gray_f64
from hoshicore.component.norma.intrinsics_from_exif import lens_type_from_exif
from hoshicore.component.norma.optimization import CameraOptimizationPolicy


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--reference-index", type=int, required=True)
    parser.add_argument("--caps", default="128,256,512,1024")
    parser.add_argument("--pair-offsets", default="1,2,4")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--fallback-focal-equiv-mm", type=float, default=20.0)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark_results/ba_sampling.json"))
    return parser.parse_args()


def _image_paths(folder: Path) -> list[Path]:
    suffixes = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in suffixes)


def _masked(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return image
    runtime = mask
    if runtime.ndim == 2:
        runtime = np.repeat(runtime[..., None], image.shape[2], axis=-1)
    runtime = cv2.resize(
        runtime.astype(np.float32), (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST) > 0.5
    return np.multiply(image, runtime)


def _sample_edges(edges, cap: int, width: int, height: int, seed: int):
    return [
        _sample_edge_pairs(
            edge, cap, width, height,
            (seed + edge.first_index * 1009
             + edge.second_index * 9176) % (2 ** 32))
        for edge in edges
    ]


def _rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(first @ second.T) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _full_residual_summary(edges, rotations, camera) -> dict[str, float | int]:
    values = [_edge_residuals(edge, rotations, camera) for edge in edges
              if edge.first_index in rotations and edge.second_index in rotations]
    residuals = np.concatenate(values)
    edge_p90 = np.asarray([np.percentile(value, 90) for value in values])
    return {
        "evaluated_pairs": int(sum(len(value) for value in values)),
        "point_p50_deg": math.degrees(float(np.percentile(residuals, 50))),
        "point_p90_deg": math.degrees(float(np.percentile(residuals, 90))),
        "point_p99_deg": math.degrees(float(np.percentile(residuals, 99))),
        "edge_p90_median_deg": math.degrees(float(np.median(edge_p90))),
        "edge_p90_max_deg": math.degrees(float(np.max(edge_p90))),
    }


def main() -> int:
    args = _args()
    logger.remove()
    caps = sorted({int(value) for value in args.caps.split(",")})
    offsets = tuple(int(value) for value in args.pair_offsets.split(","))
    paths = _image_paths(args.input_dir)
    if not 0 <= args.reference_index < len(paths):
        raise ValueError("reference index is outside the discovered sequence")
    mask = load_img(str(args.mask)) if args.mask else None
    observations = []
    started = time.perf_counter()
    for index, path in enumerate(paths):
        print(f"detect [{index + 1}/{len(paths)}] {path.name}", flush=True)
        image = load_img(str(path))
        tags = _read_exif(path)
        stars = detect_star_points(to_gray_f64(_masked(image, mask)))
        observations.append((index, stars, image.shape, tags))
    detection_seconds = time.perf_counter() - started

    _, _, shape, reference_tags = observations[args.reference_index]
    policy = CameraInitializationPolicy(
        lens_type=lens_type_from_exif(reference_tags),
        fallback_focal_equiv_mm=args.fallback_focal_equiv_mm,
        optimize_focal=True, optimize_distortion=True,
        optimize_principal_point=False)
    candidate = build_camera_candidate(
        reference_tags, shape, "distortion", init_policy=policy)
    frames = [BundleFrame(index, stars, candidate)
              for index, stars, _, _ in observations]

    edge_started = time.perf_counter()
    edges, sequence_scale = _build_edges(
        frames, offsets, args.random_seed, max_pairs_per_edge=None)
    edge_seconds = time.perf_counter() - edge_started
    accepted = [edge for edge in edges if edge.error is None]
    component = _connected_component(accepted, args.reference_index)
    accepted = [edge for edge in accepted
                if edge.first_index in component and edge.second_index in component]
    camera = candidate.camera.with_focal_length(
        candidate.camera.intrinsics.focal_length_mm * sequence_scale)
    requested = candidate.optimization_policy
    focal_only = CameraOptimizationPolicy(True, False, False, 0)

    report = {
        "input_dir": str(args.input_dir), "files": len(paths),
        "reference_index": args.reference_index, "caps": caps,
        "full_edge_count": len(accepted),
        "full_pair_count": int(sum(len(edge.first_pts) for edge in accepted)),
        "timings_seconds": {
            "detect": detection_seconds, "build_edges": edge_seconds},
        "results": [],
    }
    rotations_by_cap = {}
    for cap in caps:
        sampled = _sample_edges(
            accepted, cap, shape[1], shape[0], args.random_seed)
        solve_started = time.perf_counter()
        mode = "requested"
        reason = None
        try:
            solved_camera, rotations, retained, condition = (
                _solve_bundle_parameters(
                    sampled, args.reference_index, component, camera,
                    requested, 300))
            solved_policy = requested
        except BundleAdjustmentError as exc:
            mode = "focal_fallback"
            reason = str(exc)
            solved_camera, rotations, retained, condition = (
                _solve_bundle_parameters(
                    sampled, args.reference_index, component, camera,
                    focal_only, 300))
            solved_policy = focal_only
        solve_seconds = time.perf_counter() - solve_started
        rotations_by_cap[cap] = rotations
        row = {
            "cap": cap,
            "sampled_pairs": int(sum(len(edge.first_pts) for edge in sampled)),
            "retained_edges": len(retained),
            "solve_seconds": solve_seconds,
            "camera_solve_mode": mode,
            "camera_fallback_reason": reason,
            "active_camera_parameter_count": (
                1 if solved_policy == focal_only else 5),
            "observability_condition": condition,
            "camera": _camera_payload(solved_camera),
            "full_residual": _full_residual_summary(
                accepted, rotations, solved_camera),
        }
        report["results"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    reference_cap = max(caps)
    reference_rotations = rotations_by_cap[reference_cap]
    for row in report["results"]:
        differences = [
            _rotation_difference_deg(rotation, reference_rotations[index])
            for index, rotation in rotations_by_cap[row["cap"]].items()
            if index in reference_rotations
        ]
        row["rotation_delta_vs_max_cap_deg"] = {
            "median": float(np.median(differences)),
            "p90": float(np.percentile(differences, 90)),
            "max": float(np.max(differences)),
        }
    report["timings_seconds"]["total"] = time.perf_counter() - started
    args.output.write_text(json.dumps(
        report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
