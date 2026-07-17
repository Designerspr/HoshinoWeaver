"""Alignment pipeline benchmark with stage-level profiling."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Any

import cv2
import numpy as np
from loguru import logger

from bench.common import (
    annotate_case_units,
    collect_env_info,
    print_or_save_report,
    resolve_existing_frames,
    run_benchmark,
)
from bench.data_tools.starfield import generate_starfield_frames
from hoshicore._custom_op.ops.detection import star_detect_threshold_morph_numpy
from hoshicore.component.norma.alignment import match_star_pairs, optimize_alignment
from hoshicore.component.norma.detection import (
    _detect_star_points_cuda_hybrid,
    _detect_star_points_opencv,
    _wavelet_dec_rec,
    detect_star_points,
)
from hoshicore.component.norma.frame_align import (
    AlignmentCameraCandidate,
    align_frame_camera_model,
    align_frame_homography,
)
from hoshicore.component.norma.geometry_view import (GeometryView,
                                                      make_geometry,
                                                      to_gray_f64)
from hoshicore.component.norma.optimization import CameraOptimizationPolicy
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics


DEFAULT_CASE_NAMES = [
    "detect_stream",
    "detect_prepare_stream",
    "detect_wavelet_stream",
    "detect_extract_stream",
    "detect_bandpass_stream",
    "detect_threshold_morph_stream",
    "detect_contour_stream",
    "detect_ellipse_intensity_stream",
    "features_stream",
    "geometry_stream",
    "match_stream",
    "warp_stream",
    "homography_pipeline",
    "optimization_stream",
    "remap_stream",
    "camera_model_pipeline",
]
GPU_CASE_NAMES = [
    "detect_cuda_hybrid_stream",
]
BASELINE_CASE_NAMES = [
    "detect_opencv_stream",
]
QUALITY_CASE_NAME = "detect_cuda_hybrid_vs_contour_quality"
CASE_NAMES = [
    *DEFAULT_CASE_NAMES,
    *GPU_CASE_NAMES,
    *BASELINE_CASE_NAMES,
    QUALITY_CASE_NAME,
]
DEFAULT_CASES = DEFAULT_CASE_NAMES
SUITE_ID = "cpu.alignment"
ALL_FRAME_CASE_NAMES = {
    "detect_stream",
    "detect_prepare_stream",
    "detect_wavelet_stream",
    "detect_extract_stream",
    "detect_bandpass_stream",
    "detect_threshold_morph_stream",
    "detect_contour_stream",
    "detect_ellipse_intensity_stream",
    "features_stream",
    "geometry_stream",
    "detect_cuda_hybrid_stream",
    "detect_opencv_stream",
}
ALIGNED_FRAME_CASE_NAMES = {
    "match_stream",
    "warp_stream",
    "homography_pipeline",
    "optimization_stream",
    "remap_stream",
    "camera_model_pipeline",
}
CUDA_HYBRID_QUALITY_THRESHOLDS = {
    "max_count_diff_ratio": 0.15,
    "min_contour_to_cuda_hybrid_recall_1px": 0.95,
    "max_contour_to_cuda_hybrid_p95_px": 0.1,
    "max_pair_diff_ratio": 0.20,
    "max_homography_p95_px": 0.30,
}


@dataclasses.dataclass
class DetectPayload:
    img_gray: np.ndarray
    img_blr: np.ndarray
    mask: np.ndarray
    resize_factor: float


@dataclasses.dataclass
class DetectBandpassPayload:
    img_rec: np.ndarray
    mask: np.ndarray


@dataclasses.dataclass
class DetectThresholdPayload:
    img_rec: np.ndarray
    bw: np.ndarray


@dataclasses.dataclass
class DetectContourPayload:
    img_rec: np.ndarray
    bw: np.ndarray
    contours: list[np.ndarray]


def alignment_case_units(
    case_names: list[str],
    frame_count: int,
) -> dict[str, dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    for case_name in case_names:
        if case_name in ALL_FRAME_CASE_NAMES:
            units[case_name] = {"unit": "frame", "count": frame_count}
        elif case_name in ALIGNED_FRAME_CASE_NAMES:
            units[case_name] = {
                "unit": "aligned_frame",
                "count": max(0, frame_count - 1),
            }
    return units


def _prepare_detect_payload(frame: np.ndarray,
                            *,
                            resize_length: int = 10000,
                            gaussian_ksize: int = 9,
                            sigma: float = 2.0) -> DetectPayload:
    img_gray = to_gray_f64(frame)
    img_shape = img_gray.shape

    img_blr = cv2.GaussianBlur(img_gray, (gaussian_ksize, gaussian_ksize),
                               sigma)
    img_blr_mean = np.mean(img_blr)
    img_blr_range = np.max(img_blr) - np.min(img_blr)
    img_blr = (img_blr - img_blr_mean) / img_blr_range

    resize_factor = 1.0
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2.0

    mask = np.ones(img_shape, dtype=bool)

    return DetectPayload(
        img_gray=img_gray,
        img_blr=img_blr,
        mask=mask,
        resize_factor=resize_factor,
    )


def _prepare_detect_bandpass_payload(payload: DetectPayload) -> DetectBandpassPayload:
    img_rec = _wavelet_dec_rec(
        payload.img_blr, resize_factor=payload.resize_factor) * payload.mask
    return DetectBandpassPayload(img_rec=img_rec, mask=payload.mask)


def _threshold_morph_detect_image(payload: DetectBandpassPayload) -> np.ndarray:
    return star_detect_threshold_morph_numpy(payload.img_rec, payload.mask)


def _prepare_detect_threshold_payload(
        payload: DetectBandpassPayload) -> DetectThresholdPayload:
    return DetectThresholdPayload(
        img_rec=payload.img_rec,
        bw=_threshold_morph_detect_image(payload),
    )


def _find_detect_contours(bw: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [contour for contour in contours if len(contour) > 5]


def _prepare_detect_contour_payload(
        payload: DetectThresholdPayload) -> DetectContourPayload:
    return DetectContourPayload(
        img_rec=payload.img_rec,
        bw=payload.bw,
        contours=_find_detect_contours(payload.bw),
    )


def _measure_detect_ellipse_intensity(payload: DetectContourPayload) -> None:
    contours = payload.contours
    if not contours:
        return

    elps = [cv2.fitEllipse(contour) for contour in contours]
    areas = np.array([
        cv2.contourArea(contour) + 0.5 * len(contour) for contour in contours
    ])
    eccentricities = np.sqrt(
        np.array([1 - (elp[1][0] / elp[1][1])**2 for elp in elps]))
    _ = eccentricities

    mask_img = np.zeros(payload.bw.shape, np.uint8)
    intensities = np.zeros(areas.shape)
    for i, contour in enumerate(contours):
        cv2.drawContours(mask_img, contours, i, 255, -1)
        rect = cv2.boundingRect(contour)
        val = cv2.mean(
            payload.img_rec[rect[1]:rect[1] + rect[3] + 1,
                            rect[0]:rect[0] + rect[2] + 1],
            mask_img[rect[1]:rect[1] + rect[3] + 1,
                     rect[0]:rect[0] + rect[2] + 1])
        mask_img[rect[1]:rect[1] + rect[3] + 1,
                 rect[0]:rect[0] + rect[2] + 1] = 0
        intensities[i] = val[0]
    _ = intensities


def _extract_detect_features(payload: DetectPayload) -> None:
    img_rec = _wavelet_dec_rec(
        payload.img_blr, resize_factor=payload.resize_factor) * payload.mask
    bandpass_payload = DetectBandpassPayload(img_rec=img_rec, mask=payload.mask)
    threshold_payload = _prepare_detect_threshold_payload(bandpass_payload)
    contour_payload = _prepare_detect_contour_payload(threshold_payload)
    _measure_detect_ellipse_intensity(contour_payload)


def prepare_alignment_frames(
    *,
    frames: int,
    height: int,
    width: int,
    channels: int,
    stars: int,
    seed: int,
    max_shift: float,
    max_rotation_deg: float,
    noise_sigma: float,
    input_dir: str | None,
    input_mode: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if input_mode not in {"auto", "cache", "images", "synthetic"}:
        raise ValueError(f"unsupported input_mode: {input_mode}")

    if input_mode in {"auto", "cache", "images"}:
        existing = resolve_existing_frames(
            frames=frames,
            input_dir=input_dir,
            input_mode=input_mode,
        )
        if existing is not None:
            return existing

    if input_mode == "synthetic" or input_mode == "auto":
        generated, meta = generate_starfield_frames(
            frames=frames,
            height=height,
            width=width,
            stars=stars,
            seed=seed,
            channels=channels,
            max_shift=max_shift,
            max_rotation_deg=max_rotation_deg,
            noise_sigma=noise_sigma,
        )
        return generated, {
            "mode": "synthetic_starfield",
            "input_dir": None,
            "requested_input_mode": input_mode,
            "resolved_frames": len(generated),
            "resolved_shape": list(generated[0].shape),
            "resolved_dtype": str(generated[0].dtype),
            "stars": stars,
            "max_shift": max_shift,
            "max_rotation_deg": max_rotation_deg,
            "noise_sigma": noise_sigma,
            "transform_preview": meta[:min(3, len(meta))],
        }

    raise ValueError(f"unsupported input_mode: {input_mode}")


def bench_geometry_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        geo = make_geometry(frame)
        _ = geo.unit_vectors


def bench_detect_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        gray = to_gray_f64(frame)
        _ = detect_star_points(gray)


def bench_detect_opencv_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        gray = to_gray_f64(frame)
        _ = _detect_star_points_opencv(gray)


def bench_detect_cuda_hybrid_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        gray = to_gray_f64(frame)
        _ = _detect_star_points_cuda_hybrid(gray)


def bench_detect_prepare_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        _ = _prepare_detect_payload(frame)


def bench_detect_wavelet_stream(payloads: list[DetectPayload]) -> None:
    for payload in payloads:
        _ = _wavelet_dec_rec(
            payload.img_blr, resize_factor=payload.resize_factor)


def bench_detect_extract_stream(payloads: list[DetectPayload]) -> None:
    for payload in payloads:
        _extract_detect_features(payload)


def bench_detect_bandpass_stream(payloads: list[DetectPayload]) -> None:
    for payload in payloads:
        _ = _prepare_detect_bandpass_payload(payload)


def bench_detect_threshold_morph_stream(
        payloads: list[DetectBandpassPayload]) -> None:
    for payload in payloads:
        _ = _threshold_morph_detect_image(payload)


def bench_detect_contour_stream(payloads: list[DetectThresholdPayload]) -> None:
    for payload in payloads:
        _ = _find_detect_contours(payload.bw)


def bench_detect_ellipse_intensity_stream(
        payloads: list[DetectContourPayload]) -> None:
    for payload in payloads:
        _measure_detect_ellipse_intensity(payload)


def _nearest_stats(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if len(source) == 0 or len(target) == 0:
        return {
            "source_count": len(source),
            "target_count": len(target),
            "recall_1px": 0.0,
            "recall_2px": 0.0,
            "median_px": None,
            "p95_px": None,
            "max_px": None,
        }
    diff = source[:, None, :] - target[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    nearest = np.min(dist, axis=1)
    return {
        "source_count": len(source),
        "target_count": len(target),
        "recall_1px": float(np.mean(nearest <= 1.0)),
        "recall_2px": float(np.mean(nearest <= 2.0)),
        "median_px": float(np.median(nearest)),
        "p95_px": float(np.percentile(nearest, 95)),
        "max_px": float(np.max(nearest)),
    }


def _project_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    projected = points_h @ homography.T
    return projected[:, :2] / projected[:, 2:3]


def _homography_delta(
    h1: np.ndarray,
    h2: np.ndarray,
    shape: tuple[int, int],
) -> dict[str, float]:
    h1 = h1 / h1[2, 2]
    h2 = h2 / h2[2, 2]
    height, width = shape
    grid_x = np.linspace(0, width - 1, 16)
    grid_y = np.linspace(0, height - 1, 12)
    xx, yy = np.meshgrid(grid_x, grid_y)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    delta = np.linalg.norm(
        _project_homography(h1, points) - _project_homography(h2, points),
        axis=1,
    )
    return {
        "median_px": float(np.median(delta)),
        "p95_px": float(np.percentile(delta, 95)),
        "max_px": float(np.max(delta)),
    }


def _match_detected_stars(ref, src, shape: tuple[int, int]):
    height, width = shape
    camera = CameraModel(Intrinsics(20.0, 36.0, 24.0, width, height))
    image_gray = np.zeros(shape, dtype=np.float64)
    ref_geo = GeometryView(image_gray, camera, detected_stars=ref)
    src_geo = GeometryView(image_gray, camera, detected_stars=src)
    return match_star_pairs(ref_geo, src_geo)


def compare_detect_quality(frames: list[np.ndarray]) -> dict[str, Any]:
    contour_results = []
    cuda_hybrid_results = []
    per_frame = []
    for frame in frames:
        gray = to_gray_f64(frame)
        contour = _detect_star_points_opencv(gray)
        cuda_hybrid = _detect_star_points_cuda_hybrid(gray)
        contour_results.append(contour)
        cuda_hybrid_results.append(cuda_hybrid)
        contour_to_cuda_hybrid = _nearest_stats(contour.positions,
                                             cuda_hybrid.positions)
        cuda_hybrid_to_contour = _nearest_stats(cuda_hybrid.positions,
                                             contour.positions)
        count_diff_ratio = 0.0
        if len(contour.positions) > 0:
            count_diff_ratio = (
                len(cuda_hybrid.positions) -
                len(contour.positions)) / len(contour.positions)
        per_frame.append({
            "contour_count": len(contour.positions),
            "cuda_hybrid_count": len(cuda_hybrid.positions),
            "count_diff_ratio": float(count_diff_ratio),
            "contour_to_cuda_hybrid": contour_to_cuda_hybrid,
            "cuda_hybrid_to_contour": cuda_hybrid_to_contour,
        })

    shape = frames[0].shape[:2]
    pair_reports = []
    for idx in range(1, len(frames)):
        try:
            contour_match = _match_detected_stars(
                contour_results[0], contour_results[idx], shape)
            cuda_hybrid_match = _match_detected_stars(
                cuda_hybrid_results[0], cuda_hybrid_results[idx], shape)
        except Exception as exc:
            pair_reports.append({"index": idx, "error": f"{type(exc).__name__}: {exc}"})
            continue
        pair_diff_ratio = 0.0
        if len(contour_match.pair_idx) > 0:
            pair_diff_ratio = (
                len(cuda_hybrid_match.pair_idx) -
                len(contour_match.pair_idx)) / len(contour_match.pair_idx)
        pair_reports.append({
            "index": idx,
            "contour_pairs": len(contour_match.pair_idx),
            "cuda_hybrid_pairs": len(cuda_hybrid_match.pair_idx),
            "pair_diff_ratio": float(pair_diff_ratio),
            "homography_delta": _homography_delta(
                contour_match.homography,
                cuda_hybrid_match.homography,
                shape,
            ),
        })

    return {
        "per_frame": per_frame,
        "pairs": pair_reports,
    }


def summarize_detect_quality(quality: dict[str, Any]) -> dict[str, Any]:
    per_frame = quality.get("per_frame", [])
    pairs = quality.get("pairs", [])
    failed_pairs = [pair for pair in pairs if "error" in pair]
    successful_pairs = [pair for pair in pairs if "error" not in pair]

    max_count_diff_ratio = 0.0
    min_contour_to_cuda_hybrid_recall_1px = 1.0
    max_contour_to_cuda_hybrid_p95_px = 0.0
    for frame in per_frame:
        max_count_diff_ratio = max(
            max_count_diff_ratio,
            abs(float(frame.get("count_diff_ratio", 0.0))),
        )
        contour_to_cuda_hybrid = frame.get("contour_to_cuda_hybrid", {})
        min_contour_to_cuda_hybrid_recall_1px = min(
            min_contour_to_cuda_hybrid_recall_1px,
            float(contour_to_cuda_hybrid.get("recall_1px", 0.0)),
        )
        p95_px = contour_to_cuda_hybrid.get("p95_px")
        if p95_px is not None:
            max_contour_to_cuda_hybrid_p95_px = max(
                max_contour_to_cuda_hybrid_p95_px,
                float(p95_px),
            )

    max_pair_diff_ratio = 0.0
    max_homography_p95_px = 0.0
    for pair in successful_pairs:
        max_pair_diff_ratio = max(
            max_pair_diff_ratio,
            abs(float(pair.get("pair_diff_ratio", 0.0))),
        )
        homography_delta = pair.get("homography_delta", {})
        max_homography_p95_px = max(
            max_homography_p95_px,
            float(homography_delta.get("p95_px", 0.0)),
        )

    metrics = {
        "max_count_diff_ratio": max_count_diff_ratio,
        "min_contour_to_cuda_hybrid_recall_1px":
            min_contour_to_cuda_hybrid_recall_1px,
        "max_contour_to_cuda_hybrid_p95_px": max_contour_to_cuda_hybrid_p95_px,
        "max_pair_diff_ratio": max_pair_diff_ratio,
        "max_homography_p95_px": max_homography_p95_px,
        "failed_pair_count": len(failed_pairs),
    }
    passed = (
        len(failed_pairs) == 0
        and max_count_diff_ratio <= CUDA_HYBRID_QUALITY_THRESHOLDS[
            "max_count_diff_ratio"]
        and min_contour_to_cuda_hybrid_recall_1px >= CUDA_HYBRID_QUALITY_THRESHOLDS[
            "min_contour_to_cuda_hybrid_recall_1px"]
        and max_contour_to_cuda_hybrid_p95_px <= CUDA_HYBRID_QUALITY_THRESHOLDS[
            "max_contour_to_cuda_hybrid_p95_px"]
        and max_pair_diff_ratio <= CUDA_HYBRID_QUALITY_THRESHOLDS[
            "max_pair_diff_ratio"]
        and max_homography_p95_px <= CUDA_HYBRID_QUALITY_THRESHOLDS[
            "max_homography_p95_px"]
    )
    return {
        "passed": passed,
        "thresholds": CUDA_HYBRID_QUALITY_THRESHOLDS,
        "metrics": metrics,
    }


def bench_features_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        _ = make_geometry(frame).features


def precompute_geometries(frames: list[np.ndarray]):
    ref = _make_geometry_eager(frames[0])
    rest = [_make_geometry_eager(frame) for frame in frames[1:]]
    return ref, rest


def _make_geometry_eager(frame: np.ndarray):
    geo = make_geometry(frame)
    _ = len(geo.positions)
    _ = len(geo.volumes)
    _ = geo.unit_vectors.shape
    return geo


def build_synthetic_camera(frame: np.ndarray) -> CameraModel:
    height, width = frame.shape[:2]
    intrinsics = Intrinsics(
        focal_length_mm=35.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=width,
        image_height_px=height,
    )
    return CameraModel(intrinsics=intrinsics, distortion=Distortion())


def bench_match_stream_geometries(ref_geo, src_geos) -> None:
    for src_geo in src_geos:
        _ = match_star_pairs(ref_geo, src_geo)


def bench_match_stream(frames: list[np.ndarray]) -> None:
    ref_geo, src_geos = precompute_geometries(frames)
    bench_match_stream_geometries(ref_geo, src_geos)


def bench_warp_stream_geometries(frames: list[np.ndarray], ref_geo, src_geos) -> None:
    ref_arr = frames[0]
    h, w = ref_arr.shape[:2]
    for frame, src_geo in zip(frames[1:], src_geos):
        match = match_star_pairs(ref_geo, src_geo)
        if match.homography is None:
            continue
        H = np.linalg.inv(match.homography)
        _ = cv2.warpPerspective(frame, H, (w, h))


def bench_warp_stream(frames: list[np.ndarray]) -> None:
    ref_geo, src_geos = precompute_geometries(frames)
    bench_warp_stream_geometries(frames, ref_geo, src_geos)


def bench_optimization_stream_geometries(frames: list[np.ndarray], ref_geo, src_geos) -> None:
    ref_arr = frames[0]
    ref_camera = build_synthetic_camera(ref_arr)
    ref_geo = ref_geo.with_camera(ref_camera)
    for frame, src_geo in zip(frames[1:], src_geos):
        src_camera = build_synthetic_camera(frame)
        src_geo = src_geo.with_camera(src_camera)
        match = match_star_pairs(ref_geo, src_geo)
        _ = optimize_alignment(
            match,
            ref_camera,
            src_camera,
            same_camera=True,
        )


def bench_optimization_stream(frames: list[np.ndarray]) -> None:
    ref_geo, src_geos = precompute_geometries(frames)
    bench_optimization_stream_geometries(frames, ref_geo, src_geos)


def bench_homography_pipeline(frames: list[np.ndarray]) -> None:
    ref_arr = frames[0]
    ref_geo = make_geometry(ref_arr)
    for frame in frames[1:]:
        _ = align_frame_homography(frame, ref_geo, ref_arr)


def bench_camera_model_pipeline(frames: list[np.ndarray]) -> None:
    ref_arr = frames[0]
    ref_camera = build_synthetic_camera(ref_arr)
    ref_geo = make_geometry(ref_arr, camera=ref_camera)
    policy = CameraOptimizationPolicy()
    ref_candidate = AlignmentCameraCandidate(ref_camera, policy, "provided")
    for frame in frames[1:]:
        src_camera = build_synthetic_camera(frame)
        src_candidate = AlignmentCameraCandidate(src_camera, policy, "provided")
        _ = align_frame_camera_model(
            frame,
            ref_geo,
            ref_arr,
            ref_candidate,
            src_candidate,
            same_camera=True,
        )


def _prepare_remap_payloads(frames: list[np.ndarray]):
    ref_arr = frames[0]
    ref_camera = build_synthetic_camera(ref_arr)
    ref_geo = make_geometry(ref_arr, camera=ref_camera)
    payloads = []
    for frame in frames[1:]:
        src_camera = build_synthetic_camera(frame)
        src_geo = make_geometry(frame, camera=src_camera)
        match = match_star_pairs(ref_geo, src_geo)
        result = optimize_alignment(
            match,
            ref_camera,
            src_camera,
            same_camera=True,
        )
        payloads.append((frame, result.src_camera, result.ref_camera,
                         (ref_arr.shape[1], ref_arr.shape[0])))
    return payloads


def bench_remap_stream(payloads) -> None:
    for frame, src_camera, dst_camera, output_size in payloads:
        _ = dst_camera.project_image_from_camera(src_camera, frame, output_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--stars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shift", type=float, default=12.0)
    parser.add_argument("--max-rotation-deg", type=float, default=0.8)
    parser.add_argument("--noise-sigma", type=float, default=1.5)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--input-mode",
                        choices=["auto", "cache", "images", "synthetic"],
                        default="auto")
    parser.add_argument("--log-level", type=str, default="WARNING")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    frames, input_source = prepare_alignment_frames(
        frames=args.frames,
        height=args.height,
        width=args.width,
        channels=args.channels,
        stars=args.stars,
        seed=args.seed,
        max_shift=args.max_shift,
        max_rotation_deg=args.max_rotation_deg,
        noise_sigma=args.noise_sigma,
        input_dir=args.input_dir,
        input_mode=args.input_mode,
    )

    unknown_cases = [case for case in args.cases if case not in CASE_NAMES]
    if unknown_cases:
        raise ValueError(
            f"Unknown alignment benchmark case(s): {unknown_cases}. "
            f"Available: {list(CASE_NAMES)}")

    quality_requested = QUALITY_CASE_NAME in args.cases
    benchmark_case_names = [
        case for case in args.cases if case != QUALITY_CASE_NAME
    ]

    detect_payload_cases = {
        "detect_wavelet_stream",
        "detect_extract_stream",
        "detect_bandpass_stream",
        "detect_threshold_morph_stream",
        "detect_contour_stream",
        "detect_ellipse_intensity_stream",
    }
    detect_payloads = None
    if any(case in benchmark_case_names for case in detect_payload_cases):
        detect_payloads = [_prepare_detect_payload(frame) for frame in frames]

    bandpass_payloads = None
    if any(case in benchmark_case_names for case in (
            "detect_threshold_morph_stream",
            "detect_contour_stream",
            "detect_ellipse_intensity_stream",
    )):
        bandpass_payloads = [
            _prepare_detect_bandpass_payload(payload)
            for payload in detect_payloads
        ]

    threshold_payloads = None
    if any(case in benchmark_case_names for case in (
            "detect_contour_stream",
            "detect_ellipse_intensity_stream",
    )):
        threshold_payloads = [
            _prepare_detect_threshold_payload(payload)
            for payload in bandpass_payloads
        ]

    contour_payloads = None
    if "detect_ellipse_intensity_stream" in benchmark_case_names:
        contour_payloads = [
            _prepare_detect_contour_payload(payload)
            for payload in threshold_payloads
        ]

    remap_payloads = None
    if "remap_stream" in benchmark_case_names:
        remap_payloads = _prepare_remap_payloads(frames)

    stream_geometry = None
    if any(case in benchmark_case_names for case in (
            "match_stream",
            "warp_stream",
            "optimization_stream",
    )):
        stream_geometry = precompute_geometries(frames)

    runners: dict[str, Any] = {
        "detect_stream": lambda: bench_detect_stream(frames),
        "detect_opencv_stream": lambda: bench_detect_opencv_stream(frames),
        "detect_cuda_hybrid_stream": lambda: bench_detect_cuda_hybrid_stream(frames),
        "detect_prepare_stream": lambda: bench_detect_prepare_stream(frames),
        "detect_wavelet_stream": lambda: bench_detect_wavelet_stream(
            detect_payloads),
        "detect_extract_stream": lambda: bench_detect_extract_stream(
            detect_payloads),
        "detect_bandpass_stream": lambda: bench_detect_bandpass_stream(
            detect_payloads),
        "detect_threshold_morph_stream": lambda: bench_detect_threshold_morph_stream(
            bandpass_payloads),
        "detect_contour_stream": lambda: bench_detect_contour_stream(
            threshold_payloads),
        "detect_ellipse_intensity_stream": lambda: bench_detect_ellipse_intensity_stream(
            contour_payloads),
        "features_stream": lambda: bench_features_stream(frames),
        "geometry_stream": lambda: bench_geometry_stream(frames),
        "match_stream": lambda: bench_match_stream_geometries(
            stream_geometry[0], stream_geometry[1]),
        "warp_stream": lambda: bench_warp_stream_geometries(
            frames, stream_geometry[0], stream_geometry[1]),
        "homography_pipeline": lambda: bench_homography_pipeline(frames),
        "optimization_stream": lambda: bench_optimization_stream_geometries(
            frames, stream_geometry[0], stream_geometry[1]),
        "remap_stream": lambda: bench_remap_stream(remap_payloads),
        "camera_model_pipeline": lambda: bench_camera_model_pipeline(frames),
    }

    cases = {
        case_name: run_benchmark(
            runners[case_name],
            warmup=args.warmup,
            repeat=args.repeat,
        )
        for case_name in benchmark_case_names
    }
    annotate_case_units(cases, alignment_case_units(
        benchmark_case_names,
        len(frames),
    ))

    report = {
        "suite": "alignment",
        "env": collect_env_info(),
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
            "cases": args.cases,
            "log_level": args.log_level,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "input_source": input_source,
        "results": cases,
    }
    if quality_requested:
        quality = compare_detect_quality(frames)
        report["quality"] = quality
        report["quality_summary"] = summarize_detect_quality(quality)
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run(args)
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
