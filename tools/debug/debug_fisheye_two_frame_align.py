"""Align and diagnose an arbitrary pair of star-field images.

Despite the historical filename, this entry supports perspective, fisheye,
ideal zero-distortion perspective, and mixed camera pairs through Norma's
current camera-model pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from loguru import logger

from hoshicore.component.image_io import load_img, save_img
from hoshicore.component.exif import read_exif_data
from hoshicore.component.norma import (
    AlignmentError,
    CameraInitializationPolicy,
    GeometryView,
    StarDetectionCache,
    build_camera_candidate,
    optimize_alignment,
    run_guided_refine_stage,
)
from hoshicore.component.norma.frame_align import (
    BOOTSTRAP_MATCHING_PATHS,
    DEFAULT_MATCHING_PATH,
    DEFAULT_BOOTSTRAP_SCALES,
    MATCHING_PATH_ASTERISM,
    MATCHING_PATHS,
    _compute_radial_zone_stats,
    _select_initial_alignment_candidate,
    solve_star_alignment,
)


LENS_TYPES = ("perspective", "fisheye", "ideal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align two star-field images with Norma's camera-model pipeline."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--lens-type",
        choices=LENS_TYPES,
        default="fisheye",
        help="Shared lens type; retained as fisheye by default for compatibility.",
    )
    parser.add_argument("--ref-lens-type", choices=LENS_TYPES)
    parser.add_argument("--src-lens-type", choices=LENS_TYPES)
    parser.add_argument(
        "--focal-length-mm",
        type=float,
        help="Legacy shared 35mm-equivalent focal initialization.",
    )
    parser.add_argument("--ref-focal-equiv-mm", type=float)
    parser.add_argument("--src-focal-equiv-mm", type=float)
    parser.add_argument(
        "--fallback-focal-equiv-mm", type=float, default=20.0)
    parser.add_argument(
        "--ref-exif-json",
        help="Optional JSON override; otherwise read reference EXIF from the image.",
    )
    parser.add_argument(
        "--src-exif-json",
        help="Optional JSON override; otherwise read source EXIF from the image.",
    )
    parser.add_argument(
        "--bootstrap-scales",
        type=float,
        nargs="+",
        default=DEFAULT_BOOTSTRAP_SCALES,
    )
    parser.add_argument(
        "--same-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--median-threshold-ratio",
        type=float,
        default=1.0,
        help="Median detector binary threshold multiplier; production default is 1.0.",
    )
    parser.add_argument(
        "--matching-path",
        type=str,
        choices=MATCHING_PATHS,
        default=DEFAULT_MATCHING_PATH,
        help=("Initial matching strategy. Bootstrap strategies use median "
              "detections only when guided refinement is enabled."),
    )
    parser.add_argument(
        "--guided-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("After the first solve, rematch all detected stars with a local "
              "projected-position mutual search and optimize once more."),
    )
    parser.add_argument(
        "--guided-refine-radius-px",
        type=float,
        default=8.0,
        help="Native-image search radius for the optional guided rematch.",
    )
    parser.add_argument("--debug-dir")
    parser.add_argument(
        "--remap-map-scale",
        type=float,
        default=0.5,
        help="Coordinate-map sampling edge scale; 1.0 uses an exact full-resolution map.",
    )
    parser.add_argument(
        "--validate-remap-scales",
        type=float,
        nargs="+",
        metavar="SCALE",
        help=("Evaluate source-coordinate maps on reduced grids (for example "
              "0.5 0.333333), interpolate the maps, then perform a full-resolution remap."),
    )
    return parser.parse_args()


def _read_exif_tags(image_path: Path,
                    json_path: str | None) -> dict[str, str] | None:
    """Read EXIF from the image, with JSON retained as an explicit override.

    The JSON option is useful for synthetic images and for reproducing a
    calibration.  Normal runs now use the repository's pyexiv2-backed EXIF
    reader directly, so no sidecar file is required.
    """
    if json_path is not None:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"EXIF JSON must contain an object: {json_path}")
        return {str(key): str(value) for key, value in payload.items()}
    try:
        exif_data = read_exif_data(str(image_path))
    except Exception as exc:
        logger.warning("Failed to read EXIF from {}: {}", image_path, exc)
        exif_data = None
    return exif_data.exif if exif_data is not None else None


def _policy_lens_type(lens_type: str) -> str | None:
    return None if lens_type == "perspective" else lens_type


def _to_bgr_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.dtype == np.uint8:
        return arr.copy()

    arr_f = arr.astype(np.float32)
    finite = np.isfinite(arr_f)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.min(arr_f[finite]))
    hi = float(np.percentile(arr_f[finite], 99.5))
    if hi <= lo:
        hi = max(float(np.max(arr_f[finite])), lo + 1.0)
    return np.round(np.clip((arr_f - lo) / (hi - lo), 0.0, 1.0) *
                    255.0).astype(np.uint8)


def _draw_matches(reference: np.ndarray, source: np.ndarray,
                  pts_ref: np.ndarray, pts_src: np.ndarray,
                  residuals: np.ndarray) -> np.ndarray:
    ref_vis = _to_bgr_u8(reference)
    src_vis = _to_bgr_u8(source)
    height = max(ref_vis.shape[0], src_vis.shape[0])
    offset = ref_vis.shape[1]
    canvas = np.zeros((height, offset + src_vis.shape[1], 3), dtype=np.uint8)
    canvas[:ref_vis.shape[0], :offset] = ref_vis
    canvas[:src_vis.shape[0], offset:] = src_vis

    order = np.argsort(residuals)
    if len(order) > 180:
        order = order[np.linspace(0, len(order) - 1, 180, dtype=int)]
    scale = max(float(np.percentile(residuals, 95)), 1.0)
    for idx in order:
        color_value = int(255 * min(float(residuals[idx]) / scale, 1.0))
        color = tuple(int(v) for v in cv2.applyColorMap(
            np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0])
        p_ref = tuple(np.round(pts_ref[idx]).astype(int))
        p_src_arr = np.round(pts_src[idx]).astype(int)
        p_src = (int(p_src_arr[0] + offset), int(p_src_arr[1]))
        cv2.circle(canvas, p_ref, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p_src, 3, color, -1, cv2.LINE_AA)
        cv2.line(canvas, p_ref, p_src, color, 1, cv2.LINE_AA)
    return canvas


def _draw_residuals(source: np.ndarray, pts_src: np.ndarray,
                    pts_pred: np.ndarray) -> np.ndarray:
    canvas = _to_bgr_u8(source)
    for actual, predicted in zip(pts_src, pts_pred):
        p_actual = tuple(np.round(actual).astype(int))
        p_pred = tuple(np.round(predicted).astype(int))
        cv2.circle(canvas, p_actual, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, p_actual, p_pred, (0, 0, 255), 1,
                        cv2.LINE_AA, tipLength=0.25)
    return canvas


def _draw_descriptor_profiles(ref_features: np.ndarray,
                              src_features: np.ndarray,
                              cosine_distances: np.ndarray,
                              residuals: np.ndarray,
                              max_rows: int = 12) -> np.ndarray:
    """Plot matched angular-histogram profiles for visual comparison."""
    width = 960
    row_height = 112
    if len(ref_features) == 0:
        return np.zeros((row_height, width, 3), dtype=np.uint8)

    order = np.argsort(residuals)
    if len(order) > max_rows:
        order = order[np.linspace(0, len(order) - 1, max_rows, dtype=int)]
    canvas = np.full((row_height * len(order), width, 3), 18,
                     dtype=np.uint8)
    plot_x0 = 230
    plot_x1 = width - 24
    for row, pair_index in enumerate(order):
        y0 = row * row_height
        top = y0 + 16
        bottom = y0 + row_height - 18
        cv2.putText(
            canvas,
            f"pair={int(pair_index)}  desc_cos={cosine_distances[pair_index]:.5f}  "
            f"residual={residuals[pair_index]:.3f}px",
            (12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "ref", (12, y0 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 230, 80), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, "src", (70, y0 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 80, 230), 1,
                    cv2.LINE_AA)
        cv2.line(canvas, (plot_x0, bottom), (plot_x1, bottom),
                 (80, 80, 80), 1)
        pair_features = np.vstack((ref_features[pair_index],
                                   src_features[pair_index]))
        peak = max(float(np.max(pair_features)), 1e-12)
        x = np.linspace(plot_x0, plot_x1, pair_features.shape[1])
        for feature, color in zip(pair_features,
                                  ((80, 230, 80), (230, 80, 230))):
            y = bottom - (feature / peak) * (bottom - top)
            points = np.column_stack((x, y)).round().astype(np.int32)
            cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
    return canvas


def _write_diagnostics(debug_dir: Path, reference: np.ndarray,
                       source: np.ndarray, pts_ref: np.ndarray,
                       pts_src: np.ndarray, pts_pred: np.ndarray,
                       residuals: np.ndarray, summary: dict,
                       ref_features: np.ndarray | None,
                       src_features: np.ndarray | None,
                       descriptor_distances: np.ndarray | None) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "residual_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (debug_dir / "matched_points.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("ref_x", "ref_y", "src_x", "src_y", "pred_src_x",
                         "pred_src_y", "residual_px",
                         "descriptor_cosine_distance"))
        for index, (ref, src, pred, error) in enumerate(
                zip(pts_ref, pts_src, pts_pred, residuals)):
            descriptor_distance = (
                "" if descriptor_distances is None else
                float(descriptor_distances[index]))
            writer.writerow((*ref, *src, *pred, float(error),
                             descriptor_distance))

    save_img(str(debug_dir / "matches_side_by_side.png"),
             _draw_matches(reference, source, pts_ref, pts_src, residuals))
    save_img(str(debug_dir / "residual_vectors_on_source.png"),
             _draw_residuals(source, pts_src, pts_pred))
    if (ref_features is not None and src_features is not None
            and descriptor_distances is not None):
        save_img(str(debug_dir / "descriptor_angular_profiles.png"),
                 _draw_descriptor_profiles(ref_features, src_features,
                                           descriptor_distances, residuals))


def _match_residual_stats(alignment, match) -> dict[str, float | int | None]:
    ref_rays = alignment.ref_camera.unproject(match.ref_pts)
    predicted = alignment.src_camera.project(
        (alignment.rotation_ref_to_src @ ref_rays.T).T)
    valid = np.all(np.isfinite(predicted), axis=1)
    residuals = np.linalg.norm(
        predicted[valid] - match.src_pts[valid], axis=1)
    if len(residuals) == 0:
        return {
            "count": 0,
            "median_px": None,
            "p90_px": None,
            "p99_px": None,
            "max_px": None,
        }
    return {
        "count": int(len(residuals)),
        "median_px": float(np.median(residuals)),
        "p90_px": float(np.percentile(residuals, 90)),
        "p99_px": float(np.percentile(residuals, 99)),
        "max_px": float(np.max(residuals)),
    }


def _rotation_delta_deg(first, second) -> float:
    relative = (np.asarray(second.rotation_ref_to_src, dtype=np.float64)
                @ np.asarray(first.rotation_ref_to_src, dtype=np.float64).T)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _stage1_output_path(final_output: Path) -> Path:
    return final_output.with_name(
        f"{final_output.stem}_stage1{final_output.suffix}")


def _normalized_pixel_error(reference: np.ndarray,
                            approximate: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference).astype(np.float32)
    approx = np.asarray(approximate).astype(np.float32)
    if np.issubdtype(reference.dtype, np.integer):
        scale = float(np.iinfo(reference.dtype).max)
    else:
        finite = ref[np.isfinite(ref)]
        scale = max(float(np.max(np.abs(finite))) if finite.size else 1.0,
                    1.0)
    error = np.abs(ref - approx) / scale
    return np.mean(error, axis=2) if error.ndim == 3 else error


def _interpolated_remap_maps(scale: float, width: int, height: int,
                             ref_camera, src_camera,
                             rotation) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Evaluate a sparse coordinate map, then interpolate the map itself."""
    grid_w = max(2, round(width * scale))
    grid_h = max(2, round(height * scale))
    sx = grid_w / width
    sy = grid_h / height

    # These are the full-resolution pixel coordinates represented by OpenCV's
    # half-pixel-centered resize samples.
    xs = (np.arange(grid_w, dtype=np.float64) + 0.5) / sx - 0.5
    ys = (np.arange(grid_h, dtype=np.float64) + 0.5) / sy - 0.5
    xx, yy = np.meshgrid(xs, ys)
    dst_pixels = np.column_stack((xx.ravel(), yy.ravel()))
    dst_rays = ref_camera.unproject(dst_pixels)
    src_rays = (rotation @ dst_rays.T).T
    src_pixels = src_camera.project(src_rays)
    sparse_x = src_pixels[:, 0].reshape(grid_h, grid_w).astype(np.float32)
    sparse_y = src_pixels[:, 1].reshape(grid_h, grid_w).astype(np.float32)
    map_x = cv2.resize(sparse_x, (width, height), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(sparse_y, (width, height), interpolation=cv2.INTER_LINEAR)

    # cv2.resize clamps beyond the first/last sparse sample. Replace the four
    # full-resolution borders exactly so that this boundary convention does
    # not dominate coordinate p99/max.
    top_bottom_x = np.arange(width, dtype=np.float64)
    left_right_y = np.arange(1, height - 1, dtype=np.float64)
    border_pixels = np.concatenate((
        np.column_stack((top_bottom_x, np.zeros(width))),
        np.column_stack((top_bottom_x, np.full(width, height - 1.0))),
        np.column_stack((np.zeros(len(left_right_y)), left_right_y)),
        np.column_stack((np.full(len(left_right_y), width - 1.0), left_right_y)),
    ))
    border_rays = ref_camera.unproject(border_pixels)
    border_src = src_camera.project((rotation @ border_rays.T).T)
    bx = border_pixels[:, 0].astype(np.intp)
    by = border_pixels[:, 1].astype(np.intp)
    map_x[by, bx] = border_src[:, 0].astype(np.float32)
    map_y[by, bx] = border_src[:, 1].astype(np.float32)
    return map_x, map_y, (grid_w, grid_h)


def _coordinate_error_sample(map_x: np.ndarray, map_y: np.ndarray,
                             ref_camera, src_camera,
                             rotation) -> tuple[np.ndarray, np.ndarray]:
    """Compare an interpolated map with exact projection on <=250k pixels."""
    height, width = map_x.shape
    stride = max(1, int(np.ceil(np.sqrt((width * height) / 250_000))))
    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    pixels = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    rays = ref_camera.unproject(pixels)
    exact = src_camera.project((rotation @ rays.T).T)
    approximate = np.column_stack((map_x[yy, xx].ravel(),
                                   map_y[yy, xx].ravel()))
    valid = (np.all(np.isfinite(exact), axis=1)
             & np.all(np.isfinite(approximate), axis=1))
    error = np.linalg.norm(exact[valid] - approximate[valid], axis=1)
    radius = np.sqrt(((pixels[valid, 0] - width / 2.0) / (width / 2.0))**2
                     + ((pixels[valid, 1] - height / 2.0) /
                        (height / 2.0))**2)
    return error, radius >= 0.7


def _validate_reduced_remap(scales, full: np.ndarray, source: np.ndarray,
                            ref_camera, src_camera, rotation,
                            debug_dir: Path | None) -> list[dict]:
    full_h, full_w = full.shape[:2]
    yy, xx = np.ogrid[:full_h, :full_w]
    radius = np.sqrt(((xx - full_w / 2.0) / (full_w / 2.0))**2 +
                     ((yy - full_h / 2.0) / (full_h / 2.0))**2)
    outer_mask = radius >= 0.7
    results = []
    for scale in scales:
        if not 0.0 < scale < 1.0:
            raise ValueError("--validate-remap-scales values must be in (0, 1)")
        map_x, map_y, (grid_w, grid_h) = _interpolated_remap_maps(
            scale, full_w, full_h, ref_camera, src_camera, rotation)
        approximate = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0 if source.ndim == 2 else (0, 0, 0),
        )
        error = _normalized_pixel_error(full, approximate)
        rmse = float(np.sqrt(np.mean(np.square(error))))
        coordinate_error, coordinate_outer = _coordinate_error_sample(
            map_x, map_y, ref_camera, src_camera, rotation)
        stats = {
            "edge_scale": float(scale),
            "map_sample_fraction": float((grid_w * grid_h) /
                                         (full_w * full_h)),
            "map_grid_size": [grid_w, grid_h],
            "coordinate_samples": int(len(coordinate_error)),
            "coordinate_median_px": float(np.median(coordinate_error)),
            "coordinate_p90_px": float(np.percentile(coordinate_error, 90)),
            "coordinate_p99_px": float(np.percentile(coordinate_error, 99)),
            "coordinate_max_px": float(np.max(coordinate_error)),
            "coordinate_outer_p99_px": float(
                np.percentile(coordinate_error[coordinate_outer], 99)),
            "mae": float(np.mean(error)),
            "rmse": rmse,
            "p90": float(np.percentile(error, 90)),
            "p99": float(np.percentile(error, 99)),
            "max": float(np.max(error)),
            "psnr_db": (float("inf") if rmse == 0.0 else
                        float(-20.0 * np.log10(rmse))),
            "outer_mae": float(np.mean(error[outer_mask])),
            "outer_p99": float(np.percentile(error[outer_mask], 99)),
        }
        results.append(stats)
        logger.info(
            "Sparse-map remap: scale={:.6f} samples={:.3f} grid={}x{} "
            "coord_med={:.6f}px coord_p90={:.6f}px coord_p99={:.6f}px "
            "coord_max={:.6f}px outer_coord_p99={:.6f}px "
            "pixel_MAE={:.6f} pixel_p99={:.6f} PSNR={:.2f}dB",
            scale, stats["map_sample_fraction"], grid_w, grid_h,
            stats["coordinate_median_px"], stats["coordinate_p90_px"],
            stats["coordinate_p99_px"], stats["coordinate_max_px"],
            stats["coordinate_outer_p99_px"], stats["mae"], stats["p99"],
            stats["psnr_db"])
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            label = f"{scale:.6f}".rstrip("0").rstrip(".").replace(".", "p")
            save_img(str(debug_dir / f"remap_map_approx_{label}.tif"), approximate)
            diff_vis = np.round(np.clip(error * 8.0, 0.0, 1.0) *
                                255.0).astype(np.uint8)
            save_img(str(debug_dir / f"remap_error_x8_{label}.png"), diff_vis)
    return results


def main() -> int:
    args = parse_args()
    reference_path = Path(args.reference)
    source_path = Path(args.source)
    output_path = Path(args.output)

    reference = load_img(str(reference_path))
    source = load_img(str(source_path))
    if reference is None:
        raise FileNotFoundError(f"Failed to load reference: {reference_path}")
    if source is None:
        raise FileNotFoundError(f"Failed to load source: {source_path}")

    ref_lens = args.ref_lens_type or args.lens_type
    src_lens = args.src_lens_type or args.lens_type
    ref_focal = (args.ref_focal_equiv_mm if args.ref_focal_equiv_mm is not None
                 else args.focal_length_mm)
    src_focal = (args.src_focal_equiv_mm if args.src_focal_equiv_mm is not None
                 else args.focal_length_mm)
    ref_policy = CameraInitializationPolicy(
        lens_type=_policy_lens_type(ref_lens),
        fallback_focal_equiv_mm=args.fallback_focal_equiv_mm,
    )
    src_policy = CameraInitializationPolicy(
        lens_type=_policy_lens_type(src_lens),
        fallback_focal_equiv_mm=args.fallback_focal_equiv_mm,
    )
    
    #print(_read_exif_tags(source_path, args.src_exif_json))
    
    ref_candidate = build_camera_candidate(
        _read_exif_tags(reference_path, args.ref_exif_json), reference.shape,
        "distortion",
        focal_equiv_mm=ref_focal, init_policy=ref_policy)
    src_candidate = build_camera_candidate(
        _read_exif_tags(source_path, args.src_exif_json), source.shape,
        "distortion",
        focal_equiv_mm=src_focal, init_policy=src_policy)
    if ref_candidate is None or src_candidate is None:
        raise AlignmentError("Failed to initialize one or both cameras")

    logger.info(
        "Aligning {}({}) -> {}({}); same_camera={} bootstrap_scales={} "
        "matching_path={}",
        reference_path.name, ref_lens, source_path.name, src_lens,
        args.same_camera, tuple(args.bootstrap_scales), args.matching_path)

    pipeline_timings: dict[str, float] = {}
    bootstrap_star_counts: dict | None = None
    dense_star_counts: dict | None = None
    final_match_has_descriptors = True
    if args.matching_path in BOOTSTRAP_MATCHING_PATHS:
        ref_detection = StarDetectionCache.from_image(
            reference,
            median_threshold_ratio=args.median_threshold_ratio,
        )
        src_detection = StarDetectionCache.from_image(
            source,
            median_threshold_ratio=args.median_threshold_ratio,
        )
        ref_bootstrap_stars = ref_detection.pywt_stars
        src_bootstrap_stars = src_detection.pywt_stars
        try:
            started = perf_counter()
            first_stage_result, first_stage_match = solve_star_alignment(
                ref_bootstrap_stars,
                src_bootstrap_stars,
                ref_candidate,
                src_candidate,
                bootstrap_scales=tuple(args.bootstrap_scales),
                same_camera=args.same_camera,
                use_asterism_bootstrap=(
                    args.matching_path
                    == MATCHING_PATH_ASTERISM),
            )
            pipeline_timings["bootstrap_and_optimization"] = (
                perf_counter() - started)
        except Exception as exc:
            logger.exception("Alignment failed: {}", exc)
            return 1

        bootstrap_ref_geo = GeometryView(
            ref_bootstrap_stars, first_stage_result.ref_camera)
        bootstrap_src_geo = GeometryView(
            src_bootstrap_stars, first_stage_result.src_camera)
        ref_geo = bootstrap_ref_geo
        src_geo = bootstrap_src_geo
        result = first_stage_result
        match = first_stage_match
        bootstrap_star_counts = {
            "detector": "pywt",
            "reference": int(len(bootstrap_ref_geo.positions)),
            "source": int(len(bootstrap_src_geo.positions)),
        }

        guided_refinement = {
            "enabled": bool(args.guided_refine),
            "status": "disabled",
            "radius_px": float(args.guided_refine_radius_px),
            "first_stage_pairs": int(len(first_stage_match.pair_idx)),
            "first_stage_residuals": _match_residual_stats(
                first_stage_result, first_stage_match),
        }
        if args.guided_refine:
            ref_geo = GeometryView(
                ref_detection.median_stars, first_stage_result.ref_camera)
            src_geo = GeometryView(
                src_detection.median_stars, first_stage_result.src_camera)
            dense_star_counts = {
                "detector": "median",
                "reference": int(len(ref_geo.positions)),
                "source": int(len(src_geo.positions)),
            }
            try:
                started = perf_counter()
                result, match, refine_status = run_guided_refine_stage(
                    ref_geo,
                    src_geo,
                    first_stage_result,
                    bootstrap_match=first_stage_match,
                    same_camera=args.same_camera,
                    max_distance_px=args.guided_refine_radius_px,
                    ref_policy=ref_candidate.optimization_policy,
                    src_policy=src_candidate.optimization_policy,
                )
                pipeline_timings["guided_refine"] = perf_counter() - started
                guided_refinement["status"] = refine_status
                if refine_status == "applied":
                    final_match_has_descriptors = False
                else:
                    ref_geo = bootstrap_ref_geo
                    src_geo = bootstrap_src_geo
            except Exception as exc:
                ref_geo = bootstrap_ref_geo
                src_geo = bootstrap_src_geo
                guided_refinement.update({
                    "status": "failed_fallback",
                    "error": str(exc),
                })

            if guided_refinement["status"] == "applied":
                guided_before = _match_residual_stats(first_stage_result, match)
                guided_after = _match_residual_stats(result, match)
                guided_refinement.update({
                    "guided_pairs": int(len(match.pair_idx)),
                    "guided_residuals_before": guided_before,
                    "guided_residuals_after": guided_after,
                    "first_stage_pairs_after": _match_residual_stats(
                        result, first_stage_match),
                    "rotation_delta_deg": _rotation_delta_deg(
                        first_stage_result, result),
                })
    else:
        ref_detection = StarDetectionCache.from_image(
            reference,
            median_threshold_ratio=args.median_threshold_ratio,
        )
        src_detection = StarDetectionCache.from_image(
            source,
            median_threshold_ratio=args.median_threshold_ratio,
        )
        ref_geo = GeometryView(ref_detection.median_stars,
                               ref_candidate.camera)
        src_geo = GeometryView(src_detection.median_stars,
                               src_candidate.camera)
        try:
            ref_geo, src_geo, ref_candidate, src_candidate, match = (
                _select_initial_alignment_candidate(
                    ref_geo, src_geo, ref_candidate, src_candidate,
                    tuple(args.bootstrap_scales),
                    same_camera=args.same_camera))
            result = optimize_alignment(
                match,
                ref_candidate.camera,
                src_candidate.camera,
                same_camera=args.same_camera,
                ref_policy=ref_candidate.optimization_policy,
                src_policy=src_candidate.optimization_policy,
            )
        except Exception as exc:
            logger.exception("Alignment failed: {}", exc)
            return 1

        bootstrap_ref_geo = ref_geo
        bootstrap_src_geo = src_geo
        first_stage_result = result
        first_stage_match = match
        bootstrap_star_counts = {
            "detector": "median",
            "reference": int(len(ref_geo.positions)),
            "source": int(len(src_geo.positions)),
        }
        dense_star_counts = dict(bootstrap_star_counts)
        guided_refinement = {
            "enabled": bool(args.guided_refine),
            "status": "disabled",
            "radius_px": float(args.guided_refine_radius_px),
            "first_stage_pairs": int(len(first_stage_match.pair_idx)),
            "first_stage_residuals": _match_residual_stats(
                first_stage_result, first_stage_match),
        }
        if args.guided_refine:
            try:
                refined_result, guided_match, stage_status = (
                    run_guided_refine_stage(
                        ref_geo,
                        src_geo,
                        first_stage_result,
                        bootstrap_match=first_stage_match,
                        same_camera=args.same_camera,
                        max_distance_px=args.guided_refine_radius_px,
                        ref_policy=ref_candidate.optimization_policy,
                        src_policy=src_candidate.optimization_policy,
                    ))
                guided_before = _match_residual_stats(first_stage_result,
                                                      guided_match)
                guided_after = _match_residual_stats(refined_result,
                                                     guided_match)
                guided_refinement.update({
                    "status": stage_status,
                    "guided_pairs": int(len(guided_match.pair_idx)),
                    "guided_residuals_before": guided_before,
                    "guided_residuals_after": guided_after,
                    "first_stage_pairs_after": _match_residual_stats(
                        refined_result, first_stage_match),
                    "rotation_delta_deg": _rotation_delta_deg(
                        first_stage_result, refined_result),
                    "ref_focal_delta_mm": float(
                        refined_result.ref_camera.intrinsics.focal_length_mm
                        - first_stage_result.ref_camera.intrinsics.focal_length_mm),
                    "src_focal_delta_mm": float(
                        refined_result.src_camera.intrinsics.focal_length_mm
                        - first_stage_result.src_camera.intrinsics.focal_length_mm),
                })
                result = refined_result
                match = guided_match
            except Exception as exc:
                guided_refinement.update({
                    "status": "failed_fallback",
                    "error": str(exc),
                })
                logger.warning(
                    "Guided refinement failed; keeping first-stage result: {}",
                    exc,
                )

    ref_vecs = result.ref_camera.unproject(match.ref_pts)
    src_vecs = (result.rotation_ref_to_src @ ref_vecs.T).T
    pts_pred = result.src_camera.project(src_vecs)
    valid = np.all(np.isfinite(pts_pred), axis=1)
    valid_pair_idx = match.pair_idx[valid]
    pts_ref = match.ref_pts[valid]
    pts_src = match.src_pts[valid]
    pts_pred = pts_pred[valid]
    residuals = np.linalg.norm(pts_pred - pts_src, axis=1)
    if len(residuals) == 0:
        logger.error("No finite projected matches after optimization")
        return 1

    matched_ref_features = None
    matched_src_features = None
    descriptor_distances = None
    descriptor_summary: dict = {
        "status": "not_computed_position_guided"
    }
    if final_match_has_descriptors:
        matched_ref_features = ref_geo.features[valid_pair_idx[:, 0]]
        matched_src_features = src_geo.features[valid_pair_idx[:, 1]]
        descriptor_distances = 1.0 - np.sum(
            matched_ref_features * matched_src_features, axis=1)
        descriptor_distances = np.clip(descriptor_distances, 0.0, 2.0)
        descriptor_summary = {
            "status": "computed",
            "dimension": int(matched_ref_features.shape[1]),
            "cosine_distance_median": float(np.median(descriptor_distances)),
            "cosine_distance_p90": float(np.percentile(
                descriptor_distances, 90)),
            "cosine_distance_max": float(np.max(descriptor_distances)),
        }

    bootstrap_pair_idx = first_stage_match.pair_idx
    bootstrap_ref_features = None
    bootstrap_src_features = None
    bootstrap_descriptor_distances = None
    bootstrap_rays = first_stage_result.ref_camera.unproject(
        first_stage_match.ref_pts)
    bootstrap_pred = first_stage_result.src_camera.project(
        (first_stage_result.rotation_ref_to_src @ bootstrap_rays.T).T)
    bootstrap_valid = np.all(np.isfinite(bootstrap_pred), axis=1)
    bootstrap_profile_residuals = np.linalg.norm(
        bootstrap_pred[bootstrap_valid]
        - first_stage_match.src_pts[bootstrap_valid], axis=1)
    if args.matching_path == MATCHING_PATH_ASTERISM:
        bootstrap_descriptor_summary = {
            "status": "not_computed_asterism_voting",
            "detector": bootstrap_star_counts["detector"],
            "voted_pairs": first_stage_match.initial_pair_count,
            "ransac_pairs": int(len(first_stage_match.pair_idx)),
        }
    else:
        bootstrap_ref_features = bootstrap_ref_geo.features[
            bootstrap_pair_idx[:, 0]]
        bootstrap_src_features = bootstrap_src_geo.features[
            bootstrap_pair_idx[:, 1]]
        bootstrap_descriptor_distances = np.clip(
            1.0 - np.sum(
                bootstrap_ref_features * bootstrap_src_features, axis=1),
            0.0,
            2.0,
        )
        bootstrap_descriptor_summary = {
            "status": "computed",
            "detector": bootstrap_star_counts["detector"],
            "dimension": int(bootstrap_ref_features.shape[1]),
            "cosine_distance_median": float(np.median(
                bootstrap_descriptor_distances)),
            "cosine_distance_p90": float(np.percentile(
                bootstrap_descriptor_distances, 90)),
            "cosine_distance_max": float(np.max(
                bootstrap_descriptor_distances)),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_map_scale = (1.0 if args.validate_remap_scales else
                        args.remap_map_scale)
    if guided_refinement["status"] == "applied":
        first_stage_output = _stage1_output_path(output_path)
        first_stage_aligned = first_stage_result.ref_camera.project_image_from_camera(
            first_stage_result.src_camera,
            source,
            (reference.shape[1], reference.shape[0]),
            rotation_dst_to_src=first_stage_result.rotation_ref_to_src,
            map_scale=output_map_scale,
        )
        save_img(str(first_stage_output), first_stage_aligned)
        del first_stage_aligned
        guided_refinement["first_stage_output"] = str(first_stage_output)
        logger.info("First-stage aligned image saved to {}", first_stage_output)

    aligned = result.ref_camera.project_image_from_camera(
        result.src_camera,
        source,
        (reference.shape[1], reference.shape[0]),
        rotation_dst_to_src=result.rotation_ref_to_src,
        map_scale=output_map_scale,
    )
    save_img(str(output_path), aligned)
    guided_refinement["final_output"] = str(output_path)

    remap_validation = []
    if args.validate_remap_scales:
        remap_validation = _validate_reduced_remap(
            args.validate_remap_scales,
            aligned,
            source,
            result.ref_camera,
            result.src_camera,
            result.rotation_ref_to_src,
            Path(args.debug_dir) if args.debug_dir else None,
        )

    summary = {
        "matching_path": args.matching_path,
        "count": int(len(residuals)),
        "median_px": float(np.median(residuals)),
        "p90_px": float(np.percentile(residuals, 90)),
        "p99_px": float(np.percentile(residuals, 99)),
        "max_px": float(np.max(residuals)),
        "ref_lens_type": ref_lens,
        "src_lens_type": src_lens,
        "ref_focal_mm": result.ref_camera.intrinsics.focal_length_mm,
        "src_focal_mm": result.src_camera.intrinsics.focal_length_mm,
        "radial_zones": _compute_radial_zone_stats(
            pts_ref, residuals, reference.shape),
        "bootstrap_star_counts": bootstrap_star_counts,
        "dense_star_counts": dense_star_counts,
        "pipeline_timing_seconds": pipeline_timings,
        "descriptor_angular_histogram": descriptor_summary,
        "bootstrap_descriptor_angular_histogram":
            bootstrap_descriptor_summary,
        "guided_refinement": guided_refinement,
        "reduced_remap_validation": remap_validation,
    }
    logger.info(
        "Residuals: count={} median={:.3f}px p90={:.3f}px "
        "p99={:.3f}px max={:.3f}px",
        len(residuals), np.median(residuals),
        np.percentile(residuals, 90), np.percentile(residuals, 99),
        np.max(residuals))
    logger.info("Aligned image saved to {}", output_path)
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        _write_diagnostics(debug_dir, reference, source, pts_ref,
                           pts_src, pts_pred, residuals, summary,
                           matched_ref_features, matched_src_features,
                           descriptor_distances)
        if (bootstrap_ref_features is not None
                and bootstrap_src_features is not None
                and bootstrap_descriptor_distances is not None):
            save_img(
                str(debug_dir / "bootstrap_descriptor_angular_profiles.png"),
                _draw_descriptor_profiles(
                    bootstrap_ref_features[bootstrap_valid],
                    bootstrap_src_features[bootstrap_valid],
                    bootstrap_descriptor_distances[bootstrap_valid],
                    bootstrap_profile_residuals,
                ),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
