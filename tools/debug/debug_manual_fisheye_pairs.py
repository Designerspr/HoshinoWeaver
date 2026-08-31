"""Fit fisheye alignment from manually selected point pairs only.

This script intentionally bypasses star detection and automatic matching. It is
for checking whether a trusted set of point correspondences can be explained by
the current fisheye projection + rotation + intrinsic optimization model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from hoshicore.component.image_io import load_img
from hoshicore.component.norma import (AlignmentResult, FisheyeCameraModel,
                                       build_camera)
from hoshicore.component.norma.alignment import (
    _build_flexible_result,
    _camera_optimization_state,
)
from hoshicore.component.norma.optimization import (
    CameraOptimizationPolicy,
    FlexibleOptimizationContext,
    compute_flexible_residual_diagnostics,
    estimate_robust_scale_from_initial_residual,
    flexible_reproject_error,
    iter_optimized_camera_param_slices,
    pack_flexible_initial_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit fisheye camera alignment from manual point pairs.")
    parser.add_argument(
        "--pairs",
        default="cur/manual_fisheye_pairs.csv",
        help="CSV with ref_x,ref_y,src_x,src_y columns.",
    )
    parser.add_argument(
        "--reference",
        default=r"I:\[picked]100JPGP3\_MG_3163.jpg",
        help="Reference image path, used only for image size and camera setup.",
    )
    parser.add_argument(
        "--source",
        default=r"I:\[picked]100JPGP3\_MG_3250.jpg",
        help="Source image path, used only for image size and camera setup.",
    )
    parser.add_argument(
        "--focal-length-mm",
        type=float,
        default=16.0,
        help="Initial fisheye focal length in mm.",
    )
    parser.add_argument(
        "--output-json",
        default="cur/manual_fisheye_fit_summary.json",
        help="Where to write fit summaries.",
    )
    parser.add_argument(
        "--output-residuals",
        default="cur/manual_fisheye_fit_residuals.csv",
        help="Where to write per-point residuals for all variants.",
    )
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument(
        "--method",
        choices=("trf", "lm"),
        default="trf",
        help="least_squares method. trf is safer for small manual point sets.",
    )
    parser.add_argument(
        "--same-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share intrinsics/distortion between reference and source cameras.",
    )
    return parser.parse_args()


def load_manual_pairs(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    ref_pts: list[tuple[float, float]] = []
    src_pts: list[tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"ref_x", "ref_y", "src_x", "src_y"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append(row)
            ref_pts.append((float(row["ref_x"]), float(row["ref_y"])))
            src_pts.append((float(row["src_x"]), float(row["src_y"])))
    if len(ref_pts) < 3:
        raise ValueError(f"Need at least 3 manual point pairs, got {len(ref_pts)}")
    return (
        np.asarray(ref_pts, dtype=np.float64),
        np.asarray(src_pts, dtype=np.float64),
        rows,
    )


def build_fisheye_camera(image_shape: tuple[int, ...], focal_length_mm: float) -> FisheyeCameraModel:
    camera = build_camera(
        exif_tags=None,
        img_shape=image_shape,
        method="distortion",
        init_distortion=None,
        lens_type="fisheye",
        focal_equiv_mm=focal_length_mm,
    )
    if not isinstance(camera, FisheyeCameraModel):
        raise TypeError(
            "Expected FisheyeCameraModel from build_camera, "
            f"got {type(camera).__name__ if camera is not None else 'None'}"
        )
    return camera


def estimate_rotation_svd(
    ref_camera: FisheyeCameraModel,
    src_camera: FisheyeCameraModel,
    ref_pts: np.ndarray,
    src_pts: np.ndarray,
) -> np.ndarray:
    ref_vecs = ref_camera.unproject(ref_pts)
    src_vecs = src_camera.unproject(src_pts)
    h_mat = ref_vecs.T @ src_vecs
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat[-1, :] *= -1
        rotation = vt_mat.T @ u_mat.T
    return rotation


def make_context(
    ref_camera: FisheyeCameraModel,
    src_camera: FisheyeCameraModel,
    ref_pts: np.ndarray,
    src_pts: np.ndarray,
    same_camera: bool,
    n_dist: int,
    optimize_shared_principal_point: bool,
    residual_space: str,
    robust_loss: str | None,
    principal_point_reg: float | None,
) -> tuple[np.ndarray, FlexibleOptimizationContext]:
    rotation_init = estimate_rotation_svd(ref_camera, src_camera, ref_pts, src_pts)
    rvec, _ = cv2.Rodrigues(rotation_init)
    rvec = rvec[:, 0]

    policy = CameraOptimizationPolicy(
        optimize_focal=True,
        optimize_distortion=True,
        optimize_principal_point=optimize_shared_principal_point,
        n_dist=n_dist,
    )
    ctx = FlexibleOptimizationContext(
        ref_pts=ref_pts,
        src_pts=src_pts,
        ref_state=_camera_optimization_state(ref_camera, policy),
        src_state=_camera_optimization_state(src_camera, policy),
        same_camera=same_camera,
        robust_loss=robust_loss,
        residual_space=residual_space,
    )
    x0 = pack_flexible_initial_params(rvec, ctx)
    ctx.params0 = x0.copy()

    if principal_point_reg is not None:
        reg_weight = np.zeros_like(x0)
        for _, state, param_slice in iter_optimized_camera_param_slices(ctx):
            if state.policy.optimize_principal_point:
                reg_weight[param_slice.stop - 2:param_slice.stop] = (
                    principal_point_reg)
        ctx.reg_weight = reg_weight
    return x0, ctx


def result_pixel_residuals(result, ref_pts: np.ndarray, src_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_vecs = result.ref_camera.unproject(ref_pts)
    src_vecs = (result.rotation_ref_to_src @ ref_vecs.T).T
    src_pred = result.src_camera.project(src_vecs)
    return src_pred - src_pts, src_pred


def result_angle_residuals(result, ref_pts: np.ndarray, src_pts: np.ndarray) -> np.ndarray:
    ref_vecs = result.ref_camera.unproject(ref_pts)
    src_vecs = result.src_camera.unproject(src_pts)
    rotated = (result.rotation_ref_to_src @ ref_vecs.T).T
    dots = np.sum(rotated * src_vecs, axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def describe(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def summarize_result(label: str, result, ref_camera: FisheyeCameraModel,
                     ref_pts: np.ndarray, src_pts: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    residual_vec, src_pred = result_pixel_residuals(result, ref_pts, src_pts)
    residual_px = np.linalg.norm(residual_vec, axis=1)
    angle_rad = result_angle_residuals(result, ref_pts, src_pts)
    focal = result.ref_camera.intrinsics.focal_length_mm
    rvec, _ = cv2.Rodrigues(result.rotation_ref_to_src)
    rvec = rvec[:, 0]

    summary = {
        "label": label,
        "count": int(len(ref_pts)),
        "pixel": describe(residual_px),
        "angle_rad": describe(angle_rad),
        "angle_deg": describe(np.rad2deg(angle_rad)),
        "mean_dx_px": float(np.mean(residual_vec[:, 0])),
        "mean_dy_px": float(np.mean(residual_vec[:, 1])),
        "median_dx_px": float(np.median(residual_vec[:, 0])),
        "median_dy_px": float(np.median(residual_vec[:, 1])),
        "focal_length_mm": float(focal),
        "focal_scale": float(focal / ref_camera.intrinsics.focal_length_mm - 1.0),
        "principal_point_px": [
            float(result.ref_camera.intrinsics.principal_point_px[0]),
            float(result.ref_camera.intrinsics.principal_point_px[1]),
        ],
        "distortion": [float(v) for v in result.ref_camera.dist_k4],
        "rotation_angle_deg": float(np.linalg.norm(rvec) * 180.0 / np.pi),
        "rotation_rvec": [float(v) for v in rvec],
    }
    rows = []
    for idx in range(len(ref_pts)):
        rows.append({
            "label": label,
            "idx": idx,
            "ref_x": float(ref_pts[idx, 0]),
            "ref_y": float(ref_pts[idx, 1]),
            "src_x": float(src_pts[idx, 0]),
            "src_y": float(src_pts[idx, 1]),
            "pred_x": float(src_pred[idx, 0]),
            "pred_y": float(src_pred[idx, 1]),
            "dx_px": float(residual_vec[idx, 0]),
            "dy_px": float(residual_vec[idx, 1]),
            "residual_px": float(residual_px[idx]),
            "angle_deg": float(np.rad2deg(angle_rad[idx])),
        })
    return summary, rows


def run_variant(
    label: str,
    ref_camera: FisheyeCameraModel,
    src_camera: FisheyeCameraModel,
    ref_pts: np.ndarray,
    src_pts: np.ndarray,
    same_camera: bool,
    n_dist: int,
    optimize_shared_principal_point: bool,
    residual_space: str,
    robust_loss: str | None,
    principal_point_reg: float | None,
    method: str,
    max_nfev: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    x0, ctx = make_context(
        ref_camera,
        src_camera,
        ref_pts,
        src_pts,
        same_camera=same_camera,
        n_dist=n_dist,
        optimize_shared_principal_point=optimize_shared_principal_point,
        residual_space=residual_space,
        robust_loss=robust_loss,
        principal_point_reg=principal_point_reg,
    )
    solve_kwargs = {
        "method": method,
        "max_nfev": max_nfev,
    }
    if robust_loss is not None and method != "lm":
        solve_kwargs.update({
            "loss": robust_loss,
            "f_scale": estimate_robust_scale_from_initial_residual(x0, ctx),
        })
    opt = least_squares(
        flexible_reproject_error, x0, args=(ctx,), **solve_kwargs)
    result = _build_flexible_result(opt.x, ctx, ref_camera, src_camera)
    summary, rows = summarize_result(label, result, ref_camera, ref_pts, src_pts)
    summary.update({
        "n_dist": int(n_dist),
        "same_camera": bool(same_camera),
        "optimize_shared_principal_point": bool(optimize_shared_principal_point),
        "residual_space": residual_space,
        "robust_loss": robust_loss,
        "principal_point_reg": principal_point_reg,
        "optimizer_method": method,
        "optimizer_success": bool(opt.success),
        "optimizer_status": int(opt.status),
        "optimizer_message": str(opt.message),
        "optimizer_cost": float(opt.cost),
        "optimizer_nfev": int(opt.nfev),
        "optimizer_optimality": float(opt.optimality),
        "objective_diagnostics": compute_flexible_residual_diagnostics(
            opt.x, ctx),
    })
    return summary, rows


def main() -> None:
    args = parse_args()
    pairs_path = Path(args.pairs)
    ref_pts, src_pts, manual_rows = load_manual_pairs(pairs_path)

    reference = load_img(args.reference)
    source = load_img(args.source)
    ref_camera = build_fisheye_camera(reference.shape, args.focal_length_mm)
    src_camera = build_fisheye_camera(source.shape, args.focal_length_mm)

    rotation_init = estimate_rotation_svd(ref_camera, src_camera, ref_pts, src_pts)
    baseline = AlignmentResult(rotation_init, ref_camera, src_camera)
    baseline_summary, residual_rows = summarize_result(
        "svd_rotation_fixed_intrinsics",
        baseline,
        ref_camera,
        ref_pts,
        src_pts,
    )

    variants: list[tuple[str, int, bool, str, str | None, float | None]] = [
        ("pixel_no_robust_k3_center_pp", 3, False, "pixel", None, None),
        ("pixel_no_robust_k2_center_pp", 2, False, "pixel", None, None),
        ("pixel_no_robust_k4_center_pp", 4, False, "pixel", None, None),
        ("pixel_no_robust_k4_free_pp", 4, True, "pixel", None, None),
        ("pixel_robust_k4_center_pp", 4, False, "pixel", "huber", None),
        ("angular_robust_k3_current_like", 3, False, "angular", "huber", None),
    ]

    summaries = [baseline_summary]
    all_residual_rows = residual_rows
    for label, n_dist, opt_pp, residual_space, robust_loss, pp_reg in variants:
        if opt_pp and not args.same_camera:
            continue
        try:
            summary, rows = run_variant(
                label,
                ref_camera,
                src_camera,
                ref_pts,
                src_pts,
                same_camera=args.same_camera,
                n_dist=n_dist,
                optimize_shared_principal_point=opt_pp,
                residual_space=residual_space,
                robust_loss=robust_loss,
                principal_point_reg=pp_reg,
                method=args.method,
                max_nfev=args.max_nfev,
            )
        except Exception as exc:
            summary = {
                "label": label,
                "count": int(len(ref_pts)),
                "failed": True,
                "error": repr(exc),
                "n_dist": int(n_dist),
                "optimize_shared_principal_point": bool(opt_pp),
                "residual_space": residual_space,
                "robust_loss": robust_loss,
                "principal_point_reg": pp_reg,
                "optimizer_method": args.method,
            }
            rows = []
        summaries.append(summary)
        all_residual_rows.extend(rows)

    output = {
        "manual_pairs_csv": str(pairs_path),
        "manual_pair_count": int(len(ref_pts)),
        "reference": str(args.reference),
        "source": str(args.source),
        "initial_focal_length_mm": float(args.focal_length_mm),
        "same_camera": bool(args.same_camera),
        "manual_rows": manual_rows,
        "fit_summaries": summaries,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    output_residuals = Path(args.output_residuals)
    output_residuals.parent.mkdir(parents=True, exist_ok=True)
    if all_residual_rows:
        with output_residuals.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_residual_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_residual_rows)

    print(f"Manual pairs: {len(ref_pts)}")
    for summary in summaries:
        if summary.get("failed"):
            print(f"{summary['label']}: FAILED {summary['error']}")
            continue
        pix = summary["pixel"]
        print(
            f"{summary['label']}: "
            f"median={pix['median']:.4f}px p90={pix['p90']:.4f}px "
            f"max={pix['max']:.4f}px mean={pix['mean']:.4f}px "
            f"focal={summary['focal_length_mm']:.6f}mm "
            f"pp=({summary['principal_point_px'][0]:.3f},"
            f"{summary['principal_point_px'][1]:.3f})"
        )
    print(f"Wrote {output_json}")
    print(f"Wrote {output_residuals}")


if __name__ == "__main__":
    main()
