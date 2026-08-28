"""High-level alignment pipeline.

Provides two paths:
1. Star-point matching alignment (heavy): detect → match → optimize
2. Known-pointing projection (light): direct remap given known transforms
"""
import dataclasses
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .geometry_view import GeometryView
from .matching import (DEFAULT_COVERAGE_REPAIR, FISHEYE_ROTATION_VALIDATION,
                       AsterismMatchingConfig, MatchResult,
                       CoverageRepairContext,
                       RotationValidationConfig, find_guided_mutual_match,
                       find_asterism_initial_match, find_initial_match,
                       fine_tune_rotation)
from .optimization import (CameraOptimizationPolicy, CameraOptimizationState,
                           CameraSolvedParams, FlexibleOptimizationContext,
                           compute_flexible_residual_diagnostics,
                           make_flexible_regularization_weights,
                           pack_flexible_initial_params,
                           run_flexible_optimization, unpack_flexible_params)
from .types import BaseCameraModel, CameraModel, Distortion, FisheyeCameraModel, FisheyeDistortion


@dataclasses.dataclass(frozen=True)
class AlignmentResult:
    """Optimized ref-to-src transform and the corresponding camera models."""

    rotation_ref_to_src: NDArray[np.float64] | cv2.typing.MatLike
    ref_camera: BaseCameraModel
    src_camera: BaseCameraModel
    pair_idx: Optional[NDArray[np.int32]] = None
    def compose(self, other: "AlignmentResult") -> "AlignmentResult":
        """Chain: self is A→B, other is B→C, returns A→C."""
        return AlignmentResult(
            rotation_ref_to_src=(other.rotation_ref_to_src
                                 @ self.rotation_ref_to_src),
            ref_camera=self.ref_camera,
            src_camera=other.src_camera,
        )


def _format_array(arr: NDArray[np.float64], precision: int = 6) -> str:
    return np.array2string(
        np.asarray(arr, dtype=np.float64),
        precision=precision,
        suppress_small=False,
        separator=", ",
        max_line_width=200,
    )


def match_star_pairs(
    ref_geo: GeometryView,
    src_geo: GeometryView,
    apply_threshold_filter: bool = False,
    theta_th: float = np.pi / 6,
    rotation_validation_config:
    RotationValidationConfig = FISHEYE_ROTATION_VALIDATION,
    coverage_repair_context: CoverageRepairContext = DEFAULT_COVERAGE_REPAIR,
    random_seed: int | None = None,
) -> MatchResult:
    """Match stars between two GeometryView instances.

    GeometryView owns the matched-stage data: pixel positions, unit rays,
    detection volumes, and cached local-geometry features. Keeping this as the
    only public matching entry avoids duplicate array-based paths.

    Matching is validated in unit-ray space.
    """
    pair_idx = find_initial_match(
        ref_geo.features,
        src_geo.features,
        ref_geo.positions,
        src_geo.positions,
        vectors1=ref_geo.unit_vectors,
        vectors2=src_geo.unit_vectors,
        apply_threshold_filter=apply_threshold_filter,
        coverage_repair_context=coverage_repair_context,
        theta_th=theta_th)

    logger.debug(
        "match_star_pairs: using rotation RANSAC refine initial_pairs={}",
        len(pair_idx),
    )
    initial_pair_count = len(pair_idx)
    rotation, pair_idx = fine_tune_rotation(
        ref_geo.positions, src_geo.positions, ref_geo.unit_vectors,
        src_geo.unit_vectors, pair_idx, rotation_validation_config,
        random_seed=random_seed)
    homography = rotation_derived_homography(ref_geo.camera,
                                                  src_geo.camera,
                                                  rotation)

    return MatchResult(
        pair_idx=pair_idx,
        ref_pts=ref_geo.positions[pair_idx[:, 0]],
        src_pts=src_geo.positions[pair_idx[:, 1]],
        rotation=rotation,
        homography=homography,
        initial_pair_count=initial_pair_count,
    )


def match_star_pairs_asterism(
    ref_geo: GeometryView,
    src_geo: GeometryView,
    rotation_validation_config:
    RotationValidationConfig = FISHEYE_ROTATION_VALIDATION,
    asterism_config: AsterismMatchingConfig = AsterismMatchingConfig(),
    random_seed: int | None = None,
) -> MatchResult:
    """Match sparse stars using local spherical-triangle voting."""
    pair_idx = find_asterism_initial_match(
        ref_geo.unit_vectors,
        src_geo.unit_vectors,
        config=asterism_config,
    )
    if len(pair_idx) < 6:
        raise ValueError(
            "asterism bootstrap requires at least 6 voted pairs, "
            f"got {len(pair_idx)}")
    logger.debug(
        "match_star_pairs_asterism: using rotation RANSAC "
        "voted_pairs={}", len(pair_idx))
    initial_pair_count = len(pair_idx)
    rotation, pair_idx = fine_tune_rotation(
        ref_geo.positions,
        src_geo.positions,
        ref_geo.unit_vectors,
        src_geo.unit_vectors,
        pair_idx,
        rotation_validation_config,
        random_seed=random_seed,
    )
    return MatchResult(
        pair_idx=pair_idx,
        ref_pts=ref_geo.positions[pair_idx[:, 0]],
        src_pts=src_geo.positions[pair_idx[:, 1]],
        rotation=rotation,
        homography=rotation_derived_homography(
            ref_geo.camera, src_geo.camera, rotation),
        initial_pair_count=initial_pair_count,
    )


def rotation_derived_homography(
    ref_camera: BaseCameraModel,
    src_camera: BaseCameraModel,
    rotation_ref_to_src: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Return H = K_src @ R_ref_to_src @ inv(K_ref) when valid as a fast path.

    This is exact for zero-distortion perspective cameras. For non-zero
    perspective distortion it is an approximation; callers may still use it
    deliberately for speed, but remap is the geometrically correct backend.
    Fisheye and mixed projection pairs return None.
    """
    if not isinstance(ref_camera, CameraModel) or not isinstance(
            src_camera, CameraModel):
        return None

    if (not ref_camera.distortion.is_zero or
            not src_camera.distortion.is_zero):
        logger.warning(
            "rotation_derived_homography: using perspective H approximation "
            "with non-zero distortion; remap is geometrically correct")

    H = src_camera.K @ rotation_ref_to_src @ np.linalg.inv(ref_camera.K)
    if not np.all(np.isfinite(H)):
        return None
    if abs(float(H[2, 2])) > 1e-12:
        H = H / H[2, 2]
    return H.astype(np.float64, copy=False)


def _points_in_geometry_bounds(
    pts: NDArray[np.float64],
    geo: GeometryView,
) -> NDArray[np.bool_]:
    height, width = geo.img_shape[:2]
    return (
        np.all(np.isfinite(pts), axis=1)
        & (pts[:, 0] >= 0.0)
        & (pts[:, 0] <= width - 1.0)
        & (pts[:, 1] >= 0.0)
        & (pts[:, 1] <= height - 1.0)
    )


def guided_mutual_rematch(
    ref_geo: GeometryView,
    src_geo: GeometryView,
    alignment: AlignmentResult,
    max_distance_px: float = 8.0,
) -> MatchResult:
    """Rematch all detected stars under an already optimized projection.

    The global descriptors are intentionally not used here.  The current
    camera models and rotation predict each complete point set into the other
    image, after which a radius-limited mutual nearest-neighbor match is made
    in native pixel coordinates.
    """
    rotation = np.asarray(alignment.rotation_ref_to_src, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("guided rematch requires a finite 3x3 rotation")

    ref_rays = alignment.ref_camera.unproject(ref_geo.positions)
    predicted_src = alignment.src_camera.project((rotation @ ref_rays.T).T)
    src_rays = alignment.src_camera.unproject(src_geo.positions)
    predicted_ref = alignment.ref_camera.project(
        (rotation.T @ src_rays.T).T)

    valid_ref_to_src = _points_in_geometry_bounds(predicted_src, src_geo)
    valid_src_to_ref = _points_in_geometry_bounds(predicted_ref, ref_geo)
    predicted_src = np.asarray(predicted_src, dtype=np.float64).copy()
    predicted_ref = np.asarray(predicted_ref, dtype=np.float64).copy()
    predicted_src[~valid_ref_to_src] = np.nan
    predicted_ref[~valid_src_to_ref] = np.nan

    pair_idx = find_guided_mutual_match(
        ref_geo.positions,
        src_geo.positions,
        predicted_src,
        predicted_ref,
        max_distance_px=max_distance_px,
    )
    logger.debug(
        "Guided mutual rematch: radius_px={:.3f} ref_points={} src_points={} "
        "ref_predictions_in_bounds={} src_predictions_in_bounds={} pairs={}",
        max_distance_px,
        len(ref_geo.positions),
        len(src_geo.positions),
        int(np.count_nonzero(valid_ref_to_src)),
        int(np.count_nonzero(valid_src_to_ref)),
        len(pair_idx),
    )
    return MatchResult(
        pair_idx=pair_idx,
        ref_pts=ref_geo.positions[pair_idx[:, 0]],
        src_pts=src_geo.positions[pair_idx[:, 1]],
        rotation=rotation,
        homography=rotation_derived_homography(
            alignment.ref_camera,
            alignment.src_camera,
            rotation,
        ),
    )


def filter_guided_match_spatially(
    match: MatchResult,
    alignment: AlignmentResult,
    image_shape: tuple[int, ...],
    grid_cols: int = 24,
    grid_rows: int = 16,
    min_support: int = 8,
    sigma: float = 3.5,
    floor_px: float = 1.5,
) -> tuple[MatchResult, dict[str, float | int]]:
    """Reject only locally inconsistent guided residual vectors.

    The first-stage projection defines a residual vector for every guided pair.
    Each image cell votes for its component-wise median vector; sparse cells
    borrow the surrounding 3x3 cells.  A robust local MAD threshold removes
    points that strongly disagree with that local vote.  Non-zero, spatially
    varying residual fields are deliberately retained so they can still drive
    focal/distortion refinement.
    """
    count = len(match.pair_idx)
    if count == 0:
        return match, {"input_pairs": 0, "kept_pairs": 0, "rejected_pairs": 0}
    if grid_cols <= 0 or grid_rows <= 0:
        raise ValueError("guided spatial-filter grid dimensions must be positive")

    rotation = np.asarray(alignment.rotation_ref_to_src, dtype=np.float64)
    ref_rays = alignment.ref_camera.unproject(match.ref_pts)
    predicted_src = alignment.src_camera.project((rotation @ ref_rays.T).T)
    residual = np.asarray(match.src_pts - predicted_src, dtype=np.float64)
    finite = np.all(np.isfinite(residual), axis=1)

    height, width = image_shape[:2]
    col = np.clip((match.ref_pts[:, 0] * grid_cols / max(width, 1)).astype(int),
                  0, grid_cols - 1)
    row = np.clip((match.ref_pts[:, 1] * grid_rows / max(height, 1)).astype(int),
                  0, grid_rows - 1)
    cell = row * grid_cols + col
    members = [np.flatnonzero((cell == idx) & finite)
               for idx in range(grid_rows * grid_cols)]

    keep = np.zeros(count, dtype=bool)
    thresholds: list[float] = []
    supported_cells = 0
    for cell_idx, own_idx in enumerate(members):
        if len(own_idx) == 0:
            continue
        support_idx = own_idx
        if len(support_idx) < min_support:
            cell_row, cell_col = divmod(cell_idx, grid_cols)
            neighbors = [
                members[r * grid_cols + c]
                for r in range(max(0, cell_row - 1),
                               min(grid_rows, cell_row + 2))
                for c in range(max(0, cell_col - 1),
                               min(grid_cols, cell_col + 2))
                if len(members[r * grid_cols + c])
            ]
            if neighbors:
                support_idx = np.concatenate(neighbors)
        if len(support_idx) < min_support:
            # Insufficient evidence is not evidence of an outlier.
            keep[own_idx] = True
            continue

        supported_cells += 1
        local_vote = np.median(residual[support_idx], axis=0)
        support_deviation = np.linalg.norm(
            residual[support_idx] - local_vote, axis=1)
        deviation_median = float(np.median(support_deviation))
        mad = float(np.median(np.abs(support_deviation - deviation_median)))
        threshold = max(float(floor_px),
                        deviation_median + float(sigma) * 1.4826 * mad)
        thresholds.append(threshold)
        own_deviation = np.linalg.norm(residual[own_idx] - local_vote, axis=1)
        keep[own_idx] = own_deviation <= threshold

    kept_idx = np.flatnonzero(keep & finite)
    filtered = MatchResult(
        pair_idx=match.pair_idx[kept_idx],
        ref_pts=match.ref_pts[kept_idx],
        src_pts=match.src_pts[kept_idx],
        rotation=match.rotation,
        homography=match.homography,
    )
    stats: dict[str, float | int] = {
        "input_pairs": count,
        "kept_pairs": len(kept_idx),
        "rejected_pairs": count - len(kept_idx),
        "supported_cells": supported_cells,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "threshold_median_px": (float(np.median(thresholds))
                                if thresholds else float("nan")),
    }
    logger.debug(
        "Guided spatial filter: pairs={}->{} rejected={} grid={}x{} "
        "supported_cells={} threshold_median={:.3f}px",
        count, len(kept_idx), count - len(kept_idx), grid_cols, grid_rows,
        supported_cells, stats["threshold_median_px"],
    )
    return filtered, stats


def optimize_alignment(
    match: MatchResult,
    camera1: BaseCameraModel,
    camera2: BaseCameraModel,
    same_camera: bool = False,
    n_dist: Optional[int] = None,
    ref_policy: Optional[CameraOptimizationPolicy] = None,
    src_policy: Optional[CameraOptimizationPolicy] = None,
    focal_regularization_weight: float = 0.0,
) -> AlignmentResult:
    """Optimize rotation and camera parameters from matched points.

    Args:
        match: MatchResult from match_star_pairs.
        camera1, camera2: initial camera models for ref and src images.
        same_camera: whether both images are from the same camera.
        n_dist: number of distortion parameters to optimize. Defaults to
            3 for fisheye cameras and 4 for perspective cameras.
        focal_regularization_weight: prior weight for fisheye focal scale.
            The default is zero; pass ``1.0`` to restore the previous prior.

    Returns:
        AlignmentResult with optimized rotation and refined cameras.
    """
    ref_pts = match.ref_pts
    src_pts = match.src_pts

    ref_projection = _camera_projection_type(camera1)
    src_projection = _camera_projection_type(camera2)
    if same_camera and ref_projection != src_projection:
        logger.warning(
            "optimize_alignment: mixed projection cameras cannot use same_camera=True; forcing same_camera=False"
        )
        same_camera = False

    ref_policy = ref_policy or _default_optimization_policy(camera1, n_dist)
    src_policy = src_policy or (ref_policy if same_camera else
                                _default_optimization_policy(camera2, n_dist))
    logger.debug("optimize_alignment: "
                 f"ref_projection={ref_projection}  "
                 f"src_projection={src_projection}  "
                 f"same_camera={same_camera}  "
                 f"ref_policy={ref_policy}  "
                 f"src_policy={src_policy}")

    # SVD (Kabsch) on matched unit vectors is projection-family agnostic and
    # works for perspective, fisheye, and mixed camera pairs.
    ref_vecs = camera1.unproject(ref_pts)
    src_vecs = camera2.unproject(src_pts)
    H_svd = ref_vecs.T @ src_vecs
    U, _, Vt = np.linalg.svd(H_svd)
    R_init = Vt.T @ U.T
    if np.linalg.det(R_init) < 0:
        Vt[-1, :] *= -1
        R_init = Vt.T @ U.T

    rvec, _ = cv2.Rodrigues(R_init)
    rvec = rvec[:, 0]

    ctx = FlexibleOptimizationContext(
        ref_pts=ref_pts,
        src_pts=src_pts,
        ref_state=_camera_optimization_state(camera1, ref_policy),
        src_state=_camera_optimization_state(camera2, src_policy),
        same_camera=same_camera,
    )
    x0 = pack_flexible_initial_params(rvec, ctx)
    ctx.params0 = x0.copy()
    ctx.reg_weight = make_flexible_regularization_weights(
        ctx, x0, fisheye_focal_weight=float(focal_regularization_weight))

    # jointly optimize rotation and camera parameters, with optional regularization
    res = run_flexible_optimization(x0, ctx, max_nfev=300)
    _log_flexible_optimization_summary(ctx, res.x)
    return _build_flexible_result(res.x, ctx, camera1, camera2)


def guided_refine_alignment(
    ref_geo: GeometryView,
    src_geo: GeometryView,
    alignment: AlignmentResult,
    same_camera: bool = False,
    max_distance_px: float = 8.0,
    ref_policy: Optional[CameraOptimizationPolicy] = None,
    src_policy: Optional[CameraOptimizationPolicy] = None,
    focal_regularization_weight: float = 0.0,
) -> tuple[AlignmentResult, MatchResult]:
    """Run one optional local-rematch and camera re-optimization stage."""
    guided_match = guided_mutual_rematch(
        ref_geo,
        src_geo,
        alignment,
        max_distance_px=max_distance_px,
    )
    if len(guided_match.pair_idx) < 6:
        raise ValueError(
            "guided refinement requires at least 6 mutual pairs, "
            f"got {len(guided_match.pair_idx)}")

    refined = optimize_alignment(
        guided_match,
        alignment.ref_camera,
        alignment.src_camera,
        same_camera=same_camera,
        ref_policy=ref_policy,
        src_policy=src_policy,
        focal_regularization_weight=focal_regularization_weight,
    )
    return refined, guided_match


def _camera_projection_type(camera: BaseCameraModel) -> str:
    return "fisheye" if isinstance(camera,
                                   FisheyeCameraModel) else "perspective"


def _default_optimization_policy(
    camera: BaseCameraModel,
    n_dist: Optional[int],
) -> CameraOptimizationPolicy:
    if isinstance(camera, FisheyeCameraModel):
        return CameraOptimizationPolicy(
            optimize_focal=True,
            optimize_distortion=True,
            optimize_principal_point=False,
            n_dist=3 if n_dist is None else n_dist,
        )
    return CameraOptimizationPolicy(
        optimize_focal=True,
        optimize_distortion=True,
        optimize_principal_point=False,
        n_dist=4 if n_dist is None else n_dist,
    )


def _camera_optimization_state(
    camera: BaseCameraModel,
    policy: CameraOptimizationPolicy,
) -> CameraOptimizationState:
    intr = camera.intrinsics
    cx, cy = intr.principal_point_px

    def _camera_distortion_init(
        camera: BaseCameraModel,
        n_dist: int,
    ) -> NDArray[np.float64]:
        if n_dist <= 0:
            return np.zeros(0, dtype=np.float64)
        if camera.distortion.is_zero:
            return np.zeros(n_dist, dtype=np.float64)
        if isinstance(camera, FisheyeCameraModel):
            # Keep all four Kannala--Brandt coefficients in the optimization
            # state.  The policy may optimize only a prefix (e.g. k1..k3),
            # but the remaining initialized coefficients must still be used
            # by every residual evaluation.
            return camera.distortion.to_cv2().copy()
        return camera.distortion.to_opt_params(n_dist)

    return CameraOptimizationState(
        projection_type=_camera_projection_type(camera),
        base_focal=intr.focal_length_mm,
        sensor_w_mm=intr.sensor_width_mm,
        sensor_h_mm=intr.sensor_height_mm,
        img_w=intr.image_width_px,
        img_h=intr.image_height_px,
        base_cx=cx,
        base_cy=cy,
        base_distortion=_camera_distortion_init(camera, policy.n_dist),
        policy=policy,
    )


def _log_flexible_optimization_summary(
    ctx: FlexibleOptimizationContext,
    x_opt: NDArray[np.float64],
) -> None:
    diag = compute_flexible_residual_diagnostics(x_opt, ctx)
    _, ref_params, src_params = unpack_flexible_params(x_opt, ctx)
    reg_active = "none"
    if ctx.reg_weight is not None and np.any(ctx.reg_weight != 0):
        reg_active = ",".join(f"p{i}={weight:.3f}"
                              for i, weight in enumerate(ctx.reg_weight)
                              if weight != 0)
    logger.debug(
        "Joint optimization summary: "
        f"ref_projection={ctx.ref_state.projection_type} "
        f"src_projection={ctx.src_state.projection_type} "
        f"same_camera={ctx.same_camera} "
        f"ref_policy={ctx.ref_state.policy} src_policy={ctx.src_state.policy} "
        f"cost={diag['total_cost']:.12e} data={diag['data_cost']:.12e} reg={diag['reg_cost']:.12e} "
        f"residual_px[median={diag['raw_angle_median_px']:.6f},"
        f"p90={diag['raw_angle_p90_px']:.6f},"
        f"max={diag['raw_angle_max_px']:.6f},"
        f"rms={diag['raw_angle_rms_px']:.6f}] "
        f"ref_focal={ctx.ref_state.base_focal * (1.0 + ref_params.focal_scale):.6f}mm "
        f"src_focal={ctx.src_state.base_focal * (1.0 + src_params.focal_scale):.6f}mm "
        f"ref_distortion={_format_array(ref_params.distortion, precision=6)} "
        f"src_distortion={_format_array(src_params.distortion, precision=6)} "
        f"reg_active={reg_active}")


def _camera_with_solved_params(
    camera: BaseCameraModel,
    state: CameraOptimizationState,
    params: CameraSolvedParams,
) -> BaseCameraModel:
    focal = state.base_focal * (1.0 + params.focal_scale)
    cx = state.base_cx + params.principal_point_offset_x_px
    cy = state.base_cy + params.principal_point_offset_y_px
    refined = camera.with_intrinsics(
        camera.intrinsics.with_focal_length(focal).with_principal_point(
            cx, cy))
    if isinstance(camera, FisheyeCameraModel):
        # Only the first policy.n_dist coefficients are optimized. Preserve
        # any remaining initialized coefficients instead of padding them with
        # zeros (the default fisheye policy optimizes k1..k3 and holds k4).
        dist_arr = camera.distortion.to_cv2().copy()
        n_solved = min(len(params.distortion), len(dist_arr))
        dist_arr[:n_solved] = params.distortion[:n_solved]
        return refined.with_distortion(
            FisheyeDistortion.from_array(dist_arr))

    dist_arr = np.zeros(5, dtype=np.float64)
    if len(params.distortion) > 0:
        dist_arr[:min(len(params.distortion), 5)] = params.distortion[:5]
    return refined.with_distortion(Distortion.from_cv2(dist_arr))


def _build_flexible_result(params_flat, ctx, camera1,
                           camera2) -> AlignmentResult:
    rvec, ref_params, src_params = unpack_flexible_params(params_flat, ctx)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    cam1_refined = _camera_with_solved_params(camera1, ctx.ref_state,
                                              ref_params)
    if ctx.same_camera:
        cam2_refined = dataclasses.replace(
            camera2,
            intrinsics=cam1_refined.intrinsics,
            distortion=cam1_refined.distortion,
        )
    else:
        cam2_refined = _camera_with_solved_params(camera2, ctx.src_state,
                                                  src_params)
    return AlignmentResult(rotation_ref_to_src=R,
                           ref_camera=cam1_refined,
                           src_camera=cam2_refined)
