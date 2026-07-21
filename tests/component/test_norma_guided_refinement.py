"""Tests for projection-guided second-stage star rematching."""
import cv2
import numpy as np

from hoshicore.component.norma.alignment import (
    AlignmentResult,
    filter_guided_match_spatially,
    guided_mutual_rematch,
    guided_refine_alignment,
)
from hoshicore.component.norma.matching import MatchResult
from hoshicore.component.norma.detection import DetectedStars
from hoshicore.component.norma.geometry_view import GeometryView
from hoshicore.component.norma.optimization import CameraOptimizationPolicy
from hoshicore.component.norma.types import CameraModel, Intrinsics


def test_guided_refinement_recovers_native_point_pairs_and_rotation():
    intrinsics = Intrinsics(
        focal_length_mm=20.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=600,
        image_height_px=400,
    )
    camera = CameraModel(intrinsics=intrinsics)
    ref_positions = np.array([
        [100.0, 80.0],
        [200.0, 75.0],
        [300.0, 90.0],
        [400.0, 85.0],
        [500.0, 100.0],
        [120.0, 200.0],
        [240.0, 190.0],
        [360.0, 210.0],
        [480.0, 195.0],
        [150.0, 320.0],
        [300.0, 310.0],
        [450.0, 325.0],
    ])
    true_rvec = np.deg2rad(np.array([0.12, -0.18, 0.08]))
    true_rotation, _ = cv2.Rodrigues(true_rvec)
    src_positions = camera.project(
        (true_rotation @ camera.unproject(ref_positions).T).T)

    volumes = np.ones(len(ref_positions), dtype=np.float64)
    image_gray = np.zeros((400, 600), dtype=np.float64)
    ref_geo = GeometryView(
        image_gray,
        camera,
        detected_stars=DetectedStars(ref_positions, volumes),
    )
    src_geo = GeometryView(
        image_gray,
        camera,
        detected_stars=DetectedStars(src_positions, volumes),
    )

    initial_rvec = np.deg2rad(np.array([0.10, -0.15, 0.07]))
    initial_rotation, _ = cv2.Rodrigues(initial_rvec)
    initial = AlignmentResult(
        rotation_ref_to_src=initial_rotation,
        ref_camera=camera,
        src_camera=camera,
    )

    guided_match = guided_mutual_rematch(
        ref_geo, src_geo, initial, max_distance_px=3.0)
    np.testing.assert_array_equal(
        guided_match.pair_idx,
        np.column_stack((np.arange(len(ref_positions), dtype=np.int32), ) * 2),
    )

    fixed_camera_policy = CameraOptimizationPolicy(
        optimize_focal=False,
        optimize_distortion=False,
        optimize_principal_point=False,
        n_dist=0,
    )
    refined, refined_match = guided_refine_alignment(
        ref_geo,
        src_geo,
        initial,
        same_camera=True,
        max_distance_px=3.0,
        ref_policy=fixed_camera_policy,
        src_policy=fixed_camera_policy,
    )

    assert len(refined_match.pair_idx) == len(ref_positions)
    np.testing.assert_allclose(
        refined.rotation_ref_to_src, true_rotation, atol=1e-9)


def test_spatial_filter_rejects_local_residual_outliers_not_smooth_correction():
    intrinsics = Intrinsics(
        focal_length_mm=20.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=600,
        image_height_px=400,
    )
    camera = CameraModel(intrinsics=intrinsics)
    xx, yy = np.meshgrid(np.linspace(30, 570, 20), np.linspace(30, 370, 12))
    ref = np.column_stack((xx.ravel(), yy.ravel()))
    smooth = np.column_stack((
        2.0 + 0.002 * (ref[:, 0] - 300.0),
        -1.0 + 0.001 * (ref[:, 1] - 200.0),
    ))
    src = ref + smooth
    outliers = np.arange(0, len(ref), 30)
    src[outliers] += np.array([12.0, -9.0])
    pair_idx = np.column_stack((np.arange(len(ref)), np.arange(len(ref)))).astype(
        np.int32)
    match = MatchResult(
        pair_idx=pair_idx,
        ref_pts=ref,
        src_pts=src,
        rotation=np.eye(3),
        homography=np.eye(3),
    )
    initial = AlignmentResult(np.eye(3), camera, camera)

    filtered, stats = filter_guided_match_spatially(
        match, initial, (400, 600), grid_cols=6, grid_rows=4)

    kept = set(filtered.pair_idx[:, 0].tolist())
    assert kept.isdisjoint(outliers.tolist())
    assert len(filtered.pair_idx) == len(ref) - len(outliers)
    assert stats["rejected_pairs"] == len(outliers)
