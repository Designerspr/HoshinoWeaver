"""Tests for projection-guided second-stage star rematching."""
import cv2
import numpy as np

from hoshicore.component.norma.alignment import (
    AlignmentResult,
    guided_mutual_rematch,
    run_guided_refine_stage,
)
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
    ref_geo = GeometryView(
        DetectedStars(ref_positions, volumes), camera,
    )
    src_geo = GeometryView(
        DetectedStars(src_positions, volumes), camera,
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
    refined, refined_match, status = run_guided_refine_stage(
        ref_geo,
        src_geo,
        initial,
        bootstrap_match=guided_match,
        same_camera=True,
        max_distance_px=3.0,
        ref_policy=fixed_camera_policy,
        src_policy=fixed_camera_policy,
    )

    assert status == "applied"
    assert len(refined_match.pair_idx) == len(ref_positions)
    np.testing.assert_allclose(
        refined.rotation_ref_to_src, true_rotation, atol=1e-9)
