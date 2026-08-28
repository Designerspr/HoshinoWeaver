import numpy as np
import pytest

import hoshicore.component.norma.bundle as bundle_module
from hoshicore.component.norma.bundle import (
    BAAlignmentPlan,
    BundleFrame,
    BundleAdjustmentError,
    FrameAlignment,
    FrameAlignmentStatus,
    _BundleEdge,
    _camera_observability,
    build_bundle_plan,
)
from hoshicore.component.norma.detection import DetectedStars
from hoshicore.component.norma.frame_align import AlignmentCameraCandidate
from hoshicore.component.norma.optimization import CameraOptimizationPolicy
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics
from hoshicore.ops.alignment_ops import _evaluate_remap


def _rotation_z(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])


def _camera() -> CameraModel:
    return CameraModel(Intrinsics(20.0, 36.0, 24.0, 120, 80), Distortion())


def test_observability_eliminates_pose_nuisance_columns():
    # First column is a camera parameter independent from the pose column.
    condition = _camera_observability(np.array([[1.0, 0.0], [0.0, 1.0],
                                                 [1.0, 1.0]]), 1)
    assert condition == pytest.approx(1.0)
    with pytest.raises(BundleAdjustmentError, match="rank deficient"):
        _camera_observability(np.array([[1.0, 1.0], [2.0, 2.0]]), 1)


def test_plan_frame_access_and_masked_remap_diagnostics():
    plan = BAAlignmentPlan(
        reference_frame_index=0, shared_camera=_camera(),
        frames=(FrameAlignment(0, FrameAlignmentStatus.SOLVED, np.eye(3),
                               "bundle"),),
        accepted_edge_count=0, rejected_edge_count=0,
        active_camera_parameter_count=0,
        observability_condition=None)
    assert plan.frame(0).status == FrameAlignmentStatus.SOLVED
    with pytest.raises(IndexError):
        plan.frame(1)
    image = np.arange(64, dtype=np.float64).reshape(8, 8)
    support, zncc, l1, bad = _evaluate_remap(image, image.copy(),
                                               np.ones((8, 8), dtype=bool), grid_size=1)
    assert support == 1.0
    assert zncc == pytest.approx(1.0)
    assert l1 == pytest.approx(0.0)
    assert bad == 0.0


def test_bundle_plan_jointly_recovers_relative_rotations(monkeypatch):
    camera = _camera()
    policy = CameraOptimizationPolicy(False, False, False, 0)
    candidate = AlignmentCameraCandidate(camera, policy, "test")
    rotations = {0: np.eye(3), 1: _rotation_z(0.03), 2: _rotation_z(0.06)}
    rays = np.array([[-0.2, -0.1, 1.0], [0.1, -0.2, 1.0], [0.2, 0.1, 1.0],
                     [-0.1, 0.2, 1.0], [0.05, 0.08, 1.0], [-0.18, 0.14, 1.0]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    def fake_edge(first, second, random_seed):
        assert random_seed == 0
        if second.index == 3:
            return _BundleEdge(first.index, second.index,
                               np.empty((0, 2)), np.empty((0, 2)),
                               np.eye(3), "unmatched")
        first_pts = camera.project((rotations[first.index] @ rays.T).T)
        second_pts = camera.project((rotations[second.index] @ rays.T).T)
        relative = rotations[second.index] @ rotations[first.index].T
        return _BundleEdge(first.index, second.index, first_pts, second_pts,
                           relative)

    monkeypatch.setattr(bundle_module, "_make_edge", fake_edge)
    stars = DetectedStars(np.empty((0, 2)), np.empty(0))
    frames = [BundleFrame(index, stars, candidate) for index in range(4)]
    plan = build_bundle_plan(frames, 0, pair_offsets=(1,))
    assert [item.status for item in plan.frames] == [
        FrameAlignmentStatus.SOLVED,
        FrameAlignmentStatus.SOLVED,
        FrameAlignmentStatus.SOLVED,
        FrameAlignmentStatus.EXCLUDED,
    ]
    np.testing.assert_allclose(plan.frame(2).rotation_ref_to_src, rotations[2],
                               atol=1e-5)
    assert not plan.frame(2).rotation_ref_to_src.flags.writeable
    assert plan.accepted_edge_count == 2
    assert plan.frame(3).rotation_ref_to_src is None
    assert plan.rejected_edge_count == 1
