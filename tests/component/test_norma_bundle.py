from types import SimpleNamespace

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
    _build_edges,
    _camera_parameter_bounds,
    _camera_observability,
    _make_edge,
    _sample_edge_pairs,
    _solve_bundle_parameters,
    _spatial_pair_bins,
    build_bundle_plan,
)
from hoshicore.component.norma.detection import DetectedStars
from hoshicore.component.norma.frame_align import AlignmentCameraCandidate
from hoshicore.component.norma.optimization import CameraOptimizationPolicy
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics


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


def test_camera_parameter_bounds_limit_focal_and_distortion():
    policy = CameraOptimizationPolicy(True, True, False, 4)
    lower, upper = _camera_parameter_bounds(policy)
    np.testing.assert_allclose(lower, [-0.3, -1.0, -1.0, -1.0, -1.0])
    np.testing.assert_allclose(upper, [0.3, 1.0, 1.0, 1.0, 1.0])


def test_edge_sampling_reserves_spatial_bins_then_caps_deterministically():
    center = np.array([50.0, 50.0])
    points = []
    for normalized_radius in (0.2, 0.6, 1.0):
        for sector in range(8):
            angle = (sector + 0.25) * 2.0 * np.pi / 8.0
            point = center + 50.0 * normalized_radius * np.array([
                np.cos(angle), np.sin(angle)])
            points.extend([point, point + 0.01])
    points = np.asarray(points)
    edge = _BundleEdge(0, 1, points, points + 1.0, np.eye(3))

    sampled = _sample_edge_pairs(edge, 24, 100, 100, random_seed=7)
    repeated = _sample_edge_pairs(edge, 24, 100, 100, random_seed=7)

    assert len(sampled.first_pts) == 24
    assert len(np.unique(_spatial_pair_bins(
        sampled.first_pts, 100, 100))) == 24
    np.testing.assert_array_equal(sampled.first_pts, repeated.first_pts)
    np.testing.assert_array_equal(sampled.second_pts, repeated.second_pts)


def test_edge_sampling_can_be_disabled():
    points = np.arange(40, dtype=np.float64).reshape(20, 2)
    edge = _BundleEdge(0, 1, points, points, np.eye(3))
    assert _sample_edge_pairs(edge, None, 100, 100, 0) is edge
    assert _sample_edge_pairs(edge, 0, 100, 100, 0) is edge


def test_bundle_solver_applies_camera_bounds_and_leaves_poses_unbounded(
        monkeypatch):
    camera = _camera()
    policy = CameraOptimizationPolicy(True, True, False, 4)
    points = np.tile(
        np.array([[0.0, 0.0], [119.0, 0.0],
                  [0.0, 79.0], [119.0, 79.0]]), (6, 1))
    edge = _BundleEdge(
        0, 1, points, points.copy(), np.eye(3))
    captured = {}

    def fake_least_squares(fun, x0, *, args, bounds, **kwargs):
        captured["lower"], captured["upper"] = bounds
        return SimpleNamespace(
            success=True, x=x0, active_mask=np.zeros_like(x0),
            message="ok", jac=np.ones((72, len(x0))))

    monkeypatch.setattr(bundle_module, "least_squares", fake_least_squares)
    monkeypatch.setattr(bundle_module, "_camera_observability",
                        lambda jacobian, camera_width: 1.0)
    _solve_bundle_parameters(
        [edge], 0, {0, 1}, camera, policy, max_nfev=10)

    np.testing.assert_allclose(
        captured["lower"][:5], [-0.3, -1.0, -1.0, -1.0, -1.0])
    np.testing.assert_allclose(
        captured["upper"][:5], [0.3, 1.0, 1.0, 1.0, 1.0])
    assert np.all(np.isneginf(captured["lower"][5:]))
    assert np.all(np.isposinf(captured["upper"][5:]))


def test_bundle_falls_back_to_focal_only_without_rebuilding_edges(monkeypatch):
    camera = _camera()
    requested = CameraOptimizationPolicy(True, True, False, 4)
    candidate = AlignmentCameraCandidate(camera, requested, "manual")
    stars = DetectedStars(np.empty((0, 2)), np.empty(0))
    frames = [BundleFrame(index, stars, candidate) for index in range(2)]
    edge_calls = []

    def fake_edge(first, second, random_seed, bootstrap_scales):
        edge_calls.append((first.index, second.index))
        return _BundleEdge(
            first.index, second.index,
            np.zeros((6, 2)), np.zeros((6, 2)), np.eye(3))

    solve_policies = []

    def fake_solve(edges, reference_index, component, initial_camera,
                   policy, max_nfev):
        solve_policies.append(policy)
        if len(solve_policies) == 1:
            raise BundleAdjustmentError("shared camera parameters are rank deficient")
        return (initial_camera,
                {0: np.eye(3), 1: np.eye(3)}, list(edges), 1.0)

    monkeypatch.setattr(bundle_module, "_make_edge", fake_edge)
    monkeypatch.setattr(bundle_module, "_solve_bundle_parameters", fake_solve)
    plan = build_bundle_plan(frames, 0, pair_offsets=(1,))

    assert edge_calls == [(0, 1)]
    assert solve_policies == [
        requested,
        CameraOptimizationPolicy(True, False, False, 0),
    ]
    assert plan.camera_solve_mode == "focal_fallback"
    assert plan.camera_fallback_reason == (
        "shared camera parameters are rank deficient")
    assert plan.active_camera_parameter_count == 1


def test_plan_frame_access():
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


def test_bundle_edge_refines_rotation_without_camera_parameters(monkeypatch):
    camera = _camera()
    candidate = AlignmentCameraCandidate(
        camera, CameraOptimizationPolicy(True, True, True, 4), "manual")
    stars = DetectedStars(np.empty((0, 2)), np.empty(0))
    calls = []

    def fake_solve(ref_stars, src_stars, ref_candidate, src_candidate, **kwargs):
        calls.append((ref_candidate, src_candidate, kwargs))
        assert ref_candidate.optimization_policy == CameraOptimizationPolicy(
            False, False, False, 0)
        assert src_candidate.optimization_policy == CameraOptimizationPolicy(
            False, False, False, 0)
        alignment = SimpleNamespace(
            rotation_ref_to_src=np.eye(3),
            ref_camera=ref_candidate.camera.with_focal_length(14.0),
        )
        match = SimpleNamespace(
            pair_idx=np.zeros((6, 2), dtype=np.int32),
            ref_pts=np.zeros((6, 2)), src_pts=np.zeros((6, 2)))
        return alignment, match

    monkeypatch.setattr(bundle_module, "solve_star_alignment", fake_solve)
    edge = _make_edge(BundleFrame(0, stars, candidate),
                      BundleFrame(1, stars, candidate), 7,
                      bootstrap_scales=(0.7, 1.0, 1.3))
    assert edge.error is None
    assert edge.selected_scale == pytest.approx(0.7)
    assert calls[0][2]["bootstrap_scales"] == (0.7, 1.0, 1.3)
    assert calls[0][2]["same_camera"] is True
    assert calls[0][2]["random_seed"] == 7
    assert calls[0][2]["residual_space"] == "angular"


def test_bundle_scale_votes_reuse_preferred_and_fall_back(monkeypatch):
    camera = _camera()
    candidate = AlignmentCameraCandidate(
        camera, CameraOptimizationPolicy(True, True, False, 4), "manual")
    stars = DetectedStars(np.empty((0, 2)), np.empty(0))
    frames = [BundleFrame(index, stars, candidate) for index in range(6)]
    calls = []

    def fake_edge(first, second, random_seed, bootstrap_scales):
        scales = tuple(bootstrap_scales)
        calls.append((first.index, scales))
        if first.index < 3:
            selected = 0.7
        elif first.index == 3 and scales == (0.7,):
            return _BundleEdge(first.index, second.index,
                               np.empty((0, 2)), np.empty((0, 2)),
                               np.eye(3), "preferred failed")
        elif first.index == 3:
            selected = 1.0
        else:
            selected = 0.7
        return _BundleEdge(first.index, second.index,
                           np.empty((6, 2)), np.empty((6, 2)),
                           np.eye(3), selected_scale=selected)

    monkeypatch.setattr(bundle_module, "_make_edge", fake_edge)
    completed = []
    edges, sequence_scale = _build_edges(
        frames, (1,), random_seed=0,
        edge_completed=lambda: completed.append(True))
    assert len(edges) == 5
    assert len(completed) == 5
    assert sequence_scale == pytest.approx(0.7)
    assert calls == [
        (0, (1.0, 0.7, 1.3)),
        (1, (0.7, 1.0, 1.3)),
        (2, (0.7, 1.0, 1.3)),
        (3, (0.7,)),
        (3, (1.0, 1.3)),
        (4, (0.7,)),
    ]
    assert all(edge.error is None for edge in edges)


def test_bundle_exif_perspective_uses_only_unit_scale(monkeypatch):
    camera = _camera()
    candidate = AlignmentCameraCandidate(
        camera, CameraOptimizationPolicy(True, True, False, 4), "exif")
    stars = DetectedStars(np.empty((0, 2)), np.empty(0))
    frames = [BundleFrame(index, stars, candidate) for index in range(4)]
    calls = []

    def fake_edge(first, second, random_seed, bootstrap_scales):
        calls.append(tuple(bootstrap_scales))
        return _BundleEdge(first.index, second.index,
                           np.empty((6, 2)), np.empty((6, 2)), np.eye(3))

    monkeypatch.setattr(bundle_module, "_make_edge", fake_edge)
    _, sequence_scale = _build_edges(frames, (1,), random_seed=0)
    assert calls == [(1.0,), (1.0,), (1.0,)]
    assert sequence_scale == pytest.approx(1.0)


def test_bundle_plan_jointly_recovers_relative_rotations(monkeypatch):
    camera = _camera()
    edge_camera = camera.with_focal_length(26.0)
    policy = CameraOptimizationPolicy(False, False, False, 0)
    candidate = AlignmentCameraCandidate(camera, policy, "test")
    rotations = {0: np.eye(3), 1: _rotation_z(0.03), 2: _rotation_z(0.06)}
    rays = np.array([[-0.2, -0.1, 1.0], [0.1, -0.2, 1.0], [0.2, 0.1, 1.0],
                     [-0.1, 0.2, 1.0], [0.05, 0.08, 1.0], [-0.18, 0.14, 1.0]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    def fake_edge(first, second, random_seed, bootstrap_scales):
        assert random_seed == 0
        if second.index == 3:
            return _BundleEdge(first.index, second.index,
                               np.empty((0, 2)), np.empty((0, 2)),
                               np.eye(3), "unmatched")
        first_pts = edge_camera.project((rotations[first.index] @ rays.T).T)
        second_pts = edge_camera.project((rotations[second.index] @ rays.T).T)
        relative = rotations[second.index] @ rotations[first.index].T
        return _BundleEdge(first.index, second.index, first_pts, second_pts,
                           relative, selected_scale=1.3)

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
    assert plan.shared_camera.intrinsics.focal_length_mm == pytest.approx(26.0)
    assert not plan.frame(2).rotation_ref_to_src.flags.writeable
    assert plan.accepted_edge_count == 2
    assert plan.frame(3).rotation_ref_to_src is None
    assert plan.rejected_edge_count == 1
