"""Tests for norma matching diagnostics and refinement."""
import numpy as np
import pytest

from hoshicore.component.norma.matching import find_initial_match, fine_tune_rotation


def _make_features_with_controlled_diag_distance(diag_distances: list[float]) -> tuple[np.ndarray, np.ndarray]:
    n = len(diag_distances)
    features1 = np.zeros((n, 2 * n), dtype=np.float64)
    features2 = np.zeros((n, 2 * n), dtype=np.float64)
    for i, dist in enumerate(diag_distances):
        cos_theta = 1.0 - dist
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
        features1[i, i] = 1.0
        features2[i, i] = cos_theta
        features2[i, n + i] = sin_theta
    return features1, features2


def test_fine_tune_rotation_accepts_consistent_pairs_with_duplicates():
    vectors1 = np.array([
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 0.995],
        [-0.1, 0.0, 0.995],
        [0.0, 0.1, 0.995],
        [0.0, -0.1, 0.995],
        [0.08, 0.07, 0.994],
    ], dtype=np.float64)
    vectors1 /= np.linalg.norm(vectors1, axis=1, keepdims=True)
    angle = np.deg2rad(0.4)
    R = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ], dtype=np.float64)
    vectors2 = (R @ vectors1.T).T
    pts = np.array([
        [20.0, 20.0],
        [80.0, 20.0],
        [80.0, 80.0],
        [20.0, 80.0],
        [50.0, 50.0],
        [65.0, 40.0],
    ], dtype=np.float64)
    init_pair_idx = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [0, 0],
    ], dtype=np.int32)

    R_est, pair_idx = fine_tune_rotation(
        pts, pts, vectors1, vectors2, init_pair_idx)

    assert len(np.unique(pair_idx, axis=0)) >= 6
    np.testing.assert_allclose(R_est, R, atol=1e-10)


def test_fine_tune_rotation_rejects_when_unique_pairs_below_minimum():
    pts = np.zeros((5, 2), dtype=np.float64)
    vectors = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (5, 1))
    init_pair_idx = np.array([
        [0, 0],
        [1, 1],
        [0, 0],
        [2, 2],
        [1, 1],
    ], dtype=np.int32)

    with pytest.raises(ValueError, match="at least 6 unique"):
        fine_tune_rotation(pts, pts, vectors, vectors, init_pair_idx)


def test_find_initial_match_fallbacks_when_filtered_unique_pairs_below_four():
    n = 20
    features1, features2 = _make_features_with_controlled_diag_distance(
        [0.001] * 6 + [0.2] * (n - 6))
    pts = np.stack((np.arange(n, dtype=np.float64) * 10.0,
                    np.zeros(n, dtype=np.float64)), axis=1)
    vectors1 = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n, 1))
    vectors2 = vectors1.copy()
    vectors2[3:] = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    pair_idx = find_initial_match(
        features1, features2, pts, pts,
        vectors1=vectors1, vectors2=vectors2,
        apply_threshold_filter=True)

    assert len(pair_idx) == 6
    np.testing.assert_array_equal(
        pair_idx,
        np.column_stack((np.arange(6, dtype=np.int32), np.arange(6, dtype=np.int32))))


def test_find_initial_match_fallbacks_when_keep_ratio_too_low_on_small_set():
    n = 30
    features1, features2 = _make_features_with_controlled_diag_distance(
        [0.001] * 9 + [0.2] * (n - 9))
    pts = np.stack((np.arange(n, dtype=np.float64) * 10.0,
                    np.zeros(n, dtype=np.float64)), axis=1)
    vectors1 = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n, 1))
    vectors2 = vectors1.copy()
    vectors2[4:9] = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    pair_idx = find_initial_match(
        features1, features2, pts, pts,
        vectors1=vectors1, vectors2=vectors2,
        apply_threshold_filter=True)

    assert len(pair_idx) == 9
    np.testing.assert_array_equal(
        pair_idx,
        np.column_stack((np.arange(9, dtype=np.int32), np.arange(9, dtype=np.int32))))
