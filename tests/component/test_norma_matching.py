"""Tests for norma matching diagnostics and refinement."""
import numpy as np
import pytest

from hoshicore.component.norma.matching import (
    extract_asterism_tokens,
    extract_point_features,
    find_asterism_initial_match,
    find_guided_mutual_match,
    find_initial_match,
    fine_tune_rotation,
)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (np.eye(3) * np.cos(angle)
            + (1.0 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * cross)


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


def test_angular_histogram_features_are_invariant_to_global_rotation():
    rng = np.random.default_rng(1234)
    vectors = rng.normal(size=(40, 3))
    vectors[:, 2] += 3.0
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    volumes = rng.uniform(0.5, 3.0, size=len(vectors))
    angle = np.deg2rad(37.0)
    axis = np.array([0.3, -0.4, 0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    rotation = (np.eye(3) * np.cos(angle)
                + (1.0 - np.cos(angle)) * np.outer(axis, axis)
                + np.sin(angle) * cross)

    features = extract_point_features(vectors, volumes, k=8)
    rotated_features = extract_point_features(
        (rotation @ vectors.T).T, volumes, k=8)

    assert features.shape == (40, 120)
    assert np.all(np.isfinite(features))
    np.testing.assert_allclose(rotated_features, features, rtol=1e-9,
                               atol=1e-10)


def test_asterism_tokens_are_rotation_invariant():
    rng = np.random.default_rng(2301)
    vectors = rng.normal(size=(50, 3))
    vectors[:, 2] += 4.0
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    rotation = _axis_angle_rotation(np.array([0.4, -0.2, 0.7]),
                                    np.deg2rad(31.0))

    tokens = extract_asterism_tokens(vectors, neighbor_count=8)
    rotated = extract_asterism_tokens(
        (rotation @ vectors.T).T, neighbor_count=8)

    np.testing.assert_array_equal(rotated.anchor_indices,
                                  tokens.anchor_indices)
    np.testing.assert_allclose(rotated.values, tokens.values, rtol=1e-10,
                               atol=1e-11)


def test_asterism_matching_tolerates_missing_stars():
    rng = np.random.default_rng(2302)
    xy = rng.normal(scale=0.22, size=(80, 2))
    vectors = np.column_stack((xy, np.ones(80)))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    rotation = _axis_angle_rotation(np.array([0.3, -0.6, 0.2]),
                                    np.deg2rad(17.0))
    retained = np.sort(rng.choice(len(vectors), 60, replace=False))
    permutation = rng.permutation(len(retained))
    source = (rotation @ vectors[retained][permutation].T).T
    expected_source = {
        int(retained[permutation[index]]): index
        for index in range(len(retained))
    }

    pairs = find_asterism_initial_match(vectors, source)
    correct = sum(
        expected_source.get(int(ref_index), -1) == int(src_index)
        for ref_index, src_index in pairs)

    assert len(pairs) >= 50
    assert correct == len(pairs)


def test_asterism_matching_abstains_without_repeated_constellation_votes():
    rng = np.random.default_rng(2303)
    first = rng.normal(size=(120, 3))
    second = rng.normal(size=(110, 3))
    first[:, 2] += 4.0
    second[:, 2] += 4.0
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second /= np.linalg.norm(second, axis=1, keepdims=True)

    pairs = find_asterism_initial_match(first, second)

    assert len(pairs) == 0


def test_low_ranked_neighbor_does_not_contribute_to_original_top_k():
    vectors = [np.array([0.0, 0.0, 1.0], dtype=np.float64)]
    for index, radius in enumerate(np.linspace(0.01, 0.08, 8)):
        azimuth = index * 0.71
        vector = np.array([
            radius * np.cos(azimuth),
            radius * np.sin(azimuth),
            1.0,
        ])
        vectors.append(vector / np.linalg.norm(vector))
    vectors = np.asarray(vectors)
    volumes = np.array([1.0, 100.0, 100.0, 100.0, 0.5, 0.5, 0.0, 0.5,
                        0.5])

    without_neighbor = extract_point_features(vectors, volumes, k=3)[0]
    volumes[5] = 10.0  # Still below the top-k volume*distance cutoff.
    with_neighbor = extract_point_features(vectors, volumes, k=3)[0]

    np.testing.assert_allclose(with_neighbor, without_neighbor, rtol=0,
                               atol=1e-12)


def test_guided_mutual_match_uses_bidirectional_projected_positions():
    pts1 = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
    ])
    pts2 = np.array([
        [3.0, 1.0],
        [13.0, 1.0],
        [50.0, 1.0],
    ])
    predicted_pts2 = np.array([
        [3.2, 1.1],
        [13.2, 1.1],
        [23.2, 1.1],
    ])
    predicted_pts1 = np.array([
        [0.2, 0.1],
        [10.2, 0.1],
        [47.2, 0.1],
    ])

    pair_idx = find_guided_mutual_match(
        pts1,
        pts2,
        predicted_pts2,
        predicted_pts1,
        max_distance_px=2.0,
    )

    np.testing.assert_array_equal(
        pair_idx,
        np.array([[0, 0], [1, 1]], dtype=np.int32),
    )


def test_guided_mutual_match_rejects_one_way_nearest_neighbor():
    pts1 = np.array([[0.0, 0.0], [10.0, 0.0]])
    pts2 = np.array([[1.0, 0.0], [30.0, 0.0]])
    predicted_pts2 = np.array([[1.0, 0.0], [1.1, 0.0]])
    predicted_pts1 = np.array([[10.0, 0.0], [10.0, 0.0]])

    pair_idx = find_guided_mutual_match(
        pts1,
        pts2,
        predicted_pts2,
        predicted_pts1,
        max_distance_px=2.0,
    )

    np.testing.assert_array_equal(
        pair_idx,
        np.array([[1, 0]], dtype=np.int32),
    )


def test_fine_tune_rotation_is_reproducible_with_seed():
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

    first_rotation, first_pairs = fine_tune_rotation(
        pts, pts, vectors1, vectors2, init_pair_idx, random_seed=42)
    second_rotation, second_pairs = fine_tune_rotation(
        pts, pts, vectors1, vectors2, init_pair_idx, random_seed=42)

    np.testing.assert_array_equal(second_pairs, first_pairs)
    np.testing.assert_array_equal(second_rotation, first_rotation)
    assert len(np.unique(first_pairs, axis=0)) >= 6
    np.testing.assert_allclose(first_rotation, R, atol=1e-10)


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
