from unittest import mock

import numpy as np

from hoshicore._custom_op import (
    extract_point_features,
    find_initial_match,
)
import hoshicore._custom_op.ops.alignment as alignment_ops
import hoshicore.component.norma.matching as norma_matching


from tests.custom_ops._base import CustomOpsTestCase


def _make_alignment_match_inputs(seed: int = 0, n_points: int = 96):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(n_points, 3))
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    vec2 = vec + rng.normal(scale=1e-4, size=vec.shape)
    vec2 = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    vol = rng.random(n_points) * 10 + 1
    vol2 = vol * (1.0 + rng.normal(scale=1e-3, size=n_points))
    pts = rng.random((n_points, 2)) * 1000
    pts2 = pts + rng.normal(scale=1.0, size=pts.shape)
    return vec, vec2, vol, vol2, pts, pts2


class TestAlignmentCustomOps(CustomOpsTestCase):
    def test_extract_point_features_compiled_matches_numpy(self) -> None:
        vec, _, vol, _, _, _ = _make_alignment_match_inputs(seed=1)

        got = alignment_ops.extract_point_features_compiled(vec, vol, k=8)
        expected = alignment_ops.extract_point_features_numpy(vec, vol, k=8)

        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_find_initial_match_compiled_matches_numpy(self) -> None:
        vec, vec2, vol, vol2, pts, pts2 = _make_alignment_match_inputs(seed=2)
        features1 = alignment_ops.extract_point_features_numpy(vec, vol, k=8)
        features2 = alignment_ops.extract_point_features_numpy(vec2, vol2, k=8)

        got = alignment_ops.find_initial_match_compiled(
            features1,
            features2,
            pts,
            pts2,
            vec,
            vec2,
        )
        expected = alignment_ops.find_initial_match_numpy(
            features1,
            features2,
            pts,
            pts2,
            vec,
            vec2,
        )

        self.assertGreater(len(expected), 0)
        np.testing.assert_array_equal(got, expected)

    def test_alignment_matching_can_force_numpy_fallback(self) -> None:
        vec, vec2, vol, vol2, pts, pts2 = _make_alignment_match_inputs(seed=3)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                alignment_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                alignment_ops._select_extract_point_features_backend.cache_clear()
                alignment_ops._select_find_initial_match_backend.cache_clear()
                features1 = extract_point_features(vec, vol, k=8)
                features2 = extract_point_features(vec2, vol2, k=8)
                got = find_initial_match(features1, features2, pts, pts2, vec, vec2)

        expected_features1 = alignment_ops.extract_point_features_numpy(vec, vol, k=8)
        expected_features2 = alignment_ops.extract_point_features_numpy(vec2, vol2, k=8)
        expected = alignment_ops.find_initial_match_numpy(
            expected_features1,
            expected_features2,
            pts,
            pts2,
            vec,
            vec2,
        )
        np.testing.assert_allclose(
            features1, expected_features1, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            features2, expected_features2, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_array_equal(got, expected)

    def test_alignment_public_dispatch_uses_compiled_backend(self) -> None:
        vec, vec2, vol, vol2, pts, pts2 = _make_alignment_match_inputs(seed=4)
        alignment_ops._select_extract_point_features_backend.cache_clear()
        alignment_ops._select_find_initial_match_backend.cache_clear()

        with mock.patch.object(
            alignment_ops,
            "extract_point_features_compiled",
            wraps=alignment_ops.extract_point_features_compiled,
        ) as patched_extract:
            with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"}, clear=False
            ):
                features1 = extract_point_features(vec, vol, k=8)
                features2 = extract_point_features(vec2, vol2, k=8)

        with mock.patch.object(
            alignment_ops,
            "find_initial_match_compiled",
            wraps=alignment_ops.find_initial_match_compiled,
        ) as patched_match:
            with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"}, clear=False
            ):
                _ = find_initial_match(features1, features2, pts, pts2, vec, vec2)

        self.assertGreaterEqual(patched_extract.call_count, 2)
        patched_match.assert_called_once()

    def test_norma_matching_routes_through_alignment_custom_op(self) -> None:
        vec, vec2, vol, vol2, pts, pts2 = _make_alignment_match_inputs(seed=5)

        features1 = norma_matching.extract_point_features(vec, vol, k=8)
        features2 = norma_matching.extract_point_features(vec2, vol2, k=8)
        got = norma_matching.find_initial_match(
            features1, features2, pts, pts2, vec, vec2
        )

        expected_features1 = alignment_ops.extract_point_features_numpy(vec, vol, k=8)
        expected_features2 = alignment_ops.extract_point_features_numpy(vec2, vol2, k=8)
        expected = alignment_ops.find_initial_match_numpy(
            expected_features1,
            expected_features2,
            pts,
            pts2,
            vec,
            vec2,
        )
        np.testing.assert_allclose(
            features1, expected_features1, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            features2, expected_features2, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_array_equal(got, expected)
