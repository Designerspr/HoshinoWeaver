import os
import unittest
from unittest import mock

import numpy as np
from scipy.spatial import distance as spd

from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op.backend_registry import BackendCandidate
from hoshicore._custom_op.backend_registry import BackendSelection
import hoshicore._custom_op.cuda_memory as cuda_memory
import hoshicore._custom_op.ops.alignment as alignment_ops
import hoshicore.component.norma.matching as norma_matching


LOGICAL_OP = "matching_cosine_bidirectional_nearest"


def _random_features(
    *,
    seed: int,
    n1: int = 83,
    n2: int = 91,
    feature_dim: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n1, feature_dim)),
        rng.normal(size=(n2, feature_dim)),
    )


def _selection(backend: str) -> BackendSelection:
    kernel_name = {
        "cuda_host_io": "matching_cosine_bidirectional_nearest_cuda",
        "openmp_cpu": "matching_cosine_bidirectional_nearest_cpu",
    }[backend]
    return BackendSelection(
        BackendCandidate(LOGICAL_OP, backend, kernel_name),
        object(),
    )


def _near_tie_features() -> tuple[np.ndarray, np.ndarray]:
    features1 = np.array(
        [
            [
                1.4494748848093297,
                1.4696893531118267,
                0.6441938486509983,
                1.2342825510879065,
                -0.523976233308554,
                -1.5252258053452417,
                -1.4517599122912224,
                -0.30329993432659225,
            ],
            [
                -1.2857499092854192,
                1.363868994782076,
                0.5603988818927299,
                1.1392809293506725,
                -0.843044458971553,
                -0.7262488657658389,
                -0.6834461138619257,
                0.8075393566191182,
            ],
        ],
        dtype=np.float64,
    )
    features2 = np.array(
        [
            [
                0.5655353444984592,
                0.25487767524364063,
                -1.7232448342698463,
                -0.4170903428017445,
                1.6147536783607304,
                -0.8463109831624619,
                -1.1497356864896329,
                0.7919761765460365,
            ],
            [
                0.5655353444984592,
                0.25487767524364074,
                -1.7232448342698463,
                -0.41709034280174356,
                1.6147536783607304,
                -0.8463109831624605,
                -1.1497356864896322,
                0.7919761765460377,
            ],
            [
                -0.5570464308722523,
                -0.17289966288145642,
                -1.567832392763869,
                -0.6966074369013125,
                0.1429824634212799,
                -0.37537338093680583,
                0.5040612279530662,
                0.018591342367037984,
            ],
        ],
        dtype=np.float64,
    )
    return features1, features2


def _column_only_tie_features() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    )


class TestMatchingCosineBidirectionalNearest(unittest.TestCase):
    @staticmethod
    def _compiled_module():
        module, error = alignment_ops._load_compiled_module_result()
        if module is None:
            raise unittest.SkipTest(error or "compiled custom ops unavailable")
        return module

    def test_cpu_compiled_matches_numpy_on_rectangular_non_normalized_input(
        self,
    ) -> None:
        features1, features2 = _random_features(seed=1)
        features1 = features1[:, ::-1]
        features2 = features2[:, ::-1]

        expected = alignment_ops.matching_cosine_bidirectional_nearest_numpy(
            features1, features2)
        got = alignment_ops.matching_cosine_bidirectional_nearest_cpu_compiled(
            features1, features2)

        self.assertIsNotNone(got)
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(got[2], expected[2])
        np.testing.assert_allclose(got[3], expected[3], rtol=1e-12, atol=1e-14)

        for value in got:
            self.assertTrue(value.flags.c_contiguous)
        self.assertEqual(got[0].dtype, np.int64)
        self.assertEqual(got[1].dtype, np.float64)

    def test_cuda_matches_numpy_when_available(self) -> None:
        module = self._compiled_module()
        if not module.build_info().get("cuda"):
            self.skipTest("compiled extension is CPU-only")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))
        features1, features2 = _random_features(
            seed=2, n1=137, n2=149, feature_dim=31)

        expected = alignment_ops.matching_cosine_bidirectional_nearest_numpy(
            features1, features2)
        got = alignment_ops.matching_cosine_bidirectional_nearest_cuda(
            features1, features2)

        self.assertIsNotNone(got)
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(got[2], expected[2])
        np.testing.assert_allclose(got[3], expected[3], rtol=1e-12, atol=1e-14)

        tied = np.ones((1000, 4), dtype=np.float64)
        zero_norm = tied.copy()
        zero_norm[0] = 0.0
        nonfinite = tied.copy()
        nonfinite[0, 0] = np.nan
        near_tie = _near_tie_features()
        column_tie = _column_only_tie_features()
        for name, features_left, features_right in (
            ("tie", tied, tied),
            ("zero_norm", zero_norm, tied),
            ("nonfinite", nonfinite, tied),
            ("near_tie", near_tie[0], near_tie[1]),
            ("column_only_tie", column_tie[0], column_tie[1]),
        ):
            with self.subTest(name=name):
                self.assertIsNone(
                    module.matching_cosine_bidirectional_nearest_cuda(
                        features_left, features_right
                    )
                )

        for n1, n2 in ((1, 5), (5, 1)):
            with self.subTest(singleton_shape=(n1, n2)):
                features_left, features_right = _random_features(
                    seed=100 + n1,
                    n1=n1,
                    n2=n2,
                    feature_dim=11,
                )
                singleton_expected = (
                    alignment_ops.matching_cosine_bidirectional_nearest_numpy(
                        features_left, features_right
                    )
                )
                singleton_got = (
                    module.matching_cosine_bidirectional_nearest_cuda(
                        features_left, features_right
                    )
                )
                self.assertIsNotNone(singleton_got)
                np.testing.assert_array_equal(
                    singleton_got[0], singleton_expected[0]
                )
                np.testing.assert_array_equal(
                    singleton_got[2], singleton_expected[2]
                )

    def test_native_returns_none_for_numpy_ordering_edge_cases(self) -> None:
        module = self._compiled_module()
        tied = np.ones((1000, 4), dtype=np.float64)
        zero_norm = tied.copy()
        zero_norm[0] = 0.0
        nonfinite = tied.copy()
        nonfinite[0, 0] = np.nan
        near_tie = _near_tie_features()
        column_tie = _column_only_tie_features()

        for name, features1, features2 in (
            ("tie", tied, tied),
            ("zero_norm", zero_norm, tied),
            ("nonfinite", nonfinite, tied),
            ("near_tie", near_tie[0], near_tie[1]),
            ("column_only_tie", column_tie[0], column_tie[1]),
        ):
            with self.subTest(name=name):
                self.assertIsNone(
                    module.matching_cosine_bidirectional_nearest_cpu(
                        features1, features2))

    def test_cpu_compiled_handles_singleton_sides(self) -> None:
        for n1, n2 in ((1, 5), (5, 1)):
            with self.subTest(shape=(n1, n2)):
                features1, features2 = _random_features(
                    seed=200 + n1,
                    n1=n1,
                    n2=n2,
                    feature_dim=11,
                )
                expected = (
                    alignment_ops.matching_cosine_bidirectional_nearest_numpy(
                        features1, features2
                    )
                )
                got = (
                    alignment_ops.matching_cosine_bidirectional_nearest_cpu_compiled(
                        features1, features2
                    )
                )
                self.assertIsNotNone(got)
                np.testing.assert_array_equal(got[0], expected[0])
                np.testing.assert_allclose(
                    got[1], expected[1], rtol=1e-12, atol=1e-14
                )
                np.testing.assert_array_equal(got[2], expected[2])
                np.testing.assert_allclose(
                    got[3], expected[3], rtol=1e-12, atol=1e-14
                )

    def test_semantic_none_recomputes_with_numpy(self) -> None:
        features1, features2 = _near_tie_features()
        expected = alignment_ops.matching_cosine_bidirectional_nearest_numpy(
            features1, features2)

        with mock.patch.object(
            alignment_ops,
            "_resolve_backend",
            return_value=_selection("openmp_cpu"),
        ):
            with mock.patch.object(
                alignment_ops,
                "matching_cosine_bidirectional_nearest_numpy",
                wraps=alignment_ops.matching_cosine_bidirectional_nearest_numpy,
            ) as numpy_backend:
                got = alignment_ops.matching_cosine_bidirectional_nearest(
                    features1, features2)

        numpy_backend.assert_called_once()
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[2], expected[2])

    def test_public_wrapper_can_force_numpy(self) -> None:
        features1, features2 = _random_features(seed=3)

        with mock.patch.dict(
            os.environ, {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                alignment_ops,
                "matching_cosine_bidirectional_nearest_numpy",
                wraps=alignment_ops.matching_cosine_bidirectional_nearest_numpy,
            ) as numpy_backend:
                got = alignment_ops.matching_cosine_bidirectional_nearest(
                    features1, features2)

        numpy_backend.assert_called_once()
        self.assertEqual(got[0].shape, (len(features1),))
        self.assertEqual(got[2].shape, (len(features2),))

    def test_typed_cuda_resource_error_falls_back_to_cpu(self) -> None:
        features1, features2 = _random_features(seed=4)
        expected = alignment_ops.matching_cosine_bidirectional_nearest_numpy(
            features1, features2)

        with mock.patch.object(
            alignment_ops,
            "_resolve_backend",
            return_value=_selection("cuda_host_io"),
        ):
            with mock.patch.object(
                alignment_ops,
                "matching_cosine_bidirectional_nearest_cuda",
                side_effect=CustomOpResourceExhaustedError("estimated VRAM"),
            ):
                with mock.patch.object(
                    alignment_ops,
                    "resolve_after_resource_exhausted",
                    return_value=_selection("openmp_cpu"),
                ) as resolve:
                    got = alignment_ops.matching_cosine_bidirectional_nearest(
                        features1, features2)

        resolve.assert_called_once()
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[2], expected[2])

    def test_cuda_admission_rejection_falls_back_to_cpu(self) -> None:
        features1, features2 = _random_features(seed=12)
        expected = alignment_ops.matching_cosine_bidirectional_nearest_numpy(
            features1, features2)
        fake_module = mock.Mock()
        fake_module.matching_cosine_bidirectional_nearest_cuda = mock.Mock(
            side_effect=AssertionError("CUDA kernel must not run after rejection")
        )

        with mock.patch.object(
            alignment_ops,
            "_resolve_backend",
            return_value=_selection("cuda_host_io"),
        ):
            with mock.patch.object(
                alignment_ops,
                "_load_compiled_module_result",
                return_value=(fake_module, None),
            ):
                with mock.patch.object(
                    cuda_memory,
                    "cuda_memory_info",
                    return_value={
                        "available": True,
                        "device": 0,
                        "free_bytes": 1,
                        "total_bytes": 1 << 30,
                    },
                ) as probe:
                    with mock.patch.object(
                        cuda_memory,
                        "_clear_current_thread_cuda_cache",
                        return_value=False,
                    ):
                        with mock.patch.object(
                            alignment_ops,
                            "resolve_after_resource_exhausted",
                            return_value=_selection("openmp_cpu"),
                        ) as resolve:
                            with mock.patch.object(
                                alignment_ops,
                                "matching_cosine_bidirectional_nearest_cpu_compiled",
                                wraps=(
                                    alignment_ops.matching_cosine_bidirectional_nearest_numpy
                                ),
                            ) as cpu_backend:
                                got = (
                                    alignment_ops.matching_cosine_bidirectional_nearest(
                                        features1, features2
                                    )
                                )

        probe.assert_called_once()
        resolve.assert_called_once()
        cpu_backend.assert_called_once()
        fake_module.matching_cosine_bidirectional_nearest_cuda.assert_not_called()
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[2], expected[2])

    def test_typed_cuda_unavailable_error_falls_back_to_cpu(self) -> None:
        module = self._compiled_module()
        features1, features2 = _random_features(seed=10)
        expected = alignment_ops.matching_cosine_bidirectional_nearest_numpy(
            features1, features2)
        unavailable_error = module.CudaRuntimeUnavailableError(
            "no CUDA-capable device is detected")

        with mock.patch.object(
            alignment_ops,
            "_resolve_backend",
            return_value=_selection("cuda_host_io"),
        ):
            with mock.patch.object(
                alignment_ops,
                "matching_cosine_bidirectional_nearest_cuda",
                side_effect=unavailable_error,
            ):
                got = alignment_ops.matching_cosine_bidirectional_nearest(
                    features1, features2)

        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[2], expected[2])

    def test_unknown_cuda_error_propagates(self) -> None:
        features1, features2 = _random_features(seed=5)

        with mock.patch.object(
            alignment_ops,
            "_resolve_backend",
            return_value=_selection("cuda_host_io"),
        ):
            with mock.patch.object(
                alignment_ops,
                "matching_cosine_bidirectional_nearest_cuda",
                side_effect=RuntimeError("invalid device pointer"),
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid device pointer"):
                    alignment_ops.matching_cosine_bidirectional_nearest(
                        features1, features2)

    def test_cpu_runtime_error_propagates(self) -> None:
        features1, features2 = _random_features(seed=11)

        with mock.patch.object(
            alignment_ops,
            "_resolve_backend",
            return_value=_selection("openmp_cpu"),
        ):
            with mock.patch.object(
                alignment_ops,
                "matching_cosine_bidirectional_nearest_cpu_compiled",
                side_effect=RuntimeError("CPU matching kernel failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "CPU matching kernel failed"):
                    alignment_ops.matching_cosine_bidirectional_nearest(
                        features1, features2)

    def test_invalid_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature dimensions must match"):
            alignment_ops.matching_cosine_bidirectional_nearest_numpy(
                np.ones((4, 3)), np.ones((5, 2)))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            alignment_ops.matching_cosine_bidirectional_nearest_numpy(
                np.ones((0, 3)), np.ones((5, 3)))

    def test_norma_default_path_routes_only_nearest_core(self) -> None:
        features1, features2 = _random_features(
            seed=6, n1=40, n2=43, feature_dim=8)
        rng = np.random.default_rng(7)
        pts1 = rng.random((40, 2)) * 1000.0
        pts2 = rng.random((43, 2)) * 1000.0

        with mock.patch.object(
            norma_matching,
            "matching_cosine_bidirectional_nearest",
            wraps=norma_matching.matching_cosine_bidirectional_nearest,
        ) as nearest:
            pair_idx = norma_matching.find_initial_match(
                features1,
                features2,
                pts1,
                pts2,
                alpha=0.0,
                apply_threshold_filter=False,
            )

        nearest.assert_called_once_with(features1, features2)
        self.assertEqual(pair_idx.ndim, 2)
        self.assertEqual(pair_idx.shape[1], 2)

    def test_norma_alpha_path_keeps_existing_scipy_algorithm(self) -> None:
        features1, features2 = _random_features(
            seed=8, n1=30, n2=33, feature_dim=7)
        rng = np.random.default_rng(9)
        pts1 = rng.random((30, 2)) * 1000.0
        pts2 = rng.random((33, 2)) * 1000.0
        alpha = 0.25
        cosine_distances = spd.cdist(features1, features2, "cosine")
        points = np.vstack((pts1, pts2))
        points_mean = np.mean(points, axis=0)
        points_range = np.max(points, axis=0) - np.min(points, axis=0)
        point_distances = spd.cdist(
            (pts1 - points_mean) / points_range,
            (pts2 - points_mean) / points_range,
            "euclidean",
        )
        distance_matrix = (
            cosine_distances * (1.0 - alpha) + point_distances * alpha
        )
        row_indices = np.argsort(distance_matrix, axis=1)[:, 0]
        col_indices = np.argsort(distance_matrix, axis=0)[0, :]
        mutual = col_indices[row_indices] == np.arange(len(features1))
        row_distances = distance_matrix[np.arange(len(features1)), row_indices]
        col_distances = distance_matrix[col_indices, np.arange(len(features2))]
        distance_threshold = min(
            np.percentile(row_distances, 30),
            np.percentile(col_distances, 30),
        )
        expected = np.stack(
            (
                np.where(mutual & (row_distances < distance_threshold))[0],
                row_indices[mutual & (row_distances < distance_threshold)],
            ),
            axis=-1,
        )

        with mock.patch.object(
            norma_matching,
            "matching_cosine_bidirectional_nearest",
            side_effect=AssertionError("custom op must not run for alpha > 0"),
        ) as nearest:
            with mock.patch.object(
                norma_matching,
                "_repair_pair_coverage",
                side_effect=lambda _pts, _mutual, _dist, pairs, *_args: pairs,
            ):
                pair_idx = norma_matching.find_initial_match(
                    features1,
                    features2,
                    pts1,
                    pts2,
                    alpha=alpha,
                    apply_threshold_filter=False,
                )

        nearest.assert_not_called()
        np.testing.assert_array_equal(pair_idx, expected)
