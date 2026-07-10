from unittest import mock

import numpy as np

from hoshicore._custom_op import (
    build_info,
    fgp_add,
    fgp_accumulate,
    fgp_masked_mean_merge,
    huber_weighted_chunk,
    huber_weighted_accumulate,
)
import hoshicore._custom_op.backend_registry as backend_registry
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
import hoshicore._custom_op.ops.fgp as fgp_ops
from hoshicore.component.data_container import FastGaussianParam, HuberMeanParam
from hoshicore.component.merger import HuberWeightedMerger
from hoshicore.component.merger import MeanMerger
import hoshicore.ops.sigma_clip_ops as sigma_clip_ops


from tests.custom_ops._base import CustomOpsTestCase


class TestFgpHuberCustomOps(CustomOpsTestCase):
    def test_fgp_accumulate_matches_python_unweighted(self) -> None:
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        got = fgp_accumulate(base, fresh)
        expected = expected + FastGaussianParam(fresh, source_dtype=fresh.dtype)

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_fgp_accumulate_matches_python_weighted_int(self) -> None:
        weight = 3
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        got = fgp_accumulate(base, fresh, weight)
        patch = FastGaussianParam(fresh, source_dtype=fresh.dtype)
        patch = patch * weight
        expected = expected + patch

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_fgp_accumulate_can_force_numpy_fallback(self) -> None:
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                fgp_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                fgp_ops._select_fgp_backend.cache_clear()
                got = fgp_accumulate(base, fresh)

        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = expected + FastGaussianParam(fresh, source_dtype=fresh.dtype)

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_fgp_add_matches_python(self) -> None:
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        other = FastGaussianParam(
            np.array([[2, 4], [7, 1]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = expected + other

        got = fgp_add(base, other)

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_fgp_add_can_force_numpy_fallback(self) -> None:
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        other = FastGaussianParam(
            np.array([[2, 4], [7, 1]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                fgp_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                fgp_ops._select_fgp_add_backend.cache_clear()
                got = fgp_add(base, other)

        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = expected + other

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_huber_weighted_accumulate_matches_numpy_scalar_weight(self) -> None:
        img = np.array([[[4], [9]], [[6], [3]]], dtype=np.uint16)
        ref_mean = np.array([[[5], [7]], [[4], [2]]], dtype=np.float32)
        ref_std = np.array([[[2], [1]], [[3], [2]]], dtype=np.float32)
        base = HuberMeanParam(
            weighted_sum=np.zeros_like(img, dtype=np.float64),
            weight_total=np.zeros_like(img, dtype=np.float64),
            source_dtype=img.dtype,
        )

        got = huber_weighted_accumulate(base, img, ref_mean, ref_std, 1.345, weight=0.5)

        residual = (img.astype(np.float32) - ref_mean) / (ref_std + 1e-10)
        abs_residual = np.abs(residual)
        huber_weight = (
            np.where(
                abs_residual <= 1.345,
                np.ones_like(abs_residual, dtype=np.float32),
                (1.345 / (abs_residual + 1e-10)).astype(np.float32),
            )
            * 0.5
        )
        expected_sum = (img * huber_weight).astype(np.float64)
        expected_total = huber_weight.astype(np.float64)

        self.assertIs(got, base)
        np.testing.assert_allclose(
            base.weighted_sum, expected_sum, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            base.weight_total, expected_total, rtol=1e-6, atol=1e-6
        )

    def test_huber_weighted_accumulate_keeps_array_weight_on_numpy_path(self) -> None:
        img = np.array([[[4], [9]], [[6], [3]]], dtype=np.uint16)
        ref_mean = np.array([[[5], [7]], [[4], [2]]], dtype=np.float32)
        ref_std = np.array([[[2], [1]], [[3], [2]]], dtype=np.float32)
        frame_weight = np.array([[[1.0], [0.25]], [[0.5], [0.75]]], dtype=np.float32)
        base = HuberMeanParam(
            weighted_sum=np.zeros_like(img, dtype=np.float64),
            weight_total=np.zeros_like(img, dtype=np.float64),
            source_dtype=img.dtype,
        )

        got = huber_weighted_accumulate(
            base, img, ref_mean, ref_std, 1.345, weight=frame_weight
        )

        residual = (img.astype(np.float32) - ref_mean) / (ref_std + 1e-10)
        abs_residual = np.abs(residual)
        huber_weight = (
            np.where(
                abs_residual <= 1.345,
                np.ones_like(abs_residual, dtype=np.float32),
                (1.345 / (abs_residual + 1e-10)).astype(np.float32),
            )
            * frame_weight
        )
        expected_sum = (img * huber_weight).astype(np.float64)
        expected_total = huber_weight.astype(np.float64)

        self.assertIs(got, base)
        np.testing.assert_allclose(
            base.weighted_sum, expected_sum, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            base.weight_total, expected_total, rtol=1e-6, atol=1e-6
        )

    def test_huber_weighted_chunk_matches_numpy(self) -> None:
        stack = np.array(
            [
                [4, 9, 6, 3],
                [5, 6, 7, 1],
                [8, 2, 5, 4],
            ],
            dtype=np.uint16,
        )
        ref_mean = np.array([5, 7, 4, 2], dtype=np.float64)
        ref_std = np.array([2, 1, 3, 2], dtype=np.float64)
        weights = np.array([0.5, 0.25, 0.75], dtype=np.float64)

        got_sum, got_total = huber_weighted_chunk(
            stack, ref_mean, ref_std, 1.345, weights
        )
        expected_sum, expected_total = fgp_ops.huber_weighted_chunk_numpy(
            stack, ref_mean, ref_std, 1.345, weights
        )

        np.testing.assert_allclose(got_sum, expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got_total, expected_total, rtol=1e-6, atol=1e-6)

    def test_huber_weighted_chunk_can_force_numpy_fallback(self) -> None:
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        ref_mean = np.linspace(2, 8, 6, dtype=np.float64)
        ref_std = np.linspace(1, 3, 6, dtype=np.float64)
        expected = fgp_ops.huber_weighted_chunk_numpy(stack, ref_mean, ref_std, 1.345)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                fgp_ops,
                "huber_weighted_chunk_compiled_cuda",
                side_effect=AssertionError("native backend should not be called"),
            ):
                got = huber_weighted_chunk(stack, ref_mean, ref_std, 1.345)

        for actual, expected_arr in zip(got, expected, strict=True):
            np.testing.assert_allclose(actual, expected_arr, rtol=1e-6, atol=1e-6)

    def test_huber_weighted_chunk_preserves_double_weight_precision(self) -> None:
        stack = np.array([[0], [1]], dtype=np.uint16)
        ref_mean = np.array([0.0], dtype=np.float64)
        ref_std = np.array([1.0], dtype=np.float64)
        weights = np.array([1.0, 1.0 + 1e-8], dtype=np.float64)

        weighted_sum, weight_total = fgp_ops.huber_weighted_chunk_numpy(
            stack, ref_mean, ref_std, 10.0, weights
        )

        np.testing.assert_allclose(weighted_sum, np.array([1.0 + 1e-8]))
        np.testing.assert_allclose(weight_total, np.array([2.0 + 1e-8]))
        np.testing.assert_array_equal(
            np.round(weighted_sum / weight_total),
            np.array([1.0]),
        )

    def test_huber_weighted_chunk_cuda_matches_numpy(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA Huber chunk backend is not built")

        rng = np.random.default_rng(2468)
        for dtype, high in ((np.uint8, 220), (np.uint16, 60000)):
            for weighted in (False, True):
                with self.subTest(dtype=np.dtype(dtype).name, weighted=weighted):
                    stack = rng.integers(10, high, size=(12, 257), dtype=dtype)
                    ref_mean = np.mean(stack.astype(np.float64), axis=0)
                    ref_std = np.std(stack.astype(np.float64), axis=0, ddof=1)
                    weights = (
                        rng.random(stack.shape[0]).astype(np.float64)
                        if weighted
                        else None
                    )

                    try:
                        c_sum, c_total = fgp_ops.huber_weighted_chunk_compiled_cuda(
                            stack, ref_mean, ref_std, 1.345, weights
                        )
                    except RuntimeError as exc:
                        if is_cuda_runtime_unavailable_error(exc):
                            self.skipTest(
                                f"CUDA Huber chunk runtime unavailable: {exc}"
                            )
                        raise

                    n_sum, n_total = fgp_ops.huber_weighted_chunk_numpy(
                        stack, ref_mean, ref_std, 1.345, weights
                    )
                    np.testing.assert_allclose(c_sum, n_sum, rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(c_total, n_total, rtol=1e-6, atol=1e-6)

    def test_huber_weighted_chunk_cuda_runtime_falls_back_to_numpy(self) -> None:
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        ref_mean = np.linspace(2, 8, 6, dtype=np.float64)
        ref_std = np.linspace(1, 3, 6, dtype=np.float64)
        expected = fgp_ops.huber_weighted_chunk_numpy(stack, ref_mean, ref_std, 1.345)
        cuda_candidate = backend_registry.BackendCandidate(
            "huber_weighted_chunk",
            "cuda_host_io",
            "huber_weighted_chunk_cuda",
            priority=10,
            build_flag="cuda",
        )
        cuda_selection = backend_registry.BackendSelection(
            cuda_candidate,
            mock.Mock(),
        )

        fgp_ops._select_huber_chunk_backend.cache_clear()
        with mock.patch.object(fgp_ops, "_select_backend", return_value=cuda_selection):
            with mock.patch.object(
                fgp_ops,
                "huber_weighted_chunk_compiled_cuda",
                side_effect=RuntimeError(
                    "huber_weighted_chunk_cuda cudaGetDevice: no CUDA-capable device is detected"
                ),
            ):
                fgp_ops._select_huber_chunk_backend.cache_clear()
                got = huber_weighted_chunk(stack, ref_mean, ref_std, 1.345)
        fgp_ops._select_huber_chunk_backend.cache_clear()

        for actual, expected_arr in zip(got, expected, strict=True):
            np.testing.assert_allclose(actual, expected_arr, rtol=1e-6, atol=1e-6)

    def test_mean_merger_works_with_fgp_accumulate(self) -> None:
        merger = MeanMerger(int_weight=False)
        merger.merge(np.array([[1, 9], [2, 3]], dtype=np.uint16), None)
        merger.merge(np.array([[5, 1], [2, 8]], dtype=np.uint16), None)

        expected = FastGaussianParam(
            np.array([[1, 9], [2, 3]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = expected + FastGaussianParam(
            np.array([[5, 1], [2, 8]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        np.testing.assert_array_equal(merger.result.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(merger.result.square_sum, expected.square_sum)
        np.testing.assert_array_equal(merger.result.n, expected.n)

    def test_huber_weighted_merger_routes_through_custom_accumulate(self) -> None:
        ref_stats = FastGaussianParam(
            np.array([[5, 7], [4, 2]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        merger = HuberWeightedMerger(ref_stats=ref_stats, huber_c=1.345)
        frame_a = np.array([[4, 9], [6, 3]], dtype=np.uint16)
        frame_b = np.array([[5, 6], [7, 1]], dtype=np.uint16)

        with mock.patch(
            "hoshicore.component.merger.custom_huber_weighted_accumulate",
            wraps=fgp_ops.huber_weighted_accumulate_numpy,
        ) as patched_huber:
            merger.merge(frame_a, 0.5)
            merger.merge(frame_b, 0.25)

        self.assertEqual(patched_huber.call_count, 2)

        ref_mean = ref_stats.mu.astype(np.float32)
        ref_std = np.sqrt(np.maximum(ref_stats.var, 0)).astype(np.float32)
        expected_sum = np.zeros_like(frame_a, dtype=np.float64)
        expected_total = np.zeros_like(frame_a, dtype=np.float64)
        for frame, weight in ((frame_a, 0.5), (frame_b, 0.25)):
            residual = (frame.astype(np.float32) - ref_mean) / (ref_std + 1e-10)
            abs_residual = np.abs(residual)
            huber_weight = (
                np.where(
                    abs_residual <= 1.345,
                    np.ones_like(abs_residual, dtype=np.float32),
                    (1.345 / (abs_residual + 1e-10)).astype(np.float32),
                )
                * weight
            )
            expected_sum += (frame * huber_weight).astype(np.float64)
            expected_total += huber_weight.astype(np.float64)

        np.testing.assert_allclose(
            merger.result.weighted_sum, expected_sum, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            merger.result.weight_total, expected_total, rtol=1e-6, atol=1e-6
        )

    def test_huber_mean_iterator_falls_back_when_cuda_chunk_unavailable(self) -> None:
        frames = [
            (np.array([[4, 9], [6, 3]], dtype=np.uint16), 0.5),
            (np.array([[5, 6], [7, 1]], dtype=np.uint16), 0.25),
        ]
        fgp_total = FastGaussianParam(frames[0][0], source_dtype=np.dtype("uint16"))
        fgp_total = fgp_total + FastGaussianParam(
            frames[1][0], source_dtype=np.dtype("uint16")
        )
        op = sigma_clip_ops.HuberMeanIteratorOp("huber")
        state = op._init_chunk_state(
            {"fgp_total": fgp_total, "huber_c": 1.345}, 0, 2, 2
        )
        op._configs = {"huber_c": 1.345}

        with mock.patch.object(
            sigma_clip_ops, "custom_huber_weighted_chunk_available", return_value=True
        ):
            with mock.patch.object(
                sigma_clip_ops, "custom_huber_weighted_chunk_or_none", return_value=None
            ):
                op._run_pass(state, frames)

        expected_merger = HuberWeightedMerger(ref_stats=fgp_total, huber_c=1.345)
        for frame, weight in frames:
            expected_merger.merge(frame, weight)
        np.testing.assert_allclose(
            op._finalize_chunk(state),
            expected_merger.merged_image.data,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_huber_mean_iterator_uses_fused_chunk_when_available(self) -> None:
        frames = [
            (np.array([[4, 9], [6, 3]], dtype=np.uint16), None),
            (np.array([[5, 6], [7, 1]], dtype=np.uint16), None),
        ]
        fgp_total = FastGaussianParam(frames[0][0], source_dtype=np.dtype("uint16"))
        fgp_total = fgp_total + FastGaussianParam(
            frames[1][0], source_dtype=np.dtype("uint16")
        )
        op = sigma_clip_ops.HuberMeanIteratorOp("huber")
        state = op._init_chunk_state(
            {"fgp_total": fgp_total, "huber_c": 1.345}, 0, 2, 2
        )
        op._configs = {"huber_c": 1.345}
        weighted_sum = np.array([1.0, 7.0, 16.0, 30.0], dtype=np.float64)
        weight_total = np.array([2.0, 2.0, 3.0, 4.0], dtype=np.float64)

        with mock.patch.object(
            sigma_clip_ops, "custom_huber_weighted_chunk_available", return_value=True
        ):
            with mock.patch.object(
                sigma_clip_ops,
                "custom_huber_weighted_chunk_or_none",
                return_value=(weighted_sum, weight_total),
            ) as fused:
                with mock.patch.object(state["merger"], "merge") as merge:
                    op._run_pass(state, frames)

        fused.assert_called_once()
        merge.assert_not_called()
        expected = np.round(weighted_sum / weight_total).reshape(2, 2)
        np.testing.assert_allclose(op._finalize_chunk(state), expected)

    def test_fgp_masked_mean_merge_can_force_numpy_fallback(self) -> None:
        img = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            dtype=np.uint16,
        )
        mask = np.array([[True, False], [True, True]])
        sum_mu = np.zeros_like(img, dtype=np.uint32)
        square_sum = np.zeros_like(img, dtype=np.uint64)
        count = np.zeros_like(img, dtype=np.uint16)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                fgp_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                fgp_masked_mean_merge(img, mask, sum_mu, square_sum, count)

        expected_mask = np.broadcast_to(mask[..., None], img.shape).astype(np.uint16)
        np.testing.assert_array_equal(sum_mu, img.astype(np.uint32) * expected_mask)
        np.testing.assert_array_equal(
            square_sum,
            np.square(img, dtype=np.uint64) * expected_mask.astype(np.uint64),
        )
        np.testing.assert_array_equal(count, expected_mask)
