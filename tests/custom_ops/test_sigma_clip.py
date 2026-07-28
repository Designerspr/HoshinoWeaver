import unittest
from unittest import mock

import numpy as np

from hoshicore._custom_op import (
    build_info,
    sigma_clip_fused_masked_merge,
    sigma_clip_fused_merge,
)
import hoshicore._custom_op.backend_registry as backend_registry
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
import hoshicore._custom_op.ops.fgp as fgp_ops
import hoshicore._custom_op.ops.sigma_clip as sigma_clip_chunk_ops


from tests.custom_ops._base import CustomOpsTestCase


class TestSigmaClipCustomOps(CustomOpsTestCase):
    def test_sigma_clip_fused_merge_can_force_numpy_fallback(self) -> None:
        img = np.array([[[1], [8]], [[5], [2]]], dtype=np.uint16)
        rej_high = np.array([[[3], [6]], [[7], [4]]], dtype=np.uint16)
        rej_low = np.array([[[0], [2]], [[4], [3]]], dtype=np.uint16)
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
                sigma_clip_fused_merge(
                    img, rej_high, rej_low, sum_mu, square_sum, count
                )

        rejected = ((img < rej_low) | (img > rej_high)).astype(np.uint16)
        np.testing.assert_array_equal(sum_mu, img.astype(np.uint32) * rejected)
        np.testing.assert_array_equal(
            square_sum,
            np.square(img, dtype=np.uint64) * rejected.astype(np.uint64),
        )
        np.testing.assert_array_equal(count, rejected)

    def test_sigma_clip_fused_masked_merge_can_force_numpy_fallback(self) -> None:
        img = np.array(
            [[[1, 2], [8, 9]], [[5, 6], [2, 1]]],
            dtype=np.uint16,
        )
        mask = np.array([[True, False], [True, True]])
        rej_high = np.array(
            [[[3, 3], [6, 6]], [[7, 7], [4, 4]]],
            dtype=np.uint16,
        )
        rej_low = np.array(
            [[[0, 0], [2, 2]], [[4, 4], [3, 3]]],
            dtype=np.uint16,
        )
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
                sigma_clip_fused_masked_merge(
                    img,
                    mask,
                    rej_high,
                    rej_low,
                    sum_mu,
                    square_sum,
                    count,
                )

        rejected = (mask[..., None] & ((img < rej_low) | (img > rej_high))).astype(
            np.uint16
        )
        np.testing.assert_array_equal(sum_mu, img.astype(np.uint32) * rejected)
        np.testing.assert_array_equal(
            square_sum,
            np.square(img, dtype=np.uint64) * rejected.astype(np.uint64),
        )
        np.testing.assert_array_equal(count, rejected)

    def test_sigma_clip_chunk_rejects_outliers(self) -> None:
        """Chunk kernel correctly rejects outlier pixels."""
        np.random.seed(123)
        n_frames = 30
        plane_size = 8
        stack = np.random.randint(190, 210, (n_frames, plane_size)).astype(np.uint16)
        stack[5, 2] = 900
        stack[10, 2] = 850
        stack[15, 5] = 50

        total_sum = stack.sum(axis=0).astype(np.float64)
        total_sq = (stack.astype(np.float64) ** 2).sum(axis=0)
        total_n = np.full(plane_size, float(n_frames))

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5
        )
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk_numpy(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5
        )

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        self.assertEqual(c_n[2], 28.0)
        self.assertEqual(c_n[5], 29.0)
        for i in [0, 1, 3, 4, 6, 7]:
            self.assertEqual(c_n[i], 30.0)

    def test_sigma_clip_chunk_with_static_mask(self) -> None:
        """Chunk kernel respects a static mask (same for all frames)."""
        np.random.seed(789)
        n_frames = 20
        plane_size = 8
        stack = np.random.randint(100, 120, (n_frames, plane_size)).astype(np.uint16)
        stack[5, 3] = 900

        mask = np.ones((n_frames, plane_size), dtype=np.uint8)
        mask[:, 3] = 0
        mask[:, 7] = 0

        stack_f64 = stack.astype(np.float64)
        mask_f64 = mask.astype(np.float64)
        total_sum = (stack_f64 * mask_f64).sum(axis=0)
        total_sq = (stack_f64**2 * mask_f64).sum(axis=0)
        total_n = mask_f64.sum(axis=0)

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5, mask=mask
        )
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk_numpy(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5, mask=mask
        )

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        self.assertEqual(c_n[3], 0.0)
        self.assertEqual(c_n[7], 0.0)
        for i in [0, 1, 2, 4, 5, 6]:
            self.assertEqual(c_n[i], 20.0)

    def test_sigma_clip_chunk_with_perframe_mask(self) -> None:
        """Chunk kernel handles per-frame masks."""
        np.random.seed(321)
        n_frames = 20
        plane_size = 6
        stack = np.random.randint(95, 105, (n_frames, plane_size)).astype(np.uint16)
        stack[0, 2] = 250

        mask = np.ones((n_frames, plane_size), dtype=np.uint8)
        mask[0, 0] = 0
        mask[0, 1] = 0

        stack_f64 = stack.astype(np.float64)
        mask_f64 = mask.astype(np.float64)
        total_sum = (stack_f64 * mask_f64).sum(axis=0)
        total_sq = (stack_f64**2 * mask_f64).sum(axis=0)
        total_n = mask_f64.sum(axis=0)

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5, mask=mask
        )
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk_numpy(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5, mask=mask
        )

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        self.assertEqual(c_n[0], 19.0)
        self.assertEqual(c_n[1], 19.0)
        self.assertEqual(c_n[2], 19.0)

    def test_sigma_clip_chunk_skip_zero_rgb_matches_numpy(self) -> None:
        """Chunk kernel skips RGB all-zero pixels consistently with numpy."""
        rng = np.random.default_rng(123)
        n_frames = 12
        spatial = 17
        channels = 3
        plane_size = spatial * channels
        stack = rng.integers(90, 120, size=(n_frames, plane_size), dtype=np.uint16)
        stack[:, 6:9] = 0
        stack[4, 12:15] = 0
        stack[3, 25] = 900

        mask = (rng.random((n_frames, plane_size)) > 0.2).astype(np.uint8)
        stack_f64 = stack.astype(np.float64)
        active = mask.astype(bool)
        zero_pixels = np.all(
            stack.reshape(n_frames, spatial, channels)[..., :3] == 0,
            axis=-1,
        )
        zero_flat = np.broadcast_to(
            zero_pixels[..., None], (n_frames, spatial, channels)
        ).reshape(n_frames, plane_size)
        active &= ~zero_flat
        active_f64 = active.astype(np.float64)
        total_sum = (stack_f64 * active_f64).sum(axis=0)
        total_sq = (stack_f64**2 * active_f64).sum(axis=0)
        total_n = active_f64.sum(axis=0)

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk(
            stack,
            total_sum,
            total_sq,
            total_n,
            3.0,
            3.0,
            5,
            mask=mask,
            skip_zero_rgb=True,
            channels=channels,
        )
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk_numpy(
            stack,
            total_sum,
            total_sq,
            total_n,
            3.0,
            3.0,
            5,
            mask=mask,
            skip_zero_rgb=True,
            channels=channels,
        )

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        np.testing.assert_allclose(c_sq, n_sq, rtol=1e-10)
        np.testing.assert_array_equal(c_n[6:9], np.zeros(3))

    def test_sigma_clip_fused_chunk_with_mask(self) -> None:
        """Fused chunk kernel respects mask."""
        np.random.seed(654)
        n_frames = 25
        plane_size = 10
        stack = np.random.randint(80, 120, (n_frames, plane_size)).astype(np.uint16)
        stack[3, 4] = 250

        mask = np.ones((n_frames, plane_size), dtype=np.uint8)
        mask[:, 8] = 0

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk(
            stack, 3.0, 3.0, 5, mask=mask
        )
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(
            stack, 3.0, 3.0, 5, mask=mask
        )

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        self.assertEqual(c_n[8], 0.0)
        self.assertLess(c_n[4], 25.0)

    def test_sigma_clip_fused_chunk_can_force_numpy_fallback(self) -> None:
        """Fallback preference bypasses native fused chunk backends."""
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        expected = sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(stack)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                sigma_clip_chunk_ops,
                "sigma_clip_fused_chunk_compiled",
                side_effect=AssertionError("native backend should not be called"),
            ):
                with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "sigma_clip_fused_chunk_compiled_cuda",
                    side_effect=AssertionError("native backend should not be called"),
                ):
                    got = sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)

        for actual, expected_arr in zip(got, expected, strict=True):
            np.testing.assert_array_equal(actual, expected_arr)

    def test_sigma_clip_fused_chunk_skip_zero_rgb_matches_numpy(self) -> None:
        """Fused chunk kernel excludes RGB all-zero pixels in total stats."""
        rng = np.random.default_rng(456)
        n_frames = 14
        spatial = 19
        channels = 3
        plane_size = spatial * channels
        stack = rng.integers(80, 130, size=(n_frames, plane_size), dtype=np.uint16)
        stack[:, 9:12] = 0
        stack[2, 30:33] = 0
        stack[5, 44] = 700
        mask = (rng.random((n_frames, plane_size)) > 0.15).astype(np.uint8)

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk(
            stack, 3.0, 3.0, 5, mask=mask, skip_zero_rgb=True, channels=channels
        )
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(
            stack, 3.0, 3.0, 5, mask=mask, skip_zero_rgb=True, channels=channels
        )

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        np.testing.assert_allclose(c_sq, n_sq, rtol=1e-10)
        np.testing.assert_array_equal(c_n[9:12], np.zeros(3))

    def test_sigma_clip_fused_chunk_skip_zero_rgb_requires_full_pixels(self) -> None:
        """RGB zero-skip requires flattened data to contain complete pixels."""
        stack = np.arange(40, dtype=np.uint16).reshape(4, 10)
        with self.assertRaisesRegex(ValueError, "divisible by channels"):
            sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(
                stack, skip_zero_rgb=True, channels=3
            )
        with self.assertRaisesRegex(ValueError, "divisible by channels"):
            sigma_clip_chunk_ops.sigma_clip_fused_chunk_compiled(
                stack, skip_zero_rgb=True, channels=3
            )

    def test_sigma_clip_fused_chunk_cuda_requires_full_pixels(self) -> None:
        """CUDA wrapper validates RGB zero-skip shape before launching kernels."""
        if not build_info().get("cuda"):
            self.skipTest("CUDA sigma clip backend is not built")
        stack = np.arange(40, dtype=np.uint16).reshape(4, 10)

        with self.assertRaisesRegex(ValueError, "divisible by channels"):
            sigma_clip_chunk_ops.sigma_clip_fused_chunk_compiled_cuda(
                stack, skip_zero_rgb=True, channels=3
            )

    def test_sigma_clip_fused_chunk_cuda_matches_numpy(self) -> None:
        """CUDA fused chunk backend matches numpy when a CUDA device is available."""
        if not build_info().get("cuda"):
            self.skipTest("CUDA sigma clip backend is not built")

        rng = np.random.default_rng(987)
        n_frames = 18
        spatial = 32
        channels = 3
        plane_size = spatial * channels
        stack = rng.integers(80, 140, size=(n_frames, plane_size), dtype=np.uint16)
        stack[0, 12] = 900
        stack[:, 21:24] = 0
        mask = (rng.random((n_frames, plane_size)) > 0.2).astype(np.uint8)

        try:
            c_sum, c_sq, c_n = (
                sigma_clip_chunk_ops.sigma_clip_fused_chunk_compiled_cuda(
                    stack, 3.0, 3.0, 5, mask=mask, skip_zero_rgb=True, channels=channels
                )
            )
        except RuntimeError as exc:
            if is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA sigma clip runtime unavailable: {exc}")
            raise

        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(
            stack, 3.0, 3.0, 5, mask=mask, skip_zero_rgb=True, channels=channels
        )

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        np.testing.assert_allclose(c_sq, n_sq, rtol=1e-10)

    def test_sigma_clip_fused_chunk_cuda_runtime_falls_back_to_cpu(self) -> None:
        """Public fused dispatch falls back to CPU when CUDA runtime is unavailable."""
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        expected = tuple(
            np.full(6, value, dtype=np.float64) for value in (1.0, 2.0, 3.0)
        )
        cuda_candidate = backend_registry.BackendCandidate(
            "sigma_clip_fused_chunk",
            "cuda_host_io",
            "sigma_clip_fused_chunk_cuda",
            priority=10,
            build_flag="cuda",
        )
        cuda_selection = backend_registry.BackendSelection(
            cuda_candidate,
            mock.Mock(),
        )

        with mock.patch.object(
            sigma_clip_chunk_ops, "_resolve_backend", return_value=cuda_selection
        ):
            with mock.patch.object(
                sigma_clip_chunk_ops,
                "sigma_clip_fused_chunk_compiled_cuda",
                side_effect=RuntimeError(
                    "sigma_clip_fused_chunk_cuda cudaGetDevice: no CUDA-capable device is detected"
                ),
            ):
                cpu_fallback = mock.Mock(return_value=expected)
                with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "resolve_after_runtime_unavailable",
                    return_value=backend_registry.BackendSelection(
                        backend_registry.BackendCandidate(
                            "sigma_clip_fused_chunk",
                            "openmp_cpu",
                            "sigma_clip_fused_chunk",
                        ),
                        mock.Mock(),
                    ),
                ):
                    with mock.patch.object(
                        sigma_clip_chunk_ops,
                        "sigma_clip_fused_chunk_compiled",
                        cpu_fallback,
                    ):
                        got = sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)

        for actual, expected_arr in zip(got, expected, strict=True):
            np.testing.assert_array_equal(actual, expected_arr)
        cpu_fallback.assert_called_once()

    def test_sigma_clip_fused_chunk_cpu_backend_error_propagates(self) -> None:
        """Direct CPU backend failures are not silently converted to numpy."""
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        cpu_candidate = backend_registry.BackendCandidate(
            "sigma_clip_fused_chunk",
            "openmp_cpu",
            "sigma_clip_fused_chunk",
            priority=1,
        )
        cpu_selection = backend_registry.BackendSelection(
            cpu_candidate,
            mock.Mock(),
        )

        with mock.patch.object(
            sigma_clip_chunk_ops, "_resolve_backend", return_value=cpu_selection
        ):
            with mock.patch.object(
                sigma_clip_chunk_ops,
                "sigma_clip_fused_chunk_compiled",
                side_effect=RuntimeError("native CPU bug"),
            ):
                with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "sigma_clip_fused_chunk_numpy",
                    side_effect=AssertionError("numpy backend should not be called"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "native CPU bug"):
                        sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)

    def test_sigma_clip_fused_chunk_cuda_runtime_falls_back_to_numpy(self) -> None:
        """Public fused dispatch falls back to numpy if CUDA and CPU are unavailable."""
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        expected = sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(stack)
        cuda_candidate = backend_registry.BackendCandidate(
            "sigma_clip_fused_chunk",
            "cuda_host_io",
            "sigma_clip_fused_chunk_cuda",
            priority=10,
            build_flag="cuda",
        )
        cuda_selection = backend_registry.BackendSelection(
            cuda_candidate,
            mock.Mock(),
        )

        with mock.patch.object(
            sigma_clip_chunk_ops, "_resolve_backend", return_value=cuda_selection
        ):
            with mock.patch.object(
                sigma_clip_chunk_ops,
                "sigma_clip_fused_chunk_compiled_cuda",
                side_effect=RuntimeError(
                    "sigma_clip_fused_chunk_cuda cudaGetDevice: no CUDA-capable device is detected"
                ),
            ):
                with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "resolve_after_runtime_unavailable",
                    return_value=backend_registry.BackendSelection(
                        None,
                        mock.Mock(),
                        "mock CPU backend unavailable",
                    ),
                ):
                    got = sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)

        for actual, expected_arr in zip(got, expected, strict=True):
            np.testing.assert_array_equal(actual, expected_arr)

    def test_sigma_clip_fused_chunk_cuda_fallback_propagates_cpu_value_error(
        self,
    ) -> None:
        """CUDA runtime fallback does not hide CPU validation failures."""
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        cuda_candidate = backend_registry.BackendCandidate(
            "sigma_clip_fused_chunk",
            "cuda_host_io",
            "sigma_clip_fused_chunk_cuda",
            priority=10,
            build_flag="cuda",
        )
        cuda_selection = backend_registry.BackendSelection(
            cuda_candidate,
            mock.Mock(),
        )

        with mock.patch.object(
            sigma_clip_chunk_ops, "_resolve_backend", return_value=cuda_selection
        ):
            with mock.patch.object(
                sigma_clip_chunk_ops,
                "sigma_clip_fused_chunk_compiled_cuda",
                side_effect=RuntimeError(
                    "sigma_clip_fused_chunk_cuda cudaGetDevice: no CUDA-capable device is detected"
                ),
            ):
                cpu_fallback = mock.Mock(
                    side_effect=ValueError("bad CPU fallback input")
                )
                with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "resolve_after_runtime_unavailable",
                    return_value=backend_registry.BackendSelection(
                        backend_registry.BackendCandidate(
                            "sigma_clip_fused_chunk",
                            "openmp_cpu",
                            "sigma_clip_fused_chunk",
                        ),
                        mock.Mock(),
                    ),
                ):
                    with mock.patch.object(
                        sigma_clip_chunk_ops,
                        "sigma_clip_fused_chunk_compiled",
                        cpu_fallback,
                    ):
                        with mock.patch.object(
                            sigma_clip_chunk_ops,
                            "sigma_clip_fused_chunk_numpy",
                            side_effect=AssertionError(
                                "numpy backend should not be called"
                            ),
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "bad CPU fallback input"
                            ):
                                sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)

    def test_sigma_clip_fused_chunk_cuda_fallback_propagates_cpu_runtime_error(
        self,
    ) -> None:
        """CUDA runtime fallback does not hide CPU kernel RuntimeError failures."""
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        cuda_candidate = backend_registry.BackendCandidate(
            "sigma_clip_fused_chunk",
            "cuda_host_io",
            "sigma_clip_fused_chunk_cuda",
            priority=10,
            build_flag="cuda",
        )
        cuda_selection = backend_registry.BackendSelection(
            cuda_candidate,
            mock.Mock(),
        )

        with mock.patch.object(
            sigma_clip_chunk_ops, "_resolve_backend", return_value=cuda_selection
        ):
            with mock.patch.object(
                sigma_clip_chunk_ops,
                "sigma_clip_fused_chunk_compiled_cuda",
                side_effect=RuntimeError(
                    "sigma_clip_fused_chunk_cuda cudaGetDevice: no CUDA-capable device is detected"
                ),
            ):
                cpu_fallback = mock.Mock(side_effect=RuntimeError("native CPU bug"))
                with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "resolve_after_runtime_unavailable",
                    return_value=backend_registry.BackendSelection(
                        backend_registry.BackendCandidate(
                            "sigma_clip_fused_chunk",
                            "openmp_cpu",
                            "sigma_clip_fused_chunk",
                        ),
                        mock.Mock(),
                    ),
                ):
                    with mock.patch.object(
                        sigma_clip_chunk_ops,
                        "sigma_clip_fused_chunk_compiled",
                        cpu_fallback,
                    ):
                        with mock.patch.object(
                            sigma_clip_chunk_ops,
                            "sigma_clip_fused_chunk_numpy",
                            side_effect=AssertionError(
                                "numpy backend should not be called"
                            ),
                        ):
                            with self.assertRaisesRegex(RuntimeError, "native CPU bug"):
                                sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)

    def test_sigma_clip_fused_chunk_does_not_cache_runtime_decision(self) -> None:
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        cpu_selection = backend_registry.BackendSelection(
            backend_registry.BackendCandidate(
                "sigma_clip_fused_chunk",
                "openmp_cpu",
                "sigma_clip_fused_chunk",
            ),
            mock.Mock(),
        )
        cuda_selection = backend_registry.BackendSelection(
            backend_registry.BackendCandidate(
                "sigma_clip_fused_chunk",
                "cuda_host_io",
                "sigma_clip_fused_chunk_cuda",
            ),
            mock.Mock(),
        )
        cpu_result = tuple(
            np.full(6, value, dtype=np.float64) for value in (1.0, 2.0, 3.0)
        )
        cuda_result = tuple(
            np.full(6, value, dtype=np.float64) for value in (4.0, 5.0, 6.0)
        )

        with mock.patch.object(
            sigma_clip_chunk_ops,
            "_resolve_backend",
            side_effect=[cpu_selection, cuda_selection],
        ) as resolver:
            with mock.patch.object(
                sigma_clip_chunk_ops,
                "sigma_clip_fused_chunk_compiled",
                return_value=cpu_result,
            ):
                with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "sigma_clip_fused_chunk_compiled_cuda",
                    return_value=cuda_result,
                ) as cuda_backend:
                    first = sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)
                    second = sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)

        self.assertEqual(resolver.call_count, 2)
        cuda_backend.assert_called_once()
        np.testing.assert_array_equal(first[0], cpu_result[0])
        np.testing.assert_array_equal(second[0], cuda_result[0])

    def test_sigma_clip_fused_chunk_unstructured_oom_error_propagates(self) -> None:
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        cuda_selection = backend_registry.BackendSelection(
            backend_registry.BackendCandidate(
                "sigma_clip_fused_chunk",
                "cuda_host_io",
                "sigma_clip_fused_chunk_cuda",
            ),
            mock.Mock(),
        )

        with mock.patch.object(
            sigma_clip_chunk_ops,
            "_resolve_backend",
            return_value=cuda_selection,
        ):
            with mock.patch.object(
                sigma_clip_chunk_ops,
                "sigma_clip_fused_chunk_compiled_cuda",
                side_effect=RuntimeError("cudaMalloc: out of memory"),
            ):
                with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "sigma_clip_fused_chunk_numpy",
                    side_effect=AssertionError("numpy backend should not run"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "out of memory"):
                        sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)

    def test_sigma_clip_fused_chunk_typed_resource_error_falls_back_to_cpu(
        self,
    ) -> None:
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        cuda_selection = backend_registry.BackendSelection(
            backend_registry.BackendCandidate(
                "sigma_clip_fused_chunk",
                "cuda_host_io",
                "sigma_clip_fused_chunk_cuda",
            ),
            mock.Mock(),
        )
        cpu_selection = backend_registry.BackendSelection(
            backend_registry.BackendCandidate(
                "sigma_clip_fused_chunk",
                "openmp_cpu",
                "sigma_clip_fused_chunk",
            ),
            mock.Mock(),
        )
        expected = (
            np.full(6, 10.0, dtype=np.float64),
            np.full(6, 20.0, dtype=np.float64),
            np.full(6, 4, dtype=np.uint32),
        )

        with mock.patch.object(
            sigma_clip_chunk_ops,
            "_resolve_backend",
            return_value=cuda_selection,
        ):
            with mock.patch.object(
                sigma_clip_chunk_ops,
                "sigma_clip_fused_chunk_compiled_cuda",
                side_effect=CustomOpResourceExhaustedError(
                    "estimated VRAM is insufficient"
                ),
            ):
                with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "resolve_after_resource_exhausted",
                    return_value=cpu_selection,
                ) as resource_resolver:
                    with mock.patch.object(
                        sigma_clip_chunk_ops,
                        "sigma_clip_fused_chunk_compiled",
                        return_value=expected,
                    ) as cpu_backend:
                        result = sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)

        self.assertIs(result, expected)
        resource_resolver.assert_called_once()
        cpu_backend.assert_called_once()

    @staticmethod
    def _compiled_module():
        module, error = sigma_clip_chunk_ops._load_compiled_module_result()
        if module is None:
            raise unittest.SkipTest(error or "compiled custom ops unavailable")
        return module

    def test_sigma_clip_fused_chunk_native_rejects_partial_rgb_pixels(self) -> None:
        """Native binding validates RGB zero-skip shape without the wrapper."""
        module = self._compiled_module()
        stack = np.arange(40, dtype=np.uint16).reshape(4, 10)

        with self.assertRaisesRegex(ValueError, "divisible by channels"):
            module.sigma_clip_fused_chunk(
                stack, 3.0, 3.0, 5, None, True, 3
            )

    def test_sigma_clip_iterative_chunk_native_rejects_partial_rgb_pixels(self) -> None:
        """Native iterative binding validates RGB zero-skip shape too."""
        module = self._compiled_module()
        plane_size = 10
        stack = np.arange(4 * plane_size, dtype=np.uint16).reshape(4, plane_size)
        total_sum = np.zeros(plane_size, dtype=np.float64)
        total_sq = np.zeros(plane_size, dtype=np.float64)
        total_n = np.zeros(plane_size, dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "divisible by channels"):
            module.sigma_clip_iterative_chunk(
                stack, total_sum, total_sq, total_n, 3.0, 3.0, 5, None, True, 3
            )
