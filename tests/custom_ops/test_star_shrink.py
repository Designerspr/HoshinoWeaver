from unittest import mock

import numpy as np

from hoshicore._custom_op import (
    star_mask_dog,
    star_shrink_detect_mask,
    star_shrink_dog_process,
    star_shrink_process,
)
import hoshicore._custom_op.backend_registry as backend_registry
from hoshicore._custom_op._dispatch import cuda_memory_info
import hoshicore._custom_op.ops.star_shrink as star_shrink_ops


from tests.custom_ops._base import CustomOpsTestCase


class TestStarShrinkCustomOps(CustomOpsTestCase):
    def test_star_shrink_process_compiled_matches_numpy_uint16(self) -> None:
        rng = np.random.default_rng(2468)
        base = rng.normal(loc=12000.0, scale=1200.0, size=(16, 18)).clip(0, 50000)
        for y, x, value in [(5, 6, 42000), (8, 10, 36000), (11, 13, 39000)]:
            base[y, x] = value
            base[y - 1 : y + 2, x - 1 : x + 2] = np.maximum(
                base[y - 1 : y + 2, x - 1 : x + 2],
                value * 0.45,
            )
        image = (
            np.stack(
                [
                    base * 0.96,
                    base,
                    base * 1.04,
                ],
                axis=2,
            )
            .clip(0, 65535)
            .astype(np.uint16)
        )
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[3:13, 4:15] = 1

        got = star_shrink_ops.star_shrink_process_compiled(
            image,
            mask,
            3,
            "CIRCLE",
            1,
            1.0,
            7,
        )
        expected = star_shrink_ops.star_shrink_process_numpy(
            image,
            mask,
            3,
            "CIRCLE",
            1,
            1.0,
            7,
        )

        self.assertEqual(got.dtype, np.uint16)
        self.assertEqual(got.shape, image.shape)
        # OpenCV's float Lab conversion and the native standard-Lab implementation
        # are not bit-exact; keep the 16-bit tolerance below 0.2% of full scale.
        np.testing.assert_allclose(got, expected, rtol=0, atol=128)
        np.testing.assert_array_equal(got[mask == 0], image[mask == 0])

    def test_star_shrink_process_compiled_matches_numpy_uint8_edges(self) -> None:
        image = np.array(
            [
                [[8, 9, 10], [20, 22, 24], [40, 42, 44], [60, 62, 64], [80, 82, 84]],
                [[12, 13, 14], [45, 48, 51], [90, 94, 98], [70, 72, 74], [35, 37, 39]],
                [
                    [16, 17, 18],
                    [55, 58, 61],
                    [140, 145, 150],
                    [85, 88, 91],
                    [45, 47, 49],
                ],
                [[22, 23, 24], [40, 42, 44], [75, 78, 81], [65, 68, 71], [30, 32, 34]],
                [[28, 29, 30], [35, 37, 39], [50, 52, 54], [45, 47, 49], [25, 27, 29]],
            ],
            dtype=np.uint8,
        )
        mask = np.ones(image.shape[:2], dtype=np.uint8)

        got = star_shrink_ops.star_shrink_process_compiled(
            image,
            mask,
            3,
            "CIRCLE",
            1,
            1.0,
            3,
        )
        expected = star_shrink_ops.star_shrink_process_numpy(
            image,
            mask,
            3,
            "CIRCLE",
            1,
            1.0,
            3,
        )

        self.assertEqual(got.dtype, np.uint8)
        np.testing.assert_allclose(got, expected, rtol=0, atol=2)

    def test_star_shrink_process_float32_uses_numpy_semantics(self) -> None:
        image = np.linspace(0.0, 2.0, 9 * 11, dtype=np.float32).reshape(9, 11)
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[2:8, 3:10] = 1

        got = star_shrink_process(image, mask, 3, "RECT", 2, 0.5, 5)
        expected = star_shrink_ops.star_shrink_process_numpy(
            image,
            mask,
            3,
            "RECT",
            2,
            0.5,
            5,
        )

        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(got[mask == 0], image[mask == 0])

        with self.assertRaises(ValueError):
            star_shrink_ops.star_shrink_process_compiled(
                image,
                mask,
                3,
                "RECT",
                2,
                0.5,
                5,
            )

    def test_star_shrink_process_can_force_numpy_fallback(self) -> None:
        image = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[1:5, 2:6] = 1

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_shrink_process_compiled",
                side_effect=AssertionError("native backend should not be called"),
            ):
                star_shrink_ops._select_star_shrink_process_backend.cache_clear()
                got = star_shrink_process(image, mask, 3, "CROSS", 1, 1.0, 5)

        expected = star_shrink_ops.star_shrink_process_numpy(
            image,
            mask,
            3,
            "CROSS",
            1,
            1.0,
            5,
        )
        np.testing.assert_array_equal(got, expected)

    def test_star_shrink_process_cuda_runtime_falls_back_to_cpu(self) -> None:
        image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        candidate = backend_registry.BackendCandidate(
            "star_shrink_process",
            "cuda_host_io",
            "star_shrink_process_cuda",
            priority=10,
            build_flag="cuda",
        )
        selection = backend_registry.BackendSelection(candidate, object(), None)

        with mock.patch.object(
            star_shrink_ops,
            "_select_star_shrink_process_backend",
            return_value=selection,
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_shrink_process_compiled_cuda",
                side_effect=RuntimeError("no CUDA-capable device is detected"),
            ) as mock_cuda:
                with mock.patch.object(
                    star_shrink_ops,
                    "_compiled_cpu_available",
                    return_value=True,
                ):
                    with mock.patch.object(
                        star_shrink_ops,
                        "star_shrink_process_compiled",
                        wraps=star_shrink_ops.star_shrink_process_numpy,
                    ) as mock_cpu:
                        got = star_shrink_process(image, mask, 3, "CIRCLE", 1, 1.0, 5)

        mock_cuda.assert_called_once()
        mock_cpu.assert_called_once()
        expected = star_shrink_ops.star_shrink_process_numpy(
            image,
            mask,
            3,
            "CIRCLE",
            1,
            1.0,
            5,
        )
        np.testing.assert_array_equal(got, expected)

    def test_star_shrink_process_cpu_backend_error_propagates(self) -> None:
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        candidate = backend_registry.BackendCandidate(
            "star_shrink_process",
            "openmp_cpu",
            "star_shrink_process",
        )
        selection = backend_registry.BackendSelection(candidate, object(), None)

        with mock.patch.object(
            star_shrink_ops,
            "_select_star_shrink_process_backend",
            return_value=selection,
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_shrink_process_compiled",
                side_effect=ValueError("bad star shrink params"),
            ):
                with self.assertRaisesRegex(ValueError, "bad star shrink params"):
                    star_shrink_process(image, mask, 3, "CIRCLE", 1, 1.0, 5)

    def test_star_shrink_process_cuda_matches_compiled_when_available(self) -> None:
        info = cuda_memory_info()
        if not info.get("available"):
            self.skipTest(f"CUDA runtime unavailable: {info.get('reason', 'unknown')}")

        rng = np.random.default_rng(9753)
        image = rng.integers(100, 30000, size=(24, 28, 3), dtype=np.uint16)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[4:20, 5:24] = 1

        got = star_shrink_ops.star_shrink_process_compiled_cuda(
            image,
            mask,
            3,
            "CIRCLE",
            1,
            1.0,
            7,
        )
        expected = star_shrink_ops.star_shrink_process_compiled(
            image,
            mask,
            3,
            "CIRCLE",
            1,
            1.0,
            7,
        )

        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_star_shrink_detect_mask_compiled_matches_numpy_uint16_2d(self) -> None:
        image = np.full((15, 17), 1000, dtype=np.uint16)
        image[5, 6] = 16000
        image[6, 7] = 18000
        image[10, 12] = 22000

        got = star_shrink_ops.star_shrink_detect_mask_compiled(
            image,
            ksize=5,
            threshold_ratio=2.0,
            open_ksize=0,
            dilate_ksize=0,
        )
        expected = star_shrink_ops.star_shrink_detect_mask_numpy(
            image,
            ksize=5,
            threshold_ratio=2.0,
            open_ksize=0,
            dilate_ksize=0,
        )

        self.assertEqual(got.dtype, np.uint8)
        np.testing.assert_array_equal(got, expected)

    def test_star_shrink_detect_mask_compiled_matches_numpy_rgb_morphology(
        self,
    ) -> None:
        gray = np.full((17, 19), 40, dtype=np.uint8)
        gray[7:10, 8:11] = 255
        gray[2, 3] = 240
        image = np.repeat(gray[..., np.newaxis], 3, axis=2)

        got = star_shrink_ops.star_shrink_detect_mask_compiled(
            image,
            ksize=3,
            threshold_ratio=1.0,
            open_ksize=3,
            dilate_ksize=3,
        )
        expected = star_shrink_ops.star_shrink_detect_mask_numpy(
            image,
            ksize=3,
            threshold_ratio=1.0,
            open_ksize=3,
            dilate_ksize=3,
        )

        self.assertEqual(got.dtype, np.uint8)
        np.testing.assert_array_equal(got, expected)

    def test_star_shrink_detect_mask_can_force_numpy_fallback(self) -> None:
        image = np.arange(9 * 11, dtype=np.uint8).reshape(9, 11)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_shrink_detect_mask_compiled",
                side_effect=AssertionError("native backend should not be called"),
            ):
                star_shrink_ops._select_star_shrink_detect_mask_backend.cache_clear()
                got = star_shrink_detect_mask(
                    image,
                    ksize=3,
                    threshold_ratio=1.5,
                    open_ksize=0,
                    dilate_ksize=0,
                )

        expected = star_shrink_ops.star_shrink_detect_mask_numpy(
            image,
            ksize=3,
            threshold_ratio=1.5,
            open_ksize=0,
            dilate_ksize=0,
        )
        np.testing.assert_array_equal(got, expected)

    def test_star_shrink_detect_mask_cpu_backend_error_propagates(self) -> None:
        image = np.arange(6 * 7, dtype=np.uint8).reshape(6, 7)
        candidate = backend_registry.BackendCandidate(
            "star_shrink_detect_mask",
            "openmp_cpu",
            "star_shrink_detect_mask",
        )
        selection = backend_registry.BackendSelection(candidate, object(), None)

        with mock.patch.object(
            star_shrink_ops,
            "_select_star_shrink_detect_mask_backend",
            return_value=selection,
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_shrink_detect_mask_compiled",
                side_effect=ValueError("bad detect params"),
            ):
                with self.assertRaisesRegex(ValueError, "bad detect params"):
                    star_shrink_detect_mask(image, ksize=3)

    def test_star_mask_dog_can_force_numpy_fallback(self) -> None:
        image = np.zeros((17, 19), dtype=np.uint16)
        image[8, 9] = 50000

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_mask_dog_compiled_cuda",
                side_effect=AssertionError("native backend should not be called"),
            ):
                star_shrink_ops._select_star_mask_dog_backend.cache_clear()
                got = star_mask_dog(
                    image,
                    sigma_small=1.0,
                    sigma_large=3.0,
                    threshold_ratio=1.0,
                    open_ksize=0,
                    dilate_ksize=0,
                )

        expected = star_shrink_ops.star_mask_dog_numpy(
            image,
            sigma_small=1.0,
            sigma_large=3.0,
            threshold_ratio=1.0,
            open_ksize=0,
            dilate_ksize=0,
        )
        np.testing.assert_array_equal(got, expected)

    def test_star_mask_dog_cuda_runtime_falls_back_to_numpy(self) -> None:
        image = np.zeros((9, 11), dtype=np.uint8)
        image[4, 5] = 255
        candidate = backend_registry.BackendCandidate(
            "star_mask_dog",
            "cuda_host_io",
            "star_mask_dog_cuda",
            priority=10,
            build_flag="cuda",
        )
        selection = backend_registry.BackendSelection(candidate, object(), None)

        with mock.patch.object(
            star_shrink_ops,
            "_select_star_mask_dog_backend",
            return_value=selection,
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_mask_dog_compiled_cuda",
                side_effect=RuntimeError("no CUDA-capable device is detected"),
            ) as mock_cuda:
                got = star_mask_dog(
                    image,
                    sigma_small=1.0,
                    sigma_large=2.0,
                    threshold_ratio=1.0,
                    open_ksize=0,
                    dilate_ksize=0,
                )

        mock_cuda.assert_called_once()
        expected = star_shrink_ops.star_mask_dog_numpy(
            image,
            sigma_small=1.0,
            sigma_large=2.0,
            threshold_ratio=1.0,
            open_ksize=0,
            dilate_ksize=0,
        )
        np.testing.assert_array_equal(got, expected)

    def test_star_mask_dog_cuda_detects_synthetic_star_when_available(self) -> None:
        info = cuda_memory_info()
        if not info.get("available"):
            self.skipTest(f"CUDA runtime unavailable: {info.get('reason', 'unknown')}")

        image = np.zeros((33, 35), dtype=np.uint16)
        image[16, 17] = 60000
        image[15:18, 16:19] = np.maximum(image[15:18, 16:19], 18000)
        got = star_shrink_ops.star_mask_dog_compiled_cuda(
            image,
            sigma_small=1.0,
            sigma_large=3.0,
            threshold_ratio=1.0,
            open_ksize=0,
            dilate_ksize=0,
        )
        expected = star_shrink_ops.star_mask_dog_numpy(
            image,
            sigma_small=1.0,
            sigma_large=3.0,
            threshold_ratio=1.0,
            open_ksize=0,
            dilate_ksize=0,
        )

        self.assertEqual(got.dtype, np.uint8)
        self.assertEqual(got.shape, image.shape)
        self.assertEqual(int(got[16, 17]), 1)
        self.assertLessEqual(
            abs(int(np.count_nonzero(got)) - int(np.count_nonzero(expected))), 8
        )

    def test_star_shrink_dog_process_can_force_numpy_fallback(self) -> None:
        image = np.zeros((17, 19, 3), dtype=np.uint8)
        image[8, 9] = [255, 255, 255]

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_shrink_dog_process_compiled_cuda",
                side_effect=AssertionError("native backend should not be called"),
            ):
                star_shrink_ops._select_star_shrink_dog_process_backend.cache_clear()
                got = star_shrink_dog_process(
                    image,
                    sigma_small=1.0,
                    sigma_large=3.0,
                    threshold_ratio=1.0,
                    open_ksize=0,
                    dilate_ksize=0,
                    shrink_ksize=3,
                    shrink_shape="CIRCLE",
                    shrink_times=1,
                    shrink_ratio=1.0,
                    deringing_ksize=5,
                )

        expected = star_shrink_ops.star_shrink_dog_process_numpy(
            image,
            sigma_small=1.0,
            sigma_large=3.0,
            threshold_ratio=1.0,
            open_ksize=0,
            dilate_ksize=0,
            shrink_ksize=3,
            shrink_shape="CIRCLE",
            shrink_times=1,
            shrink_ratio=1.0,
            deringing_ksize=5,
        )
        np.testing.assert_array_equal(got, expected)

    def test_star_shrink_dog_process_cuda_runtime_falls_back_to_composed_path(
        self,
    ) -> None:
        image = np.zeros((11, 13, 3), dtype=np.uint8)
        image[5, 6] = [255, 255, 255]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[4:7, 5:8] = 1
        candidate = backend_registry.BackendCandidate(
            "star_shrink_dog_process",
            "cuda_host_io",
            "star_shrink_dog_process_cuda",
            priority=10,
            build_flag="cuda",
        )
        selection = backend_registry.BackendSelection(candidate, object(), None)

        with mock.patch.object(
            star_shrink_ops,
            "_select_star_shrink_dog_process_backend",
            return_value=selection,
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_shrink_dog_process_compiled_cuda",
                side_effect=RuntimeError("no CUDA-capable device is detected"),
            ) as mock_cuda:
                with mock.patch.object(
                    star_shrink_ops, "star_mask_dog", return_value=mask
                ) as mock_mask:
                    with mock.patch.object(
                        star_shrink_ops,
                        "star_shrink_process",
                        wraps=star_shrink_ops.star_shrink_process_numpy,
                    ) as mock_process:
                        got = star_shrink_dog_process(
                            image,
                            sigma_small=1.0,
                            sigma_large=2.0,
                            threshold_ratio=1.0,
                            open_ksize=0,
                            dilate_ksize=0,
                            shrink_ksize=3,
                            shrink_shape="CIRCLE",
                            shrink_times=1,
                            shrink_ratio=1.0,
                            deringing_ksize=5,
                        )

        mock_cuda.assert_called_once()
        mock_mask.assert_called_once()
        mock_process.assert_called_once()
        expected = star_shrink_ops.star_shrink_process_numpy(
            image,
            mask,
            3,
            "CIRCLE",
            1,
            1.0,
            5,
        )
        np.testing.assert_array_equal(got, expected)

    def test_star_shrink_dog_process_cuda_backend_error_propagates(self) -> None:
        image = np.zeros((11, 13, 3), dtype=np.uint8)
        candidate = backend_registry.BackendCandidate(
            "star_shrink_dog_process",
            "cuda_host_io",
            "star_shrink_dog_process_cuda",
            priority=10,
            build_flag="cuda",
        )
        selection = backend_registry.BackendSelection(candidate, object(), None)

        with mock.patch.object(
            star_shrink_ops,
            "_select_star_shrink_dog_process_backend",
            return_value=selection,
        ):
            with mock.patch.object(
                star_shrink_ops,
                "star_shrink_dog_process_compiled_cuda",
                side_effect=RuntimeError("kernel launch failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "kernel launch failed"):
                    star_shrink_dog_process(
                        image,
                        sigma_small=1.0,
                        sigma_large=2.0,
                        threshold_ratio=1.0,
                        open_ksize=0,
                        dilate_ksize=0,
                        shrink_ksize=3,
                        shrink_shape="CIRCLE",
                        shrink_times=1,
                        shrink_ratio=1.0,
                        deringing_ksize=5,
                    )

    def test_star_shrink_dog_process_cuda_matches_composed_when_available(self) -> None:
        info = cuda_memory_info()
        if not info.get("available"):
            self.skipTest(f"CUDA runtime unavailable: {info.get('reason', 'unknown')}")

        rng = np.random.default_rng(8642)
        image = rng.integers(200, 18000, size=(24, 27, 3), dtype=np.uint16)
        image[11:14, 12:15] = np.maximum(image[11:14, 12:15], 52000)

        got = star_shrink_ops.star_shrink_dog_process_compiled_cuda(
            image,
            sigma_small=1.0,
            sigma_large=3.0,
            threshold_ratio=1.0,
            open_ksize=0,
            dilate_ksize=0,
            shrink_ksize=3,
            shrink_shape="CIRCLE",
            shrink_times=1,
            shrink_ratio=1.0,
            deringing_ksize=5,
        )
        mask = star_shrink_ops.star_mask_dog_compiled_cuda(
            image,
            sigma_small=1.0,
            sigma_large=3.0,
            threshold_ratio=1.0,
            open_ksize=0,
            dilate_ksize=0,
        )
        expected = star_shrink_ops.star_shrink_process_compiled_cuda(
            image,
            mask,
            3,
            "CIRCLE",
            1,
            1.0,
            5,
        )

        self.assertEqual(got.dtype, image.dtype)
        self.assertEqual(got.shape, image.shape)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1)
