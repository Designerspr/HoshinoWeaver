from unittest import mock

import numpy as np

from hoshicore._custom_op import (
    equalize_noise_correct,
    noise_equalization_params,
    noise_fill_local_mean,
)
import hoshicore._custom_op.ops.noise as noise_ops
import hoshicore.component.noise_equalization as noise_equalization


from tests.custom_ops._base import CustomOpsTestCase


class TestNoiseCustomOps(CustomOpsTestCase):
    def test_equalize_noise_correct_matches_numpy(self) -> None:
        max_img = np.array([[20.0, 180.0], [90.0, 250.0]], dtype=np.float64)
        filled_std_img = np.array([[8.0, 20.0], [12.0, 25.0]], dtype=np.float64)

        got = equalize_noise_correct(max_img, filled_std_img, 10.0, 1.25, 255.0, 0.9)
        expected = noise_ops.equalize_noise_correct_numpy(
            max_img,
            filled_std_img,
            10.0,
            1.25,
            255.0,
            0.9,
        )

        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-7)

    def test_equalize_noise_correct_can_force_numpy_fallback(self) -> None:
        max_img = np.array([[20.0, 180.0], [90.0, 250.0]], dtype=np.float64)
        filled_std_img = np.array([[8.0, 20.0], [12.0, 25.0]], dtype=np.float64)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                noise_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                noise_ops._select_equalize_noise_backend.cache_clear()
                got = equalize_noise_correct(
                    max_img, filled_std_img, 10.0, 1.25, 255.0, 0.9
                )

        expected = noise_ops.equalize_noise_correct_numpy(
            max_img,
            filled_std_img,
            10.0,
            1.25,
            255.0,
            0.9,
        )
        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-7)

    def test_noise_fill_local_mean_matches_numpy(self) -> None:
        rng = np.random.default_rng(123)
        img = rng.normal(size=(7, 8, 3)).astype(np.float64)
        mask = rng.random(size=img.shape) > 0.72

        got = noise_fill_local_mean(img, mask, kernel_size=5)
        expected = noise_ops.noise_fill_local_mean_numpy(img, mask, kernel_size=5)

        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_noise_fill_local_mean_float32_2d_matches_numpy(self) -> None:
        img = np.arange(30, dtype=np.float32).reshape(5, 6)
        mask = np.zeros_like(img, dtype=bool)
        mask[0, 0] = True
        mask[2, 3] = True
        mask[-1, -1] = True

        got = noise_fill_local_mean(img, mask, kernel_size=3)
        expected = noise_ops.noise_fill_local_mean_numpy(img, mask, kernel_size=3)

        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_noise_fill_local_mean_all_valid_and_all_mask(self) -> None:
        img = np.arange(24, dtype=np.float64).reshape(4, 3, 2)

        all_valid = np.zeros_like(img, dtype=bool)
        valid_result = noise_fill_local_mean(img, all_valid, kernel_size=3)
        np.testing.assert_array_equal(valid_result, img)

        all_mask = np.ones_like(img, dtype=bool)
        masked_result = noise_fill_local_mean(img, all_mask, kernel_size=3)
        np.testing.assert_array_equal(masked_result, np.zeros_like(img))

    def test_noise_fill_local_mean_can_force_numpy_fallback(self) -> None:
        img = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float64)
        mask = np.array([[True, False], [False, True]])

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                noise_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                noise_ops._select_fill_local_mean_backend.cache_clear()
                got = noise_fill_local_mean(img, mask, kernel_size=3)

        expected = noise_ops.noise_fill_local_mean_numpy(img, mask, kernel_size=3)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_noise_equalization_params_matches_numpy(self) -> None:
        rng = np.random.default_rng(321)
        mean_img = rng.normal(loc=100.0, scale=3.0, size=(5, 6, 3)).astype(np.float64)
        std_img = rng.uniform(0.5, 8.0, size=mean_img.shape).astype(np.float64)
        max_img = mean_img + std_img * rng.uniform(0.1, 3.0, size=mean_img.shape)
        n_img = rng.integers(1, 8, size=mean_img.shape).astype(np.int32)
        std_img[1, 2, 0] = 100.0

        got = noise_equalization_params(
            max_img,
            mean_img,
            std_img,
            n_img,
            top_fraction=0.35,
            sigma_reject=2.0,
        )
        expected = noise_ops.noise_equalization_params_numpy(
            max_img,
            mean_img,
            std_img,
            n_img,
            top_fraction=0.35,
            sigma_reject=2.0,
        )

        self.assertIsNotNone(got)
        self.assertIsNotNone(expected)
        got_sigma_ref, got_c_n_eff, got_mask = got
        expected_sigma_ref, expected_c_n_eff, expected_mask = expected
        self.assertAlmostEqual(got_sigma_ref, expected_sigma_ref, places=10)
        self.assertAlmostEqual(got_c_n_eff, expected_c_n_eff, places=10)
        np.testing.assert_array_equal(got_mask, expected_mask)

    def test_noise_equalization_params_accepts_2d_count_image_for_rgb(self) -> None:
        rng = np.random.default_rng(654)
        mean_img = rng.normal(loc=50.0, scale=2.0, size=(4, 5, 3)).astype(np.float64)
        std_img = rng.uniform(0.25, 5.0, size=mean_img.shape).astype(np.float64)
        max_img = mean_img + std_img * rng.uniform(0.1, 2.5, size=mean_img.shape)
        n_img = rng.integers(1, 10, size=mean_img.shape[:2]).astype(np.int32)
        std_img[0, 1, 2] = 25.0

        got = noise_equalization_params(
            max_img,
            mean_img,
            std_img,
            n_img,
            top_fraction=0.4,
            sigma_reject=1.5,
        )
        expected = noise_ops.noise_equalization_params_numpy(
            max_img,
            mean_img,
            std_img,
            n_img,
            top_fraction=0.4,
            sigma_reject=1.5,
        )

        self.assertIsNotNone(got)
        self.assertIsNotNone(expected)
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(got[2], expected[2])

    def test_noise_equalization_params_float32_minus_only_matches_numpy(self) -> None:
        rng = np.random.default_rng(987)
        mean_img = rng.normal(loc=20.0, scale=1.0, size=(4, 4, 2)).astype(np.float32)
        std_img = rng.uniform(0.1, 3.0, size=mean_img.shape).astype(np.float32)
        max_img = mean_img + std_img * rng.uniform(
            0.2, 2.0, size=mean_img.shape
        ).astype(np.float32)
        n_img = rng.integers(1, 6, size=mean_img.shape).astype(np.int32)

        got = noise_equalization_params(
            max_img,
            mean_img,
            std_img,
            n_img,
            top_fraction=0.5,
            sigma_reject=2.0,
            minus_only=True,
        )
        expected = noise_ops.noise_equalization_params_numpy(
            max_img,
            mean_img,
            std_img,
            n_img,
            top_fraction=0.5,
            sigma_reject=2.0,
            minus_only=True,
        )

        self.assertIsNotNone(got)
        self.assertIsNotNone(expected)
        self.assertEqual(got[0], 0.0)
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(got[2], expected[2])

    def test_noise_equalization_params_can_force_numpy_fallback(self) -> None:
        mean_img = np.ones((3, 4, 3), dtype=np.float64)
        std_img = np.full_like(mean_img, 2.0)
        max_img = mean_img + std_img
        n_img = np.arange(mean_img.size).reshape(mean_img.shape)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                noise_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                noise_ops._select_equalization_params_backend.cache_clear()
                got = noise_equalization_params(max_img, mean_img, std_img, n_img)

        expected = noise_ops.noise_equalization_params_numpy(
            max_img, mean_img, std_img, n_img
        )
        self.assertIsNotNone(got)
        self.assertIsNotNone(expected)
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(got[2], expected[2])

    def test_noise_equalization_params_returns_none_without_valid_sigma(self) -> None:
        mean_img = np.ones((2, 2, 3), dtype=np.float64)
        std_img = np.zeros_like(mean_img)
        max_img = mean_img.copy()
        n_img = np.ones_like(mean_img)

        self.assertIsNone(noise_equalization_params(max_img, mean_img, std_img, n_img))

    def test_noise_equalization_params_rejects_non_median_helper_method(self) -> None:
        mean_img = np.ones((2, 2, 3), dtype=np.float64)
        std_img = np.ones_like(mean_img)
        max_img = mean_img + std_img
        n_img = np.ones_like(mean_img)

        with self.assertRaises(ValueError):
            noise_equalization_params(
                max_img,
                mean_img,
                std_img,
                n_img,
                estimate_method="ransac",
            )

    def test_equalize_noise_routes_pixel_correction_through_custom_op(self) -> None:
        max_img = np.array(
            [
                [[12.0, 15.0, 18.0], [30.0, 36.0, 42.0]],
                [[24.0, 20.0, 16.0], [48.0, 45.0, 51.0]],
            ],
            dtype=np.float64,
        )
        mean_img = np.array(
            [
                [[10.0, 11.0, 13.0], [28.0, 30.0, 35.0]],
                [[21.0, 18.0, 14.0], [40.0, 39.0, 43.0]],
            ],
            dtype=np.float64,
        )
        std_img = np.array(
            [[[2.0, 3.0, 4.0], [5.0, 7.0, 6.0]], [[3.0, 2.0, 1.5], [8.0, 7.0, 9.0]]],
            dtype=np.float64,
        )
        n_img = np.array(
            [[[10, 10, 10], [9, 9, 9]], [[10, 10, 10], [8, 8, 8]]],
            dtype=np.uint16,
        )

        max_value = float(np.max(max_img))
        threshold = np.quantile(n_img, 1.0 - 0.25)
        bg_mask = n_img >= threshold
        residual = (max_img - mean_img)[bg_mask]
        sigma_bg = std_img[bg_mask]
        valid = sigma_bg > 0
        r_valid = residual[valid]
        s_valid = sigma_bg[valid]
        c_n_eff = float(np.median(r_valid / s_valid))
        sigma_ref = np.median(s_valid)
        squeeze_std = std_img.reshape((-1, 3))
        mean_std = np.mean(squeeze_std, axis=0)
        std_std = np.std(squeeze_std, axis=0)
        mask = std_img > (mean_std + 3.0 * std_std)[None, None, ...]
        filled_std_img = noise_equalization.fill_local_mean(
            std_img, mask, kernel_size=21
        )
        expected = noise_ops.equalize_noise_correct_numpy(
            max_img,
            filled_std_img,
            sigma_ref,
            c_n_eff,
            max_value,
            0.9,
        )

        with mock.patch.object(
            noise_equalization,
            "custom_equalize_noise_correct",
            wraps=noise_ops.equalize_noise_correct_numpy,
        ) as patched_custom:
            with mock.patch.object(
                noise_equalization,
                "custom_noise_equalization_params",
                wraps=noise_ops.noise_equalization_params,
            ) as patched_params:
                got = noise_equalization.equalize_noise(
                    max_img,
                    mean_img,
                    std_img,
                    n_img,
                    top_fraction=0.25,
                    sigma_reject=3.0,
                    highlight_preserve=0.9,
                )

        patched_custom.assert_called_once()
        patched_params.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-7)

    def test_equalize_noise_ransac_bypasses_fused_prepare(self) -> None:
        max_img = np.array(
            [
                [[12.0, 15.0, 18.0], [30.0, 36.0, 42.0]],
                [[24.0, 20.0, 16.0], [48.0, 45.0, 51.0]],
            ],
            dtype=np.float64,
        )
        mean_img = np.array(
            [
                [[10.0, 11.0, 13.0], [28.0, 30.0, 35.0]],
                [[21.0, 18.0, 14.0], [40.0, 39.0, 43.0]],
            ],
            dtype=np.float64,
        )
        std_img = np.array(
            [[[2.0, 3.0, 4.0], [5.0, 7.0, 6.0]], [[3.0, 2.0, 1.5], [8.0, 7.0, 9.0]]],
            dtype=np.float64,
        )
        n_img = np.array(
            [[[10, 10, 10], [9, 9, 9]], [[10, 10, 10], [8, 8, 8]]],
            dtype=np.uint16,
        )

        with mock.patch.object(
            noise_equalization,
            "custom_noise_equalization_params",
            side_effect=AssertionError("median params helper should not be called"),
        ) as patched_params:
            with mock.patch.object(
                noise_equalization, "_ransac_ratio", return_value=1.25
            ) as patched_ransac:
                got = noise_equalization.equalize_noise(
                    max_img,
                    mean_img,
                    std_img,
                    n_img,
                    estimate_method="ransac",
                    top_fraction=0.25,
                    sigma_reject=3.0,
                    highlight_preserve=0.9,
                )

        patched_params.assert_not_called()
        patched_ransac.assert_called_once()
        self.assertEqual(got.shape, max_img.shape)
