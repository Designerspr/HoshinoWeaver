import asyncio
import unittest
from unittest import mock

import cv2
import numpy as np

from hoshicore._custom_op import (
    build_info,
    custom_ops_available,
    equalize_noise_correct,
    extract_point_features,
    fgp_add,
    fgp_accumulate,
    fgp_masked_mean_merge,
    find_initial_match,
    huber_weighted_chunk,
    huber_weighted_accumulate,
    max_combine,
    median_filter_2d,
    median_reduce_chunk,
    noise_equalization_params,
    noise_fill_local_mean,
    sigma_clip_fused_masked_merge,
    sigma_clip_fused_merge,
    star_mask_dog,
    star_shrink_detect_mask,
    star_shrink_dog_process,
    star_shrink_process,
    threshold_max_merge as custom_threshold_max_merge,
)
import hoshicore._custom_op.backend_registry as backend_registry
from hoshicore._custom_op._dispatch import cuda_memory_info
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
import hoshicore._custom_op.ops.alignment as alignment_ops
import hoshicore._custom_op.ops.fgp as fgp_ops
import hoshicore._custom_op.ops.filter as filter_ops
import hoshicore._custom_op.ops.max as max_ops
import hoshicore._custom_op.ops.median as median_ops
import hoshicore._custom_op.ops.noise as noise_ops
import hoshicore._custom_op.ops.star_shrink as star_shrink_ops
from hoshicore.component.data_container import FastGaussianParam, HuberMeanParam
from hoshicore.component.frame_buffer import MemoryFrameBuffer
import hoshicore.component.noise_equalization as noise_equalization
import hoshicore.component.norma.frame_align as frame_align
import hoshicore.component.norma.matching as norma_matching
import hoshicore.component.star_detect as star_detect
from hoshicore.component.merger import HuberWeightedMerger
from hoshicore.component.merger import MaxMerger
from hoshicore.component.merger import MeanMerger
import hoshicore.ops.sigma_clip_ops as sigma_clip_ops
from hoshicore.ops.sigma_clip_ops import MedianReduceOp
from hoshicore.ops.sigma_clip_ops import ThresholdMaxIteratorOp
import hoshicore._custom_op.ops.sigma_clip as sigma_clip_chunk_ops
from hoshicore.ops.trailstacker import MeanStackerOp


def _naive_median_filter_2d(image: np.ndarray, ksize: int) -> np.ndarray:
    radius = ksize // 2
    if image.ndim == 2:
        padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
        out = np.empty_like(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                window = padded[y:y + ksize, x:x + ksize]
                out[y, x] = np.median(window).astype(image.dtype)
        return out

    padded = np.pad(
        image,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="edge",
    )
    out = np.empty_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                window = padded[y:y + ksize, x:x + ksize, c]
                out[y, x, c] = np.median(window).astype(image.dtype)
    return out


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


class TestCustomOpsFallback(unittest.TestCase):
    def tearDown(self) -> None:
        filter_ops._load_compiled_module_result.cache_clear()
        filter_ops._select_median_filter_backend.cache_clear()
        fgp_ops._load_compiled_module_result.cache_clear()
        fgp_ops._compiled_build_info.cache_clear()
        fgp_ops._select_fgp_backend.cache_clear()
        fgp_ops._select_fgp_add_backend.cache_clear()
        fgp_ops._select_huber_backend.cache_clear()
        fgp_ops._select_huber_chunk_backend.cache_clear()
        fgp_ops._LAST_APPLIED_COMPILED_THREADS = None
        max_ops._load_compiled_module_result.cache_clear()
        max_ops._compiled_build_info.cache_clear()
        max_ops._select_max_backend.cache_clear()
        max_ops._select_threshold_max_backend.cache_clear()
        max_ops._LAST_APPLIED_COMPILED_THREADS = None
        noise_ops._load_compiled_module_result.cache_clear()
        noise_ops._compiled_build_info.cache_clear()
        noise_ops._select_equalize_noise_backend.cache_clear()
        noise_ops._select_fill_local_mean_backend.cache_clear()
        noise_ops._select_equalization_params_backend.cache_clear()
        noise_ops._LAST_APPLIED_COMPILED_THREADS = None
        star_shrink_ops._load_compiled_module_result.cache_clear()
        star_shrink_ops._select_star_mask_dog_backend.cache_clear()
        star_shrink_ops._select_star_shrink_detect_mask_backend.cache_clear()
        star_shrink_ops._select_star_shrink_dog_process_backend.cache_clear()
        star_shrink_ops._select_star_shrink_process_backend.cache_clear()
        median_ops._load_compiled_module_result.cache_clear()
        median_ops._compiled_build_info.cache_clear()
        median_ops._select_median_backend.cache_clear()
        median_ops._LAST_APPLIED_COMPILED_THREADS = None
        sigma_clip_chunk_ops._load_compiled_module_result.cache_clear()
        sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()
        alignment_ops._load_compiled_module_result.cache_clear()
        alignment_ops._select_extract_point_features_backend.cache_clear()
        alignment_ops._select_find_initial_match_backend.cache_clear()

    def test_max_combine_matches_numpy(self) -> None:
        base = np.array([[1, 5], [3, 4]], dtype=np.uint16)
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        got = max_combine(base, fresh)
        expected = np.maximum(np.array([[1, 5], [3, 4]], dtype=np.uint16), fresh)

        self.assertIs(got, base)
        np.testing.assert_array_equal(base, expected)
        np.testing.assert_array_equal(got, expected)

    def test_max_merger_works_with_custom_op_fallback(self) -> None:
        merger = MaxMerger(int_weight=False)
        merger.merge(np.array([[1, 9], [2, 3]], dtype=np.uint16), None)
        merger.merge(np.array([[5, 1], [2, 8]], dtype=np.uint16), None)

        np.testing.assert_array_equal(
            merger.merged_image,
            np.array([[5, 9], [2, 8]], dtype=np.uint16),
        )

    def test_threshold_max_merge_matches_numpy(self) -> None:
        frame = np.array([[6.0, 12.0], [4.0, 15.0]], dtype=np.float64)
        mean_img = np.array([[5.0, 8.0], [4.5, 10.0]], dtype=np.float64)
        std_img = np.array([[0.5, 1.0], [0.5, 2.0]], dtype=np.float64)
        base = np.array([[5.0, 9.0], [4.5, 11.0]], dtype=np.float64)
        expected = np.array(base, copy=True)

        got = custom_threshold_max_merge(frame, mean_img, std_img, base, 2.0, 0.5)
        max_ops.threshold_max_merge_numpy(frame, mean_img, std_img, expected, 2.0, 0.5)

        self.assertIs(got, base)
        np.testing.assert_allclose(base, expected, rtol=1e-7, atol=1e-7)

    def test_threshold_max_merge_can_force_numpy_fallback(self) -> None:
        frame = np.array([[6.0, 12.0], [4.0, 15.0]], dtype=np.float64)
        mean_img = np.array([[5.0, 8.0], [4.5, 10.0]], dtype=np.float64)
        std_img = np.array([[0.5, 1.0], [0.5, 2.0]], dtype=np.float64)
        base = np.array([[5.0, 9.0], [4.5, 11.0]], dtype=np.float64)
        expected = np.array(base, copy=True)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(max_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                max_ops._select_threshold_max_backend.cache_clear()
                got = custom_threshold_max_merge(frame, mean_img, std_img, base, 2.0, 0.5)

        max_ops.threshold_max_merge_numpy(frame, mean_img, std_img, expected, 2.0, 0.5)
        self.assertIs(got, base)
        np.testing.assert_allclose(base, expected, rtol=1e-7, atol=1e-7)

    def test_threshold_max_merge_keeps_morphology_on_numpy_path(self) -> None:
        frame = np.array([[6.0, 12.0], [4.0, 15.0]], dtype=np.float64)
        mean_img = np.array([[5.0, 8.0], [4.5, 10.0]], dtype=np.float64)
        std_img = np.array([[0.5, 1.0], [0.5, 2.0]], dtype=np.float64)
        result = np.array(mean_img, copy=True)
        expected = np.array(mean_img, copy=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        with mock.patch.object(noise_equalization, "custom_threshold_max_merge") as patched_custom:
            noise_equalization.threshold_max_merge(
                frame,
                mean_img,
                std_img,
                result,
                2.0,
                0.5,
                kernel,
            )

        mask = frame > (mean_img + 2.0 * std_img)
        mask = cv2.morphologyEx(mask.view(np.uint8), cv2.MORPH_OPEN, kernel).view(bool)
        signal = frame * 0.5
        np.maximum(expected, np.where(mask, signal, mean_img), out=expected)

        patched_custom.assert_not_called()
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(noise_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                noise_ops._select_equalize_noise_backend.cache_clear()
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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(noise_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
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
        max_img = mean_img + std_img * rng.uniform(0.2, 2.0, size=mean_img.shape).astype(np.float32)
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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(noise_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                noise_ops._select_equalization_params_backend.cache_clear()
                got = noise_equalization_params(max_img, mean_img, std_img, n_img)

        expected = noise_ops.noise_equalization_params_numpy(max_img, mean_img, std_img, n_img)
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
            [[[12.0, 15.0, 18.0], [30.0, 36.0, 42.0]],
             [[24.0, 20.0, 16.0], [48.0, 45.0, 51.0]]],
            dtype=np.float64,
        )
        mean_img = np.array(
            [[[10.0, 11.0, 13.0], [28.0, 30.0, 35.0]],
             [[21.0, 18.0, 14.0], [40.0, 39.0, 43.0]]],
            dtype=np.float64,
        )
        std_img = np.array(
            [[[2.0, 3.0, 4.0], [5.0, 7.0, 6.0]],
             [[3.0, 2.0, 1.5], [8.0, 7.0, 9.0]]],
            dtype=np.float64,
        )
        n_img = np.array(
            [[[10, 10, 10], [9, 9, 9]],
             [[10, 10, 10], [8, 8, 8]]],
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
        mask = (std_img > (mean_std + 3.0 * std_std)[None, None, ...])
        filled_std_img = noise_equalization.fill_local_mean(std_img, mask, kernel_size=21)
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
            [[[12.0, 15.0, 18.0], [30.0, 36.0, 42.0]],
             [[24.0, 20.0, 16.0], [48.0, 45.0, 51.0]]],
            dtype=np.float64,
        )
        mean_img = np.array(
            [[[10.0, 11.0, 13.0], [28.0, 30.0, 35.0]],
             [[21.0, 18.0, 14.0], [40.0, 39.0, 43.0]]],
            dtype=np.float64,
        )
        std_img = np.array(
            [[[2.0, 3.0, 4.0], [5.0, 7.0, 6.0]],
             [[3.0, 2.0, 1.5], [8.0, 7.0, 9.0]]],
            dtype=np.float64,
        )
        n_img = np.array(
            [[[10, 10, 10], [9, 9, 9]],
             [[10, 10, 10], [8, 8, 8]]],
            dtype=np.uint16,
        )

        with mock.patch.object(
            noise_equalization,
            "custom_noise_equalization_params",
            side_effect=AssertionError("median params helper should not be called"),
        ) as patched_params:
            with mock.patch.object(noise_equalization, "_ransac_ratio", return_value=1.25) as patched_ransac:
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

    def test_star_shrink_process_compiled_matches_numpy_uint16(self) -> None:
        rng = np.random.default_rng(2468)
        base = rng.normal(loc=12000.0, scale=1200.0, size=(16, 18)).clip(0, 50000)
        for y, x, value in [(5, 6, 42000), (8, 10, 36000), (11, 13, 39000)]:
            base[y, x] = value
            base[y - 1:y + 2, x - 1:x + 2] = np.maximum(
                base[y - 1:y + 2, x - 1:x + 2],
                value * 0.45,
            )
        image = np.stack(
            [
                base * 0.96,
                base,
                base * 1.04,
            ],
            axis=2,
        ).clip(0, 65535).astype(np.uint16)
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
                [[16, 17, 18], [55, 58, 61], [140, 145, 150], [85, 88, 91], [45, 47, 49]],
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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
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

    def test_star_shrink_detect_mask_compiled_matches_numpy_rgb_morphology(self) -> None:
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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
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
        self.assertLessEqual(abs(int(np.count_nonzero(got)) - int(np.count_nonzero(expected))), 8)

    def test_star_shrink_dog_process_can_force_numpy_fallback(self) -> None:
        image = np.zeros((17, 19, 3), dtype=np.uint8)
        image[8, 9] = [255, 255, 255]

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
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

    def test_star_shrink_dog_process_cuda_runtime_falls_back_to_composed_path(self) -> None:
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
                with mock.patch.object(star_shrink_ops, "star_mask_dog", return_value=mask) as mock_mask:
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

    def test_median_reduce_chunk_matches_numpy(self) -> None:
        stack = np.array(
            [
                [[[1.0], [5.0]], [[3.0], [4.0]]],
                [[[2.0], [4.0]], [[7.0], [1.0]]],
                [[[6.0], [8.0]], [[5.0], [9.0]]],
                [[[0.0], [7.0]], [[2.0], [6.0]]],
            ],
            dtype=np.float32,
        )

        got = median_reduce_chunk(stack)
        expected = np.median(stack, axis=0)

        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_median_reduce_chunk_integer_types(self) -> None:
        """Test uint8/uint16 with both odd and even frame counts."""
        for dtype in (np.uint8, np.uint16):
            # Odd frame count: exact median, no averaging
            stack_odd = np.array(
                [[[10, 50], [30, 40]],
                 [[20, 40], [70, 10]],
                 [[60, 80], [50, 90]]],
                dtype=dtype,
            )
            got = median_reduce_chunk(stack_odd)
            expected = np.median(stack_odd, axis=0).astype(dtype)
            np.testing.assert_array_equal(got, expected,
                                          err_msg=f"{dtype.__name__} odd-N")

            # Even frame count: average of two middle values (truncated)
            stack_even = np.array(
                [[[10, 50], [30, 40]],
                 [[20, 40], [70, 10]],
                 [[60, 80], [50, 90]],
                 [[0, 70], [20, 60]]],
                dtype=dtype,
            )
            got = median_reduce_chunk(stack_even)
            expected = np.median(stack_even, axis=0).astype(dtype)
            np.testing.assert_array_equal(got, expected,
                                          err_msg=f"{dtype.__name__} even-N")

    def test_median_reduce_chunk_can_force_numpy_fallback(self) -> None:
        stack = np.array(
            [
                [[[1.0], [5.0]], [[3.0], [4.0]]],
                [[[2.0], [4.0]], [[7.0], [1.0]]],
                [[[6.0], [8.0]], [[5.0], [9.0]]],
                [[[0.0], [7.0]], [[2.0], [6.0]]],
            ],
            dtype=np.float32,
        )

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(median_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                median_ops._select_median_backend.cache_clear()
                got = median_reduce_chunk(stack)

        expected = np.median(stack, axis=0)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_median_filter_2d_uint8_matches_opencv(self) -> None:
        rng = np.random.default_rng(123)
        for shape in ((17, 19), (13, 15, 3)):
            image = rng.integers(0, 255, size=shape, dtype=np.uint8)
            for ksize in (3, 7, 13):
                got = median_filter_2d(image, ksize)
                expected = cv2.medianBlur(image, ksize)
                np.testing.assert_array_equal(got, expected)

    def test_median_filter_2d_uint16_large_kernel_matches_naive(self) -> None:
        rng = np.random.default_rng(456)
        image_2d = rng.integers(0, 65535, size=(6, 7), dtype=np.uint16)
        image_3d = rng.integers(0, 65535, size=(5, 6, 3), dtype=np.uint16)

        for image, ksize in ((image_2d, 7), (image_3d, 7)):
            got = median_filter_2d(image, ksize)
            expected = _naive_median_filter_2d(image, ksize)
            np.testing.assert_array_equal(got, expected)

    def test_median_filter_2d_compiled_matches_numpy_large_kernel(self) -> None:
        rng = np.random.default_rng(654)
        for shape in ((6, 7), (5, 6, 1), (5, 6, 4)):
            image = rng.integers(0, 65535, size=shape, dtype=np.uint16)
            got = filter_ops.median_filter_2d_compiled(image, 7)
            expected = filter_ops.median_filter_2d_numpy(image, 7)
            np.testing.assert_array_equal(got, expected)

    def test_median_filter_2d_uint16_small_kernel_matches_opencv(self) -> None:
        rng = np.random.default_rng(789)
        for shape in ((11, 13), (9, 10, 1), (9, 10, 3)):
            image = rng.integers(0, 65535, size=shape, dtype=np.uint16)
            for ksize in (3, 5):
                got = median_filter_2d(image, ksize)
                expected = cv2.medianBlur(image, ksize)
                if image.ndim == 3 and image.shape[2] == 1:
                    expected = expected[:, :, None]
                np.testing.assert_array_equal(got, expected)

    def test_median_filter_2d_rejects_invalid_ksize(self) -> None:
        image = np.arange(9, dtype=np.uint16).reshape(3, 3)
        with self.assertRaises(ValueError):
            median_filter_2d(image, 4)
        with self.assertRaises(ValueError):
            median_filter_2d(image, 65537)

    def test_median_filter_2d_can_force_numpy_fallback(self) -> None:
        image = np.array(
            [[100, 2, 300], [4, 5000, 6], [700, 8, 900]],
            dtype=np.uint16,
        )

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(filter_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                filter_ops._select_median_filter_backend.cache_clear()
                got = filter_ops.median_filter_2d(image, 7)

        expected = _naive_median_filter_2d(image, 7)
        np.testing.assert_array_equal(got, expected)

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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(alignment_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
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
        np.testing.assert_allclose(features1, expected_features1, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(features2, expected_features2, rtol=1e-10, atol=1e-12)
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
            with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"}, clear=False):
                features1 = extract_point_features(vec, vol, k=8)
                features2 = extract_point_features(vec2, vol2, k=8)

        with mock.patch.object(
            alignment_ops,
            "find_initial_match_compiled",
            wraps=alignment_ops.find_initial_match_compiled,
        ) as patched_match:
            with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"}, clear=False):
                _ = find_initial_match(features1, features2, pts, pts2, vec, vec2)

        self.assertGreaterEqual(patched_extract.call_count, 2)
        patched_match.assert_called_once()

    def test_norma_matching_routes_through_alignment_custom_op(self) -> None:
        vec, vec2, vol, vol2, pts, pts2 = _make_alignment_match_inputs(seed=5)

        features1 = norma_matching.extract_point_features(vec, vol, k=8)
        features2 = norma_matching.extract_point_features(vec2, vol2, k=8)
        got = norma_matching.find_initial_match(features1, features2, pts, pts2, vec, vec2)

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
        np.testing.assert_allclose(features1, expected_features1, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(features2, expected_features2, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(got, expected)

    def test_star_detect_uint16_large_median_uses_custom_filter(self) -> None:
        image = np.zeros((9, 11), dtype=np.uint16)
        image[4, 5] = 50000
        filtered_bg = np.zeros_like(image)

        with mock.patch.object(
            star_detect,
            "median_filter_2d",
            return_value=filtered_bg,
        ) as patched_filter:
            mask = star_detect.detect_starmask_by_threshold(
                image,
                ksize=7,
                threshold_ratio=1,
                open_ksize=0,
                dilate_ksize=0,
            )

        patched_filter.assert_called_once()
        args, _ = patched_filter.call_args
        np.testing.assert_array_equal(args[0], image)
        self.assertEqual(args[1], 7)
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(mask.dtype, np.uint8)

    def test_star_detect_uint16_large_median_forced_fallback_keeps_precision(self) -> None:
        image = np.zeros((9, 11), dtype=np.uint16)
        image[4, 5] = 50000
        original_median_blur = cv2.medianBlur

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            filter_ops._select_median_filter_backend.cache_clear()
            with mock.patch.object(
                star_detect.cv2,
                "medianBlur",
                wraps=original_median_blur,
            ) as patched_median_blur:
                mask = star_detect.detect_starmask_by_threshold(
                    image,
                    ksize=7,
                    threshold_ratio=1,
                    open_ksize=0,
                    dilate_ksize=0,
                )

        patched_median_blur.assert_not_called()
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(mask.dtype, np.uint8)

    def test_median_reduce_op_routes_chunk_through_custom_op(self) -> None:
        frame_buffer = MemoryFrameBuffer()
        frames = [
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            np.array([[2, 4], [7, 1]], dtype=np.uint16),
            np.array([[6, 8], [5, 9]], dtype=np.uint16),
            np.array([[0, 7], [2, 6]], dtype=np.uint16),
        ]
        for frame in frames:
            frame_buffer.append(frame)
        frame_buffer.acquire()

        async def iter_chunk_prefetch(row_ranges):
            for row_start, row_end in row_ranges:
                yield [
                    frame_buffer.get_rows(i, row_start, row_end)
                    for i in range(len(frame_buffer))
                ]

        frame_buffer.iter_chunk_prefetch = iter_chunk_prefetch

        op = MedianReduceOp("median_reduce")
        outputs = {}

        async def run_case() -> None:
            with mock.patch.object(
                sigma_clip_ops,
                "custom_median_reduce_chunk",
                wraps=median_ops.median_reduce_chunk_numpy,
            ) as patched_custom:
                async def run_cpu(fn, *args, **kwargs):
                    return fn(*args, **kwargs)

                async def capture(payload):
                    outputs.update(payload)

                op._run_cpu = run_cpu
                op._broadcast_outputs = capture
                await op._async_execute(
                    {
                        "buffer_handle": frame_buffer,
                        "chunk_rows": 1,
                    }
                )
                self.assertEqual(patched_custom.call_count, 2)

        asyncio.run(run_case())
        expected = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint16)
        self.assertIn("result", outputs)
        np.testing.assert_array_equal(outputs["result"].data, expected)

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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
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
        huber_weight = np.where(
            abs_residual <= 1.345,
            np.ones_like(abs_residual, dtype=np.float32),
            (1.345 / (abs_residual + 1e-10)).astype(np.float32),
        ) * 0.5
        expected_sum = (img * huber_weight).astype(np.float64)
        expected_total = huber_weight.astype(np.float64)

        self.assertIs(got, base)
        np.testing.assert_allclose(base.weighted_sum, expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(base.weight_total, expected_total, rtol=1e-6, atol=1e-6)

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

        got = huber_weighted_accumulate(base, img, ref_mean, ref_std, 1.345, weight=frame_weight)

        residual = (img.astype(np.float32) - ref_mean) / (ref_std + 1e-10)
        abs_residual = np.abs(residual)
        huber_weight = np.where(
            abs_residual <= 1.345,
            np.ones_like(abs_residual, dtype=np.float32),
            (1.345 / (abs_residual + 1e-10)).astype(np.float32),
        ) * frame_weight
        expected_sum = (img * huber_weight).astype(np.float64)
        expected_total = huber_weight.astype(np.float64)

        self.assertIs(got, base)
        np.testing.assert_allclose(base.weighted_sum, expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(base.weight_total, expected_total, rtol=1e-6, atol=1e-6)

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
            stack, ref_mean, ref_std, 1.345, weights)
        expected_sum, expected_total = fgp_ops.huber_weighted_chunk_numpy(
            stack, ref_mean, ref_std, 1.345, weights)

        np.testing.assert_allclose(got_sum, expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got_total, expected_total, rtol=1e-6, atol=1e-6)

    def test_huber_weighted_chunk_can_force_numpy_fallback(self) -> None:
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        ref_mean = np.linspace(2, 8, 6, dtype=np.float64)
        ref_std = np.linspace(1, 3, 6, dtype=np.float64)
        expected = fgp_ops.huber_weighted_chunk_numpy(
            stack, ref_mean, ref_std, 1.345)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(
                    fgp_ops,
                    "huber_weighted_chunk_compiled_cuda",
                    side_effect=AssertionError("native backend should not be called")):
                got = huber_weighted_chunk(stack, ref_mean, ref_std, 1.345)

        for actual, expected_arr in zip(got, expected, strict=True):
            np.testing.assert_allclose(actual, expected_arr, rtol=1e-6, atol=1e-6)

    def test_huber_weighted_chunk_preserves_double_weight_precision(self) -> None:
        stack = np.array([[0], [1]], dtype=np.uint16)
        ref_mean = np.array([0.0], dtype=np.float64)
        ref_std = np.array([1.0], dtype=np.float64)
        weights = np.array([1.0, 1.0 + 1e-8], dtype=np.float64)

        weighted_sum, weight_total = fgp_ops.huber_weighted_chunk_numpy(
            stack, ref_mean, ref_std, 10.0, weights)

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
                            stack, ref_mean, ref_std, 1.345, weights)
                    except RuntimeError as exc:
                        if is_cuda_runtime_unavailable_error(exc):
                            self.skipTest(f"CUDA Huber chunk runtime unavailable: {exc}")
                        raise

                    n_sum, n_total = fgp_ops.huber_weighted_chunk_numpy(
                        stack, ref_mean, ref_std, 1.345, weights)
                    np.testing.assert_allclose(c_sum, n_sum, rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(c_total, n_total, rtol=1e-6, atol=1e-6)

    def test_huber_weighted_chunk_cuda_runtime_falls_back_to_numpy(self) -> None:
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        ref_mean = np.linspace(2, 8, 6, dtype=np.float64)
        ref_std = np.linspace(1, 3, 6, dtype=np.float64)
        expected = fgp_ops.huber_weighted_chunk_numpy(
            stack, ref_mean, ref_std, 1.345)
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
        with mock.patch.object(
                fgp_ops,
                "_select_backend",
                return_value=cuda_selection):
            with mock.patch.object(
                    fgp_ops,
                    "huber_weighted_chunk_compiled_cuda",
                    side_effect=RuntimeError(
                        "huber_weighted_chunk_cuda cudaGetDevice: no CUDA-capable device is detected")):
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
            huber_weight = np.where(
                abs_residual <= 1.345,
                np.ones_like(abs_residual, dtype=np.float32),
                (1.345 / (abs_residual + 1e-10)).astype(np.float32),
            ) * weight
            expected_sum += (frame * huber_weight).astype(np.float64)
            expected_total += huber_weight.astype(np.float64)

        np.testing.assert_allclose(merger.result.weighted_sum, expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(merger.result.weight_total, expected_total, rtol=1e-6, atol=1e-6)

    def test_huber_mean_iterator_falls_back_when_cuda_chunk_unavailable(self) -> None:
        frames = [
            (np.array([[4, 9], [6, 3]], dtype=np.uint16), 0.5),
            (np.array([[5, 6], [7, 1]], dtype=np.uint16), 0.25),
        ]
        fgp_total = FastGaussianParam(frames[0][0], source_dtype=np.dtype("uint16"))
        fgp_total = fgp_total + FastGaussianParam(
            frames[1][0], source_dtype=np.dtype("uint16"))
        op = sigma_clip_ops.HuberMeanIteratorOp("huber")
        state = op._init_chunk_state(
            {"fgp_total": fgp_total, "huber_c": 1.345}, 0, 2, 2)
        op._configs = {"huber_c": 1.345}

        with mock.patch.object(
                sigma_clip_ops,
                "custom_huber_weighted_chunk_available",
                return_value=True):
            with mock.patch.object(
                    sigma_clip_ops,
                    "custom_huber_weighted_chunk_or_none",
                    return_value=None):
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
            frames[1][0], source_dtype=np.dtype("uint16"))
        op = sigma_clip_ops.HuberMeanIteratorOp("huber")
        state = op._init_chunk_state(
            {"fgp_total": fgp_total, "huber_c": 1.345}, 0, 2, 2)
        op._configs = {"huber_c": 1.345}
        weighted_sum = np.array([1.0, 7.0, 16.0, 30.0], dtype=np.float64)
        weight_total = np.array([2.0, 2.0, 3.0, 4.0], dtype=np.float64)

        with mock.patch.object(
                sigma_clip_ops,
                "custom_huber_weighted_chunk_available",
                return_value=True):
            with mock.patch.object(
                    sigma_clip_ops,
                    "custom_huber_weighted_chunk_or_none",
                    return_value=(weighted_sum, weight_total)) as fused:
                with mock.patch.object(state['merger'], "merge") as merge:
                    op._run_pass(state, frames)

        fused.assert_called_once()
        merge.assert_not_called()
        expected = np.round(weighted_sum / weight_total).reshape(2, 2)
        np.testing.assert_allclose(op._finalize_chunk(state), expected)

    def test_chunk_iterator_base_caches_configs_for_hooks(self) -> None:
        frame = np.array([[1, 2]], dtype=np.uint16)

        class FakeFrameBuffer:
            def __getitem__(self, idx):
                if idx != 0:
                    raise IndexError(idx)
                return frame, None

            async def iter_chunk_prefetch(self, row_ranges):
                for _ in row_ranges:
                    yield [(frame, None)]

            def cleanup(self):
                self.cleaned = True

        class ConfigCaptureChunkOp(sigma_clip_ops.ChunkIteratorBaseOp):
            OUTPUTS = {"result": {"type": "image"}}

            def _init_chunk_state(self, configs, row_start, row_end, w):
                self.init_marker = self._configs["marker"]
                return {}

            def _run_pass(self, state, chunk_stack):
                self.run_marker = self._configs["marker"]

            def _check_convergence(self, state, pass_idx):
                return True

            def _finalize_chunk(self, state):
                return frame.copy()

            def _wrap_output(self, result, configs):
                self.wrap_marker = self._configs["marker"]
                return {"result": result}

        buffer = FakeFrameBuffer()
        op = ConfigCaptureChunkOp("capture")
        async def run_inline(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        op._run_cpu = run_inline
        configs = {"buffer_handle": buffer, "chunk_rows": 1, "marker": "ok"}

        asyncio.run(op._async_execute(configs))

        self.assertIs(op._configs, configs)
        self.assertEqual(op.init_marker, "ok")
        self.assertEqual(op.run_marker, "ok")
        self.assertEqual(op.wrap_marker, "ok")
        self.assertTrue(buffer.cleaned)

    def test_threshold_max_iterator_routes_noop_morphology_through_custom_kernel(self) -> None:
        class FakeFrameBuffer:
            def __init__(self, items):
                self._items = items
                self.cleaned = False

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

            async def iter_prefetch(self, start=0, stop=None):
                for item in self._items[start:stop]:
                    yield item

            def cleanup(self):
                self.cleaned = True

        fgp_total = FastGaussianParam(
            np.array([[5, 8], [4, 10]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        frame_buffer = FakeFrameBuffer(
            [
                (np.array([[6, 12], [4, 15]], dtype=np.uint16), 0.5),
                (np.array([[4, 9], [9, 12]], dtype=np.uint16), 1.0),
            ]
        )
        op = ThresholdMaxIteratorOp("threshold_max")
        outputs = {}

        async def run_case() -> None:
            with mock.patch.object(
                noise_equalization,
                "custom_threshold_max_merge",
                wraps=max_ops.threshold_max_merge_numpy,
            ) as patched_custom:
                async def run_cpu(fn, *args, **kwargs):
                    return fn(*args, **kwargs)

                async def capture(payload):
                    outputs.update(payload)

                op._run_cpu = run_cpu
                op._broadcast_outputs = capture
                await op._async_execute(
                    {
                        "fgp_total": fgp_total,
                        "buffer_handle": frame_buffer,
                        "n_sigma": 2.0,
                        "morph_kernel_size": 1,
                    }
                )
                self.assertEqual(patched_custom.call_count, len(frame_buffer))

        asyncio.run(run_case())
        self.assertTrue(frame_buffer.cleaned)
        self.assertIn("result", outputs)

    def test_custom_ops_available_returns_bool(self) -> None:
        self.assertIsInstance(custom_ops_available(), bool)

    def test_build_info_returns_minimal_metadata(self) -> None:
        info = build_info()
        self.assertIsInstance(info, dict)
        self.assertIn("available", info)

    def test_build_info_reports_fallback_backend(self) -> None:
        with mock.patch.object(max_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
            max_ops._select_max_backend.cache_clear()
            info = max_ops.build_info()

        self.assertFalse(info["available"])
        self.assertEqual(info["backend"], "numpy")

    def test_max_combine_can_force_numpy_fallback(self) -> None:
        base = np.array([[1, 5], [3, 4]], dtype=np.uint16)
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(max_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                max_ops._select_max_backend.cache_clear()
                got = max_ops.max_combine(base, fresh)

        expected = np.maximum(np.array([[1, 5], [3, 4]], dtype=np.uint16), fresh)
        self.assertIs(got, base)
        np.testing.assert_array_equal(base, expected)
        np.testing.assert_array_equal(got, expected)

    def test_build_info_includes_thread_policy(self) -> None:
        info = build_info()
        self.assertIn("thread_policy", info)

    def test_backend_registry_exposes_registered_candidates(self) -> None:
        candidates = backend_registry.registered_backend_candidates("median_reduce_chunk")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].logical_op, "median_reduce_chunk")
        self.assertEqual(candidates[0].backend, "openmp_cpu")
        self.assertEqual(candidates[0].kernel_name, "median_reduce_chunk")

        filter_candidates = backend_registry.registered_backend_candidates("median_filter_2d")
        self.assertEqual(len(filter_candidates), 1)
        self.assertEqual(filter_candidates[0].backend, "openmp_cpu")
        self.assertEqual(filter_candidates[0].kernel_name, "median_filter_2d")

        feature_candidates = backend_registry.registered_backend_candidates("extract_point_features")
        self.assertEqual(len(feature_candidates), 1)
        self.assertEqual(feature_candidates[0].backend, "openmp_cpu")
        self.assertEqual(feature_candidates[0].kernel_name, "extract_point_features")

        match_candidates = backend_registry.registered_backend_candidates("find_initial_match")
        self.assertEqual(len(match_candidates), 1)
        self.assertEqual(match_candidates[0].backend, "openmp_cpu")
        self.assertEqual(match_candidates[0].kernel_name, "find_initial_match")

        remap_candidates = backend_registry.registered_backend_candidates("camera_model_remap")
        self.assertTrue(
            any(candidate.backend == "cuda_host_io" and candidate.kernel_name == "camera_model_remap"
                for candidate in remap_candidates))
        self.assertTrue(
            any(candidate.backend == "openmp_cpu" and candidate.kernel_name == "camera_model_remap_cpu"
                for candidate in remap_candidates))

        huber_chunk_candidates = backend_registry.registered_backend_candidates(
            "huber_weighted_chunk")
        self.assertEqual(len(huber_chunk_candidates), 1)
        self.assertEqual(huber_chunk_candidates[0].backend, "cuda_host_io")
        self.assertEqual(
            huber_chunk_candidates[0].kernel_name, "huber_weighted_chunk_cuda")

    def test_backend_registry_reports_missing_compiled_module(self) -> None:
        selection = backend_registry.select_backend(
            "median_reduce_chunk",
            load_module=lambda: (None, "mock import error"),
        )

        self.assertFalse(selection.native)
        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.reason, "mock import error")

    def test_backend_registry_selects_native_kernel(self) -> None:
        class Module:
            pass

        module = Module()
        module.median_reduce_chunk = lambda stack: stack

        selection = backend_registry.select_backend(
            "median_reduce_chunk",
            load_module=lambda: (module, None),
        )

        self.assertTrue(selection.native)
        self.assertIs(selection.module, module)
        self.assertEqual(selection.backend, "openmp_cpu")

    def test_backend_registry_respects_build_flag(self) -> None:
        class Module:
            wavelet_dec_rec_cuda_core = lambda self: None

            def build_info(self):
                return {"cuda": False}

        module = Module()

        selection = backend_registry.select_backend(
            "wavelet_dec_rec_cuda_core",
            load_module=lambda: (module, None),
        )

        self.assertFalse(selection.native)
        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.reason, "compiled backend missing build flag: cuda")

    def test_backend_registry_selects_remap_cpu_after_cuda_build_flag_miss(self) -> None:
        class Module:
            camera_model_remap = lambda self: None
            camera_model_remap_cpu = lambda self: None

            def build_info(self):
                return {"cuda": False}

        module = Module()

        selection = backend_registry.select_backend(
            "camera_model_remap",
            load_module=lambda: (module, None),
        )

        self.assertTrue(selection.native)
        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.candidate.kernel_name, "camera_model_remap_cpu")

    def test_backend_registry_continues_after_build_flag_miss(self) -> None:
        class Module:
            wavelet_dec_rec_cuda_core = lambda self: None
            wavelet_dec_rec_cpu = lambda self: None

            def build_info(self):
                return {"cuda": False}

        module = Module()

        selection = backend_registry.select_backend(
            "wavelet_dec_rec",
            load_module=lambda: (module, None),
        )

        self.assertTrue(selection.native)
        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.candidate.kernel_name, "wavelet_dec_rec_cpu")

    def test_fgp_masked_mean_merge_can_force_numpy_fallback(self) -> None:
        img = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            dtype=np.uint16,
        )
        mask = np.array([[True, False], [True, True]])
        sum_mu = np.zeros_like(img, dtype=np.uint32)
        square_sum = np.zeros_like(img, dtype=np.uint64)
        count = np.zeros_like(img, dtype=np.uint16)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                fgp_masked_mean_merge(img, mask, sum_mu, square_sum, count)

        expected_mask = np.broadcast_to(mask[..., None], img.shape).astype(np.uint16)
        np.testing.assert_array_equal(sum_mu, img.astype(np.uint32) * expected_mask)
        np.testing.assert_array_equal(
            square_sum,
            np.square(img, dtype=np.uint64) * expected_mask.astype(np.uint64),
        )
        np.testing.assert_array_equal(count, expected_mask)

    def test_sigma_clip_fused_merge_can_force_numpy_fallback(self) -> None:
        img = np.array([[[1], [8]], [[5], [2]]], dtype=np.uint16)
        rej_high = np.array([[[3], [6]], [[7], [4]]], dtype=np.uint16)
        rej_low = np.array([[[0], [2]], [[4], [3]]], dtype=np.uint16)
        sum_mu = np.zeros_like(img, dtype=np.uint32)
        square_sum = np.zeros_like(img, dtype=np.uint64)
        count = np.zeros_like(img, dtype=np.uint16)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                sigma_clip_fused_merge(img, rej_high, rej_low, sum_mu, square_sum, count)

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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                sigma_clip_fused_masked_merge(
                    img,
                    mask,
                    rej_high,
                    rej_low,
                    sum_mu,
                    square_sum,
                    count,
                )

        rejected = (mask[..., None] & ((img < rej_low) | (img > rej_high))).astype(np.uint16)
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
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5)
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk_numpy(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5)

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
        total_sq = (stack_f64 ** 2 * mask_f64).sum(axis=0)
        total_n = mask_f64.sum(axis=0)

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5, mask=mask)
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk_numpy(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5, mask=mask)

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
        total_sq = (stack_f64 ** 2 * mask_f64).sum(axis=0)
        total_n = mask_f64.sum(axis=0)

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5, mask=mask)
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk_numpy(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5, mask=mask)

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
        stack = rng.integers(
            90, 120, size=(n_frames, plane_size), dtype=np.uint16)
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
        total_sq = (stack_f64 ** 2 * active_f64).sum(axis=0)
        total_n = active_f64.sum(axis=0)

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5,
            mask=mask, skip_zero_rgb=True, channels=channels)
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_iterative_chunk_numpy(
            stack, total_sum, total_sq, total_n, 3.0, 3.0, 5,
            mask=mask, skip_zero_rgb=True, channels=channels)

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
            stack, 3.0, 3.0, 5, mask=mask)
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(
            stack, 3.0, 3.0, 5, mask=mask)

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        self.assertEqual(c_n[8], 0.0)
        self.assertLess(c_n[4], 25.0)

    def test_sigma_clip_fused_chunk_can_force_numpy_fallback(self) -> None:
        """Fallback preference bypasses native fused chunk backends."""
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        expected = sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(stack)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "sigma_clip_fused_chunk_compiled",
                    side_effect=AssertionError("native backend should not be called")):
                with mock.patch.object(
                        sigma_clip_chunk_ops,
                        "sigma_clip_fused_chunk_compiled_cuda",
                        side_effect=AssertionError("native backend should not be called")):
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
        stack = rng.integers(
            80, 130, size=(n_frames, plane_size), dtype=np.uint16)
        stack[:, 9:12] = 0
        stack[2, 30:33] = 0
        stack[5, 44] = 700
        mask = (rng.random((n_frames, plane_size)) > 0.15).astype(np.uint8)

        c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk(
            stack, 3.0, 3.0, 5, mask=mask,
            skip_zero_rgb=True, channels=channels)
        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(
            stack, 3.0, 3.0, 5, mask=mask,
            skip_zero_rgb=True, channels=channels)

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        np.testing.assert_allclose(c_sq, n_sq, rtol=1e-10)
        np.testing.assert_array_equal(c_n[9:12], np.zeros(3))

    def test_sigma_clip_fused_chunk_skip_zero_rgb_requires_full_pixels(self) -> None:
        """RGB zero-skip requires flattened data to contain complete pixels."""
        stack = np.arange(40, dtype=np.uint16).reshape(4, 10)
        with self.assertRaisesRegex(ValueError, "divisible by channels"):
            sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(
                stack, skip_zero_rgb=True, channels=3)
        with self.assertRaisesRegex(ValueError, "divisible by channels"):
            sigma_clip_chunk_ops.sigma_clip_fused_chunk_compiled(
                stack, skip_zero_rgb=True, channels=3)

    def test_sigma_clip_fused_chunk_cuda_requires_full_pixels(self) -> None:
        """CUDA wrapper validates RGB zero-skip shape before launching kernels."""
        if not build_info().get("cuda"):
            self.skipTest("CUDA sigma clip backend is not built")
        stack = np.arange(40, dtype=np.uint16).reshape(4, 10)

        with self.assertRaisesRegex(ValueError, "divisible by channels"):
            sigma_clip_chunk_ops.sigma_clip_fused_chunk_compiled_cuda(
                stack, skip_zero_rgb=True, channels=3)

    def test_sigma_clip_fused_chunk_cuda_matches_numpy(self) -> None:
        """CUDA fused chunk backend matches numpy when a CUDA device is available."""
        if not build_info().get("cuda"):
            self.skipTest("CUDA sigma clip backend is not built")

        rng = np.random.default_rng(987)
        n_frames = 18
        spatial = 32
        channels = 3
        plane_size = spatial * channels
        stack = rng.integers(
            80, 140, size=(n_frames, plane_size), dtype=np.uint16)
        stack[0, 12] = 900
        stack[:, 21:24] = 0
        mask = (rng.random((n_frames, plane_size)) > 0.2).astype(np.uint8)

        try:
            c_sum, c_sq, c_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk_compiled_cuda(
                stack, 3.0, 3.0, 5, mask=mask,
                skip_zero_rgb=True, channels=channels)
        except RuntimeError as exc:
            if is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA sigma clip runtime unavailable: {exc}")
            raise

        n_sum, n_sq, n_n = sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy(
            stack, 3.0, 3.0, 5, mask=mask,
            skip_zero_rgb=True, channels=channels)

        np.testing.assert_array_equal(c_n, n_n)
        np.testing.assert_allclose(c_sum, n_sum, rtol=1e-10)
        np.testing.assert_allclose(c_sq, n_sq, rtol=1e-10)

    def test_sigma_clip_fused_chunk_cuda_runtime_falls_back_to_cpu(self) -> None:
        """Public fused dispatch falls back to CPU when CUDA runtime is unavailable."""
        stack = np.arange(24, dtype=np.uint16).reshape(4, 6)
        expected = tuple(np.full(6, value, dtype=np.float64)
                         for value in (1.0, 2.0, 3.0))
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

        sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()
        with mock.patch.object(
                sigma_clip_chunk_ops,
                "_select_backend",
                return_value=cuda_selection):
            with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "sigma_clip_fused_chunk_compiled_cuda",
                    side_effect=RuntimeError(
                        "sigma_clip_fused_chunk_cuda cudaGetDevice: no CUDA-capable device is detected")):
                sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()
                with mock.patch.object(
                        sigma_clip_chunk_ops,
                        "sigma_clip_fused_chunk_compiled",
                        return_value=expected) as cpu_fallback:
                    got = sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)
        sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()

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

        sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()
        with mock.patch.object(
                sigma_clip_chunk_ops,
                "_select_backend",
                return_value=cpu_selection):
            with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "sigma_clip_fused_chunk_compiled",
                    side_effect=RuntimeError("native CPU bug")):
                with mock.patch.object(
                        sigma_clip_chunk_ops,
                        "sigma_clip_fused_chunk_numpy",
                        side_effect=AssertionError("numpy backend should not be called")):
                    sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()
                    with self.assertRaisesRegex(RuntimeError, "native CPU bug"):
                        sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)
        sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()

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

        sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()
        with mock.patch.object(
                sigma_clip_chunk_ops,
                "_select_backend",
                return_value=cuda_selection):
            with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "sigma_clip_fused_chunk_compiled_cuda",
                    side_effect=RuntimeError(
                        "sigma_clip_fused_chunk_cuda cudaGetDevice: no CUDA-capable device is detected")):
                sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()
                with mock.patch.object(
                        sigma_clip_chunk_ops,
                        "sigma_clip_fused_chunk_compiled",
                        side_effect=RuntimeError("compiled CPU backend is unavailable")):
                    got = sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)
        sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()

        for actual, expected_arr in zip(got, expected, strict=True):
            np.testing.assert_array_equal(actual, expected_arr)

    def test_sigma_clip_fused_chunk_cuda_fallback_propagates_cpu_value_error(self) -> None:
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

        sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()
        with mock.patch.object(
                sigma_clip_chunk_ops,
                "_select_backend",
                return_value=cuda_selection):
            with mock.patch.object(
                    sigma_clip_chunk_ops,
                    "sigma_clip_fused_chunk_compiled_cuda",
                    side_effect=RuntimeError(
                        "sigma_clip_fused_chunk_cuda cudaGetDevice: no CUDA-capable device is detected")):
                sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()
                with mock.patch.object(
                        sigma_clip_chunk_ops,
                        "sigma_clip_fused_chunk_compiled",
                        side_effect=ValueError("bad CPU fallback input")):
                    with mock.patch.object(
                            sigma_clip_chunk_ops,
                            "sigma_clip_fused_chunk_numpy",
                            side_effect=AssertionError("numpy backend should not be called")):
                        with self.assertRaisesRegex(ValueError, "bad CPU fallback input"):
                            sigma_clip_chunk_ops.sigma_clip_fused_chunk(stack)
        sigma_clip_chunk_ops._select_sigma_clip_fused_chunk_backend.cache_clear()


if __name__ == "__main__":
    unittest.main()
