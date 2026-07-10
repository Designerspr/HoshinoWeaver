import unittest
from unittest import mock

import cv2
import numpy as np

from hoshicore._custom_op import build_info
from hoshicore._custom_op.backend_registry import registered_backend_candidates
from hoshicore._custom_op._dispatch import CustomOpUnavailableError
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
from hoshicore._custom_op.ops import detection as detection_ops
import hoshicore.component.norma.detection as star_detection


def _is_compiled_backend_unavailable(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        is_cuda_runtime_unavailable_error(exc)
        or "compiled custom op backend is unavailable" in message
        or "compiled cuda custom op backend is unavailable" in message
    )


def _threshold_morph_reference(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    mask_bool = mask > 0
    bw = ((image > np.percentile(image[mask_bool], 99.5)) * mask_bool).astype(
        np.uint8) * 255
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _connected_components_reference(image: np.ndarray, bw: np.ndarray):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (bw > 0).astype(np.uint8), 8)
    rows: list[tuple[float, float, float, float, float]] = []
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area <= 5:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels[y:y + h, x:x + w] == label
        yy, xx = np.nonzero(component)
        if len(xx) <= 5:
            continue

        cx, cy = centroids[label]
        gx = xx.astype(np.float64) + x
        gy = yy.astype(np.float64) + y
        dx = gx - cx
        dy = gy - cy
        cov_xx = float(np.mean(dx * dx))
        cov_yy = float(np.mean(dy * dy))
        cov_xy = float(np.mean(dx * dy))
        trace = cov_xx + cov_yy
        det_term = np.sqrt(max(0.0, (cov_xx - cov_yy)**2 + 4.0 * cov_xy**2))
        lambda_max = 0.5 * (trace + det_term)
        lambda_min = 0.5 * (trace - det_term)
        eccentricity = 0.0
        if lambda_max > 1e-12:
            eccentricity = float(
                np.sqrt(max(0.0, 1.0 - lambda_min / lambda_max)))
        intensity = float(np.mean(image[y:y + h, x:x + w][component]))
        rows.append((float(cx), float(cy), area, intensity, eccentricity))

    if not rows:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    values = np.array(rows, dtype=np.float64)
    return values[:, :2], values[:, 2], values[:, 3], values[:, 4]


def _sort_candidates(payload):
    positions, areas, intensities, eccentricities = payload
    order = np.lexsort((positions[:, 0], positions[:, 1]))
    return (
        positions[order],
        areas[order],
        intensities[order],
        eccentricities[order],
    )


class TestStarDetectCustomOps(unittest.TestCase):
    def tearDown(self) -> None:
        detection_ops._load_compiled_module_result.cache_clear()
        detection_ops._select_star_detect_connected_components_backend.cache_clear()
        detection_ops._select_star_detect_full_connected_components_backend.cache_clear()

    def test_star_detect_threshold_morph_numpy_matches_reference(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.normal(size=(37, 41)).astype(np.float64)
        mask = rng.random(size=image.shape) > 0.2

        expected = _threshold_morph_reference(image, mask)
        got = detection_ops.star_detect_threshold_morph_numpy(image, mask)

        np.testing.assert_array_equal(got, expected)

    def test_star_detect_bandpass_threshold_morph_numpy_matches_reference(
            self) -> None:
        rng = np.random.default_rng(20)
        image = rng.normal(size=(38, 41)).astype(np.float64)
        mask = rng.random(size=image.shape) > 0.2

        img_rec, bw = detection_ops.star_detect_bandpass_threshold_morph_numpy(
            image, mask, 1.0)

        expected_core = detection_ops.wavelet_dec_rec_core_numpy(
            image, detection_ops._wavelet_level(1.0))
        expected_img_rec = cv2.resize(
            expected_core, (image.shape[1], image.shape[0])) * mask
        expected_bw = detection_ops.star_detect_threshold_morph_numpy(
            expected_img_rec, mask)
        np.testing.assert_allclose(img_rec, expected_img_rec, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(bw, expected_bw)

    def test_staged_detection_backends_are_not_registered(self) -> None:
        for logical_op in (
                "star_detect_threshold_morph",
                "star_detect_bandpass_threshold_morph",
                "star_detect_bandpass_connected_components",
        ):
            with self.subTest(logical_op=logical_op):
                self.assertEqual(registered_backend_candidates(logical_op), ())

    def test_star_detect_full_connected_components_backend_registered(self) -> None:
        candidates = registered_backend_candidates(
            "star_detect_full_connected_components")
        self.assertTrue(
            any(candidate.kernel_name == "star_detect_full_connected_components_core"
                and candidate.backend == "cuda_host_io"
                and candidate.build_flag == "cuda"
                for candidate in candidates))

    def test_star_detect_connected_components_backend_registered(self) -> None:
        candidates = registered_backend_candidates(
            "star_detect_connected_components_candidates")
        self.assertTrue(
            any(candidate.kernel_name == "star_detect_connected_components_candidates"
                and candidate.backend == "openmp_cpu"
                for candidate in candidates))

    def test_star_detect_connected_components_reference_extracts_candidates(self) -> None:
        image = np.zeros((48, 64), dtype=np.float64)
        bw = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(bw, (12, 14), 3, 255, -1)
        cv2.circle(image, (12, 14), 3, 2.0, -1)
        cv2.rectangle(bw, (38, 30), (45, 36), 255, -1)
        cv2.rectangle(image, (38, 30), (45, 36), 4.0, -1)

        positions, areas, intensities, eccentricities = (
            _connected_components_reference(image, bw)
        )

        self.assertEqual(positions.shape, (2, 2))
        self.assertEqual(areas.shape, (2,))
        self.assertEqual(intensities.shape, (2,))
        self.assertEqual(eccentricities.shape, (2,))
        self.assertLess(
            np.max(np.min(
                np.linalg.norm(
                    positions[:, None, :] -
                    np.array([[12.0, 14.0], [41.5, 33.0]])[None, :, :],
                    axis=2,
                ),
                axis=1,
            )),
            0.1,
        )
        self.assertTrue(np.all(areas > 20))
        self.assertTrue(np.all(intensities > 0))

    def test_star_detect_connected_components_rejects_numpy_fallback(self) -> None:
        image = np.zeros((32, 32), dtype=np.float64)
        bw = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(bw, (16, 16), 4, 255, -1)
        cv2.circle(image, (16, 16), 4, 3.0, -1)

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"},
                clear=False):
            detection_ops._select_star_detect_connected_components_backend.cache_clear()
            with self.assertRaisesRegex(RuntimeError, "requires the compiled"):
                detection_ops.star_detect_connected_components_candidates(image, bw)

    def test_star_detect_connected_components_compiled_matches_numpy(self) -> None:
        image = np.zeros((72, 80), dtype=np.float64)
        bw = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(bw, (10, 12), 3, 255, -1)
        cv2.circle(image, (10, 12), 3, 2.0, -1)
        cv2.rectangle(bw, (30, 8), (38, 15), 255, -1)
        cv2.rectangle(image, (30, 8), (38, 15), 5.0, -1)
        cv2.circle(bw, (60, 50), 5, 255, -1)
        cv2.circle(image, (60, 50), 5, 7.0, -1)

        expected = _sort_candidates(_connected_components_reference(image, bw))
        try:
            got = _sort_candidates(
                detection_ops.star_detect_connected_components_candidates_compiled(
                    image, bw)
            )
        except RuntimeError as exc:
            if "compiled custom op backend is unavailable" not in str(exc).lower():
                raise
            self.skipTest(f"compiled backend unavailable: {exc}")

        for got_arr, expected_arr in zip(got, expected):
            np.testing.assert_allclose(got_arr, expected_arr, rtol=1e-12, atol=1e-12)

    def test_star_detect_full_connected_components_rejects_numpy_fallback(
            self) -> None:
        image = np.zeros((64, 64), dtype=np.float64)
        mask = np.ones(image.shape, dtype=np.uint8)

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"},
                clear=False):
            detection_ops._select_star_detect_full_connected_components_backend.cache_clear()
            with self.assertRaisesRegex(CustomOpUnavailableError, "requires the compiled CUDA"):
                detection_ops.star_detect_full_connected_components(
                    image, mask, 1.0)

    def test_star_detect_full_connected_components_wrapper_can_use_compiled_backend(
            self) -> None:
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        mask = np.zeros(image.shape, dtype=bool)
        mask[1:-1, 2:-2] = True
        expected = (
            np.array([[2.0, 3.0]], dtype=np.float64),
            np.array([7.0], dtype=np.float64),
            np.array([5.0], dtype=np.float64),
            np.array([0.1], dtype=np.float64),
        )

        compiled = mock.Mock(return_value=expected)
        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"},
                clear=False):
            with mock.patch.object(
                    detection_ops,
                    "_select_star_detect_full_connected_components_backend",
                    return_value=("compiled", compiled)):
                got = detection_ops.star_detect_full_connected_components(
                    image, mask, 0.5, gaussian_ksize=7, sigma=1.5)

        compiled.assert_called_once()
        args = compiled.call_args.args
        self.assertEqual(args[0].dtype, np.float64)
        np.testing.assert_allclose(args[0], image.astype(np.float64))
        self.assertEqual(args[1].dtype, np.uint8)
        np.testing.assert_array_equal(args[1], mask.astype(np.uint8))
        self.assertEqual(args[2], 0.5)
        self.assertEqual(args[3], 7)
        self.assertEqual(args[4], 1.5)
        self.assertIs(got, expected)

    def test_star_detect_full_connected_components_compiled_matches_opencv_on_synthetic(
            self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA full detector backend is not built")

        image = np.zeros((384, 384), dtype=np.float64)
        stars = [
            (80, 90, 7.0),
            (140, 260, 9.0),
            (220, 130, 8.0),
            (300, 300, 10.0),
        ]
        for x, y, value in stars:
            cv2.circle(image, (x, y), 5, value, -1)
            cv2.circle(image, (x, y), 9, value * 0.35, 1)
        image += np.linspace(0.0, 0.05, image.shape[1], dtype=np.float64)[None, :]

        try:
            candidates = detection_ops.star_detect_full_connected_components_compiled(
                image,
                None,
                1.0,
                gaussian_ksize=9,
                sigma=2.0,
            )
        except RuntimeError as exc:
            if _is_compiled_backend_unavailable(exc):
                self.skipTest(f"CUDA full detector runtime unavailable: {exc}")
            raise

        got = star_detection._component_candidates_to_detected(*candidates)
        expected = star_detection._detect_star_points_opencv(
            image,
            min_star_points=0,
            resize_length=10000,
            gaussian_ksize=9,
            sigma=2,
        )
        self.assertGreaterEqual(len(got.positions), 3)
        self.assertGreaterEqual(len(expected.positions), 3)

        distances = np.linalg.norm(
            expected.positions[:, None, :] - got.positions[None, :, :],
            axis=2,
        )
        nearest = np.min(distances, axis=1)
        self.assertLess(float(np.percentile(nearest, 95)), 1.5)

    def test_detect_star_points_uses_gpu_fused_path_by_default(self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)
        expected = star_detection.DetectedStars(
            positions=np.array([[1.0, 2.0]], dtype=np.float64),
            volumes=np.array([3.0], dtype=np.float64),
        )

        with mock.patch.object(
                star_detection,
                "_detect_star_points_full_gpu",
                return_value=expected) as full_gpu:
            with mock.patch.object(
                    star_detection, "_detect_star_points_opencv") as opencv:
                got = star_detection.detect_star_points(
                    image, min_star_points=0)

        full_gpu.assert_called_once_with(
            image,
            mask=None,
            resize_length=10000,
            gaussian_ksize=9,
            sigma=2,
            min_star_points=0,
        )
        opencv.assert_not_called()
        self.assertIs(got, expected)

    def test_detect_star_points_gpu_failure_falls_back_to_opencv_not_cpu_cc(
            self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)
        expected = star_detection.DetectedStars(
            positions=np.array([[4.0, 5.0]], dtype=np.float64),
            volumes=np.array([6.0], dtype=np.float64),
        )

        with mock.patch.object(
                star_detection,
                "_detect_star_points_full_gpu",
                side_effect=CustomOpUnavailableError("mock CUDA unavailable")) as full_gpu:
            with mock.patch.object(
                    star_detection,
                    "_detect_star_points_opencv",
                    return_value=expected) as opencv:
                with mock.patch.object(
                        detection_ops,
                        "star_detect_connected_components_candidates",
                        side_effect=AssertionError("CPU CC must not be production fallback")):
                    got = star_detection.detect_star_points(
                        image, min_star_points=0)

        full_gpu.assert_called_once_with(
            image,
            mask=None,
            resize_length=10000,
            gaussian_ksize=9,
            sigma=2,
            min_star_points=0,
        )
        opencv.assert_called_once_with(
            image,
            mask=None,
            resize_length=10000,
            gaussian_ksize=9,
            sigma=2,
            min_star_points=0,
        )
        self.assertIs(got, expected)

    def test_detect_star_points_gpu_kernel_error_propagates(self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)

        with mock.patch.object(
                star_detection,
                "_detect_star_points_full_gpu",
                side_effect=RuntimeError("kernel bug")):
            with mock.patch.object(
                    star_detection,
                    "_detect_star_points_opencv",
                    side_effect=AssertionError("opencv fallback should not be called")):
                with self.assertRaisesRegex(RuntimeError, "kernel bug"):
                    star_detection.detect_star_points(image, min_star_points=0)

    def test_opencv_prepared_detector_uses_cpu_bandpass_helper(self) -> None:
        img_blr = np.zeros((32, 32), dtype=np.float64)
        mask = np.ones(img_blr.shape, dtype=bool)
        img_rec = np.zeros_like(img_blr)
        bw = np.zeros(img_blr.shape, dtype=np.uint8)
        cv2.ellipse(bw, (16, 16), (5, 4), 0.0, 0.0, 360.0, 255, -1)
        cv2.ellipse(img_rec, (16, 16), (5, 4), 0.0, 0.0, 360.0, 10.0, -1)

        with mock.patch.object(
                star_detection,
                "star_detect_bandpass_threshold_morph_numpy",
                return_value=(img_rec, bw)) as cpu_helper:
            got = star_detection._detect_star_points_opencv_prepared(
                img_blr, mask, 1.0, min_star_points=0)

        cpu_helper.assert_called_once_with(img_blr, mask, 1.0)
        self.assertIsInstance(got, star_detection.DetectedStars)

    def test_star_detect_public_facade_exports_only_full_detector(self) -> None:
        import hoshicore._custom_op as custom_op
        import hoshicore._custom_op.api as api
        from hoshicore._custom_op import star_detect_full_connected_components
        from hoshicore._custom_op.api import (
            star_detect_full_connected_components as api_star_detect_full_connected_components,
        )

        self.assertIs(
            star_detect_full_connected_components,
            detection_ops.star_detect_full_connected_components,
        )
        self.assertIs(
            api_star_detect_full_connected_components,
            detection_ops.star_detect_full_connected_components,
        )
        for name in (
                "star_detect_threshold_morph",
                "star_detect_bandpass_threshold_morph",
                "star_detect_bandpass_connected_components",
                "star_detect_connected_components_candidates",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, custom_op.__all__)
                self.assertNotIn(name, api.__all__)
                self.assertFalse(hasattr(custom_op, name))
                self.assertFalse(hasattr(api, name))
