from unittest import mock

import cv2
import numpy as np

import hoshicore.component.norma.detection as detection_module
from hoshicore.component.norma.detection import (
    _candidate_volumes_with_rescue,
    detect_star_points_median,
    detect_star_points_median_detailed,
)


def _make_star_image() -> tuple[np.ndarray, list[tuple[int, int]]]:
    image = np.zeros((160, 220), dtype=np.float64)
    centers = [
        (30, 30),
        (70, 35),
        (110, 45),
        (150, 55),
        (190, 70),
        (45, 115),
        (95, 120),
        (145, 125),
        (190, 130),
    ]
    for index, center in enumerate(centers):
        cv2.circle(
            image,
            center,
            2 + index % 3,
            0.35 + 0.06 * index,
            -1,
        )
    return image, centers


def test_median_detector_returns_norma_contract_and_honors_mask() -> None:
    image, _ = _make_star_image()
    mask = np.ones(image.shape, dtype=bool)
    mask[:, 175:] = False

    detected = detect_star_points_median(
        image,
        mask=mask,
        median_ksize=13,
        threshold_ratio=2.5,
        open_ksize=3,
        min_star_points=0,
    )

    assert detected.positions.ndim == 2
    assert detected.positions.shape[1] == 2
    assert detected.volumes.shape == (len(detected.positions),)
    assert len(detected.positions) >= 3
    assert np.all(detected.positions[:, 0] < 175)
    assert np.all(detected.volumes > 0)


def test_median_detector_handles_empty_foreground() -> None:
    detected = detect_star_points_median(
        np.zeros((32, 48), dtype=np.float64),
        median_ksize=5,
        min_star_points=0,
    )

    assert detected.positions.shape == (0, 2)
    assert detected.volumes.shape == (0,)


def test_median_detector_detailed_result_exposes_candidate_stages() -> None:
    image, _ = _make_star_image()

    details = detect_star_points_median_detailed(
        image,
        median_ksize=13,
        threshold_ratio=2.5,
        open_ksize=3,
        min_star_points=0,
    )

    candidate_count = len(details.candidate_positions)
    assert details.response.shape == image.shape
    assert details.star_mask.shape == image.shape
    assert details.star_mask.dtype == np.uint8
    assert details.threshold > 0
    assert details.areas.shape == (candidate_count,)
    assert details.intensities.shape == (candidate_count,)
    assert details.eccentricities.shape == (candidate_count,)
    assert details.strict_valid_stars.shape == (candidate_count,)
    assert details.rescued_stars.shape == (candidate_count,)
    assert details.valid_stars.shape == (candidate_count,)
    assert not np.any(details.rescued_stars)
    np.testing.assert_array_equal(details.valid_stars,
                                  details.strict_valid_stars)
    np.testing.assert_array_equal(
        details.detected_stars.positions,
        details.candidate_positions[details.valid_stars],
    )


def test_median_detector_detailed_result_reuses_mask_threshold() -> None:
    image = np.zeros((32, 48), dtype=np.float64)
    star_mask = np.zeros(image.shape, dtype=np.uint8)
    response = np.zeros(image.shape, dtype=np.float32)

    with mock.patch.object(
            detection_module,
            "median_star_mask",
            return_value=(star_mask, response, 0.125)):
        with mock.patch.object(
                detection_module.np,
                "std",
                side_effect=AssertionError("threshold must not be recomputed")):
            details = detect_star_points_median_detailed(
                image,
                median_ksize=13,
                min_star_points=0,
            )

    assert details.threshold == 0.125


def test_outer_rescue_disables_eccentricity_rejection_experimentally() -> None:
    image, _ = _make_star_image()
    details = detect_star_points_median_detailed(
        image,
        median_ksize=13,
        threshold_ratio=2.5,
        open_ksize=3,
        min_star_points=0,
        max_eccentricity=0.0,
        enable_outer_rescue=True,
    )

    assert not np.any(details.strict_valid_stars)
    assert np.any(details.rescued_stars)
    np.testing.assert_array_equal(details.valid_stars, details.rescued_stars)


def test_rescue_volume_is_positive_finite_and_downweighted() -> None:
    areas = np.array([20.0, 30.0])
    intensities = np.array([2.0, 4.0])
    eccentricities = np.array([0.80, 0.99])
    rescued = np.array([False, True])

    volumes = _candidate_volumes_with_rescue(
        areas, intensities, eccentricities, rescued, 0.85)
    raw = areas * intensities

    assert np.all(np.isfinite(volumes))
    assert np.all(volumes > 0)
    assert volumes[0] == raw[0]
    assert volumes[1] <= raw[1]
    assert volumes[1] == raw[1] * 0.15
