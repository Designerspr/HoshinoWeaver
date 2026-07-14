"""Alternative star detection backends for norma alignment."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from hoshicore._custom_op.ops.detection import (
    star_detect_bandpass_threshold_morph_numpy,
    star_detect_connected_components_candidates,
)

from .detection import DetectedStars, _detect_star_points_opencv


@dataclasses.dataclass
class ComponentStarCandidates:
    positions: NDArray[np.float64]
    areas: NDArray[np.float64]
    intensities: NDArray[np.float64]
    eccentricities: NDArray[np.float64]


def _empty_detected_stars() -> DetectedStars:
    return DetectedStars(
        positions=np.empty((0, 2), dtype=np.float64),
        volumes=np.empty((0,), dtype=np.float64),
    )


def _empty_candidates() -> ComponentStarCandidates:
    return ComponentStarCandidates(
        positions=np.empty((0, 2), dtype=np.float64),
        areas=np.empty((0,), dtype=np.float64),
        intensities=np.empty((0,), dtype=np.float64),
        eccentricities=np.empty((0,), dtype=np.float64),
    )


def extract_connected_component_candidates(
    img_rec: NDArray[np.float64],
    bw: NDArray[np.uint8],
    *,
    allow_python_fallback: bool = True,
) -> ComponentStarCandidates:
    try:
        positions, areas, intensities, eccentricities = (
            star_detect_connected_components_candidates(img_rec, bw)
        )
    except RuntimeError as exc:
        if not allow_python_fallback:
            raise
        logger.warning(
            "Compiled connected-component detector is unavailable; using "
            "the Python/OpenCV fallback: {}", exc)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (np.asarray(bw) > 0).astype(np.uint8), connectivity=8)
        positions_list = []
        areas_list = []
        intensities_list = []
        eccentricities_list = []
        for label in range(1, count):
            component = labels == label
            ys, xs = np.nonzero(component)
            if len(xs) == 0:
                continue
            positions_list.append(centroids[label])
            areas_list.append(float(stats[label, cv2.CC_STAT_AREA]))
            intensities_list.append(float(np.mean(img_rec[component])))
            centered = np.column_stack((xs - np.mean(xs), ys - np.mean(ys)))
            covariance = centered.T @ centered / max(len(centered), 1)
            eigenvalues = np.linalg.eigvalsh(covariance)
            major = float(max(eigenvalues[-1], 0.0))
            minor = float(max(eigenvalues[0], 0.0))
            eccentricities_list.append(
                0.0 if major <= 1e-12 else np.sqrt(max(0.0, 1.0 - minor / major)))
        positions = np.asarray(positions_list, dtype=np.float64).reshape(-1, 2)
        areas = np.asarray(areas_list, dtype=np.float64)
        intensities = np.asarray(intensities_list, dtype=np.float64)
        eccentricities = np.asarray(eccentricities_list, dtype=np.float64)
    return ComponentStarCandidates(
        positions=positions,
        areas=areas,
        intensities=intensities,
        eccentricities=eccentricities,
    )


def filter_component_star_candidates(
    candidates: ComponentStarCandidates,
    *,
    min_area: float = 10.0,
    max_eccentricity: float = 0.8,
) -> DetectedStars:
    if len(candidates.positions) == 0:
        return _empty_detected_stars()
    areas = candidates.areas
    eccentricities = candidates.eccentricities
    intensities = candidates.intensities
    valid_stars = np.logical_and(areas > min_area,
                                 eccentricities < max_eccentricity)
    valid_stars = np.logical_and(
        np.logical_and(valid_stars, areas > np.percentile(areas, 10)),
        intensities > np.percentile(intensities, 10),
    )
    return DetectedStars(
        positions=candidates.positions[valid_stars],
        volumes=areas[valid_stars] * intensities[valid_stars],
    )


def detect_stars_connected_components(
    img_rec: NDArray[np.float64],
    bw: NDArray[np.uint8],
    *,
    min_area: float = 10.0,
    max_eccentricity: float = 0.8,
) -> DetectedStars:
    candidates = extract_connected_component_candidates(img_rec, bw)
    return filter_component_star_candidates(
        candidates,
        min_area=min_area,
        max_eccentricity=max_eccentricity,
    )


def _prepare_detection_inputs(
    img_gray: NDArray,
    mask=None,
    resize_length: int = 10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float]:
    img_shape = img_gray.shape
    if img_gray.dtype != np.float64:
        if np.issubdtype(img_gray.dtype, np.integer):
            img_gray = img_gray.astype(np.float64) / np.iinfo(img_gray.dtype).max
        else:
            img_gray = img_gray.astype(np.float64)

    img_blr = cv2.GaussianBlur(img_gray, (gaussian_ksize, gaussian_ksize),
                               sigma)
    img_blr_mean = np.mean(img_blr)
    img_blr_range = np.max(img_blr) - np.min(img_blr)
    img_blr = (img_blr - img_blr_mean) / img_blr_range

    resize_factor = 1.0
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2.0

    tmp_mask = cv2.resize(img_gray, None, fx=resize_factor, fy=resize_factor)
    tmp_mask_10percent = np.percentile(tmp_mask, 10)
    tmp_mask = (tmp_mask < min(tmp_mask_10percent, 0.15)).astype(
        np.uint8) * 255

    dilate_size = int(max(img_shape) * 0.003 * resize_factor)
    tmp_mask = 255 - cv2.dilate(
        tmp_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (dilate_size, dilate_size)))
    tmp_mask = cv2.resize(tmp_mask, (img_shape[1], img_shape[0]))
    if mask is None:
        detection_mask = tmp_mask > 127
    else:
        detection_mask = np.logical_and(tmp_mask > 127, mask > 0)
    mask_rate = np.sum(detection_mask) * 100.0 / np.prod(detection_mask.shape)
    if mask_rate < 50:
        detection_mask = np.ones(tmp_mask.shape, dtype=bool)

    return img_blr, detection_mask, resize_factor


def detect_star_points_connected_components(
    img_gray: NDArray,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
    min_star_points: int = 400,
) -> DetectedStars:
    img_blr, detection_mask, resize_factor = _prepare_detection_inputs(
        img_gray,
        mask=mask,
        resize_length=resize_length,
        gaussian_ksize=gaussian_ksize,
        sigma=sigma,
    )

    while True:
        img_rec, bw = star_detect_bandpass_threshold_morph_numpy(
            img_blr, detection_mask, resize_factor)
        try:
            candidates = extract_connected_component_candidates(
                img_rec, bw, allow_python_fallback=False)
        except RuntimeError as exc:
            logger.warning(
                f"Experimental CC detector unavailable, falling back to contour detector: {exc}"
            )
            return _detect_star_points_opencv(
                img_gray,
                mask=mask,
                resize_length=resize_length,
                gaussian_ksize=gaussian_ksize,
                sigma=sigma,
                min_star_points=min_star_points,
            )
        detected = filter_component_star_candidates(candidates)
        candidate_count = len(candidates.positions)
        # 和原 contour 路径一致：重试依据候选区域数，而不是最终过滤后的星点数。
        if candidate_count < min_star_points and resize_factor < 1:
            resize_factor *= 2
            continue
        break

    if candidate_count < min_star_points:
        logger.warning(
            f"Not enough points: expected {min_star_points}, got {candidate_count}"
        )
    return detected
