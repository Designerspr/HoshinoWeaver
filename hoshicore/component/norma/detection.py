"""Star point detection from images."""
import dataclasses

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from hoshicore._custom_op._dispatch import CustomOpUnavailableError
from hoshicore._custom_op.ops.detection import (
    star_detect_bandpass_threshold_morph_numpy,
    star_detect_full_connected_components,
)
from hoshicore._custom_op.ops.wavelet import wavelet_dec_rec


@dataclasses.dataclass
class DetectedStars:
    positions: NDArray[np.float64]
    volumes: NDArray[np.float64]


FULL_GPU_COMPONENT_FILTER_PERCENTILE = 22.5


def _wavelet_dec_rec(img_blr, resize_factor=0.25):
    return wavelet_dec_rec(img_blr, resize_factor=resize_factor)


def _empty_detected_stars() -> DetectedStars:
    return DetectedStars(
        positions=np.empty((0, 2), dtype=np.float64),
        volumes=np.empty((0,), dtype=np.float64),
    )


def _normalize_gray(img_gray: NDArray) -> NDArray[np.float64]:
    if img_gray.dtype == np.float64:
        return img_gray
    if np.issubdtype(img_gray.dtype, np.integer):
        return img_gray.astype(np.float64) / np.iinfo(img_gray.dtype).max
    return img_gray.astype(np.float64)


def _prepare_detection_inputs(
    img_gray: NDArray,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float]:
    img_shape = img_gray.shape
    img_gray = _normalize_gray(img_gray)

    img_blr = cv2.GaussianBlur(img_gray, (gaussian_ksize, gaussian_ksize),
                               sigma)
    img_blr_mean = np.mean(img_blr)
    img_blr_range = np.max(img_blr) - np.min(img_blr)
    img_blr = (img_blr - img_blr_mean) / img_blr_range

    resize_factor = 1
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2

    logger.debug("Mask logical selection")
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
        mask = tmp_mask > 127
    else:
        mask = np.logical_and(tmp_mask > 127, mask > 0)
    logger.debug("Mask calculation Complete")
    mask_rate = np.sum(mask) * 100.0 / np.prod(mask.shape)
    logger.debug(f"mask rate: {mask_rate:.2f}")
    if mask_rate < 50:
        mask = np.ones(tmp_mask.shape, dtype="bool")
    return img_blr, mask, resize_factor


def _component_candidates_to_detected(
    positions: NDArray[np.float64],
    areas: NDArray[np.float64],
    intensities: NDArray[np.float64],
    eccentricities: NDArray[np.float64],
) -> DetectedStars:
    if len(positions) == 0:
        return _empty_detected_stars()
    valid_stars = np.logical_and(areas > 20, eccentricities < .8)
    # GPU CC 会比 contour 路径多保留少量弱候选；略收紧分位过滤可稳定下游匹配。
    filter_percentile = FULL_GPU_COMPONENT_FILTER_PERCENTILE
    valid_stars = np.logical_and(
        np.logical_and(
            valid_stars, areas > np.percentile(areas, filter_percentile)),
        intensities > np.percentile(intensities, filter_percentile),
    )
    return DetectedStars(
        positions=positions[valid_stars],
        volumes=areas[valid_stars] * intensities[valid_stars],
    )


def _detect_star_points_full_gpu(
    img_gray: NDArray,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
    min_star_points: int = 400,
) -> DetectedStars:
    img_shape = img_gray.shape
    img_gray = _normalize_gray(img_gray)

    resize_factor = 1.0
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2.0

    while True:
        positions, areas, intensities, eccentricities = (
            star_detect_full_connected_components(
                img_gray,
                mask,
                resize_factor,
                gaussian_ksize=gaussian_ksize,
                sigma=sigma,
            )
        )
        candidate_count = len(positions)
        logger.debug(f"{candidate_count} full GPU star candidates detected")
        if candidate_count < min_star_points and resize_factor < 1:
            logger.debug(
                "Not enough points, resize factor is now increasing by 2")
            resize_factor *= 2
            continue
        break

    if candidate_count < min_star_points:
        logger.warning(
            f"Not enough points: expected {min_star_points}, got {candidate_count}"
        )
    logger.debug(f"final resize factor = {resize_factor:.3f}")

    detected = _component_candidates_to_detected(
        positions, areas, intensities, eccentricities)
    logger.debug(f"Final star points = {len(detected.positions)}")
    return detected


def _detect_star_points_opencv(
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

    return _detect_star_points_opencv_prepared(
        img_blr, detection_mask, resize_factor, min_star_points=min_star_points)


def _detect_star_points_opencv_prepared(
    img_blr: NDArray[np.float64],
    detection_mask: NDArray[np.bool_],
    resize_factor: float,
    *,
    min_star_points: int = 400,
) -> DetectedStars:
    while True:
        # Production fallback 必须保持纯 CPU/OpenCV，不经过 staged GPU dispatch。
        img_rec, bw = star_detect_bandpass_threshold_morph_numpy(
            img_blr, detection_mask, resize_factor)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)
        contours = [contour for contour in contours if len(contour) > 5]
        logger.debug(f"{len(contours)} star points detected")
        if len(contours) < min_star_points and resize_factor < 1:
            logger.debug(
                "Not enough points, resize factor is now increasing by 2")
            resize_factor *= 2
            continue
        else:
            break

    if len(contours) < min_star_points:
        logger.warning(
            f"Not enough points: expected {min_star_points}, got {len(contours)}"
        )
    logger.debug(f"final resize factor = {resize_factor:.3f}")

    elps = [cv2.fitEllipse(contour) for contour in contours]
    centroids = np.array([e[0] for e in elps])
    areas = np.array([
        cv2.contourArea(contour) + 0.5 * len(contour) for contour in contours
    ])
    eccentricities = np.sqrt(
        np.array([1 - (elp[1][0] / elp[1][1])**2 for elp in elps]))

    mask_img = np.zeros(bw.shape, np.uint8)
    intensities = np.zeros(areas.shape)
    for i in range(len(contours)):
        cv2.drawContours(mask_img, contours, i, 255, -1)
        rect = cv2.boundingRect(contours[i])
        val = cv2.mean(
            img_rec[rect[1]:rect[1] + rect[3] + 1,
                    rect[0]:rect[0] + rect[2] + 1],
            mask_img[rect[1]:rect[1] + rect[3] + 1,
                     rect[0]:rect[0] + rect[2] + 1])
        mask_img[rect[1]:rect[1] + rect[3] + 1,
                 rect[0]:rect[0] + rect[2] + 1] = 0
        intensities[i] = val[0]

    valid_stars = np.logical_and(areas > 20, eccentricities < .8)
    valid_stars = np.logical_and(
        np.logical_and(valid_stars, areas > np.percentile(areas, 20)),
        intensities > np.percentile(intensities, 20)
    )

    star_pts = centroids[valid_stars]
    areas = areas[valid_stars]
    intensities = intensities[valid_stars]
    logger.debug(f"Final star points = {len(star_pts)}")

    return DetectedStars(positions=star_pts, volumes=areas * intensities)


def detect_star_points(
    img_gray: NDArray,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
    min_star_points: int = 400,
) -> DetectedStars:
    try:
        return _detect_star_points_full_gpu(
            img_gray,
            mask=mask,
            resize_length=resize_length,
            gaussian_ksize=gaussian_ksize,
            sigma=sigma,
            min_star_points=min_star_points,
        )
    except CustomOpUnavailableError as exc:
        # Full GPU detector 不可用时，生产 fallback 必须回到原始 contour/OpenCV 路径。
        logger.debug(
            f"Full GPU star detector unavailable, falling back to contour detector: {exc}"
        )
        return _detect_star_points_opencv(
            img_gray,
            mask=mask,
            resize_length=resize_length,
            gaussian_ksize=gaussian_ksize,
            sigma=sigma,
            min_star_points=min_star_points,
        )
