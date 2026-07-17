"""Star point detection from images."""
import dataclasses

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from hoshicore._custom_op import wavelet_dec_rec
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import CustomOpUnavailableError
from hoshicore._custom_op.ops.detection import (
    StarDetectCapacityError,
    star_detect_fused_pixel_components,
)

MIN_STAR_AREA = 10
STAR_FILTER_PERCENTILE = 10


@dataclasses.dataclass
class DetectedStars:
    positions: NDArray[np.float64]
    volumes: NDArray[np.float64]


class _CudaHybridGeometryMismatch(RuntimeError):
    """CUDA component summaries cannot be mapped to exact host contours."""

    pass


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


def _wavelet_dec_rec(img_blr, resize_factor=0.25):
    return wavelet_dec_rec(img_blr, resize_factor=resize_factor)


def _bandpass_dog(img_blr: np.ndarray, resize_factor: float = 0.25) -> np.ndarray:
    h, w = img_blr.shape
    small = cv2.resize(img_blr, None, fx=resize_factor, fy=resize_factor,
                       interpolation=cv2.INTER_AREA)
    diag_len = (h**2+w**2)**(1/2)
    # 假设原图最大星点 20px in 7000px
    fine_width = diag_len * 0.0001
    coarse_width = diag_len * 0.001
    # 粗尺度去背景，细尺度去噪，差值保留星点
    coarse = cv2.GaussianBlur(small, (0, 0), sigmaX=coarse_width)
    fine   = cv2.GaussianBlur(small, (0, 0), sigmaX=fine_width)
    dog = coarse - fine
    return cv2.resize(dog, (w, h), interpolation=cv2.INTER_LINEAR)


def _find_star_contours(bw: NDArray[np.uint8]) -> list[NDArray[np.int32]]:
    contours, _ = cv2.findContours(
        bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [contour for contour in contours if len(contour) > 5]


def _filter_star_candidates(
    positions: NDArray[np.float64],
    areas: NDArray[np.float64],
    intensities: NDArray[np.float64],
    eccentricities: NDArray[np.float64],
) -> DetectedStars:
    if len(positions) == 0:
        return _empty_detected_stars()

    area_percentile = np.percentile(areas, STAR_FILTER_PERCENTILE)
    intensity_percentile = np.percentile(
        intensities, STAR_FILTER_PERCENTILE)
    valid_stars = np.logical_and(
        areas > MIN_STAR_AREA,
        eccentricities < .8,
    )
    valid_stars = np.logical_and(
        valid_stars,
        np.logical_and(
            areas > area_percentile,
            intensities > intensity_percentile,
        ),
    )
    return DetectedStars(
        positions=positions[valid_stars],
        volumes=areas[valid_stars] * intensities[valid_stars],
    )


def _measure_contour_candidates(
    contours: list[NDArray[np.int32]],
    img_rec: NDArray[np.float64],
    bw: NDArray[np.uint8],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    ellipses = [cv2.fitEllipse(contour) for contour in contours]
    positions = np.asarray([ellipse[0] for ellipse in ellipses])
    areas = np.asarray([
        cv2.contourArea(contour) + 0.5 * len(contour)
        for contour in contours
    ])
    eccentricities = np.sqrt(
        np.asarray([
            1 - (ellipse[1][0] / ellipse[1][1])**2
            for ellipse in ellipses
        ]))

    mask_img = np.zeros(bw.shape, np.uint8)
    intensities = np.zeros(areas.shape)
    for index, contour in enumerate(contours):
        cv2.drawContours(mask_img, contours, index, 255, -1)
        x, y, width, height = cv2.boundingRect(contour)
        roi = np.s_[y:y + height + 1, x:x + width + 1]
        intensities[index] = cv2.mean(img_rec[roi], mask_img[roi])[0]
        mask_img[roi] = 0
    return positions, areas, intensities, eccentricities


def _measure_cuda_hybrid_contour_candidates(
    component_positions: NDArray[np.float64],
    component_intensities: NDArray[np.float64],
    binary_mask: NDArray[np.uint8],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Recover exact contour geometry and attach CUDA component intensities."""
    contours = _find_star_contours(binary_mask)
    if not contours:
        empty = np.empty((0,), dtype=np.float64)
        return np.empty((0, 2), dtype=np.float64), empty, empty, empty

    ellipses = [cv2.fitEllipse(contour) for contour in contours]
    positions = np.asarray([ellipse[0] for ellipse in ellipses])
    if len(component_positions) < len(positions):
        raise _CudaHybridGeometryMismatch(
            "CUDA detector returned fewer components than contours")
    distances = np.linalg.norm(
        positions[:, None, :] - component_positions[None, :, :], axis=2)
    component_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[
        np.arange(len(positions)), component_indices]
    if (len(np.unique(component_indices)) != len(component_indices)
            or np.any(nearest_distances > 1.0)):
        raise _CudaHybridGeometryMismatch(
            "CUDA components cannot be mapped one-to-one to contours")

    areas = np.asarray([
        cv2.contourArea(contour) + 0.5 * len(contour)
        for contour in contours
    ])
    eccentricities = np.sqrt(
        np.asarray([
            1 - (ellipse[1][0] / ellipse[1][1])**2
            for ellipse in ellipses
        ]))
    intensities = component_intensities[component_indices]
    return positions, areas, intensities, eccentricities


def _detect_star_points_contour(
    img_gray: NDArray,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
    min_star_points: int = 400,
) -> DetectedStars:
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

    resize_factor = 1
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2

    logger.debug("Mask logical selection")
    if mask is None:
        mask = np.ones(img_shape, dtype=bool)
        logger.debug("Mask calculation Complete (auto mask disabled; using full image)")
    else:
        mask = mask > 0
        logger.debug("Mask calculation Complete (using external mask only)")
    mask_rate = np.sum(mask) * 100.0 / np.prod(mask.shape)
    logger.debug(f"mask rate: {mask_rate:.2f}")

    while True:
        img_rec = _wavelet_dec_rec(img_blr, resize_factor=resize_factor) * mask
        bw = ((img_rec > np.percentile(img_rec[mask], 99.5)) * mask).astype(
            np.uint8) * 255
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours = _find_star_contours(bw)
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

    if not contours:
        return _empty_detected_stars()

    candidates = _measure_contour_candidates(contours, img_rec, bw)
    detected = _filter_star_candidates(*candidates)
    logger.debug(f"Final star points = {len(detected.positions)}")
    return detected


def _detect_star_points_opencv(*args, **kwargs) -> DetectedStars:
    """Compatibility name for the production Python/OpenCV detector."""
    return _detect_star_points_contour(*args, **kwargs)


def _detect_star_points_cuda_hybrid(
    img_gray: NDArray,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
    min_star_points: int = 400,
) -> DetectedStars:
    """Run fused CUDA pixel processing with exact OpenCV contour geometry."""
    img_shape = img_gray.shape
    img_gray = _normalize_gray(img_gray)
    if np.ptp(img_gray) == 0:
        return _empty_detected_stars()
    resize_factor = 1.0
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2.0

    while True:
        component_positions, component_intensities, binary_mask = (
            star_detect_fused_pixel_components(
                img_gray,
                mask,
                resize_factor,
                gaussian_ksize=gaussian_ksize,
                sigma=sigma,
            ))
        candidates = _measure_cuda_hybrid_contour_candidates(
            component_positions, component_intensities, binary_mask)
        candidate_count = len(candidates[0])
        logger.debug(f"{candidate_count} CUDA hybrid star candidates detected")
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
    detected = _filter_star_candidates(*candidates)
    logger.debug(f"Final CUDA hybrid star points = {len(detected.positions)}")
    return detected


def detect_star_points(
    img_gray: NDArray,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
    min_star_points: int = 400,
) -> DetectedStars:
    """Detect stars with CUDA pixel work plus exact host contour geometry."""
    kwargs = {
        "mask": mask,
        "resize_length": resize_length,
        "gaussian_ksize": gaussian_ksize,
        "sigma": sigma,
        "min_star_points": min_star_points,
    }
    try:
        return _detect_star_points_cuda_hybrid(img_gray, **kwargs)
    except CustomOpUnavailableError as exc:
        logger.debug(
            "CUDA star detector unavailable; using contour fallback: {}", exc)
    except _CudaHybridGeometryMismatch as exc:
        logger.warning(
            "CUDA star detector geometry guard failed; using contour fallback: {}",
            exc,
        )
    except StarDetectCapacityError as exc:
        logger.warning(
            "CUDA star detector capacity guard reached; using contour fallback: {}",
            exc,
        )
    except CustomOpResourceExhaustedError as exc:
        logger.warning(
            "CUDA star detector resources exhausted; using contour fallback: {}",
            exc,
        )
    return _detect_star_points_contour(img_gray, **kwargs)
