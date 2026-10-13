"""Wavelet recovery selected from candidate filtering statistics.

Only compact observations survive a probe. Contour probes share one wavelet
response; native probes retain the original strict backend and also prepare the
relaxed host profile. All frames ultimately use one fixed profile.
"""
import dataclasses
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from hoshicore._custom_op.ops.detection import GraySource

from . import detection as det

RECOVERY_PROBE_FRAMES = 3
RECOVERY_SHAPE_FRACTION = .25
RECOVERY_MIN_ELIGIBLE = 64
RECOVERY_MIN_RECOVERABLE = 32
RECOVERY_RESPONSE_PERCENTILE = 99.3
RECOVERY_MAX_ECCENTRICITY = .95


@dataclasses.dataclass(frozen=True)
class WaveletRecoveryStats:
    eligible: int
    strict: int
    recoverable: int

    @property
    def recoverable_fraction(self) -> float:
        return self.recoverable / self.eligible if self.eligible else 0.0

    @property
    def needs_recovery(self) -> bool:
        return (self.eligible >= RECOVERY_MIN_ELIGIBLE
                and self.recoverable >= RECOVERY_MIN_RECOVERABLE
                and self.recoverable_fraction >= RECOVERY_SHAPE_FRACTION)


@dataclasses.dataclass(frozen=True)
class WaveletRecoveryProbe:
    strict_stars: det.DetectedStars
    relaxed_stars: det.DetectedStars
    stats: WaveletRecoveryStats


def candidate_recovery_stats(
        areas: NDArray[np.floating], intensities: NDArray[np.floating],
        eccentricities: NDArray[np.floating]) -> WaveletRecoveryStats:
    """Count recoverable shape rejections conditional on other filters passing.

    Trails beyond the relaxed cap do not vote for relaxation. Natural sky-density
    differences and candidates already failing area/response do not enter the ratio.
    """
    if len(areas) == 0:
        return WaveletRecoveryStats(0, 0, 0)
    eligible = ((areas > det.MIN_STAR_AREA)
                & (areas > np.percentile(areas, det.STAR_FILTER_PERCENTILE))
                & (intensities > np.percentile(intensities,
                                               det.STAR_FILTER_PERCENTILE))
                & np.isfinite(eccentricities))
    strict = eligible & (eccentricities < .8)
    recoverable = eligible & (eccentricities >= .8) & (
        eccentricities < RECOVERY_MAX_ECCENTRICITY)
    return WaveletRecoveryStats(int(eligible.sum()), int(strict.sum()),
                                int(recoverable.sum()))


def select_wavelet_recovery(stats: Sequence[WaveletRecoveryStats]) -> str:
    """Require a strict majority of the first up to three frames to agree."""
    votes = sum(item.needs_recovery for item in stats)
    return "relaxed" if stats and votes > len(stats) // 2 else "strict"


def probe_wavelet_recovery(gray,
                           mask=None,
                           *,
                           resize_length=10000,
                           min_star_points=400,
                           mode="contour",
                           only_if_needed=False) -> WaveletRecoveryProbe:
    """Prepare compact observations, preserving the strict backend.

    Sequence votes need both profiles even for a frame that does not vote for
    recovery. Single-image callers set only_if_needed to avoid preparing the
    relaxed profile unless this image satisfies the gate.
    """
    if mode != "contour":
        measured = []
        strict = det.detect_star_points(
            gray,
            mask,
            mode=mode,
            resize_length=resize_length,
            min_star_points=min_star_points,
            candidate_observer=lambda candidates: measured.append(
                candidate_recovery_stats(*candidates[1:])))
        if not measured:
            return WaveletRecoveryProbe(strict, strict,
                                        WaveletRecoveryStats(0, 0, 0))
        if only_if_needed and not measured[-1].needs_recovery:
            return WaveletRecoveryProbe(strict, strict, measured[-1])
        host = gray.host_gray() if isinstance(gray, GraySource) else gray
        relaxed = detect_recovered_wavelet_stars(host, mask)
        return WaveletRecoveryProbe(strict, relaxed, measured[-1])
    if isinstance(gray, GraySource):
        gray = gray.host_gray()
    gray = det._normalize_gray(gray)
    valid = np.ones(gray.shape, bool) if mask is None else np.asarray(mask) > 0
    if valid.shape != gray.shape or not np.any(valid):
        raise ValueError(
            "wavelet recovery mask must match the image and select pixels")
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    image_range = np.ptp(blur)
    if image_range == 0:
        empty = det._empty_detected_stars()
        return WaveletRecoveryProbe(empty, empty,
                                    WaveletRecoveryStats(0, 0, 0))
    blur = (blur - blur.mean()) / image_range
    factor = 1.0
    while max(gray.shape) * factor > resize_length:
        factor /= 2
    kernel = np.ones((3, 3), np.uint8)

    def contours_at(response, percentile):
        threshold = np.percentile(response[valid], percentile)
        binary = ((response > threshold) & valid).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary, det._find_star_contours(binary)

    while True:
        response = det._wavelet_dec_rec(blur, resize_factor=factor) * valid
        binary, contours = contours_at(response, 99.5)
        if len(contours) < min_star_points and factor < 1:
            factor *= 2
            continue
        break
    if contours:
        candidates = det._measure_contour_candidates(contours, response,
                                                     binary)
        strict = det._filter_star_candidates(*candidates)
        stats = candidate_recovery_stats(*candidates[1:])
    else:
        strict = det._empty_detected_stars()
        stats = WaveletRecoveryStats(0, 0, 0)
    if only_if_needed and not stats.needs_recovery:
        return WaveletRecoveryProbe(strict, strict, stats)
    binary, contours = contours_at(response, RECOVERY_RESPONSE_PERCENTILE)
    relaxed = (det._filter_star_candidates(
        *det._measure_contour_candidates(contours, response, binary),
        max_eccentricity=RECOVERY_MAX_ECCENTRICITY)
               if contours else det._empty_detected_stars())
    return WaveletRecoveryProbe(strict, relaxed, stats)


def detect_recovered_wavelet_stars(gray, mask=None) -> det.DetectedStars:
    """Use the same relaxed profile throughout the remainder of the sequence."""
    return det._detect_star_points_contour(
        gray,
        mask,
        response_percentile=RECOVERY_RESPONSE_PERCENTILE,
        max_eccentricity=RECOVERY_MAX_ECCENTRICITY,
        resize_selection_percentile=99.5)
