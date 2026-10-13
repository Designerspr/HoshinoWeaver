"""Image detection caching and image-free star geometry."""
from functools import cached_property
from typing import Optional

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from hoshicore._custom_op.ops.detection import GraySource
from hoshicore._custom_op import detection_gray_f64, detection_gray_u16

from .detection import DetectedStars, detect_star_points, detect_star_points_median
from .matching import adaptive_k, extract_point_features
from .types import BaseCameraModel
from .wavelet_recovery import (WaveletRecoveryProbe,
                               detect_recovered_wavelet_stars,
                               probe_wavelet_recovery)


def to_gray_f64(arr: np.ndarray) -> NDArray[np.float64]:
    """Convert a project-convention BGR image to grayscale in ``[0, 1]``."""
    return detection_gray_f64(arr)


def to_median_gray_u16(arr: np.ndarray) -> NDArray[np.uint16] | None:
    """The median detector's uint16 gray of a uint16 frame, without float64.

    Bitwise equal to quantizing :func:`to_gray_f64` back to uint16: for every
    float32 gray ``g`` in ``[0, 65535]``, ``rint(float32(g / 65535) * 65535)``
    equals ``rint(g)`` (checked over all such float32 values). Returns None for
    other dtypes or a gray outside that range, which take the float64 path.
    """
    return detection_gray_u16(arr)


class StarDetectionCache:
    """One grayscale image with separately cached detector results.

    This object is intentionally image-backed. A long-lived reference cache
    avoids repeating detection across source frames; source caches are normally
    discarded after one alignment.

    Wavelet results are selected using the image's shape-filter statistics
    before returning observations to feature consumers. Sequence callers can
    disable this decision and lock a single profile across their frames.
    """

    def __init__(self, gray: Optional[NDArray[np.float64]] = None,
                 mask: Optional[np.ndarray] = None,
                 median_threshold_ratio: float = 1.0,
                 star_detection_mode: str = "auto", *,
                 image: Optional[np.ndarray] = None,
                 star_detection_recovery: bool = True):
        if (gray is None) == (image is None):
            raise ValueError("StarDetectionCache needs exactly one of gray or image")
        self._gray = gray
        self._image = image
        self._mask = mask
        self._median_threshold_ratio = median_threshold_ratio
        self._star_detection_mode = star_detection_mode
        self._star_detection_recovery = star_detection_recovery
        self.star_detection_profile = "strict"
        self.star_recovery_stats = None

    @classmethod
    def from_image(cls, image: np.ndarray, mask: Optional[np.ndarray] = None,
                   median_threshold_ratio: float = 1.0,
                   star_detection_mode: str = "auto", *,
                   star_detection_recovery: bool = True) -> "StarDetectionCache":
        return cls(mask=mask, median_threshold_ratio=median_threshold_ratio,
                   star_detection_mode=star_detection_mode, image=image,
                   star_detection_recovery=star_detection_recovery)

    @cached_property
    def gray(self) -> NDArray[np.float64]:
        """Host gray, converted from the image only when a detector needs it."""
        return self._gray if self._gray is not None else to_gray_f64(self._image)

    def _make_wavelet_source(self) -> NDArray[np.float64] | GraySource:
        # From an image, the CUDA detector converts the gray on the device.
        return (self.gray if self._image is None
                else GraySource(self._image, to_gray_f64, host_gray=lambda: self.gray))

    def probe_wavelet_recovery(self, *, only_if_needed: bool = False) -> WaveletRecoveryProbe:
        """Prepare observations and stats using this cache's input and backend.

        Sequence callers request both profiles and retain only the returned
        compact observations. Single-image callers may prepare recovery only
        when the image satisfies the gate.
        """
        return probe_wavelet_recovery(
            self._make_wavelet_source(), self._mask,
            mode=self._star_detection_mode, only_if_needed=only_if_needed)

    @cached_property
    def recovered_pywt_stars(self) -> DetectedStars:
        """Detect with the relaxed profile selected by a sequence caller."""
        return detect_recovered_wavelet_stars(self.gray, self._mask)

    @cached_property
    def pywt_stars(self) -> DetectedStars:
        if self._star_detection_recovery:
            probe = self.probe_wavelet_recovery(only_if_needed=True)
            self.star_recovery_stats = probe.stats
            self.star_detection_profile = "relaxed" if probe.stats.needs_recovery else "strict"
            logger.debug(
                "Wavelet recovery: image profile={} eligible={} strict={} recoverable={} fraction={:.4f}",
                self.star_detection_profile, probe.stats.eligible, probe.stats.strict,
                probe.stats.recoverable, probe.stats.recoverable_fraction)
            return probe.relaxed_stars if probe.stats.needs_recovery else probe.strict_stars
        return detect_star_points(self._make_wavelet_source(), self._mask,
                                  mode=self._star_detection_mode)

    @cached_property
    def median_stars(self) -> DetectedStars:
        gray = None if self._image is None else to_median_gray_u16(self._image)
        return detect_star_points_median(
            self.gray if gray is None else gray, self._mask,
            threshold_ratio=self._median_threshold_ratio)


class GeometryView:
    """Camera geometry and features for exactly one detected star set."""

    def __init__(self, stars: DetectedStars, camera: BaseCameraModel):
        self._stars = stars
        self._camera = camera
        self.img_shape = (camera.intrinsics.image_height_px,
                          camera.intrinsics.image_width_px)

    @property
    def stars(self) -> DetectedStars:
        return self._stars

    @property
    def camera(self) -> BaseCameraModel:
        return self._camera

    @property
    def positions(self) -> NDArray[np.float64]:
        return self._stars.positions

    @property
    def volumes(self) -> NDArray[np.float64]:
        return self._stars.volumes

    @property
    def intensities(self) -> NDArray[np.float64] | None:
        return self._stars.intensities

    @cached_property
    def unit_vectors(self) -> NDArray[np.float64]:
        return self._camera.unproject(self.positions)

    @cached_property
    def features(self) -> NDArray[np.float64]:
        k = adaptive_k(len(self.positions))
        return extract_point_features(self.unit_vectors, self.volumes, k=k)

    def with_camera(self, camera: BaseCameraModel) -> "GeometryView":
        return GeometryView(self._stars, camera)
