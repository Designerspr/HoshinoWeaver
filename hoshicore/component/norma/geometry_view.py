"""GeometryView construction and cached star-geometry features.

This module owns the image-to-geometry boundary:

raw image -> grayscale -> detected stars -> camera unit rays -> match features
"""
from functools import cached_property
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from .detection import DetectedStars, detect_star_points, detect_star_points_median
from .intrinsics_from_exif import intrinsics_from_focal_equiv
from .matching import adaptive_k, extract_point_features
from .types import BaseCameraModel, CameraModel


def to_gray_f64(arr: np.ndarray) -> np.ndarray:
    """Convert an image array to float64 grayscale in the [0, 1] range."""
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr.astype(np.float32),
                            cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        gray = arr.astype(np.float64)

    if np.issubdtype(arr.dtype, np.integer):
        gray /= np.iinfo(arr.dtype).max
    else:
        max_val = gray.max()
        if max_val > 1.0:
            gray /= max_val

    return gray


class GeometryView:
    """Cached geometry view of one image under one camera model.

    The detected stars are camera-independent and are reused by with_camera().
    Unit vectors and local match features are camera-dependent and are cached
    per GeometryView instance.
    """

    def __init__(
        self,
        image_gray: NDArray[np.float64],
        camera: BaseCameraModel,
        mask: Optional[np.ndarray] = None,
        detected_stars: Optional[DetectedStars] = None,
        median_threshold_ratio: float = 1.0,
    ):
        self._image_gray = image_gray
        self._mask = mask
        self._camera = camera
        self._detected_stars_seed = detected_stars
        self._median_threshold_ratio = median_threshold_ratio
        self.img_shape = image_gray.shape

    @property
    def camera(self) -> BaseCameraModel:
        return self._camera

    @property
    def image_gray(self) -> NDArray[np.float64]:
        return self._image_gray

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    @cached_property
    def detected_stars(self) -> DetectedStars:
        if self._detected_stars_seed is not None:
            return self._detected_stars_seed
        return detect_star_points_median(
            self._image_gray,
            self._mask,
            threshold_ratio=self._median_threshold_ratio,
        )

    @cached_property
    def pywt_detected_stars(self) -> DetectedStars:
        return detect_star_points(self._image_gray, self._mask)

    @property
    def positions(self) -> NDArray[np.float64]:
        return self.detected_stars.positions

    @property
    def volumes(self) -> NDArray[np.float64]:
        return self.detected_stars.volumes

    @property
    def intensities(self) -> NDArray[np.float64] | None:
        return self.detected_stars.intensities

    @cached_property
    def unit_vectors(self) -> NDArray[np.float64]:
        return self._camera.unproject(self.positions)

    @cached_property
    def features(self) -> NDArray[np.float64]:
        k = adaptive_k(len(self.positions))
        return extract_point_features(self.unit_vectors, self.volumes, k=k)

    def with_camera(self, camera: BaseCameraModel) -> "GeometryView":
        """Create a new view with another camera while reusing star detection."""
        return GeometryView(
            self._image_gray,
            camera,
            mask=self._mask,
            detected_stars=self.detected_stars,
            median_threshold_ratio=self._median_threshold_ratio,
        )


def make_geometry(arr: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  camera: Optional[BaseCameraModel] = None,
                  fallback_focal_equiv_mm: float = 20.0,
                  median_threshold_ratio: float = 1.0) -> GeometryView:
    """Build a GeometryView from a raw image array.

    When camera is None, use a zero-distortion perspective fallback camera.
    """
    gray = to_gray_f64(arr)
    if camera is None:
        h, w = arr.shape[:2]
        camera = CameraModel(intrinsics=intrinsics_from_focal_equiv(
            fallback_focal_equiv_mm, w, h))
    return GeometryView(
        gray,
        camera,
        mask=mask,
        median_threshold_ratio=median_threshold_ratio,
    )
