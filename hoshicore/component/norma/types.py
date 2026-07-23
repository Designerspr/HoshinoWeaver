"""Immutable value types for the norma package."""
import abc
import dataclasses
from enum import Enum
from functools import cached_property
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from hoshicore._custom_op import \
    camera_model_remap as custom_camera_model_remap

from .projection import (make_intrinsic_matrix, project_fisheye_vectors,
                         project_vectors, unproject_fisheye_pixels,
                         unproject_pixels)
from .sky_model import altaz_to_radec, compute_parallactic_angle


class CoordSystem(Enum):
    ALTAZ = "altaz"
    RADEC = "radec"
    CAMERA = "camera"


@dataclasses.dataclass(frozen=True)
class BaseDistortion(abc.ABC):
    """Abstract base for distortion coefficient types."""

    @property
    @abc.abstractmethod
    def is_zero(self) -> bool:
        """Return True if all distortion coefficients are zero."""
        ...

    @abc.abstractmethod
    def to_opt_params(self, n: int) -> NDArray[np.float64]:
        """Return first n optimization parameters as a 1D array."""
        ...


@dataclasses.dataclass(frozen=True)
class Distortion(BaseDistortion):
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def is_zero(self) -> bool:
        return (self.k1 == 0 and self.k2 == 0 and self.k3 == 0 and self.p1 == 0
                and self.p2 == 0)

    def to_cv2(self) -> NDArray[np.float64]:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3],
                        dtype=np.float64)

    @classmethod
    def from_cv2(cls, arr) -> "Distortion":
        if len(arr) == 5:
            return cls(k1=arr[0], k2=arr[1], p1=arr[2], p2=arr[3], k3=arr[4])
        elif len(arr) == 4:
            return cls(k1=arr[0], k2=arr[1], p1=arr[2], p2=arr[3])
        elif len(arr) == 2:
            return cls(k1=arr[0], k2=arr[1])
        else:
            raise ValueError(f"Unexpected distortion array length: {len(arr)}")

    def to_opt_params(self, n: int) -> NDArray[np.float64]:
        """Return first n optimization parameters: [k1, k2, p1, p2][:n]."""
        return np.array([self.k1, self.k2, self.p1, self.p2],
                        dtype=np.float64)[:n]


@dataclasses.dataclass(frozen=True)
class FisheyeDistortion(BaseDistortion):
    """Kannala-Brandt fisheye distortion coefficients (cv2.fisheye model).

    Zero coefficients represent a pure equidistant lens (r = f·θ),
    which is the most common fisheye projection and a suitable default
    for self-calibration when the lens type is unknown.
    """
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0

    @property
    def is_zero(self) -> bool:
        return self.k1 == 0 and self.k2 == 0 and self.k3 == 0 and self.k4 == 0

    def to_cv2(self) -> NDArray[np.float64]:
        return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)

    def to_opt_params(self, n: int) -> NDArray[np.float64]:
        """Return first n optimization parameters: [k1, k2, k3, k4][:n]."""
        return np.array([self.k1, self.k2, self.k3, self.k4],
                        dtype=np.float64)[:n]

    @classmethod
    def from_array(cls, arr) -> "FisheyeDistortion":
        arr = list(arr) + [0.0] * 4
        return cls(k1=arr[0], k2=arr[1], k3=arr[2], k4=arr[3])


@dataclasses.dataclass(frozen=True)
class Intrinsics:
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int
    cx_px: Optional[float] = None
    cy_px: Optional[float] = None

    @cached_property
    def K(self) -> NDArray[np.float64]:
        return make_intrinsic_matrix(self.focal_length_mm,
                                     self.sensor_width_mm,
                                     self.sensor_height_mm,
                                     self.image_width_px,
                                     self.image_height_px,
                                     cx_px=self.cx_px,
                                     cy_px=self.cy_px)

    def with_focal_length(self, focal_length_mm: float) -> "Intrinsics":
        return dataclasses.replace(self, focal_length_mm=focal_length_mm)

    @property
    def principal_point_px(self) -> tuple[float, float]:
        cx = self.image_width_px / 2.0 if self.cx_px is None else float(
            self.cx_px)
        cy = self.image_height_px / 2.0 if self.cy_px is None else float(
            self.cy_px)
        return cx, cy

    def with_principal_point(self, cx_px: float, cy_px: float):
        return dataclasses.replace(self,
                                   cx_px=float(cx_px),
                                   cy_px=float(cy_px))


@dataclasses.dataclass(frozen=True)
class CameraFieldOfView:
    """Estimated angular coverage of a camera model, in degrees."""

    horizontal_deg: float
    vertical_deg: float
    diagonal_deg: float
    max_off_axis_deg: float


@dataclasses.dataclass(frozen=True)
class Pointing:
    lon_deg: float
    lat_deg: float
    roll_deg: float
    system: CoordSystem

    @cached_property
    def R(self) -> NDArray[np.float64]:
        """Build a world-to-camera rotation matrix from spherical pointing + roll.

        Works for both AltAz (lon=az, lat=alt) and RA/Dec (lon=ra, lat=dec)
        because both share the same spherical geometry with z-axis as pole.

        Returns:
            3x3 rotation matrix (world → camera, OpenCV convention: X-right, Y-down, Z-forward).
        """
        ra = np.deg2rad(self.lon_deg)
        dec = np.deg2rad(self.lat_deg)
        roll = np.deg2rad(self.roll_deg)

        forward = np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ])

        north_pole = np.array([0.0, 0.0, 1.0])

        if abs(self.lat_deg) > 89.99:
            up_raw = np.array([np.cos(ra), np.sin(ra), 0.0])
            if self.lat_deg > 0:
                up_raw = -up_raw
        else:
            up_raw = north_pole - np.dot(north_pole, forward) * forward

        up = up_raw / np.linalg.norm(up_raw)
        down = -up

        right = np.cross(down, forward)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        down = down / np.linalg.norm(down)

        R_no_roll = np.array([right, down, forward])

        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        R_roll = np.array([
            [cos_roll, sin_roll, 0],
            [-sin_roll, cos_roll, 0],
            [0, 0, 1],
        ])

        return R_roll @ R_no_roll

    @classmethod
    def from_view(cls,
                  view: "View",
                  mode: str = "auto") -> Optional["Pointing"]:
        """Build an absolute pointing from the orientation fields of a view."""
        if mode == "auto":
            if (view.az_deg is not None and view.alt_deg is not None
                    and view.world_roll_deg is not None):
                mode = CoordSystem.ALTAZ.value
            elif (view.ra_deg is not None and view.dec_deg is not None
                  and view.sky_roll_deg is not None):
                mode = CoordSystem.RADEC.value
            else:
                return None

        if mode == CoordSystem.ALTAZ.value:
            if (view.az_deg is None or view.alt_deg is None
                    or view.world_roll_deg is None):
                raise ValueError(
                    "Azimuth, altitude, and world roll are required for AltAz pointing"
                )
            return cls(view.az_deg, view.alt_deg, view.world_roll_deg,
                       CoordSystem.ALTAZ)

        if mode == CoordSystem.RADEC.value:
            if (view.ra_deg is None or view.dec_deg is None
                    or view.sky_roll_deg is None):
                raise ValueError(
                    "RA, Dec, and sky roll are required for RA/Dec pointing")
            return cls(view.ra_deg, view.dec_deg, view.sky_roll_deg,
                       CoordSystem.RADEC)

        if mode == CoordSystem.CAMERA.value:
            return None
        raise ValueError(f"Unsupported pointing mode: {mode!r}")


@dataclasses.dataclass
class View:
    """Complete view description."""
    focal_length: float
    sensor_width_mm: float
    sensor_height_mm: float
    img_width: int
    img_height: int
    az_deg: Optional[float] = None
    alt_deg: Optional[float] = None
    world_roll_deg: Optional[float] = None
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    sky_roll_deg: Optional[float] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    jd: Optional[float] = None

    def altaz_to_radec(self):
        if self.az_deg is None or self.alt_deg is None or self.world_roll_deg is None:
            raise ValueError(
                "Azimuth, Altitude and Roll deg must be set to convert to RA/Dec."
            )
        if self.latitude_deg is None or self.longitude_deg is None or self.jd is None:
            raise ValueError(
                "Latitude, Longitude, and Julian Day must be set to convert to RA/Dec."
            )
        ra, dec = altaz_to_radec(azimuth_deg=self.az_deg,
                                 elevation_deg=self.alt_deg,
                                 latitude_deg=self.latitude_deg,
                                 longitude_deg=self.longitude_deg,
                                 jd=self.jd)
        parallactic_angle = compute_parallactic_angle(self.az_deg,
                                                      self.alt_deg,
                                                      self.latitude_deg)
        self.ra_deg = ra
        self.dec_deg = dec
        self.sky_roll_deg = (self.world_roll_deg + parallactic_angle) % 360.0


@dataclasses.dataclass(frozen=True)
class BaseCameraModel(abc.ABC):
    """Abstract base for camera projection models.

    Camera models contain calibration only: intrinsics, distortion, and the
    projection family. Pose and optimization state are held separately.
    """
    intrinsics: Intrinsics
    distortion: BaseDistortion = dataclasses.field(
        default_factory=BaseDistortion)

    @abc.abstractmethod
    def unproject(self, pts: "NDArray[np.float64]") -> "NDArray[np.float64]":
        """Map pixel coordinates (n, 2) to unit vectors (n, 3)."""
        ...

    @abc.abstractmethod
    def project(self, vecs: "NDArray[np.float64]") -> "NDArray[np.float64]":
        """Map unit vectors (n, 3) to pixel coordinates (n, 2)."""
        ...

    @property
    @abc.abstractmethod
    def fov(self) -> CameraFieldOfView:
        """Estimate horizontal, vertical, and diagonal field of view."""
        ...

    @property
    def remap_projection(self) -> Optional[str]:
        """Projection identifier for fused remap, or ``None`` if unsupported.

        Keeping this capability optional preserves compatibility with external
        camera-model subclasses, which continue through the generic path.
        """
        return None

    @property
    def remap_dist_coeffs(self) -> Optional[NDArray[np.float64]]:
        """Projection-specific distortion coefficients for fused remap."""
        return None

    def _estimate_fov_from_model(self) -> CameraFieldOfView:
        """Estimate FOV by unprojecting the continuous image boundary.

        Using ``unproject`` keeps focal length, principal point, and the
        concrete distortion model on the same calculation path as alignment.
        Opposite off-axis angles are added instead of taking the shortest
        angle between rays, so fisheye coverage may correctly exceed 180°.
        """
        w = self.intrinsics.image_width_px
        h = self.intrinsics.image_height_px
        cx, cy = self.intrinsics.principal_point_px
        boundary_points = np.array([[0.0, cy], [w, cy], [cx, 0.0], [cx, h],
                                    [0.0, 0.0], [w, 0.0], [w, h], [0.0, h]],
                                   dtype=np.float64)
        rays = self.unproject(boundary_points)
        if rays.shape != (8, 3) or not np.all(np.isfinite(rays)):
            raise ValueError("Camera model produced invalid boundary rays")

        norms = np.linalg.norm(rays, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("Camera model produced zero-length boundary rays")
        cos_theta = rays[:, 2] / norms
        off_axis_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        horizontal = float(off_axis_deg[0] + off_axis_deg[1])
        vertical = float(off_axis_deg[2] + off_axis_deg[3])
        diagonal = float(
            max(off_axis_deg[4] + off_axis_deg[6],
                off_axis_deg[5] + off_axis_deg[7]))
        return CameraFieldOfView(
            horizontal_deg=horizontal,
            vertical_deg=vertical,
            diagonal_deg=diagonal,
            max_off_axis_deg=float(np.max(off_axis_deg)),
        )

    @cached_property
    def K(self) -> NDArray[np.float64]:
        return self.intrinsics.K

    def with_intrinsics(self, intrinsics: "Intrinsics"):
        return dataclasses.replace(self, intrinsics=intrinsics)

    def with_distortion(self, distortion: Any):
        return dataclasses.replace(self, distortion=distortion)

    def with_focal_length(self, focal_length_mm: float):
        return self.with_intrinsics(
            self.intrinsics.with_focal_length(focal_length_mm))

    @staticmethod
    def _remap_rotation_dst_to_src(
        rotation_dst_to_src: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        rotation = np.eye(3, dtype=np.float64)
        if rotation_dst_to_src is not None:
            rotation = np.asarray(rotation_dst_to_src, dtype=np.float64)
            if rotation.shape != (3, 3):
                raise ValueError("rotation_dst_to_src must have shape (3, 3)")
        return rotation

    def _project_image_from_camera_fused(
        self,
        camera: "BaseCameraModel",
        img: NDArray[np.uint8],
        output_size: tuple[int, int],
        roi=None,
        interpolation=cv2.INTER_LINEAR,
        rotation_dst_to_src: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.uint8] | None:
        if roi is not None or interpolation != cv2.INTER_LINEAR:
            return None

        src_projection = camera.remap_projection
        dst_projection = self.remap_projection
        if src_projection is None or dst_projection is None:
            return None

        target_width, target_height = output_size
        return custom_camera_model_remap(
            image=img,
            out_height=target_height,
            out_width=target_width,
            fx_src=float(camera.K[0, 0]),
            fy_src=float(camera.K[1, 1]),
            cx_src=float(camera.K[0, 2]),
            cy_src=float(camera.K[1, 2]),
            fx_dst=float(self.K[0, 0]),
            fy_dst=float(self.K[1, 1]),
            cx_dst=float(self.K[0, 2]),
            cy_dst=float(self.K[1, 2]),
            rotation_dst_to_src=self._remap_rotation_dst_to_src(
                rotation_dst_to_src),
            src_dist_coeffs=camera.remap_dist_coeffs,
            dst_dist_coeffs=self.remap_dist_coeffs,
            src_projection=src_projection,
            dst_projection=dst_projection,
        )

    def project_image_from_camera(
        self,
        camera: "BaseCameraModel",
        img: NDArray[np.uint8],
        output_size: tuple[int, int],
        roi=None,
        interpolation=cv2.INTER_LINEAR,
        rotation_dst_to_src: Optional[NDArray[np.float64]] = None,
        map_scale: float = 0.5,
    ) -> NDArray[np.uint8]:
        """Per-pixel remap: unproject dst pixels → project into src → remap.

        ``rotation_dst_to_src`` maps destination camera-local rays into source
        camera-local rays. Identity is used when it is omitted.
        Supported perspective/fisheye combinations prefer the exact fused
        custom op. ``map_scale`` controls the generic coordinate-map fallback
        used when that native path is unavailable.
        """
        if not 0.0 < map_scale <= 1.0:
            raise ValueError("map_scale must be in (0, 1]")

        fused = self._project_image_from_camera_fused(
            camera,
            img,
            output_size,
            roi=roi,
            interpolation=interpolation,
            rotation_dst_to_src=rotation_dst_to_src,
        )
        if fused is not None:
            return fused

        target_width, target_height = output_size
        use_sparse_map = (map_scale < 1.0 and roi is None
                          and interpolation == cv2.INTER_LINEAR)
        grid_width = (max(2, round(target_width * map_scale))
                      if use_sparse_map else target_width)
        grid_height = (max(2, round(target_height * map_scale))
                       if use_sparse_map else target_height)
        if use_sparse_map:
            sx = grid_width / target_width
            sy = grid_height / target_height
            u_dst = ((np.arange(grid_width, dtype=np.float64) + 0.5) / sx
                     - 0.5)
            v_dst = ((np.arange(grid_height, dtype=np.float64) + 0.5) / sy
                     - 0.5)
        else:
            u_dst = np.arange(grid_width, dtype=np.float64)
            v_dst = np.arange(grid_height, dtype=np.float64)
        u_grid, v_grid = np.meshgrid(u_dst, v_dst)
        dst_pixels = np.stack([u_grid.ravel(), v_grid.ravel()],
                              axis=1).astype(np.float64)

        src_vecs = self.unproject(dst_pixels)
        if rotation_dst_to_src is not None:
            rotation = np.asarray(rotation_dst_to_src, dtype=np.float64)
            if rotation.shape != (3, 3):
                raise ValueError("rotation_dst_to_src must have shape (3, 3)")
            src_vecs = (rotation @ src_vecs.T).T
        src_pixels = camera.project(src_vecs)

        map_x = src_pixels[:, 0].reshape(grid_height,
                                         grid_width).astype(np.float32)
        map_y = src_pixels[:, 1].reshape(grid_height,
                                         grid_width).astype(np.float32)
        if use_sparse_map:
            map_x = cv2.resize(map_x, (target_width, target_height),
                               interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(map_y, (target_width, target_height),
                               interpolation=cv2.INTER_LINEAR)
            border_x = np.arange(target_width, dtype=np.float64)
            border_y = np.arange(1, target_height - 1, dtype=np.float64)
            border_pixels = np.concatenate((
                np.column_stack((border_x, np.zeros(target_width))),
                np.column_stack((border_x,
                                 np.full(target_width, target_height - 1.0))),
                np.column_stack((np.zeros(len(border_y)), border_y)),
                np.column_stack((np.full(len(border_y), target_width - 1.0),
                                 border_y)),
            ))
            border_vecs = self.unproject(border_pixels)
            if rotation_dst_to_src is not None:
                border_vecs = (rotation @ border_vecs.T).T
            border_src = camera.project(border_vecs)
            bx = border_pixels[:, 0].astype(np.intp)
            by = border_pixels[:, 1].astype(np.intp)
            map_x[by, bx] = border_src[:, 0].astype(np.float32)
            map_y[by, bx] = border_src[:, 1].astype(np.float32)

        if roi is not None:
            x1, y1, x2, y2 = roi
            img_use = img[y1:y2, x1:x2]
            map_x = map_x - x1
            map_y = map_y - y1
        else:
            img_use = img

        remapped = cv2.remap(img_use,
                             map_x,
                             map_y,
                             interpolation=interpolation,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
        if img_use.ndim == 3 and img_use.shape[2] == 1 and remapped.ndim == 2:
            return remapped[:, :, None]
        return remapped


@dataclasses.dataclass(frozen=True)
class CameraModel(BaseCameraModel):
    distortion: Distortion = dataclasses.field(default_factory=Distortion)

    @property
    def dist_coeffs(self) -> Optional[NDArray[np.float64]]:
        if self.distortion.is_zero:
            return None
        return self.distortion.to_cv2()

    @property
    def remap_projection(self) -> str:
        return "perspective"

    @property
    def remap_dist_coeffs(self) -> Optional[NDArray[np.float64]]:
        return self.dist_coeffs

    def unproject(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return unproject_pixels(pts, self.K, self.dist_coeffs)

    def project(self, vecs: NDArray[np.float64]) -> NDArray[np.float64]:
        return project_vectors(vecs, self.K, self.dist_coeffs)

    @cached_property
    def fov(self) -> CameraFieldOfView:
        return self._estimate_fov_from_model()

    def with_distortion(self, distortion: Distortion) -> "CameraModel":
        return super().with_distortion(distortion)

    @classmethod
    def from_view(cls, view: View) -> "CameraModel":
        """Build camera calibration from a view; orientation stays separate."""
        intrinsics = Intrinsics(
            focal_length_mm=view.focal_length,
            sensor_width_mm=view.sensor_width_mm,
            sensor_height_mm=view.sensor_height_mm,
            image_width_px=view.img_width,
            image_height_px=view.img_height,
        )
        return cls(intrinsics=intrinsics)


@dataclasses.dataclass(frozen=True)
class FisheyeCameraModel(BaseCameraModel):
    """Camera model using Kannala-Brandt fisheye projection (cv2.fisheye).

    Supports any fisheye type (equidistant, equisolid, orthographic, etc.)
    via the k1..k4 polynomial. Zero k1..k4 = pure equidistant model.
    """
    distortion: FisheyeDistortion = dataclasses.field(
        default_factory=FisheyeDistortion)

    @property
    def dist_k4(self) -> NDArray[np.float64]:
        return self.distortion.to_cv2()

    @property
    def remap_projection(self) -> str:
        return "fisheye"

    @property
    def remap_dist_coeffs(self) -> Optional[NDArray[np.float64]]:
        if self.distortion.is_zero:
            return None
        return self.dist_k4

    def unproject(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return unproject_fisheye_pixels(pts, self.K, self.dist_k4)

    def project(self, vecs: NDArray[np.float64]) -> NDArray[np.float64]:
        return project_fisheye_vectors(vecs, self.K, self.dist_k4)

    @cached_property
    def fov(self) -> CameraFieldOfView:
        return self._estimate_fov_from_model()

    def with_distortion(self,
                        distortion: FisheyeDistortion) -> "FisheyeCameraModel":
        return super().with_distortion(distortion)
