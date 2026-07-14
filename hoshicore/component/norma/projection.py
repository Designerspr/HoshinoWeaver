"""Pure functions for camera projection and unprojection."""
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


def _pad_fisheye_dist_k4(dist_k4: Optional[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Return fisheye distortion coefficients padded to [k1, k2, k3, k4]."""
    if dist_k4 is None:
        return np.zeros(4, dtype=np.float64)
    coeffs = np.asarray(dist_k4, dtype=np.float64).ravel()
    if coeffs.size >= 4:
        return coeffs[:4]
    padded = np.zeros(4, dtype=np.float64)
    padded[:coeffs.size] = coeffs
    return padded


def make_intrinsic_matrix(focal_mm: float, sensor_w_mm: float,
                          sensor_h_mm: float, img_w_px: int,
                          img_h_px: int,
                          cx_px: Optional[float] = None,
                          cy_px: Optional[float] = None) -> NDArray[np.float64]:
    fx = focal_mm * img_w_px / sensor_w_mm
    fy = focal_mm * img_h_px / sensor_h_mm
    cx = img_w_px / 2.0 if cx_px is None else float(cx_px)
    cy = img_h_px / 2.0 if cy_px is None else float(cy_px)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def undistort_points(pts: NDArray[np.float64], K: NDArray[np.float64],
                     dist_coeffs: Optional[NDArray[np.float64]]
                     ) -> NDArray[np.float64]:
    """Remove distortion from pixel coordinates.
    pts: (n, 2), K: 3x3, dist_coeffs: 5-element or None.
    Returns (n, 2) undistorted pixel coordinates.
    """
    if dist_coeffs is not None and np.any(dist_coeffs != 0):
        upts = cv2.undistortPoints(pts[:, None, :].astype(np.float64), K,
                                   dist_coeffs, P=K)
        return upts[:, 0, :]
    return pts


def unproject_pixels(pts: NDArray[np.float64], K: NDArray[np.float64],
                     dist_coeffs: Optional[NDArray[np.float64]] = None,
                     R: Optional[NDArray[np.float64]] = None
                     ) -> NDArray[np.float64]:
    """Pixel coordinates to unit direction vectors in world frame.
    pts: (n, 2), K: 3x3, dist_coeffs: 5-element or None, R: 3x3 or None.
    Returns (n, 3) unit vectors.
    """
    upts = undistort_points(pts, K, dist_coeffs)
    xyz_h = np.concatenate([upts, np.ones((upts.shape[0], 1))], axis=1)
    vec = (np.linalg.inv(K) @ xyz_h.T).T
    vec = vec / np.linalg.norm(vec, axis=1)[:, None]
    if R is not None:
        vec = (R.T @ vec.T).T
    return vec


def project_vectors(v: NDArray[np.float64], K: NDArray[np.float64],
                    dist_coeffs: Optional[NDArray[np.float64]] = None,
                    R: Optional[NDArray[np.float64]] = None
                    ) -> NDArray[np.float64]:
    """Unit direction vectors in world frame to pixel coordinates.
    v: (n, 3), K: 3x3, dist_coeffs: 5-element or None, R: 3x3 or None.
    Returns (n, 2). NaN for vectors behind the camera.
    """
    assert v.shape[1] == 3 and len(v.shape) == 2
    n = v.shape[0]

    rotated = (R @ v.T).T if R is not None else v
    valid = np.where(rotated[:, 2] > 0)[0]
    result = np.full((n, 2), np.nan, dtype=np.float64)

    if len(valid) == 0:
        return result

    rv = rotated[valid]

    if dist_coeffs is not None and np.any(dist_coeffs != 0):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(rv[None, ...], rvec, tvec, K,
                                            dist_coeffs)
        result[valid, :] = image_points[:, 0, :]
    else:
        normalized = (rv * (1 / rv[:, 2][:, None])).T
        image_points = (K @ normalized).T[:, :2]
        result[valid, :] = image_points

    return result


def _fisheye_solve_theta(r_d: NDArray[np.float64],
                         k: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized Newton iteration: solve θ from
    r_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸).

    Initial guess θ₀ = r_d (accurate when coefficients are small).
    Converges in ~10 steps for all realistic Kannala-Brandt coefficients.
    """
    k1, k2, k3, k4 = k[0], k[1], k[2], k[3]
    theta = r_d.copy()
    for _ in range(10):
        th2 = theta * theta
        th4 = th2 * th2
        th6 = th4 * th2
        th8 = th4 * th4
        f  = theta * (1.0 + k1*th2 + k2*th4 + k3*th6 + k4*th8) - r_d
        df = 1.0 + 3.0*k1*th2 + 5.0*k2*th4 + 7.0*k3*th6 + 9.0*k4*th8
        theta = theta - f / np.where(np.abs(df) > 1e-12, df, 1e-12)
    return theta


def unproject_fisheye_pixels(pts: NDArray[np.float64], K: NDArray[np.float64],
                             dist_k4: Optional[NDArray[np.float64]] = None,
                             R: Optional[NDArray[np.float64]] = None
                             ) -> NDArray[np.float64]:
    """Fisheye pixel coordinates to unit direction vectors (Kannala-Brandt model).
    pts: (n, 2), K: 3x3, dist_k4: [k1,k2,k3,k4] or None, R: 3x3 or None.
    Returns (n, 3) unit vectors.

    OpenCV fisheye: r_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
    where r_d = sqrt(x_d²+y_d²), x_d = (u-cx)/fx.
    Zero k → pure equidistant model (r_d = θ).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_d = (pts[:, 0] - cx) / fx
    y_d = (pts[:, 1] - cy) / fy
    r_d = np.sqrt(x_d * x_d + y_d * y_d)  # = θ_d (distorted angle)

    k = _pad_fisheye_dist_k4(dist_k4)
    if np.any(k != 0):
        theta = _fisheye_solve_theta(r_d, k)
    else:
        theta = r_d  # equidistant: θ = r_d directly

    safe_r = np.where(r_d > 0, r_d, 1.0)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    vx = sin_t * x_d / safe_r
    vy = sin_t * y_d / safe_r
    vz = cos_t
    # Center pixel: r_d=0 → sin_t=0, direction undefined; vz=cos(0)=1 ✓
    vec = np.stack([vx, vy, vz], axis=1)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    vec = vec / np.where(norms > 0, norms, 1.0)
    if R is not None:
        vec = (R.T @ vec.T).T
    return vec


def project_fisheye_vectors(v: NDArray[np.float64], K: NDArray[np.float64],
                            dist_k4: Optional[NDArray[np.float64]] = None,
                            R: Optional[NDArray[np.float64]] = None
                            ) -> NDArray[np.float64]:
    """Unit direction vectors to fisheye pixel coordinates (Kannala-Brandt model).
    v: (n, 3), K: 3x3, dist_k4: [k1,k2,k3,k4] or None, R: 3x3 or None.
    Returns (n, 2). Unlike perspective, fisheye can see vectors with z < 0 (θ > 90°).
    """
    assert v.shape[1] == 3 and v.ndim == 2

    rotated = (R @ v.T).T if R is not None else v
    n = rotated.shape[0]
    result = np.full((n, 2), np.nan, dtype=np.float64)

    norms = np.linalg.norm(rotated, axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        return result

    vecs = rotated[valid] / norms[valid, None]
    xy_norm = np.linalg.norm(vecs[:, :2], axis=1)
    theta = np.arctan2(xy_norm, vecs[:, 2])

    k = _pad_fisheye_dist_k4(dist_k4)
    th2 = theta * theta
    th4 = th2 * th2
    th6 = th4 * th2
    th8 = th4 * th4
    theta_d = theta * (1.0 + k[0] * th2 + k[1] * th4 + k[2] * th6 + k[3] * th8)

    safe_xy = np.where(xy_norm > 1e-12, xy_norm, 1.0)
    x_d = theta_d * vecs[:, 0] / safe_xy
    y_d = theta_d * vecs[:, 1] / safe_xy
    x_d = np.where(xy_norm > 1e-12, x_d, 0.0)
    y_d = np.where(xy_norm > 1e-12, y_d, 0.0)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    result[valid, 0] = fx * x_d + cx
    result[valid, 1] = fy * y_d + cy
    return result


def distort_image(img_undist: NDArray[np.uint8], K: NDArray[np.float64],
                  dist_coeffs: NDArray[np.float64],
                  output_size: tuple[int, int]) -> NDArray[np.uint8]:
    """Apply distortion to an undistorted image via remap."""
    target_width, target_height = output_size

    ys, xs = np.meshgrid(np.arange(target_height),
                         np.arange(target_width),
                         indexing='ij')
    pixels_dist = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)

    pixels_norm_undist = cv2.undistortPoints(pixels_dist.reshape(-1, 1, 2), K,
                                            dist_coeffs,
                                            P=None).reshape(-1, 2)

    pixels_h = np.hstack(
        [pixels_norm_undist,
         np.ones((len(pixels_norm_undist), 1))])
    pixels_undist = (K @ pixels_h.T).T[:, :2]

    map_xy = pixels_undist.reshape(target_height, target_width,
                                   2).astype(np.float32)

    return cv2.remap(img_undist,
                     map_xy[..., 0],
                     map_xy[..., 1],
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
