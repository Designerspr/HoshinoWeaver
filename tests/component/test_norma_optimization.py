"""Tests for norma alignment optimization residuals."""
import numpy as np
from scipy.optimize import least_squares

from hoshicore.component.norma.optimization import (
    CameraOptimizationPolicy,
    CameraOptimizationState,
    FlexibleOptimizationContext,
    flexible_reproject_error,
    make_flexible_regularization_weights,
)


def _scipy_constant_residual_cost(
    errors: np.ndarray,
    loss: str,
    f_scale: float,
) -> float:
    res = least_squares(
        lambda _: errors,
        np.zeros(1, dtype=np.float64),
        method="trf",
        loss=loss,
        f_scale=f_scale,
        max_nfev=1,
    )
    return float(res.cost)


def test_scipy_huber_loss_matches_expected_cost():
    errors = np.array([-0.004, -0.001, 0.0, 0.001, 0.004], dtype=np.float64)
    threshold = 0.002
    abs_error = np.abs(errors)
    expected_loss = np.where(
        abs_error < threshold,
        0.5 * errors**2,
        threshold * (abs_error - 0.5 * threshold),
    )

    cost = _scipy_constant_residual_cost(errors, "huber", threshold)

    np.testing.assert_allclose(cost, np.sum(expected_loss))


def test_scipy_cauchy_loss_matches_expected_cost():
    errors = np.array([-0.006, -0.002, 0.0, 0.002, 0.006], dtype=np.float64)
    scale = 0.003
    expected_loss = 0.5 * scale**2 * np.log(1 + (errors / scale)**2)

    cost = _scipy_constant_residual_cost(errors, "cauchy", scale)

    np.testing.assert_allclose(cost, np.sum(expected_loss))


def test_same_camera_fisheye_pixel_residual_uses_shared_distortion():
    pts = np.array([
        [2600.0, 1700.0],
        [3100.0, 2100.0],
        [1800.0, 900.0],
    ], dtype=np.float64)
    policy = CameraOptimizationPolicy(
        optimize_focal=True,
        optimize_distortion=True,
        optimize_principal_point=False,
        n_dist=3,
    )
    state = CameraOptimizationState(
        projection_type="fisheye",
        base_focal=16.0,
        sensor_w_mm=36.0,
        sensor_h_mm=24.0,
        img_w=5472,
        img_h=3648,
        base_cx=2736.0,
        base_cy=1824.0,
        base_distortion=np.array([0.02, -0.01, 0.004, -0.002], dtype=np.float64),
        policy=policy,
    )
    x = np.array([
        0.0, 0.0, 0.0,       # rvec
        0.0,                 # focal scale
        0.02, -0.01, 0.004, # optimized k1..k3; initialized k4 stays fixed
    ], dtype=np.float64)
    ctx = FlexibleOptimizationContext(
        ref_pts=pts,
        src_pts=pts.copy(),
        ref_state=state,
        src_state=state,
        same_camera=True,
        robust_loss=None,
        residual_space="pixel",
    )

    residual = flexible_reproject_error(x, ctx)

    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_fisheye_focal_prior_is_context_weight():
    policy = CameraOptimizationPolicy(optimize_focal=True, optimize_distortion=False)
    state = CameraOptimizationState(
        projection_type="fisheye",
        base_focal=16.0,
        sensor_w_mm=36.0,
        sensor_h_mm=24.0,
        img_w=5472,
        img_h=3648,
        base_cx=2736.0,
        base_cy=1824.0,
        base_distortion=np.zeros(4, dtype=np.float64),
        policy=policy,
    )
    ctx = FlexibleOptimizationContext(
        ref_pts=np.zeros((1, 2), dtype=np.float64),
        src_pts=np.zeros((1, 2), dtype=np.float64),
        ref_state=state,
        src_state=state,
        same_camera=True,
    )
    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    weights = make_flexible_regularization_weights(ctx, x0)

    np.testing.assert_allclose(weights, np.array([0.0, 0.0, 0.0, 1.0]))
