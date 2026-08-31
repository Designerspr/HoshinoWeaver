"""Debug helper for star detection threshold inspection.

Usage:
    python -m tools.debug.debug_star_detection <input_image> [options]

Options:
    --output-dir DIR          Output directory (default: debug_star_detection_median_out/)
    --mask PATH               External mask image (0 = excluded, non-zero = valid)
    --detector NAME           median (default) or wavelet-debug
    --median-ksize N          Median background kernel size (default: 13)
    --threshold-ratio F       Residual sigma multiplier (default: 5.0)
    --open-ksize N            Morphological opening kernel (default: 3)
    --min-area F              Median candidate minimum area (default: 7)
    --resize-length N         Max image side length before downscale (default: 10000)
    --gaussian-ksize N        Gaussian kernel size (default: 9)
    --sigma F                 Gaussian sigma (default: 2.0)
    --min-star-points N       Minimum contour count target (default: 400)
    --display-scale N         Max side length for overlay images (default: 2000)
    --no-density-recovery     Disable density-based recovery
    --density-grid H W        Density recovery grid size (default: 6 6)
    --density-threshold F     Recovery trigger factor, using p75_ref * F (default: 0.5)
    --relaxed-ecc-max F       Relaxed eccentricity cap inside sparse cells (default: 0.92)
    --centroid-inspect        Save a full-res centroid-crosshair contact sheet
                              (median detector only, sampled center/mid/corner)
    --centroid-crop-size N    Crop size in source px around each sample (default: 32)
    --centroid-zoom N         Zoom factor per crop tile (default: 8)
    --centroid-per-region N   Samples per center/mid/corner region (default: 6)

Examples:
    python -m tools.debug.debug_star_detection sky.tif
    python -m tools.debug.debug_star_detection sky.tif --display-scale 3000 --output-dir out/
    python -m tools.debug.debug_star_detection sky.tif --no-density-recovery
    python -m tools.debug.debug_star_detection sky.tif --density-grid 4 4 --density-threshold 0.4
"""

import argparse
import os
from math import log

import cv2
import numpy as np
import pywt

from hoshicore.component.norma.detection import (
    detect_star_points_median_detailed,
)

MIN_STAR_AREA = 10
STAR_FILTER_PERCENTILE = 10


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load 8-bit / 16-bit PNG/TIF and return BGR uint8 or uint16."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def to_display_uint8(img: np.ndarray) -> np.ndarray:
    """Convert arbitrary dtype input to uint8 for visualization."""
    if img.dtype == np.uint8:
        return img.copy()
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    if img.dtype.kind == 'f':
        lo, hi = img.min(), img.max()
        if hi > lo:
            return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return np.zeros(img.shape[:2] + ((3,) if img.ndim == 2 else ()), dtype=np.uint8)
    return img.astype(np.uint8)


def save_image(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if img.dtype == np.uint16:
        cv2.imwrite(path, img)
    elif img.dtype != np.uint8:
        img = to_display_uint8(img)
    cv2.imwrite(path, img)


def to_gray_float64(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.issubdtype(gray.dtype, np.integer):
        return gray.astype(np.float64) / np.iinfo(gray.dtype).max
    return gray.astype(np.float64)


# ---------------------------------------------------------------------------
# Core function copied from detection.py
# ---------------------------------------------------------------------------

def _wavelet_dec_rec(img_blr: np.ndarray, resize_factor: float = 0.25) -> np.ndarray:
    img_shape = img_blr.shape
    level = int(6 - log(1 / resize_factor, 2))
    img_blr_resize = cv2.resize(img_blr, None, fx=resize_factor, fy=resize_factor)
    coeffs = pywt.wavedec2(img_blr_resize, "db8", level=level)
    coeffs[0].fill(0)
    coeffs[-1][0].fill(0)
    coeffs[-1][1].fill(0)
    coeffs[-1][2].fill(0)
    img_rec_resize = pywt.waverec2(coeffs, "db8")
    img_rec = cv2.resize(img_rec_resize, (img_shape[1], img_shape[0]))
    return img_rec


# ---------------------------------------------------------------------------
# Density grid visualization
# ---------------------------------------------------------------------------

def _save_density_grid(
    path: str,
    display_small: np.ndarray,
    density_grid: tuple,
    cell_counts_before: np.ndarray,
    cell_counts_after: np.ndarray,
    cell_mask_frac: np.ndarray,
    triggered_cells: np.ndarray,
    avg_density: float,
    density_threshold: float,
):
    """Draw the density grid on a scaled display image."""
    vis = display_small.copy()
    h_vis, w_vis = vis.shape[:2]
    grid_h, grid_w = density_grid

    ys_v = np.linspace(0, h_vis, grid_h + 1).astype(int)
    xs_v = np.linspace(0, w_vis, grid_w + 1).astype(int)

    for gy in range(grid_h):
        for gx in range(grid_w):
            y0, y1 = ys_v[gy], ys_v[gy + 1]
            x0, x1 = xs_v[gx], xs_v[gx + 1]
            frac = cell_mask_frac[gy, gx]
            before = cell_counts_before[gy, gx]
            after  = cell_counts_after[gy, gx]
            trig   = triggered_cells[gy, gx]

            # Background shading
            if frac < 0.2:
                # Non-sky cell: dark mask
                overlay = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
                overlay[:] = (30, 30, 30)
                cv2.addWeighted(vis[y0:y1, x0:x1], 0.4, overlay, 0.6, 0,
                                vis[y0:y1, x0:x1])
            elif trig:
                # Triggered recovery cell: blue tint
                overlay = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
                overlay[:] = (180, 80, 0)
                cv2.addWeighted(vis[y0:y1, x0:x1], 0.55, overlay, 0.45, 0,
                                vis[y0:y1, x0:x1])

            # Cell border
            border_color = (80, 80, 80) if frac < 0.2 else ((200, 120, 0) if trig else (160, 160, 160))
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), border_color, 1)

            # Text labels
            cx_text = x0 + 4
            cy_text = y0 + 14
            if frac < 0.2:
                cv2.putText(vis, "---", (cx_text, cy_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
            else:
                if trig:
                    label = f"{before}->{after}"
                    color = (0, 220, 255)
                else:
                    label = str(before)
                    color = (220, 220, 220)
                cv2.putText(vis, label, (cx_text, cy_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
                # Mask coverage
                cv2.putText(vis, f"{frac*100:.0f}%", (cx_text, cy_text + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1, cv2.LINE_AA)

    # Summary bar
    bar_h = 20
    bar = np.zeros((bar_h, w_vis, 3), dtype=np.uint8)
    info = (f"grid={grid_h}x{grid_w}  p75_ref={avg_density:.1f}  "
            f"threshold={avg_density * density_threshold:.1f}  "
            f"triggered={triggered_cells.sum()}")
    cv2.putText(bar, info, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (200, 200, 200), 1, cv2.LINE_AA)
    vis = np.vstack([bar, vis])

    cv2.imwrite(path, vis)


# ---------------------------------------------------------------------------
# 主诊断流程
# ---------------------------------------------------------------------------

def _format_threshold_label(label: str, value: float) -> str:
    if not np.isfinite(value):
        return f"{label}=nan"
    if abs(value) >= 100 or abs(value - round(value)) < 1e-9:
        return f"{label}={value:.0f}"
    if abs(value) >= 1:
        return f"{label}={value:.2f}"
    return f"{label}={value:.4f}"


def _series_hist_image(
    values: np.ndarray,
    title: str,
    threshold_lines=None,
    width: int = 520,
    height: int = 320,
    bins: int = 40,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    img = np.full((height, width, 3), 18, dtype=np.uint8)

    cv2.putText(img, title, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (230, 230, 230), 1, cv2.LINE_AA)
    if values.size == 0:
        cv2.putText(img, "no data", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (180, 180, 180), 1, cv2.LINE_AA)
        return img

    left, right = 52, width - 24
    top, bottom = 40, height - 54
    plot_w = max(1, right - left)
    plot_h = max(1, bottom - top)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax))
    hist_max = max(int(hist.max()), 1)
    cv2.rectangle(img, (left, top), (right, bottom), (90, 90, 90), 1)
    for i in range(1, 5):
        y = bottom - int(i / 5 * plot_h)
        cv2.line(img, (left, y), (right, y), (40, 40, 40), 1)

    bin_w = plot_w / bins
    for idx, count in enumerate(hist):
        x0 = int(left + idx * bin_w)
        x1 = max(x0 + 1, int(left + (idx + 1) * bin_w) - 1)
        bar_h = int(count / hist_max * (plot_h - 1))
        cv2.rectangle(img, (x0, bottom - bar_h), (x1, bottom), (0, 190, 230), -1)

    for idx, item in enumerate(threshold_lines or []):
        color, label, value = item
        if not np.isfinite(value):
            continue
        px = left + int((value - vmin) / (vmax - vmin) * plot_w)
        px = int(np.clip(px, left, right))
        cv2.line(img, (px, top), (px, bottom), color, 2)
        cv2.putText(img, _format_threshold_label(label, value), (px + 4, top + 16 + idx * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    p10 = float(np.percentile(values, 10))
    p50 = float(np.percentile(values, 50))
    p90 = float(np.percentile(values, 90))
    cv2.putText(img, f"p10={p10:.3g}  p50={p50:.3g}  p90={p90:.3g}",
                (left, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"n={len(values)}", (16, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(img, _format_threshold_label("min", vmin), (left, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(img, _format_threshold_label("max", vmax), (right - 92, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def _scatter_plot_image(
    x: np.ndarray,
    y: np.ndarray,
    colors: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    x_thresholds=None,
    y_thresholds=None,
    width: int = 520,
    height: int = 320,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    colors = colors[valid]

    img = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(img, title, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (230, 230, 230), 1, cv2.LINE_AA)
    if x.size == 0:
        cv2.putText(img, "no data", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (180, 180, 180), 1, cv2.LINE_AA)
        return img

    left, right = 60, width - 24
    top, bottom = 40, height - 54
    plot_w = max(1, right - left)
    plot_h = max(1, bottom - top)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if x_max <= x_min:
        x_max = x_min + 1e-6
    if y_max <= y_min:
        y_max = y_min + 1e-6

    cv2.rectangle(img, (left, top), (right, bottom), (90, 90, 90), 1)
    for i in range(1, 5):
        px = left + int(i / 5 * plot_w)
        py = bottom - int(i / 5 * plot_h)
        cv2.line(img, (px, top), (px, bottom), (40, 40, 40), 1)
        cv2.line(img, (left, py), (right, py), (40, 40, 40), 1)

    for idx, item in enumerate(x_thresholds or []):
        color, label, value = item
        px = left + int((value - x_min) / (x_max - x_min) * plot_w)
        px = int(np.clip(px, left, right))
        cv2.line(img, (px, top), (px, bottom), color, 1)
        cv2.putText(img, _format_threshold_label(label, value), (px + 3, top + 16 + idx * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    for idx, item in enumerate(y_thresholds or []):
        color, label, value = item
        py = bottom - int((value - y_min) / (y_max - y_min) * plot_h)
        py = int(np.clip(py, top, bottom))
        cv2.line(img, (left, py), (right, py), color, 1)
        cv2.putText(img, _format_threshold_label(label, value), (left + 4, py - 4 - idx * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    for xv, yv, color in zip(x, y, colors):
        px = left + int((xv - x_min) / (x_max - x_min) * plot_w)
        py = bottom - int((yv - y_min) / (y_max - y_min) * plot_h)
        cv2.circle(img, (px, py), 2, tuple(int(c) for c in color), -1, cv2.LINE_AA)

    cv2.putText(img, x_label, (left, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, y_label, (16, top + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)
    return img


def _save_img_rec_distribution(path: str, img_rec: np.ndarray, mask: np.ndarray,
                               threshold: float,
                               threshold_label: str = "p99.5",
                               title: str = "img_rec pixel distribution"):
    img = _series_hist_image(
        img_rec[mask],
        title,
        threshold_lines=[((0, 80, 255), threshold_label, threshold)],
        bins=60,
    )
    cv2.imwrite(path, img)


def _save_feature_distributions(path: str, areas: np.ndarray,
                                eccentricities: np.ndarray,
                                intensities: np.ndarray,
                                area_pct: float,
                                intensity_pct: float,
                                min_area: float = MIN_STAR_AREA,
                                max_eccentricity: float = 0.8):
    area_img = _series_hist_image(
        areas,
        "candidate area",
        threshold_lines=[
            ((0, 80, 255), "min_area", min_area),
            ((0, 220, 220), f"p{STAR_FILTER_PERCENTILE}", area_pct),
        ],
    )
    ecc_img = _series_hist_image(
        eccentricities,
        "candidate eccentricity",
        threshold_lines=[((0, 150, 255), "ecc_max", max_eccentricity)],
    )
    intensity_img = _series_hist_image(
        intensities,
        "candidate intensity",
        threshold_lines=[((220, 100, 220), f"p{STAR_FILTER_PERCENTILE}", intensity_pct)],
    )
    filler = np.full_like(intensity_img, 18)
    cv2.putText(filler, "threshold summary", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(filler, f"area min: {min_area:g}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (0, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(filler, f"area p{STAR_FILTER_PERCENTILE}: {area_pct:.3g}", (16, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(filler, f"ecc max: {max_eccentricity:g}", (16, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (0, 150, 255), 1, cv2.LINE_AA)
    cv2.putText(filler, f"intensity p{STAR_FILTER_PERCENTILE}: {intensity_pct:.3g}", (16, 166),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 100, 220), 1, cv2.LINE_AA)
    canvas = np.vstack([np.hstack([area_img, ecc_img]), np.hstack([intensity_img, filler])])
    cv2.imwrite(path, canvas)


def _save_candidate_scatter(path: str, areas: np.ndarray, eccentricities: np.ndarray,
                            intensities: np.ndarray, valid_stars: np.ndarray,
                            recovered_stars: np.ndarray, fail_area_min: np.ndarray,
                            fail_ecc: np.ndarray, fail_area_pct: np.ndarray,
                            fail_intensity: np.ndarray, area_pct: float,
                            intensity_pct: float,
                            min_area: float = MIN_STAR_AREA,
                            max_eccentricity: float = 0.8):
    colors = np.zeros((len(areas), 3), dtype=np.uint8)
    colors[:] = (220, 100, 220)
    colors[fail_area_pct] = (0, 220, 220)
    colors[fail_ecc] = (0, 100, 255)
    colors[fail_area_min] = (0, 0, 220)
    colors[valid_stars] = (0, 230, 0)
    colors[recovered_stars] = (255, 210, 0)
    colors[fail_intensity] = (220, 100, 220)

    area_intensity = _scatter_plot_image(
        areas, intensities, colors,
        "area vs intensity", "area", "intensity",
        x_thresholds=[
            ((0, 80, 255), "min_area", min_area),
            ((0, 220, 220), f"p{STAR_FILTER_PERCENTILE}", area_pct),
        ],
        y_thresholds=[((220, 100, 220), f"p{STAR_FILTER_PERCENTILE}", intensity_pct)],
    )
    ecc_area = _scatter_plot_image(
        eccentricities, areas, colors,
        "eccentricity vs area", "eccentricity", "area",
        x_thresholds=[((0, 150, 255), "ecc_max", max_eccentricity)],
        y_thresholds=[
            ((0, 80, 255), "min_area", min_area),
            ((0, 220, 220), f"p{STAR_FILTER_PERCENTILE}", area_pct),
        ],
    )

    legend = np.full((120, area_intensity.shape[1] * 2, 3), 18, dtype=np.uint8)
    items = [
        ((0, 230, 0), "valid"),
        ((255, 210, 0), "recovered"),
        ((0, 0, 220), "fail area min"),
        ((0, 100, 255), "fail eccentricity"),
        ((0, 220, 220), "fail area percentile"),
        ((220, 100, 220), "fail intensity percentile"),
    ]
    for idx, (color, label) in enumerate(items):
        col = idx % 3
        row = idx // 3
        x0 = 16 + col * 220
        y0 = 32 + row * 36
        cv2.rectangle(legend, (x0, y0 - 12), (x0 + 16, y0 + 4), color, -1)
        cv2.putText(legend, label, (x0 + 24, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)

    canvas = np.vstack([np.hstack([area_intensity, ecc_area]), legend])
    cv2.imwrite(path, canvas)


def _save_density_distribution(path: str, cell_counts_before: np.ndarray,
                               cell_counts_after: np.ndarray,
                               cell_mask_frac: np.ndarray,
                               triggered_cells: np.ndarray,
                               avg_density: float,
                               density_threshold: float):
    sky_cells = cell_mask_frac >= 0.2
    before = cell_counts_before[sky_cells].astype(np.float64)
    after = cell_counts_after[sky_cells].astype(np.float64)
    trigger_threshold = avg_density * density_threshold
    bins = max(10, min(40, int(before.size) if before.size else 10))

    before_img = _series_hist_image(
        before,
        "sky-cell count before recovery",
        threshold_lines=[
            ((0, 220, 220), "p75_ref", avg_density),
            ((0, 80, 255), "trigger", trigger_threshold),
        ],
        bins=bins,
    )
    after_img = _series_hist_image(
        after,
        "sky-cell count after recovery",
        threshold_lines=[
            ((0, 220, 220), "p75_ref", avg_density),
            ((0, 80, 255), "trigger", trigger_threshold),
        ],
        bins=bins,
    )

    summary = np.full((120, before_img.shape[1] * 2, 3), 18, dtype=np.uint8)
    cv2.putText(summary, f"sky_cells={int(sky_cells.sum())}  triggered={int(triggered_cells.sum())}",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(summary, _format_threshold_label("p75_ref", avg_density),
                (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(summary, _format_threshold_label("trigger", trigger_threshold),
                (220, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(summary, "reference density uses p75 of sky-cell counts",
                (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1, cv2.LINE_AA)

    canvas = np.vstack([np.hstack([before_img, after_img]), summary])
    cv2.imwrite(path, canvas)


def _region_label(norm_r: float) -> str:
    if norm_r < 0.33:
        return "center"
    if norm_r < 0.66:
        return "mid"
    return "corner"


def _save_centroid_inspection_sheet(
    path: str,
    img_gray: np.ndarray,
    positions: np.ndarray,
    labels: list,
    crop_size: int = 32,
    zoom: int = 8,
    cols: int = 6,
):
    """Crop full-resolution patches around selected points and draw the
    algorithm-reported centroid as a crosshair, so sub-pixel accuracy can be
    checked by eye without downscaling or overlay clutter from neighbors.
    """
    if len(positions) == 0:
        return
    half = crop_size / 2.0
    h_img, w_img = img_gray.shape
    disp = to_display_uint8(img_gray)
    if disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    tiles = []
    for (x, y), label in zip(positions, labels):
        x0 = int(round(x - half))
        y0 = int(round(y - half))
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        pad_l = max(0, -x0)
        pad_t = max(0, -y0)
        pad_r = max(0, x1 - w_img)
        pad_b = max(0, y1 - h_img)
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w_img, x1), min(h_img, y1)
        crop = disp[cy0:cy1, cx0:cx1]
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        tile = cv2.resize(crop, (crop_size * zoom, crop_size * zoom),
                          interpolation=cv2.INTER_NEAREST)

        # Sub-pixel crosshair position within the tile.
        rel_x = (x - x0) * zoom
        rel_y = (y - y0) * zoom
        px, py = int(round(rel_x)), int(round(rel_y))
        cv2.line(tile, (px, 0), (px, tile.shape[0] - 1), (0, 0, 255), 1,
                 cv2.LINE_AA)
        cv2.line(tile, (0, py), (tile.shape[1] - 1, py), (0, 0, 255), 1,
                 cv2.LINE_AA)
        cv2.circle(tile, (px, py), 3, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                      (90, 90, 90), 1)
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, f"({x:.1f},{y:.1f})", (4, tile.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
                    cv2.LINE_AA)
        tiles.append(tile)

    tile_h, tile_w = tiles[0].shape[:2]
    n_rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((n_rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        canvas[row * tile_h:(row + 1) * tile_h,
               col * tile_w:(col + 1) * tile_w] = tile
    cv2.imwrite(path, canvas)


def _select_centroid_inspection_samples(
    positions: np.ndarray,
    valid_stars: np.ndarray,
    img_shape: tuple,
    per_region: int = 6,
    rng_seed: int = 0,
):
    """Pick a spread of valid-star indices across center/mid/corner regions."""
    h_img, w_img = img_shape
    cx, cy = w_img / 2.0, h_img / 2.0
    corner_radius = max(float(np.hypot(cx, cy)), 1.0)
    valid_indices = np.flatnonzero(valid_stars)
    if len(valid_indices) == 0:
        return np.empty((0,), dtype=np.int64), []

    norm_r = np.hypot(positions[valid_indices, 0] - cx,
                      positions[valid_indices, 1] - cy) / corner_radius
    rng = np.random.default_rng(rng_seed)
    selected = []
    labels = []
    for region_name, lo, hi in (("center", 0.0, 0.33), ("mid", 0.33, 0.66),
                                ("corner", 0.66, 1.01)):
        region_mask = (norm_r >= lo) & (norm_r < hi)
        region_indices = valid_indices[region_mask]
        if len(region_indices) == 0:
            continue
        count = min(per_region, len(region_indices))
        picked = rng.choice(region_indices, size=count, replace=False)
        selected.extend(picked.tolist())
        labels.extend([region_name] * count)
    return np.asarray(selected, dtype=np.int64), labels


def run_median_detection_debug(
    img_bgr: np.ndarray,
    mask_ext=None,
    median_ksize: int = 13,
    threshold_ratio: float = 1.0,
    open_ksize: int = 3,
    min_star_points: int = 400,
    min_area: float = 7.0,
    enable_outer_rescue: bool = True,
    output_dir: str = "debug_star_detection_median_out",
    display_scale: int = 2000,
    high_value_fraction: float = 0.30,
    centroid_inspect: bool = False,
    centroid_crop_size: int = 32,
    centroid_zoom: int = 8,
    centroid_per_region: int = 6,
):
    """Run the Norma median detector through its public alternative interface."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, "step")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = None if mask_ext is None else mask_ext > 0

    print(f"  Input image: {img_gray.shape[1]}x{img_gray.shape[0]}, "
          f"dtype={img_bgr.dtype}")
    max_eccentricity = 0.85
    print("  Detector: detect_star_points_median_detailed()")
    print(f"  median_ksize={median_ksize}, threshold_ratio={threshold_ratio}, "
          f"open_ksize={open_ksize}, min_area={min_area:g}")
    print(f"  outer_rescue={enable_outer_rescue} "
          "(enabled means no eccentricity rejection)")

    details = detect_star_points_median_detailed(
        img_gray,
        mask=mask,
        median_ksize=median_ksize,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        min_star_points=min_star_points,
        min_area=min_area,
        max_eccentricity=max_eccentricity,
        enable_outer_rescue=enable_outer_rescue,
    )
    detected = details.detected_stars
    analysis_mask = (np.ones(img_gray.shape, dtype=bool)
                     if mask is None else mask)

    save_image(f"{prefix}_0_original.png", img_bgr)
    if mask is not None:
        save_image(f"{prefix}_3_mask.png", mask.astype(np.uint8) * 255)
    save_image(f"{prefix}_4a_median_response.png", details.response)
    save_image(f"{prefix}_4b_binary.png", details.star_mask * np.uint8(255))
    _save_img_rec_distribution(
        f"{prefix}_4c_response_distribution.png",
        details.response,
        analysis_mask,
        details.threshold,
        threshold_label=f"sigma*{threshold_ratio:g}",
        title="median residual pixel distribution",
    )

    positions = details.candidate_positions
    areas = details.areas
    intensities = details.intensities
    eccentricities = details.eccentricities
    valid_stars = details.valid_stars
    strict_valid_stars = details.strict_valid_stars
    recovered_stars = details.rescued_stars
    if not 0.0 < high_value_fraction <= 1.0:
        raise ValueError("high_value_fraction must be in (0, 1]")
    high_value_stars = np.zeros(len(positions), dtype=bool)
    valid_indices = np.flatnonzero(valid_stars)
    if len(valid_indices):
        high_value_count = max(
            1, int(np.ceil(len(valid_indices) * high_value_fraction)))
        ranked = valid_indices[np.argpartition(
            intensities[valid_indices], -high_value_count)[-high_value_count:]]
        high_value_stars[ranked] = True
    fail_area_min = areas < min_area
    fail_ecc = eccentricities >= max_eccentricity
    fail_area_pct = areas <= details.area_percentile
    fail_intensity = intensities <= details.intensity_percentile

    if len(positions):
        _save_feature_distributions(
            f"{prefix}_5a_feature_distributions.png",
            areas,
            eccentricities,
            intensities,
            details.area_percentile,
            details.intensity_percentile,
            min_area=min_area,
            max_eccentricity=max_eccentricity,
        )
        _save_candidate_scatter(
            f"{prefix}_5b_candidate_scatter.png",
            areas,
            eccentricities,
            intensities,
            valid_stars,
            recovered_stars,
            fail_area_min,
            fail_ecc,
            fail_area_pct,
            fail_intensity,
            details.area_percentile,
            details.intensity_percentile,
            min_area=min_area,
            max_eccentricity=max_eccentricity,
        )

    display = to_display_uint8(img_bgr)
    h, w = display.shape[:2]
    scale = min(display_scale / max(h, w), 1.0)
    display_small = (
        cv2.resize(display, (int(w * scale), int(h * scale)))
        if scale < 1.0 else display.copy()
    )

    candidates_overlay = display_small.copy()
    for index, (x, y) in enumerate(positions):
        if recovered_stars[index]:
            color = (255, 210, 0)
        elif strict_valid_stars[index]:
            color = (0, 230, 0)
        elif fail_area_min[index]:
            color = (0, 0, 220)
        elif fail_ecc[index]:
            color = (0, 100, 255)
        elif fail_area_pct[index]:
            color = (0, 220, 220)
        else:
            color = (220, 100, 220)
        center = (int(round(x * scale)), int(round(y * scale)))
        cv2.circle(candidates_overlay, center, 4, color, 1, cv2.LINE_AA)
    save_image(f"{prefix}_5_candidates_overlay.png", candidates_overlay)

    valid_overlay = display_small.copy()
    for index in np.flatnonzero(valid_stars):
        x, y = positions[index]
        color = ((255, 210, 0) if recovered_stars[index]
                 else (0, 230, 0))
        center = (int(round(x * scale)), int(round(y * scale)))
        cv2.circle(valid_overlay, center, 5, color, 1, cv2.LINE_AA)
        cv2.circle(valid_overlay, center, 1, color, -1, cv2.LINE_AA)
    save_image(f"{prefix}_6_valid_overlay.png", valid_overlay)

    high_value_overlay = display_small.copy()
    for index in np.flatnonzero(high_value_stars):
        x, y = positions[index]
        center = (int(round(x * scale)), int(round(y * scale)))
        cv2.circle(high_value_overlay, center, 5, (0, 255, 255), 1,
                   cv2.LINE_AA)
        cv2.circle(high_value_overlay, center, 1, (0, 255, 255), -1,
                   cv2.LINE_AA)
    cv2.putText(
        high_value_overlay,
        f"top {high_value_fraction * 100:.1f}% valid stars by median intensity "
        f"({int(high_value_stars.sum())}/{int(valid_stars.sum())})",
        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
        cv2.LINE_AA)
    save_image(f"{prefix}_6b_high_value_overlay.png", high_value_overlay)

    csv_path = os.path.join(output_dir, "candidates.csv")
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write("idx,cx,cy,area,eccentricity,intensity,volume,strict_valid,"
                   "rescued,valid,high_value,fail_area_min,fail_ecc,"
                   "fail_area_pct,fail_intensity\n")
        for index, (x, y) in enumerate(positions):
            quality = (np.clip(
                (1.0 - eccentricities[index]) / (1.0 - max_eccentricity),
                0.15, 1.0) if recovered_stars[index] else 1.0)
            volume = areas[index] * intensities[index] * quality
            file.write(
                f"{index},{x:.4f},{y:.4f},{areas[index]:.6g},"
                f"{eccentricities[index]:.6g},{intensities[index]:.8g},"
                f"{volume:.8g},{int(strict_valid_stars[index])},"
                f"{int(recovered_stars[index])},{int(valid_stars[index])},"
                f"{int(high_value_stars[index])},"
                f"{int(fail_area_min[index])},{int(fail_ecc[index])},"
                f"{int(fail_area_pct[index])},{int(fail_intensity[index])}\n"
            )

    print(f"  Median response threshold: {details.threshold:.8g}")
    print(f"  Binary foreground pixels: {int(details.star_mask.sum())}")
    print(f"  Candidate contours: {len(positions)}")
    print(f"  Rejected by area < {min_area:g}: {int(fail_area_min.sum())}")
    print(f"  Rejected by eccentricity >= {max_eccentricity}: "
          f"{int(fail_ecc.sum())}")
    print(f"  Rejected by area p{STAR_FILTER_PERCENTILE}: "
          f"{int(fail_area_pct.sum())}")
    print(f"  Rejected by intensity p{STAR_FILTER_PERCENTILE}: "
          f"{int(fail_intensity.sum())}")
    print(f"  Strict detected stars: {int(strict_valid_stars.sum())}")
    print(f"  Eccentricity-filter rescued stars: {int(recovered_stars.sum())}")
    print(f"  Final detected stars: {len(detected.positions)}")
    if np.any(high_value_stars):
        high_positions = positions[high_value_stars]
        canvas_min = np.min(positions[valid_stars], axis=0)
        canvas_max = np.max(positions[valid_stars], axis=0)
        canvas_center = 0.5 * (canvas_min + canvas_max)
        canvas_radius = max(float(np.linalg.norm(
            0.5 * (canvas_max - canvas_min))), 1.0)
        relative = high_positions - canvas_center
        radius_norm = np.linalg.norm(relative, axis=1) / canvas_radius
        radial = (
            int(np.count_nonzero(radius_norm < 0.33)),
            int(np.count_nonzero((radius_norm >= 0.33) &
                                 (radius_norm < 0.66))),
            int(np.count_nonzero(radius_norm >= 0.66)),
        )
        sectors = np.floor(
            ((np.degrees(np.arctan2(relative[:, 1], relative[:, 0])) + 360.0)
             % 360.0) / 45.0).astype(np.int32)
        print(f"  High-value intensity top {high_value_fraction * 100:.1f}%: "
              f"{int(high_value_stars.sum())}, radial={radial[0]}/"
              f"{radial[1]}/{radial[2]}, sectors={len(np.unique(sectors))}/8")
    print(f"  [4a] median response saved")
    print(f"  [4b] binary mask saved")
    print(f"  [4c] response distribution saved")
    print(f"  [5] candidate overlay saved")
    if len(positions):
        print(f"  [5a] feature distributions saved")
        print(f"  [5b] candidate scatter saved")
    print(f"  [6] valid overlay: {prefix}_6_valid_overlay.png")
    print(f"  [6b] high-value overlay: {prefix}_6b_high_value_overlay.png")
    print(f"  [CSV] candidates.csv saved")

    if centroid_inspect and len(positions):
        sample_indices, sample_labels = _select_centroid_inspection_samples(
            positions, valid_stars, img_gray.shape,
            per_region=centroid_per_region)
        if len(sample_indices):
            sheet_labels = [
                f"{label}#{index}"
                for label, index in zip(sample_labels, sample_indices)
            ]
            _save_centroid_inspection_sheet(
                f"{prefix}_8_centroid_inspection.png",
                img_gray,
                positions[sample_indices],
                sheet_labels,
                crop_size=centroid_crop_size,
                zoom=centroid_zoom,
            )
            print(f"  [8] centroid inspection sheet saved "
                  f"({len(sample_indices)} samples, "
                  f"crop={centroid_crop_size}px, zoom={centroid_zoom}x)")
        else:
            print("  [8] centroid inspection skipped: no valid stars")

    print(f"\nDone. Output directory: {os.path.abspath(output_dir)}/")


def run_detection_debug(
    img_bgr: np.ndarray,
    mask_ext=None,
    resize_length: int = 10000,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
    min_star_points: int = 400,
    output_dir: str = "debug_star_detection_out",
    display_scale: int = 2000,
    density_recovery: bool = True,
    density_grid: tuple = (6, 6),
    density_threshold: float = 0.5,
    relaxed_ecc_max: float = 0.92,
):
    """Run the same pipeline as detect_star_points and save debug artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, "step")

    # 0. Original image
    img_gray = to_gray_float64(img_bgr)
    img_shape = img_gray.shape
    print(f"  Input image: {img_shape[1]}x{img_shape[0]}, dtype={img_bgr.dtype}")

    save_image(f"{prefix}_0_original.png", img_bgr)
    print(f"  [0] original saved")

    # 1. Gaussian blur + mean/range normalization
    img_blr = cv2.GaussianBlur(img_gray, (gaussian_ksize, gaussian_ksize), sigma)
    img_blr_mean = np.mean(img_blr)
    img_blr_range = np.max(img_blr) - np.min(img_blr)
    img_blr_norm = (img_blr - img_blr_mean) / img_blr_range

    save_image(f"{prefix}_1_blur_normalized.png", img_blr_norm.astype(np.float32))
    print(f"  [1] blur_normalized saved  (mean={img_blr_mean:.5f}, range={img_blr_range:.5f})")

    # 2. Choose resize_factor
    resize_factor = 1.0
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2
    print(f"  [2] initial resize_factor = {resize_factor:.4f}")

    # 3. Mask
    if mask_ext is None:
        mask = np.ones(img_shape, dtype=bool)
        print("  [3] auto mask disabled: using full-image mask")
    else:
        mask = mask_ext > 0
        print("  [3] using external mask only")
    mask_rate = np.sum(mask) * 100.0 / np.prod(mask.shape)
    print(f"  [3] mask computed: {mask_rate:.2f}% valid pixels")

    save_image(f"{prefix}_3_mask.png", (mask.astype(np.uint8) * 255))
    print(f"  [3] mask saved")

    # 4. Wavelet reconstruction loop
    final_resize_factor = resize_factor
    attempt = 0
    while True:
        attempt += 1
        img_rec = _wavelet_dec_rec(img_blr_norm, resize_factor=final_resize_factor) * mask
        threshold = np.percentile(img_rec[mask], 99.5)
        bw = ((img_rec > threshold) * mask).astype(np.uint8) * 255
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours_valid = [c for c in contours if len(c) > 5]
        print(f"  [4] attempt {attempt}: resize_factor={final_resize_factor:.4f}, "
              f"threshold={threshold:.6f}, contours={len(contours_valid)}")
        if len(contours_valid) < min_star_points and final_resize_factor < 1.0:
            final_resize_factor *= 2
            continue
        else:
            break

    print(f"  [4] final resize_factor = {final_resize_factor:.4f}")
    save_image(f"{prefix}_4a_wavelet_rec.png", img_rec.astype(np.float32))
    save_image(f"{prefix}_4b_binary.png", bw)
    _save_img_rec_distribution(f"{prefix}_4c_img_rec_distribution.png", img_rec, mask, threshold)
    print(f"  [4a] wavelet_rec saved")
    print(f"  [4b] binary saved")
    print(f"  [4c] img_rec_distribution saved")

    # 5. Ellipse fitting + feature extraction
    contours = contours_valid
    if not contours:
        print("  [!] No candidate contours detected; exiting")
        return

    elps = [cv2.fitEllipse(c) for c in contours]
    centroids = np.array([e[0] for e in elps])
    areas = np.array([cv2.contourArea(c) + 0.5 * len(c) for c in contours])

    # Same eccentricity formula as detection.py
    eccentricities = np.sqrt(
        np.array([1 - (elp[1][0] / elp[1][1])**2 for elp in elps]))

    mask_img = np.zeros(bw.shape, np.uint8)
    intensities = np.zeros(len(contours))
    for i, c in enumerate(contours):
        cv2.drawContours(mask_img, contours, i, 255, -1)
        rect = cv2.boundingRect(c)
        val = cv2.mean(
            img_rec[rect[1]:rect[1] + rect[3] + 1,
                    rect[0]:rect[0] + rect[2] + 1],
            mask_img[rect[1]:rect[1] + rect[3] + 1,
                     rect[0]:rect[0] + rect[2] + 1])
        mask_img[rect[1]:rect[1] + rect[3] + 1,
                 rect[0]:rect[0] + rect[2] + 1] = 0
        intensities[i] = val[0]

    # 6. Filtering (same as detect_star_points)
    area_pct = np.percentile(areas, STAR_FILTER_PERCENTILE)
    intensity_pct = np.percentile(intensities, STAR_FILTER_PERCENTILE)
    _save_feature_distributions(
        f"{prefix}_5a_feature_distributions.png",
        areas,
        eccentricities,
        intensities,
        area_pct,
        intensity_pct,
    )
    print(f"  [5a] feature_distributions saved")

    fail_area_min  = areas < MIN_STAR_AREA
    fail_ecc       = eccentricities >= 0.8
    fail_area_pct  = areas < area_pct
    fail_intensity = intensities < intensity_pct
    valid_stars    = ~(fail_area_min | fail_ecc | fail_area_pct | fail_intensity)

    n_total = len(contours)
    print(f"\n  === Base filter summary ===")
    print(f"  Total candidates:       {n_total}")
    print(f"  area<{MIN_STAR_AREA} rejected:      {int(fail_area_min.sum())}")
    print(f"  eccentricity>=0.8: {int(fail_ecc.sum())}")
    print(
        f"  area<pct{STAR_FILTER_PERCENTILE} rejected:   {int(fail_area_pct.sum())}  "
        f"(pct{STAR_FILTER_PERCENTILE}={area_pct:.2f})"
    )
    print(
        f"  intensity<pct{STAR_FILTER_PERCENTILE}:   {int(fail_intensity.sum())}  "
        f"(pct{STAR_FILTER_PERCENTILE}={intensity_pct:.6f})"
    )
    print(f"  Valid stars after filtering: {int(valid_stars.sum())}")
    print(f"  Note: one candidate may fail multiple rules")

    # 7. Density recovery pass
    recovered_stars = np.zeros(n_total, dtype=bool)
    cell_counts_before = np.zeros(density_grid, dtype=int)
    cell_counts_after  = np.zeros(density_grid, dtype=int)
    cell_mask_frac     = np.zeros(density_grid)
    triggered_cells    = np.zeros(density_grid, dtype=bool)
    avg_density        = 0.0

    if density_recovery:
        grid_h, grid_w = density_grid
        h_img, w_img = img_shape
        ys = np.linspace(0, h_img, grid_h + 1)
        xs = np.linspace(0, w_img, grid_w + 1)

        # Mask coverage per cell
        for gy in range(grid_h):
            for gx in range(grid_w):
                patch = mask[int(ys[gy]):int(ys[gy + 1]),
                             int(xs[gx]):int(xs[gx + 1])]
                cell_mask_frac[gy, gx] = patch.mean() if patch.size > 0 else 0.0

        # Base valid-star count before recovery
        for i in np.where(valid_stars)[0]:
            cx, cy = centroids[i]
            gx = min(int(cx / w_img * grid_w), grid_w - 1)
            gy = min(int(cy / h_img * grid_h), grid_h - 1)
            cell_counts_before[gy, gx] += 1

        cell_counts_after = cell_counts_before.copy()

        # Reference density: p75 among sky cells (mask coverage >= 20%)
        sky_cells = cell_mask_frac >= 0.2
        if sky_cells.any() and cell_counts_before[sky_cells].sum() > 0:
            avg_density = float(np.percentile(cell_counts_before[sky_cells], 75))

        print(f"\n  === Density recovery grid ({grid_h}x{grid_w}) ===")
        print(f"  Reference density (p75 of sky cells): {avg_density:.1f} stars/cell")

        if avg_density >= 1:
            trigger_threshold = avg_density * density_threshold
            print(f"  Recovery trigger threshold: < {trigger_threshold:.1f} stars/cell")
            print(f"  Relaxed eccentricity cap: {relaxed_ecc_max}")

            ecc_only = fail_ecc & ~fail_area_min & ~fail_area_pct & ~fail_intensity
            print(f"  Eccentricity-only failures: {int(ecc_only.sum())}")

            # Pre-mark sparse cells from the baseline counts only.
            # This avoids recovery order changing which stars get restored.
            sparse_cells = sky_cells & (cell_counts_before < trigger_threshold)

            for i in np.where(ecc_only)[0]:
                if eccentricities[i] >= relaxed_ecc_max:
                    continue
                cx, cy = centroids[i]
                gx = min(int(cx / w_img * grid_w), grid_w - 1)
                gy = min(int(cy / h_img * grid_h), grid_h - 1)
                if sparse_cells[gy, gx]:
                    valid_stars[i] = True
                    recovered_stars[i] = True
                    cell_counts_after[gy, gx] += 1
                    triggered_cells[gy, gx] = True

            print(f"  Triggered cells: {int(triggered_cells.sum())}")
            print(f"  Recovered stars: {int(recovered_stars.sum())}")
        else:
            print(f"  avg_density < 1, skip recovery")

        # Density grid ASCII table
        print(f"\n  Cell density table (before -> after | ---=non-sky  *=triggered):")
        col_w = 9
        header = "      " + "".join(f"  C{gx:<{col_w-2}}" for gx in range(grid_w))
        print(header)
        for gy in range(grid_h):
            row = f"  R{gy}  "
            for gx in range(grid_w):
                frac = cell_mask_frac[gy, gx]
                if frac < 0.2:
                    cell_str = "---"
                elif triggered_cells[gy, gx]:
                    b, a = cell_counts_before[gy, gx], cell_counts_after[gy, gx]
                    cell_str = f"*{b}->{a}"
                else:
                    cell_str = str(cell_counts_before[gy, gx])
                row += f"  {cell_str:<{col_w}}"
            print(row)

    n_valid    = int(valid_stars.sum())
    n_rejected = n_total - n_valid
    print(f"\n  Final valid stars: {n_valid}  (recovered {int(recovered_stars.sum())})")

    # 8. Overlay images
    display = to_display_uint8(img_bgr)
    h_d, w_d = display.shape[:2]
    scale = min(display_scale / max(h_d, w_d), 1.0)
    dw, dh = int(w_d * scale), int(h_d * scale)
    display_small = cv2.resize(display, (dw, dh)) if scale < 1.0 else display.copy()

    # Color mapping:
    # green     = valid from original filter
    # cyan      = valid after density recovery
    # red       = rejected by min area
    # orange    = rejected by eccentricity
    # yellow    = rejected by area percentile
    # magenta   = rejected by intensity percentile
    def get_color(i):
        if recovered_stars[i]:
            return (255, 210, 0)          # cyan: recovered
        if valid_stars[i]:
            return (0, 230, 0)            # green: kept
        if fail_area_min[i]:
            return (0, 0, 220)            # red: area too small
        if fail_ecc[i]:
            return (0, 100, 255)          # orange: eccentricity too high
        if fail_area_pct[i]:
            return (0, 220, 220)          # yellow: area below percentile
        return (220, 100, 220)            # magenta: intensity below percentile

    # All-candidate overlay
    overlay = display_small.copy()
    for i, elp in enumerate(elps):
        color = get_color(i)
        cx, cy = centroids[i]
        center_s = (int(elp[0][0] * scale), int(elp[0][1] * scale))
        axes_s   = (max(1, int(elp[1][0] * scale / 2)),
                    max(1, int(elp[1][1] * scale / 2)))
        cv2.ellipse(overlay, center_s, axes_s, elp[2], 0, 360, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, (int(cx * scale), int(cy * scale)), 2, color, -1, cv2.LINE_AA)

    save_image(f"{prefix}_5_candidates_overlay.png", overlay)
    _save_candidate_scatter(
        f"{prefix}_5b_candidate_scatter.png",
        areas,
        eccentricities,
        intensities,
        valid_stars,
        recovered_stars,
        fail_area_min,
        fail_ecc,
        fail_area_pct,
        fail_intensity,
        area_pct,
        intensity_pct,
    )
    print(f"\n  [5] candidates_overlay saved  "
          f"({int((valid_stars & ~recovered_stars).sum())} kept, "
          f"{int(recovered_stars.sum())} recovered, "
          f"{n_rejected} rejected)")
    print(f"  [5b] candidate_scatter saved")

    # Valid-star overlay only
    overlay_valid = display_small.copy()
    for i in range(n_total):
        if not valid_stars[i]:
            continue
        elp = elps[i]
        color = (255, 210, 0) if recovered_stars[i] else (0, 230, 0)
        center_s = (int(elp[0][0] * scale), int(elp[0][1] * scale))
        axes_s   = (max(1, int(elp[1][0] * scale / 2)),
                    max(1, int(elp[1][1] * scale / 2)))
        cv2.ellipse(overlay_valid, center_s, axes_s, elp[2], 0, 360,
                    color, 1, cv2.LINE_AA)
        cv2.circle(overlay_valid, (int(centroids[i, 0] * scale),
                                   int(centroids[i, 1] * scale)),
                   2, color, -1, cv2.LINE_AA)

    save_image(f"{prefix}_6_valid_overlay.png", overlay_valid)
    print(f"  [6] valid_overlay saved")

    # Density grid visualizations
    if density_recovery:
        _save_density_grid(
            f"{prefix}_7_density_grid.png",
            display_small,
            density_grid,
            cell_counts_before, cell_counts_after,
            cell_mask_frac, triggered_cells,
            avg_density, density_threshold,
        )
        _save_density_distribution(
            f"{prefix}_7b_density_distribution.png",
            cell_counts_before,
            cell_counts_after,
            cell_mask_frac,
            triggered_cells,
            avg_density,
            density_threshold,
        )
        print(f"  [7] density_grid saved")
        print(f"  [7b] density_distribution saved")

    # Legend
    _save_legend(f"{prefix}_legend.png", density_recovery)

    # CSV export
    csv_path = os.path.join(output_dir, "candidates.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("idx,cx,cy,area,eccentricity,intensity,volume,valid,recovered,"
                "fail_area_min,fail_ecc,fail_area_pct,fail_intensity\n")
        for i in range(n_total):
            volume = areas[i] * intensities[i] if valid_stars[i] else 0
            f.write(f"{i},{centroids[i,0]:.2f},{centroids[i,1]:.2f},"
                    f"{areas[i]:.2f},{eccentricities[i]:.4f},{intensities[i]:.6f},"
                    f"{volume:.4f},{int(valid_stars[i])},{int(recovered_stars[i])},"
                    f"{int(fail_area_min[i])},{int(fail_ecc[i])},"
                    f"{int(fail_area_pct[i])},{int(fail_intensity[i])}\n")
    print(f"  [CSV] candidates.csv saved  ({n_total} rows)")

    print(f"\nDone. Output directory: {os.path.abspath(output_dir)}/")


def _save_legend(path: str, density_recovery: bool):
    items = [
        ((0, 230, 0),    "Valid star (original)"),
    ]
    if density_recovery:
        items.append(((255, 210, 0), "Valid star (density recovery)"))
    items += [
        ((0, 0, 220),    f"Rejected: area < {MIN_STAR_AREA}"),
        ((0, 100, 255),  "Rejected: eccentricity >= 0.8"),
        ((0, 220, 220),  f"Rejected: area < percentile({STAR_FILTER_PERCENTILE})"),
        ((220, 100, 220),f"Rejected: intensity < percentile({STAR_FILTER_PERCENTILE})"),
    ]
    h_item = 28
    w_legend = 380
    h_legend = h_item * len(items) + 10
    img = np.zeros((h_legend, w_legend, 3), dtype=np.uint8)
    for i, (color, label) in enumerate(items):
        y = 8 + i * h_item
        cv2.rectangle(img, (8, y), (24, y + 16), color, -1)
        cv2.putText(img, label, (32, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Input image path")
    p.add_argument("--output-dir", default="debug_star_detection_median_out")
    p.add_argument("--mask", default=None)
    p.add_argument("--detector", choices=("median", "wavelet-debug"),
                   default="median",
                   help="Detection path (default: median)")
    p.add_argument("--median-ksize", type=int, default=13,
                   help="Median background kernel size (default: 13)")
    p.add_argument("--threshold-ratio", type=float, default=1.0,
                   help="Residual sigma multiplier (default: 5.0)")
    p.add_argument("--open-ksize", type=int, default=3,
                   help="Morphological opening kernel size (default: 3)")
    p.add_argument("--min-area", type=float, default=7.0,
                   help="Median candidate minimum area (default: 7)")
    p.add_argument("--outer-rescue", action="store_true",
                   help="Disable eccentricity rejection for median candidates")
    p.add_argument("--resize-length", type=int, default=10000)
    p.add_argument("--gaussian-ksize", type=int, default=9)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--min-star-points", type=int, default=400)
    p.add_argument("--display-scale", type=int, default=2000)
    p.add_argument("--high-value-fraction", type=float, default=0.30,
                   help="Top valid-star fraction by median intensity to highlight")
    p.add_argument("--no-density-recovery", action="store_true",
                   help="Disable density recovery")
    p.add_argument("--density-grid", type=int, nargs=2, default=[6, 6],
                   metavar=("H", "W"), help="Density recovery grid size (default: 6 6)")
    p.add_argument("--density-threshold", type=float, default=0.5,
                   help="Recovery trigger factor (default: 0.5)")
    p.add_argument("--relaxed-ecc-max", type=float, default=0.92,
                   help="Relaxed eccentricity cap (default: 0.92)")
    p.add_argument("--centroid-inspect", action="store_true",
                   help="Save a full-resolution centroid-crosshair contact "
                        "sheet sampled across center/mid/corner regions "
                        "(median detector only)")
    p.add_argument("--centroid-crop-size", type=int, default=32,
                   help="Crop size in source pixels around each sampled "
                        "centroid (default: 32)")
    p.add_argument("--centroid-zoom", type=int, default=8,
                   help="Zoom factor applied to each crop tile (default: 8)")
    p.add_argument("--centroid-per-region", type=int, default=6,
                   help="Samples per center/mid/corner region (default: 6)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input} ...")
    img = load_image(args.input)

    mask_ext = None
    if args.mask:
        print(f"Loading mask {args.mask} ...")
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(f"Failed to read mask: {args.mask}")
        mask_ext = mask_img

    if args.detector == "median":
        run_median_detection_debug(
            img_bgr=img,
            mask_ext=mask_ext,
            median_ksize=args.median_ksize,
            threshold_ratio=args.threshold_ratio,
            open_ksize=args.open_ksize,
            min_star_points=args.min_star_points,
            min_area=args.min_area,
            enable_outer_rescue=args.outer_rescue,
            output_dir=args.output_dir,
            display_scale=args.display_scale,
            high_value_fraction=args.high_value_fraction,
            centroid_inspect=args.centroid_inspect,
            centroid_crop_size=args.centroid_crop_size,
            centroid_zoom=args.centroid_zoom,
            centroid_per_region=args.centroid_per_region,
        )
    else:
        run_detection_debug(
            img_bgr=img,
            mask_ext=mask_ext,
            resize_length=args.resize_length,
            gaussian_ksize=args.gaussian_ksize,
            sigma=args.sigma,
            min_star_points=args.min_star_points,
            output_dir=args.output_dir,
            display_scale=args.display_scale,
            density_recovery=not args.no_density_recovery,
            density_grid=tuple(args.density_grid),
            density_threshold=args.density_threshold,
            relaxed_ecc_max=args.relaxed_ecc_max,
        )


if __name__ == "__main__":
    main()
