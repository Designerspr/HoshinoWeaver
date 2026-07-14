import cv2
import numpy as np

from bench.data_tools.starfield import make_starfield_base
from hoshicore._custom_op.ops import detection as detection_ops
from hoshicore.component.norma.detection import _detect_star_points_opencv
import hoshicore.component.norma.fast_detection as fast_detection
from hoshicore.component.norma.fast_detection import (
    ComponentStarCandidates,
    detect_star_points_connected_components,
    detect_stars_connected_components,
    extract_connected_component_candidates,
    filter_component_star_candidates,
)
from hoshicore.component.norma.geometry_view import to_gray_f64


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = source[:, None, :] - target[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(dist, axis=1)


def _use_compiled_cc(monkeypatch) -> None:
    monkeypatch.delenv("HNW_CUSTOM_OPS_FALLBACK", raising=False)
    detection_ops._select_star_detect_connected_components_backend.cache_clear()


def test_connected_component_candidates_measure_blob_centers(monkeypatch) -> None:
    _use_compiled_cc(monkeypatch)
    img_rec = np.zeros((96, 128), dtype=np.float64)
    bw = np.zeros(img_rec.shape, dtype=np.uint8)
    centers = np.array([[25.0, 30.0], [70.0, 50.0], [100.0, 72.0]])
    radii = [4, 5, 6]
    for idx, (center, radius) in enumerate(zip(centers, radii)):
        xy = tuple(center.astype(int))
        cv2.circle(bw, xy, radius, 255, -1)
        cv2.circle(img_rec, xy, radius, float(idx + 1), -1)

    candidates = extract_connected_component_candidates(img_rec, bw)

    assert len(candidates.positions) == 3
    assert np.max(_nearest_distances(centers, candidates.positions)) < 0.1
    assert np.all(candidates.areas > 20)
    assert np.all(candidates.intensities > 0)


def test_connected_component_detection_handles_empty_foreground(monkeypatch) -> None:
    _use_compiled_cc(monkeypatch)
    img_rec = np.zeros((32, 48), dtype=np.float64)
    bw = np.zeros(img_rec.shape, dtype=np.uint8)

    detected = detect_stars_connected_components(img_rec, bw)

    assert detected.positions.shape == (0, 2)
    assert detected.volumes.shape == (0,)


def test_component_candidate_filter_preserves_volume_semantics() -> None:
    candidates = ComponentStarCandidates(
        positions=np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
        areas=np.array([10.0, 30.0, 40.0, 50.0, 60.0]),
        intensities=np.array([10.0, 10.0, 1.0, 30.0, 40.0]),
        eccentricities=np.array([0.1, 0.9, 0.1, 0.1, 0.1]),
    )

    detected = filter_component_star_candidates(candidates)

    np.testing.assert_array_equal(
        detected.positions,
        np.array([[3.0, 0.0], [4.0, 0.0]]),
    )
    np.testing.assert_allclose(detected.volumes, np.array([1500.0, 2400.0]))


def test_connected_component_detector_retries_on_candidate_count(monkeypatch) -> None:
    monkeypatch.setattr(
        fast_detection,
        "_prepare_detection_inputs",
        lambda *args, **kwargs: (
            np.zeros((4, 4), dtype=np.float64),
            np.ones((4, 4), dtype=bool),
            0.25,
        ),
    )

    resize_factors: list[float] = []

    def fake_bandpass(img_blr, mask, resize_factor):
        resize_factors.append(resize_factor)
        return np.zeros((4, 4), dtype=np.float64), np.zeros((4, 4), dtype=np.uint8)

    def fake_extract(img_rec, bw, **kwargs):
        count = 2 if len(resize_factors) == 1 else 5
        return ComponentStarCandidates(
            positions=np.zeros((count, 2), dtype=np.float64),
            areas=np.full(count, 30.0, dtype=np.float64),
            intensities=np.ones(count, dtype=np.float64),
            eccentricities=np.zeros(count, dtype=np.float64),
        )

    monkeypatch.setattr(
        fast_detection, "star_detect_bandpass_threshold_morph_numpy",
        fake_bandpass,
    )
    monkeypatch.setattr(
        fast_detection, "extract_connected_component_candidates", fake_extract)

    detected = detect_star_points_connected_components(
        np.zeros((4, 4), dtype=np.float64),
        min_star_points=4,
    )

    assert resize_factors == [0.25, 0.5]
    assert detected.positions.shape == (0, 2)


def test_connected_component_detector_tracks_contour_detector_on_synthetic(
        monkeypatch) -> None:
    _use_compiled_cc(monkeypatch)
    frame = make_starfield_base(
        height=512,
        width=768,
        stars=280,
        seed=7,
        dtype=np.uint8,
        channels=3,
    )
    gray = to_gray_f64(frame)

    contour = _detect_star_points_opencv(gray, min_star_points=0)
    cc = detect_star_points_connected_components(gray, min_star_points=0)

    assert len(contour.positions) > 20
    assert len(cc.positions) > 20
    count_ratio = abs(len(cc.positions) - len(contour.positions)) / len(
        contour.positions)
    assert count_ratio <= 0.25
    distances = _nearest_distances(contour.positions, cc.positions)
    assert np.mean(distances <= 1.0) >= 0.95
    assert np.percentile(distances, 95) <= 0.25
