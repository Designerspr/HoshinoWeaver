from functools import lru_cache

from functools import lru_cache

import numpy as np
import pytest

from hoshicore.component.norma import detection
from hoshicore.component.norma.wavelet_recovery import (
    WaveletRecoveryStats, candidate_recovery_stats, probe_wavelet_recovery,
    select_wavelet_recovery)


@pytest.mark.parametrize("stats,expected", [
    (WaveletRecoveryStats(128, 96, 32), True),
    (WaveletRecoveryStats(129, 97, 32), False),
    (WaveletRecoveryStats(63, 0, 63), False),
    (WaveletRecoveryStats(100, 69, 31), False),
    (WaveletRecoveryStats(0, 0, 0), False),
])
def test_gate_requires_fraction_and_absolute_evidence(stats, expected):
    assert stats.needs_recovery is expected


def test_shape_gate_excludes_other_rejections_and_unrecoverable_trails():
    areas = np.array([1, 2, 20, 21, 22, 23, 24, 25, 26, 27.])
    intensities = np.arange(1., 11.)
    eccentricities = np.array([.9, .9, .7, .8, .949, .95, .999, .9, np.nan, .7])
    stats = candidate_recovery_stats(areas, intensities, eccentricities)
    assert stats == WaveletRecoveryStats(7, 2, 3)
    assert candidate_recovery_stats(np.array([]), np.array([]), np.array([])) == WaveletRecoveryStats(0, 0, 0)


def test_sequence_gate_requires_majority_and_handles_short_sequences():
    yes = WaveletRecoveryStats(128, 64, 64)
    no = WaveletRecoveryStats(128, 120, 8)
    assert select_wavelet_recovery([no, yes, yes]) == "relaxed"
    assert select_wavelet_recovery([yes, no, no]) == "strict"
    assert select_wavelet_recovery([yes, no]) == "strict"
    assert select_wavelet_recovery([yes]) == "relaxed"
    assert select_wavelet_recovery([]) == "strict"


@lru_cache(maxsize=2)
@lru_cache(maxsize=2)
def _field(height=320, width=480):
    rng = np.random.default_rng(9)
    yy, xx = np.mgrid[:height, :width]
    field = rng.normal(.03, .0001, size=xx.shape)
    for i, (y, x) in enumerate((y, x) for y in range(24, height - 20, 24) for x in range(24, width - 24, 24)):
        sigma_x = 4.5 if i % 3 else 1.8
        field += rng.uniform(.15, .65) * np.exp(-((xx-x)**2/sigma_x**2 + (yy-y)**2/1.8**2)/2)
    return field


@pytest.mark.parametrize("mode", ["contour", "auto", "native_relaxed"])
def test_probe_preserves_original_strict_backend_result(mode):
    gray = _field()
    mask = np.ones(gray.shape, np.uint8)
    mask[:, :20] = 0
    original = detection.detect_star_points(gray, mask, mode=mode)
    probe = probe_wavelet_recovery(gray, mask, mode=mode)
    assert len(original.positions) > 0
    np.testing.assert_array_equal(probe.strict_stars.positions, original.positions)
    np.testing.assert_allclose(probe.strict_stars.volumes, original.volumes, rtol=1e-12)
    assert np.all(probe.relaxed_stars.positions[:, 0] >= 20)


def test_contour_probe_reuses_wavelet_response(monkeypatch):
    original = detection._wavelet_dec_rec
    calls = []
    def wavelet(*args, **kwargs):
        calls.append(kwargs["resize_factor"])
        return original(*args, **kwargs)
    monkeypatch.setattr(detection, "_wavelet_dec_rec", wavelet)
    probe_wavelet_recovery(_field(), mode="contour")
    assert calls == [1.0]


def test_constant_image_and_invalid_mask():
    gray = np.zeros((32, 48))
    probe = probe_wavelet_recovery(gray)
    assert probe.stats == WaveletRecoveryStats(0, 0, 0)
    assert len(probe.strict_stars.positions) == len(probe.relaxed_stars.positions) == 0
    with pytest.raises(ValueError, match="mask"):
        probe_wavelet_recovery(gray, np.zeros(gray.shape))


@pytest.mark.parametrize("mode", ["contour", "auto", "native_relaxed"])
def test_cache_recovers_real_deformed_field_before_geometry(mode):
    from hoshicore.component.norma.geometry_view import GeometryView, StarDetectionCache
    from hoshicore.component.norma.types import CameraModel, Intrinsics

    gray = _field(640, 960)
    probe = probe_wavelet_recovery(gray, mode=mode)
    assert probe.stats.needs_recovery
    cache = StarDetectionCache(gray, star_detection_mode=mode)
    camera = CameraModel(Intrinsics(20, 36, 24, 960, 640))
    geometry = GeometryView(cache.pywt_stars, camera)
    assert cache.star_detection_profile == "relaxed"
    assert cache.star_recovery_stats == probe.stats
    assert len(geometry.positions) > len(probe.strict_stars.positions)
    np.testing.assert_array_equal(geometry.positions, probe.relaxed_stars.positions)
    np.testing.assert_array_equal(geometry.volumes, probe.relaxed_stars.volumes)
    assert cache.pywt_stars is geometry.stars


@pytest.mark.parametrize("mode", ["contour", "auto", "native_relaxed"])
def test_nontriggering_cache_preserves_strict_observations(mode):
    from hoshicore.component.norma.geometry_view import StarDetectionCache

    gray = _field()
    strict = detection.detect_star_points(gray, mode=mode)
    cache = StarDetectionCache(gray, star_detection_mode=mode)
    np.testing.assert_array_equal(cache.pywt_stars.positions, strict.positions)
    np.testing.assert_allclose(cache.pywt_stars.volumes, strict.volumes, rtol=1e-12)
    assert cache.star_detection_profile == "strict"
    assert not cache.star_recovery_stats.needs_recovery


def test_nontriggering_native_probe_skips_relaxed_detection(monkeypatch):
    import hoshicore.component.norma.wavelet_recovery as module

    def unexpected(*args, **kwargs):
        raise AssertionError("normal images must not run a second wavelet detection")

    monkeypatch.setattr(module, "detect_recovered_wavelet_stars", unexpected)
    probe = module.probe_wavelet_recovery(_field(), mode="auto", only_if_needed=True)
    assert not probe.stats.needs_recovery
    assert probe.strict_stars is probe.relaxed_stars


def test_recovery_opt_out_restores_strict_deformed_field():
    from hoshicore.component.norma.geometry_view import StarDetectionCache

    gray = _field(640, 960)
    strict = detection.detect_star_points(gray, mode="contour")
    cache = StarDetectionCache(gray, star_detection_mode="contour", star_detection_recovery=False)
    np.testing.assert_array_equal(cache.pywt_stars.positions, strict.positions)
    np.testing.assert_array_equal(cache.pywt_stars.volumes, strict.volumes)
    assert cache.star_detection_profile == "strict"
    assert cache.star_recovery_stats is None