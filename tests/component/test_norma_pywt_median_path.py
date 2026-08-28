"""Tests for the experimental pywt-bootstrap/median-guided path."""
import sys

import numpy as np
import pytest

from hoshicore.component.norma.alignment import AlignmentResult
from hoshicore.component.norma.detection import DetectedStars
from hoshicore.component.norma.frame_align import (
    AlignmentCameraCandidate,
    AlignmentError,
    DEFAULT_MATCHING_PATH,
    MATCHING_PATH_ASTERISM,
    MATCHING_PATH_HISTOGRAM,
    solve_pywt_alignment,
)
from hoshicore.component.norma.matching import MatchResult
from hoshicore.component.norma.optimization import CameraOptimizationPolicy
from hoshicore.component.norma.types import CameraModel, Intrinsics


def _camera() -> CameraModel:
    return CameraModel(intrinsics=Intrinsics(
        focal_length_mm=20.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=200,
        image_height_px=120,
    ))


def _stars(count: int, offset: float = 0.0) -> DetectedStars:
    positions = np.column_stack((
        np.linspace(10.0, 190.0, count),
        np.linspace(10.0 + offset, 110.0 + offset, count),
    ))
    return DetectedStars(
        positions=positions,
        volumes=np.ones(count, dtype=np.float64),
    )


def _match(ref_geo, src_geo, count: int, high_index: bool = False):
    if high_index:
        ref_idx = np.arange(len(ref_geo.positions) - count,
                            len(ref_geo.positions), dtype=np.int32)
        src_idx = np.arange(len(src_geo.positions) - count,
                            len(src_geo.positions), dtype=np.int32)
    else:
        ref_idx = np.arange(count, dtype=np.int32)
        src_idx = np.arange(count, dtype=np.int32)
    pair_idx = np.column_stack((ref_idx, src_idx))
    return MatchResult(
        pair_idx=pair_idx,
        ref_pts=ref_geo.positions[ref_idx],
        src_pts=src_geo.positions[src_idx],
        rotation=np.eye(3),
        initial_pair_count=count,
    )


def _candidate(camera: CameraModel) -> AlignmentCameraCandidate:
    return AlignmentCameraCandidate(
        camera=camera,
        optimization_policy=CameraOptimizationPolicy(),
        init_source="test",
    )


def test_dual_path_keeps_bootstrap_and_dense_index_spaces_separate(
        monkeypatch):
    import hoshicore.component.norma.frame_align as module
    import hoshicore.component.norma.alignment as alignment_module

    camera = _camera()
    pywt_calls = []
    median_calls = []
    optimize_matches = []

    def fake_pywt(image, mask=None):
        pywt_calls.append(image)
        return _stars(24, offset=float(len(pywt_calls)))

    def fake_median(image, mask=None, threshold_ratio=1.0):
        median_calls.append((image, threshold_ratio))
        return _stars(32, offset=float(len(median_calls)))

    def fake_select(ref_geo, src_geo, ref_candidate, src_candidate,
                    bootstrap_scales, same_camera=False, **kwargs):
        assert len(ref_geo.positions) == 24
        assert len(src_geo.positions) == 24
        return (ref_geo, src_geo, ref_candidate, src_candidate,
                _match(ref_geo, src_geo, 8))

    def fake_guided(ref_geo, src_geo, alignment, max_distance_px=8.0):
        assert len(ref_geo.positions) == 32
        assert len(src_geo.positions) == 32
        return _match(ref_geo, src_geo, 7, high_index=True)

    def fake_optimize(match, ref_camera, src_camera, **kwargs):
        optimize_matches.append(match)
        return AlignmentResult(np.eye(3), ref_camera, src_camera)

    def fake_residual_p90(match, alignment):
        return 0.0

    monkeypatch.setattr(module, "detect_star_points", fake_pywt)
    monkeypatch.setattr(module, "detect_star_points_median", fake_median)
    monkeypatch.setattr(module, "_select_initial_alignment_candidate",
                        fake_select)
    monkeypatch.setattr(alignment_module, "guided_mutual_rematch",
                        fake_guided)
    monkeypatch.setattr(module, "optimize_alignment", fake_optimize)
    monkeypatch.setattr(alignment_module, "optimize_alignment", fake_optimize)
    monkeypatch.setattr(alignment_module, "_guided_rematch_residual_p90",
                        fake_residual_p90)
    monkeypatch.setattr(
        module.GeometryView,
        "features",
        property(lambda self: (_ for _ in ()).throw(
            AssertionError("dual-path solver must not read features directly"))),
    )

    image = np.zeros((120, 200), dtype=np.float64)
    result = solve_pywt_alignment(
        image,
        image,
        _candidate(camera),
        _candidate(camera),
        bootstrap_scales=(1.0,),
        guided_refine=True,
    )

    assert len(pywt_calls) == 2
    assert len(median_calls) == 2
    assert len(optimize_matches) == 2
    assert result.guided_status == "applied"
    assert len(result.bootstrap_ref_geo.positions) == 24
    assert len(result.dense_ref_geo.positions) == 32
    assert int(np.max(result.bootstrap_match.pair_idx)) < 24
    assert int(np.max(result.final_match.pair_idx)) >= 24


def test_dual_path_without_guided_refine_skips_median_detection(monkeypatch):
    import hoshicore.component.norma.frame_align as module

    camera = _camera()

    monkeypatch.setattr(module, "detect_star_points",
                        lambda image, mask=None: _stars(24))
    monkeypatch.setattr(
        module,
        "detect_star_points_median",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("median detection must be skipped")),
    )
    monkeypatch.setattr(
        module,
        "_select_initial_alignment_candidate",
        lambda ref_geo, src_geo, ref_candidate, src_candidate,
        bootstrap_scales, same_camera=False, **kwargs:
        (ref_geo, src_geo, ref_candidate, src_candidate,
         _match(ref_geo, src_geo, 8)),
    )
    monkeypatch.setattr(
        module,
        "optimize_alignment",
        lambda match, ref_camera, src_camera, **kwargs:
        AlignmentResult(np.eye(3), ref_camera, src_camera),
    )

    image = np.zeros((120, 200), dtype=np.float64)
    result = solve_pywt_alignment(
        image,
        image,
        _candidate(camera),
        _candidate(camera),
        bootstrap_scales=(1.0,),
        guided_refine=False,
    )

    assert result.guided_status == "disabled"
    assert result.dense_ref_geo is None
    assert result.dense_src_geo is None
    assert result.final_match is result.bootstrap_match


def test_dual_path_selects_asterism_bootstrap_matcher(monkeypatch):
    import hoshicore.component.norma.frame_align as module

    camera = _camera()
    selected_matchers = []
    monkeypatch.setattr(module, "detect_star_points",
                        lambda image, mask=None: _stars(24))

    def fake_select(ref_geo, src_geo, ref_candidate, src_candidate,
                    bootstrap_scales, same_camera=False,
                    match_function=None):
        selected_matchers.append(match_function)
        return (ref_geo, src_geo, ref_candidate, src_candidate,
                _match(ref_geo, src_geo, 8))

    monkeypatch.setattr(module, "_select_initial_alignment_candidate",
                        fake_select)
    monkeypatch.setattr(
        module,
        "optimize_alignment",
        lambda match, ref_camera, src_camera, **kwargs:
        AlignmentResult(np.eye(3), ref_camera, src_camera),
    )

    image = np.zeros((120, 200), dtype=np.float64)
    solve_pywt_alignment(
        image,
        image,
        _candidate(camera),
        _candidate(camera),
        bootstrap_scales=(1.0,),
        guided_refine=False,
        use_asterism_bootstrap=True,
    )

    assert selected_matchers == [module.match_star_pairs_asterism]


def test_dual_path_bootstrap_failure_does_not_fall_back_to_median(monkeypatch):
    import hoshicore.component.norma.frame_align as module

    camera = _camera()
    monkeypatch.setattr(module, "detect_star_points",
                        lambda image, mask=None: _stars(4))
    monkeypatch.setattr(
        module,
        "detect_star_points_median",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("median fallback must not run")),
    )

    image = np.zeros((120, 200), dtype=np.float64)
    with pytest.raises(AlignmentError, match="Insufficient stars"):
        solve_pywt_alignment(
            image,
            image,
            _candidate(camera),
            _candidate(camera),
            bootstrap_scales=(1.0,),
            guided_refine=True,
        )


def test_dual_path_guided_failure_falls_back_to_pywt_result(monkeypatch):
    import hoshicore.component.norma.frame_align as module
    import hoshicore.component.norma.alignment as alignment_module

    camera = _camera()
    monkeypatch.setattr(module, "detect_star_points",
                        lambda image, mask=None: _stars(24))
    monkeypatch.setattr(
        module, "detect_star_points_median",
        lambda image, mask=None, threshold_ratio=1.0: _stars(32))
    monkeypatch.setattr(
        module,
        "_select_initial_alignment_candidate",
        lambda ref_geo, src_geo, ref_candidate, src_candidate,
        bootstrap_scales, same_camera=False, **kwargs:
        (ref_geo, src_geo, ref_candidate, src_candidate,
         _match(ref_geo, src_geo, 8)),
    )
    bootstrap_alignment = AlignmentResult(np.eye(3), camera, camera)
    monkeypatch.setattr(module, "optimize_alignment",
                        lambda *args, **kwargs: bootstrap_alignment)
    monkeypatch.setattr(
        alignment_module,
        "guided_mutual_rematch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("guided failed")),
    )

    image = np.zeros((120, 200), dtype=np.float64)
    result = solve_pywt_alignment(
        image,
        image,
        _candidate(camera),
        _candidate(camera),
        bootstrap_scales=(1.0,),
        guided_refine=True,
    )

    assert result.guided_status == "failed_fallback"
    assert result.guided_error == "guided failed"
    assert result.final_alignment is bootstrap_alignment
    assert result.final_match is result.bootstrap_match


def test_debug_and_benchmark_matching_path_cli_defaults_and_overrides(
        monkeypatch):
    import benchmark_norma_alignment as benchmark
    import debug_fisheye_two_frame_align as debug

    monkeypatch.setattr(sys, "argv", [
        "debug_fisheye_two_frame_align.py",
        "--reference", "ref.tif",
        "--source", "src.tif",
        "--output", "out.tif",
    ])
    assert debug.parse_args().matching_path == DEFAULT_MATCHING_PATH

    monkeypatch.setattr(sys, "argv", [
        "debug_fisheye_two_frame_align.py",
        "--reference", "ref.tif",
        "--source", "src.tif",
        "--output", "out.tif",
        "--matching-path", MATCHING_PATH_HISTOGRAM,
    ])
    assert (debug.parse_args().matching_path
            == MATCHING_PATH_HISTOGRAM)

    monkeypatch.setattr(sys, "argv", [
        "benchmark_norma_alignment.py", "dataset.json"
    ])
    assert benchmark.parse_args().matching_path is None

    monkeypatch.setattr(sys, "argv", [
        "benchmark_norma_alignment.py", "dataset.json",
        "--matching-path", MATCHING_PATH_HISTOGRAM,
    ])
    assert (benchmark.parse_args().matching_path
            == MATCHING_PATH_HISTOGRAM)

    monkeypatch.setattr(sys, "argv", [
        "debug_fisheye_two_frame_align.py",
        "--reference", "ref.tif",
        "--source", "src.tif",
        "--output", "out.tif",
        "--matching-path",
        MATCHING_PATH_ASTERISM,
    ])
    assert (debug.parse_args().matching_path
            == MATCHING_PATH_ASTERISM)


def test_benchmark_exports_first_and_final_rotation_summaries():
    import benchmark_norma_alignment as benchmark

    angle_rad = np.deg2rad(10.0)
    rotation = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [np.sin(angle_rad), np.cos(angle_rad), 0.0],
        [0.0, 0.0, 1.0],
    ])
    summary = benchmark._rotation_summary(rotation)

    assert summary["angle_deg"] == pytest.approx(10.0)
    assert summary["rotation_vector_deg"] == pytest.approx([0.0, 0.0, 10.0])
    assert np.asarray(summary["matrix"]) == pytest.approx(rotation)

    row = benchmark._summary_row({
        "id": "rotation-export",
        "success": True,
        "first_stage_alignment": {
            "rotation_ref_to_src": summary,
            "cameras": {
                "ref": {"focal_length_mm": 14.0},
                "src": {"focal_length_mm": 14.0},
            },
        },
        "final_alignment": {"rotation_ref_to_src": summary},
        "stage_alignment_delta": {
            "rotation_deg": 0.125,
            "ref_focal_mm": 0.25,
            "src_focal_mm": 0.25,
        },
    })

    assert row["first_rotation_z_deg"] == pytest.approx(10.0)
    assert row["final_rotation_angle_deg"] == pytest.approx(10.0)
    assert row["stage_rotation_delta_deg"] == pytest.approx(0.125)
    assert row["first_ref_focal_mm"] == pytest.approx(14.0)
