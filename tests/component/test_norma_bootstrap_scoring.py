from types import SimpleNamespace

import numpy as np

from hoshicore.component.norma.frame_align import (
    _bootstrap_candidate_pairs,
    _bootstrap_candidate_score,
)
from hoshicore.component.norma.matching import RotationDiagnostics


def _diagnostics(
    matches: int,
    median_deg: float,
    p90_deg: float,
    coverage: float,
    outer: int,
) -> RotationDiagnostics:
    return RotationDiagnostics(
        inlier_count=matches,
        median_angle_error_rad=np.deg2rad(median_deg),
        p90_angle_error_rad=np.deg2rad(p90_deg),
        max_angle_error_rad=np.deg2rad(0.75),
        coverage_ratio=coverage,
        radial_bin_count=3,
        active_sector_count=8,
        outer_inlier_count=outer,
    )


def test_bootstrap_score_prefers_rotation_quality_after_coverage_is_sufficient():
    dense_wrong_focal = _diagnostics(1578, 0.1071, 0.2432, 0.6833, 200)
    clean_correct_focal = _diagnostics(1114, 0.0307, 0.0901, 0.6608, 121)

    wrong_score, _ = _bootstrap_candidate_score(
        dense_wrong_focal, 1578, 196, 8, 0.7, 0.7)
    correct_score, _ = _bootstrap_candidate_score(
        clean_correct_focal, 1114, 125, 8, 1.0, 1.0)

    assert correct_score > wrong_score


def test_same_camera_bootstrap_uses_only_equal_scale_pairs():
    ref = [SimpleNamespace(scale=scale) for scale in (0.7, 1.0, 1.3)]
    src = [SimpleNamespace(scale=scale) for scale in (0.7, 1.0, 1.3)]

    diagonal = _bootstrap_candidate_pairs(ref, src, same_camera=True)
    cartesian = _bootstrap_candidate_pairs(ref, src, same_camera=False)

    assert [(a.scale, b.scale) for a, b in diagonal] == [
        (0.7, 0.7),
        (1.0, 1.0),
        (1.3, 1.3),
    ]
    assert len(cartesian) == 9
