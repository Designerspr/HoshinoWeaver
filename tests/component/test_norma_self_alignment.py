from types import SimpleNamespace

import numpy as np

from hoshicore.component.norma.frame_align import _has_identical_star_geometry


def _geometry(*, positions=None, volumes=None, features=None):
    return SimpleNamespace(
        positions=np.array([[10.0, 20.0], [30.0, 40.0]])
        if positions is None else positions,
        volumes=np.array([2.0, 3.0]) if volumes is None else volumes,
        features=np.array([[0.1, 0.2], [0.3, 0.4]])
        if features is None else features,
    )


def test_identical_star_geometry_accepts_exact_feature_identity():
    assert _has_identical_star_geometry(_geometry(), _geometry())


def test_identical_descriptors_do_not_hide_different_positions():
    ref = _geometry()
    src = _geometry(positions=np.array([[11.0, 20.0], [30.0, 40.0]]))

    assert not _has_identical_star_geometry(ref, src)


def test_identical_positions_do_not_hide_different_photometry():
    ref = _geometry()
    src = _geometry(volumes=np.array([2.0, 4.0]))

    assert not _has_identical_star_geometry(ref, src)


def test_identical_points_and_photometry_still_require_same_descriptors():
    ref = _geometry()
    src = _geometry(features=np.array([[0.1, 0.2], [0.3, 0.5]]))

    assert not _has_identical_star_geometry(ref, src)
