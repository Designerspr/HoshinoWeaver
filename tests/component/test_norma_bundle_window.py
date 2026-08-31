from types import SimpleNamespace

import numpy as np
import pytest

from hoshicore.component.norma.bundle import (
    BAAlignmentPlan,
    FrameAlignment,
    FrameAlignmentStatus,
)
from hoshicore.component.norma.bundle_window import (
    build_bundle_window_schedule,
)


def _rotation_z(angle: float) -> np.ndarray:
    return np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])


def _plan(statuses) -> BAAlignmentPlan:
    frames = []
    for index, status in enumerate(statuses):
        rotation = (_rotation_z(index * 0.1)
                    if status == FrameAlignmentStatus.SOLVED else None)
        frames.append(FrameAlignment(
            index=index,
            status=status,
            rotation_ref_to_src=rotation,
            pose_source="bundle" if rotation is not None else "none",
        ))
    return BAAlignmentPlan(
        reference_frame_index=0,
        shared_camera=SimpleNamespace(),
        frames=tuple(frames),
        accepted_edge_count=0,
        rejected_edge_count=0,
        active_camera_parameter_count=0,
        observability_condition=None,
    )


def test_bundle_window_schedule_uses_shrinking_edges_and_skips_excluded():
    solved = FrameAlignmentStatus.SOLVED
    excluded = FrameAlignmentStatus.EXCLUDED
    plan = _plan((solved, solved, excluded, solved, solved))

    schedule = build_bundle_window_schedule(
        plan, 3, min_contributors=2)

    assert [window.center_index for window in schedule.windows] == [0, 1, 3, 4]
    assert [[source.index for source in window.sources]
            for window in schedule.windows] == [[0, 1], [0, 1], [3, 4], [3, 4]]
    expected = (_rotation_z(0.1) @ _rotation_z(0.0).T)
    np.testing.assert_allclose(
        schedule.windows[0].sources[1].rotation_center_to_source,
        expected,
    )
    assert not schedule.windows[0].sources[1].rotation_center_to_source.flags.writeable


@pytest.mark.parametrize("window_size", [0, 2, -1])
def test_bundle_window_schedule_requires_positive_odd_window(window_size):
    with pytest.raises(ValueError, match="positive odd"):
        build_bundle_window_schedule(
            _plan((FrameAlignmentStatus.SOLVED,)),
            window_size,
            min_contributors=1,
        )


def test_bundle_window_schedule_rejects_solved_frame_without_pose():
    plan = _plan((FrameAlignmentStatus.SOLVED,))
    plan = BAAlignmentPlan(
        reference_frame_index=0,
        shared_camera=plan.shared_camera,
        frames=(FrameAlignment(
            index=0,
            status=FrameAlignmentStatus.SOLVED,
            rotation_ref_to_src=None,
            pose_source="bundle",
        ),),
        accepted_edge_count=0,
        rejected_edge_count=0,
        active_camera_parameter_count=0,
        observability_condition=None,
    )
    with pytest.raises(ValueError, match="no reference-relative pose"):
        build_bundle_window_schedule(plan, 1, min_contributors=1)
