from hoshicore.component.norma.bundle import (BAAlignmentPlan, FrameAlignment,
                                               FrameAlignmentStatus)
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics
from hoshicore.ops.bundle_ops import (BundleWindowFrameReport,
                                      BundleWindowReport)
from hoshicore.ops.report_ops import format_bundle_window_report


def _frame(index, status, reason=None):
    return FrameAlignment(
        index=index,
        status=status,
        rotation_ref_to_src=None,
        pose_source="test",
        reason=reason,
    )


def test_format_bundle_window_report_derives_all_skip_reasons():
    plan = BAAlignmentPlan(
        reference_frame_index=0,
        shared_camera=CameraModel(
            Intrinsics(20.0, 36.0, 24.0, 120, 80), Distortion()),
        frames=(
            _frame(0, FrameAlignmentStatus.SOLVED),
            _frame(1, FrameAlignmentStatus.EXCLUDED, "weak outer field"),
            _frame(2, FrameAlignmentStatus.SOLVED),
            _frame(3, FrameAlignmentStatus.SOLVED),
        ),
        accepted_edge_count=2,
        rejected_edge_count=1,
        active_camera_parameter_count=0,
        observability_condition=None,
    )
    report = BundleWindowReport((
        BundleWindowFrameReport(0, (0, 2)),
        BundleWindowFrameReport(2, (0, 2, 3)),
    ))

    summary = format_bundle_window_report(plan, report)

    assert "4 input, 3 solved, 1 excluded" in summary
    assert "Shared camera:" in summary
    assert "Bundle edges: 2 accepted, 1 rejected" in summary
    assert "2 emitted, 1 insufficient contributors" in summary
    assert "min=2, median=2.5, max=3" in summary
    assert "1: weak outer field" in summary
    assert "3: insufficient contributors" in summary
