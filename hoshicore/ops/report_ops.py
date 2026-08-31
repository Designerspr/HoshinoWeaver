"""Terminal report formatters for command-line DAGs."""
from __future__ import annotations

from statistics import median
from typing import Any

from loguru import logger

from ..component.norma.bundle import BAAlignmentPlan, FrameAlignmentStatus
from ..engine.registry import register_op
from .base import BaseOp
from .bundle_ops import BundleWindowReport


def format_bundle_window_report(
    plan: BAAlignmentPlan,
    report: BundleWindowReport,
) -> str:
    """Return a concise, self-contained BA window summary."""
    if not isinstance(plan, BAAlignmentPlan):
        raise TypeError("alignment_plan must be a BAAlignmentPlan")
    if not isinstance(report, BundleWindowReport):
        raise TypeError("window_report must be a BundleWindowReport")

    emitted = {
        frame.center_index: frame for frame in report.frames
        if frame.status == "ready"
    }
    excluded = [
        frame for frame in plan.frames
        if frame.status == FrameAlignmentStatus.EXCLUDED
    ]
    solved = [
        frame for frame in plan.frames
        if frame.status == FrameAlignmentStatus.SOLVED
    ]
    explicit_insufficient = [
        frame for frame in report.frames
        if frame.status == "insufficient_contributors"
    ]
    reported_indices = {frame.center_index for frame in report.frames}
    legacy_insufficient = [
        frame for frame in solved
        if frame.index not in reported_indices
    ]
    contributor_counts = [
        len(frame.contributor_indices) for frame in emitted.values()
    ]

    lines = [
        f"Bundle alignment: {len(plan.frames)} input, {len(solved)} solved, "
        f"{len(excluded)} excluded",
        "Shared camera: "
        f"{type(plan.shared_camera).__name__}, "
        f"focal={plan.shared_camera.intrinsics.focal_length_mm:.6g}mm, "
        f"distortion={plan.shared_camera.distortion.to_cv2().tolist()}",
        f"Bundle edges: {plan.accepted_edge_count} accepted, "
        f"{plan.rejected_edge_count} rejected, "
        f"condition={plan.observability_condition}",
        f"Camera solve: {plan.camera_solve_mode}",
        f"Window output: {len(emitted)} emitted, "
        f"{len(explicit_insufficient) + len(legacy_insufficient)} "
        "insufficient contributors",
    ]
    if plan.camera_fallback_reason:
        lines.append(
            f"Camera fallback reason: {plan.camera_fallback_reason}")
    if contributor_counts:
        lines.append(
            "Contributors: "
            f"min={min(contributor_counts)}, "
            f"median={median(contributor_counts):g}, "
            f"max={max(contributor_counts)}")
    if excluded:
        lines.append("Excluded frames:")
        lines.extend(
            f"  {frame.index}: {frame.reason or 'excluded by bundle adjustment'}"
            for frame in excluded
        )
    if explicit_insufficient or legacy_insufficient:
        lines.append("Skipped window centers:")
        lines.extend(
            f"  {frame.center_index}: {frame.reason or 'insufficient contributors'}"
            for frame in explicit_insufficient
        )
        lines.extend(
            f"  {frame.index}: insufficient contributors"
            for frame in legacy_insufficient
        )
    return "\n".join(lines)


@register_op()
class BundleWindowReportDisplayOp(BaseOp):
    """Log a window summary and expose the same text for future frontends."""

    CONFIGS: dict[str, Any] = {
        "alignment_plan": {"type": "object", "required": True},
        "window_report": {"type": "object", "required": True},
    }
    OUTPUTS: dict[str, Any] = {
        "summary": {"type": "str"},
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        summary = format_bundle_window_report(
            configs["alignment_plan"], configs["window_report"])
        logger.info("\n{}", summary)
        await self._broadcast_outputs({"summary": summary})
