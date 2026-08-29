"""Sequence-wide bundle adjustment and BA-plan consumers."""
import dataclasses
from typing import Any, Optional

import numpy as np

from ..component.data_container import FloatImage
from ..component.norma.bundle import (BAAlignmentPlan, BundleAdjustmentError,
                                      BundleFrame, FrameAlignmentStatus,
                                      build_bundle_plan)
from ..component.norma.frame_align import (CameraInitializationPolicy,
                                           build_camera_candidate)
from ..component.norma.geometry_view import StarDetectionCache
from ..component.norma.intrinsics_from_exif import lens_type_from_exif
from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from .base import BaseOp, FilterBaseOp


@dataclasses.dataclass(frozen=True)
class BundleRemapFrameReport:
    index: int
    status: FrameAlignmentStatus
    emitted: bool
    reason: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class BundleRemapReport:
    frames: tuple[BundleRemapFrameReport, ...]


@register_op()
class BundleAdjustmentOp(BaseOp):
    """Collect a same-camera sequence and emit its geometry-only BA plan.

    This op intentionally has no sequence output. A downstream plan consumer
    must use a second load of the original files after this object is available;
    this avoids a plan/data join deadlock in the streaming graph.
    """

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "reference_frame_index": {"type": "int", "required": True},
        "method": {"type": "str", "default": "distortion"},
        "lens_type": {"type": "str", "default": None},
        "distortion": {"type": "list", "default": None},
        "focal_length_mm": {"type": "float", "default": None},
        "crop_factor": {"type": "float", "default": 1.0},
        "fallback_focal_equiv_mm": {"type": "float", "default": 20.0},
        "optimize_focal": {"type": "bool", "default": None},
        "optimize_distortion": {"type": "bool", "default": None},
        "optimize_principal_point": {"type": "bool", "default": None},
        "pair_offsets": {"type": "list", "default": [1, 2, 4]},
        "random_seed": {"type": "int", "default": 0},
    }
    OUTPUTS: dict[str, Any] = {
        "alignment_plan": {"type": "object"},
    }

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = (configs, dtype_bytes, n_frames)
        # Only compact star observations survive collection; pixel memory is
        # bounded by the frame currently being detected.
        return (frame_bytes * 2, 0)

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        if configs.get("method", "distortion") != "distortion":
            raise ValueError(
                "BundleAdjustmentOp supports camera-model distortion mode only")
        exifs_active = self.inputs["exifs"].active
        focal_length = configs.get("focal_length_mm")
        focal_equiv = (float(focal_length) * float(configs.get("crop_factor") or 1.0)
                       if focal_length is not None else None)
        fallback = float(configs.get("fallback_focal_equiv_mm", 20.0))
        reference_index = int(configs["reference_frame_index"])
        observations = []
        shared_candidate = None
        reference_shape = None
        for index in self._input_range():
            data = self._async_convert_inputs()
            try:
                image = await data["data"]
            except StreamExhausted:
                break
            exif_obj = None
            if exifs_active:
                try:
                    exif_obj = await data["exifs"]
                except StreamExhausted as exc:
                    raise ValueError("data/exifs sequences have different lengths") from exc
            array = image.data if isinstance(image, FloatImage) else image
            if index == reference_index:
                tags = exif_obj.exif if exif_obj is not None else None
                lens_type = configs.get("lens_type") or lens_type_from_exif(tags)
                policy = CameraInitializationPolicy(
                    lens_type=lens_type,
                    fallback_focal_equiv_mm=fallback,
                    optimize_focal=configs.get("optimize_focal"),
                    optimize_distortion=configs.get("optimize_distortion"),
                    optimize_principal_point=configs.get("optimize_principal_point"),
                )
                shared_candidate = build_camera_candidate(
                    tags, array.shape, "distortion", configs.get("distortion"),
                    focal_equiv, policy)
                reference_shape = array.shape[:2]
            detection = await self._run_cpu(
                StarDetectionCache.from_image, array)
            stars = await self._run_cpu(lambda: detection.pywt_stars)
            observations.append((index, stars, array.shape[:2]))
            self.tracker.update(self.name)
        if shared_candidate is None:
            raise BundleAdjustmentError(
                "reference_frame_index must identify one collected frame")
        if any(shape != reference_shape for _, _, shape in observations):
            raise BundleAdjustmentError(
                "sequence BA requires a single image geometry")
        frames = [BundleFrame(index=index, stars=stars,
                              candidate=shared_candidate)
                  for index, stars, _ in observations]
        try:
            plan = await self._run_cpu(
                build_bundle_plan, frames,
                reference_frame_index=reference_index,
                pair_offsets=tuple(
                    int(value) for value in
                    configs.get("pair_offsets") or (1, 2, 4)),
                random_seed=configs.get("random_seed", 0))
        except BundleAdjustmentError:
            raise
        except Exception as exc:
            raise BundleAdjustmentError(f"sequence BA failed: {exc}") from exc
        await self._broadcast_outputs({"alignment_plan": plan})


@register_op()
class BundleReferenceRemapOp(FilterBaseOp):
    """Apply a geometry-only BA plan to a second image stream in input order."""

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "alignment_plan": {"type": "object", "required": True},
        "remap_map_scale": {"type": "float", "default": 0.5},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
        "aligned_exifs": {"type": "sequence"},
        "remap_report": {"type": "object"},
    }

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = (configs, n_frames, dtype_bytes)
        # One input and one remapped output. Projection-specific workspace is
        # temporary and does not grow with the number or position of frames.
        return (2 * frame_bytes, 0)

    async def _async_filter(self, configs: dict[str, Any]) -> None:
        plan = configs["alignment_plan"]
        if not isinstance(plan, BAAlignmentPlan):
            raise TypeError("alignment_plan must be a BAAlignmentPlan")
        map_scale = float(configs.get("remap_map_scale", 0.5))
        if not 0.0 < map_scale <= 1.0:
            raise ValueError("remap_map_scale must be in (0, 1]")
        if not 0 <= plan.reference_frame_index < len(plan.frames):
            raise ValueError("AlignmentPlan reference index is outside its frames")
        exifs_active = self.inputs["exifs"].active
        reports: list[BundleRemapFrameReport] = []
        seen_count = 0
        intrinsics = plan.shared_camera.intrinsics
        expected_shape = (intrinsics.image_height_px,
                          intrinsics.image_width_px)
        output_size = (intrinsics.image_width_px,
                       intrinsics.image_height_px)

        for index in self._input_range():
            data = self._async_convert_inputs()
            try:
                frame = await data["data"]
            except StreamExhausted:
                break
            exif_obj = None
            if exifs_active:
                try:
                    exif_obj = await data["exifs"]
                except StreamExhausted as exc:
                    raise ValueError("data/exifs sequences have different lengths") from exc
            seen_count += 1
            if index >= len(plan.frames):
                raise ValueError(
                    "second image load is longer than AlignmentPlan: "
                    f"got at least {seen_count}, expected {len(plan.frames)}")
            entry = plan.frame(index)
            if entry.index != index:
                raise ValueError(
                    "AlignmentPlan frames must use contiguous stream indices")
            array = frame.data if isinstance(frame, FloatImage) else frame
            if array.shape[:2] != expected_shape:
                raise ValueError(
                    f"frame {index} geometry differs from AlignmentPlan: "
                    f"got {array.shape[:2]}, expected {expected_shape}")

            if entry.status == FrameAlignmentStatus.EXCLUDED:
                reports.append(BundleRemapFrameReport(
                    index, entry.status, False, entry.reason))
                self.tracker.update(self.name)
                continue
            if entry.rotation_ref_to_src is None:
                raise ValueError(
                    f"solved frame {index} has no reference-relative pose")

            if index == plan.reference_frame_index:
                output = frame
            else:
                aligned = await self._run_cpu(
                    plan.shared_camera.project_image_from_camera,
                    plan.shared_camera,
                    array,
                    output_size,
                    rotation_dst_to_src=entry.rotation_ref_to_src,
                    map_scale=map_scale,
                )
                output = (FloatImage(data=aligned, dtype=frame.dtype)
                          if isinstance(frame, FloatImage) else aligned)
            await self._broadcast_outputs(
                {"result": output, "aligned_exifs": exif_obj})
            reports.append(BundleRemapFrameReport(
                index, entry.status, True, entry.reason))
            self.tracker.update(self.name)
        if seen_count != len(plan.frames):
            raise ValueError(
                "second image load length differs from AlignmentPlan: "
                f"got {seen_count}, expected {len(plan.frames)}")
        await self._broadcast_outputs({
            "remap_report": BundleRemapReport(tuple(reports))})
