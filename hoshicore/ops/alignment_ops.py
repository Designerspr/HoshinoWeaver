"""
对齐算子：星点对齐等帧间配准操作。

StarAlignmentOp 支持两条显式对齐路径：
  1. 2D 单应性（homography）：zero-distortion CameraModel + warpPerspective（无需 EXIF）
  2. 相机模型优化（camera_model）：Intrinsics + optimize_alignment + remap

相机参数按 EXIF、手动焦距和 fallback 顺序构造。
对齐失败的帧被丢弃，输出为变长序列（sentinel 驱动）。
"""
import dataclasses
from typing import Any, Optional

import numpy as np
from loguru import logger

from ..component.norma.frame_align import (AlignmentError,
                                           CameraInitializationPolicy,
                                           DEFAULT_MATCHING_PATH,
                                           MATCHING_PATH_MEDIAN,
                                           align_frame_camera_model,
                                           align_frame_homography,
                                           build_camera_candidate)
from ..component.norma.bundle import (BAAlignmentPlan, BundleAdjustmentError,
                                      BundleFrame, FrameAlignmentStatus,
                                      build_bundle_plan)
from ..component.norma.geometry_view import GeometryView, StarDetectionCache
from ..component.norma.intrinsics_from_exif import lens_type_from_exif
from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from ..component.data_container import FloatImage
from .base import BaseOp, FilterBaseOp


@dataclasses.dataclass(frozen=True)
class IntegrationFrameReport:
    index: int
    status: FrameAlignmentStatus
    emitted: bool
    reason: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class IntegrationReport:
    frames: tuple[IntegrationFrameReport, ...]


@register_op()
class StarAlignmentOp(FilterBaseOp):
    """星点对齐：将序列帧对齐到参考帧。

    支持两条路径，显式选择：
    - homography: 仅优化旋转+焦距，无需相机信息
    - distortion: 联合优化旋转+焦距+畸变，需要 EXIF 提供内参

    对齐失败的帧被丢弃（不输出），输出为变长序列。
    """

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "reference":        {"type": "image",  "required": True},
        "reference_exif":   {"type": "exif",   "default": None},
        "method":           {"type": "str",    "default": "distortion"},
        "camera_setup_mode": {"type": "str",   "default": None},
        "same_camera":      {"type": "bool",   "default": True},
        "matching_path":    {"type": "str",    "default": DEFAULT_MATCHING_PATH},
        "distortion":       {"type": "list",   "default": None},
        "lens_type":        {"type": "str",    "default": None},
        "ref_lens_type":    {"type": "str",    "default": None},
        "src_lens_type":    {"type": "str",    "default": None},
        "focal_length_mm":  {"type": "float",  "default": None},
        "crop_factor":      {"type": "float",  "default": 1.0},
        "fallback_focal_equiv_mm": {"type": "float", "default": 20.0},
        "bootstrap_scales": {"type": "list", "default": None},
        "remap_map_scale": {"type": "float", "default": 0.5},
        "guided_refine": {"type": "bool", "default": False},
        "optimize_focal": {"type": "bool", "default": None},
        "optimize_distortion": {"type": "bool", "default": None},
        "optimize_principal_point": {"type": "bool", "default": None},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
        "aligned_exifs": {"type": "sequence"},
    }

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = dtype_bytes
        # 持有 1 张参考帧 + 1 张当前帧的对齐输出
        # TODO: 对齐本身资源未计算
        return (2 * frame_bytes, 0)

    async def _async_filter(self, configs: dict[str, Any]) -> None:
        method = configs.get('method', 'distortion')
        camera_setup_mode = configs.get('camera_setup_mode')
        same_camera = configs.get('same_camera', True)
        matching_path = configs.get('matching_path', DEFAULT_MATCHING_PATH)
        init_distortion = configs.get('distortion')
        shared_lens_type = configs.get('lens_type')
        configured_ref_lens = configs.get('ref_lens_type')
        configured_src_lens = configs.get('src_lens_type')
        focal_length_mm = configs.get('focal_length_mm')
        crop_factor = configs.get('crop_factor') or 1.0

        # camera_setup_mode is the user-facing controller.  Keep
        # same_camera and the individual lens fields as a backwards-compatible
        # low-level API for callers which do not provide the mode.
        if camera_setup_mode is not None:
            if camera_setup_mode == "auto":
                same_camera = True
                shared_lens_type = None
                configured_ref_lens = None
                configured_src_lens = None
                focal_length_mm = None
                crop_factor = 1.0
            elif camera_setup_mode == "manual":
                same_camera = True
                configured_ref_lens = None
                configured_src_lens = None
            elif camera_setup_mode == "separate":
                same_camera = False
                shared_lens_type = None
            else:
                raise ValueError(
                    f"Unsupported camera_setup_mode {camera_setup_mode!r}; "
                    "expected 'auto', 'manual', or 'separate'")

        focal_equiv_mm = focal_length_mm * crop_factor if focal_length_mm else None
        fallback_focal_equiv_mm: float = configs.get('fallback_focal_equiv_mm', 20.0)
        bootstrap_scales = configs.get('bootstrap_scales')
        if bootstrap_scales is None:
            bootstrap_scales_tuple = (0.7, 1.0, 1.3)
        else:
            bootstrap_scales_tuple = tuple(float(x) for x in bootstrap_scales)
        remap_map_scale = float(configs.get('remap_map_scale', 0.5))
        guided_refine = bool(configs.get('guided_refine', False))
        policy_kwargs = dict(
            fallback_focal_equiv_mm=float(fallback_focal_equiv_mm),
            optimize_focal=configs.get('optimize_focal'),
            optimize_distortion=configs.get('optimize_distortion'),
            optimize_principal_point=configs.get('optimize_principal_point'),
        )

        exifs_active = self.inputs['exifs'].active

        # Initialize the mandatory reference before consuming the frame
        # stream. This removes the previous first-frame/fallback mode.
        reference = configs['reference']
        ref_arr = reference.data if isinstance(reference, FloatImage) else reference

        ref_exif_obj = configs.get('reference_exif')
        ref_exif_tags = (ref_exif_obj.exif
                         if ref_exif_obj is not None else None)
        ref_lens_type = (configured_ref_lens if configured_ref_lens is not None
                         else shared_lens_type)
        if ref_lens_type is None:
            ref_lens_type = lens_type_from_exif(ref_exif_tags)
        ref_policy = CameraInitializationPolicy(
            lens_type=ref_lens_type, **policy_kwargs)

        ref_candidate = build_camera_candidate(
            ref_exif_tags, ref_arr.shape, method, init_distortion,
            focal_equiv_mm, ref_policy)
        ref_camera = ref_candidate.camera
        ref_detection = await self._run_cpu(
            StarDetectionCache.from_image, ref_arr)
        if method == "homography" or matching_path == MATCHING_PATH_MEDIAN:
            ref_stars = await self._run_cpu(
                lambda: ref_detection.median_stars)
        else:
            ref_stars = await self._run_cpu(
                lambda: ref_detection.pywt_stars)
        ref_geo = GeometryView(ref_stars, ref_camera)
        ref_refine_geo = None
        if (method != "homography" and guided_refine
                and matching_path != MATCHING_PATH_MEDIAN):
            ref_refine_stars = await self._run_cpu(
                lambda: ref_detection.median_stars)
            ref_refine_geo = GeometryView(ref_refine_stars, ref_camera)
        path_name = ("fixed-camera homography fast path"
                     if method == "homography" else "camera model path")
        logger.info(
            f"{self.name}: {path_name} enabled for mandatory reference "
            f"(focal={ref_camera.intrinsics.focal_length_mm:.1f}mm, "
            f"lens_type={ref_lens_type or 'perspective'}, "
            f"source={ref_candidate.init_source})")
        aligned_count = 0
        skipped_count = 0

        for i in self._input_range():
            data = self._async_convert_inputs()
            try:
                frame = await data['data']
            except StreamExhausted:
                break

            # 消费 EXIF 并拆包为 dict
            exif_tags = None
            exif_obj = None
            if exifs_active:
                try:
                    exif_obj = await data['exifs']
                    exif_tags = exif_obj.exif if exif_obj is not None else None
                except StreamExhausted:
                    pass

            frame_arr = frame.data if isinstance(frame, FloatImage) else frame

            # 对齐
            try:
                src_lens_type = (configured_src_lens
                                 if configured_src_lens is not None else
                                 shared_lens_type)
                if src_lens_type is None:
                    src_lens_type = lens_type_from_exif(exif_tags)
                src_policy = CameraInitializationPolicy(
                    lens_type=src_lens_type, **policy_kwargs)
                src_candidate = build_camera_candidate(
                    exif_tags, frame_arr.shape, method, init_distortion,
                    focal_equiv_mm, src_policy)
                src_camera = src_candidate.camera

                if ref_camera and src_camera and method != "homography":
                    aligned_arr = await self._run_cpu(
                        align_frame_camera_model,
                        frame_arr, ref_geo, ref_arr,
                        ref_candidate, src_candidate, same_camera,
                         bootstrap_scales_tuple, remap_map_scale,
                         guided_refine=guided_refine,
                         matching_path=matching_path,
                         ref_refine_geo=ref_refine_geo)
                else:
                    aligned_arr = await self._run_cpu(
                        align_frame_homography,
                        frame_arr,
                        ref_geo,
                        ref_arr,
                        float(fallback_focal_equiv_mm),
                        src_camera,
                    )

                aligned = (FloatImage(data=aligned_arr, dtype=frame.dtype)
                           if isinstance(frame, FloatImage) else aligned_arr)
                await self._broadcast_outputs(
                    {"result": aligned, "aligned_exifs": exif_obj})
                aligned_count += 1
            except AlignmentError as e:
                skipped_count += 1
                logger.warning(
                    f"{self.name}: frame {i} alignment failed ({e}), skipping")
            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                logger.error(traceback_str)
                raise e

            self.tracker.update(self.name)

        logger.info(
            f"{self.name}: aligned {aligned_count} frames, "
            f"skipped {skipped_count}")


@register_op()
class BAAlignmentOp(BaseOp):
    """Collect a same-camera sequence and emit its geometry-only BA plan.

    This op intentionally has no sequence output.  A following integration
    stage must consume a second load of the original files after this object is
    available; this avoids a plan/data join deadlock in the streaming graph.
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
            raise ValueError("BAAlignmentOp supports camera-model distortion mode only")
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
                pair_offsets=tuple(int(value) for value in configs.get("pair_offsets") or (1, 2, 4)),
                random_seed=configs.get("random_seed", 0))
        except BundleAdjustmentError:
            raise
        except Exception as exc:
            raise BundleAdjustmentError(f"sequence BA failed: {exc}") from exc
        await self._broadcast_outputs({"alignment_plan": plan})


@register_op()
class BAIntegrationOp(FilterBaseOp):
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
        "integration_report": {"type": "object"},
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
        reports: list[IntegrationFrameReport] = []
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
                reports.append(IntegrationFrameReport(
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
            reports.append(IntegrationFrameReport(
                index, entry.status, True, entry.reason))
            self.tracker.update(self.name)
        if seen_count != len(plan.frames):
            raise ValueError(
                "second image load length differs from AlignmentPlan: "
                f"got {seen_count}, expected {len(plan.frames)}")
        await self._broadcast_outputs({
            "integration_report": IntegrationReport(tuple(reports))})
