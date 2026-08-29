"""Sequence-wide bundle adjustment and BA-plan consumers."""
import dataclasses
from collections import Counter
from typing import Any, Optional

import numpy as np

from .._custom_op import median_reduce_chunk, sigma_clip_fused_chunk
from ..component.data_container import FloatImage
from ..component.merger import MaxMerger, MeanMerger
from ..component.norma.bundle import (BAAlignmentPlan, BundleAdjustmentError,
                                      BundleFrame, FrameAlignmentStatus,
                                      build_bundle_plan)
from ..component.norma.bundle_window import (BundleWindowSpec,
                                             build_bundle_window_schedule)
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


@dataclasses.dataclass(frozen=True)
class BundleWindowFrameReport:
    center_index: int
    contributor_indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class BundleWindowReport:
    frames: tuple[BundleWindowFrameReport, ...]


def _append_coverage_channel(
    image: np.ndarray,
) -> tuple[np.ndarray, int | float]:
    """Append an opacity channel so one remap also yields border coverage."""
    dtype = image.dtype
    if np.issubdtype(dtype, np.integer):
        coverage_scale = int(np.iinfo(dtype).max)
    elif np.issubdtype(dtype, np.floating):
        coverage_scale = 1.0
    else:
        raise TypeError(f"unsupported image dtype for window mean: {dtype}")
    coverage = np.full(image.shape[:2], coverage_scale, dtype=dtype)
    if image.ndim == 2:
        return np.stack((image, coverage), axis=-1), coverage_scale
    if image.ndim != 3 or image.shape[2] <= 0:
        raise ValueError("window images must have shape (H, W) or (H, W, C)")
    return np.concatenate((image, coverage[..., None]), axis=-1), coverage_scale


def _mean_bundle_window(
    camera: Any,
    spec: BundleWindowSpec,
    images: dict[int, np.ndarray],
    output_size: tuple[int, int],
    map_scale: float,
) -> np.ndarray:
    """Remap one window and merge only fully supported samples."""
    center = images[spec.center_index]
    merger = MeanMerger()

    for sample, valid in _iter_bundle_window_samples(
            camera, spec, images, output_size, map_scale):
        merger.merge(sample, spatial_mask=valid)

    statistics = merger.result
    if statistics is None:
        raise RuntimeError("bundle window has no valid contributors")
    if np.issubdtype(center.dtype, np.floating):
        denominator = np.where(statistics.n > 0, statistics.n, 1)
        return (statistics.sum_mu / denominator).astype(center.dtype)
    merged = merger.merged_image
    assert merged is not None
    return merged.data.astype(center.dtype)


def _iter_bundle_window_samples(
    camera: Any,
    spec: BundleWindowSpec,
    images: dict[int, np.ndarray],
    output_size: tuple[int, int],
    map_scale: float,
):
    """Yield each center-coordinate sample and its full-coverage mask."""
    for source in spec.sources:
        image = images[source.index]
        if source.index == spec.center_index:
            yield image, np.ones(image.shape[:2], dtype=np.uint8)
            continue
        augmented, coverage_scale = _append_coverage_channel(image)
        remapped = camera.project_image_from_camera(
            camera,
            augmented,
            output_size,
            rotation_dst_to_src=source.rotation_center_to_source,
            map_scale=map_scale,
        )
        sample = remapped[..., :-1]
        if image.ndim == 2:
            sample = sample[..., 0]
        coverage = remapped[..., -1]
        if np.issubdtype(coverage.dtype, np.integer):
            valid = coverage == coverage_scale
        else:
            valid = coverage >= float(coverage_scale) - 1e-6
        yield sample, valid.astype(np.uint8)


def _collect_bundle_window_stack(
    camera: Any,
    spec: BundleWindowSpec,
    images: dict[int, np.ndarray],
    output_size: tuple[int, int],
    map_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one reducer-local stack for non-streaming algorithms."""
    center = images[spec.center_index]
    stack = np.empty((len(spec.sources), *center.shape), dtype=center.dtype)
    masks = np.empty((len(spec.sources), *center.shape[:2]), dtype=np.uint8)
    for position, (sample, valid) in enumerate(_iter_bundle_window_samples(
            camera, spec, images, output_size, map_scale)):
        stack[position] = sample
        masks[position] = valid
    return stack, masks


def _max_bundle_window(camera: Any, spec: BundleWindowSpec,
                       images: dict[int, np.ndarray],
                       output_size: tuple[int, int],
                       map_scale: float) -> np.ndarray:
    merger = MaxMerger()
    for sample, valid in _iter_bundle_window_samples(
            camera, spec, images, output_size, map_scale):
        merger.merge(sample, spatial_mask=valid)
    result = merger.merged_image
    if result is None:
        raise RuntimeError("bundle window has no valid contributors")
    return result


def _masked_median(stack: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Exact per-pixel median with a frame-level spatial validity mask."""
    if np.all(masks):
        return median_reduce_chunk(stack)
    expanded = masks if stack.ndim == 3 else masks[..., None]
    fill = (np.iinfo(stack.dtype).max
            if np.issubdtype(stack.dtype, np.integer) else np.inf)
    np.copyto(stack, fill, where=~expanded.astype(bool))
    stack.sort(axis=0)
    counts = masks.sum(axis=0, dtype=np.intp)
    lower_indices = (counts - 1) // 2
    upper_indices = counts // 2
    if stack.ndim == 4:
        lower_indices = np.broadcast_to(lower_indices[..., None], stack.shape[1:])
        upper_indices = np.broadcast_to(upper_indices[..., None], stack.shape[1:])
    lower = np.take_along_axis(stack, lower_indices[None, ...], axis=0)[0]
    upper = np.take_along_axis(stack, upper_indices[None, ...], axis=0)[0]
    if np.issubdtype(stack.dtype, np.integer):
        result = (lower.astype(np.float64) + upper.astype(np.float64)) / 2.0
        return result.astype(stack.dtype)
    return ((lower + upper) * 0.5).astype(stack.dtype)


def _median_bundle_window(camera: Any, spec: BundleWindowSpec,
                          images: dict[int, np.ndarray],
                          output_size: tuple[int, int],
                          map_scale: float) -> np.ndarray:
    stack, masks = _collect_bundle_window_stack(
        camera, spec, images, output_size, map_scale)
    return _masked_median(stack, masks)


def _sigma_clip_bundle_window(
    camera: Any,
    spec: BundleWindowSpec,
    images: dict[int, np.ndarray],
    output_size: tuple[int, int],
    map_scale: float,
    rej_high: float,
    rej_low: float,
    max_iter: int,
) -> np.ndarray:
    stack, masks = _collect_bundle_window_stack(
        camera, spec, images, output_size, map_scale)
    if stack.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("BundleWindowSigmaClipStackOp supports uint8/uint16")
    channels = stack.shape[-1] if stack.ndim == 4 else 1
    flat_stack = np.ascontiguousarray(stack.reshape(len(stack), -1))
    expanded_masks = (masks if channels == 1 else np.broadcast_to(
        masks[..., None], stack.shape))
    flat_masks = np.ascontiguousarray(expanded_masks.reshape(len(stack), -1),
                                      dtype=np.uint8)
    accepted_sum, _, accepted_n = sigma_clip_fused_chunk(
        flat_stack,
        rej_high=rej_high,
        rej_low=rej_low,
        max_iter=max_iter,
        mask=flat_masks,
        channels=channels,
    )
    mean = np.divide(
        accepted_sum,
        accepted_n,
        out=np.zeros_like(accepted_sum),
        where=accepted_n > 0,
    ).reshape(stack.shape[1:])
    return np.rint(mean).clip(0, np.iinfo(stack.dtype).max).astype(stack.dtype)


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
        "reference_frame_index": {"type": "int", "default": None},
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
        configured_reference = configs.get("reference_frame_index")
        observations = []
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
            tags = exif_obj.exif if exif_obj is not None else None
            detection = await self._run_cpu(
                StarDetectionCache.from_image, array)
            stars = await self._run_cpu(lambda: detection.pywt_stars)
            observations.append((index, stars, array.shape, tags))
            self.tracker.update(self.name)
        if not observations:
            raise BundleAdjustmentError("cannot bundle-adjust an empty sequence")
        reference_index = (
            observations[len(observations) // 2][0]
            if configured_reference is None else int(configured_reference))
        reference_observation = next(
            (item for item in observations if item[0] == reference_index), None)
        if reference_observation is None:
            raise BundleAdjustmentError(
                "reference_frame_index must identify one collected frame")
        _, _, reference_array_shape, reference_tags = reference_observation
        lens_type = (configs.get("lens_type")
                     or lens_type_from_exif(reference_tags))
        policy = CameraInitializationPolicy(
            lens_type=lens_type,
            fallback_focal_equiv_mm=fallback,
            optimize_focal=configs.get("optimize_focal"),
            optimize_distortion=configs.get("optimize_distortion"),
            optimize_principal_point=configs.get("optimize_principal_point"),
        )
        shared_candidate = build_camera_candidate(
            reference_tags, reference_array_shape, "distortion",
            configs.get("distortion"), focal_equiv, policy)
        reference_shape = reference_array_shape[:2]
        if any(shape[:2] != reference_shape
               for _, _, shape, _ in observations):
            raise BundleAdjustmentError(
                "sequence BA requires a single image geometry")
        frames = [BundleFrame(index=index, stars=stars,
                              candidate=shared_candidate)
                  for index, stars, _, _ in observations]
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


class _BundleWindowStackOp(FilterBaseOp):
    """Shared stream scheduling for concrete bundle-window reducers."""

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "alignment_plan": {"type": "object", "required": True},
        "window_size": {"type": "int", "default": 5},
        "min_contributors": {"type": "int", "default": 2},
        "remap_map_scale": {"type": "float", "default": 0.5},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
        "aligned_exifs": {"type": "sequence"},
        "center_indices": {"type": "sequence"},
        "window_report": {"type": "object"},
    }

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs):
        raise NotImplementedError

    async def _async_filter(self, configs: dict[str, Any]) -> None:
        plan = configs["alignment_plan"]
        schedule = build_bundle_window_schedule(
            plan,
            int(configs.get("window_size", 5)),
            min_contributors=int(configs.get("min_contributors", 2)),
        )
        map_scale = float(configs.get("remap_map_scale", 0.5))
        if not 0.0 < map_scale <= 1.0:
            raise ValueError("remap_map_scale must be in (0, 1]")

        intrinsics = plan.shared_camera.intrinsics
        expected_shape = (intrinsics.image_height_px,
                          intrinsics.image_width_px)
        output_size = (intrinsics.image_width_px,
                       intrinsics.image_height_px)
        exifs_active = self.inputs["exifs"].active
        remaining_uses = Counter(
            source.index
            for window in schedule.windows
            for source in window.sources
        )
        frame_cache: dict[int, tuple[Any, np.ndarray, Any]] = {}
        reports: list[BundleWindowFrameReport] = []
        next_window = 0
        seen_count = 0
        data_dtype = None
        float_image_input = None

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
                    raise ValueError(
                        "data/exifs sequences have different lengths") from exc

            seen_count += 1
            if index >= schedule.frame_count:
                raise ValueError(
                    "second image load is longer than AlignmentPlan: "
                    f"got at least {seen_count}, expected {schedule.frame_count}")
            array = frame.data if isinstance(frame, FloatImage) else frame
            if not isinstance(array, np.ndarray):
                raise TypeError("window input frames must be numpy arrays or FloatImage")
            if array.shape[:2] != expected_shape:
                raise ValueError(
                    f"frame {index} geometry differs from AlignmentPlan: "
                    f"got {array.shape[:2]}, expected {expected_shape}")
            if array.ndim not in (2, 3):
                raise ValueError("window images must have shape (H, W) or (H, W, C)")
            if data_dtype is None:
                data_dtype = array.dtype
                float_image_input = isinstance(frame, FloatImage)
            elif array.dtype != data_dtype:
                raise ValueError("window input frames must share one array dtype")
            elif isinstance(frame, FloatImage) != float_image_input:
                raise ValueError(
                    "window input frames must use one container representation")

            if remaining_uses[index] > 0:
                frame_cache[index] = (frame, array, exif_obj)

            while next_window < len(schedule.windows):
                spec = schedule.windows[next_window]
                ready_index = max(source.index for source in spec.sources)
                if ready_index > index:
                    break
                missing = [source.index for source in spec.sources
                           if source.index not in frame_cache]
                if missing:
                    raise RuntimeError(
                        f"window {spec.center_index} lost buffered frames {missing}")
                images = {
                    source.index: frame_cache[source.index][1]
                    for source in spec.sources
                }
                result = await self._run_cpu(
                    self._reduce_window,
                    plan.shared_camera,
                    spec,
                    images,
                    output_size,
                    map_scale,
                    configs,
                )
                center_frame, _, center_exif = frame_cache[spec.center_index]
                output = (FloatImage(data=result, dtype=center_frame.dtype)
                          if isinstance(center_frame, FloatImage) else result)
                contributor_indices = tuple(
                    source.index for source in spec.sources)
                await self._broadcast_outputs({
                    "result": output,
                    "aligned_exifs": center_exif,
                    "center_indices": spec.center_index,
                })
                reports.append(BundleWindowFrameReport(
                    center_index=spec.center_index,
                    contributor_indices=contributor_indices,
                ))
                for source_index in contributor_indices:
                    remaining_uses[source_index] -= 1
                    if remaining_uses[source_index] == 0:
                        frame_cache.pop(source_index, None)
                next_window += 1

            if remaining_uses[index] == 0:
                frame_cache.pop(index, None)
            self.tracker.update(self.name)

        if seen_count != schedule.frame_count:
            raise ValueError(
                "second image load length differs from AlignmentPlan: "
                f"got {seen_count}, expected {schedule.frame_count}")
        if next_window != len(schedule.windows):
            raise RuntimeError("not all bundle windows were emitted")
        await self._broadcast_outputs({
            "window_report": BundleWindowReport(tuple(reports)),
        })


@register_op()
class BundleWindowMeanStackOp(_BundleWindowStackOp):
    """Remap each BA window to its center frame and emit a masked mean."""

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs):
        _ = configs
        return _mean_bundle_window(
            camera, spec, images, output_size, map_scale)

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = n_frames
        window_size = int(configs.get("window_size", 5))
        source_bytes = max(1, int(dtype_bytes or 2))
        work_multiplier = 3 + 16 / source_bytes
        return (int((window_size + work_multiplier) * frame_bytes), 0)


@register_op()
class BundleWindowMaxStackOp(_BundleWindowStackOp):
    """Remap each BA window to its center frame and emit a masked maximum."""

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs):
        _ = configs
        return _max_bundle_window(
            camera, spec, images, output_size, map_scale)

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = (n_frames, dtype_bytes)
        window_size = int(configs.get("window_size", 5))
        return ((window_size + 4) * frame_bytes, 0)


@register_op()
class BundleWindowMedianStackOp(_BundleWindowStackOp):
    """Remap and materialize one window for an exact masked median."""

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs):
        _ = configs
        return _median_bundle_window(
            camera, spec, images, output_size, map_scale)

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = n_frames
        window_size = int(configs.get("window_size", 5))
        source_bytes = max(1, int(dtype_bytes or 2))
        multiplier = 2 * window_size + window_size / source_bytes + 3
        return (int(multiplier * frame_bytes), 0)


@register_op()
class BundleWindowSigmaClipStackOp(_BundleWindowStackOp):
    """Remap and materialize one window for iterative sigma clipping."""

    CONFIGS = {
        **_BundleWindowStackOp.CONFIGS,
        "rej_high": {"type": "float", "default": 3.0},
        "rej_low": {"type": "float", "default": 3.0},
        "max_iter": {"type": "int", "default": 5},
    }

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs):
        rej_high = float(configs.get("rej_high", 3.0))
        rej_low = float(configs.get("rej_low", 3.0))
        max_iter = int(configs.get("max_iter", 5))
        if rej_high <= 0.0 or rej_low <= 0.0:
            raise ValueError("sigma clipping thresholds must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        return _sigma_clip_bundle_window(
            camera, spec, images, output_size, map_scale,
            rej_high, rej_low, max_iter)

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = n_frames
        window_size = int(configs.get("window_size", 5))
        source_bytes = max(1, int(dtype_bytes or 2))
        multiplier = (2 * window_size + window_size / source_bytes
                      + 24 / source_bytes + 3)
        return (int(multiplier * frame_bytes), 0)
