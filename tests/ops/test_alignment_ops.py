import asyncio
from types import SimpleNamespace

import numpy as np

import hoshicore.ops.bundle_ops as bundle_ops
from hoshicore.component.norma.bundle import (BAAlignmentPlan, FrameAlignment,
                                               FrameAlignmentStatus)
from hoshicore.ops.bundle_ops import (BundleAdjustmentOp,
                                      BundleReferenceRemapOp,
                                      BundleRemapReport)


class _Camera:
    def __init__(self, calls):
        self.intrinsics = SimpleNamespace(
            image_height_px=8, image_width_px=12)
        self.calls = calls

    def project_image_from_camera(
        self,
        camera,
        image,
        output_size,
        *,
        rotation_dst_to_src,
        map_scale,
    ):
        self.calls.append((image, output_size, rotation_dst_to_src, map_scale))
        assert camera is self
        return image + 10


def _plan(camera, frames, reference=0):
    return BAAlignmentPlan(
        reference_frame_index=reference,
        shared_camera=camera,
        frames=tuple(frames),
        accepted_edge_count=1,
        rejected_edge_count=0,
        active_camera_parameter_count=0,
        observability_condition=None,
    )


def _frame(index, status=FrameAlignmentStatus.SOLVED, rotation=None,
           reason=None):
    if rotation is None and status == FrameAlignmentStatus.SOLVED:
        rotation = np.eye(3)
    return FrameAlignment(
        index=index,
        status=status,
        rotation_ref_to_src=rotation,
        pose_source="bundle" if status == FrameAlignmentStatus.SOLVED else "none",
        reason=reason,
    )


def test_bundle_adjustment_uses_reference_camera_for_every_frame(monkeypatch):
    op = BundleAdjustmentOp("bundle")
    op.length = 3
    op.inputs["exifs"] = SimpleNamespace(active=True)
    images = [np.zeros((8, 12), dtype=np.uint16) for _ in range(3)]
    exifs = [SimpleNamespace(exif={"frame": index}) for index in range(3)]
    position = 0

    async def ready(value):
        return value

    def inputs():
        nonlocal position
        current = position
        position += 1
        return {"data": ready(images[current]), "exifs": ready(exifs[current])}

    async def run_cpu(function, *args, **kwargs):
        return function(*args, **kwargs)

    shared_candidate = object()
    candidate_calls = []

    def build_candidate(tags, shape, method, distortion, focal, policy):
        candidate_calls.append((tags, shape, method, distortion, focal, policy))
        return shared_candidate

    captured = {}

    def build_plan(frames, **kwargs):
        captured["frames"] = frames
        captured["kwargs"] = kwargs
        return "plan"

    async def broadcast(result):
        captured["output"] = result

    monkeypatch.setattr(op, "_async_convert_inputs", inputs)
    monkeypatch.setattr(op, "_run_cpu", run_cpu)
    monkeypatch.setattr(op, "_broadcast_outputs", broadcast)
    monkeypatch.setattr(
        bundle_ops.StarDetectionCache, "from_image",
        staticmethod(lambda image: SimpleNamespace(pywt_stars=object())))
    monkeypatch.setattr(bundle_ops, "build_camera_candidate", build_candidate)
    monkeypatch.setattr(bundle_ops, "build_bundle_plan", build_plan)

    asyncio.run(op._async_execute({
        "reference_frame_index": 1,
        "method": "distortion",
        "focal_length_mm": 14.0,
        "crop_factor": 1.0,
        "fallback_focal_equiv_mm": 20.0,
        "pair_offsets": [1, 2],
        "random_seed": 7,
    }))

    assert len(candidate_calls) == 1
    assert candidate_calls[0][0] == {"frame": 1}
    assert all(frame.candidate is shared_candidate for frame in captured["frames"])
    assert captured["kwargs"] == {
        "reference_frame_index": 1,
        "pair_offsets": (1, 2),
        "random_seed": 7,
    }
    assert captured["output"] == {"alignment_plan": "plan"}


def test_bundle_reference_remap_streams_in_order_and_skips_excluded(monkeypatch):
    op = BundleReferenceRemapOp("remap")
    op.length = 4
    op.inputs["exifs"] = SimpleNamespace(active=True)
    images = [np.full((8, 12), index, dtype=np.uint16)
              for index in range(4)]
    exifs = [object() for _ in images]
    position = 0

    async def ready(value):
        return value

    def inputs():
        nonlocal position
        current = position
        position += 1
        return {
            "data": ready(images[current]),
            "exifs": ready(exifs[current]),
        }

    async def run_cpu(function, *args, **kwargs):
        return function(*args, **kwargs)

    broadcasts = []

    async def broadcast(result):
        broadcasts.append(result)

    remap_calls = []
    camera = _Camera(remap_calls)
    plan = _plan(camera, (
        _frame(0),
        _frame(1, FrameAlignmentStatus.EXCLUDED, reason="disconnected"),
        _frame(2),
        _frame(3),
    ), reference=2)
    monkeypatch.setattr(op, "_async_convert_inputs", inputs)
    monkeypatch.setattr(op, "_run_cpu", run_cpu)
    monkeypatch.setattr(op, "_broadcast_outputs", broadcast)

    asyncio.run(op._async_filter({
        "alignment_plan": plan,
        "remap_map_scale": 0.25,
    }))

    frame_outputs = broadcasts[:-1]
    assert [int(result["result"][0, 0]) for result in frame_outputs] == [10, 2, 13]
    assert [result["aligned_exifs"] for result in frame_outputs] == [
        exifs[0], exifs[2], exifs[3]]
    assert len(remap_calls) == 2
    assert [int(call[0][0, 0]) for call in remap_calls] == [0, 3]
    assert all(call[1] == (12, 8) and call[3] == 0.25
               for call in remap_calls)

    report = broadcasts[-1]["remap_report"]
    assert isinstance(report, BundleRemapReport)
    assert [item.index for item in report.frames] == [0, 1, 2, 3]
    assert [item.emitted for item in report.frames] == [True, False, True, True]
    assert report.frames[1].reason == "disconnected"


def test_bundle_reference_remap_rejects_solved_frame_without_pose(monkeypatch):
    op = BundleReferenceRemapOp("remap")
    op.length = 1
    op.inputs["exifs"] = SimpleNamespace(active=False)
    image = np.zeros((8, 12), dtype=np.uint16)

    async def ready(value):
        return value

    monkeypatch.setattr(
        op, "_async_convert_inputs", lambda: {"data": ready(image)})
    camera = _Camera([])
    missing_pose = FrameAlignment(
        index=0,
        status=FrameAlignmentStatus.SOLVED,
        rotation_ref_to_src=None,
        pose_source="bundle",
    )

    with np.testing.assert_raises_regex(
            ValueError, "solved frame 0 has no reference-relative pose"):
        asyncio.run(op._async_filter({
            "alignment_plan": _plan(camera, (missing_pose,)),
        }))


def test_bundle_reference_remap_validates_second_load_geometry(monkeypatch):
    op = BundleReferenceRemapOp("remap")
    op.length = 1
    op.inputs["exifs"] = SimpleNamespace(active=False)

    async def ready(value):
        return value

    monkeypatch.setattr(
        op, "_async_convert_inputs",
        lambda: {"data": ready(np.zeros((7, 12), dtype=np.uint16))})
    camera = _Camera([])

    with np.testing.assert_raises_regex(
            ValueError, "geometry differs from AlignmentPlan"):
        asyncio.run(op._async_filter({
            "alignment_plan": _plan(camera, (_frame(0),)),
        }))


def test_bundle_reference_remap_resource_estimate_is_sequence_constant():
    assert BundleReferenceRemapOp.estimate_resources({}, 1024, 3) == (2048, 0)
    assert BundleReferenceRemapOp.estimate_resources({}, 1024, 300) == (2048, 0)
