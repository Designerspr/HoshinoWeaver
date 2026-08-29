import asyncio

import numpy as np
import pytest

import hoshicore.ops.image_saver as image_saver
from hoshicore.ops.image_saver import BatchImageSaveOp, _format_output_path


async def _ready(value):
    return value


def test_format_output_path_supports_sequence_and_frame_indices():
    assert _format_output_path(
        "frame_{index:04d}_src_{frame_index:04d}.png",
        3,
        20,
        frame_index=11,
    ) == "frame_0003_src_0011.png"


def test_format_output_path_requires_wired_frame_index():
    with pytest.raises(ValueError, match="frame_indices is not wired"):
        _format_output_path("frame_{frame_index:04d}.png", 3, 20)


def test_batch_saver_passes_frame_index_and_exif(monkeypatch, tmp_path):
    calls = []

    def fake_save(path, frame, **kwargs):
        calls.append((path, frame, kwargs))

    monkeypatch.setattr(image_saver, "save_img", fake_save)
    op = BatchImageSaveOp("save")
    op.inputs["frame_indices"].active = True
    op.inputs["exifs"].active = True
    op._frame_counter = 0
    frame = np.zeros((2, 3), dtype=np.uint8)
    exif = object()

    result = asyncio.run(op._async_execute_single(
        {
            "data": _ready(frame),
            "frame_indices": _ready(7),
            "exifs": _ready(exif),
        },
        {
            "output_dir": str(tmp_path),
            "output_template": "frame_{index:03d}_{frame_index:03d}.png",
            "output_dtype": None,
            "png_compressing": 4,
            "jpg_quality": 91,
        },
    ))

    assert result["result"].endswith("frame_000_007.png")
    assert calls[0][1] is frame
    assert calls[0][2] == {
        "png_compressing": 4,
        "jpg_quality": 91,
        "exif": exif,
    }


def test_batch_saver_propagates_save_failure(monkeypatch, tmp_path):
    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(image_saver, "save_img", fail_save)
    op = BatchImageSaveOp("save")
    op.inputs["frame_indices"].active = False
    op.inputs["exifs"].active = False
    op._frame_counter = 0

    with pytest.raises(RuntimeError, match="disk full"):
        asyncio.run(op._async_execute_single(
            {"data": _ready(np.zeros((2, 3), dtype=np.uint8))},
            {
                "output_dir": str(tmp_path),
                "output_template": "frame_{index:03d}.png",
                "output_dtype": None,
            },
        ))
