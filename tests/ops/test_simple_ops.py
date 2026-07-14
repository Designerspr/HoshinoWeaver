import asyncio

import numpy as np
import pytest

from hoshicore.component.queue import RichContextQueue
from hoshicore.ops.simple_ops import CalibrationSubtractImageOp


pytestmark = pytest.mark.asyncio


async def _run_single_image_subtract(frame, reference):
    op = CalibrationSubtractImageOp("subtract_image")
    sink = RichContextQueue(maxsize=1)
    op.outputs["result"].append(sink)
    await op.config["data"].put(frame)
    await op.config["reference"].put(reference)
    task = asyncio.create_task(op.execute())
    result = await asyncio.wait_for(sink.get(), timeout=0.1)
    await asyncio.wait_for(task, timeout=0.1)
    return result


async def test_calibration_subtract_image_passthrough_when_reference_is_none():
    frame = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    result = await _run_single_image_subtract(frame, None)

    assert result is frame


async def test_calibration_subtract_image_subtracts_reference():
    frame = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    reference = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    result = await _run_single_image_subtract(frame, reference)

    np.testing.assert_array_equal(result, frame - reference)
