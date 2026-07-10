import asyncio
from unittest import mock

import cv2
import numpy as np

from hoshicore._custom_op import (
    max_combine,
    threshold_max_merge as custom_threshold_max_merge,
)
import hoshicore._custom_op.ops.max as max_ops
from hoshicore.component.data_container import FastGaussianParam
import hoshicore.component.noise_equalization as noise_equalization
from hoshicore.component.merger import MaxMerger
from hoshicore.ops.sigma_clip_ops import ThresholdMaxIteratorOp


from tests.custom_ops._base import CustomOpsTestCase


class TestMaxCustomOps(CustomOpsTestCase):
    def test_max_combine_matches_numpy(self) -> None:
        base = np.array([[1, 5], [3, 4]], dtype=np.uint16)
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        got = max_combine(base, fresh)
        expected = np.maximum(np.array([[1, 5], [3, 4]], dtype=np.uint16), fresh)

        self.assertIs(got, base)
        np.testing.assert_array_equal(base, expected)
        np.testing.assert_array_equal(got, expected)

    def test_max_merger_works_with_custom_op_fallback(self) -> None:
        merger = MaxMerger(int_weight=False)
        merger.merge(np.array([[1, 9], [2, 3]], dtype=np.uint16), None)
        merger.merge(np.array([[5, 1], [2, 8]], dtype=np.uint16), None)

        np.testing.assert_array_equal(
            merger.merged_image,
            np.array([[5, 9], [2, 8]], dtype=np.uint16),
        )

    def test_threshold_max_merge_matches_numpy(self) -> None:
        frame = np.array([[6.0, 12.0], [4.0, 15.0]], dtype=np.float64)
        mean_img = np.array([[5.0, 8.0], [4.5, 10.0]], dtype=np.float64)
        std_img = np.array([[0.5, 1.0], [0.5, 2.0]], dtype=np.float64)
        base = np.array([[5.0, 9.0], [4.5, 11.0]], dtype=np.float64)
        expected = np.array(base, copy=True)

        got = custom_threshold_max_merge(frame, mean_img, std_img, base, 2.0, 0.5)
        max_ops.threshold_max_merge_numpy(frame, mean_img, std_img, expected, 2.0, 0.5)

        self.assertIs(got, base)
        np.testing.assert_allclose(base, expected, rtol=1e-7, atol=1e-7)

    def test_threshold_max_merge_can_force_numpy_fallback(self) -> None:
        frame = np.array([[6.0, 12.0], [4.0, 15.0]], dtype=np.float64)
        mean_img = np.array([[5.0, 8.0], [4.5, 10.0]], dtype=np.float64)
        std_img = np.array([[0.5, 1.0], [0.5, 2.0]], dtype=np.float64)
        base = np.array([[5.0, 9.0], [4.5, 11.0]], dtype=np.float64)
        expected = np.array(base, copy=True)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                max_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                max_ops._select_threshold_max_backend.cache_clear()
                got = custom_threshold_max_merge(
                    frame, mean_img, std_img, base, 2.0, 0.5
                )

        max_ops.threshold_max_merge_numpy(frame, mean_img, std_img, expected, 2.0, 0.5)
        self.assertIs(got, base)
        np.testing.assert_allclose(base, expected, rtol=1e-7, atol=1e-7)

    def test_threshold_max_merge_keeps_morphology_on_numpy_path(self) -> None:
        frame = np.array([[6.0, 12.0], [4.0, 15.0]], dtype=np.float64)
        mean_img = np.array([[5.0, 8.0], [4.5, 10.0]], dtype=np.float64)
        std_img = np.array([[0.5, 1.0], [0.5, 2.0]], dtype=np.float64)
        result = np.array(mean_img, copy=True)
        expected = np.array(mean_img, copy=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        with mock.patch.object(
            noise_equalization, "custom_threshold_max_merge"
        ) as patched_custom:
            noise_equalization.threshold_max_merge(
                frame,
                mean_img,
                std_img,
                result,
                2.0,
                0.5,
                kernel,
            )

        mask = frame > (mean_img + 2.0 * std_img)
        mask = cv2.morphologyEx(mask.view(np.uint8), cv2.MORPH_OPEN, kernel).view(bool)
        signal = frame * 0.5
        np.maximum(expected, np.where(mask, signal, mean_img), out=expected)

        patched_custom.assert_not_called()
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_threshold_max_iterator_routes_noop_morphology_through_custom_kernel(
        self,
    ) -> None:
        class FakeFrameBuffer:
            def __init__(self, items):
                self._items = items
                self.cleaned = False

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

            async def iter_prefetch(self, start=0, stop=None):
                for item in self._items[start:stop]:
                    yield item

            def cleanup(self):
                self.cleaned = True

        fgp_total = FastGaussianParam(
            np.array([[5, 8], [4, 10]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        frame_buffer = FakeFrameBuffer(
            [
                (np.array([[6, 12], [4, 15]], dtype=np.uint16), 0.5),
                (np.array([[4, 9], [9, 12]], dtype=np.uint16), 1.0),
            ]
        )
        op = ThresholdMaxIteratorOp("threshold_max")
        outputs = {}

        async def run_case() -> None:
            with mock.patch.object(
                noise_equalization,
                "custom_threshold_max_merge",
                wraps=max_ops.threshold_max_merge_numpy,
            ) as patched_custom:

                async def run_cpu(fn, *args, **kwargs):
                    return fn(*args, **kwargs)

                async def capture(payload):
                    outputs.update(payload)

                op._run_cpu = run_cpu
                op._broadcast_outputs = capture
                await op._async_execute(
                    {
                        "fgp_total": fgp_total,
                        "buffer_handle": frame_buffer,
                        "n_sigma": 2.0,
                        "morph_kernel_size": 1,
                    }
                )
                self.assertEqual(patched_custom.call_count, len(frame_buffer))

        asyncio.run(run_case())
        self.assertTrue(frame_buffer.cleaned)
        self.assertIn("result", outputs)

    def test_max_combine_can_force_numpy_fallback(self) -> None:
        base = np.array([[1, 5], [3, 4]], dtype=np.uint16)
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                max_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                max_ops._select_max_backend.cache_clear()
                got = max_ops.max_combine(base, fresh)

        expected = np.maximum(np.array([[1, 5], [3, 4]], dtype=np.uint16), fresh)
        self.assertIs(got, base)
        np.testing.assert_array_equal(base, expected)
        np.testing.assert_array_equal(got, expected)
