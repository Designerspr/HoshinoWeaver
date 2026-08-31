import asyncio

import numpy as np

import hoshicore.ops.sigma_clip_ops as sigma_clip_ops


from tests.custom_ops._base import CustomOpsTestCase


class TestCustomOpIntegration(CustomOpsTestCase):
    def test_chunk_iterator_base_caches_configs_for_hooks(self) -> None:
        frame = np.array([[1, 2]], dtype=np.uint16)

        class FakeFrameBuffer:
            def __getitem__(self, idx):
                if idx != 0:
                    raise IndexError(idx)
                return frame, None

            async def iter_chunk_prefetch(self, row_ranges):
                for _ in row_ranges:
                    yield [(frame, None)]

            def cleanup(self):
                self.cleaned = True

        class ConfigCaptureChunkOp(sigma_clip_ops.ChunkIteratorBaseOp):
            OUTPUTS = {"result": {"type": "image"}}

            def _init_chunk_state(self, configs, row_start, row_end, w):
                self.init_marker = self._configs["marker"]
                return {}

            def _run_pass(self, state, chunk_stack):
                self.run_marker = self._configs["marker"]

            def _check_convergence(self, state, pass_idx):
                return True

            def _finalize_chunk(self, state):
                return frame.copy()

            def _wrap_output(self, result, configs):
                self.wrap_marker = self._configs["marker"]
                return {"result": result}

        buffer = FakeFrameBuffer()
        op = ConfigCaptureChunkOp("capture")

        async def run_inline(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        op._run_cpu = run_inline
        configs = {"buffer_handle": buffer, "chunk_rows": 1, "marker": "ok"}

        asyncio.run(op._async_execute(configs))

        self.assertIs(op._configs, configs)
        self.assertEqual(op.init_marker, "ok")
        self.assertEqual(op.run_marker, "ok")
        self.assertEqual(op.wrap_marker, "ok")
        self.assertTrue(buffer.cleaned)
