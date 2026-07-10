import unittest
from unittest import mock

import numpy as np

import hoshicore._custom_op._dispatch as custom_op_dispatch
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
import hoshicore._custom_op.ops.sigma_clip as sigma_clip_chunk_ops


class TestCustomOpDispatchHelpers(unittest.TestCase):
    def test_compiled_threads_are_applied_on_every_native_call(self) -> None:
        module = mock.Mock()
        module.set_openmp_threads.return_value = True
        sample = np.zeros((8, 8), dtype=np.float32)

        with (
            mock.patch.object(
                custom_op_dispatch,
                "load_compiled_module",
                return_value=(module, None),
            ),
            mock.patch.object(
                custom_op_dispatch,
                "compiled_build_info",
                return_value={"openmp": True},
            ),
            mock.patch.object(
                custom_op_dispatch.thread_tuning,
                "resolve_runtime_threads",
                return_value=3,
            ),
        ):
            custom_op_dispatch.apply_compiled_threads("max_combine", sample)
            custom_op_dispatch.apply_compiled_threads("max_combine", sample)

        self.assertEqual(module.set_openmp_threads.call_count, 2)
        module.set_openmp_threads.assert_has_calls([mock.call(3), mock.call(3)])

    def test_cuda_runtime_unavailable_classifier_does_not_hide_resource_errors(
        self,
    ) -> None:
        self.assertFalse(
            is_cuda_runtime_unavailable_error(
                RuntimeError("kernel cudaMalloc input: out of memory")
            )
        )
        self.assertFalse(
            is_cuda_runtime_unavailable_error(
                RuntimeError("kernel cudaMallocHost input: memory allocation failed")
            )
        )
        self.assertFalse(
            is_cuda_runtime_unavailable_error(
                RuntimeError("kernel launch failed: invalid device pointer")
            )
        )
        self.assertFalse(
            is_cuda_runtime_unavailable_error(
                RuntimeError("kernel synchronize failed: CUDA unknown error")
            )
        )
        self.assertTrue(
            is_cuda_runtime_unavailable_error(
                RuntimeError("cudaGetDevice: no CUDA-capable device is detected")
            )
        )
        self.assertTrue(
            is_cuda_runtime_unavailable_error(
                RuntimeError(
                    "kernel launch: no kernel image is available for execution on the device"
                )
            )
        )

    def test_cuda_runtime_unavailable_classifier_prefers_structured_error(self) -> None:
        class CudaRuntimeUnavailableError(RuntimeError):
            pass

        module = mock.Mock(CudaRuntimeUnavailableError=CudaRuntimeUnavailableError)
        with mock.patch.object(
            custom_op_dispatch,
            "load_compiled_module",
            return_value=(module, None),
        ):
            self.assertTrue(
                is_cuda_runtime_unavailable_error(
                    CudaRuntimeUnavailableError("structured CUDA runtime failure")
                )
            )
            self.assertFalse(
                is_cuda_runtime_unavailable_error(
                    RuntimeError("structured CUDA runtime failure")
                )
            )

    def test_sigma_clip_cpu_entrypoints_apply_compiled_threads(self) -> None:
        module = mock.Mock()
        module.sigma_clip_iterative_chunk.return_value = (
            np.zeros(4),
            np.zeros(4),
            np.zeros(4),
        )
        module.sigma_clip_fused_chunk.return_value = (
            np.zeros(4),
            np.zeros(4),
            np.zeros(4),
        )
        stack = np.ones((2, 4), dtype=np.uint16)
        total_sum = np.ones(4, dtype=np.float64)
        total_sq = np.ones(4, dtype=np.float64)
        total_n = np.ones(4, dtype=np.float64)

        with (
            mock.patch.object(
                sigma_clip_chunk_ops,
                "_load_compiled_module_result",
                return_value=(module, None),
            ),
            mock.patch.object(
                sigma_clip_chunk_ops,
                "_apply_compiled_threads",
            ) as apply_threads,
        ):
            sigma_clip_chunk_ops.sigma_clip_iterative_chunk_compiled(
                stack, total_sum, total_sq, total_n
            )
            sigma_clip_chunk_ops.sigma_clip_fused_chunk_compiled(stack)

        self.assertEqual(
            [call.args[0] for call in apply_threads.call_args_list],
            ["sigma_clip_iterative_chunk", "sigma_clip_fused_chunk"],
        )
        for call in apply_threads.call_args_list:
            self.assertIs(call.args[1], stack)
