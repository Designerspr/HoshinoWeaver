import os
import unittest
from unittest import mock

import numpy as np

import hoshicore._custom_op as custom_ops
import hoshicore._custom_op._dispatch as custom_op_dispatch
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
from hoshicore._custom_op._dispatch import is_cuda_resource_exhausted_error
import hoshicore._custom_op.ops.sigma_clip as sigma_clip_chunk_ops


class TestCustomOpDispatchHelpers(unittest.TestCase):
    def tearDown(self) -> None:
        custom_op_dispatch.set_backend_preference(None)

    def test_backend_preference_accepts_cpu_from_environment(self) -> None:
        with mock.patch.dict(
            os.environ, {"HNW_CUSTOM_OPS_FALLBACK": "cpu"}, clear=False
        ):
            self.assertEqual(
                custom_op_dispatch.get_backend_preference(), "cpu")

    def test_backend_preference_runtime_override_and_reset(self) -> None:
        with mock.patch.dict(
            os.environ, {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            custom_op_dispatch.set_backend_preference("cpu")
            self.assertEqual(
                custom_op_dispatch.get_backend_preference(), "cpu")
            custom_op_dispatch.set_backend_preference(None)
            self.assertEqual(
                custom_op_dispatch.get_backend_preference(), "numpy")

    def test_backend_preference_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto, cpu, numpy"):
            custom_op_dispatch.set_backend_preference("gpu")

    def test_backend_preference_is_exposed_by_public_facade(self) -> None:
        custom_ops.set_backend_preference("cpu")
        self.assertEqual(custom_ops.get_backend_preference(), "cpu")

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
        self.assertFalse(
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

    def test_cuda_runtime_unavailable_classifier_accepts_probe_exception(self) -> None:
        exc = custom_op_dispatch.CustomOpCudaRuntimeUnavailableError(
            "CUDA compute capability 5.2 is unsupported",
            reason_code="cuda_compute_capability_unsupported",
        )

        self.assertTrue(is_cuda_runtime_unavailable_error(exc))
        self.assertEqual(
            exc.reason_code,
            "cuda_compute_capability_unsupported",
        )

    def test_cuda_resource_classifier_requires_structured_error(self) -> None:
        class CudaResourceExhaustedError(RuntimeError):
            pass

        module = mock.Mock(
            CudaResourceExhaustedError=CudaResourceExhaustedError)
        with mock.patch.object(
            custom_op_dispatch,
            "load_compiled_module",
            return_value=(module, None),
        ):
            self.assertTrue(
                is_cuda_resource_exhausted_error(
                    CudaResourceExhaustedError("out of memory")
                )
            )
            self.assertFalse(
                is_cuda_resource_exhausted_error(
                    RuntimeError("cudaMalloc: out of memory")
                )
            )

    def test_cuda_resource_classifier_accepts_actual_native_exception(self) -> None:
        module, error = custom_op_dispatch.load_compiled_module()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        if not hasattr(module, "CudaResourceExhaustedError"):
            self.skipTest("CUDA resource exception is not built")

        exc = module.CudaResourceExhaustedError("typed native resource error")
        self.assertTrue(is_cuda_resource_exhausted_error(exc))
        self.assertFalse(is_cuda_runtime_unavailable_error(exc))
        self.assertFalse(
            is_cuda_resource_exhausted_error(
                RuntimeError("cudaMalloc: out of memory")
            )
        )

    def test_cuda_memory_probe_propagates_runtime_errors(self) -> None:
        module = mock.Mock()
        module.cuda_memory_info.side_effect = RuntimeError(
            "cudaMemGetInfo: out of memory"
        )
        with mock.patch.object(
            custom_op_dispatch,
            "load_compiled_module",
            return_value=(module, None),
        ):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                custom_op_dispatch.cuda_memory_info()

    def test_cuda_memory_probe_reports_supported_compute_capability(self) -> None:
        module, error = custom_op_dispatch.load_compiled_module()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        if not module.build_info().get("cuda"):
            self.skipTest("CUDA backend is not built")

        payload = custom_op_dispatch.cuda_memory_info()
        if not payload.get("available"):
            self.skipTest(str(payload.get("reason") or "CUDA runtime unavailable"))

        capability = (
            payload["compute_capability_major"],
            payload["compute_capability_minor"],
        )
        minimum = (
            payload["minimum_compute_capability_major"],
            payload["minimum_compute_capability_minor"],
        )
        self.assertGreaterEqual(capability, minimum)
        self.assertEqual(minimum, (6, 0))

    def test_cuda_memory_probe_rejects_incomplete_available_payload(self) -> None:
        module = mock.Mock()
        module.cuda_memory_info.return_value = {
            "available": True,
            "status": "available",
        }
        with mock.patch.object(
            custom_op_dispatch,
            "load_compiled_module",
            return_value=(module, None),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing byte counts"):
                custom_op_dispatch.cuda_memory_info()

    def test_cuda_memory_probe_raises_structured_error(self) -> None:
        module = mock.Mock()
        module.cuda_memory_info.return_value = {
            "available": False,
            "status": "error",
            "reason_code": "cuda_runtime_error",
            "error_code": 2,
            "category": "resource",
            "reason": "out of memory",
        }
        with mock.patch.object(
            custom_op_dispatch,
            "load_compiled_module",
            return_value=(module, None),
        ):
            with self.assertRaisesRegex(
                custom_op_dispatch.CudaProbeError, "out of memory"
            ) as caught:
                custom_op_dispatch.cuda_memory_info()

        self.assertEqual(caught.exception.error_code, 2)
        self.assertEqual(caught.exception.category, "resource")

    def test_cuda_memory_probe_rejects_inconsistent_status(self) -> None:
        module = mock.Mock()
        module.cuda_memory_info.return_value = {
            "available": True,
            "status": "error",
            "free_bytes": 1024,
            "total_bytes": 2048,
        }
        with mock.patch.object(
            custom_op_dispatch,
            "load_compiled_module",
            return_value=(module, None),
        ):
            with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                custom_op_dispatch.cuda_memory_info()

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
