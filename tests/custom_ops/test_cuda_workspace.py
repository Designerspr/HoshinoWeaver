from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

import numpy as np

import hoshicore._custom_op.ops.max as max_ops


class TestCudaHostIoWorkspace(unittest.TestCase):
    @staticmethod
    def _compiled_module():
        module, error = max_ops._load_compiled_module_result()
        if module is None:
            raise unittest.SkipTest(error or "compiled custom ops unavailable")
        return module

    def test_cuda_host_io_cache_metadata_is_bounded_and_clearable(self) -> None:
        module = self._compiled_module()
        info = module.cuda_host_io_cache_info()

        self.assertIsInstance(info, dict)
        if module.build_info().get("cuda"):
            self.assertTrue(info["available"])
            self.assertGreaterEqual(info["per_thread_limit_bytes"], 0)
            self.assertGreaterEqual(info["process_limit_bytes"], 0)
            self.assertGreaterEqual(info["current_thread_device_bytes"], 0)
            self.assertGreaterEqual(info["current_thread_pinned_bytes"], 0)
            self.assertTrue(module.clear_cuda_host_io_cache())
        else:
            self.assertFalse(info["available"])
            self.assertFalse(module.clear_cuda_host_io_cache())

    def test_cuda_host_io_cache_reuses_slots_across_ops(self) -> None:
        module = self._compiled_module()
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        self.assertTrue(module.clear_cuda_host_io_cache())
        self.addCleanup(module.clear_cuda_host_io_cache)
        stack = np.arange(4 * 64, dtype=np.uint16).reshape(4, 64)
        ref_mean = np.mean(stack, axis=0, dtype=np.float64)
        ref_std = np.std(stack, axis=0, dtype=np.float64)

        module.huber_weighted_chunk_cuda(stack, ref_mean, ref_std, 1.5)
        after_huber = module.cuda_host_io_cache_info()
        self.assertGreater(after_huber["current_thread_device_bytes"], 0)
        self.assertEqual(after_huber["current_thread_pinned_bytes"], 0)

        module.sigma_clip_fused_chunk_cuda(stack, 3.0, 3.0, 2, None, False, 1)
        after_sigma = module.cuda_host_io_cache_info()
        self.assertEqual(
            after_sigma["current_thread_device_bytes"],
            after_huber["current_thread_device_bytes"],
        )

        image = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
        module.camera_model_remap(
            image,
            8,
            8,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            np.eye(3, dtype=np.float32),
        )
        after_remap = module.cuda_host_io_cache_info()
        self.assertEqual(
            after_remap["current_thread_device_bytes"],
            after_huber["current_thread_device_bytes"],
        )
        self.assertGreater(after_remap["current_thread_pinned_bytes"], 0)
        self.assertGreaterEqual(
            after_remap["process_device_bytes"],
            after_remap["current_thread_device_bytes"],
        )

        self.assertTrue(module.clear_cuda_host_io_cache())
        after_clear = module.cuda_host_io_cache_info()
        self.assertEqual(after_clear["current_thread_device_bytes"], 0)
        self.assertEqual(after_clear["current_thread_pinned_bytes"], 0)

    def test_cuda_host_io_cache_isolated_per_worker_and_released_on_exit(self) -> None:
        module = self._compiled_module()
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        self.assertTrue(module.clear_cuda_host_io_cache())
        start_barrier = threading.Barrier(2)
        retained_barrier = threading.Barrier(2)

        def run_worker(offset: int) -> tuple[int, dict]:
            stack = np.arange(4 * 64, dtype=np.uint16).reshape(4, 64) + offset
            ref_mean = np.mean(stack, axis=0, dtype=np.float64)
            ref_std = np.std(stack, axis=0, dtype=np.float64)
            start_barrier.wait(timeout=10)
            module.huber_weighted_chunk_cuda(stack, ref_mean, ref_std, 1.5)
            info = module.cuda_host_io_cache_info()
            retained_barrier.wait(timeout=10)
            return threading.get_ident(), info

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run_worker, (0, 1)))

        self.assertEqual(len({thread_id for thread_id, _ in results}), 2)
        for _, info in results:
            self.assertGreater(info["current_thread_device_bytes"], 0)
            self.assertEqual(info["current_thread_pinned_bytes"], 0)

        deadline = time.monotonic() + 1.0
        after_shutdown = module.cuda_host_io_cache_info()
        while (
            after_shutdown["process_device_bytes"] != 0
            or after_shutdown["process_pinned_bytes"] != 0
        ) and time.monotonic() < deadline:
            time.sleep(0.001)
            after_shutdown = module.cuda_host_io_cache_info()
        self.assertEqual(after_shutdown["process_device_bytes"], 0)
        self.assertEqual(after_shutdown["process_pinned_bytes"], 0)
