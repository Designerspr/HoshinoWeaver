from unittest import mock


from hoshicore._custom_op import (
    build_info,
    custom_ops_available,
)
import hoshicore._custom_op.backend_registry as backend_registry
import hoshicore._custom_op.ops.max as max_ops


from tests.custom_ops._base import CustomOpsTestCase


class TestBackendRegistry(CustomOpsTestCase):
    def test_custom_ops_available_returns_bool(self) -> None:
        self.assertIsInstance(custom_ops_available(), bool)

    def test_build_info_returns_minimal_metadata(self) -> None:
        info = build_info()
        self.assertIsInstance(info, dict)
        self.assertIn("available", info)

    def test_build_info_reports_fallback_backend(self) -> None:
        with mock.patch.object(
            max_ops, "_load_compiled_module_result", return_value=(None, "mock error")
        ):
            max_ops._select_max_backend.cache_clear()
            info = max_ops.build_info()

        self.assertFalse(info["available"])
        self.assertEqual(info["backend"], "numpy")

    def test_build_info_includes_thread_policy(self) -> None:
        info = build_info()
        self.assertIn("thread_policy", info)

    def test_backend_registry_exposes_registered_candidates(self) -> None:
        candidates = backend_registry.registered_backend_candidates(
            "median_reduce_chunk"
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].logical_op, "median_reduce_chunk")
        self.assertEqual(candidates[0].backend, "openmp_cpu")
        self.assertEqual(candidates[0].kernel_name, "median_reduce_chunk")

        filter_candidates = backend_registry.registered_backend_candidates(
            "median_filter_2d"
        )
        self.assertEqual(len(filter_candidates), 1)
        self.assertEqual(filter_candidates[0].backend, "openmp_cpu")
        self.assertEqual(filter_candidates[0].kernel_name, "median_filter_2d")

        feature_candidates = backend_registry.registered_backend_candidates(
            "extract_point_features"
        )
        self.assertEqual(len(feature_candidates), 1)
        self.assertEqual(feature_candidates[0].backend, "openmp_cpu")
        self.assertEqual(feature_candidates[0].kernel_name, "extract_point_features")

        match_candidates = backend_registry.registered_backend_candidates(
            "find_initial_match"
        )
        self.assertEqual(len(match_candidates), 1)
        self.assertEqual(match_candidates[0].backend, "openmp_cpu")
        self.assertEqual(match_candidates[0].kernel_name, "find_initial_match")

        remap_candidates = backend_registry.registered_backend_candidates(
            "camera_model_remap"
        )
        self.assertTrue(
            any(
                candidate.backend == "cuda_host_io"
                and candidate.kernel_name == "camera_model_remap"
                for candidate in remap_candidates
            )
        )
        self.assertTrue(
            any(
                candidate.backend == "openmp_cpu"
                and candidate.kernel_name == "camera_model_remap_cpu"
                for candidate in remap_candidates
            )
        )

        huber_chunk_candidates = backend_registry.registered_backend_candidates(
            "huber_weighted_chunk"
        )
        self.assertEqual(len(huber_chunk_candidates), 1)
        self.assertEqual(huber_chunk_candidates[0].backend, "cuda_host_io")
        self.assertEqual(
            huber_chunk_candidates[0].kernel_name, "huber_weighted_chunk_cuda"
        )

    def test_backend_registry_reports_missing_compiled_module(self) -> None:
        selection = backend_registry.select_backend(
            "median_reduce_chunk",
            load_module=lambda: (None, "mock import error"),
        )

        self.assertFalse(selection.native)
        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.reason, "mock import error")

    def test_backend_registry_selects_native_kernel(self) -> None:
        class Module:
            pass

        module = Module()
        module.median_reduce_chunk = lambda stack: stack

        selection = backend_registry.select_backend(
            "median_reduce_chunk",
            load_module=lambda: (module, None),
        )

        self.assertTrue(selection.native)
        self.assertIs(selection.module, module)
        self.assertEqual(selection.backend, "openmp_cpu")

    def test_backend_registry_respects_build_flag(self) -> None:
        class Module:
            def wavelet_dec_rec_cuda_core(self):
                return None

            def build_info(self):
                return {"cuda": False}

        module = Module()

        selection = backend_registry.select_backend(
            "wavelet_dec_rec_cuda_core",
            load_module=lambda: (module, None),
        )

        self.assertFalse(selection.native)
        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.reason, "compiled backend missing build flag: cuda")

    def test_backend_registry_selects_remap_cpu_after_cuda_build_flag_miss(
        self,
    ) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": False}

        module = Module()

        selection = backend_registry.select_backend(
            "camera_model_remap",
            load_module=lambda: (module, None),
        )

        self.assertTrue(selection.native)
        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.candidate.kernel_name, "camera_model_remap_cpu")

    def test_backend_registry_can_exclude_failed_runtime_backend(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        module = Module()

        selection = backend_registry.select_backend(
            "camera_model_remap",
            load_module=lambda: (module, None),
            exclude_backends={"cuda_host_io"},
        )

        self.assertTrue(selection.native)
        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.candidate.kernel_name, "camera_model_remap_cpu")

    def test_backend_registry_reports_when_all_candidates_are_excluded(self) -> None:
        class Module:
            def huber_weighted_chunk_cuda(self):
                return None

            def build_info(self):
                return {"cuda": True}

        module = Module()

        selection = backend_registry.select_backend(
            "huber_weighted_chunk",
            load_module=lambda: (module, None),
            exclude_backends={"cuda_host_io"},
        )

        self.assertFalse(selection.native)
        self.assertEqual(selection.backend, "numpy")
        self.assertIn("after excluding: cuda_host_io", selection.reason)

    def test_backend_registry_continues_after_build_flag_miss(self) -> None:
        class Module:
            def wavelet_dec_rec_cuda_core(self):
                return None

            def wavelet_dec_rec_cpu(self):
                return None

            def build_info(self):
                return {"cuda": False}

        module = Module()

        selection = backend_registry.select_backend(
            "wavelet_dec_rec",
            load_module=lambda: (module, None),
        )

        self.assertTrue(selection.native)
        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.candidate.kernel_name, "wavelet_dec_rec_cpu")
