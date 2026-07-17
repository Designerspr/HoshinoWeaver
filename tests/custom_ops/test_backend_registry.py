import json
from unittest import mock


from hoshicore._custom_op import (
    build_info,
    custom_ops_available,
)
import hoshicore._custom_op.backend_registry as backend_registry
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op.cuda_memory import cuda_chunk_memory_model
from hoshicore._custom_op.cuda_memory import cuda_memory_estimate
from hoshicore._custom_op.cuda_memory import cuda_memory_model_kind
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

        matching_candidates = backend_registry.registered_backend_candidates(
            "matching_cosine_bidirectional_nearest"
        )
        self.assertEqual(len(matching_candidates), 2)
        self.assertEqual(
            {candidate.backend for candidate in matching_candidates},
            {"cuda_host_io", "openmp_cpu"},
        )

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

    def test_builtin_cuda_candidates_declare_consumed_memory_models(self) -> None:
        candidates = backend_registry.registered_backend_candidates()

        backend_registry.validate_cuda_memory_policy_declarations(
            candidates,
            require_memory_models=True,
        )
        cuda_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.backend == "cuda_host_io"
        )
        self.assertTrue(cuda_candidates)
        self.assertTrue(
            all(candidate.memory_model is not None for candidate in cuda_candidates)
        )
        for candidate in cuda_candidates:
            self.assertEqual(
                candidate.memory_model,
                cuda_memory_model_kind(candidate.logical_op),
            )
        self.assertEqual(
            {candidate.logical_op for candidate in cuda_candidates},
            {
                "camera_model_remap",
                "huber_weighted_chunk",
                "matching_cosine_bidirectional_nearest",
                "sigma_clip_fused_chunk",
                "star_detect_fused_pixel_components",
                "star_mask_dog",
                "star_shrink_dog_process",
                "star_shrink_process",
                "wavelet_dec_rec",
                "wavelet_dec_rec_cuda_core",
            },
        )

    def test_registered_cuda_chunk_models_are_consumable(self) -> None:
        chunk_candidates = (
            candidate
            for candidate in backend_registry.registered_backend_candidates()
            if candidate.memory_model == "cuda_chunk"
        )

        for candidate in chunk_candidates:
            model = cuda_chunk_memory_model(
                candidate.logical_op,
                n_frames=4,
                row_bytes=128,
                dtype_bytes=2,
            )
            self.assertGreater(model.device_bytes_per_row, 0)

    def test_registered_cuda_non_chunk_models_are_consumable(self) -> None:
        sample_args = {
            "camera_model_remap": {
                "source_height": 8,
                "source_width": 10,
                "channels": 3,
                "dtype_bytes": 2,
                "out_height": 6,
                "out_width": 7,
            },
            "matching_cosine_bidirectional_nearest": {
                "n1": 17,
                "n2": 19,
                "feature_dim": 11,
            },
            "star_detect_fused_pixel_components": {
                "height": 64,
                "width": 80,
                "small_height": 16,
                "small_width": 20,
                "level": 2,
                "gaussian_ksize": 9,
            },
            "star_mask_dog": {
                "height": 32,
                "width": 48,
                "channels": 3,
                "dtype_bytes": 2,
                "small_kernel_size": 9,
                "large_kernel_size": 73,
            },
            "star_shrink_dog_process": {
                "height": 32,
                "width": 48,
                "channels": 3,
                "dtype_bytes": 2,
                "small_kernel_size": 9,
                "large_kernel_size": 73,
            },
            "star_shrink_process": {
                "height": 32,
                "width": 48,
                "channels": 3,
                "dtype_bytes": 2,
            },
            "wavelet_dec_rec": {"height": 64, "width": 80, "level": 2},
            "wavelet_dec_rec_cuda_core": {
                "height": 64,
                "width": 80,
                "level": 2,
            },
        }
        candidates = (
            candidate
            for candidate in backend_registry.registered_backend_candidates()
            if candidate.backend == "cuda_host_io"
            and candidate.memory_model != "cuda_chunk"
        )

        for candidate in candidates:
            estimate = cuda_memory_estimate(
                candidate.logical_op,
                **sample_args[candidate.logical_op],
            )
            self.assertEqual(estimate.logical_op, candidate.logical_op)
            self.assertGreater(estimate.peak_device_bytes, 0)

    def test_cuda_candidate_without_memory_declaration_is_rejected(self) -> None:
        candidate = backend_registry.BackendCandidate(
            "missing_memory_policy",
            "cuda_host_io",
            "missing_memory_policy_cuda",
            build_flag="cuda",
        )

        with self.assertRaisesRegex(RuntimeError, "must declare"):
            backend_registry.validate_cuda_memory_policy_declarations(
                (candidate,))

    def test_explicit_third_party_memory_deferral_is_compatible_but_not_builtin(
        self,
    ) -> None:
        candidate = backend_registry.BackendCandidate(
            "third_party_cuda_op",
            "cuda_host_io",
            "third_party_cuda_op_kernel",
            build_flag="cuda",
            memory_model_reason="third-party compatibility declaration",
        )

        backend_registry.validate_cuda_memory_policy_declarations((candidate,))
        with self.assertRaisesRegex(RuntimeError, "must declare a consumed"):
            backend_registry.validate_cuda_memory_policy_declarations(
                (candidate,),
                require_memory_models=True,
            )

    def test_cuda_candidate_with_unknown_memory_model_is_rejected(self) -> None:
        candidate = backend_registry.BackendCandidate(
            "unknown_memory_model",
            "cuda_host_io",
            "unknown_memory_model_cuda",
            build_flag="cuda",
            memory_model="typo",
        )

        with self.assertRaisesRegex(RuntimeError, "unknown memory model"):
            backend_registry.validate_cuda_memory_policy_declarations(
                (candidate,))

    def test_cuda_candidate_model_must_resolve_to_an_estimator(self) -> None:
        candidate = backend_registry.BackendCandidate(
            "missing_static_estimator",
            "cuda_host_io",
            "missing_static_estimator_cuda",
            build_flag="cuda",
            memory_model="static_estimator",
        )

        with self.assertRaisesRegex(RuntimeError, "without a registered estimator"):
            backend_registry.validate_cuda_memory_policy_declarations((candidate,))

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

    def test_backend_decision_is_serializable_and_sanitized(self) -> None:
        class Module:
            def median_reduce_chunk(self):
                return None

        selection = backend_registry.select_backend(
            "median_reduce_chunk",
            load_module=lambda: (Module(), None),
        )

        payload = selection.to_decision("median_reduce_chunk").to_dict()

        json.dumps(payload)
        self.assertEqual(payload["backend"], "openmp_cpu")
        self.assertEqual(payload["reason_code"], "selected_native")
        self.assertNotIn("module", payload)
        self.assertNotIn("reason_detail", payload)

    def test_runtime_resolver_keeps_available_cuda_candidate(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        selection = backend_registry.resolve_backend(
            "camera_model_remap",
            load_module=lambda: (Module(), None),
            cuda_probe=lambda: {
                "available": True,
                "status": "available",
                "free_bytes": 1024,
                "total_bytes": 2048,
            },
        )

        self.assertEqual(selection.backend, "cuda_host_io")
        self.assertIsNotNone(selection.decision)
        self.assertEqual(selection.decision.reason_code, "selected_native")

    def test_runtime_resolver_falls_back_to_cpu_when_cuda_unavailable(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        selection = backend_registry.resolve_backend(
            "camera_model_remap",
            load_module=lambda: (Module(), None),
            cuda_probe=lambda: {
                "available": False,
                "status": "explicitly_unavailable",
                "reason": "mock no device",
            },
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.reason_code, "cuda_runtime_unavailable")
        self.assertEqual(selection.decision.reason_code, "cuda_runtime_unavailable")

    def test_runtime_resolver_does_not_hide_probe_errors(self) -> None:
        class Module:
            def huber_weighted_chunk_cuda(self):
                return None

            def build_info(self):
                return {"cuda": True}

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            backend_registry.resolve_backend(
                "huber_weighted_chunk",
                load_module=lambda: (Module(), None),
                cuda_probe=lambda: (_ for _ in ()).throw(
                    RuntimeError("cudaMemGetInfo: out of memory")
                ),
            )

    def test_runtime_resolver_rejects_generic_unavailable_status(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        with self.assertRaisesRegex(RuntimeError, "without an unavailable status"):
            backend_registry.resolve_backend(
                "camera_model_remap",
                load_module=lambda: (Module(), None),
                cuda_probe=lambda: {
                    "available": False,
                    "status": "unavailable",
                    "reason": "out of memory",
                },
            )

    def test_forced_numpy_skips_module_load_and_cuda_probe(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("module should not load"))
        probe = mock.Mock(side_effect=AssertionError("CUDA should not probe"))

        selection = backend_registry.resolve_backend(
            "camera_model_remap",
            "numpy",
            load_module=loader,
            cuda_probe=probe,
        )

        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.decision.reason_code, "forced_numpy")
        loader.assert_not_called()
        probe.assert_not_called()

    def test_runtime_retry_rejects_resource_errors_before_reresolve(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("resolver should not run"))

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            backend_registry.resolve_after_runtime_unavailable(
                "sigma_clip_fused_chunk",
                "cuda_host_io",
                RuntimeError("cudaMalloc: out of memory"),
                load_module=loader,
            )

        loader.assert_not_called()

    def test_resource_resolver_selects_cpu_with_distinct_reason(self) -> None:
        class Module:
            def sigma_clip_fused_chunk_cuda(self):
                return None

            def sigma_clip_fused_chunk(self):
                return None

            def build_info(self):
                return {"cuda": True}

        selection = backend_registry.resolve_after_resource_exhausted(
            "sigma_clip_fused_chunk",
            "cuda_host_io",
            CustomOpResourceExhaustedError("estimated VRAM is insufficient"),
            load_module=lambda: (Module(), None),
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.reason_code, "cuda_resource_exhausted")
        self.assertEqual(
            selection.decision.reason_code, "cuda_resource_exhausted")

    def test_resource_resolver_selects_numpy_without_cpu_candidate(self) -> None:
        class Module:
            def huber_weighted_chunk_cuda(self):
                return None

            def build_info(self):
                return {"cuda": True}

        selection = backend_registry.resolve_after_resource_exhausted(
            "huber_weighted_chunk",
            "cuda_host_io",
            CustomOpResourceExhaustedError("estimated VRAM is insufficient"),
            load_module=lambda: (Module(), None),
        )

        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.reason_code, "cuda_resource_exhausted")
        self.assertEqual(
            selection.decision.reason_code, "cuda_resource_exhausted")

    def test_resource_resolver_rejects_unstructured_oom_errors(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("resolver should not run"))

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            backend_registry.resolve_after_resource_exhausted(
                "sigma_clip_fused_chunk",
                "cuda_host_io",
                RuntimeError("cudaMalloc: out of memory"),
                load_module=loader,
            )

        loader.assert_not_called()
