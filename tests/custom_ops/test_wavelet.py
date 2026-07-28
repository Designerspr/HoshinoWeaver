import unittest
from unittest import mock

import cv2
import numpy as np

from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op.backend_registry import BackendCandidate
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import registered_backend_candidates
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
from hoshicore._custom_op.ops import wavelet as wavelet_ops
import hoshicore.component.norma.detection as detection
import hoshicore._custom_op.backend_registry as backend_registry
import hoshicore._custom_op.cuda_memory as cuda_memory


def _is_compiled_backend_unavailable(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        is_cuda_runtime_unavailable_error(exc)
        or "compiled custom op backend is unavailable" in message
        or "compiled cuda custom op backend is unavailable" in message
    )


class TestWaveletDecRecCustomOp(unittest.TestCase):
    @staticmethod
    def _admission(granted: bool) -> mock.MagicMock:
        admission = mock.MagicMock()
        admission.__enter__.return_value = mock.Mock(
            granted=granted,
            estimated_peak_bytes=123456,
        )
        return admission

    def tearDown(self) -> None:
        wavelet_ops._load_compiled_module_result.cache_clear()
        wavelet_ops._select_wavelet_dec_rec_backend.cache_clear()
        wavelet_ops._select_wavelet_dec_rec_cuda_core_backend.cache_clear()

    def test_wavelet_dec_rec_core_compiled_matches_pywavelets(self) -> None:
        rng = np.random.default_rng(0)
        cases = [
            ((32, 33), 2),
            ((25, 25), 4),
            ((64, 64), 4),
            ((100, 101), 3),
            ((101, 100), 6),
            ((128, 192), 4),
        ]

        for shape, level in cases:
            with self.subTest(shape=shape, level=level):
                image = np.ascontiguousarray(rng.normal(size=shape))
                expected = wavelet_ops.wavelet_dec_rec_core_numpy(image, level)
                got = wavelet_ops.wavelet_dec_rec_core_compiled(image, level)

                self.assertEqual(got.shape, expected.shape)
                np.testing.assert_allclose(
                    got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_cuda_core_matches_pywavelets(self) -> None:
        rng = np.random.default_rng(12)
        cases = [
            ((32, 33), 2),
            ((64, 63), 4),
            ((101, 100), 6),
        ]

        for shape, level in cases:
            with self.subTest(shape=shape, level=level):
                image = np.ascontiguousarray(rng.normal(size=shape))
                expected = wavelet_ops.wavelet_dec_rec_core_numpy(image, level)
                try:
                    got = wavelet_ops.wavelet_dec_rec_core_cuda(image, level)
                except RuntimeError as exc:
                    if not _is_compiled_backend_unavailable(exc):
                        raise
                    self.skipTest(f"compiled CUDA backend unavailable: {exc}")

                self.assertEqual(got.shape, expected.shape)
                np.testing.assert_allclose(
                    got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_cuda_admission_denial_stops_before_kernel(
        self,
    ) -> None:
        image = np.ones((24, 25), dtype=np.float64)
        native = mock.Mock()
        module = mock.Mock(wavelet_dec_rec_cuda_core=native)

        with mock.patch.object(
            wavelet_ops,
            "_load_compiled_module_result",
            return_value=(module, None),
        ):
            with mock.patch.object(
                cuda_memory,
                "cuda_memory_admission",
                return_value=self._admission(False),
            ):
                with self.assertRaisesRegex(
                    CustomOpResourceExhaustedError,
                    "estimated peak",
                ):
                    wavelet_ops.wavelet_dec_rec_core_cuda(image, 2)

        native.assert_not_called()

    def test_wavelet_cuda_backend_uses_selected_logical_op_memory_model(
        self,
    ) -> None:
        image = np.ones((24, 25), dtype=np.float64)
        expected = np.full_like(image, 2.0)
        native = mock.Mock(return_value=expected)
        module = mock.Mock(wavelet_dec_rec_cuda_core=native)
        selection = BackendSelection(
            BackendCandidate(
                "wavelet_dec_rec",
                "cuda_host_io",
                "wavelet_dec_rec_cuda_core",
            ),
            module,
        )

        with mock.patch.object(
            wavelet_ops,
            "_load_compiled_module_result",
            return_value=(module, None),
        ):
            with mock.patch.object(
                wavelet_ops,
                "cuda_memory_estimate",
                return_value=mock.Mock(logical_op="wavelet_dec_rec"),
            ) as estimate:
                with mock.patch.object(
                    cuda_memory,
                    "cuda_memory_admission",
                    return_value=self._admission(True),
                ):
                    backend_name, backend = wavelet_ops._wavelet_dec_rec_backend(
                        selection
                    )
                    got = backend(image, 2)

        self.assertEqual(backend_name, "cuda")
        estimate.assert_called_once_with(
            "wavelet_dec_rec",
            height=image.shape[0],
            width=image.shape[1],
            level=2,
        )
        native.assert_called_once()
        self.assertIs(got, expected)

    def test_wavelet_dec_rec_wrapper_matches_pywavelets(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.normal(size=(101, 100))

        for resize_factor in (1.0, 0.5, 0.25):
            with self.subTest(resize_factor=resize_factor):
                expected = wavelet_ops.wavelet_dec_rec_core_numpy(
                    cv2.resize(image, None, fx=resize_factor, fy=resize_factor),
                    wavelet_ops._wavelet_level(resize_factor),
                )
                expected = cv2.resize(expected, (image.shape[1], image.shape[0]))
                got = wavelet_ops.wavelet_dec_rec(image, resize_factor)

                self.assertEqual(got.shape, image.shape)
                np.testing.assert_allclose(
                    got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_wrapper_can_use_compiled_backend(self) -> None:
        rng = np.random.default_rng(11)
        image = rng.normal(size=(24, 26))
        expected = wavelet_ops.wavelet_dec_rec(image, 1.0)

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"},
                clear=False):
            with mock.patch.object(
                    wavelet_ops, "MIN_COMPILED_WAVELET_PIXELS", 0):
                with mock.patch.object(
                        wavelet_ops, "MIN_CUDA_WAVELET_PIXELS", 0):
                    with mock.patch.object(
                            wavelet_ops,
                            "wavelet_dec_rec_core_compiled",
                            wraps=wavelet_ops.wavelet_dec_rec_core_compiled,
                    ) as compiled:
                        with mock.patch.object(
                                wavelet_ops,
                                "_select_wavelet_dec_rec_backend",
                                return_value=("compiled", compiled)):
                            with mock.patch.object(
                                    wavelet_ops,
                                    "_apply_compiled_threads") as apply_threads:
                                got = wavelet_ops.wavelet_dec_rec(image, 1.0)

        compiled.assert_called_once()
        apply_threads.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_can_force_numpy_fallback(self) -> None:
        rng = np.random.default_rng(2)
        image = rng.normal(size=(32, 34))

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"},
                clear=False):
            with mock.patch.object(
                    wavelet_ops,
                    "_load_compiled_module_result",
                    return_value=(None, "mock error")):
                with mock.patch.object(
                        wavelet_ops,
                        "MIN_COMPILED_WAVELET_PIXELS",
                        0):
                    with mock.patch.object(
                            wavelet_ops, "MIN_CUDA_WAVELET_PIXELS", 0):
                        wavelet_ops._select_wavelet_dec_rec_backend.cache_clear()
                        got = wavelet_ops.wavelet_dec_rec(image, 0.5)

        expected = wavelet_ops.wavelet_dec_rec_core_numpy(
            cv2.resize(image, None, fx=0.5, fy=0.5),
            wavelet_ops._wavelet_level(0.5),
        )
        expected = cv2.resize(expected, (image.shape[1], image.shape[0]))
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_missing_compiled_backend_falls_back(self) -> None:
        rng = np.random.default_rng(3)
        image = rng.normal(size=(30, 32))
        expected = wavelet_ops.wavelet_dec_rec(image, 1.0)

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"},
                clear=False):
            with mock.patch.object(
                    wavelet_ops, "_load_compiled_module_result",
                    return_value=(None, "mock error")) as load_module:
                with mock.patch.object(
                        wavelet_ops, "MIN_COMPILED_WAVELET_PIXELS", 0):
                    with mock.patch.object(
                            wavelet_ops, "MIN_CUDA_WAVELET_PIXELS", 0):
                        wavelet_ops._select_wavelet_dec_rec_backend.cache_clear()
                        got = wavelet_ops.wavelet_dec_rec(image, 1.0)

        load_module.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_cuda_runtime_falls_back_to_cpu(self) -> None:
        rng = np.random.default_rng(14)
        image = rng.normal(size=(24, 26))
        expected = np.full_like(image, 7.0)

        def raise_no_cuda(_image: np.ndarray, _level: int) -> np.ndarray:
            raise RuntimeError(
                "wavelet_dec_rec_cuda_core cudaMalloc input: "
                "no CUDA-capable device is detected")

        cpu_selection = BackendSelection(
            BackendCandidate(
                "wavelet_dec_rec",
                "openmp_cpu",
                "wavelet_dec_rec_cpu",
            ),
            object(),
        )

        with mock.patch.object(
                wavelet_ops, "_select_wavelet_dec_rec_backend",
                return_value=("cuda", raise_no_cuda)):
            with mock.patch.object(
                    backend_registry,
                    "resolve_after_runtime_unavailable",
                    return_value=cpu_selection):
                with mock.patch.object(
                        wavelet_ops,
                        "wavelet_dec_rec_core_compiled",
                        return_value=expected) as cpu_backend:
                    got = wavelet_ops.wavelet_dec_rec_core(image, 6)

        cpu_backend.assert_called_once()
        self.assertIs(got, expected)

    def test_wavelet_dec_rec_cuda_fallback_propagates_cpu_runtime_error(self) -> None:
        rng = np.random.default_rng(15)
        image = rng.normal(size=(24, 26))

        def raise_no_cuda(_image: np.ndarray, _level: int) -> np.ndarray:
            raise RuntimeError(
                "wavelet_dec_rec_cuda_core cudaMalloc input: "
                "no CUDA-capable device is detected")

        cpu_selection = BackendSelection(
            BackendCandidate(
                "wavelet_dec_rec",
                "openmp_cpu",
                "wavelet_dec_rec_cpu",
            ),
            object(),
        )

        with mock.patch.object(
                wavelet_ops, "_select_wavelet_dec_rec_backend",
                return_value=("cuda", raise_no_cuda)):
            with mock.patch.object(
                    backend_registry,
                    "resolve_after_runtime_unavailable",
                    return_value=cpu_selection):
                with mock.patch.object(
                        wavelet_ops,
                        "wavelet_dec_rec_core_compiled",
                        side_effect=RuntimeError("native CPU wavelet bug")):
                    with mock.patch.object(
                            wavelet_ops,
                            "wavelet_dec_rec_core_numpy",
                            side_effect=AssertionError("numpy backend should not be called")):
                        with self.assertRaisesRegex(RuntimeError, "native CPU wavelet bug"):
                            wavelet_ops.wavelet_dec_rec_core(image, 6)

    def test_wavelet_dec_rec_cuda_resource_exhaustion_falls_back_to_cpu(
        self,
    ) -> None:
        image = np.ones((24, 26), dtype=np.float64)
        expected = np.full_like(image, 5.0)

        def raise_resource(_image: np.ndarray, _level: int) -> np.ndarray:
            raise CustomOpResourceExhaustedError("admission denied")

        cpu_selection = BackendSelection(
            BackendCandidate(
                "wavelet_dec_rec",
                "openmp_cpu",
                "wavelet_dec_rec_cpu",
            ),
            object(),
        )
        with mock.patch.object(
            wavelet_ops,
            "_select_wavelet_dec_rec_backend",
            return_value=("cuda", raise_resource),
        ):
            with mock.patch.object(
                backend_registry,
                "resolve_after_resource_exhausted",
                return_value=cpu_selection,
            ) as resolve:
                with mock.patch.object(
                    wavelet_ops,
                    "wavelet_dec_rec_core_compiled",
                    return_value=expected,
                ) as cpu_backend:
                    got = wavelet_ops.wavelet_dec_rec_core(image, 2)

        resolve.assert_called_once()
        cpu_backend.assert_called_once()
        self.assertIs(got, expected)

    def test_wavelet_cuda_core_resource_exhaustion_falls_back_to_numpy(
        self,
    ) -> None:
        image = np.ones((24, 26), dtype=np.float64)
        expected = np.full_like(image, 4.0)

        def raise_resource(_image: np.ndarray, _level: int) -> np.ndarray:
            raise CustomOpResourceExhaustedError("admission denied")

        numpy_selection = BackendSelection(
            None,
            None,
            "CUDA resource exhausted",
            "cuda_resource_exhausted",
        )
        with mock.patch.object(
            wavelet_ops,
            "_select_wavelet_dec_rec_cuda_core_backend",
            return_value=("compiled", raise_resource),
        ):
            with mock.patch.object(
                backend_registry,
                "resolve_after_resource_exhausted",
                return_value=numpy_selection,
            ) as resolve:
                with mock.patch.object(
                    wavelet_ops,
                    "wavelet_dec_rec_core_numpy",
                    return_value=expected,
                ) as numpy_backend:
                    got = wavelet_ops.wavelet_dec_rec_core_cuda_or_numpy(image, 2)

        resolve.assert_called_once()
        numpy_backend.assert_called_once()
        self.assertIs(got, expected)

    def test_wavelet_dec_rec_backend_registered(self) -> None:
        candidates = registered_backend_candidates("wavelet_dec_rec")
        self.assertTrue(
            any(candidate.kernel_name == "wavelet_dec_rec_cpu"
                and candidate.backend == "openmp_cpu"
                for candidate in candidates))
        self.assertTrue(
            any(candidate.kernel_name == "wavelet_dec_rec_cuda_core"
                and candidate.backend == "cuda_host_io"
                and candidate.build_flag == "cuda"
                for candidate in candidates))

    def test_wavelet_dec_rec_cuda_core_backend_registered(self) -> None:
        candidates = registered_backend_candidates("wavelet_dec_rec_cuda_core")
        self.assertTrue(
            any(candidate.kernel_name == "wavelet_dec_rec_cuda_core"
                and candidate.backend == "cuda_host_io"
                and candidate.build_flag == "cuda"
                for candidate in candidates))

    def test_wavelet_dec_rec_cuda_core_falls_back_when_unavailable(self) -> None:
        rng = np.random.default_rng(13)
        image = rng.normal(size=(24, 25))
        expected = wavelet_ops.wavelet_dec_rec_core_numpy(image, 2)

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"},
                clear=False):
            with mock.patch.object(
                    wavelet_ops,
                    "_load_compiled_module_result",
                    return_value=(None, "mock error")) as load_module:
                wavelet_ops._select_wavelet_dec_rec_cuda_core_backend.cache_clear()
                got = wavelet_ops.wavelet_dec_rec_core_cuda_or_numpy(image, 2)

        load_module.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_public_facade_exported(self) -> None:
        from hoshicore._custom_op import wavelet_dec_rec
        from hoshicore._custom_op.api import wavelet_dec_rec as api_wavelet_dec_rec

        self.assertIs(wavelet_dec_rec, wavelet_ops.wavelet_dec_rec)
        self.assertIs(api_wavelet_dec_rec, wavelet_ops.wavelet_dec_rec)

    def test_detection_routes_wavelet_through_custom_op(self) -> None:
        image = np.ones((8, 10), dtype=np.float64)
        expected = np.full_like(image, 3.0)

        with mock.patch.object(
                detection, "wavelet_dec_rec", return_value=expected) as patched:
            got = detection._wavelet_dec_rec(image, resize_factor=0.5)

        patched.assert_called_once_with(image, resize_factor=0.5)
        self.assertIs(got, expected)
