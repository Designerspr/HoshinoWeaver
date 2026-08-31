import unittest
from unittest import mock

import numpy as np

from hoshicore._custom_op import build_info, camera_model_remap
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
import hoshicore._custom_op.backend_registry as backend_registry
import hoshicore._custom_op.cuda_memory as cuda_memory
import hoshicore._custom_op.ops.remap as remap_ops
import hoshicore.component.norma.types as norma_types
from hoshicore.component.norma.types import (
    BaseCameraModel,
    CameraModel,
    Distortion,
    FisheyeCameraModel,
    FisheyeDistortion,
    Intrinsics,
)


# OpenCV's float32 INTER_LINEAR path is not bit-exact across builds. Keep this
# tolerance scoped to float remap-vs-OpenCV checks; integer paths use tighter
# dtype-scale tolerances below.
REMAP_FLOAT_RTOL = 1e-5
REMAP_FLOAT_ATOL = 5e-3
REMAP_NATIVE_FLOAT_RTOL = 1e-6
REMAP_NATIVE_FLOAT_ATOL = 2e-7


def _cpu_remap_selection() -> mock.Mock:
    return mock.Mock(
        native=True,
        candidate=mock.Mock(kernel_name="camera_model_remap_cpu"),
    )


def _numpy_remap_selection() -> mock.Mock:
    return mock.Mock(native=False, candidate=None)


def _projection_models(
    width: int,
    height: int,
) -> tuple[dict[str, BaseCameraModel], dict[str, BaseCameraModel]]:
    src_intrinsics = Intrinsics(
        focal_length_mm=8.7,
        sensor_width_mm=12.0,
        sensor_height_mm=9.0,
        image_width_px=width,
        image_height_px=height,
        cx_px=width / 2.0 - 0.3,
        cy_px=height / 2.0 + 0.2,
    )
    dst_intrinsics = Intrinsics(
        focal_length_mm=8.1,
        sensor_width_mm=11.5,
        sensor_height_mm=8.7,
        image_width_px=width,
        image_height_px=height,
        cx_px=width / 2.0 + 0.25,
        cy_px=height / 2.0 - 0.15,
    )
    src_models: dict[str, BaseCameraModel] = {
        "perspective": CameraModel(
            src_intrinsics,
            Distortion(k1=-0.018, k2=0.002, p1=0.0004, p2=-0.0003),
        ),
        "fisheye": FisheyeCameraModel(
            src_intrinsics,
            FisheyeDistortion(k1=0.011, k2=-0.002, k3=0.0002),
        ),
    }
    dst_models: dict[str, BaseCameraModel] = {
        "perspective": CameraModel(
            dst_intrinsics,
            Distortion(k1=-0.014, k2=0.0015, p1=-0.0003, p2=0.0002),
        ),
        "fisheye": FisheyeCameraModel(
            dst_intrinsics,
            FisheyeDistortion(k1=-0.008, k2=0.0014, k3=-0.0001),
        ),
    }
    return src_models, dst_models


def _projection_remap_kwargs(
    image: np.ndarray,
    src_camera: BaseCameraModel,
    dst_camera: BaseCameraModel,
    rotation: np.ndarray,
) -> dict[str, object]:
    height, width = image.shape[:2]
    return {
        "image": image,
        "out_height": height,
        "out_width": width,
        "fx_src": float(src_camera.K[0, 0]),
        "fy_src": float(src_camera.K[1, 1]),
        "cx_src": float(src_camera.K[0, 2]),
        "cy_src": float(src_camera.K[1, 2]),
        "fx_dst": float(dst_camera.K[0, 0]),
        "fy_dst": float(dst_camera.K[1, 1]),
        "cx_dst": float(dst_camera.K[0, 2]),
        "cy_dst": float(dst_camera.K[1, 2]),
        "rotation_dst_to_src": rotation,
        "src_dist_coeffs": src_camera.remap_dist_coeffs,
        "dst_dist_coeffs": dst_camera.remap_dist_coeffs,
        "src_projection": src_camera.remap_projection,
        "dst_projection": dst_camera.remap_projection,
    }


def _wide_fov_mixed_remap_kwargs(image: np.ndarray) -> dict[str, object]:
    height, width = image.shape[:2]
    angle = np.deg2rad(110.0)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ], dtype=np.float64)
    return {
        "image": image,
        "out_height": height,
        "out_width": width,
        "fx_src": 5.0,
        "fy_src": 5.0,
        "cx_src": (width - 1) / 2.0,
        "cy_src": (height - 1) / 2.0,
        "fx_dst": 18.0,
        "fy_dst": 18.0,
        "cx_dst": (width - 1) / 2.0,
        "cy_dst": (height - 1) / 2.0,
        "rotation_dst_to_src": rotation,
        "src_dist_coeffs": np.array(
            [0.008, -0.001, 0.0001, 0.0], dtype=np.float64),
        "dst_dist_coeffs": None,
        "src_projection": "fisheye",
        "dst_projection": "perspective",
    }


class TestCameraModelRemapCustomOp(unittest.TestCase):
    def test_project_image_from_camera_defaults_to_half_scale_coordinate_map(self) -> None:
        intrinsics = Intrinsics(
            focal_length_mm=20.0,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            image_width_px=12,
            image_height_px=8,
        )
        camera = CameraModel(intrinsics=intrinsics)
        image = np.arange(8 * 12, dtype=np.uint16).reshape(8, 12)

        # Isolate the generic fallback: supported built-in camera pairs prefer
        # the exact fused path even when the fallback map scale is below 1.
        with mock.patch.object(
                CameraModel,
                "_project_image_from_camera_fused",
                return_value=None):
            approximate = camera.project_image_from_camera(
                camera, image, (12, 8))
            exact = camera.project_image_from_camera(camera, image, (12, 8),
                                                     map_scale=1.0)

        np.testing.assert_array_equal(approximate, exact)

    def test_project_image_from_camera_rejects_invalid_map_scale(self) -> None:
        intrinsics = Intrinsics(20.0, 36.0, 24.0, 12, 8)
        camera = CameraModel(intrinsics=intrinsics)
        with self.assertRaisesRegex(ValueError, "map_scale"):
            camera.project_image_from_camera(camera,
                                             np.zeros((8, 12), dtype=np.uint8),
                                             (12, 8), map_scale=0.0)

    def tearDown(self) -> None:
        remap_ops._load_compiled_module_result.cache_clear()
        remap_ops._select_camera_model_remap_backend.cache_clear()

    def test_camera_model_remap_matches_numpy(self) -> None:
        image = np.linspace(0.0, 1.0, num=4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
        pitch = np.deg2rad(1.5)
        rotation = np.array([
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ], dtype=np.float32)

        got = camera_model_remap(
            image=image,
            out_height=3,
            out_width=4,
            fx_src=9.0,
            fy_src=8.5,
            cx_src=2.0,
            cy_src=1.5,
            fx_dst=8.0,
            fy_dst=7.5,
            cx_dst=1.5,
            cy_dst=1.0,
            rotation_dst_to_src=rotation,
        )
        expected = remap_ops.camera_model_remap_numpy(
            image=image,
            out_height=3,
            out_width=4,
            fx_src=9.0,
            fy_src=8.5,
            cx_src=2.0,
            cy_src=1.5,
            fx_dst=8.0,
            fy_dst=7.5,
            cx_dst=1.5,
            cy_dst=1.0,
            rotation_dst_to_src=rotation,
        )

        np.testing.assert_allclose(
            got, expected, rtol=REMAP_FLOAT_RTOL, atol=REMAP_FLOAT_ATOL)

    def test_camera_model_remap_numpy_preserves_singleton_channel(self) -> None:
        image = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5, 1)
        got = remap_ops.camera_model_remap_numpy(
            image=image,
            out_height=4,
            out_width=5,
            fx_src=8.0,
            fy_src=8.0,
            cx_src=2.0,
            cy_src=1.5,
            fx_dst=8.0,
            fy_dst=8.0,
            cx_dst=2.0,
            cy_dst=1.5,
            rotation_dst_to_src=np.eye(3, dtype=np.float64),
            src_projection="fisheye",
            dst_projection="fisheye",
        )

        self.assertEqual(got.shape, image.shape)
        np.testing.assert_array_equal(got, image)

    def test_camera_model_remap_distortion_matches_numpy(self) -> None:
        image = np.linspace(0.0, 1.0, num=7 * 8 * 3, dtype=np.float32).reshape(7, 8, 3)
        yaw = np.deg2rad(0.8)
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        src_dist = np.array([0.015, -0.002, 0.001, -0.0005, 0.0001],
                            dtype=np.float32)
        dst_dist = np.array([-0.012, 0.0015, -0.0007, 0.0004, -0.00008],
                            dtype=np.float32)

        got = camera_model_remap(
            image=image,
            out_height=6,
            out_width=7,
            fx_src=12.0,
            fy_src=11.5,
            cx_src=3.5,
            cy_src=3.0,
            fx_dst=10.5,
            fy_dst=10.0,
            cx_dst=3.0,
            cy_dst=2.5,
            rotation_dst_to_src=rotation,
            src_dist_coeffs=src_dist,
            dst_dist_coeffs=dst_dist,
        )
        expected = remap_ops.camera_model_remap_numpy(
            image=image,
            out_height=6,
            out_width=7,
            fx_src=12.0,
            fy_src=11.5,
            cx_src=3.5,
            cy_src=3.0,
            fx_dst=10.5,
            fy_dst=10.0,
            cx_dst=3.0,
            cy_dst=2.5,
            rotation_dst_to_src=rotation,
            src_dist_coeffs=src_dist,
            dst_dist_coeffs=dst_dist,
        )

        np.testing.assert_allclose(
            got, expected, rtol=REMAP_FLOAT_RTOL, atol=REMAP_FLOAT_ATOL)

    def test_camera_model_remap_distortion_compiled_uint16_matches_numpy(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        image = (np.arange(7 * 8 * 2, dtype=np.uint16).reshape(7, 8, 2) * 17)
        rotation = np.array([
            [0.9999, -0.0100, 0.0020],
            [0.0100, 0.9999, -0.0015],
            [-0.0020, 0.0015, 1.0000],
        ], dtype=np.float32)
        src_dist = np.array([0.012, -0.0015, 0.0008, -0.0004, 0.0001],
                            dtype=np.float32)
        dst_dist = np.array([-0.010, 0.0012, -0.0006, 0.0003, -0.00008],
                            dtype=np.float32)

        kwargs = {
            "image": image,
            "out_height": 6,
            "out_width": 7,
            "fx_src": 13.0,
            "fy_src": 12.5,
            "cx_src": 3.5,
            "cy_src": 3.0,
            "fx_dst": 11.5,
            "fy_dst": 11.0,
            "cx_dst": 3.0,
            "cy_dst": 2.5,
            "rotation_dst_to_src": rotation,
            "src_dist_coeffs": src_dist,
            "dst_dist_coeffs": dst_dist,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        try:
            got = remap_ops.camera_model_remap_compiled(**kwargs)
        except RuntimeError as exc:
            if is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA runtime unavailable: {exc}")
            raise

        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_camera_model_remap_cpu_compiled_uint16_matches_numpy(self) -> None:
        # C3 is a production channel count; see the dispatch test for why C2/C5
        # cannot be compared against cv2 across architectures.
        image = (np.arange(8 * 9 * 3, dtype=np.uint16).reshape(8, 9, 3) * 19)
        rotation = np.array([
            [0.9998, -0.0170, 0.0030],
            [0.0170, 0.9998, -0.0020],
            [-0.0030, 0.0020, 1.0000],
        ], dtype=np.float32)
        src_dist = np.array([0.012, -0.0015, 0.0008, -0.0004, 0.0001],
                            dtype=np.float32)
        dst_dist = np.array([-0.010, 0.0012, -0.0006, 0.0003, -0.00008],
                            dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 7,
            "out_width": 8,
            "fx_src": 13.0,
            "fy_src": 12.5,
            "cx_src": 4.0,
            "cy_src": 3.5,
            "fx_dst": 11.5,
            "fy_dst": 11.0,
            "cx_dst": 3.5,
            "cy_dst": 3.0,
            "rotation_dst_to_src": rotation,
            "src_dist_coeffs": src_dist,
            "dst_dist_coeffs": dst_dist,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        got = remap_ops.camera_model_remap_cpu_compiled(**kwargs)

        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_camera_model_remap_cpu_matches_opencv5_channel_dispatch(self) -> None:
        # OpenCV 5's exact path handles C1/C3/C4 on every architecture; its 1/32
        # table fallback for other channel counts is x86_64-only (arm64 is exact
        # throughout). These fractional samples distinguish the two paths.
        cases = (
            (np.uint8, 255, 0.1, 3, 2),
            (np.uint16, 65535, 0.02, 26, 64),
        )
        for dtype, peak, offset, exact_value, table_value in cases:
            base = np.array([[0, 0], [0, peak]], dtype=dtype)
            kwargs = {
                "out_height": 1,
                "out_width": 1,
                "fx_src": 1.0,
                "fy_src": 1.0,
                "cx_src": offset,
                "cy_src": offset,
                "fx_dst": 1.0,
                "fy_dst": 1.0,
                "cx_dst": 0.0,
                "cy_dst": 0.0,
                "rotation_dst_to_src": np.eye(3, dtype=np.float64),
            }

            for channels in (1, 2, 3, 4, 5):
                with self.subTest(dtype=dtype.__name__, channels=channels):
                    image = (
                        base
                        if channels == 1
                        else np.repeat(base[:, :, None], channels, axis=2)
                    )
                    got = remap_ops.camera_model_remap_cpu_compiled(image=image, **kwargs)
                    if channels in {1, 3, 4}:
                        expected = remap_ops.camera_model_remap_numpy(
                            image=image, **kwargs)
                        np.testing.assert_array_equal(
                            expected, np.full_like(expected, exact_value))
                        np.testing.assert_array_equal(got, expected)
                    else:
                        # cv2 is not comparable here, so pin the kernel's own contract.
                        np.testing.assert_array_equal(
                            got, np.full_like(got, table_value))

    def test_camera_model_remap_cuda_matches_opencv5_channel_dispatch(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        cases = ((np.uint8, 255, 0.1), (np.uint16, 65535, 0.02))
        for dtype, peak, offset in cases:
            base = np.array([[0, 0], [0, peak]], dtype=dtype)
            kwargs = {
                "out_height": 1,
                "out_width": 1,
                "fx_src": 1.0,
                "fy_src": 1.0,
                "cx_src": offset,
                "cy_src": offset,
                "fx_dst": 1.0,
                "fy_dst": 1.0,
                "cx_dst": 0.0,
                "cy_dst": 0.0,
                "rotation_dst_to_src": np.eye(3, dtype=np.float64),
            }

            for channels in (1, 2, 3, 4, 5):
                with self.subTest(dtype=dtype.__name__, channels=channels):
                    image = (
                        base
                        if channels == 1
                        else np.repeat(base[:, :, None], channels, axis=2)
                    )
                    expected = remap_ops.camera_model_remap_numpy(image=image, **kwargs)
                    try:
                        got = remap_ops.camera_model_remap_compiled(image=image, **kwargs)
                    except RuntimeError as exc:
                        if is_cuda_runtime_unavailable_error(exc):
                            self.skipTest(f"CUDA runtime unavailable: {exc}")
                        raise

                    np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_cpu_compiled_uint8_half_pixel_rounds_to_even(
            self) -> None:
        # Half-pixel ties are implementation-defined in OpenCV. This test fixes
        # the CPU kernel's contract and keeps it aligned with the CUDA path.
        image = np.array([[0, 1]], dtype=np.uint8)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 1,
            "out_width": 1,
            "fx_src": 1.0,
            "fy_src": 1.0,
            "cx_src": 0.5,
            "cy_src": 0.0,
            "fx_dst": 1.0,
            "fy_dst": 1.0,
            "cx_dst": 0.0,
            "cy_dst": 0.0,
            "rotation_dst_to_src": rotation,
        }
        got = remap_ops.camera_model_remap_cpu_compiled(**kwargs)

        self.assertEqual(int(got[0, 0]), 0)

    def test_camera_model_remap_cpu_compiled_uint16_half_pixel_rounds_to_even(
            self) -> None:
        # Keep integer tie handling consistent across uint8/uint16 CPU kernels.
        image = np.array([[0, 1]], dtype=np.uint16)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 1,
            "out_width": 1,
            "fx_src": 1.0,
            "fy_src": 1.0,
            "cx_src": 0.5,
            "cy_src": 0.0,
            "fx_dst": 1.0,
            "fy_dst": 1.0,
            "cx_dst": 0.0,
            "cy_dst": 0.0,
            "rotation_dst_to_src": rotation,
        }
        got = remap_ops.camera_model_remap_cpu_compiled(**kwargs)

        self.assertEqual(int(got[0, 0]), 0)

    def test_camera_model_remap_cpu_compiled_float32_matches_numpy(self) -> None:
        image = np.linspace(0.0, 1.0, num=7 * 8 * 3, dtype=np.float32).reshape(7, 8, 3)
        rotation = np.array([
            [0.9999, -0.0100, 0.0020],
            [0.0100, 0.9999, -0.0015],
            [-0.0020, 0.0015, 1.0000],
        ], dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 6,
            "out_width": 7,
            "fx_src": 12.0,
            "fy_src": 11.5,
            "cx_src": 3.5,
            "cy_src": 3.0,
            "fx_dst": 10.5,
            "fy_dst": 10.0,
            "cx_dst": 3.0,
            "cy_dst": 2.5,
            "rotation_dst_to_src": rotation,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        got = remap_ops.camera_model_remap_cpu_compiled(**kwargs)

        np.testing.assert_allclose(
            got, expected, rtol=REMAP_FLOAT_RTOL, atol=REMAP_FLOAT_ATOL)

    def test_camera_model_remap_numpy_projection_pairs_match_camera_models(
            self) -> None:
        height, width = 17, 19
        image = np.linspace(
            0.0, 1.0, num=height * width * 3, dtype=np.float32
        ).reshape(height, width, 3)
        angle = np.deg2rad(1.7)
        rotation = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ], dtype=np.float64)
        src_models, dst_models = _projection_models(width, height)

        for src_name, src_camera in src_models.items():
            for dst_name, dst_camera in dst_models.items():
                with self.subTest(src=src_name, dst=dst_name):
                    got = remap_ops.camera_model_remap_numpy(
                        **_projection_remap_kwargs(
                            image, src_camera, dst_camera, rotation))
                    expected = dst_camera.project_image_from_camera(
                        src_camera,
                        image,
                        (width, height),
                        roi=(0, 0, width, height),
                        rotation_dst_to_src=rotation,
                    )
                    np.testing.assert_allclose(
                        got,
                        expected,
                        rtol=REMAP_NATIVE_FLOAT_RTOL,
                        atol=REMAP_NATIVE_FLOAT_ATOL,
                    )

    def test_camera_model_remap_cpu_projection_pairs_match_numpy(self) -> None:
        height, width = 17, 19
        image = np.linspace(
            0.0, 1.0, num=height * width * 3, dtype=np.float32
        ).reshape(height, width, 3)
        angle = np.deg2rad(-1.3)
        rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ], dtype=np.float32)
        src_models, dst_models = _projection_models(width, height)

        for src_name, src_camera in src_models.items():
            for dst_name, dst_camera in dst_models.items():
                with self.subTest(src=src_name, dst=dst_name):
                    kwargs = _projection_remap_kwargs(
                        image, src_camera, dst_camera, rotation)
                    expected = remap_ops.camera_model_remap_numpy(**kwargs)
                    got = remap_ops.camera_model_remap_cpu_compiled(**kwargs)
                    np.testing.assert_allclose(
                        got,
                        expected,
                        rtol=REMAP_NATIVE_FLOAT_RTOL,
                        atol=REMAP_NATIVE_FLOAT_ATOL,
                    )

    def test_camera_model_remap_cpu_projection_pairs_integer_dtypes_match_numpy(
            self) -> None:
        height, width = 19, 23
        rotation = np.eye(3, dtype=np.float64)
        src_models, dst_models = _projection_models(width, height)

        for dtype, high in ((np.uint8, 256), (np.uint16, 65536)):
            image = np.random.default_rng(31).integers(
                0, high, size=(height, width, 3), dtype=dtype)
            for src_name, src_camera in src_models.items():
                for dst_name, dst_camera in dst_models.items():
                    with self.subTest(
                            dtype=dtype.__name__, src=src_name, dst=dst_name):
                        kwargs = _projection_remap_kwargs(
                            image, src_camera, dst_camera, rotation)
                        expected = remap_ops.camera_model_remap_numpy(**kwargs)
                        got = remap_ops.camera_model_remap_cpu_compiled(**kwargs)
                        # Integer ties differ across OpenCV builds; native CPU
                        # and CUDA deliberately share ties-to-even semantics.
                        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_camera_model_remap_cpu_wide_fov_mixed_matches_numpy(self) -> None:
        image = np.linspace(
            0.0, 1.0, num=31 * 35 * 3, dtype=np.float32
        ).reshape(31, 35, 3)
        kwargs = _wide_fov_mixed_remap_kwargs(image)

        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        got = remap_ops.camera_model_remap_cpu_compiled(**kwargs)

        np.testing.assert_allclose(
            got,
            expected,
            rtol=REMAP_NATIVE_FLOAT_RTOL,
            atol=REMAP_NATIVE_FLOAT_ATOL,
        )

    def test_camera_model_remap_cuda_projection_pairs_match_numpy(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        height, width = 17, 19
        image = np.linspace(
            0.0, 1.0, num=height * width * 3, dtype=np.float32
        ).reshape(height, width, 3)
        rotation = np.eye(3, dtype=np.float32)
        src_models, dst_models = _projection_models(width, height)

        for src_name, src_camera in src_models.items():
            for dst_name, dst_camera in dst_models.items():
                with self.subTest(src=src_name, dst=dst_name):
                    kwargs = _projection_remap_kwargs(
                        image, src_camera, dst_camera, rotation)
                    expected = remap_ops.camera_model_remap_numpy(**kwargs)
                    try:
                        got = remap_ops.camera_model_remap_compiled(**kwargs)
                    except RuntimeError as exc:
                        if is_cuda_runtime_unavailable_error(exc):
                            self.skipTest(f"CUDA runtime unavailable: {exc}")
                        raise
                    np.testing.assert_allclose(
                        got,
                        expected,
                        rtol=REMAP_NATIVE_FLOAT_RTOL,
                        atol=REMAP_NATIVE_FLOAT_ATOL,
                    )

    def test_camera_model_remap_cuda_shared_staging_handles_larger_output(
            self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        image = np.arange(7 * 9 * 3, dtype=np.uint16).reshape(7, 9, 3)
        kwargs = {
            "image": image,
            "out_height": 11,
            "out_width": 13,
            "fx_src": 12.0,
            "fy_src": 12.0,
            "cx_src": 4.0,
            "cy_src": 3.0,
            "fx_dst": 16.0,
            "fy_dst": 16.0,
            "cx_dst": 6.0,
            "cy_dst": 5.0,
            "rotation_dst_to_src": np.eye(3, dtype=np.float64),
            "src_dist_coeffs": np.array(
                [0.01, -0.001, 0.0005, -0.0002, 0.0001],
                dtype=np.float64,
            ),
            "dst_dist_coeffs": np.array(
                [-0.008, 0.0008, -0.0003, 0.0001, -0.00005],
                dtype=np.float64,
            ),
        }
        expected = remap_ops.camera_model_remap_cpu_compiled(**kwargs)
        try:
            got = remap_ops.camera_model_remap_compiled(**kwargs)
        except RuntimeError as exc:
            if is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA runtime unavailable: {exc}")
            raise

        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_cuda_projection_pairs_integer_dtypes_match_cpu(
            self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        height, width = 19, 23
        rotation = np.eye(3, dtype=np.float64)
        src_models, dst_models = _projection_models(width, height)

        for dtype, high in ((np.uint8, 256), (np.uint16, 65536)):
            image = np.random.default_rng(43).integers(
                0, high, size=(height, width, 3), dtype=dtype)
            for src_name, src_camera in src_models.items():
                for dst_name, dst_camera in dst_models.items():
                    with self.subTest(
                            dtype=dtype.__name__, src=src_name, dst=dst_name):
                        kwargs = _projection_remap_kwargs(
                            image, src_camera, dst_camera, rotation)
                        expected = remap_ops.camera_model_remap_cpu_compiled(
                            **kwargs)
                        try:
                            got = remap_ops.camera_model_remap_compiled(**kwargs)
                        except RuntimeError as exc:
                            if is_cuda_runtime_unavailable_error(exc):
                                self.skipTest(f"CUDA runtime unavailable: {exc}")
                            raise
                        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_cuda_wide_fov_mixed_matches_numpy(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        image = np.linspace(
            0.0, 1.0, num=31 * 35 * 3, dtype=np.float32
        ).reshape(31, 35, 3)
        kwargs = _wide_fov_mixed_remap_kwargs(image)
        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        try:
            got = remap_ops.camera_model_remap_compiled(**kwargs)
        except RuntimeError as exc:
            if is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA runtime unavailable: {exc}")
            raise

        np.testing.assert_allclose(
            got,
            expected,
            rtol=REMAP_NATIVE_FLOAT_RTOL,
            atol=REMAP_NATIVE_FLOAT_ATOL,
        )

    def test_camera_model_remap_compiled_zero_distortion_matches_numpy(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        image = np.linspace(0.0, 1.0, num=6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3)
        rotation = np.array([
            [0.9999, -0.0120, 0.0020],
            [0.0120, 0.9999, -0.0010],
            [-0.0020, 0.0010, 1.0000],
        ], dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 5,
            "out_width": 6,
            "fx_src": 11.0,
            "fy_src": 10.5,
            "cx_src": 3.0,
            "cy_src": 2.5,
            "fx_dst": 9.5,
            "fy_dst": 9.0,
            "cx_dst": 2.5,
            "cy_dst": 2.0,
            "rotation_dst_to_src": rotation,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        try:
            got = remap_ops.camera_model_remap_compiled(**kwargs)
        except RuntimeError as exc:
            if is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA runtime unavailable: {exc}")
            raise

        np.testing.assert_allclose(
            got, expected, rtol=REMAP_FLOAT_RTOL, atol=REMAP_FLOAT_ATOL)

    def test_camera_model_remap_distortion_compiled_float32_rgb_matches_numpy(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        image = np.linspace(0.0, 1.0, num=9 * 10 * 3, dtype=np.float32).reshape(9, 10, 3)
        rotation = np.array([
            [0.9998, -0.0180, 0.0030],
            [0.0180, 0.9998, -0.0020],
            [-0.0030, 0.0020, 1.0000],
        ], dtype=np.float32)
        src_dist = np.array([0.010, -0.0012, 0.0006, -0.0003, 0.00008],
                            dtype=np.float32)
        dst_dist = np.array([-0.011, 0.0014, -0.0007, 0.0004, -0.00007],
                            dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 8,
            "out_width": 9,
            "fx_src": 15.0,
            "fy_src": 14.5,
            "cx_src": 4.5,
            "cy_src": 4.0,
            "fx_dst": 13.5,
            "fy_dst": 13.0,
            "cx_dst": 4.0,
            "cy_dst": 3.5,
            "rotation_dst_to_src": rotation,
            "src_dist_coeffs": src_dist,
            "dst_dist_coeffs": dst_dist,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        try:
            got = remap_ops.camera_model_remap_compiled(**kwargs)
        except RuntimeError as exc:
            if is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA runtime unavailable: {exc}")
            raise

        np.testing.assert_allclose(
            got, expected, rtol=REMAP_FLOAT_RTOL, atol=REMAP_FLOAT_ATOL)

    def test_camera_model_remap_can_force_numpy_fallback(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(remap_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                remap_ops._select_camera_model_remap_backend.cache_clear()
                got = camera_model_remap(
                    image=image,
                    out_height=2,
                    out_width=2,
                    fx_src=6.0,
                    fy_src=6.0,
                    cx_src=1.0,
                    cy_src=1.0,
                    fx_dst=5.0,
                    fy_dst=5.0,
                    cx_dst=1.0,
                    cy_dst=1.0,
                    rotation_dst_to_src=rotation,
                )

        expected = remap_ops.camera_model_remap_numpy(
            image=image,
            out_height=2,
            out_width=2,
            fx_src=6.0,
            fy_src=6.0,
            cx_src=1.0,
            cy_src=1.0,
            fx_dst=5.0,
            fy_dst=5.0,
            cx_dst=1.0,
            cy_dst=1.0,
            rotation_dst_to_src=rotation,
        )
        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_distortion_can_force_numpy_fallback(self) -> None:
        image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
        rotation = np.eye(3, dtype=np.float32)
        src_dist = np.array([0.012, -0.001, 0.0005, -0.0003],
                            dtype=np.float32)
        dst_dist = np.array([-0.01, 0.001, -0.0004, 0.0002, 0.0],
                            dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 5,
            "out_width": 6,
            "fx_src": 8.0,
            "fy_src": 8.0,
            "cx_src": 3.0,
            "cy_src": 2.5,
            "fx_dst": 8.0,
            "fy_dst": 8.0,
            "cx_dst": 3.0,
            "cy_dst": 2.5,
            "rotation_dst_to_src": rotation,
            "src_dist_coeffs": src_dist,
            "dst_dist_coeffs": dst_dist,
            "src_projection": "fisheye",
            "dst_projection": "perspective",
        }

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(remap_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                remap_ops._select_camera_model_remap_backend.cache_clear()
                got = camera_model_remap(**kwargs)

        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_float64_uses_float32_compiled_boundary(self) -> None:
        image = np.arange(18, dtype=np.float64).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
            "src_dist_coeffs": np.array(
                [0.01, -0.001, 0.0001, 0.0], dtype=np.float64),
            "src_projection": "fisheye",
            "dst_projection": "perspective",
        }
        compiled = mock.Mock(return_value=np.full((2, 2, 2), 0.25,
                                                   dtype=np.float32))

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("compiled", compiled)):
            got = camera_model_remap(**kwargs)

        compiled.assert_called_once()
        self.assertEqual(compiled.call_args.kwargs["image"].dtype, np.float32)
        self.assertEqual(got.dtype, np.float64)
        np.testing.assert_array_equal(got, np.full((2, 2, 2), 0.25,
                                                   dtype=np.float64))

    def test_camera_model_remap_rejects_nonfinite_distortion(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite"):
            camera_model_remap(
                image=image,
                out_height=2,
                out_width=2,
                fx_src=6.0,
                fy_src=6.0,
                cx_src=1.0,
                cy_src=1.0,
                fx_dst=5.0,
                fy_dst=5.0,
                cx_dst=1.0,
                cy_dst=1.0,
                rotation_dst_to_src=rotation,
                src_dist_coeffs=np.array([np.nan, 0.0, 0.0, 0.0, 0.0],
                                         dtype=np.float32),
            )

    def test_camera_model_remap_rejects_invalid_camera_params(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
        }
        with self.assertRaisesRegex(ValueError, "fx_src"):
            camera_model_remap(**{**kwargs, "fx_src": 0.0})
        with self.assertRaisesRegex(ValueError, "src_projection"):
            camera_model_remap(
                **{**kwargs, "src_projection": "equirectangular"})
        bad_rotation = rotation.copy()
        bad_rotation[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "rotation_dst_to_src"):
            camera_model_remap(**{**kwargs, "rotation_dst_to_src": bad_rotation})

    def test_camera_model_remap_falls_back_when_cuda_runtime_is_unavailable(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
            "src_dist_coeffs": np.array(
                [0.01, -0.001, 0.0001, 0.0], dtype=np.float64),
            "src_projection": "fisheye",
            "dst_projection": "perspective",
        }
        expected = np.full((2, 2, 2), 7, dtype=np.uint8)

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("cuda", mock.Mock(side_effect=RuntimeError(
                    "camera_model_remap cudaMalloc(image): no CUDA-capable device is detected")))):
            with mock.patch.object(
                    backend_registry,
                    "resolve_after_runtime_unavailable",
                    return_value=_cpu_remap_selection()):
                with mock.patch.object(
                        remap_ops,
                        "camera_model_remap_cpu_compiled",
                        return_value=expected) as cpu_fallback:
                    got = camera_model_remap(**kwargs)

        np.testing.assert_array_equal(got, expected)
        cpu_fallback.assert_called_once()
        self.assertEqual(
            cpu_fallback.call_args.kwargs["src_projection"], "fisheye")
        self.assertEqual(
            cpu_fallback.call_args.kwargs["dst_projection"], "perspective")

    def test_camera_model_remap_falls_back_to_numpy_when_cuda_and_cpu_unavailable(
            self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("cuda", mock.Mock(side_effect=RuntimeError(
                    "camera_model_remap cudaMalloc(image): no CUDA-capable device is detected")))):
            with mock.patch.object(
                    backend_registry,
                    "resolve_after_runtime_unavailable",
                    return_value=_numpy_remap_selection()):
                got = camera_model_remap(**kwargs)

        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_cuda_fallback_propagates_cpu_runtime_error(
            self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
        }

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("cuda", mock.Mock(side_effect=RuntimeError(
                    "camera_model_remap cudaMalloc(image): no CUDA-capable device is detected")))):
            with mock.patch.object(
                    backend_registry,
                    "resolve_after_runtime_unavailable",
                    return_value=_cpu_remap_selection()):
                with mock.patch.object(
                        remap_ops,
                        "camera_model_remap_cpu_compiled",
                        side_effect=RuntimeError("native CPU remap bug")):
                    with mock.patch.object(
                            remap_ops,
                            "camera_model_remap_numpy",
                            side_effect=AssertionError("numpy fallback should not be called")):
                        with self.assertRaisesRegex(RuntimeError, "native CPU remap bug"):
                            camera_model_remap(**kwargs)

    def test_camera_model_remap_propagates_unsupported_kernel_image(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
        }
        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("cuda", mock.Mock(side_effect=RuntimeError(
                    "camera_model_remap kernel launch: no kernel image is available for execution on the device")))):
            with mock.patch.object(
                    remap_ops,
                    "camera_model_remap_cpu_compiled",
                    side_effect=AssertionError("CPU fallback should not run")):
                with mock.patch.object(
                        remap_ops,
                        "camera_model_remap_numpy",
                        side_effect=AssertionError("numpy fallback should not run")):
                    with self.assertRaisesRegex(RuntimeError, "no kernel image"):
                        camera_model_remap(**kwargs)

    def test_camera_model_remap_propagates_untyped_cuda_allocation_failure(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
        }
        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("cuda", mock.Mock(side_effect=RuntimeError(
                    "camera_model_remap cudaMallocHost(image): out of memory")))):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                camera_model_remap(**kwargs)

    def test_camera_model_remap_typed_resource_error_falls_back_to_cpu(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": np.eye(3, dtype=np.float32),
        }
        expected = np.full((2, 2, 2), 9, dtype=np.uint8)

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("cuda", mock.Mock(side_effect=
                                                  CustomOpResourceExhaustedError(
                                                      "estimated VRAM is insufficient")))):
            with mock.patch.object(
                    backend_registry,
                    "resolve_after_resource_exhausted",
                    return_value=_cpu_remap_selection()) as resolve:
                with mock.patch.object(
                        remap_ops,
                        "camera_model_remap_cpu_compiled",
                        return_value=expected) as cpu_fallback:
                    got = camera_model_remap(**kwargs)

        np.testing.assert_array_equal(got, expected)
        resolve.assert_called_once()
        cpu_fallback.assert_called_once()

    def test_camera_model_remap_typed_resource_error_falls_back_to_numpy(
            self) -> None:
        image = np.arange(9, dtype=np.uint8).reshape(3, 3)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": np.eye(3, dtype=np.float32),
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("cuda", mock.Mock(side_effect=
                                                  CustomOpResourceExhaustedError(
                                                      "cudaMalloc output")))):
            with mock.patch.object(
                    backend_registry,
                    "resolve_after_resource_exhausted",
                    return_value=_numpy_remap_selection()):
                got = camera_model_remap(**kwargs)

        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_compiled_stops_before_kernel_when_denied(
            self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        native = mock.Mock()
        module = mock.Mock(camera_model_remap=native)
        admission = mock.MagicMock()
        admission.return_value.__enter__.return_value = mock.Mock(
            granted=False,
            estimated_peak_bytes=52,
        )

        with mock.patch.object(
                remap_ops,
                "_load_compiled_module_result",
                return_value=(module, None)):
            with mock.patch.object(
                    cuda_memory,
                    "cuda_memory_admission",
                    admission):
                with self.assertRaisesRegex(
                        CustomOpResourceExhaustedError,
                        "estimated peak 52 bytes"):
                    remap_ops.camera_model_remap_compiled(
                        image=image,
                        out_height=2,
                        out_width=2,
                        fx_src=6.0,
                        fy_src=6.0,
                        cx_src=1.0,
                        cy_src=1.0,
                        fx_dst=5.0,
                        fy_dst=5.0,
                        cx_dst=1.0,
                        cy_dst=1.0,
                        rotation_dst_to_src=np.eye(3, dtype=np.float32),
                    )

        native.assert_not_called()

    def test_project_image_from_camera_routes_zero_distortion_through_custom_fused(self) -> None:
        img = np.arange(16, dtype=np.uint8).reshape(4, 4)
        intrinsics = Intrinsics(
            focal_length_mm=8.0,
            sensor_width_mm=8.0,
            sensor_height_mm=8.0,
            image_width_px=4,
            image_height_px=4,
        )
        src_camera = CameraModel(intrinsics=intrinsics)
        dst_camera = CameraModel(intrinsics=intrinsics)

        expected = remap_ops.camera_model_remap_numpy(
            image=img,
            out_height=4,
            out_width=4,
            fx_src=float(src_camera.K[0, 0]),
            fy_src=float(src_camera.K[1, 1]),
            cx_src=float(src_camera.K[0, 2]),
            cy_src=float(src_camera.K[1, 2]),
            fx_dst=float(dst_camera.K[0, 0]),
            fy_dst=float(dst_camera.K[1, 1]),
            cx_dst=float(dst_camera.K[0, 2]),
            cy_dst=float(dst_camera.K[1, 2]),
            rotation_dst_to_src=np.eye(3, dtype=np.float32),
        )

        with mock.patch.object(
                norma_types,
                "custom_camera_model_remap",
                wraps=norma_types.custom_camera_model_remap) as patched_custom:
            got = dst_camera.project_image_from_camera(src_camera, img, (4, 4),
                                                       map_scale=1.0)

        patched_custom.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_project_image_from_camera_routes_distortion_through_custom_fused(self) -> None:
        img = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
        intrinsics = Intrinsics(
            focal_length_mm=8.0,
            sensor_width_mm=8.0,
            sensor_height_mm=8.0,
            image_width_px=6,
            image_height_px=5,
        )
        src_camera = CameraModel(
            intrinsics=intrinsics,
            distortion=Distortion(k1=0.012, k2=-0.001, p1=0.0005, p2=-0.0003),
        )
        dst_camera = CameraModel(
            intrinsics=intrinsics,
            distortion=Distortion(k1=-0.01, k2=0.001, p1=-0.0004, p2=0.0002),
        )

        expected = remap_ops.camera_model_remap_numpy(
            image=img,
            out_height=5,
            out_width=6,
            fx_src=float(src_camera.K[0, 0]),
            fy_src=float(src_camera.K[1, 1]),
            cx_src=float(src_camera.K[0, 2]),
            cy_src=float(src_camera.K[1, 2]),
            fx_dst=float(dst_camera.K[0, 0]),
            fy_dst=float(dst_camera.K[1, 1]),
            cx_dst=float(dst_camera.K[0, 2]),
            cy_dst=float(dst_camera.K[1, 2]),
            rotation_dst_to_src=np.eye(3, dtype=np.float32),
            src_dist_coeffs=src_camera.dist_coeffs,
            dst_dist_coeffs=dst_camera.dist_coeffs,
        )

        with mock.patch.object(
                norma_types,
                "custom_camera_model_remap",
                wraps=norma_types.custom_camera_model_remap) as patched_custom:
            got = dst_camera.project_image_from_camera(src_camera, img, (6, 5),
                                                       map_scale=1.0)

        patched_custom.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_project_image_from_camera_distortion_matches_generic_path(self) -> None:
        img = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
        intrinsics = Intrinsics(
            focal_length_mm=8.0,
            sensor_width_mm=8.0,
            sensor_height_mm=8.0,
            image_width_px=6,
            image_height_px=5,
        )
        src_camera = CameraModel(
            intrinsics=intrinsics,
            distortion=Distortion(k1=0.012, k2=-0.001, p1=0.0005, p2=-0.0003),
        )
        dst_camera = CameraModel(
            intrinsics=intrinsics,
            distortion=Distortion(k1=-0.01, k2=0.001, p1=-0.0004, p2=0.0002),
        )

        got = dst_camera.project_image_from_camera(src_camera, img, (6, 5),
                                                   map_scale=1.0)
        expected = dst_camera.project_image_from_camera(
            src_camera,
            img,
            (6, 5),
            roi=(0, 0, 6, 5),
        )

        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_project_image_from_camera_routes_all_projection_pairs_through_fused(
            self) -> None:
        height, width = 7, 9
        image = np.arange(height * width, dtype=np.uint8).reshape(height, width)
        expected = np.full_like(image, 23)
        src_models, dst_models = _projection_models(width, height)

        for src_name, src_camera in src_models.items():
            for dst_name, dst_camera in dst_models.items():
                with self.subTest(src=src_name, dst=dst_name):
                    with mock.patch.object(
                            norma_types,
                            "custom_camera_model_remap",
                            return_value=expected) as custom:
                        got = dst_camera.project_image_from_camera(
                            src_camera, image, (width, height))

                    np.testing.assert_array_equal(got, expected)
                    custom.assert_called_once()
                    call_kwargs = custom.call_args.kwargs
                    self.assertEqual(call_kwargs["src_projection"], src_name)
                    self.assertEqual(call_kwargs["dst_projection"], dst_name)

    def test_project_image_from_camera_keeps_generic_subclass_compatible(
            self) -> None:
        class GenericOnlyCameraModel(CameraModel):

            @property
            def remap_projection(self) -> None:
                return None

        height, width = 7, 9
        image = np.arange(height * width, dtype=np.uint8).reshape(height, width)
        intrinsics = Intrinsics(
            focal_length_mm=8.0,
            sensor_width_mm=12.0,
            sensor_height_mm=9.0,
            image_width_px=width,
            image_height_px=height,
        )
        camera = GenericOnlyCameraModel(intrinsics)

        with mock.patch.object(
                norma_types, "custom_camera_model_remap") as custom:
            got = camera.project_image_from_camera(
                camera, image, (width, height))

        custom.assert_not_called()
        np.testing.assert_array_equal(got, image)

    def test_project_image_from_camera_projection_pairs_match_generic_path(
            self) -> None:
        height, width = 17, 19
        image = np.linspace(
            0.0, 1.0, num=height * width, dtype=np.float32
        ).reshape(height, width)
        rotation = np.eye(3, dtype=np.float64)
        src_models, dst_models = _projection_models(width, height)

        for src_name, src_camera in src_models.items():
            for dst_name, dst_camera in dst_models.items():
                with self.subTest(src=src_name, dst=dst_name):
                    got = dst_camera.project_image_from_camera(
                        src_camera,
                        image,
                        (width, height),
                        rotation_dst_to_src=rotation,
                        map_scale=1.0,
                    )
                    expected = dst_camera.project_image_from_camera(
                        src_camera,
                        image,
                        (width, height),
                        roi=(0, 0, width, height),
                        rotation_dst_to_src=rotation,
                    )
                    np.testing.assert_allclose(
                        got,
                        expected,
                        rtol=REMAP_FLOAT_RTOL,
                        atol=REMAP_FLOAT_ATOL,
                    )
