from unittest import mock

import numpy as np

from hoshicore._custom_op import calibration_divide
from hoshicore._custom_op import calibration_subtract
import hoshicore._custom_op.ops.calibration as calibration_ops
import hoshicore.component.calibration as component_calibration


def teardown_function() -> None:
    calibration_ops._load_compiled_module_result.cache_clear()
    calibration_ops._select_calibration_subtract_backend.cache_clear()
    calibration_ops._select_calibration_divide_backend.cache_clear()


def test_calibration_subtract_compiled_matches_numpy_uint16() -> None:
    frame = np.array([[100, 5], [65535, 1000]], dtype=np.uint16)
    reference = np.array([[90, 10], [1, 2000]], dtype=np.uint16)

    got, got_dtype = calibration_subtract(frame, reference, frame.dtype, reference.dtype)
    expected, expected_dtype = calibration_ops.calibration_subtract_numpy(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )

    assert got_dtype == expected_dtype == np.dtype("uint16")
    np.testing.assert_array_equal(got, expected)


def test_calibration_subtract_direct_compiled_matches_numpy_uint16() -> None:
    frame = np.array([[100, 5], [65535, 1000]], dtype=np.uint16)
    reference = np.array([[90, 10], [1, 2000]], dtype=np.uint16)

    got, got_dtype = calibration_ops.calibration_subtract_compiled(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )
    expected, expected_dtype = calibration_ops.calibration_subtract_numpy(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )

    assert got_dtype == expected_dtype == np.dtype("uint16")
    np.testing.assert_array_equal(got, expected)


def test_calibration_subtract_aligns_mixed_uint_dtypes() -> None:
    frame = np.array([[1, 255]], dtype=np.uint8)
    reference = np.array([[257, 65535]], dtype=np.uint16)

    got, got_dtype = calibration_subtract(frame, reference, frame.dtype, reference.dtype)
    expected, expected_dtype = calibration_ops.calibration_subtract_numpy(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )

    assert got_dtype == expected_dtype == np.dtype("uint16")
    np.testing.assert_array_equal(got, expected)


def test_calibration_divide_compiled_matches_numpy_with_zero_reference() -> None:
    frame = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    reference = np.array([[100, 0], [200, 400]], dtype=np.uint16)

    got, got_dtype = calibration_divide(frame, reference, frame.dtype, reference.dtype)
    expected, expected_dtype = calibration_ops.calibration_divide_numpy(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )

    assert got_dtype == expected_dtype == np.dtype("uint16")
    np.testing.assert_array_equal(got, expected)


def test_calibration_divide_direct_compiled_matches_numpy_uint16() -> None:
    frame = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    reference = np.array([[100, 50], [200, 400]], dtype=np.uint16)

    got, got_dtype = calibration_ops.calibration_divide_compiled(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )
    expected, expected_dtype = calibration_ops.calibration_divide_numpy(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )

    assert got_dtype == expected_dtype == np.dtype("uint16")
    np.testing.assert_array_equal(got, expected)


def test_calibration_broadcast_shape_keeps_numpy_semantics() -> None:
    frame = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    reference = np.array([[10, 20]], dtype=np.uint16)

    got, got_dtype = calibration_subtract(frame, reference, frame.dtype, reference.dtype)
    expected, expected_dtype = calibration_ops.calibration_subtract_numpy(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )

    assert got_dtype == expected_dtype == np.dtype("uint16")
    np.testing.assert_array_equal(got, expected)


def test_calibration_subtract_compiled_matches_numpy_float32() -> None:
    frame = np.array([[1.5, 0.25], [10.0, 2.0]], dtype=np.float32)
    reference = np.array([[1.0, 0.5], [3.0, 5.0]], dtype=np.float32)

    got, got_dtype = calibration_subtract(frame, reference, frame.dtype, reference.dtype)
    expected, expected_dtype = calibration_ops.calibration_subtract_numpy(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )

    assert got_dtype == expected_dtype == np.dtype("float32")
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_calibration_can_force_numpy_fallback() -> None:
    frame = np.array([[100, 200]], dtype=np.uint16)
    reference = np.array([[90, 250]], dtype=np.uint16)

    with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
        calibration_ops._select_calibration_subtract_backend.cache_clear()
        got, got_dtype = calibration_subtract(frame, reference, frame.dtype, reference.dtype)

    expected, expected_dtype = calibration_ops.calibration_subtract_numpy(
        frame,
        reference,
        frame.dtype,
        reference.dtype,
    )
    assert got_dtype == expected_dtype
    np.testing.assert_array_equal(got, expected)


def test_component_calibration_routes_through_custom_op_wrapper() -> None:
    frame = np.array([[10, 20]], dtype=np.uint16)
    reference = np.array([[1, 2]], dtype=np.uint16)
    sentinel = np.array([[9, 18]], dtype=np.uint16)

    with mock.patch(
        "hoshicore._custom_op.ops.calibration.calibration_subtract",
        return_value=(sentinel, np.dtype("uint16")),
    ) as patched:
        got, got_dtype = component_calibration.calibration_subtract(
            frame,
            reference,
            frame.dtype,
            reference.dtype,
        )

    patched.assert_called_once_with(frame, reference, frame.dtype, reference.dtype)
    assert got_dtype == np.dtype("uint16")
    np.testing.assert_array_equal(got, sentinel)
