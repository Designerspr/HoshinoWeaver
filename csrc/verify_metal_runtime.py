"""Fail-fast Metal runtime and first-kernel smoke for macOS CI."""

from __future__ import annotations

import json
import math
import time

import numpy as np

from hoshicore._custom_op import _C
from hoshicore._custom_op import _metal
from hoshicore._custom_op.metal_memory import metal_memory_estimate
from hoshicore._custom_op.ops import star_shrink as star_shrink_ops


# Many threadgroups per kernel, still inside a hosted runner's working set.
_TIMED_SHAPE = (1152, 1536, 3)
_TIMED_PARAMS = (3, "CIRCLE", 1, 1.0, 5)
_TIMING_REPEATS = 3
# Megapixel-scale so fixed dispatch/sync overheads are amortized as in
# production, and both dimensions exceed the 73-tap large Gaussian so interior
# taps dominate. This timing informs the star_mask_dog priority decision.
_DOG_TIMED_SHAPE = (1024, 1280, 3)


def _assert_high_water(image: np.ndarray, logical_op: str = "star_shrink_process",
                       **extra: int) -> int:
    cache = dict(_metal.metal_host_io_cache_info())
    estimate = metal_memory_estimate(
        logical_op,
        height=image.shape[0],
        width=image.shape[1],
        channels=image.shape[2] if image.ndim == 3 else 1,
        dtype_bytes=image.dtype.itemsize,
        **extra,
    )
    if cache.get("last_logical_peak_bytes") != estimate.peak_device_bytes:
        raise RuntimeError(
            f"Metal workspace high-water does not match estimator for {logical_op}: "
            f"{cache.get('last_logical_peak_bytes')} != {estimate.peak_device_bytes}"
        )
    return estimate.peak_device_bytes


def _best_seconds(fn, *args) -> float:
    fn(*args)  # warmup: pipeline compilation, buffer pool, OpenMP thread spin-up
    best = None
    for _ in range(_TIMING_REPEATS):
        started = time.perf_counter()
        fn(*args)
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)
    return best


def _paired_timing() -> dict[str, object]:
    """Smoke-level Metal/OpenMP timing through the production wrappers.

    Hosted runners virtualize the GPU, so this only proves both paths run at
    scale and agree numerically; it cannot decide backend priority.
    """
    rng = np.random.default_rng(20260811)
    image = rng.integers(100, 50000, size=_TIMED_SHAPE, dtype=np.uint16)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[8:-8, 8:-8] = 1

    metal_result = star_shrink_ops.star_shrink_process_compiled_metal(
        image, mask, *_TIMED_PARAMS)
    cpu_result = star_shrink_ops.star_shrink_process_compiled(
        image, mask, *_TIMED_PARAMS)
    # A multi-threadgroup dispatch exercises grid indexing that the small
    # correctness cases above cannot reach.
    np.testing.assert_allclose(metal_result, cpu_result, rtol=0, atol=1)
    np.testing.assert_array_equal(metal_result[mask == 0], image[mask == 0])
    peak_bytes = _assert_high_water(image)

    metal_seconds = _best_seconds(
        star_shrink_ops.star_shrink_process_compiled_metal,
        image, mask, *_TIMED_PARAMS)
    cpu_seconds = _best_seconds(
        star_shrink_ops.star_shrink_process_compiled,
        image, mask, *_TIMED_PARAMS)
    return {
        "shape": list(_TIMED_SHAPE),
        "metal_seconds": round(metal_seconds, 4),
        "openmp_seconds": round(cpu_seconds, 4),
        "metal_speedup": round(cpu_seconds / metal_seconds, 3),
        "peak_bytes": peak_bytes,
        "note": "smoke-level only; virtualized runner, not a priority decision",
    }


def _dog_kernel_sizes(sigma_small: float, sigma_large: float) -> dict[str, int]:
    return {
        "small_kernel_size": 2 * max(1, math.ceil(3.0 * sigma_small)) + 1,
        "large_kernel_size": 2 * max(1, math.ceil(3.0 * sigma_large)) + 1,
    }


def _verify_star_mask_dog() -> dict[str, object]:
    """Compare the Metal DoG mask against the OpenMP kernel on fixed fixtures.

    Strict equality is a regression gate on these fixtures, not a mathematical
    guarantee: the mask is thresholded, so fp32 last-bit differences in the gray
    and blur stages are usually but not provably absorbed.
    """
    rng = np.random.default_rng(20260812)
    peak = 0
    checked = []
    cases = (
        ("u16_gray_open", (37, 41), np.uint16, 4096, 60000, (1.5, 12.0, 3.0, 3, 0)),
        ("u16_rgb_open", (33, 29, 3), np.uint16, 4096, 60000, (1.5, 12.0, 3.0, 3, 0)),
        # uint8 exercises the other gray kernel; dilate_ksize > 0 the other branch.
        ("u8_rgb_open_dilate", (31, 35, 3), np.uint8, 64, 255, (1.5, 8.0, 2.5, 3, 5)),
        ("u8_gray_dilate_only", (29, 31), np.uint8, 64, 255, (2.0, 9.0, 3.0, 0, 3)),
    )
    for name, shape, dtype, background, peak_value, params in cases:
        image = rng.integers(0, background, size=shape, dtype=dtype)
        image[8:12, 9:13] = peak_value  # a blob the DoG must survive thresholding
        actual = _metal.star_mask_dog_metal(image, *params)
        expected = _C.star_mask_dog_cpu(image, *params)
        np.testing.assert_array_equal(actual, expected)
        peak = _assert_high_water(
            image, "star_mask_dog", **_dog_kernel_sizes(params[0], params[1])
        )
        checked.append(name)
    return {"cases": checked, "logical_peak": peak}


def _dog_paired_timing() -> dict[str, object]:
    """Medium-size DoG gate: interior Gaussian path, quality bound, timing.

    Both dimensions exceed 2 * large_radius so most pixels take the interior
    path the small fixtures never reach. Mask equality is a bounded quality gate,
    not strict: gray/blur run in fp32 and the compilers may contract differently.
    """
    sigma_small, sigma_large = 1.5, 12.0
    params = (sigma_small, sigma_large, 3.0, 3, 0)
    large_radius = max(1, math.ceil(3.0 * sigma_large))
    rng = np.random.default_rng(20260813)
    image = rng.integers(200, 40000, size=_DOG_TIMED_SHAPE, dtype=np.uint16)
    image[64:70, 96:102] = 65535
    if min(_DOG_TIMED_SHAPE[0], _DOG_TIMED_SHAPE[1]) <= 2 * large_radius:
        raise RuntimeError("DoG timing shape must exceed the large Gaussian diameter")

    metal_mask = star_shrink_ops.star_mask_dog_compiled_metal(image, *params)
    cpu_mask = star_shrink_ops.star_mask_dog_compiled_cpu(image, *params)
    mismatched = int(np.count_nonzero(metal_mask != cpu_mask))
    total = int(metal_mask.size)
    if mismatched > max(1, total // 10000):
        raise RuntimeError(
            f"Metal DoG mask disagrees with OpenMP on {mismatched}/{total} pixels"
        )
    peak_bytes = _assert_high_water(
        image, "star_mask_dog", **_dog_kernel_sizes(sigma_small, sigma_large)
    )

    metal_seconds = _best_seconds(
        star_shrink_ops.star_mask_dog_compiled_metal, image, *params)
    cpu_seconds = _best_seconds(
        star_shrink_ops.star_mask_dog_compiled_cpu, image, *params)
    return {
        "shape": list(_DOG_TIMED_SHAPE),
        "metal_seconds": round(metal_seconds, 4),
        "openmp_seconds": round(cpu_seconds, 4),
        "metal_speedup": round(cpu_seconds / metal_seconds, 3),
        "mask_mismatched_pixels": mismatched,
        "mask_pixels": total,
        "peak_bytes": peak_bytes,
        "note": "smoke-level only; virtualized runner, not a priority decision",
    }


def main() -> None:
    info = dict(_metal.metal_device_info())
    if not info.get("available"):
        raise RuntimeError(f"Metal runtime unavailable: {info.get('reason', 'unknown')}")
    if not info.get("has_unified_memory"):
        raise RuntimeError("Metal CI gate requires an Apple unified-memory device")

    rng = np.random.default_rng(20260810)
    cases = (
        (
            "uint8_gray_circle",
            rng.integers(0, 256, size=(17, 19), dtype=np.uint8),
            (3, "CIRCLE", 1, 1.0, 5),
        ),
        (
            "uint16_rgb_rect_blended",
            rng.integers(100, 30000, size=(24, 28, 3), dtype=np.uint16),
            (5, "RECT", 2, 0.5, 7),
        ),
        (
            "uint16_rgb_cross_repeated",
            rng.integers(100, 50000, size=(15, 17, 3), dtype=np.uint16),
            (3, "CROSS", 3, 1.0 / 3.0, 9),
        ),
    )
    checked_cases = []
    last_image = None
    for case_name, image, params in cases:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[: image.shape[0] - 2, 2:] = 1
        actual = _metal.star_shrink_process_metal(image, mask, *params)
        expected = _C.star_shrink_process(image, mask, *params)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1)
        np.testing.assert_array_equal(actual[mask == 0], image[mask == 0])
        checked_cases.append(case_name)
        last_image = image

    if last_image is None:
        raise RuntimeError("Metal runtime gate did not execute any kernel cases")

    logical_peak = _assert_high_water(last_image)
    dog = _verify_star_mask_dog()
    dog["timing"] = _dog_paired_timing()
    timing = _paired_timing()
    print(
        "HNW_METAL_RUNTIME_OK "
        + json.dumps(
            {
                "device": info.get("name"),
                "registry_id": info.get("registry_id"),
                "working_set": info.get("recommended_max_working_set_bytes"),
                "logical_peak": logical_peak,
                "dog": dog,
                "cases": checked_cases,
                "timing": timing,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
