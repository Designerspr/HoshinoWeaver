"""Minimal native-extension smoke entry point for frozen-package validation."""

from __future__ import annotations

import json
import os
import platform

import numpy as np

from hoshicore._custom_op import _C


SMOKE_MARKER = "HNW_NATIVE_PACKAGE_SMOKE_OK"


def run_native_package_smoke() -> dict[str, object]:
    """Import the bundled extension and execute a direct native CPU kernel."""
    build = dict(_C.build_info())
    if not build.get("openmp"):
        raise RuntimeError("packaged _C was not built with OpenMP support")

    base = np.array([1, 4, 2], dtype=np.uint16)
    fresh = np.array([3, 2, 5], dtype=np.uint16)
    _C.max_combine(base, fresh)
    np.testing.assert_array_equal(base, np.array([3, 4, 5], dtype=np.uint16))

    require_metal = os.environ.get("HNW_REQUIRE_METAL_RUNTIME") == "1"
    if platform.system() == "Darwin" and require_metal:
        from hoshicore._custom_op import _metal

        metal_info = dict(_metal.metal_device_info())
        if not metal_info.get("available"):
            raise RuntimeError(
                "packaged Metal runtime is unavailable: "
                f"{metal_info.get('reason', 'unknown')}"
            )
        image = np.arange(9 * 11, dtype=np.uint8).reshape(9, 11)
        mask = np.zeros((9, 11), dtype=np.uint8)
        mask[2:8, 3:10] = 1
        actual = _metal.star_shrink_process_metal(
            image, mask, 3, "CIRCLE", 1, 1.0, 5
        )
        expected = _C.star_shrink_process(
            image, mask, 3, "CIRCLE", 1, 1.0, 5
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1)
        build["metal_device"] = metal_info.get("name", "unknown")
    return build


def main() -> None:
    build = run_native_package_smoke()
    print(f"{SMOKE_MARKER} {json.dumps(build, sort_keys=True)}")


if __name__ == "__main__":
    main()
