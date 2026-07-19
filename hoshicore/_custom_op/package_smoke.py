"""Minimal native-extension smoke entry point for frozen-package validation."""

from __future__ import annotations

import json

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
    return build


def main() -> None:
    build = run_native_package_smoke()
    print(f"{SMOKE_MARKER} {json.dumps(build, sort_keys=True)}")


if __name__ == "__main__":
    main()
