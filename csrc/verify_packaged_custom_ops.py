"""Build and execute a minimal frozen custom-op smoke application."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_ENTRY = PROJECT_ROOT / "hoshicore" / "_custom_op" / "package_smoke.py"
DEFAULT_WORK_DIR = PROJECT_ROOT / "build" / "native-package-smoke"
SMOKE_MARKER = "HNW_NATIVE_PACKAGE_SMOKE_OK"


def _clean_runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "CONDA_PREFIX",
        "CUDA_HOME",
        "CUDA_PATH",
        "INCLUDE",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "LIB",
        "LIBPATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ):
        env.pop(name, None)

    if os.name == "nt":
        system_root = env.get("SystemRoot", r"C:\Windows")
        env["PATH"] = os.pathsep.join(
            (
                str(Path(system_root) / "System32"),
                system_root,
                str(Path(system_root) / "System32" / "Wbem"),
            )
        )
    else:
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    return env


def _extension_patterns() -> tuple[str, ...]:
    if os.name == "nt":
        return ("_C*.pyd",)
    if platform.system() == "Darwin":
        return ("_C*.so", "_C*.dylib")
    return ("_C*.so",)


def verify_packaged_custom_ops(work_dir: Path) -> None:
    dist_dir = work_dir / "dist"
    build_dir = work_dir / "build"
    spec_dir = work_dir / "spec"
    shutil.rmtree(work_dir, ignore_errors=True)
    spec_dir.mkdir(parents=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onedir",
        "--name",
        "hnw_native_smoke",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(spec_dir),
        "--paths",
        str(PROJECT_ROOT),
        "--hidden-import",
        "hoshicore._custom_op._C",
        str(SMOKE_ENTRY),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    package_dir = dist_dir / "hnw_native_smoke"
    executable = package_dir / (
        "hnw_native_smoke.exe" if os.name == "nt" else "hnw_native_smoke"
    )
    extensions = [
        path
        for pattern in _extension_patterns()
        for path in package_dir.rglob(pattern)
    ]
    if not extensions:
        raise RuntimeError("PyInstaller output does not contain hoshicore._custom_op._C")

    result = subprocess.run(
        [str(executable)],
        cwd=package_dir,
        env=_clean_runtime_environment(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or SMOKE_MARKER not in result.stdout:
        raise RuntimeError(
            "frozen custom-op smoke failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    print(result.stdout.strip())
    print(f"packaged_extension={extensions[0]}")


def main() -> None:
    verify_packaged_custom_ops(DEFAULT_WORK_DIR)


if __name__ == "__main__":
    main()
