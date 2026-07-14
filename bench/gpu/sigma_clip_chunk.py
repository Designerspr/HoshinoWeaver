"""Benchmark sigma-clip fused chunk host-in/host-out backends."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from bench.common import run_benchmark
from hoshicore._custom_op import build_info as custom_op_build_info
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
import hoshicore._custom_op.ops.sigma_clip as sigma_clip_ops


SUITE_ID = "gpu.sigma_clip_chunk"
CASE_NAMES = [
    "numpy",
    "openmp_cpu",
    "cuda_host_io",
]
DEFAULT_CASES = [
    "openmp_cpu",
    "cuda_host_io",
]


def parse_cases(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_CASES)
    cases = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(cases) - set(CASE_NAMES))
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(unknown)}")
    return cases


def _make_stack(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.dtype(args.dtype)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("--dtype must be uint8 or uint16")
    plane_size = args.chunk_rows * args.width * args.channels
    rng = np.random.default_rng(args.seed)
    high = 220 if dtype == np.dtype("uint8") else 4096
    stack = rng.integers(
        64, high, size=(args.frames, plane_size), dtype=dtype)

    outliers = max(1, plane_size // 4096)
    outlier_cols = rng.choice(plane_size, size=outliers, replace=False)
    stack[0, outlier_cols] = np.iinfo(dtype).max

    mask = (rng.random((args.frames, plane_size)) > args.mask_drop_rate).astype(
        np.uint8)
    if args.channels >= 3:
        spatial = plane_size // args.channels
        zero_pixels = max(1, spatial // 2048)
        zero_cols = rng.choice(spatial, size=zero_pixels, replace=False)
        for px in zero_cols:
            start = int(px) * args.channels
            stack[:, start:start + min(3, args.channels)] = 0
    return stack, mask


def _case_fn(
    case_name: str,
    stack: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> Callable[[], None]:
    fn = {
        "numpy": sigma_clip_ops.sigma_clip_fused_chunk_numpy,
        "openmp_cpu": sigma_clip_ops.sigma_clip_fused_chunk_compiled,
        "cuda_host_io": sigma_clip_ops.sigma_clip_fused_chunk_compiled_cuda,
    }[case_name]

    def run_once() -> None:
        _ = fn(
            stack,
            args.rej_high,
            args.rej_low,
            args.max_iter,
            mask=mask,
            skip_zero_rgb=args.skip_zero_rgb,
            channels=args.channels,
        )

    return run_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--width", type=int, default=6000)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--dtype", choices=["uint8", "uint16"], default="uint16")
    parser.add_argument("--mask-drop-rate", type=float, default=0.05)
    parser.add_argument("--rej-high", type=float, default=3.0)
    parser.add_argument("--rej-low", type=float, default=3.0)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument(
        "--skip-zero-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    cases = parse_cases(args.cases)
    stack, mask = _make_stack(args)
    build = custom_op_build_info()
    results: dict[str, dict[str, object]] = {}

    for case_name in cases:
        if case_name == "cuda_host_io" and not build.get("cuda"):
            results[case_name] = {
                "skipped": True,
                "reason": "CUDA backend is not built",
            }
            continue
        try:
            results[case_name] = run_benchmark(
                _case_fn(case_name, stack, mask, args),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        except RuntimeError as exc:
            if case_name == "cuda_host_io" and is_cuda_runtime_unavailable_error(exc):
                results[case_name] = {
                    "skipped": True,
                    "reason": f"CUDA runtime unavailable: {exc}",
                }
                continue
            raise

    return {
        "suite": SUITE_ID,
        "config": {
            "cases": cases,
            "frames": args.frames,
            "chunk_rows": args.chunk_rows,
            "width": args.width,
            "channels": args.channels,
            "dtype": args.dtype,
            "mask_drop_rate": args.mask_drop_rate,
            "skip_zero_rgb": args.skip_zero_rgb,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "input_source": {
            "mode": "synthetic",
            "resolved_frames": args.frames,
            "resolved_shape": [args.chunk_rows, args.width, args.channels],
            "resolved_dtype": args.dtype,
        },
        "custom_ops": build,
        "results": results,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from bench.common import print_or_save_report

    print_or_save_report(run(args), args.output_json)


if __name__ == "__main__":
    main()
