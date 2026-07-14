"""Benchmark Huber weighted chunk host-in/host-out backends."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from bench.common import run_benchmark
from hoshicore._custom_op import build_info as custom_op_build_info
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
import hoshicore._custom_op.ops.fgp as fgp_ops


SUITE_ID = "gpu.huber_chunk"
CASE_NAMES = [
    "numpy",
    "cuda_host_io",
]
DEFAULT_CASES = [
    "numpy",
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


def _make_inputs(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    dtype = np.dtype(args.dtype)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("--dtype must be uint8 or uint16")
    plane_size = args.chunk_rows * args.width * args.channels
    rng = np.random.default_rng(args.seed)
    high = 220 if dtype == np.dtype("uint8") else 4096
    stack = rng.integers(
        64, high, size=(args.frames, plane_size), dtype=dtype)
    ref_mean = np.mean(stack.astype(np.float64), axis=0)
    ref_std = np.std(stack.astype(np.float64), axis=0, ddof=1)
    weights = None
    if args.weighted:
        weights = rng.random(args.frames).astype(np.float64)
    return stack, ref_mean, ref_std, weights


def _case_fn(
    case_name: str,
    stack: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    weights: np.ndarray | None,
    args: argparse.Namespace,
) -> Callable[[], None]:
    fn = {
        "numpy": fgp_ops.huber_weighted_chunk_numpy,
        "cuda_host_io": fgp_ops.huber_weighted_chunk_compiled_cuda,
    }[case_name]

    def run_once() -> None:
        _ = fn(stack, ref_mean, ref_std, args.huber_c, weights)

    return run_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--width", type=int, default=6000)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--dtype", choices=["uint8", "uint16"], default="uint16")
    parser.add_argument("--huber-c", type=float, default=1.345)
    parser.add_argument(
        "--weighted",
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
    stack, ref_mean, ref_std, weights = _make_inputs(args)
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
                _case_fn(case_name, stack, ref_mean, ref_std, weights, args),
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
            "huber_c": args.huber_c,
            "weighted": args.weighted,
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
