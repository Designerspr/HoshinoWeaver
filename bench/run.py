"""Unified benchmark suite driver."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SuiteSpec:
    module: str
    description: str


SUITES: dict[str, SuiteSpec] = {
    "cpu.kernels": SuiteSpec(
        module="bench.cpu.kernels",
        description="CPU/custom-op kernel microbenchmarks",
    ),
    "cpu.max_stack": SuiteSpec(
        module="bench.cpu.max_stack",
        description="Max stacking stream benchmarks",
    ),
    "cpu.fgp_accumulate": SuiteSpec(
        module="bench.cpu.fgp_accumulate",
        description="FastGaussianParam accumulation benchmarks",
    ),
    "cpu.alignment": SuiteSpec(
        module="bench.cpu.alignment",
        description="Alignment stage benchmarks",
    ),
    "gpu.original_remap": SuiteSpec(
        module="bench.gpu.original_remap",
        description="Camera-model remap host-in/host-out benchmarks",
    ),
    "gpu.original_homography": SuiteSpec(
        module="bench.gpu.original_homography",
        description="OpenCV homography warp baseline benchmark",
    ),
    "pipeline.alignment": SuiteSpec(
        module="bench.pipeline.alignment",
        description="End-to-end production alignment pipeline benchmark",
    ),
    "pipeline.compute": SuiteSpec(
        module="bench.pipeline.compute",
        description="Representative production compute-path overview benchmark",
    ),
    "pipeline.all": SuiteSpec(
        module="bench.pipeline.workflow",
        description="End-to-end YAML/DAG workflow benchmark",
    ),
    "pipeline.workflow": SuiteSpec(
        module="bench.pipeline.workflow",
        description="End-to-end YAML/DAG workflow benchmark",
    ),
}


def _load_suite(suite_id: str):
    try:
        spec = SUITES[suite_id]
    except KeyError as exc:
        available = ", ".join(sorted(SUITES))
        raise SystemExit(f"Unknown benchmark suite: {suite_id}. Available: {available}") from exc
    return importlib.import_module(spec.module)


def _normalize_remainder(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _suite_cases(module) -> list[str]:
    cases = getattr(module, "CASE_NAMES", None)
    if cases is None:
        return []
    return [str(case) for case in cases]


def _suite_default_cases(module) -> list[str]:
    cases = getattr(module, "DEFAULT_CASES", None)
    if cases is None:
        return []
    return [str(case) for case in cases]


def _run_suite(
    suite_id: str,
    suite_args: list[str],
    *,
    print_report: bool = True,
) -> dict[str, Any]:
    module = _load_suite(suite_id)
    parser = module.build_parser()
    args = parser.parse_args(_normalize_remainder(suite_args))
    report = module.run(args)
    if isinstance(report, dict):
        report["suite"] = suite_id
    if print_report:
        from bench.common import print_or_save_report

        print_or_save_report(report, getattr(args, "output_json", None))
    return report


def list_suites() -> None:
    for suite_id in sorted(SUITES):
        spec = SUITES[suite_id]
        print(f"{suite_id}\t{spec.description}")


def list_cases(suite_id: str | None) -> None:
    suite_ids = [suite_id] if suite_id else sorted(SUITES)
    for idx, current in enumerate(suite_ids):
        module = _load_suite(current)
        print(f"[{current}]")
        defaults = set(_suite_default_cases(module))
        for case in _suite_cases(module):
            suffix = " (default)" if case in defaults else ""
            print(f"  {case}{suffix}")
        if idx + 1 < len(suite_ids):
            print()


def _profile_suite_args(
    *,
    profile: str,
    input_dir: str | None,
    input_mode: str,
    frames: int,
    height: int,
    width: int,
    channels: int,
    dtype: str,
    warmup: int,
    repeat: int,
) -> list[tuple[str, list[str]]]:
    input_args = ["--input-mode", input_mode]
    if input_dir is not None:
        input_args.extend(["--input-dir", input_dir])

    if profile == "smoke":
        return [
            (
                "cpu.max_stack",
                [
                    "--frames", "2",
                    "--height", "64",
                    "--width", "64",
                    "--channels", "1",
                    "--dtype", "uint8",
                    "--input-mode", "synthetic",
                    "--cases", "single_numpy_stream,single_openmp_stream",
                    "--warmup", "0",
                    "--repeat", "1",
                ],
            ),
            (
                "cpu.fgp_accumulate",
                [
                    "--frames", "2",
                    "--height", "64",
                    "--width", "64",
                    "--channels", "1",
                    "--dtype", "uint8",
                    "--input-mode", "synthetic",
                    "--cases", "single_python_stream,single_numpy_stream",
                    "--warmup", "0",
                    "--repeat", "1",
                ],
            ),
            (
                "gpu.original_remap",
                [
                    "--height", "64",
                    "--width", "64",
                    "--channels", "1",
                    "--input-mode", "synthetic",
                    "--cases", "numpy_grid", "opencv_remap",
                    "--skip-accuracy",
                    "--warmup", "0",
                    "--repeat", "1",
                ],
            ),
            (
                "gpu.original_homography",
                [
                    "--height", "64",
                    "--width", "64",
                    "--channels", "1",
                    "--input-mode", "synthetic",
                    "--cases", "opencv_warp",
                    "--warmup", "0",
                    "--repeat", "1",
                ],
            ),
        ]

    if profile != "local":
        raise SystemExit(f"Unknown profile: {profile}")

    common_frame_args = [
        "--frames", str(frames),
        "--height", str(height),
        "--width", str(width),
        "--channels", str(channels),
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        *input_args,
    ]
    return [
        (
            "cpu.kernels",
            [
                "--frames", str(frames),
                "--height", str(height),
                "--width", str(width),
                "--channels", str(channels),
                "--dtype", dtype,
                *input_args,
                "--cases", "max_combine_stream_numpy,max_combine_stream_compiled,wavelet_dec_rec_auto",
                "--warmup", str(warmup),
                "--repeat", str(repeat),
            ],
        ),
        (
            "cpu.alignment",
            [
                *common_frame_args,
                "--cases", "detect_stream", "match_stream", "remap_stream",
            ],
        ),
        (
            "gpu.original_remap",
            [
                "--height", str(height),
                "--width", str(width),
                "--channels", str(channels),
                *input_args,
                "--cases", "custom_op_fused", "opencv_remap",
                "--skip-accuracy",
                "--warmup", str(warmup),
                "--repeat", str(repeat),
            ],
        ),
        (
            "gpu.original_homography",
            [
                "--height", str(height),
                "--width", str(width),
                "--channels", str(channels),
                *input_args,
                "--cases", "opencv_warp",
                "--warmup", str(warmup),
                "--repeat", str(repeat),
            ],
        ),
    ]


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    input_sources: dict[str, Any] = {}
    flattened_results: dict[str, Any] = {}
    suite_runs = _profile_suite_args(
        profile=args.profile,
        input_dir=args.input_dir,
        input_mode=args.input_mode,
        frames=args.frames,
        height=args.height,
        width=args.width,
        channels=args.channels,
        dtype=args.dtype,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    for suite_id, suite_args in suite_runs:
        report = _run_suite(suite_id, suite_args, print_report=False)
        reports[suite_id] = report
        input_source = report.get("input_source")
        if isinstance(input_source, dict):
            input_sources[suite_id] = input_source
        results = report.get("results", {})
        if isinstance(results, dict):
            for case_name, payload in results.items():
                flattened_results[f"{suite_id}:{case_name}"] = payload

    from bench.common import collect_env_info, print_or_save_report

    aggregate = {
        "suite": f"bench.cli.{args.profile}",
        "env": collect_env_info(),
        "config": {
            "profile": args.profile,
            "input_dir": args.input_dir,
            "input_mode": args.input_mode,
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "channels": args.channels,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "suite_runs": [
                {"suite": suite_id, "args": suite_args}
                for suite_id, suite_args in suite_runs
            ],
        },
        "input_sources": input_sources,
        "results": flattened_results,
        "reports": reports,
    }
    print_or_save_report(aggregate, args.output_json)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List benchmark suites")

    list_cases_parser = subparsers.add_parser(
        "list-cases",
        help="List cases for one suite or all suites",
    )
    list_cases_parser.add_argument("suite", nargs="?")

    run_parser = subparsers.add_parser("run", help="Run one benchmark suite")
    run_parser.add_argument("suite", choices=sorted(SUITES))
    run_parser.add_argument("suite_args", nargs=argparse.REMAINDER)

    profile_parser = subparsers.add_parser(
        "profile",
        help="Run a predefined benchmark profile",
    )
    profile_parser.add_argument("profile", choices=["smoke", "local"])
    profile_parser.add_argument("--input-dir", type=str, default=None)
    profile_parser.add_argument(
        "--input-mode",
        choices=["auto", "cache", "images", "synthetic"],
        default="auto",
    )
    profile_parser.add_argument("--frames", type=int, default=10)
    profile_parser.add_argument("--height", type=int, default=2048)
    profile_parser.add_argument("--width", type=int, default=3072)
    profile_parser.add_argument("--channels", type=int, default=3)
    profile_parser.add_argument("--dtype", type=str, default="uint16")
    profile_parser.add_argument("--warmup", type=int, default=1)
    profile_parser.add_argument("--repeat", type=int, default=3)
    profile_parser.add_argument("--output-json", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "list":
        list_suites()
    elif args.command == "list-cases":
        list_cases(args.suite)
    elif args.command == "run":
        _run_suite(args.suite, args.suite_args)
    elif args.command == "profile":
        run_profile(args)
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
