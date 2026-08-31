"""CPU benchmark for pure homography warp."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import cv2
import numpy as np

from bench.common import (
    collect_env_info,
    prepare_frames,
    print_or_save_report,
    run_benchmark,
)


DEFAULT_HEIGHT = 2048
DEFAULT_WIDTH = 3072


CASE_NAMES = [
    "opencv_warp",
]
DEFAULT_CASES = CASE_NAMES
SUITE_ID = "gpu.original_homography"


@dataclass(frozen=True)
class WarpConfig:
    height: int
    width: int
    channels: int
    homography_src_to_dst: np.ndarray


def _make_homography(args: argparse.Namespace) -> np.ndarray:
    tx = args.tx_px
    ty = args.ty_px
    angle = math.radians(args.rotation_deg)
    scale = args.scale
    persp_x = args.persp_x
    persp_y = args.persp_y

    c = math.cos(angle) * scale
    s = math.sin(angle) * scale
    return np.array([
        [c, -s, tx],
        [s, c, ty],
        [persp_x, persp_y, 1.0],
    ], dtype=np.float32)


def _build_config(
    args: argparse.Namespace,
    image_shape: tuple[int, ...] | None,
) -> WarpConfig:
    if image_shape is not None:
        height = int(image_shape[0]) if args.height is None else args.height
        width = int(image_shape[1]) if args.width is None else args.width
        channels = int(image_shape[2]) if len(image_shape) == 3 else 1
    else:
        height = DEFAULT_HEIGHT if args.height is None else args.height
        width = DEFAULT_WIDTH if args.width is None else args.width
        channels = args.channels

    return WarpConfig(
        height=height,
        width=width,
        channels=channels,
        homography_src_to_dst=_make_homography(args),
    )


def warp_with_cv2(image: np.ndarray,
                  h_src_to_dst: np.ndarray,
                  output_size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(
        image,
        h_src_to_dst,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--tx-px", type=float, default=12.0)
    parser.add_argument("--ty-px", type=float, default=-8.0)
    parser.add_argument("--rotation-deg", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=1.0005)
    parser.add_argument("--persp-x", type=float, default=1e-6)
    parser.add_argument("--persp-y", type=float, default=-8e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument(
        "--input-mode",
        choices=["auto", "cache", "images", "synthetic"],
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    unknown_cases = [case for case in args.cases if case not in CASE_NAMES]
    if unknown_cases:
        raise ValueError(
            f"Unknown original homography benchmark case(s): {unknown_cases}. "
            f"Available: {list(CASE_NAMES)}")

    input_height = DEFAULT_HEIGHT if args.height is None else args.height
    input_width = DEFAULT_WIDTH if args.width is None else args.width
    frames, input_source = prepare_frames(
        frames=1,
        height=input_height,
        width=input_width,
        dtype=np.float32,
        channels=args.channels,
        seed=args.seed,
        input_dir=args.input_dir,
        input_mode=args.input_mode,
    )
    image = frames[0]
    if input_source.get("mode") == "synthetic" and args.channels == 1 and image.ndim == 2:
        image = image[:, :, None]
        input_source = {
            **input_source,
            "resolved_shape": list(image.shape),
        }
    cfg = _build_config(args, image.shape)
    output_size = (cfg.width, cfg.height)

    runners = {
        "opencv_warp": lambda: warp_with_cv2(
            image, cfg.homography_src_to_dst, output_size),
    }
    results = {
        case_name: run_benchmark(
            runners[case_name],
            warmup=args.warmup,
            repeat=args.repeat,
        )
        for case_name in args.cases
    }

    report = {
        "suite": "original_homography",
        "env": {
            **collect_env_info(),
            "cv2": cv2.__version__,
        },
        "config": {
            "height": cfg.height,
            "width": cfg.width,
            "channels": cfg.channels,
            "requested_channels": args.channels,
            "requested_height": args.height,
            "requested_width": args.width,
            "tx_px": args.tx_px,
            "ty_px": args.ty_px,
            "rotation_deg": args.rotation_deg,
            "scale": args.scale,
            "persp_x": args.persp_x,
            "persp_y": args.persp_y,
            "input_dir": args.input_dir,
            "input_mode": args.input_mode,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cases": args.cases,
        },
        "input_source": input_source,
        "results": results,
    }
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run(args)
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
