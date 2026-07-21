"""Benchmark for camera-model remap host-in/host-out paths.

Compares the camera-model reference path against the fused custom-op path.  The
benchmark covers perspective and Kannala-Brandt fisheye projection pairs.
"""

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
from hoshicore._custom_op import build_info as custom_op_build_info
from hoshicore._custom_op.ops import remap as remap_ops
from hoshicore.component.norma.projection import (
    project_fisheye_vectors,
    project_vectors,
    unproject_fisheye_pixels,
    unproject_pixels,
)


DEFAULT_HEIGHT = 2048
DEFAULT_WIDTH = 3072


CASE_NAMES = [
    "numpy_grid",
    "custom_op_fused",
    "custom_op_auto",
    "custom_op_cpu_fused",
    "opencv_remap",
    "original_remap",
]
DEFAULT_CASES = [
    "numpy_grid",
    "custom_op_fused",
    "opencv_remap",
    "original_remap",
]
SUITE_ID = "gpu.original_remap"


@dataclass(frozen=True)
class RemapConfig:
    height: int
    width: int
    src_height: int
    src_width: int
    fx_src: float
    fy_src: float
    cx_src: float
    cy_src: float
    fx_dst: float
    fy_dst: float
    cx_dst: float
    cy_dst: float
    rotation_dst_to_src: np.ndarray
    src_dist_coeffs: np.ndarray | None
    dst_dist_coeffs: np.ndarray | None
    src_projection: str
    dst_projection: str


def _make_rotation_matrix(yaw_deg: float,
                          pitch_deg: float,
                          roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float32)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float32)
    return rz @ ry @ rx


def _build_config(
    args: argparse.Namespace,
    image_shape: tuple[int, ...] | None,
) -> RemapConfig:
    if image_shape is not None:
        input_height = int(image_shape[0])
        input_width = int(image_shape[1])
    else:
        input_height = DEFAULT_HEIGHT if args.height is None else args.height
        input_width = DEFAULT_WIDTH if args.width is None else args.width

    if image_shape is not None:
        if args.src_height is not None and args.src_height != input_height:
            raise ValueError(
                f"--src-height={args.src_height} does not match input image height={input_height}"
            )
        if args.src_width is not None and args.src_width != input_width:
            raise ValueError(
                f"--src-width={args.src_width} does not match input image width={input_width}"
            )

    height = input_height if args.height is None else args.height
    width = input_width if args.width is None else args.width
    src_height = input_height if args.src_height is None else args.src_height
    src_width = input_width if args.src_width is None else args.src_width

    cx_src = (src_width - 1) * 0.5
    cy_src = (src_height - 1) * 0.5
    cx_dst = (width - 1) * 0.5
    cy_dst = (height - 1) * 0.5

    return RemapConfig(
        height=height,
        width=width,
        src_height=src_height,
        src_width=src_width,
        fx_src=args.src_focal_px,
        fy_src=args.src_focal_px,
        cx_src=cx_src,
        cy_src=cy_src,
        fx_dst=args.dst_focal_px,
        fy_dst=args.dst_focal_px,
        cx_dst=cx_dst,
        cy_dst=cy_dst,
        rotation_dst_to_src=_make_rotation_matrix(
            args.yaw_deg, args.pitch_deg, args.roll_deg),
        src_dist_coeffs=_make_dist_coeffs(
            args.distortion_scale, args.src_projection),
        dst_dist_coeffs=_make_dist_coeffs(
            -args.distortion_scale, args.dst_projection),
        src_projection=args.src_projection,
        dst_projection=args.dst_projection,
    )


def _make_dist_coeffs(scale: float, projection: str) -> np.ndarray | None:
    if scale == 0.0:
        return None
    if projection == "fisheye":
        return np.array(
            [0.01 * scale, -0.0015 * scale, 0.0002 * scale, -0.00002 * scale],
            dtype=np.float32,
        )
    return np.array(
        [0.01 * scale, -0.0015 * scale, 0.0008 * scale, -0.0004 * scale, 0.0001 * scale],
        dtype=np.float32,
    )


def build_grid_numpy(cfg: RemapConfig) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(cfg.width, dtype=np.float64)
    ys = np.arange(cfg.height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    dst_pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    k_dst = _make_camera_matrix(cfg.fx_dst, cfg.fy_dst, cfg.cx_dst, cfg.cy_dst)
    if cfg.dst_projection == "fisheye":
        dst_rays = unproject_fisheye_pixels(
            dst_pixels, k_dst, cfg.dst_dist_coeffs)
    else:
        dst_rays = unproject_pixels(dst_pixels, k_dst, cfg.dst_dist_coeffs)
    src_rays = (cfg.rotation_dst_to_src.astype(np.float64) @ dst_rays.T).T
    k_src = _make_camera_matrix(cfg.fx_src, cfg.fy_src, cfg.cx_src, cfg.cy_src)
    if cfg.src_projection == "fisheye":
        src_pixels = project_fisheye_vectors(
            src_rays, k_src, cfg.src_dist_coeffs)
    else:
        src_pixels = project_vectors(src_rays, k_src, cfg.src_dist_coeffs)

    return (
        src_pixels[:, 0].reshape(cfg.height, cfg.width).astype(np.float32),
        src_pixels[:, 1].reshape(cfg.height, cfg.width).astype(np.float32),
    )


def _make_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def remap_with_cv2(image: np.ndarray,
                   map_x: np.ndarray,
                   map_y: np.ndarray) -> np.ndarray:
    remapped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
        hint=cv2.ALGO_HINT_ACCURATE,
    )
    if image.ndim == 3 and image.shape[2] == 1 and remapped.ndim == 2:
        return remapped[:, :, None]
    return remapped


def remap_with_custom_op(image: np.ndarray, cfg: RemapConfig) -> np.ndarray:
    return remap_ops.camera_model_remap_compiled(
        image=image,
        out_height=cfg.height,
        out_width=cfg.width,
        fx_src=cfg.fx_src,
        fy_src=cfg.fy_src,
        cx_src=cfg.cx_src,
        cy_src=cfg.cy_src,
        fx_dst=cfg.fx_dst,
        fy_dst=cfg.fy_dst,
        cx_dst=cfg.cx_dst,
        cy_dst=cfg.cy_dst,
        rotation_dst_to_src=cfg.rotation_dst_to_src,
        src_dist_coeffs=cfg.src_dist_coeffs,
        dst_dist_coeffs=cfg.dst_dist_coeffs,
        src_projection=cfg.src_projection,
        dst_projection=cfg.dst_projection,
    )


def remap_with_auto_custom_op(image: np.ndarray, cfg: RemapConfig) -> np.ndarray:
    return remap_ops.camera_model_remap(
        image=image,
        out_height=cfg.height,
        out_width=cfg.width,
        fx_src=cfg.fx_src,
        fy_src=cfg.fy_src,
        cx_src=cfg.cx_src,
        cy_src=cfg.cy_src,
        fx_dst=cfg.fx_dst,
        fy_dst=cfg.fy_dst,
        cx_dst=cfg.cx_dst,
        cy_dst=cfg.cy_dst,
        rotation_dst_to_src=cfg.rotation_dst_to_src,
        src_dist_coeffs=cfg.src_dist_coeffs,
        dst_dist_coeffs=cfg.dst_dist_coeffs,
        src_projection=cfg.src_projection,
        dst_projection=cfg.dst_projection,
    )


def remap_with_cpu_custom_op(image: np.ndarray, cfg: RemapConfig) -> np.ndarray:
    return remap_ops.camera_model_remap_cpu_compiled(
        image=image,
        out_height=cfg.height,
        out_width=cfg.width,
        fx_src=cfg.fx_src,
        fy_src=cfg.fy_src,
        cx_src=cfg.cx_src,
        cy_src=cfg.cy_src,
        fx_dst=cfg.fx_dst,
        fy_dst=cfg.fy_dst,
        cx_dst=cfg.cx_dst,
        cy_dst=cfg.cy_dst,
        rotation_dst_to_src=cfg.rotation_dst_to_src,
        src_dist_coeffs=cfg.src_dist_coeffs,
        dst_dist_coeffs=cfg.dst_dist_coeffs,
        src_projection=cfg.src_projection,
        dst_projection=cfg.dst_projection,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--src-height", type=int, default=None)
    parser.add_argument("--src-width", type=int, default=None)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--src-focal-px", type=float, default=2400.0)
    parser.add_argument("--dst-focal-px", type=float, default=2400.0)
    parser.add_argument(
        "--src-projection",
        choices=["perspective", "fisheye"],
        default="perspective",
    )
    parser.add_argument(
        "--dst-projection",
        choices=["perspective", "fisheye"],
        default="perspective",
    )
    parser.add_argument("--yaw-deg", type=float, default=0.30)
    parser.add_argument("--pitch-deg", type=float, default=0.15)
    parser.add_argument("--roll-deg", type=float, default=0.05)
    parser.add_argument("--distortion-scale", type=float, default=0.0)
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
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip CPU reference accuracy check for large custom-op benchmarks.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    unknown_cases = [case for case in args.cases if case not in CASE_NAMES]
    if unknown_cases:
        raise ValueError(
            f"Unknown original remap benchmark case(s): {unknown_cases}. "
            f"Available: {list(CASE_NAMES)}")

    input_height = args.src_height
    if input_height is None:
        input_height = DEFAULT_HEIGHT if args.height is None else args.height
    input_width = args.src_width
    if input_width is None:
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
    channel_count = int(image.shape[2]) if image.ndim == 3 else 1
    map_cache: tuple[np.ndarray, np.ndarray] | None = None

    def get_cached_maps() -> tuple[np.ndarray, np.ndarray]:
        nonlocal map_cache
        if map_cache is None:
            map_cache = build_grid_numpy(cfg)
        return map_cache

    runners = {
        "numpy_grid": lambda: build_grid_numpy(cfg),
        "custom_op_fused": lambda: remap_with_custom_op(image, cfg),
        "custom_op_auto": lambda: remap_with_auto_custom_op(image, cfg),
        "custom_op_cpu_fused": lambda: remap_with_cpu_custom_op(image, cfg),
        "opencv_remap": lambda: remap_with_cv2(image, *get_cached_maps()),
        "original_remap": lambda: remap_with_cv2(image, *build_grid_numpy(cfg)),
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
        "suite": "original_remap",
        "env": {
            **collect_env_info(),
            "cv2": cv2.__version__,
            "custom_op_build": custom_op_build_info(),
        },
        "config": {
            "height": cfg.height,
            "width": cfg.width,
            "src_height": cfg.src_height,
            "src_width": cfg.src_width,
            "requested_height": args.height,
            "requested_width": args.width,
            "channels": channel_count,
            "requested_channels": args.channels,
            "src_focal_px": args.src_focal_px,
            "dst_focal_px": args.dst_focal_px,
            "src_projection": cfg.src_projection,
            "dst_projection": cfg.dst_projection,
            "yaw_deg": args.yaw_deg,
            "pitch_deg": args.pitch_deg,
            "roll_deg": args.roll_deg,
            "distortion_scale": args.distortion_scale,
            "input_dir": args.input_dir,
            "input_mode": args.input_mode,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cases": args.cases,
            "skip_accuracy": args.skip_accuracy,
        },
        "input_source": input_source,
        "results": results,
    }
    custom_backends = {}
    accuracy_by_case = {}
    if "custom_op_fused" in args.cases:
        custom_backends["custom_op_fused"] = "compiled_cuda_host_io"
        if not args.skip_accuracy:
            reference = remap_with_cv2(image, *get_cached_maps()).astype(np.float64)
            custom = remap_with_custom_op(image, cfg).astype(np.float64)
            abs_err = np.abs(custom - reference)
            accuracy_by_case["custom_op_fused"] = {
                "max_abs_err": float(np.max(abs_err)),
                "mean_abs_err": float(np.mean(abs_err)),
            }
    if "custom_op_auto" in args.cases:
        custom_backends["custom_op_auto"] = "auto_dispatch"
        if not args.skip_accuracy:
            reference = remap_with_cv2(image, *get_cached_maps()).astype(np.float64)
            custom = remap_with_auto_custom_op(image, cfg).astype(np.float64)
            abs_err = np.abs(custom - reference)
            accuracy_by_case["custom_op_auto"] = {
                "max_abs_err": float(np.max(abs_err)),
                "mean_abs_err": float(np.mean(abs_err)),
            }
    if "custom_op_cpu_fused" in args.cases:
        custom_backends["custom_op_cpu_fused"] = "compiled_openmp_cpu"
        if not args.skip_accuracy:
            reference = remap_with_cv2(image, *get_cached_maps()).astype(np.float64)
            custom = remap_with_cpu_custom_op(image, cfg).astype(np.float64)
            abs_err = np.abs(custom - reference)
            accuracy_by_case["custom_op_cpu_fused"] = {
                "max_abs_err": float(np.max(abs_err)),
                "mean_abs_err": float(np.mean(abs_err)),
            }
    if custom_backends:
        report["custom_backends"] = custom_backends
    if accuracy_by_case:
        report["accuracy_by_case"] = accuracy_by_case
        if "custom_op_fused" in accuracy_by_case:
            report["accuracy"] = accuracy_by_case["custom_op_fused"]
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run(args)
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
