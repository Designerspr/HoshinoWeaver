"""独立诊断脚本：按 star_shrink 流程逐步处理并保存中间结果。

用法：
    python -m tools.debug.debug_star_shrink <input_image> [options]

选项：
    --output-dir DIR     输出目录（默认 debug_star_shrink_out/）
    --mode MODE ...      测试模式列表（默认 trail standard post_align）
    --detect-method M    检测方法 threshold / dog（默认 threshold）
    --detect-ksize N     检测核大小（默认 13）
    --detect-threshold F 检测阈值（默认 5.0）
    --detect-dilate N    检测膨胀（默认 2）
    --no-compare         不生成水平对比图

示例：
    python -m tools.debug.debug_star_shrink my_star.tif
    python -m tools.debug.debug_star_shrink my_star.tif --mode trail post_align --output-dir out/
"""

import argparse
import os

import cv2
import numpy as np

from hoshicore.component.star_shrink import apply_mask


# ---------------------------------------------------------------------------
# 图像 I/O 工具
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """支持 8-bit / 16-bit PNG/TIF，返回 BGR uint8 或 uint16。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def save_image(path: str, img: np.ndarray):
    """保存图像，uint16 直接写，其他先转 uint8 再写。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if img.dtype == np.uint16:
        cv2.imwrite(path, img)
    else:
        if img.dtype != np.uint8:
            # float32 [0,1] 或其他 → uint8
            if img.dtype.kind == 'f':
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        cv2.imwrite(path, img)


def to_display_uint8(img: np.ndarray) -> np.ndarray:
    """任意 dtype → uint8，用于生成对比图。"""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    if img.dtype.kind == 'f':
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img.astype(np.uint8)


def save_mask_visual(path: str, mask: np.ndarray):
    """将二值 star_mask 保存为白底可视化图。"""
    vis = (mask * 255).astype(np.uint8)
    cv2.imwrite(path, vis)


# ---------------------------------------------------------------------------
# Top-hat 全局压制
# ---------------------------------------------------------------------------

def tophat_suppress(img: np.ndarray, ksize: int, strength: float) -> np.ndarray:
    """用形态学 top-hat 压制未被蒙版覆盖的小星点。

    opening = morph_open(img, ksize)  →  保留所有大于核尺寸的平滑结构
    tophat  = img - opening           →  仅包含小于核尺寸的点状亮结构
    result  = img - strength * tophat →  按比例减小这些结构对背景的亮度超出量

    strength=1.0 将所有小于核的点状结构完全压到局部背景水平；
    strength=0.5 减半其亮度超出量。
    不接触梯度、星云、背景噪声基底（它们在 opening 中被保留）。
    """
    from hoshicore.component.star_shrink import get_morph_kernel

    raw_dtype = img.dtype
    if img.dtype.kind == 'f':
        img_f = img.astype(np.float32)
        max_val = 1.0
    else:
        max_val = float(np.iinfo(img.dtype).max)
        img_f = img.astype(np.float32) / max_val

    kernel = get_morph_kernel("CIRCLE", ksize)
    opening = cv2.morphologyEx(img_f, cv2.MORPH_OPEN, kernel)
    tophat = np.maximum(img_f - opening, 0.0)
    suppressed_f = img_f - strength * tophat
    suppressed_f = np.clip(suppressed_f, 0.0, 1.0)

    if raw_dtype.kind == 'f':
        return suppressed_f.astype(raw_dtype)
    return np.round(suppressed_f * max_val).astype(raw_dtype)

def run_pipeline_with_intermediates(
    img: np.ndarray,
    mode: str,
    detect_configs: dict,
    output_dir: str,
    do_timing: bool = False,
) -> dict[str, np.ndarray]:
    """执行完整流水线，每步结果保存到磁盘，返回 {step_name: array} 字典。"""
    import time
    from hoshicore.component.star_detect import (
        detect_starmask_by_threshold, detect_starmask_by_dog)
    from hoshicore.component.star_shrink import (
        apply_mask, deringing, morph_shrink_luma)
    from hoshicore.ops.star_ops import SHRINK_MODE_PRESETS

    prefix = os.path.join(output_dir, f"{mode}")

    if mode == "custom":
        p = detect_configs
    elif mode in SHRINK_MODE_PRESETS:
        p = SHRINK_MODE_PRESETS[mode]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    results: dict[str, np.ndarray] = {}
    _t: dict[str, float] = {}

    # 0 原图
    results["0_original"] = img
    save_image(f"{prefix}_0_original.png", img)
    print(f"  [0] original saved")

    # 1 star mask
    method = detect_configs["detect_method"]
    detect_kwargs = {
        "threshold_ratio": detect_configs["detect_threshold"],
        "open_ksize":      detect_configs["detect_open"],
        "dilate_ksize":    detect_configs["detect_dilate"],
    }
    if method == "threshold":
        detect_kwargs["ksize"] = detect_configs["detect_ksize"]
        t0 = time.perf_counter()
        star_mask = detect_starmask_by_threshold(img, **detect_kwargs)
    elif method == "dog":
        detect_kwargs["sigma_small"] = detect_configs["dog_sigma_small"]
        detect_kwargs["sigma_large"] = detect_configs["dog_sigma_large"]
        t0 = time.perf_counter()
        star_mask = detect_starmask_by_dog(img, **detect_kwargs)
    else:
        raise ValueError(f"Unknown detect_method: {method}")
    if do_timing:
        _t["1_detect"] = time.perf_counter() - t0

    results["1_star_mask"] = star_mask
    save_mask_visual(f"{prefix}_1_starmask.png", star_mask)
    print(f"  [1] star_mask saved  ({star_mask.mean() * 100:.2f}% masked)")

    # 2 morph_shrink_luma
    raw_ratio = p.get("shrink_ratio", 0.0)
    shrink_ratio = None if raw_ratio == 0.0 else raw_ratio
    t0 = time.perf_counter()
    shrunk = morph_shrink_luma(
        img,
        ksize=p["shrink_ksize"],
        times=p["shrink_times"],
        ratio=shrink_ratio,
    )
    if do_timing:
        _t["2_morph"] = time.perf_counter() - t0
    results["2_after_morph_shrink_luma"] = shrunk
    save_image(f"{prefix}_2_morph_shrink_luma.png", shrunk)
    ratio_str = (f"{shrink_ratio:.2f}" if shrink_ratio is not None
                 else f"auto={1.0/p['shrink_times']:.2f}")
    print(f"  [2] morph_shrink_luma  (ksize={p['shrink_ksize']}, "
          f"times={p['shrink_times']}, ratio={ratio_str})")

    # 3 deringing
    t0 = time.perf_counter()
    deringed = deringing(img, shrunk, algo="mean", ksize=51)
    if do_timing:
        _t["3_deringing"] = time.perf_counter() - t0
    results["3_after_deringing"] = deringed
    save_image(f"{prefix}_3_deringing.png", deringed)
    print(f"  [3] deringing  (ksize=11)")

    # 4 final
    t0 = time.perf_counter()
    final = apply_mask(img, deringed, star_mask)
    if do_timing:
        _t["4_mask"] = time.perf_counter() - t0
    results["4_final"] = final
    save_image(f"{prefix}_4_final.png", final)
    print(f"  [4] final saved")

    # diff ×4
    diff = np.clip(
        (final.astype(np.float32) - img.astype(np.float32)) * 4 + 128, 0, 255
    ).astype(np.uint8)
    cv2.imwrite(f"{prefix}_diff_4x.png", diff)
    print(f"  [D] diff_4x saved")

    if do_timing:
        total = sum(_t.values()) or 1e-9
        print(f"\n  [{mode}] per-step timing:")
        for name, elapsed in _t.items():
            print(f"    {name:16s}: {elapsed * 1000:7.1f} ms  ({elapsed / total * 100:.0f}%)")

    return results


# ---------------------------------------------------------------------------
# 对比图
# ---------------------------------------------------------------------------

def crop_center_fraction(img: np.ndarray, fraction: float) -> np.ndarray:
    """从图像中心裁取面积比例为 fraction 的矩形，保持原始长宽比。

    Args:
        img: 输入图像，2D 或 3D。
        fraction: 裁剪面积占原图面积的比例 (0, 1]。
                  0.1 → 边长各缩为 sqrt(0.1) ≈ 31.6%，面积精确为 10%。

    Returns:
        裁剪后的图像，dtype 与输入一致。
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    scale = fraction ** 0.5
    h, w = img.shape[:2]
    ch = max(1, round(h * scale))
    cw = max(1, round(w * scale))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]

def make_comparison(
    all_results: dict[str, dict[str, np.ndarray]],
    output_dir: str,
    step: str = "4_final",
    label: str | None = None,
):
    """将不同 mode 的同一步骤结果水平拼接并保存。"""
    frames = []
    labels = []
    for mode, res in all_results.items():
        if step in res:
            frames.append(to_display_uint8(res[step]))
            labels.append(mode)

    if len(frames) < 2:
        return

    # 统一高度
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    strip = np.hstack([cv2.resize(f, (w, h)) for f in frames])

    # 标签（绿色文字）
    label_h = 30
    label_bar = np.zeros((label_h, strip.shape[1], 3), dtype=np.uint8)
    x = 0
    for lbl in labels:
        cv2.putText(label_bar, lbl, (x + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA)
        x += w

    out = np.vstack([label_bar, strip])
    path = os.path.join(output_dir, f"compare_{label or step}.png")
    cv2.imwrite(path, out)
    print(f"\n[compare] {path}")


def make_step_strip(results: dict[str, np.ndarray], mode: str, output_dir: str):
    """将单个 mode 的所有步骤水平拼接。"""
    step_order = [
        "0_original",
        "2_after_morph_shrink_luma",
        "3_after_deringing",
        "4_final",
    ]
    step_labels = ["original", "luma_erode", "deringing", "final"]
    frames = [to_display_uint8(results[k]) for k in step_order if k in results]

    if not frames:
        return

    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    strip = np.hstack([cv2.resize(f, (w, h)) for f in frames])

    label_h = 28
    label_bar = np.zeros((label_h, strip.shape[1], 3), dtype=np.uint8)
    for i, lbl in enumerate(step_labels[:len(frames)]):
        cv2.putText(label_bar, lbl, (i * w + 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 0), 1, cv2.LINE_AA)

    out = np.vstack([label_bar, strip])
    path = os.path.join(output_dir, f"{mode}_steps.png")
    cv2.imwrite(path, out)
    print(f"  [strip] {path}")


def compare_mode_outputs(
    all_results: dict[str, dict[str, np.ndarray]],
    output_dir: str,
    step: str = "4_final",
) -> None:
    """Print PSNR / max_diff / mean_diff between every pair of mode outputs.

    Saves a ×4-amplified absolute diff image for each pair.
    """
    from itertools import combinations
    pairs = list(combinations(all_results.keys(), 2))
    if not pairs:
        return

    print("\n=== 像素级对比 ===")
    for m1, m2 in pairs:
        r1 = all_results[m1].get(step)
        r2 = all_results[m2].get(step)
        if r1 is None or r2 is None:
            continue

        f1 = r1.astype(np.float64)
        f2 = r2.astype(np.float64)
        abs_diff = np.abs(f1 - f2)
        max_val = float(np.iinfo(r1.dtype).max) if r1.dtype.kind != 'f' else 1.0
        mse = float(np.mean((f1 - f2) ** 2))
        psnr = (20.0 * np.log10(max_val / np.sqrt(mse))) if mse > 1e-10 else float('inf')

        print(f"\n  {m1} vs {m2}:")
        print(f"    PSNR:       {psnr:8.2f} dB")
        print(f"    max_diff:   {abs_diff.max():8.1f}")
        print(f"    mean_diff:  {abs_diff.mean():8.3f}")

        diff_vis = np.clip(abs_diff * 4, 0, max_val).astype(r1.dtype)
        path = os.path.join(output_dir, f"cmp_{m1}_vs_{m2}.png")
        save_image(path, diff_vis)
        print(f"    diff_img  → {path}")


def time_pipeline_steps(
    img: np.ndarray,
    mode: str,
    detect_configs: dict,
    n_runs: int,
) -> None:
    """Run only compute steps (no image saving) n_runs times; report avg/min per step."""
    import time
    from hoshicore.component.star_detect import (
        detect_starmask_by_threshold, detect_starmask_by_dog)
    from hoshicore.component.star_shrink import apply_mask, deringing, morph_shrink_luma
    from hoshicore.ops.star_ops import SHRINK_MODE_PRESETS

    p = SHRINK_MODE_PRESETS[mode]
    method = detect_configs["detect_method"]
    detect_kwargs = {
        "threshold_ratio": detect_configs["detect_threshold"],
        "open_ksize":      detect_configs["detect_open"],
        "dilate_ksize":    detect_configs["detect_dilate"],
    }
    if method == "threshold":
        detect_kwargs["ksize"] = detect_configs["detect_ksize"]
        detect_fn = detect_starmask_by_threshold
    else:
        detect_kwargs["sigma_small"] = detect_configs["dog_sigma_small"]
        detect_kwargs["sigma_large"] = detect_configs["dog_sigma_large"]
        detect_fn = detect_starmask_by_dog

    raw_ratio = p.get("shrink_ratio", 0.0)
    shrink_ratio = None if raw_ratio == 0.0 else raw_ratio

    accum: dict[str, list] = {"detect": [], "morph": [], "deringing": [], "mask": []}

    for _ in range(n_runs):
        t = time.perf_counter()
        mask = detect_fn(img, **detect_kwargs)
        accum["detect"].append(time.perf_counter() - t)

        t = time.perf_counter()
        shrunk = morph_shrink_luma(
            img, ksize=p["shrink_ksize"], shape=p["shrink_shape"],
            times=p["shrink_times"], ratio=shrink_ratio)
        accum["morph"].append(time.perf_counter() - t)

        t = time.perf_counter()
        deringed = deringing(img, shrunk, algo="gaussian", ksize=p["deringing_ksize"])
        accum["deringing"].append(time.perf_counter() - t)

        t = time.perf_counter()
        _ = apply_mask(img, deringed, mask)
        accum["mask"].append(time.perf_counter() - t)

    total_avg = sum(sum(v) / n_runs for v in accum.values()) or 1e-9
    print(f"\n  [{mode}] timing ({n_runs} runs):")
    for step_name, times_list in accum.items():
        avg_ms = sum(times_list) / n_runs * 1000
        min_ms = min(times_list) * 1000
        pct = (sum(times_list) / n_runs) / total_avg * 100
        print(f"    {step_name:12s}: avg={avg_ms:7.1f}ms  min={min_ms:7.1f}ms  ({pct:.0f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="输入图像路径")
    p.add_argument("--output-dir", default="debug_star_shrink_out",
                   help="输出目录（默认 debug_star_shrink_out/）")
    p.add_argument("--mode", nargs="+",
                   default=["light", "moderate", "strong", "aggressive", "removal"],
                   help="测试模式（可多个，空格分隔）")
    p.add_argument("--detect-method", default="threshold",
                   choices=["threshold", "dog"])
    p.add_argument("--detect-ksize", type=int, default=13)
    p.add_argument("--detect-threshold", type=float, default=1.0)
    p.add_argument("--detect-open", type=int, default=3)
    p.add_argument("--detect-dilate", type=int, default=2)
    p.add_argument("--dog-sigma-small", type=float, default=1.5)
    p.add_argument("--dog-sigma-large", type=float, default=12.0)
    p.add_argument("--no-compare", action="store_true",
                   help="不生成对比图")
    p.add_argument("--crop-center", type=float, default=None, metavar="FRACTION",
                   help="生成额外的裁剪对比图：从图像中心截取面积比例为 FRACTION 的区域，"
                        "保持原始长宽比（例如 0.1 = 中心 10%% 面积）")
    p.add_argument("--timeit", type=int, default=0, metavar="N",
                   help="每个 mode 重复运行 N 次（仅计算，不保存图像），报告逐步耗时")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input} ...")
    img = load_image(args.input)
    print(f"  shape={img.shape}  dtype={img.dtype}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.crop_center is not None:
        img = crop_center_fraction(img, args.crop_center)
        print(f"  → cropped to {img.shape[1]}×{img.shape[0]} "
              f"(center {args.crop_center:.0%} area)")

    detect_configs = {
        "detect_method":    args.detect_method,
        "detect_ksize":     args.detect_ksize,
        "detect_threshold": args.detect_threshold,
        "detect_open":      args.detect_open,
        "detect_dilate":    args.detect_dilate,
        "dog_sigma_small":  args.dog_sigma_small,
        "dog_sigma_large":  args.dog_sigma_large,
    }

    all_results: dict[str, dict[str, np.ndarray]] = {}

    do_timing = args.timeit == 1

    for mode in args.mode:
        print(f"\n=== mode: {mode} ===")
        results = run_pipeline_with_intermediates(
            img, mode, detect_configs, args.output_dir, do_timing=do_timing)
        all_results[mode] = results
        if not args.no_compare:
            make_step_strip(results, mode, args.output_dir)

    if not args.no_compare and len(all_results) > 1:
        make_comparison(all_results, args.output_dir, step="4_final")
        make_comparison(all_results, args.output_dir, step="0_original")
        compare_mode_outputs(all_results, args.output_dir, step="4_final")

    if args.timeit > 0:
        print(f"\n=== 性能计时（{args.timeit} 次，纯计算）===")
        for mode in args.mode:
            time_pipeline_steps(img, mode, detect_configs, args.timeit)

    print(f"\n完成。输出目录：{os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
