# Norma benchmark

该目录包含 Norma 双帧相机模型对齐的本地开发基准。它直接调用
`solve_pywt_alignment()`，用于检查默认求解路线的匹配覆盖、收敛误差，
以及提供 mask 时的图像域残余位移。

## 边界

- 算法编排由 `hoshicore.component.norma` 提供；benchmark 不重复实现检测、
  匹配、优化或 refine。
- benchmark 只接受 `solve_pywt_alignment()` 真实支持的 bootstrap 路径和参数。
- 本地数据集放在 `benchmarks/local/`，不进入版本控制。
- 根目录 `debug_*.py` 和 notebook 是独立诊断工具，不属于 benchmark API，
  也不应被正式测试导入。

## 运行

```powershell
python -m benchmarks.benchmark_norma_alignment `
  benchmarks/local/norma_alignment_local.json `
  --output-dir benchmark_results/local_current `
  --write-remap
```

可使用 `--case ID`、`--tag TAG` 筛选样本，使用 `--seed` 固定 RANSAC
随机种子。默认 seed 为 `0`。

## 数据集

数据集包含可选的 `defaults` 和必需的 `cases`。每个 case 至少包含：

- `id`
- `reference`
- `source`

`alignment` 中可设置生产 solver 已公开的参数，例如 `matching_path`、
`same_camera`、`bootstrap_scales`、相机初始化参数以及 `guided_refine`。
当前回归基线应保持 `matching_path: "asterism"` 和 `guided_refine: false`。
未公开或已移除的实验参数会被拒绝，而不是被静默忽略。

提供 `mask`，或在 `evaluation` 中提供 `reference_mask` / `source_mask` 后，
会评价 remap 图像的局部 residual-shift P90。没有 mask 时只评价匹配点收敛。

## 核心指标

- `final_pairs`：最终参与评价的匹配数量。
- `coverage_ratio`、`outer_pairs`：防止匹配集中在图像中心。
- `p90_px`：最终匹配点投影残差，仅表示 solver 内部收敛。
- `remap_residual_shift_p90_px`：图像域残余位移。
- `remap_evaluated_tiles`、`remap_evaluated_coverage`：图像指标支持度。

输出包括：

- `results.json`：单一、紧凑的 case 结果。
- `summary.csv`：核心指标投影。
- `<case>/src_aligned.tif` 与 `<case>/tgt_reference.tif`：启用
  `--write-remap` 时输出的人工检查图。

运行信息会记录 Git revision、Python、NumPy、OpenCV、平台和每个 case 的
随机 seed。
