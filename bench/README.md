# Benchmark Suite

按职责拆成三层：

- `bench/cpu/`
  CPU kernel 与对齐阶段 benchmark
- `bench/gpu/`
  GPU custom-op、CPU 基线与 host-in/host-out benchmark
- `bench/data_tools/`
  raw cache、测试图片、synthetic starfield 数据生成工具

共享基础设施保留在：

- `bench/common.py`
  公共加载、计时、JSON 输出与输入选择逻辑
- `bench/data/`
  benchmark 输入数据目录

## 当前主入口

### 统一入口

- `python -m bench.cli list`
  列出当前可运行的 benchmark suite。
- `python -m bench.cli list-cases <suite>`
  列出某个 suite 的 case 名称。
- `python -m bench.cli run <suite> -- <suite args...>`
  通过统一入口运行单个 suite；`--` 后面的参数原样传给对应 suite。
- `python -m bench.cli profile smoke`
  跑一组很小的 synthetic smoke benchmark，用于确认入口和依赖可用。
- `python -m bench.cli profile local --input-dir <image-dir> --input-mode images --frames 10`
  跑本机常用组合，并使用指定真实图片目录。
- `python -m bench.cli profile local --input-dir <cache-dir> --input-mode cache --frames 10`
  跑本机常用组合，并使用指定 raw cache。
- `python -m bench.cli run pipeline.all -- --input-dir <image-dir>`
  推荐的 workflow 总览入口。走 YAML/meta/preflight/runtime planner/executor，
  用于评估本机运行真实 DAG compute workflow 的耗时。
- `python -m bench.cli run pipeline.workflow -- --input-dir <image-dir>`
  与 `pipeline.all` 等价；语义上表示 YAML/DAG workflow benchmark。
- `python -m bench.cli run pipeline.compute -- --input-dir <image-dir>`
  跑代表性的独立生产计算路径，包括对齐和叠加；用于看具体计算路径收益。
- `python -m bench.cli run pipeline.alignment -- --input-dir <image-dir>`
  只看完整生产对齐链路时使用；终端打印 pipeline 总耗时，单方法运行时也会打印阶段耗时。

`bench.cli` 是 suite 级调度器，负责发现入口、统一 profile 和聚合 JSON；
它不是完整 DAG pipeline profiler。真正按 DAG 节点采样的 profiler 后续会单独放在
pipeline/engine 侧设计。

当前 profile：

| profile | 输入策略 | 覆盖内容 |
|---------|----------|----------|
| `smoke` | 固定 synthetic 小图；忽略 `--input-dir/--input-mode/--frames` | 最小依赖与入口可用性检查 |
| `local` | 使用传入的 `--input-dir/--input-mode`；未传输入时遵循各 suite 默认搜索/生成策略，alignment 默认生成 starfield | kernel 代表 case、alignment 分阶段、remap/homography 单帧口径 |

真实 DAG workflow benchmark 需要显式运行。日常在不同机器上测试整体性能时，
优先使用：

```bash
python -m bench.cli run pipeline.all -- --input-dir <image-dir>
```

这个命令默认 `frames=10, warmup=0, repeat=1`，终端只打印每条 workflow 的一次
总耗时 `mean/min/max`。它会跑 `pipeline.all` 里的全部默认 case；
真实大图上可能耗时较长。只跑部分路径时用 `--cases`：

```bash
python -m bench.cli run pipeline.all -- --input-dir <image-dir> --cases stack_mean,stack_sigma_clip
```

如果需要重复采样，再显式加 `--repeat`：

```bash
python -m bench.cli run pipeline.all -- --input-dir <image-dir> --repeat 3
```

如果要强制禁用 custom-op compiled/CUDA 路径，测试 fallback 口径：

```bash
python -m bench.cli run pipeline.all -- --input-dir <image-dir> --backend numpy
python -m bench.cli run pipeline.compute -- --input-dir <image-dir> --backend numpy
```

`--backend` 是 `pipeline.all` / `pipeline.workflow` / `pipeline.compute` 的参数。
`pipeline.alignment` 没有这个参数；它直接走生产 wrapper 的默认选择。如果需要临时禁用
custom-op，可以在命令外设置 `HNW_CUSTOM_OPS_FALLBACK=numpy`。

`pipeline.all` 当前覆盖两类 YAML/DAG workflow：

- `base/stacker.meta.yaml` compute workflow：输入帧在 benchmark 准备阶段读入内存，
  然后通过 engine 执行，完整经过 meta route、preflight、runtime planner 和 executor。
  这类 case 不计入 `ImgDataLoaderOp` 和 `ImageSaveOp` 文件 IO。
- 顶层 file workflow：通过 `stack.meta.yaml`、`startrail.meta.yaml`、
  `calibration_stack.meta.yaml` 等顶层 DAG 执行，计入 loader / saver / executor。

需要看具体计算路径时使用 `pipeline.compute`。

只看对齐链路时用：

```bash
python -m bench.cli run pipeline.alignment -- --input-dir <image-dir>
python -m bench.cli run pipeline.alignment -- --input-dir <image-dir> --method all
python -m bench.cli run pipeline.alignment -- --input-mode synthetic --frames 10 --height 2048 --width 3072 --method homography
```

`pipeline.all` 是 engine workflow 总览入口。终端只打印每条 workflow 的
`mean/min/max`，输出 shape/dtype 等信息保留在 JSON report 中。`--cases all`
运行默认稳定 workflow；`--cases everything` 会额外包含对真实星点和足够图像尺寸
有要求的 `sky_ground_mean`。当前 case：

| case | 覆盖 workflow |
|------|---------------|
| `stack_mean` | `base/stacker.meta.yaml` → `MeanStackerOp` |
| `stack_median` | `base/stacker.meta.yaml` → `median_stack_core.yaml` |
| `stack_sigma_clip` | `base/stacker.meta.yaml` → `sigma_clip_fused.yaml` |
| `stack_huber_mean` | `base/stacker.meta.yaml` → `huber_mean.yaml` |
| `stack_max` | `base/stacker.meta.yaml` → `TrailStackerOp` |
| `stack_file_mean` / `stack_file_median` | `stack.meta.yaml` 顶层 loader → stacker → saver |
| `stack_file_sigma_clip` / `stack_file_huber_mean` | `stack.meta.yaml` 顶层迭代 stacker workflow |
| `startrail_fifo` / `startrail_mix` | `startrail.meta.yaml` 顶层星轨 workflow |
| `calibration_mean` / `calibration_median` | `calibration_stack.meta.yaml`，bias/dark/flat 默认 none |
| `calibration_sigma_clip` / `calibration_huber_mean` | `calibration_stack.meta.yaml` 主帧迭代叠加 |
| `sky_ground_mean` | `sky_ground_stack.meta.yaml`，生成临时 sky/ground mask；需真实星点或足够大的 synthetic 输入 |

`pipeline.compute` 是独立生产计算路径总览。这些 case 直接调用生产组件或
custom-op wrapper；`stack_*` 不消费对齐输出。终端只打印每条路径的
`mean/min/max`，详细阶段和输出信息保留在 JSON report 中。当前 case：

| case | 覆盖路径 |
|------|----------|
| `alignment_homography` | 星点检测、匹配、单应性估计、`cv2.warpPerspective` |
| `alignment_camera_model` | 星点检测、匹配、相机模型优化、`camera_model_remap` |
| `stack_max` / `stack_min` | `MaxMerger` / `MinMerger` |
| `stack_mean` | `MeanMerger` + `fgp_accumulate` |
| `stack_mean_weighted` | 带标量权重的 `MeanMerger(int_weight=True)` |
| `stack_mean_masked` | 带空间 mask 的 `MeanMerger` |
| `stack_median` | `median_reduce_chunk` 中位数叠加 |
| `stack_sigma_clip_fused` | fused sigma-clip chunk kernel |
| `stack_huber` | `HuberWeightedMerger` 鲁棒均值 |
| `noise_equalization` | 最大值叠加噪声均匀化，统计量使用 sigma-clipped stats |
| `star_mask_threshold` / `star_mask_dog` | 星点 mask 检测 |
| `star_shrink` | 缩星核心路径 |
| `satellite_clean_window` | 卫星线清理窗口中位数核心 smoke，不含星点匹配 |
| `warp_homography` | 单独 homography warp |
| `remap_camera_model` | 单独 camera-model remap，默认使用轻微 focal 差异 |

`pipeline.alignment` 走生产对齐逻辑：`make_geometry/detect_star_points`、
`match_star_pairs`、`optimize_alignment`、`camera_model_remap` 或
`cv2.warpPerspective`。custom-op 后端由生产 wrapper 自己选择；如果本机 CUDA
可用并且当前路径支持 CUDA，就会按生产逻辑使用 CUDA，否则走已有 CPU/OpenCV fallback。
`--method all` 会顺序跑 `homography` 和 `camera_model` 两条主要对齐路径；
终端只打印两条 pipeline 的 `mean/min/max`，阶段明细保留在 JSON `results` 中。
单独运行 `--method homography` 或 `--method camera_model` 时，终端会同时打印阶段耗时。
输入选择遵循统一 benchmark 规则：显式 `--input-dir` 优先，其次扫描
`bench/data/cache`、`bench/data/input`、`bench/data/generated`，最后才使用
synthetic starfield。

### CPU

- `python -m bench.cpu.kernels`
  算法内核微基准。默认运行已注册的代表性 kernel case，覆盖 stack、FGP、sigma clip、median/filter、alignment matching、wavelet/detection 等热点；需要精确对比时用 `--cases` 指定成对 backend。
- `python -m bench.cpu.max_stack`
  大尺寸 `max` 专项 benchmark。比较单进程 `NumPy in-place stream`、多进程 `NumPy local-reduce`、`custom op OpenMP stream`。
- `python -m bench.cpu.fgp_accumulate`
  大尺寸 `FastGaussianParam` 统计累加专项 benchmark。比较 Python、NumPy、`custom op OpenMP` 路径。
- `python -m bench.cpu.mean_stack`
  `fgp_accumulate` 的兼容 shim。
- `python -m bench.cpu.alignment`
  对齐 pipeline 分阶段 benchmark。当前覆盖 synthetic starfield / image-dir 输入下的 `detect / features / geometry / matching / warp / homography pipeline / optimization / remap / camera-model pipeline`。

### GPU

- `python -m bench.gpu.original_remap`
  当前 camera-model remap 口径集合。覆盖 `NumPy grid`、fused `camera_model_remap` custom-op、`cv2.remap` 与原主线路径。
- `python -m bench.gpu.original_homography`
  纯 homography warp 的 CPU 基线。只测 `cv2.warpPerspective`，不含 detect / features / match。
- `python -m bench.cli run gpu.sigma_clip_chunk`
  fused sigma-clip chunk 的 host-in/host-out 口径。默认对比 OpenMP CPU 与 CUDA backend；
  CUDA 不可用时会在结果中标记 skipped。

### 数据工具

- `python -m bench.data_tools.generate_array_cache`
  生成 raw cache，供 benchmark 直接加载，避免重复图片解码。
- `python -m bench.data_tools.generate_dataset`
  生成图片目录输入，主要用于 smoke test 和输入扫描链路验证。
- `python -m bench.data_tools.generate_starfield_dataset`
  生成合成星点图数据集，供对齐 benchmark 或检测 smoke test 使用。
- `bench.data_tools.starfield`
  合成星点图公共 helper。生成带平移/轻旋转/噪声的 synthetic starfield 序列。

## 安装建议

benchmark 的最小依赖放在 [bench/requirements.txt](./requirements.txt)。

当前最小运行依赖只有：

- `numpy`
- `loguru`
- `opencv-python`

如果要本地跑测试或编译 `hoshicore._custom_op._C`，直接安装仓库根目录的 [requirements.txt](../requirements.txt) 即可。

## 输入数据

benchmark 默认 `--input-mode auto`，会优先使用本地数据，最后才生成 synthetic 输入。

扫描策略是类型优先：

1. 在输入 root 下查找 raw cache
2. 在输入 root 下查找图片
3. 使用当前 suite 自己的 synthetic 输入

输入 root 的来源是：

1. 显式 `--input-dir`
2. 未传 `--input-dir` 时依次使用 `bench/data/cache/`、`bench/data/input/`、
   `bench/data/generated/`

输入模式说明：

- `auto`
  先扫 `raw_cache`，再扫 `images`，最后回退到当前 suite 的 `synthetic`。
- `cache`
  推荐的默认性能输入。适合重复 benchmark，不需要图片解码。
- `images`
  保留给 smoke test 或输入链路检查，建议只保留小规模图片集。
- `synthetic`
  强制使用当前 suite 的生成输入。

如果显式传了 `--input-dir`，`auto` 只会在该路径下查找 raw cache 或图片；
找不到会直接报错，不会悄悄回退 synthetic。这样可以避免“以为测了真实图，
实际测了生成数据”的情况。

不同 suite 的 synthetic 语义不同：

| 入口 | 输入语义 |
|------|----------|
| `bench.cpu.kernels` / `max_stack` / `fgp_accumulate` | 共用 raw cache/images/random ndarray |
| `bench.cpu.alignment` | 共用 raw cache/images；fallback 为 synthetic starfield |
| `bench.gpu.original_remap` | 共用 raw cache/images；fallback 为 synthetic camera image/config |
| `bench.gpu.original_homography` | 共用 raw cache/images；fallback 为 synthetic image/homography |
| `pipeline.all` | 共用 raw cache/images/random ndarray |
| `pipeline.compute` | 共用 raw cache/images；fallback 为 synthetic starfield |
| `pipeline.alignment` | 共用 raw cache/images；fallback 为 synthetic starfield |

如果你希望“命令行写的小尺寸 synthetic 就一定测 synthetic”，显式传：

```bash
--input-mode synthetic
```

生成 raw cache：

```bash
python -m bench.data_tools.generate_array_cache --name max_u8_100x24mp_cache --frames 100 --height 4000 --width 6000 --dtype uint8
python -m bench.data_tools.generate_array_cache --name max_u8_1000x24mp_from_images --input-dir bench/data/generated/max_u8_100x24mp_jpg --frames 1000
```

生成图片目录：

```bash
python -m bench.data_tools.generate_dataset --name max_u8_100x24mp_jpg --frames 100 --height 4000 --width 6000 --dtype uint8 --format jpg
```

生成对齐用合成星点图：

```bash
python -m bench.data_tools.generate_starfield_dataset --name align_u16_32f --frames 32 --height 2048 --width 3072 --dtype uint16 --stars 1200 --format tif
```

## 运行建议

建议按下面顺序使用：

1. `python -m bench.cli run pipeline.all -- --input-dir <image-dir>`
   先看真实 YAML/DAG workflow 在本机的一次端到端耗时。默认后端是 `auto`，
   会按生产 wrapper 选择 compiled/CUDA/CPU fallback；需要禁用 custom-op
   时加 `--backend numpy`。
2. `python -m bench.cli run pipeline.compute -- --input-dir <image-dir>`
   再看不经过 DAG engine 的代表性生产计算路径，便于区分 engine workflow
   开销和核心计算路径耗时。
3. `python -m bench.cli run pipeline.alignment -- --input-dir <image-dir> --method all`
   单独确认对齐链路的 `homography` 和 `camera_model` 两条路径。
4. `python -m bench.cpu.kernels`
   先看当前 stack kernel 热点方向，也包含 `fgp_masked_mean_merge`、`sigma_clip_fused_*`、`fgp_add_partial_reduce` 的独立 `numpy / compiled` microbenchmark。
5. `python -m bench.cpu.max_stack` / `python -m bench.cpu.fgp_accumulate`
   跟进 `max / fgp_accumulate` 的 CPU kernel 优化。
6. `python -m bench.cpu.alignment`
   先确认对齐链阶段热点，避免直接 GPU 化错对象。
7. `python -m bench.gpu.original_remap`
   测当前正式 camera-model remap 路径，包括 CUDA custom-op、CPU custom-op、OpenCV reference 等口径。
8. `python -m bench.gpu.original_homography`
   单独判断纯 homography warp 的 OpenCV baseline。

示例：

```bash
python -m bench.cli run pipeline.all -- --input-dir <image-dir>
python -m bench.cli run pipeline.all -- --input-dir <image-dir> --backend numpy
python -m bench.cli run pipeline.compute -- --input-dir <image-dir> --cases alignment_homography,alignment_camera_model,remap_camera_model
python -m bench.cli run pipeline.alignment -- --input-dir <image-dir> --method all
python -m bench.cpu.kernels --frames 128 --height 1080 --width 1920 --dtype uint16 --input-mode synthetic
python -m bench.cpu.kernels --frames 64 --height 2048 --width 3072 --dtype uint16 --input-mode synthetic --cases fgp_masked_mean_merge_stream_numpy,fgp_masked_mean_merge_stream_compiled,sigma_clip_fused_merge_stream_numpy,sigma_clip_fused_merge_stream_compiled,sigma_clip_fused_masked_merge_stream_numpy,sigma_clip_fused_masked_merge_stream_compiled,fgp_add_partial_reduce_numpy,fgp_add_partial_reduce_compiled
python -m bench.cpu.kernels --frames 16 --height 2048 --width 3072 --dtype uint16 --input-mode synthetic --cases median_reduce_chunk_numpy,median_reduce_chunk_compiled --chunk-rows 32
python -m bench.cpu.max_stack --frames 100 --height 4000 --width 6000 --dtype uint8 --workers 4 --openmp-threads auto --input-mode cache
python -m bench.cpu.fgp_accumulate --frames 100 --height 4000 --width 6000 --dtype uint8 --openmp-threads auto --input-mode cache
python -m bench.cpu.max_stack --frames 1000 --input-dir bench/data/cache/max_u8_100x24mp_cache --output-json bench-results/max-1000.json
python -m bench.cpu.alignment --frames 16 --height 2048 --width 3072 --stars 1200 --input-mode synthetic
python -m bench.cpu.alignment --frames 16 --input-dir bench/data/generated/align_u16_32f --input-mode images
python -m bench.cpu.alignment --frames 16 --input-dir bench/data/generated/align_u16_32f --input-mode images --cases detect_stream match_stream
python -m bench.cpu.alignment --frames 16 --input-dir bench/data/generated/align_u16_32f --input-mode images --cases detect_wavelet_stream detect_extract_stream remap_stream
python -m bench.gpu.original_remap --input-dir <image-dir> --input-mode images --cases custom_op_fused opencv_remap --skip-accuracy
python -m bench.gpu.original_homography --input-dir <image-dir> --input-mode images --cases opencv_warp
```

## 输出格式

benchmark 默认在终端打印简短摘要，主要展示各 case 的运行时间。

如果传入 `--output-json /path/to/report.json`，会写出完整 JSON 文件。

`repeat` 只用于重复采样并取均值；`mean_sec / min_sec / max_sec`
始终表示一次 case 调用的耗时。对 stream 或 batch case，终端会额外打印
`per_frame_mean`、`per_chunk_mean` 等派生值，JSON 中也会写入对应的
`min_per_unit_sec / mean_per_unit_sec / median_per_unit_sec / max_per_unit_sec`
和 `unit_count`。

输出字段主要包括：

- `suite`
- `env`
- `custom_ops`
- `config`
- `input_source`
- `results`

`results` 中每个 case 通常包含：

- `samples_sec`
- `min_sec`
- `mean_sec`
- `median_sec`
- `max_sec`
- `unit` / `unit_count` / `*_per_unit_sec`（仅 stream 或 batch case）
