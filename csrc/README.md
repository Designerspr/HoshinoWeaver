# C++ Custom Ops

`csrc/` 负责 `hoshicore._custom_op._C` 的原生实现与本地构建。

设计边界：

- 只覆盖 custom-op 原生层
- Python 公共入口保持为 `hoshicore._custom_op`
- 运行时通过 backend registry 选择 CUDA/OpenMP/NumPy；只有明确的 runtime
  unavailable 或类型化 resource exhausted 才按各 wrapper 契约 fallback
- 构建统一走 `CMake + Ninja`

## 目录

```text
csrc/
  build_ops.py
  CMakePresets.json
  CMakeLists.txt
  module.cpp
  common/
  ops/
    cpu/
      fgp/
      max/
      median/
      noise/
      sigma_clip/
    cuda/
```

职责：

- `module.cpp`
  pybind11 模块入口，注册 `_C` 内的算子
- `ops/cpu/<name>/`
  单个算子的 compiled CPU 实现与绑定；OpenMP 是可选并行能力
- `ops/cuda/<name>/`
  单个算子的 CUDA 实现与绑定
- `build_ops.py`
  统一本地构建入口
- `CMakeLists.txt` / `CMakePresets.json`
  custom-op 的 CMake/Ninja 构建骨架

## 构建环境

不要求必须使用 conda。只要当前解释器环境里具备以下组件即可：

- Python 3.10+
- `pybind11`
- Python development headers
- `cmake`
- `ninja`
- 可用的 C/C++ 编译器

如果当前环境缺少 CMake/Ninja，可额外安装：

```bash
pip install -r csrc/requirements.txt
```

这个文件只补 native 构建工具，不重复根目录 `requirements.txt` 里的项目依赖；
编译器和 OpenMP runtime 仍然是系统工具链要求。

## 调用方式

已激活目标环境时，直接运行：

```bash
python csrc/build_ops.py
```

`build_ops.py` 会把当前 `sys.executable` 传给 CMake 的 `Python3_EXECUTABLE`。
需要显式指定解释器时，直接用解释器路径调用脚本即可。

## 常用命令

默认构建：

```bash
python csrc/build_ops.py
```

显式系统 GCC：

```bash
python csrc/build_ops.py --cc /usr/bin/gcc --cxx /usr/bin/g++
```

CUDA 构建：

```bash
python csrc/build_ops.py --cuda --cc /usr/bin/gcc --cxx /usr/bin/g++
```

显式解释器：

```bash
/path/to/python csrc/build_ops.py --cc /usr/bin/gcc --cxx /usr/bin/g++
```

只看配置：

```bash
python csrc/build_ops.py --dry-run
```

## C++ / CUDA 格式检查

仓库固定使用 `clang-format 20.1.8`，规则位于根目录 `.clang-format`：

```bash
# 只检查
python csrc/check_format.py

# 应用格式化
python csrc/check_format.py --fix
```

检查覆盖 `csrc/` 下的 C++/CUDA 源码并排除生成的 build tree；CI 使用同一入口。

## 常用参数

- `--preset`
  指定 CMake preset；日常路径通常不需要手动传
- `--cc / --cxx`
  显式指定编译器
- `--cuda`
  打开 CUDA 构建，自动根据平台选择对应 CUDA preset
- `--compiler gcc|clang|msvc|auto`
  选择编译器家族
- `--no-openmp`
  关闭 OpenMP
- `--march-native`
  启用本机 CPU 指令集优化；只建议本机 benchmark 使用
- `--lto`
  启用 LTO
- `--omp-simd`
  为支持的 kernel 启用显式 OpenMP SIMD pragma
- `--clean`
  清理旧产物后全量重编
- `--verbose-build`
  打印完整 backend 输出

## 输出与中间产物

- 扩展模块输出到 `hoshicore/_custom_op/_C*.so|.pyd`
- `cmake` 中间产物默认在 `csrc/build/<preset>/`
- CUDA custom-op 包含 fused `camera_model_remap`，支持 perspective/fisheye
  的四种源目标投影组合，并与 OpenMP backend 共用双精度投影数学
- CUDA custom-op 还包含 alignment descriptor cosine 双向最近邻、standalone
  wavelet、star-shrink、sigma-clip/Huber chunk 与 Norma fused pixel/component
  detection；生产候选均声明并消费显存模型
- OpenMP custom-op 为 Norma fused pixel/component detection 与 star-shrink DoG
  mask 提供无 GPU 的原生路径；detection 仍在 host 侧复用同一套 OpenCV contour
  几何和候选过滤语义

## 打包约定

最终发布为 PyInstaller single-folder 模式。CUDA runtime 静态链接到 `_C`；
OpenMP 在 Linux/Windows 为动态链接（PyInstaller 自动收集），macOS 为静态链接。

### 链接策略

| 依赖 | 链接方式 | 说明 |
|------|----------|------|
| OpenMP (Linux + GCC) | 动态（`libgomp.so`） | 系统自带，PyInstaller 自动收集到产物目录 |
| OpenMP (Windows + MSVC) | 动态（`vcomp140.dll`） | VC++ Redistributable 组件，PyInstaller 自动收集 |
| OpenMP (macOS) | 静态（Homebrew `libomp.a`） | 需先 `brew install libomp`，编译时自动检测并静态链接 |
| CUDA runtime | 静态（`cudart_static`） | 消除 `libcudart.so` / `cudart64_*.dll` 依赖 |

### 验证依赖

```bash
# Linux — 确认 cudart 已静态链接，libgomp 为动态（PyInstaller 会收集）
ldd hoshicore/_custom_op/_C*.so | grep -E "cudart|gomp"
# 预期：只看到 libgomp.so，不应出现 libcudart.so

# Windows (Developer Command Prompt)
dumpbin /dependents hoshicore/_custom_op/_C*.pyd
# 预期：出现 VCOMP140.DLL（正常），不应出现 cudart64_*.dll

# 跨平台最小 frozen-package smoke；会清理 Python/编译器/CUDA 环境路径后启动
python csrc/verify_packaged_custom_ops.py
```

### PyInstaller 收集

OpenMP 动态库由 PyInstaller 自动收集。spec file 确保 `_C` 模块被包含即可：

```python
# PyInstaller spec — hiddenimports 确保 _C 被打包
hiddenimports=['hoshicore._custom_op._C']
```

若后续使用了 cuBLAS/cuFFT 等额外 CUDA 库且无法静态链接，再按需添加到
`binaries=[]` 中。

## GPU 构建说明

### CUDA

构建 CUDA 算子需要安装 CUDA Toolkit，版本选择支持本机 GPU 的即可。
最终发布建议 12.8+，覆盖本项目支持的 NVIDIA GPU 架构。

发布构建与运行时支持的最低 compute capability 为 6.0（Pascal）。消费级产品可
近似理解为 GTX 10 系及更新型号；Maxwell（GTX 900 系）及更早设备会在 runtime
probe 阶段被明确排除并使用 CPU fallback，不会启动不兼容的 CUDA kernel。发布
架构列表显式包含 `sm_60` 与消费级 GTX 10 系使用的 `sm_61`；使用 CUDA Toolkit
12.8+ 的 Windows 发布构建还会生成 `sm_100`、`sm_101`、`sm_120`，覆盖数据中心
与 GeForce RTX 50 系 Blackwell。较旧 toolkit 不会尝试编译这些目标。

构建路径：

- Linux: `python csrc/build_ops.py --cuda`
- Windows: 使用 `windows-msvc-cuda` preset

Preset 参考：

| Preset | 平台 |
|--------|------|
| `linux-gcc-cuda` | Linux |
| `windows-msvc-cuda` | Windows |

### 其他 GPU 后端（规划中）

| 方向 | 状态 |
|------|------|
| AMD (ROCm/HIP) | 待评估 |
| macOS (Metal/MPS) | 待设计 |
| Vulkan (compute shader) | 待评估 |

所有 GPU 后端保持 CPU fallback 语义不变。

运行时可通过 `HNW_CUSTOM_OPS_FALLBACK=cpu` 禁用 CUDA 并保留 OpenMP，或在
pipeline 启动前调用 `hoshicore._custom_op.set_backend_preference("cpu")`。传入
`"auto"` 恢复自动选择，传入 `None` 则恢复由环境变量控制。该接口为进程级设置，
暂不包含 GUI wiring。

### CUDA host-I/O workspace 与显存治理

生产 CUDA host-I/O kernel 统一通过 `common/cuda_host_io_workspace.cuh` 获取
current-device stream 与可复用 device/pinned buffer。workspace 的 high-water 是
一次 logical operation 内仍然存活的 lease 总量，不等同于 thread-local cache 的
retained bytes；`cuda_host_io_cache_info()` 分别报告两者。

Python compiled 边界在调用 kernel 前消费 `cuda_memory.py` 中同一 logical op 的
estimator，并通过 admission 检查当前 free VRAM、headroom 和进程内 reservation。
admission 拒绝与实际 `cudaMalloc` OOM 都使用类型化
`CudaResourceExhaustedError`；无设备/驱动不可用使用独立的
`CudaRuntimeUnavailableError`。普通 OOM 字符串、invalid pointer、launch failure
等非类型化错误不会被静默 fallback。

新增 `cuda_host_io` candidate 时必须提供由 wrapper/planner 实际消费的
`memory_model`。registry 会把声明的类别解析到 `cuda_memory.py` 的 logical-op
model；non-chunk wrapper 统一经 `cuda_memory_estimate()` 获取 estimator，chunk
wrapper/planner 统一经 `cuda_chunk_memory_model()` 获取模型。第三方扩展可显式写
deferral reason 保持兼容，但仓库内建候选的 registry 校验不接受长期豁免。

---

普通用户只需安装 NVIDIA 驱动（>= 570.65，对应发布构建的 CUDA 12.8），不需要 CUDA Toolkit。
驱动版本要求与 GPU 型号无关，只要驱动足够新且设备 compute capability >= 6.0
即可；驱动过旧、无 NVIDIA GPU 或设备架构过旧时自动回退 CPU。受支持设备上的
`NoKernelImageForDevice` / PTX JIT 错误仍作为构建或运行时错误传播，避免掩盖损坏的
发布包。

## 新增算子

最小流程：

1. 在 `csrc/ops/cpu/<name>/` 新增 CPU `.h/.cpp`
2. 在 `CMakeLists.txt` 新增 static library target 并链接到 `_C`
3. 在 `module.cpp` 中注册 `bind_*_ops(m)`
4. 在 `hoshicore/_custom_op/ops/` 增加 Python 包装与 numpy fallback
5. 在 `hoshicore/_custom_op/backend_registry.py` 注册 `BackendCandidate`
6. 在 `hoshicore/_custom_op/api.py` + `__init__.py` 导出
7. 补 focused tests（`tests/custom_ops/test_<logical_op>.py`）
8. 补 microbenchmark（`bench/cpu/kernels.py`）

`BackendCandidate` 用于运行时判断当前包是否实际包含 native kernel。若 CMake /
打包未包含某个 kernel，Python wrapper 必须回到 numpy/CPU fallback；缺失 native
backend 只能影响性能，不能影响 public API 可用性。

CUDA 算子沿用同样流程，但额外需要：

1. 在 `csrc/ops/cuda/<name>/` 新增 CUDA `.h/.cpp/.cu`，并在 `CMakeLists.txt` 的 `HNW_ENABLE_CUDA` 分支接入
2. 保持 CPU fallback 语义不变
3. `.cpp` 绑定文件需 `#include "common/compat.h"`（MSVC `ssize_t` 兼容）
4. 在 `BackendCandidate` 中标注对应 backend（如 `cuda_host_io`）和 build flag（如 `cuda`）
5. 为生产 CUDA candidate 增加 estimator/admission、workspace high-water 测试和类型化资源回退测试

## 参考

- 构建架构与 preset 设计详见 [CMAKE_MIGRATION.md](./CMAKE_MIGRATION.md)
