import numpy as np
import tifffile

from hoshicore.engine.build import ValidatedDag
from hoshicore.engine.preflight import PreflightReport, ResourceEstimate
from hoshicore.engine.runtime_plan import apply_runtime_plan, plan_runtime
import hoshicore.engine.runtime_plan as runtime_plan_module
import hoshicore._custom_op.backend_registry as backend_registry
from hoshicore.ops.base import BaseOp
from hoshicore.ops.sigma_clip_ops import (
    HuberMeanIteratorOp,
    MedianReduceOp,
    SigmaClipFusedChunkOp,
    SigmaClipIteratorOp,
)


def _make_dag() -> ValidatedDag:
    return ValidatedDag(
        nodes={
            "median": {
                "op": "MedianReduceOp",
                "configs": {"chunk_rows": "configs.chunk_rows"},
                "outputs": {"result": {"type": "image"}},
            }
        },
        global_inputs={},
        global_configs={},
        output_links={},
        node_deps={},
        exec_order=["median"],
    )


def _make_no_chunk_dag() -> ValidatedDag:
    return ValidatedDag(
        nodes={
            "noop": {
                "op": "NoChunkOp",
                "configs": {},
                "outputs": {"result": {"type": "image"}},
            }
        },
        global_inputs={},
        global_configs={},
        output_links={},
        node_deps={},
        exec_order=["noop"],
    )


def _make_multi_chunk_dag() -> ValidatedDag:
    return ValidatedDag(
        nodes={
            "chunk_a": {"op": "FixedChunkOpA", "configs": {}},
            "chunk_b": {"op": "FixedChunkOpB", "configs": {}},
        },
        global_inputs={},
        global_configs={},
        output_links={},
        node_deps={},
        exec_order=["chunk_a", "chunk_b"],
    )


def _make_cuda_chunk_dag() -> ValidatedDag:
    return ValidatedDag(
        nodes={
            "chunk": {"op": "FixedCudaChunkOp", "configs": {}},
        },
        global_inputs={},
        global_configs={},
        output_links={},
        node_deps={},
        exec_order=["chunk"],
    )


def _report(non_chunk_mem: int = 0) -> PreflightReport:
    return PreflightReport(
        estimate=ResourceEstimate(0, 0),
        available_memory_bytes=0,
        available_disk_bytes=0,
        non_chunk_mem=non_chunk_mem,
    )


def _mock_available_memory(monkeypatch, budget: int, non_chunk_mem: int = 0) -> None:
    available = int(
        (runtime_plan_module.MEMORY_FIXED_OVERHEAD + budget + non_chunk_mem) /
        runtime_plan_module.MEMORY_SAFETY_FACTOR) + 16

    class FakeVMem:
        pass

    FakeVMem.available = available
    monkeypatch.setattr(
        runtime_plan_module.psutil,
        "virtual_memory",
        lambda: FakeVMem())


def _mock_cuda_memory(monkeypatch, budget: int, available: bool = True) -> None:
    if not available:
        monkeypatch.setattr(
            runtime_plan_module,
            "cuda_memory_info",
            lambda: {"available": False, "reason": "mock unavailable"},
        )
        return
    free_bytes = int(
        (runtime_plan_module.MEMORY_FIXED_OVERHEAD + budget) /
        runtime_plan_module.MEMORY_SAFETY_FACTOR) + 16
    monkeypatch.setattr(
        runtime_plan_module,
        "cuda_memory_info",
        lambda: {
            "available": True,
            "free_bytes": free_bytes,
            "total_bytes": free_bytes * 2,
        },
    )


def _mock_cuda_backend(
    monkeypatch,
    logical_op: str,
    *,
    cpu_available: bool = False,
) -> None:
    candidate = backend_registry.BackendCandidate(
        logical_op,
        "cuda_host_io",
        f"{logical_op}_cuda",
        priority=10,
        build_flag="cuda",
    )
    cpu_candidate = backend_registry.BackendCandidate(
        logical_op,
        "openmp_cpu",
        f"{logical_op}_cpu",
    )
    selection = backend_registry.BackendSelection(candidate, object())
    cpu_selection = backend_registry.BackendSelection(cpu_candidate, object())

    def select(logical_op_arg: str, preference: str = "auto", **kwargs):
        if preference == "numpy":
            return backend_registry.BackendSelection(
                None, None, "numpy backend forced by preference")
        if logical_op_arg == logical_op:
            build_info = kwargs.get("build_info")
            if isinstance(build_info, dict) and not build_info.get("cuda", True):
                if cpu_available:
                    return cpu_selection
                return backend_registry.BackendSelection(
                    None, object(), "mock CPU backend unavailable")
            return selection
        return backend_registry.BackendSelection(None, None, "mock unavailable")

    monkeypatch.setattr(runtime_plan_module, "select_backend", select)


class NoChunkOp(BaseOp):
    pass


class FixedChunkOpA(BaseOp):
    CHUNK_PLANNED = True

    @classmethod
    def chunk_cost_per_row(cls, n_frames, row_bytes, dtype_bytes):
        _ = n_frames, row_bytes, dtype_bytes
        return 40


class FixedChunkOpB(BaseOp):
    CHUNK_PLANNED = True

    @classmethod
    def chunk_cost_per_row(cls, n_frames, row_bytes, dtype_bytes):
        _ = n_frames, row_bytes, dtype_bytes
        return 60


class FixedCudaChunkOp(BaseOp):
    CHUNK_PLANNED = True
    BACKEND_LOGICAL_OP = "fixed_cuda_chunk"

    @classmethod
    def chunk_cost_per_row(cls, n_frames, row_bytes, dtype_bytes):
        _ = n_frames, row_bytes, dtype_bytes
        return 100


class FixedCudaFallbackChunkOp(BaseOp):
    CHUNK_PLANNED = True
    BACKEND_LOGICAL_OP = "fixed_cuda_fallback_chunk"

    @classmethod
    def chunk_cost_per_row(cls, n_frames, row_bytes, dtype_bytes):
        _ = n_frames, row_bytes, dtype_bytes
        return 100

    @classmethod
    def chunk_cost_per_row_for_backend(cls, backend, n_frames, row_bytes, dtype_bytes):
        _ = n_frames, row_bytes, dtype_bytes
        if backend == "openmp_cpu":
            return 200
        if backend == "numpy":
            return 400
        return 100


def _median_registry() -> dict[str, type[BaseOp]]:
    return {"MedianReduceOp": MedianReduceOp}


def test_runtime_planner_disabled_returns_empty(tmp_path):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": False},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
    )

    assert plan.config_overrides == {}
    assert plan.decisions == []


def test_runtime_planner_sets_chunk_rows_when_enabled(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    non_chunk_mem = 1000
    monkeypatch.setattr(
        runtime_plan_module.psutil,
        "virtual_memory",
        lambda: type("FakeVMem", (), {"available": 100_000})())

    plan = plan_runtime(
        _make_dag(),
        {
            "runtime_planner": True,
        },
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(non_chunk_mem),
    )

    assert plan.config_overrides["chunk_rows"] == 1
    assert plan.decisions[0].key == "chunk_rows"


def test_runtime_planner_uses_preflight_formula(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    non_chunk_mem = 1000
    # 灰度图 row_bytes = 10 * 1 * 2，Median cost = (4 + 1) * 20 = 100 bytes/row.
    _mock_available_memory(monkeypatch, budget=12800, non_chunk_mem=non_chunk_mem)

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(non_chunk_mem),
    )

    assert plan.config_overrides["chunk_rows"] == 128


def test_runtime_planner_sums_multiple_chunk_op_costs(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    # 两个 planned op 的 cost 分别为 40 和 60，总成本 100 bytes/row。
    _mock_available_memory(monkeypatch, budget=12800)

    plan = plan_runtime(
        _make_multi_chunk_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry={
            "FixedChunkOpA": FixedChunkOpA,
            "FixedChunkOpB": FixedChunkOpB,
        },
        preflight_report=_report(),
    )

    assert plan.config_overrides["chunk_rows"] == 128


def test_runtime_planner_uses_cuda_budget_for_cuda_chunk_op(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    monkeypatch.delenv("HNW_CUSTOM_OPS_FALLBACK", raising=False)
    _mock_available_memory(monkeypatch, budget=12800)
    _mock_cuda_memory(monkeypatch, budget=6400)
    _mock_cuda_backend(monkeypatch, "fixed_cuda_chunk")

    plan = plan_runtime(
        _make_cuda_chunk_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry={"FixedCudaChunkOp": FixedCudaChunkOp},
        preflight_report=_report(),
    )

    assert plan.config_overrides["chunk_rows"] == 64
    assert "gpu=" in plan.decisions[0].reason


def test_runtime_planner_ignores_unavailable_cuda_budget(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    monkeypatch.delenv("HNW_CUSTOM_OPS_FALLBACK", raising=False)
    _mock_available_memory(monkeypatch, budget=12800)
    _mock_cuda_memory(monkeypatch, budget=0, available=False)
    _mock_cuda_backend(monkeypatch, "fixed_cuda_chunk")

    plan = plan_runtime(
        _make_cuda_chunk_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry={"FixedCudaChunkOp": FixedCudaChunkOp},
        preflight_report=_report(),
    )

    assert plan.config_overrides["chunk_rows"] == 128
    assert "gpu=" not in plan.decisions[0].reason


def test_runtime_planner_uses_cpu_cost_when_cuda_runtime_unavailable(
        tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    monkeypatch.delenv("HNW_CUSTOM_OPS_FALLBACK", raising=False)
    _mock_available_memory(monkeypatch, budget=12800)
    _mock_cuda_memory(monkeypatch, budget=0, available=False)
    _mock_cuda_backend(
        monkeypatch,
        "fixed_cuda_fallback_chunk",
        cpu_available=True,
    )
    dag = ValidatedDag(
        nodes={
            "chunk": {"op": "FixedCudaFallbackChunkOp", "configs": {}},
        },
        global_inputs={},
        global_configs={},
        output_links={},
        node_deps={},
        exec_order=["chunk"],
    )

    plan = plan_runtime(
        dag,
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry={"FixedCudaFallbackChunkOp": FixedCudaFallbackChunkOp},
        preflight_report=_report(),
    )

    assert plan.config_overrides["chunk_rows"] == 64
    assert "gpu=" not in plan.decisions[0].reason


def test_runtime_planner_ignores_cuda_budget_when_numpy_forced(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    _mock_available_memory(monkeypatch, budget=12800)
    _mock_cuda_memory(monkeypatch, budget=6400)
    _mock_cuda_backend(monkeypatch, "fixed_cuda_chunk")
    monkeypatch.setenv("HNW_CUSTOM_OPS_FALLBACK", "numpy")

    plan = plan_runtime(
        _make_cuda_chunk_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry={"FixedCudaChunkOp": FixedCudaChunkOp},
        preflight_report=_report(),
    )

    assert plan.config_overrides["chunk_rows"] == 128
    assert "gpu=" not in plan.decisions[0].reason


def test_runtime_planner_uses_numpy_chunk_cost_when_numpy_forced(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    _mock_available_memory(monkeypatch, budget=12800)
    monkeypatch.setenv("HNW_CUSTOM_OPS_FALLBACK", "numpy")

    dag = ValidatedDag(
        nodes={
            "chunk": {"op": "SigmaClipFusedChunkOp", "configs": {}},
        },
        global_inputs={},
        global_configs={},
        output_links={},
        node_deps={},
        exec_order=["chunk"],
    )

    plan = plan_runtime(
        dag,
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry={"SigmaClipFusedChunkOp": SigmaClipFusedChunkOp},
        preflight_report=_report(),
    )

    assert plan.config_overrides["chunk_rows"] == 8


def test_runtime_planner_clamps_to_default_max_chunk_rows(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((4000, 10), dtype=np.uint16))
    _mock_available_memory(monkeypatch, budget=10_000_000_000)

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(),
    )

    assert plan.config_overrides["chunk_rows"] == runtime_plan_module.DEFAULT_MAX_CHUNK_ROWS


def test_chunk_cost_per_row_formulas():
    n_frames = 5
    row_bytes = 120
    dtype_bytes = 2
    float64_row = row_bytes // dtype_bytes * 8

    assert MedianReduceOp.chunk_cost_per_row(
        n_frames, row_bytes, dtype_bytes) == (n_frames + 1) * row_bytes
    assert SigmaClipFusedChunkOp.chunk_cost_per_row(
        n_frames, row_bytes, dtype_bytes) == (
            2 * n_frames * row_bytes +
            n_frames * row_bytes +
            n_frames * row_bytes // dtype_bytes +
            3 * float64_row
        )
    assert SigmaClipFusedChunkOp.chunk_cost_per_row_for_backend(
        "numpy", n_frames, row_bytes, dtype_bytes) == (
            2 * n_frames * row_bytes +
            2 * n_frames * float64_row +
            2 * n_frames * (row_bytes // dtype_bytes) +
            3 * float64_row +
            6 * float64_row
        )
    assert SigmaClipIteratorOp.chunk_cost_per_row(
        n_frames, row_bytes, dtype_bytes) == (
            2 * n_frames * row_bytes +
            n_frames * row_bytes +
            n_frames * row_bytes // dtype_bytes +
            3 * float64_row +
            3 * float64_row
        )
    assert HuberMeanIteratorOp.chunk_cost_per_row(
        n_frames, row_bytes, dtype_bytes) == (
            2 * n_frames * row_bytes +
            4 * float64_row
        )


def test_runtime_planner_requires_preflight_report(tmp_path):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
    )

    assert plan.config_overrides == {}


def test_runtime_planner_keeps_explicit_chunk_rows(tmp_path):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": True, "chunk_rows": 32},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(),
        explicit_config_keys={"chunk_rows"},
    )

    assert plan.config_overrides == {}


def test_runtime_planner_allows_auto_chunk_rows(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))
    monkeypatch.setattr(
        runtime_plan_module.psutil,
        "virtual_memory",
        lambda: type("FakeVMem", (), {"available": 100_000})())

    plan = plan_runtime(
        _make_dag(),
        {
            "runtime_planner": True,
            "chunk_rows": "auto",
        },
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(),
        explicit_config_keys={"chunk_rows"},
    )

    configs = {"chunk_rows": "auto"}
    apply_runtime_plan(plan, configs)
    assert configs["chunk_rows"] == 1


def test_runtime_planner_no_chunk_planned_op_returns_empty(tmp_path):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))

    plan = plan_runtime(
        _make_no_chunk_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        {"NoChunkOp": NoChunkOp},
        preflight_report=_report(),
    )

    assert plan.config_overrides == {}


def test_runtime_planner_clamps_to_min_when_budget_is_exhausted(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))
    monkeypatch.setattr(
        runtime_plan_module.psutil,
        "virtual_memory",
        lambda: type("FakeVMem", (), {"available": 100_000})())

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(non_chunk_mem=10_000),
    )

    assert plan.config_overrides["chunk_rows"] == 1
