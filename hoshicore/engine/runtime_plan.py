"""Runtime execution planning helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import psutil
from loguru import logger

from .._custom_op._dispatch import cuda_memory_info
from .._custom_op._dispatch import fallback_preference
from .._custom_op.backend_registry import BackendDecision, resolve_backend
from ..component.image_io import peek_shape
from ..ops.base import BaseOp
from .build import ValidatedDag
from .preflight import PreflightReport
from .registry import REGISTERED_OP

MEMORY_SAFETY_FACTOR = 0.7
MEMORY_FIXED_OVERHEAD = 200 * 1024 * 1024
DEFAULT_MIN_CHUNK_ROWS = 1
DEFAULT_MAX_CHUNK_ROWS = 1024
CHUNK_ROW_ALIGNMENT = 16
CudaProbe = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class RuntimePlanDecision:
    key: str
    value: Any
    reason: str


@dataclass(frozen=True)
class RuntimePlan:
    config_overrides: dict[str, Any] = field(default_factory=dict)
    decisions: list[RuntimePlanDecision] = field(default_factory=list)
    backend_hints: dict[str, BackendDecision] = field(default_factory=dict)


@dataclass(frozen=True)
class _ChunkPlannedNode:
    node_name: str
    op_cls: type[BaseOp]


def plan_runtime(
    dag: ValidatedDag,
    effective_configs: dict[str, Any],
    global_inputs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
    *,
    preflight_report: Optional[PreflightReport] = None,
    explicit_config_keys: set[str] | None = None,
) -> RuntimePlan:
    if not _planner_enabled(effective_configs.get("runtime_planner", False)):
        return RuntimePlan()

    registry = op_registry or REGISTERED_OP
    cuda_probe = _cached_cuda_probe()
    backend_hints = _resolve_backend_hints(dag, registry, cuda_probe)
    base_plan = RuntimePlan(backend_hints=backend_hints)
    chunk_ops = _find_chunk_planned_ops(dag, registry)
    if not chunk_ops:
        return base_plan

    explicit_keys = explicit_config_keys or set()
    current_chunk_rows = effective_configs.get("chunk_rows")
    if "chunk_rows" in explicit_keys and current_chunk_rows != "auto":
        return base_plan

    shape_info = _peek_input_shape(global_inputs)
    if shape_info is None:
        return base_plan
    shape, dtype_bytes, n_frames = shape_info
    chunk_rows, budget_info = _plan_chunk_rows(
        shape, dtype_bytes, n_frames, effective_configs,
        chunk_ops, backend_hints, preflight_report, cuda_probe)
    if chunk_rows is None:
        return base_plan

    non_chunk_mem = preflight_report.non_chunk_mem if preflight_report else 0
    budget_reason = f", budget={budget_info}" if budget_info else ""
    reason = (
        f"planned chunk_rows={chunk_rows} for shape={shape}, "
        f"frames={n_frames}, dtype_bytes={dtype_bytes}, "
        f"non_chunk_mem={non_chunk_mem}{budget_reason}"
    )
    return RuntimePlan(
        config_overrides={"chunk_rows": chunk_rows},
        decisions=[RuntimePlanDecision("chunk_rows", chunk_rows, reason)],
        backend_hints=backend_hints,
    )


def apply_runtime_plan(plan: RuntimePlan, effective_configs: dict[str, Any]) -> None:
    for key, value in plan.config_overrides.items():
        old_value = effective_configs.get(key)
        effective_configs[key] = value
        logger.info(f"[RuntimePlan] {key}: {old_value} -> {value}")
    for decision in plan.decisions:
        logger.info(f"[RuntimePlan] {decision.reason}")
    for node_name, hint in plan.backend_hints.items():
        logger.info(
            f"[RuntimePlan] node={node_name}, logical_op={hint.logical_op}, "
            f"planned_backend={hint.backend}, kernel={hint.kernel_name}, "
            f"reason_code={hint.reason_code}"
        )


def _planner_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "auto"}
    return bool(value)


def _find_chunk_planned_ops(
    dag: ValidatedDag,
    registry: dict[str, type[BaseOp]],
) -> list[_ChunkPlannedNode]:
    chunk_ops: list[_ChunkPlannedNode] = []
    for node_name in dag.exec_order:
        node_spec = dag.nodes[node_name]
        op_cls = registry.get(node_spec["op"])
        if op_cls is None:
            continue
        if getattr(op_cls, "CHUNK_PLANNED", False):
            chunk_ops.append(_ChunkPlannedNode(node_name, op_cls))
    return chunk_ops


def _resolve_backend_hints(
    dag: ValidatedDag,
    registry: dict[str, type[BaseOp]],
    cuda_probe: CudaProbe,
) -> dict[str, BackendDecision]:
    hints: dict[str, BackendDecision] = {}
    preference = fallback_preference()
    for node_name in dag.exec_order:
        node_spec = dag.nodes[node_name]
        op_cls = registry.get(node_spec["op"])
        if op_cls is None:
            continue
        logical_op = getattr(op_cls, "BACKEND_LOGICAL_OP", None)
        if not logical_op:
            continue
        selection = resolve_backend(
            logical_op,
            preference,
            cuda_probe=cuda_probe,
        )
        hints[node_name] = selection.decision or selection.to_decision(logical_op)
    return hints


def _cached_cuda_probe() -> CudaProbe:
    cached: dict[str, dict[str, Any]] = {}

    def probe() -> dict[str, Any]:
        if "value" not in cached:
            cached["value"] = cuda_memory_info()
        return cached["value"]

    return probe


def _peek_input_shape(global_inputs: dict[str, Any]) -> tuple[tuple[int, ...], int, int] | None:
    fnames = global_inputs.get("fnames")
    if not fnames or not isinstance(fnames, (list, tuple)):
        return None
    if len(fnames) == 0:
        return None
    try:
        shape, dtype_bytes = peek_shape(fnames[0])
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.warning(f"[RuntimePlan] cannot peek first frame: {exc}")
        return None
    return tuple(shape), int(dtype_bytes), len(fnames)


def _plan_chunk_rows(
    shape: tuple[int, ...],
    dtype_bytes: int,
    n_frames: int,
    configs: dict[str, Any],
    chunk_ops: list[_ChunkPlannedNode],
    backend_hints: dict[str, BackendDecision],
    preflight_report: PreflightReport | None,
    cuda_probe: CudaProbe,
) -> tuple[int | None, str | None]:
    if preflight_report is None:
        return None, None
    if len(shape) < 2 or n_frames <= 0 or dtype_bytes <= 0:
        return None, None

    height = int(shape[0])
    width = int(shape[1])
    channels = int(shape[2]) if len(shape) >= 3 else 1
    row_bytes = width * channels * dtype_bytes
    if height <= 0 or row_bytes <= 0:
        return None, None

    cost_per_row = sum(
        _chunk_cost_per_row(
            chunk_op,
            backend_hints,
            n_frames,
            row_bytes,
            dtype_bytes,
        )
        for chunk_op in chunk_ops
    )
    if cost_per_row <= 0:
        return None, None

    # preflight 负责估算非 chunk 常驻内存；planner 只拿剩余预算分配 chunk_rows。
    cpu_budget = _memory_budget_bytes() - int(preflight_report.non_chunk_mem)
    gpu_budget = _cuda_chunk_memory_budget_bytes(
        chunk_ops, backend_hints, cuda_probe)
    chunk_budget = cpu_budget if gpu_budget is None else min(cpu_budget, gpu_budget)
    if chunk_budget <= 0:
        rows = DEFAULT_MIN_CHUNK_ROWS
    else:
        rows = max(DEFAULT_MIN_CHUNK_ROWS, chunk_budget // cost_per_row)
    rows = _round_down_to_multiple(int(rows), CHUNK_ROW_ALIGNMENT)
    min_rows = _positive_int(
        configs.get("runtime_planner_min_chunk_rows"), DEFAULT_MIN_CHUNK_ROWS)
    max_rows = _positive_int(
        configs.get("runtime_planner_max_chunk_rows"), DEFAULT_MAX_CHUNK_ROWS)
    rows = max(min_rows, min(rows, max_rows, height))
    budget_info = f"cpu={cpu_budget}"
    if gpu_budget is not None:
        budget_info += f", gpu={gpu_budget}"
    return max(1, int(rows)), budget_info


def _memory_budget_bytes() -> int:
    avail_mem = psutil.virtual_memory().available
    return int(avail_mem * MEMORY_SAFETY_FACTOR) - MEMORY_FIXED_OVERHEAD


def _cuda_chunk_memory_budget_bytes(
    chunk_ops: list[_ChunkPlannedNode],
    backend_hints: dict[str, BackendDecision],
    cuda_probe: CudaProbe,
) -> int | None:
    if not _uses_cuda_host_io_chunk_backend(chunk_ops, backend_hints):
        return None
    info = cuda_probe()
    if not info.get("available"):
        return None
    try:
        free_bytes = int(info["free_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("available CUDA memory info has invalid free_bytes") from exc
    return int(free_bytes * MEMORY_SAFETY_FACTOR) - MEMORY_FIXED_OVERHEAD


def _uses_cuda_host_io_chunk_backend(
    chunk_ops: list[_ChunkPlannedNode],
    backend_hints: dict[str, BackendDecision],
) -> bool:
    for chunk_op in chunk_ops:
        hint = backend_hints.get(chunk_op.node_name)
        if hint is not None and hint.native and hint.backend == "cuda_host_io":
            return True
    return False


def _chunk_cost_per_row(
    chunk_op: _ChunkPlannedNode,
    backend_hints: dict[str, BackendDecision],
    n_frames: int,
    row_bytes: int,
    dtype_bytes: int,
) -> int:
    op_cls = chunk_op.op_cls
    cost_for_backend = getattr(op_cls, "chunk_cost_per_row_for_backend", None)
    if cost_for_backend is None:
        return int(op_cls.chunk_cost_per_row(n_frames, row_bytes, dtype_bytes))
    hint = backend_hints.get(chunk_op.node_name)
    backend = hint.backend if hint is not None and hint.native else "numpy"
    return int(cost_for_backend(backend, n_frames, row_bytes, dtype_bytes))


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if value < multiple:
        return value
    return max(multiple, (value // multiple) * multiple)
