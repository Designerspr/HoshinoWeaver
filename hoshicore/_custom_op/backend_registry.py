"""Runtime backend candidates for custom-op wrappers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Collection, Mapping

from hoshicore._custom_op._dispatch import (
    cuda_memory_info,
    is_cuda_runtime_unavailable_error,
    load_compiled_module,
)

ModuleLoader = Callable[[], tuple[Any | None, str | None]]
CudaProbe = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class BackendCandidate:
    logical_op: str
    backend: str
    kernel_name: str
    placement: str = "host_to_host"
    priority: int = 0
    requires_contiguous: bool = True
    dtypes: tuple[str, ...] = ()
    fallback: str = "numpy"
    build_flag: str | None = None


@dataclass(frozen=True)
class BackendDecision:
    logical_op: str
    backend: str
    kernel_name: str | None
    placement: str
    fallback: str | None
    native: bool
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_op": self.logical_op,
            "backend": self.backend,
            "kernel_name": self.kernel_name,
            "placement": self.placement,
            "fallback": self.fallback,
            "native": self.native,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class BackendSelection:
    candidate: BackendCandidate | None
    module: Any | None
    reason: str | None = None
    reason_code: str = "unspecified"
    decision: BackendDecision | None = None

    @property
    def native(self) -> bool:
        return self.candidate is not None and self.module is not None

    @property
    def backend(self) -> str:
        if self.candidate is None:
            return "numpy"
        return self.candidate.backend

    def to_decision(self, logical_op: str) -> BackendDecision:
        candidate = self.candidate
        if candidate is None:
            return BackendDecision(
                logical_op=logical_op,
                backend="numpy",
                kernel_name=None,
                placement="host_to_host",
                fallback=None,
                native=False,
                reason_code=self.reason_code,
            )
        return BackendDecision(
            logical_op=logical_op,
            backend=candidate.backend,
            kernel_name=candidate.kernel_name,
            placement=candidate.placement,
            fallback=candidate.fallback,
            native=self.native,
            reason_code=self.reason_code,
        )


_CANDIDATES: tuple[BackendCandidate, ...] = (
    BackendCandidate("extract_point_features", "openmp_cpu", "extract_point_features"),
    BackendCandidate("find_initial_match", "openmp_cpu", "find_initial_match"),
    BackendCandidate("calibration_subtract", "openmp_cpu", "calibration_subtract"),
    BackendCandidate("calibration_divide", "openmp_cpu", "calibration_divide"),
    BackendCandidate("fgp_accumulate", "openmp_cpu", "fgp_accumulate"),
    BackendCandidate("fgp_masked_mean_merge", "openmp_cpu", "fgp_masked_mean_merge"),
    BackendCandidate("median_filter_2d", "openmp_cpu", "median_filter_2d"),
    BackendCandidate("huber_weighted_accumulate", "openmp_cpu", "huber_weighted_accumulate"),
    BackendCandidate("sigma_clip_fused_merge", "openmp_cpu", "sigma_clip_fused_merge"),
    BackendCandidate(
        "sigma_clip_fused_masked_merge",
        "openmp_cpu",
        "sigma_clip_fused_masked_merge",
    ),
    BackendCandidate("max_combine", "openmp_cpu", "max_combine"),
    BackendCandidate("threshold_max_merge", "openmp_cpu", "threshold_max_merge"),
    BackendCandidate("median_reduce_chunk", "openmp_cpu", "median_reduce_chunk"),
    BackendCandidate("equalize_noise_correct", "openmp_cpu", "equalize_noise_correct"),
    BackendCandidate("noise_fill_local_mean", "openmp_cpu", "noise_fill_local_mean"),
    BackendCandidate("noise_equalization_params", "openmp_cpu", "noise_equalization_params"),
    BackendCandidate(
        "star_shrink_process",
        "cuda_host_io",
        "star_shrink_process_cuda",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
    ),
    BackendCandidate("star_shrink_process", "openmp_cpu", "star_shrink_process"),
    BackendCandidate("star_shrink_detect_mask", "openmp_cpu", "star_shrink_detect_mask"),
    BackendCandidate(
        "star_mask_dog",
        "cuda_host_io",
        "star_mask_dog_cuda",
        priority=10,
        build_flag="cuda",
    ),
    BackendCandidate(
        "star_shrink_dog_process",
        "cuda_host_io",
        "star_shrink_dog_process_cuda",
        priority=10,
        build_flag="cuda",
    ),
    BackendCandidate("sigma_clip_iterative_chunk", "openmp_cpu", "sigma_clip_iterative_chunk"),
    BackendCandidate(
        "huber_weighted_chunk",
        "cuda_host_io",
        "huber_weighted_chunk_cuda",
        priority=10,
        build_flag="cuda",
    ),
    BackendCandidate(
        "sigma_clip_fused_chunk",
        "cuda_host_io",
        "sigma_clip_fused_chunk_cuda",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
    ),
    BackendCandidate("sigma_clip_fused_chunk", "openmp_cpu", "sigma_clip_fused_chunk"),
    # Experimental norma detector path; production fallback remains OpenCV contour.
    BackendCandidate(
        "star_detect_connected_components_candidates",
        "openmp_cpu",
        "star_detect_connected_components_candidates",
    ),
    BackendCandidate(
        "wavelet_dec_rec",
        "cuda_host_io",
        "wavelet_dec_rec_cuda_core",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
    ),
    BackendCandidate("wavelet_dec_rec", "openmp_cpu", "wavelet_dec_rec_cpu"),
    BackendCandidate(
        "wavelet_dec_rec_cuda_core",
        "cuda_host_io",
        "wavelet_dec_rec_cuda_core",
        build_flag="cuda",
    ),
    BackendCandidate(
        "star_detect_full_connected_components",
        "cuda_host_io",
        "star_detect_full_connected_components_core",
        build_flag="cuda",
    ),
    BackendCandidate(
        "camera_model_remap",
        "cuda_host_io",
        "camera_model_remap",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
    ),
    BackendCandidate("camera_model_remap", "openmp_cpu", "camera_model_remap_cpu"),
)

_CANDIDATES_BY_OP: dict[str, tuple[BackendCandidate, ...]] = {}
for _candidate in _CANDIDATES:
    _CANDIDATES_BY_OP.setdefault(_candidate.logical_op, ())
    _CANDIDATES_BY_OP[_candidate.logical_op] += (_candidate,)


def registered_backend_candidates(logical_op: str | None = None) -> tuple[BackendCandidate, ...]:
    if logical_op is None:
        return _CANDIDATES
    return _CANDIDATES_BY_OP.get(logical_op, ())


def select_backend(
    logical_op: str,
    preference: str = "auto",
    *,
    load_module: ModuleLoader = load_compiled_module,
    build_info: Mapping[str, Any] | None = None,
    exclude_backends: Collection[str] = (),
) -> BackendSelection:
    if preference == "numpy":
        return BackendSelection(
            None,
            None,
            "numpy backend forced by preference",
            "forced_numpy",
        )

    candidates = registered_backend_candidates(logical_op)
    if not candidates:
        return BackendSelection(
            None,
            None,
            f"no backend candidate registered for {logical_op}",
            "logical_op_unregistered",
        )

    module, module_error = load_module()
    if module is None:
        return BackendSelection(
            None,
            None,
            module_error or "compiled backend unavailable",
            "compiled_module_unavailable",
        )

    missing_kernel: str | None = None
    missing_build_flag: str | None = None
    excluded = frozenset(exclude_backends)
    for candidate in sorted(candidates, key=lambda item: item.priority, reverse=True):
        if candidate.backend in excluded:
            continue
        if not _has_static_attr(module, candidate.kernel_name):
            missing_kernel = candidate.kernel_name
            continue
        if candidate.build_flag is not None:
            info = build_info if build_info is not None else _module_build_info(module)
            if info and not info.get(candidate.build_flag):
                missing_build_flag = candidate.build_flag
                continue
        return BackendSelection(candidate, module, None, "selected_native")

    if missing_kernel is not None:
        return BackendSelection(
            None,
            module,
            f"compiled backend missing kernel: {missing_kernel}",
            "kernel_unavailable",
        )
    if missing_build_flag is not None:
        return BackendSelection(
            None,
            module,
            f"compiled backend missing build flag: {missing_build_flag}",
            "build_flag_unavailable",
        )
    if excluded:
        excluded_names = ", ".join(sorted(excluded))
        return BackendSelection(
            None,
            module,
            f"no available backend candidate for {logical_op} after excluding: "
            f"{excluded_names}",
            "backends_excluded",
        )
    return BackendSelection(
        None,
        module,
        f"no available backend candidate for {logical_op}",
        "backend_unavailable",
    )


def resolve_backend(
    logical_op: str,
    preference: str = "auto",
    *,
    load_module: ModuleLoader = load_compiled_module,
    build_info: Mapping[str, Any] | None = None,
    exclude_backends: Collection[str] = (),
    cuda_probe: CudaProbe = cuda_memory_info,
) -> BackendSelection:
    selection = select_backend(
        logical_op,
        preference,
        load_module=load_module,
        build_info=build_info,
        exclude_backends=exclude_backends,
    )
    decision = selection.to_decision(logical_op)
    if (
        not selection.native
        or selection.candidate is None
        or selection.candidate.backend != "cuda_host_io"
    ):
        return BackendSelection(
            selection.candidate,
            selection.module,
            selection.reason,
            selection.reason_code,
            decision,
        )

    probe = cuda_probe()
    if probe.get("available") is True:
        return BackendSelection(
            selection.candidate,
            selection.module,
            selection.reason,
            selection.reason_code,
            decision,
        )
    status = probe.get("status")
    if status != "explicitly_unavailable":
        raise RuntimeError("CUDA runtime probe failed without an unavailable status")

    excluded = set(exclude_backends)
    excluded.add("cuda_host_io")
    fallback_selection = select_backend(
        logical_op,
        "auto",
        load_module=load_module,
        build_info=build_info,
        exclude_backends=excluded,
    )
    fallback_decision = fallback_selection.to_decision(logical_op)
    fallback_decision = BackendDecision(
        logical_op=fallback_decision.logical_op,
        backend=fallback_decision.backend,
        kernel_name=fallback_decision.kernel_name,
        placement=fallback_decision.placement,
        fallback=fallback_decision.fallback,
        native=fallback_decision.native,
        reason_code="cuda_runtime_unavailable",
    )
    return BackendSelection(
        fallback_selection.candidate,
        fallback_selection.module,
        probe.get("reason") or fallback_selection.reason,
        "cuda_runtime_unavailable",
        fallback_decision,
    )


def resolve_after_runtime_unavailable(
    logical_op: str,
    failed_backend: str,
    exc: RuntimeError,
    *,
    load_module: ModuleLoader = load_compiled_module,
    build_info: Mapping[str, Any] | None = None,
) -> BackendSelection:
    if failed_backend != "cuda_host_io" or not is_cuda_runtime_unavailable_error(exc):
        raise exc
    selection = resolve_backend(
        logical_op,
        "auto",
        load_module=load_module,
        build_info=build_info,
        exclude_backends={failed_backend},
    )
    decision = selection.to_decision(logical_op)
    decision = BackendDecision(
        logical_op=decision.logical_op,
        backend=decision.backend,
        kernel_name=decision.kernel_name,
        placement=decision.placement,
        fallback=decision.fallback,
        native=decision.native,
        reason_code="cuda_runtime_unavailable",
    )
    return BackendSelection(
        selection.candidate,
        selection.module,
        str(exc),
        "cuda_runtime_unavailable",
        decision,
    )


def native_backend_available(
    logical_op: str,
    preference: str = "auto",
    *,
    load_module: ModuleLoader = load_compiled_module,
    build_info: Mapping[str, Any] | None = None,
    exclude_backends: Collection[str] = (),
) -> tuple[bool, str | None]:
    selection = select_backend(
        logical_op,
        preference,
        load_module=load_module,
        build_info=build_info,
        exclude_backends=exclude_backends,
    )
    return selection.native, selection.reason


def _module_build_info(module: Any) -> Mapping[str, Any]:
    if not _has_static_attr(module, "build_info"):
        return {}
    payload = module.build_info()
    return payload if isinstance(payload, Mapping) else {}


def _has_static_attr(value: Any, name: str) -> bool:
    try:
        inspect.getattr_static(value, name)
    except AttributeError:
        return False
    return True
