"""Sequence bundle adjustment for a same-camera star-field capture.

The public result deliberately contains geometry only. Applying that geometry
to image streams belongs to downstream ops such as ``BundleReferenceRemapOp``.

The solve has four stages:

1. Match nearby frame pairs selected by ``pair_offsets``.
2. Keep the graph component connected to the reference frame and propagate
   pairwise rotations to obtain an initial pose for every connected frame.
3. Jointly optimize one shared camera model and one rotation per frame.
4. Reject high-residual edges, refit when necessary, then describe frames that
   were disconnected from the final graph.
"""
import dataclasses
import enum
from collections import defaultdict, deque
from typing import Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .detection import DetectedStars
from .frame_align import (DEFAULT_BOOTSTRAP_SCALES, AlignmentCameraCandidate,
                          solve_star_alignment)
from .optimization import CameraOptimizationPolicy
from .types import (BaseCameraModel, CameraModel, Distortion, FisheyeCameraModel,
                    FisheyeDistortion)


class BundleAdjustmentError(RuntimeError):
    """The requested sequence does not have a trustworthy global solution."""


class FrameAlignmentStatus(str, enum.Enum):
    SOLVED = "solved"
    EXCLUDED = "excluded"


@dataclasses.dataclass(frozen=True)
class BundleFrame:
    """Compact star observation retained while collecting a sequence."""

    index: int
    stars: DetectedStars
    candidate: AlignmentCameraCandidate


@dataclasses.dataclass(frozen=True)
class FrameAlignment:
    index: int
    status: FrameAlignmentStatus
    rotation_ref_to_src: Optional[NDArray[np.float64]]
    pose_source: str
    residual_p90_rad: Optional[float] = None
    incident_edge_count: int = 0
    reason: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class BAAlignmentPlan:
    """Immutable geometry hand-off from BA collection to stream integration."""

    reference_frame_index: int
    shared_camera: BaseCameraModel
    frames: tuple[FrameAlignment, ...]
    accepted_edge_count: int
    rejected_edge_count: int
    active_camera_parameter_count: int
    observability_condition: Optional[float]

    def frame(self, index: int) -> FrameAlignment:
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"frame index {index} is outside this plan")
        return self.frames[index]


@dataclasses.dataclass
class _BundleEdge:
    """Pairwise observations and an initial rotation from first to second."""

    first_index: int
    second_index: int
    first_pts: NDArray[np.float64]
    second_pts: NDArray[np.float64]
    initial_rotation: NDArray[np.float64]
    error: Optional[str] = None
    selected_scale: float = 1.0


_SCALE_PROBE_EDGE_COUNT = 3
_ROTATION_ONLY_POLICY = CameraOptimizationPolicy(False, False, False, 0)


def _camera_parameter_count(policy: CameraOptimizationPolicy) -> int:
    return ((1 if policy.optimize_focal else 0)
            + (policy.n_dist if policy.optimize_distortion else 0)
            + (2 if policy.optimize_principal_point else 0))


def _pack_camera_parameters(camera: BaseCameraModel,
                            policy: CameraOptimizationPolicy) -> NDArray[np.float64]:
    """Pack shared camera variables before the per-frame rotation variables.

    Focal length and principal point use offsets from the initial camera;
    distortion coefficients use their current model values directly.
    """
    parts: list[NDArray[np.float64]] = []
    if policy.optimize_focal:
        parts.append(np.zeros(1, dtype=np.float64))
    if policy.optimize_distortion:
        parts.append(np.asarray(camera.distortion.to_opt_params(policy.n_dist),
                                dtype=np.float64))
    if policy.optimize_principal_point:
        parts.append(np.zeros(2, dtype=np.float64))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def _camera_from_parameters(camera: BaseCameraModel,
                            policy: CameraOptimizationPolicy,
                            values: NDArray[np.float64]) -> BaseCameraModel:
    offset = 0
    intrinsics = camera.intrinsics
    if policy.optimize_focal:
        intrinsics = intrinsics.with_focal_length(
            intrinsics.focal_length_mm * (1.0 + float(values[offset])))
        offset += 1
    distortion = camera.distortion
    if policy.optimize_distortion:
        solved = np.asarray(values[offset:offset + policy.n_dist], dtype=np.float64)
        offset += policy.n_dist
        if isinstance(camera, FisheyeCameraModel):
            full = camera.distortion.to_cv2().copy()
            full[:len(solved)] = solved
            distortion = FisheyeDistortion.from_array(full)
        else:
            full = camera.distortion.to_cv2().copy()
            full[:len(solved)] = solved
            distortion = Distortion.from_cv2(full)
    if policy.optimize_principal_point:
        cx, cy = camera.intrinsics.principal_point_px
        intrinsics = intrinsics.with_principal_point(
            cx + float(values[offset]), cy + float(values[offset + 1]))
    return camera.with_intrinsics(intrinsics).with_distortion(distortion)


def _rotation_from_rvec(rvec: NDArray[np.float64]) -> NDArray[np.float64]:
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]


def _rvec_from_rotation(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    return cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0].reshape(3)


def _readonly_rotation(rotation: Optional[NDArray[np.float64]]) -> Optional[NDArray[np.float64]]:
    if rotation is None:
        return None
    result = np.asarray(rotation, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _connected_component(edges: Sequence[_BundleEdge], root: int) -> set[int]:
    neighbours: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        neighbours[edge.first_index].add(edge.second_index)
        neighbours[edge.second_index].add(edge.first_index)
    seen = {root}
    pending = deque([root])
    while pending:
        current = pending.popleft()
        for candidate in neighbours[current]:
            if candidate not in seen:
                seen.add(candidate)
                pending.append(candidate)
    return seen


def _initial_rotations(edges: Sequence[_BundleEdge], root: int,
                       component: set[int]) -> dict[int, NDArray[np.float64]]:
    """Propagate pairwise rotations over a BFS spanning tree.

    The returned rotation for frame ``i`` maps reference-frame rays to frame
    ``i`` rays. All graph edges, not only this implicit tree, are used later by
    the global optimizer.
    """
    graph: dict[int, list[tuple[int, NDArray[np.float64]]]] = defaultdict(list)
    for edge in edges:
        graph[edge.first_index].append((edge.second_index, edge.initial_rotation))
        graph[edge.second_index].append((edge.first_index, edge.initial_rotation.T))
    rotations = {root: np.eye(3, dtype=np.float64)}
    pending = deque([root])
    while pending:
        first = pending.popleft()
        for second, first_to_second in graph[first]:
            if second in rotations or second not in component:
                continue
            rotations[second] = first_to_second @ rotations[first]
            pending.append(second)
    if rotations.keys() != component:
        raise BundleAdjustmentError("reference component has incomplete pose initialization")
    return rotations


def _edge_residuals(edge: _BundleEdge, rotations: dict[int, NDArray[np.float64]],
                    camera: BaseCameraModel) -> NDArray[np.float64]:
    first = camera.unproject(edge.first_pts)
    second = camera.unproject(edge.second_pts)
    predicted = (rotations[edge.second_index] @ rotations[edge.first_index].T
                 @ first.T).T
    dots = np.sum(predicted * second, axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def _coverage_precheck(edges: Sequence[_BundleEdge], policy: CameraOptimizationPolicy,
                       camera: BaseCameraModel) -> None:
    if not policy.optimize_distortion:
        return
    all_points = np.concatenate([edge.first_pts for edge in edges])
    h, w = camera.intrinsics.image_height_px, camera.intrinsics.image_width_px
    center = np.array([w / 2.0, h / 2.0])
    radial = np.linalg.norm(all_points - center, axis=1)
    outer = np.count_nonzero(radial >= 0.66 * np.hypot(w / 2.0, h / 2.0))
    if len(all_points) < 20 or outer < max(4, int(0.08 * len(all_points))):
        raise BundleAdjustmentError(
            "shared distortion is not observable: insufficient outer-field coverage")


def _camera_observability(jacobian: NDArray[np.float64], camera_width: int) -> Optional[float]:
    """Check shared-camera observability after removing pose directions."""
    if camera_width == 0:
        return None
    camera_j = np.asarray(jacobian[:, :camera_width], dtype=np.float64)
    pose_j = np.asarray(jacobian[:, camera_width:], dtype=np.float64)
    if pose_j.size:
        q_mat, _ = np.linalg.qr(pose_j, mode="reduced")
        camera_j = camera_j - q_mat @ (q_mat.T @ camera_j)
    singular = np.linalg.svd(camera_j, compute_uv=False)
    if len(singular) < camera_width or singular[-1] <= 1e-10:
        raise BundleAdjustmentError("shared camera parameters are rank deficient")
    condition = float(singular[0] / singular[-1])
    if not np.isfinite(condition) or condition > 1e8:
        raise BundleAdjustmentError(
            f"shared camera parameters are ill-conditioned ({condition:.3e})")
    return condition


def _make_edge(first: BundleFrame, second: BundleFrame,
               random_seed: int | None,
               bootstrap_scales: Sequence[float] = DEFAULT_BOOTSTRAP_SCALES,
               ) -> _BundleEdge:
    """Match one edge and refine only its initial relative rotation."""
    try:
        ref_candidate = dataclasses.replace(
            first.candidate, optimization_policy=_ROTATION_ONLY_POLICY)
        src_candidate = dataclasses.replace(
            second.candidate, optimization_policy=_ROTATION_ONLY_POLICY)
        alignment, match = solve_star_alignment(
            first.stars,
            second.stars,
            ref_candidate,
            src_candidate,
            bootstrap_scales=tuple(bootstrap_scales),
            same_camera=True,
            use_asterism_bootstrap=True,
            random_seed=random_seed,
        )
        if len(match.pair_idx) < 6:
            raise ValueError(f"only {len(match.pair_idx)} matched stars")
        base_focal = first.candidate.camera.intrinsics.focal_length_mm
        selected_scale = alignment.ref_camera.intrinsics.focal_length_mm / base_focal
        return _BundleEdge(
            first.index, second.index, match.ref_pts, match.src_pts,
            np.asarray(alignment.rotation_ref_to_src, dtype=np.float64),
            selected_scale=float(selected_scale))
    except Exception as exc:
        return _BundleEdge(first.index, second.index,
                           np.empty((0, 2), dtype=np.float64),
                           np.empty((0, 2), dtype=np.float64),
                           np.eye(3), str(exc))


def _ordered_scales(votes: dict[float, int]) -> tuple[float, ...]:
    """Try the most reliable sequence scale first; prefer 1.0 on ties."""
    original_order = {scale: index for index, scale in enumerate(votes)}
    return tuple(sorted(
        votes,
        key=lambda scale: (-votes[scale], abs(scale - 1.0),
                           original_order[scale]),
    ))


def _vote_for_scale(votes: dict[float, int], selected: float) -> None:
    nearest = min(votes, key=lambda scale: abs(scale - selected))
    votes[nearest] += 1


def _sequence_scales(frames: Sequence[BundleFrame]) -> tuple[float, ...]:
    """Keep reliable perspective EXIF/provided sequences at scale 1.0."""
    reliable_perspective = all(
        not isinstance(frame.candidate.camera, FisheyeCameraModel)
        and frame.candidate.init_source in ("exif", "provided")
        for frame in frames)
    return ((1.0,) if reliable_perspective else
            tuple(float(scale) for scale in DEFAULT_BOOTSTRAP_SCALES))


def _build_edges(frames: Sequence[BundleFrame], offsets: Sequence[int],
                 random_seed: int | None) -> list[_BundleEdge]:
    """Build edges with a short full-search probe, then preferred-scale reuse.

    A successful single-scale solve already passed the matcher's rotation
    quality checks. It is accepted without a vote because the other scales
    were not compared. Failed preferred-scale solves evaluate every remaining
    scale and vote for the best successful fallback.
    """
    votes = {scale: 0 for scale in _sequence_scales(frames)}
    successful_probes = 0
    edges = []
    for pos, first in enumerate(frames):
        for offset in offsets:
            if pos + offset >= len(frames):
                continue
            second = frames[pos + offset]
            ordered_scales = _ordered_scales(votes)
            if successful_probes < _SCALE_PROBE_EDGE_COUNT:
                edge = _make_edge(first, second, random_seed, ordered_scales)
                if edge.error is None:
                    _vote_for_scale(votes, edge.selected_scale)
                    successful_probes += 1
            else:
                edge = _make_edge(first, second, random_seed,
                                  ordered_scales[:1])
                if edge.error is not None and len(ordered_scales) > 1:
                    edge = _make_edge(first, second, random_seed,
                                      ordered_scales[1:])
                    if edge.error is None:
                        _vote_for_scale(votes, edge.selected_scale)
            edges.append(edge)
    return edges


def build_bundle_plan(frames: Sequence[BundleFrame], reference_frame_index: int,
                      *,
                      pair_offsets: Sequence[int] = (1, 2, 4),
                      random_seed: int | None = 0,
                      max_nfev: int = 300) -> BAAlignmentPlan:
    """Build a robust same-camera geometry plan from collected frames.

    Frames contain detected stars rather than source pixels. Integration remaps
    the original sequence separately after the plan is final.
    """
    if not frames:
        raise BundleAdjustmentError("cannot bundle-adjust an empty sequence")
    # A duplicate index is overwritten by the dict comprehension, so the
    # length comparison also validates that frame indices are unique.
    by_index = {frame.index: frame for frame in frames}
    if len(by_index) != len(frames) or reference_frame_index not in by_index:
        raise BundleAdjustmentError("reference_frame_index must identify one collected frame")
    # All frames share the reference camera's projection family, dimensions,
    # and optimization policy. The global solve below owns the shared values.
    reference = by_index[reference_frame_index]
    camera = reference.candidate.camera
    policy = reference.candidate.optimization_policy
    if any(type(frame.candidate.camera) is not type(camera) for frame in frames):
        raise BundleAdjustmentError("sequence BA requires one camera projection family")
    if any(frame.candidate.camera.intrinsics.image_width_px != camera.intrinsics.image_width_px
           or frame.candidate.camera.intrinsics.image_height_px != camera.intrinsics.image_height_px
           for frame in frames):
        raise BundleAdjustmentError("sequence BA requires a single image geometry")

    # Offsets address positions in the time-ordered sequence, not numeric frame
    # index differences. With (1, 2, 4), the graph is a chain plus skip edges.
    offsets = tuple(sorted({int(value) for value in pair_offsets if int(value) > 0}))
    if not offsets:
        raise ValueError("pair_offsets must contain at least one positive offset")
    ordered = sorted(frames, key=lambda item: item.index)
    edges = _build_edges(ordered, offsets, random_seed)
    accepted = [edge for edge in edges if edge.error is None]

    # Only the reference-connected component has a defined reference-relative
    # pose. Failed or disconnected frames are handled after the global solve.
    component = _connected_component(accepted, reference_frame_index)
    if len(component) < 2:
        raise BundleAdjustmentError("reference frame has no reliable sequence connection")
    accepted = [edge for edge in accepted if edge.first_index in component and edge.second_index in component]
    _coverage_precheck(accepted, policy, camera)
    initial = _initial_rotations(accepted, reference_frame_index, component)
    variable_indices = [index for index in sorted(component) if index != reference_frame_index]
    camera0 = _pack_camera_parameters(camera, policy)

    # Optimizer layout: [shared camera parameters, 3D rotation vectors...].
    # The reference rotation is fixed to identity to remove gauge freedom.
    x0 = np.concatenate((camera0, *[_rvec_from_rotation(initial[index]) for index in variable_indices]))

    def unpack(values):
        solved_camera = _camera_from_parameters(camera, policy, values[:len(camera0)])
        rotations = {reference_frame_index: np.eye(3, dtype=np.float64)}
        offset = len(camera0)
        for index in variable_indices:
            rotations[index] = _rotation_from_rvec(values[offset:offset + 3])
            offset += 3
        return solved_camera, rotations

    def residual(values, selected_edges):
        solved_camera, rotations = unpack(values)
        values_out = []
        for edge in selected_edges:
            first = solved_camera.unproject(edge.first_pts)
            second = solved_camera.unproject(edge.second_pts)
            # Convert the first-frame ray through the shared reference system
            # into the second frame, then compare the two unit directions.
            predicted = (rotations[edge.second_index] @ rotations[edge.first_index].T @ first.T).T
            values_out.append(np.cross(predicted, second).reshape(-1))
        return np.concatenate(values_out)

    # First pass uses every successfully matched edge in the component. Huber
    # loss limits individual bad pairs before edge-level rejection is possible.
    initial_residual = residual(x0, accepted)
    scale = max(float(np.median(np.abs(initial_residual))) * 2.0, 1e-6)
    fit = least_squares(residual, x0, args=(accepted,), method="trf", loss="huber",
                        f_scale=scale, max_nfev=max_nfev)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise BundleAdjustmentError(f"global bundle optimization failed: {fit.message}")
    solved_camera, rotations = unpack(fit.x)

    # Reject complete edges only when their robust edge statistic is an obvious
    # outlier.  This prevents one structured correspondence field from steering
    # the shared camera model.
    edge_p90 = np.array([np.percentile(_edge_residuals(edge, rotations, solved_camera), 90)
                         for edge in accepted])
    median = float(np.median(edge_p90))
    mad = float(np.median(np.abs(edge_p90 - median))) * 1.4826
    cutoff = max(median + 3.0 * mad, np.deg2rad(0.15))
    retained = [edge for edge, p90 in zip(accepted, edge_p90) if p90 <= cutoff]
    if len(retained) < len(accepted):
        # Recompute connectivity because removing one edge may detach frames,
        # then refit all surviving poses and the same shared camera parameters.
        component = _connected_component(retained, reference_frame_index)
        retained = [edge for edge in retained if edge.first_index in component and edge.second_index in component]
        if len(component) < 2:
            raise BundleAdjustmentError("outlier rejection disconnected the reference frame")
        variable_indices = [index for index in sorted(component) if index != reference_frame_index]
        initial = {index: rotations[index] for index in component}
        x0 = np.concatenate((fit.x[:len(camera0)], *[_rvec_from_rotation(initial[index]) for index in variable_indices]))
        fit = least_squares(residual, x0, args=(retained,), method="trf", loss="huber",
                            f_scale=scale, max_nfev=max_nfev)
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise BundleAdjustmentError(f"refined bundle optimization failed: {fit.message}")
        solved_camera, rotations = unpack(fit.x)
    _coverage_precheck(retained, policy, solved_camera)
    condition = _camera_observability(fit.jac, len(camera0))

    edge_by_frame: dict[int, list[float]] = defaultdict(list)
    for edge in retained:
        p90 = float(np.percentile(_edge_residuals(edge, rotations, solved_camera), 90))
        edge_by_frame[edge.first_index].append(p90)
        edge_by_frame[edge.second_index].append(p90)
    frame_results = []
    for index in sorted(by_index):
        solved = index in rotations
        pose = rotations.get(index)
        residual_p90 = (float(np.median(edge_by_frame[index]))
                        if edge_by_frame[index] else None)
        frame_results.append(FrameAlignment(
            index=index,
            status=(FrameAlignmentStatus.SOLVED if solved
                    else FrameAlignmentStatus.EXCLUDED),
            rotation_ref_to_src=_readonly_rotation(pose),
            pose_source="bundle" if solved else "none",
            residual_p90_rad=residual_p90,
            incident_edge_count=len(edge_by_frame[index]),
            reason=(None if solved else
                    "frame is outside the reference-connected solve graph")))
    retained_keys = {(edge.first_index, edge.second_index) for edge in retained}
    accepted_edge_count = len(retained_keys)
    rejected_edge_count = len(edges) - accepted_edge_count
    return BAAlignmentPlan(
        reference_frame_index=reference_frame_index, shared_camera=solved_camera,
        frames=tuple(frame_results), accepted_edge_count=accepted_edge_count,
        rejected_edge_count=rejected_edge_count,
        active_camera_parameter_count=len(camera0),
        observability_condition=condition)
