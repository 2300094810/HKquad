"""Connection-aware algebra for kernelised N-RoSy fields.

This module deliberately separates two concerns:

* a heat-kernel evaluator supplies non-negative spatial weights; and
* this module transports and combines tangent-space N-RoSy coefficients.

The transport angle is explicit.  There is no globally unique parallel
transport between two points on a curved surface unless a path (or a
connection heat kernel) has been selected by the caller.
"""

from typing import NamedTuple, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = [
    "RosyFieldOutput",
    "QuadFieldLossOutput",
    "RosySingularityOutput",
    "aggregate_kernel_rosy_field",
    "connection_direction_loss",
    "connection_smoothness_loss",
    "detect_face_field_singularities",
    "decode_rosy_angle",
    "field_strength_loss",
    "edge_integrability_loss",
    "field_quality_loss",
    "large_direction_jump_loss",
    "log_size_smoothness_loss",
    "log_size_target_loss",
    "normalize_power_field",
    "quad_field_objective",
    "rosy_alignment_loss",
    "rosy_angular_error",
    "rotate_power_field",
]


class RosyFieldOutput(NamedTuple):
    """Outputs of a kernelised N-RoSy field evaluation.

    ``power_field`` is the primary field.  Its zeros must be preserved because
    they are potential field singularities.  ``unit_power_field`` and
    ``angle`` are decoded views and are only meaningful where ``valid`` is
    true.
    """

    power_field: Tensor
    unit_power_field: Tensor
    magnitude: Tensor
    coherence: Tensor
    angle: Tensor
    valid: Tensor
    coverage: Tensor
    target_size: Optional[Tensor]


class RosySingularityOutput(NamedTuple):
    """Quantized singularity indices of a face-based N-RoSy field.

    Boundary and otherwise invalid vertices have ``valid == False`` and a
    reported index/residual of zero.  ``residual`` is the mean physical-angle
    mismatch around the one-ring after choosing the closest symmetry branch;
    it is useful as a confidence diagnostic, not as the singularity index.
    """

    index: Tensor
    valid: Tensor
    residual: Tensor


class QuadFieldLossOutput(NamedTuple):
    """Individual differentiable terms and their weighted total."""

    total: Tensor
    direction_smoothness: Tensor
    guidance_alignment: Tensor
    boundary_alignment: Tensor
    size_smoothness: Tensor
    size_target: Tensor
    field_strength: Tensor
    integrability: Tensor
    direction_tail: Tensor
    coherence_quality: Tensor
    coverage_quality: Tensor


def normalize_power_field(
    power_field: Tensor, eps: float = 1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize a real-imaginary power field without destroying its zeros."""
    if power_field.shape[-1] != 2:
        raise ValueError("power_field must have final dimension 2 (real, imag)")

    magnitude = torch.linalg.vector_norm(power_field, dim=-1, keepdim=True)
    valid = magnitude.squeeze(-1) > eps
    unit = power_field / magnitude.clamp_min(eps)
    unit = torch.where(valid.unsqueeze(-1), unit, torch.zeros_like(unit))
    return unit, magnitude, valid


def rotate_power_field(
    power_field: Tensor,
    tangent_rotation: Tensor,
    degree: int = 4,
) -> Tensor:
    """Transport an N-RoSy power coefficient by a tangent-frame rotation.

    ``tangent_rotation`` is the rotation, in radians, experienced by an
    ordinary tangent vector.  An N-RoSy power coefficient therefore rotates
    by ``degree * tangent_rotation``.
    """
    if degree < 1:
        raise ValueError("degree must be positive")
    if power_field.shape[-1] != 2:
        raise ValueError("power_field must have final dimension 2 (real, imag)")

    phase = degree * tangent_rotation
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    real, imag = power_field.unbind(dim=-1)
    return torch.stack(
        (
            cos_phase * real - sin_phase * imag,
            sin_phase * real + cos_phase * imag,
        ),
        dim=-1,
    )


def decode_rosy_angle(unit_power_field: Tensor, degree: int = 4) -> Tensor:
    """Decode the representative angle of an N-RoSy power field."""
    if degree < 1:
        raise ValueError("degree must be positive")
    if unit_power_field.shape[-1] != 2:
        raise ValueError(
            "unit_power_field must have final dimension 2 (real, imag)"
        )
    return torch.atan2(unit_power_field[..., 1], unit_power_field[..., 0]) / degree


def _gather_kernel_values(values: Tensor, kernel_indices: Optional[Tensor]) -> Tensor:
    if kernel_indices is None:
        return values
    if kernel_indices.dtype != torch.int64:
        kernel_indices = kernel_indices.long()
    return values[kernel_indices]


def aggregate_kernel_rosy_field(
    weights: Tensor,
    kernel_coefficients: Tensor,
    *,
    kernel_indices: Optional[Tensor] = None,
    transport_angles: Optional[Tensor] = None,
    kernel_log_sizes: Optional[Tensor] = None,
    degree: int = 4,
    eps: float = 1e-8,
) -> RosyFieldOutput:
    """Combine sparse kernel coefficients into an N-RoSy field.

    Args:
        weights: Heat/support weights with shape ``[..., K]``.
        kernel_coefficients: Unconstrained complex coefficients stored as
            ``[G, 2]`` real-imaginary pairs.  Their phase represents N times
            the physical direction and their magnitude is kernel amplitude.
        kernel_indices: Optional global kernel indices with shape ``[..., K]``.
            Omit when the last weight dimension directly enumerates all G
            kernels.
        transport_angles: Optional ordinary tangent-vector transport rotations
            with shape broadcastable to ``weights``.  ``None`` means all
            coefficients already use the query frame (appropriate for a plane
            or for tests only).
        kernel_log_sizes: Optional per-kernel logarithmic target edge sizes.
        degree: Rotational symmetry order; use 4 for quad cross fields.

    The returned unnormalised ``power_field`` is intentionally not divided by
    total heat weight.  Its magnitude and zeros contain coverage, cancellation,
    and singularity information.  ``coherence`` removes the overall weight
    scale and lies in [0, 1] for non-negative weights.
    """
    if weights.ndim < 1:
        raise ValueError("weights must have at least one dimension")
    if kernel_coefficients.ndim != 2 or kernel_coefficients.shape[-1] != 2:
        raise ValueError("kernel_coefficients must have shape [G, 2]")

    coefficients = _gather_kernel_values(kernel_coefficients, kernel_indices)
    if kernel_indices is None:
        if weights.shape[-1] != kernel_coefficients.shape[0]:
            raise ValueError(
                "Without kernel_indices, weights[..., K] must enumerate all G "
                "kernel_coefficients"
            )
        view_shape = (1,) * (weights.ndim - 1) + kernel_coefficients.shape
        coefficients = kernel_coefficients.view(view_shape).expand(*weights.shape, 2)
    elif coefficients.shape[:-1] != weights.shape:
        raise ValueError("kernel_indices and weights must have the same shape")

    if transport_angles is not None:
        try:
            transport_angles = torch.broadcast_to(transport_angles, weights.shape)
        except RuntimeError as exc:
            raise ValueError(
                "transport_angles must be broadcastable to weights"
            ) from exc
        coefficients = rotate_power_field(coefficients, transport_angles, degree)

    weighted_coefficients = weights.unsqueeze(-1) * coefficients
    power_field = weighted_coefficients.sum(dim=-2)

    unit_field, magnitude, valid = normalize_power_field(power_field, eps=eps)
    coefficient_amplitudes = torch.linalg.vector_norm(coefficients, dim=-1)
    total_amplitude = (
        weights.abs() * coefficient_amplitudes
    ).sum(dim=-1, keepdim=True)
    coherence = (magnitude / total_amplitude.clamp_min(eps)).clamp(0.0, 1.0)
    coverage = weights.abs().sum(dim=-1, keepdim=True)

    angle = decode_rosy_angle(unit_field, degree=degree)
    angle = torch.where(valid, angle, torch.zeros_like(angle))

    target_size = None
    if kernel_log_sizes is not None:
        if kernel_log_sizes.ndim != 1:
            raise ValueError("kernel_log_sizes must have shape [G]")
        local_log_sizes = _gather_kernel_values(kernel_log_sizes, kernel_indices)
        if kernel_indices is None:
            if kernel_log_sizes.shape[0] != weights.shape[-1]:
                raise ValueError(
                    "kernel_log_sizes and the final weights dimension must agree"
                )
            view_shape = (1,) * (weights.ndim - 1) + kernel_log_sizes.shape
            local_log_sizes = kernel_log_sizes.view(view_shape).expand_as(weights)
        positive_weights = weights.clamp_min(0.0)
        size_weight_sum = positive_weights.sum(dim=-1, keepdim=True)
        blended_log_size = (
            positive_weights * local_log_sizes
        ).sum(dim=-1, keepdim=True) / size_weight_sum.clamp_min(eps)
        target_size = torch.exp(blended_log_size)
        target_size = torch.where(
            size_weight_sum > eps, target_size, torch.zeros_like(target_size)
        )

    return RosyFieldOutput(
        power_field=power_field,
        unit_power_field=unit_field,
        magnitude=magnitude,
        coherence=coherence,
        angle=angle,
        valid=valid,
        coverage=coverage,
        target_size=target_size,
    )


def connection_smoothness_loss(
    power_field: Tensor,
    edges: Tensor,
    transport_angles: Tensor,
    *,
    edge_weights: Optional[Tensor] = None,
    degree: int = 4,
    normalize_field: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    """Connection-aware Dirichlet energy on oriented field edges.

    Each edge is oriented ``source -> target`` and ``transport_angles[e]``
    transports an ordinary tangent vector from the source frame into the
    target frame.
    """
    if power_field.ndim != 2 or power_field.shape[-1] != 2:
        raise ValueError("power_field must have shape [N, 2]")
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("edges must have shape [E, 2]")
    if transport_angles.shape != edges.shape[:1]:
        raise ValueError("transport_angles must have shape [E]")

    field = power_field
    if normalize_field:
        field, _, _ = normalize_power_field(field, eps=eps)

    source = edges[:, 0].long()
    target = edges[:, 1].long()
    transported_source = rotate_power_field(
        field[source], transport_angles, degree=degree
    )
    energy = (field[target] - transported_source).square().sum(dim=-1)
    if edge_weights is not None:
        if edge_weights.shape != energy.shape:
            raise ValueError("edge_weights must have shape [E]")
        energy = energy * edge_weights

    if reduction == "none":
        return energy
    if reduction == "sum":
        return energy.sum()
    if reduction == "mean":
        return energy.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def _weighted_mean(
    values: Tensor,
    *,
    weights: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    if weights is not None and weights.shape != values.shape:
        raise ValueError("weights must have the same shape as values")
    if valid is not None and valid.shape != values.shape:
        raise ValueError("valid must have the same shape as values")

    effective_weights = torch.ones_like(values)
    if weights is not None:
        effective_weights = effective_weights * weights
    if valid is not None:
        effective_weights = effective_weights * valid.to(values.dtype)

    denominator = effective_weights.sum()
    numerator = (values * effective_weights).sum()
    return torch.where(
        denominator > eps,
        numerator / denominator.clamp_min(eps),
        values.sum() * 0.0,
    )


def connection_direction_loss(
    power_field: Tensor,
    edges: Tensor,
    transport_angles: Tensor,
    *,
    edge_weights: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
    degree: int = 4,
    eps: float = 1e-8,
) -> Tensor:
    """Mean unit-field smoothness over valid connection edges."""
    unit_field, _, node_valid = normalize_power_field(power_field, eps=eps)
    source = edges[:, 0].long()
    target = edges[:, 1].long()
    edge_valid = node_valid[source] & node_valid[target]
    if valid is not None:
        if valid.shape != node_valid.shape:
            raise ValueError("valid must have shape [N]")
        edge_valid = edge_valid & valid[source].bool() & valid[target].bool()

    energy = connection_smoothness_loss(
        unit_field,
        edges,
        transport_angles,
        degree=degree,
        reduction="none",
    )
    return _weighted_mean(
        energy,
        weights=edge_weights,
        valid=edge_valid,
        eps=eps,
    )


def rosy_alignment_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    weights: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Cosine alignment loss for N-RoSy power coefficients.

    Both arguments already live in power space, so the objective automatically
    respects the N-fold rotational symmetry.  Invalid zero coefficients are
    excluded; combine this with :func:`field_strength_loss` when collapse to
    the all-zero field must be prevented during optimisation.
    """
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must have matching shape [..., 2]")

    prediction_unit, _, prediction_valid = normalize_power_field(prediction, eps)
    target_unit, _, target_valid = normalize_power_field(target, eps)
    sample_valid = prediction_valid & target_valid
    if valid is not None:
        if valid.shape != sample_valid.shape:
            raise ValueError("valid must match prediction.shape[:-1]")
        sample_valid = sample_valid & valid.bool()

    cosine = (prediction_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    return _weighted_mean(
        1.0 - cosine,
        weights=weights,
        valid=sample_valid,
        eps=eps,
    )


def field_strength_loss(
    power_field: Tensor,
    *,
    target_strength: float = 1.0,
    weights: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Keep sampled power coefficients away from the trivial zero solution."""
    if target_strength < 0:
        raise ValueError("target_strength must be non-negative")
    if power_field.shape[-1] != 2:
        raise ValueError("power_field must have final dimension 2")
    magnitude = torch.linalg.vector_norm(power_field, dim=-1)
    energy = (magnitude - target_strength).square()
    return _weighted_mean(energy, weights=weights, valid=valid, eps=eps)


def large_direction_jump_loss(
    power_field: Tensor,
    edges: Tensor,
    transport_angles: Tensor,
    *,
    maximum_physical_angle: float = torch.pi / 18.0,
    edge_weights: Optional[Tensor] = None,
    degree: int = 4,
    eps: float = 1e-8,
) -> Tensor:
    """Penalize only the high-error tail of connection-aware edge angles.

    Ordinary Dirichlet smoothing averages all edges and can hide a small set
    of severe rotations that later create spurious singularity pairs.  This
    term leaves small discrepancies alone and targets those outliers.
    """
    if maximum_physical_angle <= 0:
        raise ValueError("maximum_physical_angle must be positive")
    unit, _, node_valid = normalize_power_field(power_field, eps=eps)
    source = edges[:, 0].long()
    target = edges[:, 1].long()
    transported = rotate_power_field(
        unit[source], transport_angles, degree=degree
    )
    cosine = (transported * unit[target]).sum(dim=-1).clamp(-1.0, 1.0)
    threshold = torch.cos(
        power_field.new_tensor(float(degree) * maximum_physical_angle)
    )
    energy = torch.relu(threshold - cosine).square()
    return _weighted_mean(
        energy,
        weights=edge_weights,
        valid=node_valid[source] & node_valid[target],
        eps=eps,
    )


def edge_integrability_loss(
    power_field: Tensor,
    log_sizes: Tensor,
    edges: Tensor,
    source_edge_coordinates: Tensor,
    target_edge_coordinates: Tensor,
    *,
    edge_weights: Optional[Tensor] = None,
    degree: int = 4,
    eps: float = 1e-8,
) -> Tensor:
    """Local compatibility of the scaled cross differential across an edge.

    For an integrable seamless parameterization, the UV increment measured
    along a shared primal edge must agree in both incident faces, up to one of
    the four quarter-turn chart transitions.  This is a necessary local
    condition and is substantially closer to the downstream UV objective than
    direction and size smoothness optimized independently.
    """
    if log_sizes.ndim == 2 and log_sizes.shape[-1] == 1:
        log_sizes = log_sizes.squeeze(-1)
    if log_sizes.shape != power_field.shape[:-1]:
        raise ValueError("log_sizes must match power_field.shape[:-1]")
    if source_edge_coordinates.shape != edges.shape:
        raise ValueError("source_edge_coordinates must have shape [E, 2]")
    if target_edge_coordinates.shape != edges.shape:
        raise ValueError("target_edge_coordinates must have shape [E, 2]")

    unit, _, node_valid = normalize_power_field(power_field, eps=eps)
    angle = torch.atan2(unit[:, 1], unit[:, 0]) / float(degree)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    direction_u = torch.stack((cosine, sine), dim=-1)
    direction_v = torch.stack((-sine, cosine), dim=-1)
    inverse_size = torch.exp(-log_sizes)

    source = edges[:, 0].long()
    target = edges[:, 1].long()

    def increments(face_ids: Tensor, coordinates: Tensor) -> Tensor:
        du = (
            direction_u[face_ids] * coordinates
        ).sum(dim=-1) * inverse_size[face_ids]
        dv = (
            direction_v[face_ids] * coordinates
        ).sum(dim=-1) * inverse_size[face_ids]
        return torch.stack((du, dv), dim=-1)

    source_delta = increments(source, source_edge_coordinates)
    target_delta = increments(target, target_edge_coordinates)
    source_rotations = torch.stack(
        (
            source_delta,
            torch.stack((source_delta[:, 1], -source_delta[:, 0]), dim=-1),
            -source_delta,
            torch.stack((-source_delta[:, 1], source_delta[:, 0]), dim=-1),
        ),
        dim=1,
    )
    squared_error = (
        source_rotations - target_delta.unsqueeze(1)
    ).square().sum(dim=-1)
    best_error = squared_error.min(dim=1).values
    scale = 0.5 * (
        source_delta.square().sum(dim=-1)
        + target_delta.square().sum(dim=-1)
    )
    relative_error = best_error / scale.clamp_min(eps)
    return _weighted_mean(
        relative_error,
        weights=edge_weights,
        valid=node_valid[source] & node_valid[target],
        eps=eps,
    )


def field_quality_loss(
    values: Tensor,
    minimum: float,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Squared hinge that concentrates unavoidable low-quality regions."""
    if minimum < 0:
        raise ValueError("minimum must be non-negative")
    values = values.reshape(-1)
    if values.numel() == 0:
        return values.sum() * 0.0
    return torch.relu(values.new_tensor(minimum) - values).square().mean()


def log_size_smoothness_loss(
    log_sizes: Tensor,
    edges: Tensor,
    *,
    edge_weights: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Dirichlet energy for a positive target-size field in log space."""
    if log_sizes.ndim == 2 and log_sizes.shape[-1] == 1:
        log_sizes = log_sizes.squeeze(-1)
    if log_sizes.ndim != 1:
        raise ValueError("log_sizes must have shape [N] or [N, 1]")
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("edges must have shape [E, 2]")

    source = edges[:, 0].long()
    target = edges[:, 1].long()
    energy = (log_sizes[target] - log_sizes[source]).square()
    if edge_weights is not None:
        if edge_weights.shape != energy.shape:
            raise ValueError("edge_weights must have shape [E]")
        energy = energy * edge_weights

    if reduction == "none":
        return energy
    if reduction == "sum":
        return energy.sum()
    if reduction == "mean":
        return energy.mean() if energy.numel() else log_sizes.sum() * 0.0
    raise ValueError(f"Unknown reduction: {reduction}")


def log_size_target_loss(
    log_sizes: Tensor,
    target_log_sizes: Tensor,
    *,
    weights: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Weighted regression objective for positive sizes represented in log space."""
    if log_sizes.shape != target_log_sizes.shape:
        raise ValueError("log_sizes and target_log_sizes must have matching shapes")
    if log_sizes.ndim == 2 and log_sizes.shape[-1] == 1:
        log_sizes = log_sizes.squeeze(-1)
        target_log_sizes = target_log_sizes.squeeze(-1)
    if log_sizes.ndim != 1:
        raise ValueError("log sizes must have shape [N] or [N, 1]")
    return _weighted_mean(
        (log_sizes - target_log_sizes).square(),
        weights=weights,
        valid=valid,
        eps=eps,
    )


def quad_field_objective(
    power_field: Tensor,
    edges: Tensor,
    transport_angles: Tensor,
    *,
    guidance_power: Optional[Tensor] = None,
    guidance_indices: Optional[Tensor] = None,
    guidance_weights: Optional[Tensor] = None,
    boundary_power: Optional[Tensor] = None,
    boundary_indices: Optional[Tensor] = None,
    boundary_weights: Optional[Tensor] = None,
    log_sizes: Optional[Tensor] = None,
    target_log_sizes: Optional[Tensor] = None,
    target_size_weights: Optional[Tensor] = None,
    edge_weights: Optional[Tensor] = None,
    source_edge_coordinates: Optional[Tensor] = None,
    target_edge_coordinates: Optional[Tensor] = None,
    coherence: Optional[Tensor] = None,
    coverage: Optional[Tensor] = None,
    direction_smoothness_weight: float = 1.0,
    guidance_alignment_weight: float = 1.0,
    boundary_alignment_weight: float = 1.0,
    size_smoothness_weight: float = 0.1,
    size_target_weight: float = 1.0,
    field_strength_weight: float = 0.01,
    integrability_weight: float = 0.0,
    direction_tail_weight: float = 0.0,
    coherence_weight: float = 0.0,
    coverage_weight: float = 0.0,
    maximum_direction_jump: float = torch.pi / 18.0,
    minimum_coherence: float = 0.5,
    minimum_coverage: float = 0.25,
    degree: int = 4,
    eps: float = 1e-8,
) -> QuadFieldLossOutput:
    """Compose the differentiable MVP objective for a sampled quad field.

    ``guidance_power`` is typically built from reliable principal-curvature
    directions.  ``boundary_power`` is typically returned by
    ``boundary_cross_constraints``.  Both are optional so curvature-ambiguous
    or closed regions can rely on connection smoothness alone.
    """
    zero = power_field.sum() * 0.0
    direction_smoothness = connection_direction_loss(
        power_field,
        edges,
        transport_angles,
        edge_weights=edge_weights,
        degree=degree,
        eps=eps,
    )

    guidance_alignment = zero
    if guidance_power is not None:
        prediction = (
            power_field
            if guidance_indices is None
            else power_field[guidance_indices.long()]
        )
        guidance_alignment = rosy_alignment_loss(
            prediction,
            guidance_power,
            weights=guidance_weights,
            eps=eps,
        )
    elif guidance_indices is not None or guidance_weights is not None:
        raise ValueError("guidance indices/weights require guidance_power")

    boundary_alignment = zero
    if boundary_power is not None:
        if boundary_indices is None:
            raise ValueError("boundary_power requires boundary_indices")
        boundary_alignment = rosy_alignment_loss(
            power_field[boundary_indices.long()],
            boundary_power,
            weights=boundary_weights,
            eps=eps,
        )
    elif boundary_indices is not None or boundary_weights is not None:
        raise ValueError("boundary indices/weights require boundary_power")

    size_smoothness = zero
    if log_sizes is not None:
        size_smoothness = log_size_smoothness_loss(
            log_sizes, edges, edge_weights=edge_weights
        )

    size_target = zero
    if target_log_sizes is not None:
        if log_sizes is None:
            raise ValueError("target_log_sizes requires log_sizes")
        size_target = log_size_target_loss(
            log_sizes,
            target_log_sizes,
            weights=target_size_weights,
            eps=eps,
        )
    elif target_size_weights is not None:
        raise ValueError("target_size_weights requires target_log_sizes")

    strength = field_strength_loss(power_field, eps=eps)
    integrability = zero
    if source_edge_coordinates is not None or target_edge_coordinates is not None:
        if source_edge_coordinates is None or target_edge_coordinates is None:
            raise ValueError("both source and target edge coordinates are required")
        if log_sizes is None:
            raise ValueError("edge integrability requires log_sizes")
        integrability = edge_integrability_loss(
            power_field,
            log_sizes,
            edges,
            source_edge_coordinates,
            target_edge_coordinates,
            edge_weights=edge_weights,
            degree=degree,
            eps=eps,
        )

    direction_tail = large_direction_jump_loss(
        power_field,
        edges,
        transport_angles,
        maximum_physical_angle=maximum_direction_jump,
        edge_weights=edge_weights,
        degree=degree,
        eps=eps,
    )
    coherence_quality = (
        zero
        if coherence is None
        else field_quality_loss(coherence, minimum_coherence, eps=eps)
    )
    coverage_quality = (
        zero
        if coverage is None
        else field_quality_loss(coverage, minimum_coverage, eps=eps)
    )
    total = (
        direction_smoothness_weight * direction_smoothness
        + guidance_alignment_weight * guidance_alignment
        + boundary_alignment_weight * boundary_alignment
        + size_smoothness_weight * size_smoothness
        + size_target_weight * size_target
        + field_strength_weight * strength
        + integrability_weight * integrability
        + direction_tail_weight * direction_tail
        + coherence_weight * coherence_quality
        + coverage_weight * coverage_quality
    )
    return QuadFieldLossOutput(
        total,
        direction_smoothness,
        guidance_alignment,
        boundary_alignment,
        size_smoothness,
        size_target,
        strength,
        integrability,
        direction_tail,
        coherence_quality,
        coverage_quality,
    )


def _ordered_vertex_face_cycles(
    faces: Tensor,
    num_vertices: int,
) -> list[Optional[list[int]]]:
    """Order incident faces counter-clockwise around interior vertices."""
    faces_list = faces.detach().cpu().long().tolist()
    edge_faces: dict[tuple[int, int], list[int]] = {}
    vertex_faces: list[list[int]] = [[] for _ in range(num_vertices)]

    for face_id, face in enumerate(faces_list):
        for vertex in face:
            if vertex < 0 or vertex >= num_vertices:
                raise ValueError("faces contains a vertex outside num_vertices")
            vertex_faces[vertex].append(face_id)
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (a, b) if a < b else (b, a)
            edge_faces.setdefault(edge, []).append(face_id)

    if any(len(incident) > 2 for incident in edge_faces.values()):
        raise ValueError("singularity detection requires a manifold triangle mesh")

    cycles: list[Optional[list[int]]] = [None] * num_vertices
    for vertex, incident_faces in enumerate(vertex_faces):
        if not incident_faces:
            continue

        successor: dict[int, Optional[int]] = {}
        boundary_vertex = False
        for face_id in incident_faces:
            face = faces_list[face_id]
            corner = face.index(vertex)
            previous_vertex = face[(corner - 1) % 3]
            edge = (
                (vertex, previous_vertex)
                if vertex < previous_vertex
                else (previous_vertex, vertex)
            )
            owners = edge_faces[edge]
            if len(owners) == 1:
                successor[face_id] = None
                boundary_vertex = True
            else:
                successor[face_id] = owners[0] if owners[1] == face_id else owners[1]

        if boundary_vertex:
            continue

        start = min(incident_faces)
        cycle: list[int] = []
        current: Optional[int] = start
        while current is not None and current not in cycle:
            cycle.append(current)
            current = successor[current]
        if current != start or len(cycle) != len(incident_faces):
            raise ValueError(
                "faces must be consistently oriented and manifold at every vertex"
            )
        cycles[vertex] = cycle
    return cycles


@torch.no_grad()
def detect_face_field_singularities(
    power_field: Tensor,
    faces: Tensor,
    face_adjacency: Tensor,
    transport_angles: Tensor,
    vertex_angle_defects: Tensor,
    *,
    num_vertices: Optional[int] = None,
    degree: int = 4,
    eps: float = 1e-8,
) -> RosySingularityOutput:
    """Detect quantized indices of a face-based N-RoSy field.

    The face mesh must be an oriented manifold.  Each one-ring is traversed in
    the orientation induced by its incident faces.  Across every dual edge we
    remove the discrete Levi-Civita transport and choose the nearest 2*pi
    branch in power space.  The accumulated physical rotation is combined
    with the mesh's lifted vertex angle defect before quantisation.  Using the
    geometric defect rather than a wrapped sum of connection angles preserves
    the sign when holonomy is exactly pi, removes integer frame winding, and
    makes the result invariant to rotations of the per-face coordinate gauges.
    """
    if degree < 1:
        raise ValueError("degree must be positive")
    if power_field.ndim != 2 or power_field.shape[-1] != 2:
        raise ValueError("power_field must have shape [F, 2]")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if power_field.shape[0] != faces.shape[0]:
        raise ValueError("power_field and faces must contain the same face count")
    if face_adjacency.ndim != 2 or face_adjacency.shape[-1] != 2:
        raise ValueError("face_adjacency must have shape [E, 2]")
    if transport_angles.shape != face_adjacency.shape[:1]:
        raise ValueError("transport_angles must have shape [E]")

    inferred_vertices = int(faces.max().item()) + 1 if faces.numel() else 0
    if num_vertices is None:
        num_vertices = inferred_vertices
    if num_vertices < inferred_vertices:
        raise ValueError("num_vertices is smaller than an index in faces")
    if vertex_angle_defects.shape != (num_vertices,):
        raise ValueError("vertex_angle_defects must have shape [V]")
    vertex_angle_defects = vertex_angle_defects.to(power_field.device)

    cycles = _ordered_vertex_face_cycles(faces, num_vertices)
    adjacency_list = face_adjacency.detach().cpu().long().tolist()
    connection_lookup: dict[tuple[int, int], tuple[int, float]] = {}
    for edge_id, (source, target) in enumerate(adjacency_list):
        if (source, target) in connection_lookup or (target, source) in connection_lookup:
            raise ValueError("face_adjacency contains a duplicate undirected edge")
        connection_lookup[(source, target)] = (edge_id, 1.0)
        connection_lookup[(target, source)] = (edge_id, -1.0)

    unit_field, _, field_valid = normalize_power_field(power_field, eps=eps)
    phases = torch.atan2(unit_field[:, 1], unit_field[:, 0])
    transport_angles = transport_angles.to(power_field.device)
    indices = power_field.new_zeros((num_vertices,))
    residuals = power_field.new_zeros((num_vertices,))
    valid_vertices = torch.zeros(
        (num_vertices,), dtype=torch.bool, device=power_field.device
    )

    two_pi = 2.0 * torch.pi
    for vertex, cycle in enumerate(cycles):
        if cycle is None:
            continue
        cycle_tensor = torch.tensor(cycle, dtype=torch.long, device=power_field.device)
        if not bool(field_valid[cycle_tensor].all()):
            continue

        edge_ids: list[int] = []
        edge_signs: list[float] = []
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            connection = connection_lookup.get((source, target))
            if connection is None:
                raise ValueError("face_adjacency is missing a one-ring dual edge")
            edge_id, sign = connection
            edge_ids.append(edge_id)
            edge_signs.append(sign)

        edge_id_tensor = torch.tensor(
            edge_ids, dtype=torch.long, device=power_field.device
        )
        edge_sign_tensor = power_field.new_tensor(edge_signs)
        connection_angles = transport_angles[edge_id_tensor] * edge_sign_tensor
        next_cycle_tensor = torch.roll(cycle_tensor, shifts=-1)
        raw_transition = (
            phases[next_cycle_tensor]
            - phases[cycle_tensor]
            - degree * connection_angles
        )
        branch = torch.round(raw_transition / two_pi)
        principal_transition = raw_transition - two_pi * branch
        continuous_index = (
            principal_transition.sum() / (two_pi * degree)
            + vertex_angle_defects[vertex] / two_pi
        )

        indices[vertex] = torch.round(continuous_index * degree) / degree
        residuals[vertex] = principal_transition.abs().mean() / degree
        valid_vertices[vertex] = True

    return RosySingularityOutput(indices, valid_vertices, residuals)


def rosy_angular_error(
    prediction: Tensor,
    target: Tensor,
    *,
    degree: int = 4,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    """Smallest physical angular error between two N-RoSy power fields."""
    pred_unit = F.normalize(prediction, dim=-1, eps=eps)
    target_unit = F.normalize(target, dim=-1, eps=eps)
    cos_power_angle = (pred_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    error = torch.acos(cos_power_angle) / degree

    if reduction == "none":
        return error
    if reduction == "sum":
        return error.sum()
    if reduction == "mean":
        return error.mean()
    raise ValueError(f"Unknown reduction: {reduction}")
