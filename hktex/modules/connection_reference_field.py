"""Low-frequency connection-Laplacian references for kernel cross fields."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from torch import Tensor

from .rosy_field import detect_face_field_singularities, normalize_power_field

__all__ = ["ConnectionReferenceField", "build_connection_reference_field"]


class ConnectionReferenceField(NamedTuple):
    power_field: Tensor
    singularity_index: Tensor
    singularity_valid: Tensor
    eigenvalue: float
    candidate_count: int
    nonzero_singularity_count: int
    wrong_sign_singularity_count: int
    high_order_singularity_count: int
    alignment_loss: float


def _connection_laplacian(
    face_count: int,
    adjacency: np.ndarray,
    connection_angles: np.ndarray,
    edge_weights: np.ndarray,
    degree: int,
) -> sp.csr_matrix:
    rotation = np.exp(1j * float(degree) * connection_angles)
    source = adjacency[:, 0]
    target = adjacency[:, 1]
    rows = np.concatenate((source, target, source, target))
    columns = np.concatenate((source, target, target, source))
    values = np.concatenate(
        (
            edge_weights,
            edge_weights,
            -edge_weights * np.conjugate(rotation),
            -edge_weights * rotation,
        )
    )
    return sp.coo_matrix(
        (values, (rows, columns)), shape=(face_count, face_count)
    ).tocsr()


def build_connection_reference_field(
    guidance_power: Tensor,
    guidance_weights: Tensor,
    faces: Tensor,
    adjacency: Tensor,
    connection_angles: Tensor,
    angle_defects: Tensor,
    edge_weights: Tensor,
    *,
    num_vertices: int,
    degree: int = 4,
    eigenvector_count: int = 4,
    random_candidate_count: int = 48,
    seed: int = 0,
    eps: float = 1e-10,
) -> ConnectionReferenceField:
    """Choose a smooth, topology-aware field from a low connection spectrum.

    The candidate selector strongly prefers the quarter-index singularity
    pattern implied by the Euler characteristic, then uses weighted curvature
    alignment only as a tie breaker.  This prevents noisy principal directions
    from creating many cancellable positive/negative singularity pairs.
    """
    if degree < 1 or eigenvector_count < 1 or random_candidate_count < 0:
        raise ValueError("degree/eigenvector_count must be positive")
    unit_guidance, _, guidance_valid = normalize_power_field(guidance_power, eps)
    face_count = int(guidance_power.shape[0])
    if face_count < 2:
        return ConnectionReferenceField(
            unit_guidance,
            guidance_power.new_zeros((num_vertices,)),
            torch.zeros(num_vertices, dtype=torch.bool, device=guidance_power.device),
            0.0,
            1,
            0,
            0,
            0,
            0.0,
        )

    adjacency_np = adjacency.detach().cpu().long().numpy()
    angles_np = connection_angles.detach().cpu().double().numpy()
    weights_np = edge_weights.detach().cpu().double().numpy()
    laplacian = _connection_laplacian(
        face_count, adjacency_np, angles_np, weights_np, degree
    )
    basis_count = min(eigenvector_count, face_count - 1)
    regularized = laplacian + 1e-10 * sp.eye(
        face_count, dtype=np.complex128, format="csr"
    )
    eigenvalues, basis = eigsh(
        regularized,
        k=basis_count,
        sigma=0.0,
        which="LM",
        tol=1e-8,
        v0=np.ones(face_count, dtype=np.complex128),
    )
    order = np.argsort(eigenvalues)
    eigenvalues = np.real(eigenvalues[order])
    basis = basis[:, order]

    guide_np = unit_guidance.detach().cpu().double().numpy()
    guide_complex = guide_np[:, 0] + 1j * guide_np[:, 1]
    guide_weights = (
        guidance_weights.detach().cpu().double().numpy()
        * guidance_valid.detach().cpu().numpy().astype(np.float64)
    )
    guide_weight_sum = max(float(guide_weights.sum()), eps)

    candidates = [basis[:, index] for index in range(basis_count)]
    projection = basis @ (np.conjugate(basis).T @ (guide_weights * guide_complex))
    candidates.append(projection)
    rng = np.random.default_rng(seed)
    for _ in range(random_candidate_count):
        coefficients = rng.standard_normal(basis_count) + 1j * rng.standard_normal(
            basis_count
        )
        candidates.append(basis @ coefficients)

    # Use mesh combinatorics instead of summed angle defects here.  The latter
    # needs a separate pi boundary convention and can therefore report a
    # misleading Euler characteristic on open meshes.
    faces_np = faces.detach().cpu().long().numpy()
    used_vertex_count = int(np.unique(faces_np).size)
    undirected_edges = np.concatenate(
        (
            faces_np[:, [0, 1]],
            faces_np[:, [1, 2]],
            faces_np[:, [2, 0]],
        ),
        axis=0,
    )
    undirected_edges.sort(axis=1)
    _, edge_incidence = np.unique(
        undirected_edges, axis=0, return_counts=True
    )
    edge_count = int(edge_incidence.shape[0])
    euler_characteristic = used_vertex_count - edge_count + face_count
    closed_mesh = bool(np.all(edge_incidence == 2))
    # On a closed surface Poincare-Hopf fixes the total interior index.  On an
    # open surface the boundary turning/alignment carries part of that index;
    # forcing the closed formula on interior vertices creates fake defects.
    expected_count = abs(euler_characteristic) * degree if closed_mesh else 0
    expected_sign = (
        0
        if not closed_mesh or euler_characteristic == 0
        else int(np.sign(euler_characteristic))
    )

    best = None
    for candidate_index, candidate in enumerate(candidates):
        magnitude = np.abs(candidate)
        if float(magnitude.max()) <= eps:
            continue
        candidate = candidate / np.maximum(magnitude, eps)
        correlation = np.sum(
            guide_weights * np.conjugate(guide_complex) * candidate
        )
        if abs(correlation) > eps:
            candidate *= np.exp(-1j * np.angle(correlation))
        candidate_tensor = torch.from_numpy(
            np.stack((candidate.real, candidate.imag), axis=-1)
        ).to(dtype=guidance_power.dtype)
        singularities = detect_face_field_singularities(
            candidate_tensor,
            faces.detach().cpu().long(),
            adjacency.detach().cpu().long(),
            connection_angles.detach().cpu().to(candidate_tensor.dtype),
            angle_defects.detach().cpu().to(candidate_tensor.dtype),
            num_vertices=num_vertices,
            degree=degree,
        )
        nonzero = singularities.valid & (singularities.index.abs() > 1e-8)
        indices = singularities.index[nonzero]
        wrong_sign = int(
            ((indices * expected_sign) < -1e-8).sum().item()
            if expected_sign
            else nonzero.sum().item()
        )
        high_order = int((indices.abs() > (1.0 / degree + 1e-6)).sum().item())
        count = int(nonzero.sum().item())
        alignment = float(
            np.sum(
                guide_weights
                * (
                    candidate.real * guide_complex.real
                    + candidate.imag * guide_complex.imag
                )
            )
            / guide_weight_sum
        )
        alignment_loss = 1.0 - alignment
        residual = float(
            singularities.residual[singularities.valid].mean().item()
            if bool(singularities.valid.any())
            else 0.0
        )
        score = (
            wrong_sign,
            high_order,
            abs(count - expected_count),
            count,
            alignment_loss,
            residual,
            candidate_index,
        )
        if best is None or score < best[0]:
            best = (
                score,
                candidate_tensor,
                singularities,
                count,
                wrong_sign,
                high_order,
                alignment_loss,
            )

    if best is None:
        raise RuntimeError("connection reference search produced no valid candidate")
    _, field, singularities, count, wrong_sign, high_order, alignment_loss = best
    return ConnectionReferenceField(
        field.to(device=guidance_power.device, dtype=guidance_power.dtype),
        singularities.index.to(guidance_power.device),
        singularities.valid.to(guidance_power.device),
        float(eigenvalues[0]),
        len(candidates),
        count,
        wrong_sign,
        high_order,
        alignment_loss,
    )
