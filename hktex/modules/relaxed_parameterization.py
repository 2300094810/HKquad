"""Least-squares field integration used before seamless parameterization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr

__all__ = [
    "RelaxedParameterizationResult",
    "solve_relaxed_parameterization",
    "save_relaxed_parameterization",
]


class RelaxedParameterizationResult(NamedTuple):
    """Vertex UVs and diagnostics from independent gradient integration."""

    uv: np.ndarray
    reconstructed_gradient_u: np.ndarray
    reconstructed_gradient_v: np.ndarray
    per_face_residual: np.ndarray
    rms_residual: float
    solver_iterations: tuple[int, int]


def _as_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _gradient_operator(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    eps: float = 1e-14,
) -> tuple[sp.csr_matrix, np.ndarray]:
    triangles = vertices[faces]
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    doubled_area = np.linalg.norm(normal, axis=1)
    if np.any(doubled_area <= eps):
        raise ValueError("cannot parameterize a mesh with degenerate triangles")
    denominator = np.square(doubled_area)
    gradients = np.stack(
        (
            np.cross(normal, triangles[:, 2] - triangles[:, 1]),
            np.cross(normal, triangles[:, 0] - triangles[:, 2]),
            np.cross(normal, triangles[:, 1] - triangles[:, 0]),
        ),
        axis=1,
    ) / denominator[:, None, None]

    face_count = faces.shape[0]
    rows = np.broadcast_to(
        np.arange(face_count * 3).reshape(face_count, 3, 1),
        (face_count, 3, 3),
    )
    columns = np.broadcast_to(faces[:, None, :], (face_count, 3, 3))
    values = gradients.transpose(0, 2, 1)
    operator = sp.coo_matrix(
        (values.reshape(-1), (rows.reshape(-1), columns.reshape(-1))),
        shape=(face_count * 3, vertices.shape[0]),
    ).tocsr()
    return operator, doubled_area * 0.5


def _component_anchors(vertex_count: int, faces: np.ndarray) -> sp.csr_matrix:
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    graph = sp.coo_matrix(
        (
            np.ones(edges.shape[0] * 2),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    anchors = np.array(
        [np.flatnonzero(labels == component)[0] for component in range(component_count)]
    )
    return sp.coo_matrix(
        (
            np.ones(component_count),
            (np.arange(component_count), anchors),
        ),
        shape=(component_count, vertex_count),
    ).tocsr()


def solve_relaxed_parameterization(
    vertices,
    faces,
    desired_gradient_u,
    desired_gradient_v,
    *,
    seam_edges: Optional[np.ndarray] = None,
    atol: float = 1e-10,
    btol: float = 1e-10,
    max_iterations: Optional[int] = None,
) -> RelaxedParameterizationResult:
    """Integrate target face gradients into a globally continuous UV map.

    This is an integrability diagnostic and initialization, not a seamless MIQ
    solver. It rejects non-empty candidate seams because those require vertex
    duplication and integer quarter-turn/translation transitions.
    """
    vertices = _as_numpy(vertices, np.float64)
    faces = _as_numpy(faces, np.int64)
    gradient_u = _as_numpy(desired_gradient_u, np.float64)
    gradient_v = _as_numpy(desired_gradient_v, np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if gradient_u.shape != (faces.shape[0], 3):
        raise ValueError("desired_gradient_u must have shape [F, 3]")
    if gradient_v.shape != gradient_u.shape:
        raise ValueError("desired_gradient_v must match desired_gradient_u")
    if seam_edges is not None and bool(_as_numpy(seam_edges, bool).any()):
        raise ValueError(
            "relaxed global integration cannot cross candidate seams; "
            "use a seamless parameterization backend"
        )
    finite_inputs = (
        np.isfinite(vertices).all()
        and np.isfinite(gradient_u).all()
        and np.isfinite(gradient_v).all()
    )
    if not finite_inputs:
        raise ValueError("parameterization inputs must be finite")

    operator, face_areas = _gradient_operator(vertices, faces)
    sqrt_weights = np.repeat(np.sqrt(face_areas), 3)
    weighted_operator = sp.diags(sqrt_weights) @ operator
    anchors = _component_anchors(vertices.shape[0], faces)
    system = sp.vstack((weighted_operator, anchors), format="csr")

    def solve(target: np.ndarray):
        rhs = np.concatenate(
            (target.reshape(-1) * sqrt_weights, np.zeros(anchors.shape[0]))
        )
        result = lsmr(
            system,
            rhs,
            atol=atol,
            btol=btol,
            maxiter=max_iterations,
        )
        return result[0], int(result[2])

    u, iterations_u = solve(gradient_u)
    v, iterations_v = solve(gradient_v)
    uv = np.stack((u, v), axis=-1)
    reconstructed_u = (operator @ u).reshape(-1, 3)
    reconstructed_v = (operator @ v).reshape(-1, 3)
    residual_u = np.sum(np.square(reconstructed_u - gradient_u), axis=-1)
    residual_v = np.sum(np.square(reconstructed_v - gradient_v), axis=-1)
    per_face_residual = np.sqrt(residual_u + residual_v)
    rms = float(
        np.sqrt(
            np.sum(face_areas * np.square(per_face_residual))
            / np.sum(face_areas)
        )
    )
    return RelaxedParameterizationResult(
        uv=uv,
        reconstructed_gradient_u=reconstructed_u,
        reconstructed_gradient_v=reconstructed_v,
        per_face_residual=per_face_residual,
        rms_residual=rms,
        solver_iterations=(iterations_u, iterations_v),
    )


def save_relaxed_parameterization(
    result: RelaxedParameterizationResult,
    vertices,
    faces,
    output_dir: str | Path,
) -> Path:
    """Save diagnostic UV arrays and an OBJ carrying per-vertex texture coords."""
    vertices = _as_numpy(vertices, np.float64)
    faces = _as_numpy(faces, np.int64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "relaxed_parameterization.npz",
        uv=result.uv,
        reconstructed_gradient_u=result.reconstructed_gradient_u,
        reconstructed_gradient_v=result.reconstructed_gradient_v,
        per_face_residual=result.per_face_residual,
    )
    with (output_dir / "relaxed_parameterization.obj").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("# HKTex2Quad relaxed (not seamless) parameterization\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for uv in result.uv:
            handle.write(f"vt {uv[0]:.17g} {uv[1]:.17g}\n")
        for face in faces:
            a, b, c = face + 1
            handle.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    report = {
        "format": "hktex2quad-relaxed-parameterization-v1",
        "seamless": False,
        "rms_gradient_residual": result.rms_residual,
        "solver_iterations": list(result.solver_iterations),
        "warning": (
            "This globally continuous least-squares UV is an integrability "
            "diagnostic, not an integer-grid seamless parameterization."
        ),
    }
    report_path = output_dir / "relaxed_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path
