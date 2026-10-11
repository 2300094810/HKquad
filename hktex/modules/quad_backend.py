"""Backend-neutral interchange helpers for HKTex2Quad dense face fields."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "BackendFieldBundle",
    "comb_face_cross_field",
    "cross_field_edge_transitions",
    "decode_face_cross_vectors",
    "load_dense_face_field",
    "make_backend_field_bundle",
    "save_backend_bundle",
    "validate_dense_face_field",
]


class BackendFieldBundle(NamedTuple):
    """Explicit 3D field quantities expected by parameterization backends."""

    vertices: Tensor
    faces: Tensor
    direction_u: Tensor
    direction_v: Tensor
    target_size: Tensor
    desired_gradient_u: Tensor
    desired_gradient_v: Tensor
    valid: Tensor
    face_adjacency: Tensor
    connection_angles: Tensor
    branch_quarter_turns: Tensor
    edge_quarter_turns: Tensor
    seam_edges: Tensor
    singularity_index: Tensor
    singularity_valid: Tensor


def _require(payload: dict, key: str) -> Tensor:
    if key not in payload:
        raise ValueError(f"dense field is missing required key: {key}")
    value = payload[key]
    if not isinstance(value, Tensor):
        raise ValueError(f"dense field key {key!r} must be a torch.Tensor")
    return value


def validate_dense_face_field(payload: dict, *, eps: float = 1e-6) -> None:
    """Validate the versioned stage-5 interchange payload."""
    if payload.get("format") != "hktex2quad-face-field-v1":
        raise ValueError("unsupported or missing dense-field format version")
    degree = payload.get("rosy_degree")
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("rosy_degree must be a positive integer")

    vertices = _require(payload, "vertices")
    faces = _require(payload, "faces")
    tangent_u = _require(payload, "face_tangent_u")
    tangent_v = _require(payload, "face_tangent_v")
    normals = _require(payload, "face_normals")
    power = _require(payload, "unit_power_field")
    target_size = _require(payload, "target_size")
    valid = _require(payload, "field_valid")
    adjacency = _require(payload, "face_adjacency")
    connection_angles = _require(payload, "connection_angles")
    singularity_index = _require(payload, "singularity_index")
    singularity_valid = _require(payload, "singularity_valid")

    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    face_count = faces.shape[0]
    for name, value, width in (
        ("face_tangent_u", tangent_u, 3),
        ("face_tangent_v", tangent_v, 3),
        ("face_normals", normals, 3),
        ("unit_power_field", power, 2),
    ):
        if value.shape != (face_count, width):
            raise ValueError(f"{name} must have shape [{face_count}, {width}]")
    if target_size.shape not in ((face_count,), (face_count, 1)):
        raise ValueError(f"target_size must have shape [{face_count}] or [{face_count}, 1]")
    if valid.shape != (face_count,):
        raise ValueError(f"field_valid must have shape [{face_count}]")
    if adjacency.ndim != 2 or adjacency.shape[-1] != 2:
        raise ValueError("face_adjacency must have shape [E, 2]")
    if connection_angles.shape != adjacency.shape[:1]:
        raise ValueError("connection_angles must have shape [E]")
    if adjacency.numel() and (
        int(adjacency.min().item()) < 0
        or int(adjacency.max().item()) >= face_count
    ):
        raise ValueError("face_adjacency contains an invalid face index")
    if singularity_index.shape != (vertices.shape[0],):
        raise ValueError("singularity_index must have shape [V]")
    if singularity_valid.shape != (vertices.shape[0],):
        raise ValueError("singularity_valid must have shape [V]")
    tensors = (vertices, tangent_u, tangent_v, normals, power, target_size)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("dense field contains NaN or infinite values")
    if bool((target_size <= 0).any()):
        raise ValueError("target_size must be strictly positive")
    if faces.numel() and (
        int(faces.min().item()) < 0
        or int(faces.max().item()) >= vertices.shape[0]
    ):
        raise ValueError("faces contains an invalid vertex index")

    for name, vector in (
        ("face_tangent_u", tangent_u),
        ("face_tangent_v", tangent_v),
        ("face_normals", normals),
    ):
        norms = torch.linalg.vector_norm(vector, dim=-1)
        if not bool(torch.allclose(norms, torch.ones_like(norms), atol=eps, rtol=eps)):
            raise ValueError(f"{name} must contain unit vectors")
    if bool(((tangent_u * tangent_v).sum(dim=-1).abs() > eps).any()):
        raise ValueError("face tangent axes must be orthogonal")
    if bool(((tangent_u * normals).sum(dim=-1).abs() > eps).any()):
        raise ValueError("face_tangent_u must be tangent to face_normals")
    if bool(((tangent_v * normals).sum(dim=-1).abs() > eps).any()):
        raise ValueError("face_tangent_v must be tangent to face_normals")


def decode_face_cross_vectors(
    unit_power_field: Tensor,
    tangent_u: Tensor,
    tangent_v: Tensor,
    *,
    degree: int = 4,
) -> tuple[Tensor, Tensor]:
    """Decode a deterministic representative pair of orthogonal 3D axes."""
    if degree < 1:
        raise ValueError("degree must be positive")
    if unit_power_field.ndim != 2 or unit_power_field.shape[-1] != 2:
        raise ValueError("unit_power_field must have shape [F, 2]")
    if tangent_u.shape != (unit_power_field.shape[0], 3):
        raise ValueError("tangent_u must have shape [F, 3]")
    if tangent_v.shape != tangent_u.shape:
        raise ValueError("tangent_v must match tangent_u")

    power_angle = torch.atan2(unit_power_field[:, 1], unit_power_field[:, 0])
    representative_angle = power_angle / degree
    cosine = torch.cos(representative_angle).unsqueeze(-1)
    sine = torch.sin(representative_angle).unsqueeze(-1)
    direction_u = cosine * tangent_u + sine * tangent_v
    direction_v = -sine * tangent_u + cosine * tangent_v
    return direction_u, direction_v


def comb_face_cross_field(payload: dict) -> tuple[Tensor, Tensor]:
    """Choose coherent quarter-turn branches and identify transition seams.

    A 4-RoSy coefficient fixes a cross but not which arm is called ``d1``.
    Breadth-first propagation chooses this label consistently using the
    connection angles.  Non-tree edges whose propagated labels disagree are
    reported as candidate seams; singular holonomy makes such seams necessary.

    Returns:
        ``branch_quarter_turns[F]`` in ``{0,1,2,3}`` and ``seam_edges[E]``.
    """
    validate_dense_face_field(payload)
    if payload["rosy_degree"] != 4:
        raise ValueError("quad backend combing currently requires rosy_degree == 4")

    power = payload["unit_power_field"]
    base_angle = torch.atan2(power[:, 1], power[:, 0]) / 4.0
    adjacency = payload["face_adjacency"].detach().cpu().long()
    connection = payload["connection_angles"].detach().cpu()
    valid = payload["field_valid"].detach().cpu().bool()
    face_count = power.shape[0]

    neighbors: list[list[tuple[int, int, float]]] = [
        [] for _ in range(face_count)
    ]
    for edge_id, ((source, target), angle) in enumerate(
        zip(adjacency.tolist(), connection.tolist())
    ):
        neighbors[source].append((target, edge_id, float(angle)))
        neighbors[target].append((source, edge_id, -float(angle)))
    for entries in neighbors:
        entries.sort(key=lambda value: (value[0], value[1]))

    branch = torch.full((face_count,), -1, dtype=torch.long)
    seams = torch.zeros((adjacency.shape[0],), dtype=torch.bool)
    base_cpu = base_angle.detach().cpu()
    quarter_turn = math.pi / 2.0

    for root in range(face_count):
        if branch[root] >= 0:
            continue
        branch[root] = 0
        if not valid[root]:
            continue
        queue = [root]
        head = 0
        while head < len(queue):
            source = queue[head]
            head += 1
            source_angle = (
                float(base_cpu[source].item())
                + int(branch[source].item()) * quarter_turn
            )
            for target, edge_id, transport in neighbors[source]:
                if not valid[target]:
                    seams[edge_id] = True
                    continue
                proposed = int(
                    round(
                        (source_angle + transport - float(base_cpu[target].item()))
                        / quarter_turn
                    )
                ) % 4
                if branch[target] < 0:
                    branch[target] = proposed
                    queue.append(target)
                elif int(branch[target].item()) != proposed:
                    seams[edge_id] = True

    branch = branch.clamp_min(0).to(device=power.device)
    seams = seams.to(device=power.device)
    return branch, seams


def cross_field_edge_transitions(
    payload: dict,
    branch_quarter_turns: Tensor,
) -> Tensor:
    """Return quarter-turn UV transitions for every oriented dual edge."""
    validate_dense_face_field(payload)
    if payload["rosy_degree"] != 4:
        raise ValueError("quad transitions currently require rosy_degree == 4")
    adjacency = payload["face_adjacency"].long()
    if branch_quarter_turns.shape != (payload["faces"].shape[0],):
        raise ValueError("branch_quarter_turns must have shape [F]")
    base_angle = torch.atan2(
        payload["unit_power_field"][:, 1],
        payload["unit_power_field"][:, 0],
    ) / 4.0
    quarter_turn = torch.pi / 2.0
    source = adjacency[:, 0]
    target = adjacency[:, 1]
    transported_source_angle = (
        base_angle[source]
        + branch_quarter_turns[source].to(base_angle.dtype) * quarter_turn
        + payload["connection_angles"]
    )
    proposed_target_branch = torch.round(
        (transported_source_angle - base_angle[target]) / quarter_turn
    ).long()
    return (
        branch_quarter_turns[target].long() - proposed_target_branch
    ).remainder(4)


def make_backend_field_bundle(payload: dict) -> BackendFieldBundle:
    """Convert a validated power-field payload to explicit backend vectors."""
    validate_dense_face_field(payload)
    branch_quarter_turns, seam_edges = comb_face_cross_field(payload)
    edge_quarter_turns = cross_field_edge_transitions(
        payload, branch_quarter_turns
    )
    seam_edges = seam_edges | (edge_quarter_turns != 0)
    branch_angle = branch_quarter_turns.to(payload["unit_power_field"].dtype) * (
        torch.pi / 2.0
    )
    # decode_face_cross_vectors cannot distinguish quarter-turn branches in
    # power space, so apply the selected label explicitly in tangent space.
    cosine = torch.cos(branch_angle).unsqueeze(-1)
    sine = torch.sin(branch_angle).unsqueeze(-1)
    uncombed_u, uncombed_v = decode_face_cross_vectors(
        payload["unit_power_field"],
        payload["face_tangent_u"],
        payload["face_tangent_v"],
        degree=payload["rosy_degree"],
    )
    direction_u = cosine * uncombed_u + sine * uncombed_v
    direction_v = -sine * uncombed_u + cosine * uncombed_v
    target_size = payload["target_size"].reshape(-1)
    inverse_size = target_size.reciprocal().unsqueeze(-1)
    return BackendFieldBundle(
        vertices=payload["vertices"],
        faces=payload["faces"].long(),
        direction_u=direction_u,
        direction_v=direction_v,
        target_size=target_size,
        desired_gradient_u=direction_u * inverse_size,
        desired_gradient_v=direction_v * inverse_size,
        valid=payload["field_valid"].bool(),
        face_adjacency=payload["face_adjacency"].long(),
        connection_angles=payload["connection_angles"],
        branch_quarter_turns=branch_quarter_turns,
        edge_quarter_turns=edge_quarter_turns,
        seam_edges=seam_edges,
        singularity_index=payload["singularity_index"],
        singularity_valid=payload["singularity_valid"].bool(),
    )


def load_dense_face_field(filename: str | Path) -> dict:
    """Load and validate a trainer export without importing the full HKTex stack."""
    payload = torch.load(filename, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("dense face-field file must contain a dictionary")
    validate_dense_face_field(payload)
    return payload


def _as_numpy(value: Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _write_obj(filename: Path, vertices: Tensor, faces: Tensor) -> None:
    with filename.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# HKTex2Quad source triangle mesh\n")
        for vertex in _as_numpy(vertices):
            handle.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for face in _as_numpy(faces.long()):
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def save_backend_bundle(payload: dict, output_dir: str | Path) -> Path:
    """Save a portable mesh/field bundle for a parameterization adapter.

    The bundle deliberately does not claim to be MIQ- or QuadriFlow-specific.
    It contains all continuous quantities needed by such an adapter while
    preserving the original versioned PyTorch payload as the source of truth.
    """
    bundle = make_backend_field_bundle(payload)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_obj(output_dir / "source_mesh.obj", bundle.vertices, bundle.faces)
    np.savez_compressed(
        output_dir / "cross_size_field.npz",
        direction_u=_as_numpy(bundle.direction_u),
        direction_v=_as_numpy(bundle.direction_v),
        target_size=_as_numpy(bundle.target_size),
        desired_gradient_u=_as_numpy(bundle.desired_gradient_u),
        desired_gradient_v=_as_numpy(bundle.desired_gradient_v),
        field_valid=_as_numpy(bundle.valid),
        face_adjacency=_as_numpy(bundle.face_adjacency),
        connection_angles=_as_numpy(bundle.connection_angles),
        branch_quarter_turns=_as_numpy(bundle.branch_quarter_turns),
        edge_quarter_turns=_as_numpy(bundle.edge_quarter_turns),
        seam_edges=_as_numpy(bundle.seam_edges),
        singularity_index=_as_numpy(bundle.singularity_index),
        singularity_valid=_as_numpy(bundle.singularity_valid),
    )

    table = np.concatenate(
        (
            np.arange(bundle.faces.shape[0], dtype=np.int64)[:, None],
            _as_numpy(bundle.direction_u),
            _as_numpy(bundle.direction_v),
            _as_numpy(bundle.target_size)[:, None],
            _as_numpy(bundle.valid).astype(np.int64)[:, None],
            _as_numpy(bundle.branch_quarter_turns)[:, None],
        ),
        axis=1,
    )
    with (output_dir / "face_field.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "face_id",
                "d1_x",
                "d1_y",
                "d1_z",
                "d2_x",
                "d2_y",
                "d2_z",
                "target_size",
                "valid",
                "branch_quarter_turns",
            ]
        )
        writer.writerows(table.tolist())

    manifest = {
        "format": "hktex2quad-backend-bundle-v1",
        "source_format": payload["format"],
        "rosy_degree": payload["rosy_degree"],
        "vertex_count": int(bundle.vertices.shape[0]),
        "triangle_count": int(bundle.faces.shape[0]),
        "valid_face_count": int(bundle.valid.sum().item()),
        "candidate_seam_count": int(bundle.seam_edges.sum().item()),
        "files": {
            "mesh": "source_mesh.obj",
            "field": "cross_size_field.npz",
            "human_readable_field": "face_field.csv",
        },
        "convention": {
            "indices": "zero-based in NPZ/CSV; one-based in OBJ",
            "direction_u_direction_v": "orthonormal representative of the 4-RoSy cross",
            "desired_gradients": "direction / target_size",
            "target_size": "desired surface edge length",
            "branch_quarter_turns": "d1 label rotation in multiples of pi/2",
            "edge_quarter_turns": "UV rotation over each oriented face adjacency",
            "seam_edges": "boolean mask over face_adjacency candidate cuts",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path
