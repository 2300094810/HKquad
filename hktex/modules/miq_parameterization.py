"""Adapter for the libigl/CoMISo mixed-integer quadrangulation backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

__all__ = [
    "MIQParameterizationResult",
    "MIQRawOutput",
    "load_miq_output",
    "save_miq_parameterization",
    "solve_miq_parameterization",
    "write_miq_input",
]


class MIQRawOutput(NamedTuple):
    uv_vertices: np.ndarray
    uv_faces: np.ndarray
    seam_corners: np.ndarray
    mismatch: np.ndarray
    singular: np.ndarray
    singularity_index: np.ndarray


class MIQParameterizationResult(NamedTuple):
    """Per-corner UV map returned by a true CoMISo mixed-integer solve."""

    corner_uv: np.ndarray
    face_charts: np.ndarray
    uv_vertices: np.ndarray
    uv_faces: np.ndarray
    seam_edges: np.ndarray
    seam_corners: np.ndarray
    mismatch: np.ndarray
    singular: np.ndarray
    singularity_index: np.ndarray
    reconstructed_gradient_u: np.ndarray
    reconstructed_gradient_v: np.ndarray
    alignment_quarter_turns: np.ndarray
    per_face_residual: np.ndarray
    rms_gradient_residual: float
    relative_gradient_residual: float
    flipped_face_count: int
    flipped_face_fraction: float
    cut_vertex_count: int
    chart_count: int
    gradient_scale: float
    stiffness: float
    iterations: int
    local_iterations: int
    direct_round: bool
    round_singularities: bool
    executable: str
    solver_stdout: str


def _as_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _validate_inputs(
    vertices: np.ndarray,
    faces: np.ndarray,
    direction_u: np.ndarray,
    direction_v: np.ndarray,
    target_size: np.ndarray,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if direction_u.shape != (faces.shape[0], 3):
        raise ValueError("direction_u must have shape [F, 3]")
    if direction_v.shape != direction_u.shape:
        raise ValueError("direction_v must match direction_u")
    if target_size.shape != (faces.shape[0],):
        raise ValueError("target_size must have shape [F]")
    if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
        raise ValueError("faces contain an invalid vertex index")
    if not all(
        np.isfinite(value).all()
        for value in (vertices, direction_u, direction_v, target_size)
    ):
        raise ValueError("MIQ inputs contain NaN or infinite values")
    if np.any(target_size <= 0.0):
        raise ValueError("target_size must be strictly positive")
    if np.any(np.linalg.norm(direction_u, axis=1) <= 1e-14):
        raise ValueError("direction_u contains a zero vector")
    if np.any(np.linalg.norm(direction_v, axis=1) <= 1e-14):
        raise ValueError("direction_v contains a zero vector")


def write_miq_input(
    filename: str | Path,
    vertices,
    faces,
    direction_u,
    direction_v,
    target_size,
) -> Path:
    """Write the versioned, dependency-free input consumed by the C++ solver."""
    filename = Path(filename)
    vertices = _as_numpy(vertices, np.float64)
    faces = _as_numpy(faces, np.int64)
    direction_u = _as_numpy(direction_u, np.float64)
    direction_v = _as_numpy(direction_v, np.float64)
    target_size = _as_numpy(target_size, np.float64).reshape(-1)
    _validate_inputs(vertices, faces, direction_u, direction_v, target_size)

    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("HKTEX2QUAD_MIQ_INPUT_V1\n")
        handle.write(f"{vertices.shape[0]} {faces.shape[0]}\n")
        for vertex in vertices:
            handle.write(" ".join(f"{value:.17g}" for value in vertex) + "\n")
        for face in faces:
            handle.write(" ".join(str(int(value)) for value in face) + "\n")
        for first, second, size in zip(direction_u, direction_v, target_size):
            values = np.concatenate((first, second, np.asarray([size])))
            handle.write(" ".join(f"{value:.17g}" for value in values) + "\n")
    return filename


def load_miq_output(
    filename: str | Path,
    *,
    expected_face_count: Optional[int] = None,
    expected_vertex_count: Optional[int] = None,
) -> MIQRawOutput:
    """Read and strictly validate the C++ backend output."""
    tokens = Path(filename).read_text(encoding="utf-8").split()
    if not tokens or tokens[0] != "HKTEX2QUAD_MIQ_OUTPUT_V1":
        raise ValueError("unsupported or missing MIQ output format")
    cursor = 1

    def take_int() -> int:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("truncated MIQ output")
        value = int(tokens[cursor])
        cursor += 1
        return value

    def take_float() -> float:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("truncated MIQ output")
        value = float(tokens[cursor])
        cursor += 1
        return value

    uv_count = take_int()
    face_count = take_int()
    vertex_count = take_int()
    if uv_count <= 0 or face_count <= 0 or vertex_count <= 0:
        raise ValueError("MIQ output contains invalid array dimensions")
    if expected_face_count is not None and face_count != expected_face_count:
        raise ValueError("MIQ output face count does not match the input")
    if expected_vertex_count is not None and vertex_count != expected_vertex_count:
        raise ValueError("MIQ output vertex count does not match the input")

    uv = np.asarray(
        [[take_float(), take_float()] for _ in range(uv_count)],
        dtype=np.float64,
    )
    uv_faces = np.asarray(
        [[take_int(), take_int(), take_int()] for _ in range(face_count)],
        dtype=np.int64,
    )
    seams = np.asarray(
        [[take_int(), take_int(), take_int()] for _ in range(face_count)],
        dtype=np.int64,
    )
    mismatch = np.asarray(
        [[take_int(), take_int(), take_int()] for _ in range(face_count)],
        dtype=np.int64,
    )
    singular_data = np.asarray(
        [[take_int(), take_int()] for _ in range(vertex_count)],
        dtype=np.int64,
    )
    if cursor != len(tokens):
        raise ValueError("MIQ output contains unexpected trailing data")
    if not np.isfinite(uv).all():
        raise ValueError("MIQ output contains NaN or infinite UV coordinates")
    if uv_faces.min() < 0 or uv_faces.max() >= uv_count:
        raise ValueError("MIQ output contains an invalid UV vertex index")
    return MIQRawOutput(
        uv_vertices=uv,
        uv_faces=uv_faces,
        seam_corners=seams,
        mismatch=mismatch,
        singular=singular_data[:, 0].astype(bool),
        singularity_index=singular_data[:, 1],
    )


def _resolve_executable(executable: Optional[str | Path]) -> Path:
    if executable is None:
        executable = os.environ.get("HKTEX2QUAD_MIQ_EXECUTABLE")
    candidates: list[Path] = []
    if executable is not None:
        explicit = Path(executable).expanduser()
        candidates.append(explicit)
        found = shutil.which(str(executable))
        if found:
            candidates.append(Path(found))
    repository = Path(__file__).resolve().parents[2]
    build = repository / "cpp" / "miq_backend" / "build"
    candidates.extend(
        (
            build / "hktex2quad_miq",
            build / "hktex2quad_miq.exe",
            build / "Release" / "hktex2quad_miq.exe",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "the libigl/CoMISo MIQ executable was not found. Build it with:\n"
        "  cmake -S cpp/miq_backend -B cpp/miq_backend/build "
        "-DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build cpp/miq_backend/build --config Release -j\n"
        f"Checked:\n  - {checked}"
    )


def _face_gradient_data(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    eps: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    doubled_area = np.linalg.norm(normal, axis=1)
    if np.any(doubled_area <= eps):
        raise ValueError("cannot parameterize a mesh with degenerate triangles")
    gradients = np.stack(
        (
            np.cross(normal, triangles[:, 2] - triangles[:, 1]),
            np.cross(normal, triangles[:, 0] - triangles[:, 2]),
            np.cross(normal, triangles[:, 1] - triangles[:, 0]),
        ),
        axis=1,
    ) / np.square(doubled_area)[:, None, None]
    return gradients, doubled_area * 0.5, normal / doubled_area[:, None]


def _shared_corner_pairs(
    faces: np.ndarray,
    first: int,
    second: int,
) -> list[tuple[int, int]]:
    second_lookup = {
        int(vertex): corner for corner, vertex in enumerate(faces[second])
    }
    pairs = [
        (corner, second_lookup[int(vertex)])
        for corner, vertex in enumerate(faces[first])
        if int(vertex) in second_lookup
    ]
    if len(pairs) != 2:
        raise ValueError("adjacent faces must share exactly one edge")
    return pairs


def _miq_face_charts(
    faces: np.ndarray,
    uv_faces: np.ndarray,
    face_adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seam_edges = np.zeros(face_adjacency.shape[0], dtype=bool)
    for edge_id, (first, second) in enumerate(face_adjacency):
        pairs = _shared_corner_pairs(faces, int(first), int(second))
        seam_edges[edge_id] = any(
            uv_faces[int(first), first_corner]
            != uv_faces[int(second), second_corner]
            for first_corner, second_corner in pairs
        )
    kept = face_adjacency[~seam_edges]
    if kept.size:
        graph = sp.coo_matrix(
            (
                np.ones(kept.shape[0] * 2),
                (
                    np.concatenate((kept[:, 0], kept[:, 1])),
                    np.concatenate((kept[:, 1], kept[:, 0])),
                ),
            ),
            shape=(faces.shape[0], faces.shape[0]),
        ).tocsr()
    else:
        graph = sp.csr_matrix((faces.shape[0], faces.shape[0]))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64), seam_edges


def _build_result(
    raw: MIQRawOutput,
    vertices: np.ndarray,
    faces: np.ndarray,
    desired_gradient_u: np.ndarray,
    desired_gradient_v: np.ndarray,
    face_adjacency: np.ndarray,
    *,
    gradient_scale: float,
    stiffness: float,
    iterations: int,
    local_iterations: int,
    direct_round: bool,
    round_singularities: bool,
    executable: Path,
    solver_stdout: str,
) -> MIQParameterizationResult:
    corner_uv = raw.uv_vertices[raw.uv_faces]
    face_charts, seam_edges = _miq_face_charts(
        faces, raw.uv_faces, face_adjacency
    )
    gradients, areas, normals = _face_gradient_data(vertices, faces)
    reconstructed_u = np.einsum("fvc,fv->fc", gradients, corner_uv[:, :, 0])
    reconstructed_v = np.einsum("fvc,fv->fc", gradients, corner_uv[:, :, 1])
    target_u = gradient_scale * desired_gradient_u
    target_v = gradient_scale * desired_gradient_v
    # A 4-RoSy frame has four equivalent labels. MIQ deliberately re-combs
    # these labels before solving, so comparing against one fixed label would
    # report a large false residual on faces rotated by a quarter turn.
    candidate_u = np.stack((target_u, target_v, -target_u, -target_v))
    candidate_v = np.stack((target_v, -target_u, -target_v, target_u))
    residual_sq_by_turn = np.sum(
        np.square(reconstructed_u[None, ...] - candidate_u), axis=-1
    )
    residual_sq_by_turn += np.sum(
        np.square(reconstructed_v[None, ...] - candidate_v), axis=-1
    )
    alignment_quarter_turns = np.argmin(residual_sq_by_turn, axis=0)
    residual_sq = np.take_along_axis(
        residual_sq_by_turn,
        alignment_quarter_turns[None, :],
        axis=0,
    )[0]
    per_face_residual = np.sqrt(residual_sq)
    rms = float(np.sqrt(np.sum(areas * residual_sq) / np.sum(areas)))
    target_energy = np.sum(np.square(target_u), axis=-1)
    target_energy += np.sum(np.square(target_v), axis=-1)
    target_rms = float(
        np.sqrt(np.sum(areas * target_energy) / np.sum(areas))
    )
    orientation = np.einsum(
        "fc,fc->f", np.cross(reconstructed_u, reconstructed_v), normals
    )
    flipped = int(np.count_nonzero(orientation <= 0.0))
    chart_count = int(face_charts.max() + 1) if face_charts.size else 0
    return MIQParameterizationResult(
        corner_uv=corner_uv,
        face_charts=face_charts,
        uv_vertices=raw.uv_vertices,
        uv_faces=raw.uv_faces,
        seam_edges=seam_edges,
        seam_corners=raw.seam_corners,
        mismatch=raw.mismatch,
        singular=raw.singular,
        singularity_index=raw.singularity_index,
        reconstructed_gradient_u=reconstructed_u,
        reconstructed_gradient_v=reconstructed_v,
        alignment_quarter_turns=alignment_quarter_turns.astype(np.int64),
        per_face_residual=per_face_residual,
        rms_gradient_residual=rms,
        relative_gradient_residual=rms / max(target_rms, 1e-14),
        flipped_face_count=flipped,
        flipped_face_fraction=flipped / max(faces.shape[0], 1),
        cut_vertex_count=int(raw.uv_vertices.shape[0]),
        chart_count=chart_count,
        gradient_scale=float(gradient_scale),
        stiffness=float(stiffness),
        iterations=int(iterations),
        local_iterations=int(local_iterations),
        direct_round=bool(direct_round),
        round_singularities=bool(round_singularities),
        executable=str(executable),
        solver_stdout=solver_stdout,
    )


def solve_miq_parameterization(
    vertices,
    faces,
    direction_u,
    direction_v,
    target_size,
    desired_gradient_u,
    desired_gradient_v,
    face_adjacency,
    work_dir: str | Path,
    *,
    executable: Optional[str | Path] = None,
    gradient_scale: float = 1.0,
    stiffness: float = 5.0,
    iterations: int = 5,
    local_iterations: int = 5,
    direct_round: bool = False,
    round_singularities: bool = True,
    timeout: Optional[float] = 3600.0,
) -> MIQParameterizationResult:
    """Run libigl/CoMISo MIQ and convert its indexed UV map to face corners."""
    if gradient_scale <= 0.0:
        raise ValueError("gradient_scale must be positive")
    if stiffness < 0.0:
        raise ValueError("stiffness must be non-negative")
    if iterations < 0 or local_iterations < 1:
        raise ValueError("MIQ iteration counts are invalid")
    vertices = _as_numpy(vertices, np.float64)
    faces = _as_numpy(faces, np.int64)
    direction_u = _as_numpy(direction_u, np.float64)
    direction_v = _as_numpy(direction_v, np.float64)
    target_size = _as_numpy(target_size, np.float64).reshape(-1)
    desired_gradient_u = _as_numpy(desired_gradient_u, np.float64)
    desired_gradient_v = _as_numpy(desired_gradient_v, np.float64)
    face_adjacency = _as_numpy(face_adjacency, np.int64)
    _validate_inputs(vertices, faces, direction_u, direction_v, target_size)
    if desired_gradient_u.shape != direction_u.shape:
        raise ValueError("desired_gradient_u must have shape [F, 3]")
    if desired_gradient_v.shape != direction_v.shape:
        raise ValueError("desired_gradient_v must have shape [F, 3]")
    if face_adjacency.ndim != 2 or face_adjacency.shape[1] != 2:
        raise ValueError("face_adjacency must have shape [E, 2]")

    solver = _resolve_executable(executable)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = write_miq_input(
        work_dir / "miq_input.txt",
        vertices,
        faces,
        direction_u,
        direction_v,
        target_size,
    )
    output_path = work_dir / "miq_output.txt"
    command = [
        str(solver),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--gradient-scale",
        str(float(gradient_scale)),
        "--stiffness",
        str(float(stiffness)),
        "--iterations",
        str(int(iterations)),
        "--local-iterations",
        str(int(local_iterations)),
        "--direct-round",
        "1" if direct_round else "0",
        "--round-singularities",
        "1" if round_singularities else "0",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"MIQ exceeded its {timeout:g}s timeout") from exc
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"MIQ failed with exit code {completed.returncode}: {details}"
        )
    if not output_path.is_file():
        raise RuntimeError("MIQ exited successfully but produced no output file")
    raw = load_miq_output(
        output_path,
        expected_face_count=faces.shape[0],
        expected_vertex_count=vertices.shape[0],
    )
    return _build_result(
        raw,
        vertices,
        faces,
        desired_gradient_u,
        desired_gradient_v,
        face_adjacency,
        gradient_scale=gradient_scale,
        stiffness=stiffness,
        iterations=iterations,
        local_iterations=local_iterations,
        direct_round=direct_round,
        round_singularities=round_singularities,
        executable=solver,
        solver_stdout=completed.stdout.strip(),
    )


def save_miq_parameterization(
    result: MIQParameterizationResult,
    vertices,
    faces,
    output_dir: str | Path,
) -> Path:
    """Save indexed and per-corner MIQ data plus reproducibility metadata."""
    vertices = _as_numpy(vertices, np.float64)
    faces = _as_numpy(faces, np.int64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "miq_parameterization.npz",
        uv_vertices=result.uv_vertices,
        uv_faces=result.uv_faces,
        corner_uv=result.corner_uv,
        face_charts=result.face_charts,
        seam_edges=result.seam_edges,
        seam_corners=result.seam_corners,
        mismatch=result.mismatch,
        singular=result.singular,
        singularity_index=result.singularity_index,
        alignment_quarter_turns=result.alignment_quarter_turns,
        per_face_residual=result.per_face_residual,
    )
    with (output_dir / "miq_parameterization.obj").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("# HKTex2Quad libigl/CoMISo MIQ parameterization\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for uv in result.uv_vertices:
            handle.write(f"vt {uv[0]:.17g} {uv[1]:.17g}\n")
        for face, uv_face in zip(faces + 1, result.uv_faces + 1):
            handle.write(
                "f "
                + " ".join(
                    f"{vertex_id}/{uv_id}"
                    for vertex_id, uv_id in zip(face, uv_face)
                )
                + "\n"
            )
    report = {
        "format": "hktex2quad-miq-parameterization-v1",
        "solver": "libigl v2.5.0 reference MIQ with CoMISo mixed integers",
        "executable": result.executable,
        "gradient_scale": result.gradient_scale,
        "stiffness": result.stiffness,
        "iterations": result.iterations,
        "local_iterations": result.local_iterations,
        "direct_round": result.direct_round,
        "round_singularities": result.round_singularities,
        "rms_gradient_residual": result.rms_gradient_residual,
        "relative_gradient_residual": result.relative_gradient_residual,
        "residual_convention": (
            "minimum over the four equivalent C4 frame labels per face"
        ),
        "alignment_quarter_turn_histogram": {
            str(turn): int(count)
            for turn, count in enumerate(
                np.bincount(result.alignment_quarter_turns, minlength=4)
            )
        },
        "flipped_face_count": result.flipped_face_count,
        "flipped_face_fraction": result.flipped_face_fraction,
        "cut_vertex_count": result.cut_vertex_count,
        "seam_edge_count": int(result.seam_edges.sum()),
        "chart_count": result.chart_count,
        "miq_singularity_count": int(result.singular.sum()),
        "solver_stdout": result.solver_stdout,
        "license_note": (
            "The optional MIQ executable links CoMISo and is subject to its "
            "copyleft license terms."
        ),
    }
    report_path = output_dir / "miq_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report_path
