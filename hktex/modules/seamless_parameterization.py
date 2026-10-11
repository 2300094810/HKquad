"""Cut-mesh seamless parameterization with refined integer transitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr

__all__ = [
    "FixedTransitionParameterizationResult",
    "quarter_turn_matrix",
    "save_fixed_transition_parameterization",
    "solve_fixed_transition_parameterization",
]


class FixedTransitionParameterizationResult(NamedTuple):
    """Per-corner seamless UV map with fixed integer transitions."""

    corner_uv: np.ndarray
    face_charts: np.ndarray
    edge_quarter_turns: np.ndarray
    edge_translations: np.ndarray
    seam_edges: np.ndarray
    reconstructed_gradient_u: np.ndarray
    reconstructed_gradient_v: np.ndarray
    per_face_residual: np.ndarray
    transition_residual: np.ndarray
    rms_gradient_residual: float
    max_transition_residual: float
    flipped_face_count: int
    solver_iterations: int
    cut_vertex_count: int
    integer_refinement_iterations: int
    translation_update_count: int
    relative_gradient_residual: float
    rms_transition_residual: float
    flipped_face_fraction: float


def _as_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def quarter_turn_matrix(turns: int) -> np.ndarray:
    """Map source UV coordinates into a target chart for ``turns * pi/2``."""
    turns = int(turns) % 4
    matrices = (
        ((1.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (-1.0, 0.0)),
        ((-1.0, 0.0), (0.0, -1.0)),
        ((0.0, -1.0), (1.0, 0.0)),
    )
    return np.asarray(matrices[turns], dtype=np.float64)


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


def _shared_corners(
    faces: np.ndarray,
    source: int,
    target: int,
) -> list[tuple[int, int]]:
    target_positions = {
        int(vertex): corner for corner, vertex in enumerate(faces[target])
    }
    shared = [
        (source_corner, target_positions[int(vertex)])
        for source_corner, vertex in enumerate(faces[source])
        if int(vertex) in target_positions
    ]
    if len(shared) != 2:
        raise ValueError("every face adjacency must share exactly one edge")
    return shared


def _face_components(
    face_count: int,
    adjacency: np.ndarray,
    blocked: Optional[np.ndarray] = None,
) -> tuple[int, np.ndarray]:
    if blocked is None:
        blocked = np.zeros(adjacency.shape[0], dtype=bool)
    kept = adjacency[~blocked]
    if kept.size:
        graph = sp.coo_matrix(
            (
                np.ones(kept.shape[0] * 2),
                (
                    np.concatenate((kept[:, 0], kept[:, 1])),
                    np.concatenate((kept[:, 1], kept[:, 0])),
                ),
            ),
            shape=(face_count, face_count),
        ).tocsr()
    else:
        graph = sp.csr_matrix((face_count, face_count))
    return connected_components(graph, directed=False)


def _cut_corner_classes(
    faces: np.ndarray,
    adjacency: np.ndarray,
    seams: np.ndarray,
) -> np.ndarray:
    """Return cut-mesh vertex ids for every face corner.

    Face corners are joined only through non-seam primal edges.  This is
    deliberately different from merely splitting the dual graph into face
    components: cutting a non-separating cycle can leave every face in one
    component while still duplicating vertices along the two sides of the
    cut.
    """
    corner_count = faces.shape[0] * 3
    parent = np.arange(corner_count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for edge_id, (source, target) in enumerate(adjacency):
        if seams[edge_id]:
            continue
        for source_corner, target_corner in _shared_corners(
            faces, int(source), int(target)
        ):
            union(
                3 * int(source) + source_corner,
                3 * int(target) + target_corner,
            )

    roots = np.array([find(index) for index in range(corner_count)])
    _, classes = np.unique(roots, return_inverse=True)
    return classes.reshape(-1, 3)


def _solve_cut_initialization(
    faces: np.ndarray,
    gradients: np.ndarray,
    areas: np.ndarray,
    desired_u: np.ndarray,
    desired_v: np.ndarray,
    cut_corners: np.ndarray,
    face_charts: np.ndarray,
) -> np.ndarray:
    """Integrate gradients on the vertex-duplicated cut mesh."""
    face_count = faces.shape[0]
    cut_vertex_count = int(cut_corners.max()) + 1 if cut_corners.size else 0
    rows = np.broadcast_to(
        np.arange(face_count * 3).reshape(face_count, 3, 1),
        (face_count, 3, 3),
    )
    columns = np.broadcast_to(cut_corners[:, None, :], (face_count, 3, 3))
    values = gradients.transpose(0, 2, 1)
    operator = sp.coo_matrix(
        (values.reshape(-1), (rows.reshape(-1), columns.reshape(-1))),
        shape=(face_count * 3, cut_vertex_count),
    ).tocsr()
    weights = np.repeat(np.sqrt(areas), 3)
    weighted = sp.diags(weights) @ operator
    chart_count = int(face_charts.max()) + 1 if face_charts.size else 0
    anchor_columns = np.array(
        [
            cut_corners[int(np.flatnonzero(face_charts == chart)[0]), 0]
            for chart in range(chart_count)
        ],
        dtype=np.int64,
    )
    anchor = sp.coo_matrix(
        (
            np.ones(chart_count),
            (np.arange(chart_count), anchor_columns),
        ),
        shape=(chart_count, cut_vertex_count),
    ).tocsr()
    system = sp.vstack((weighted, anchor), format="csr")

    def solve(target: np.ndarray) -> np.ndarray:
        rhs = np.concatenate((target.reshape(-1) * weights, np.zeros(chart_count)))
        return lsmr(system, rhs, atol=1e-10, btol=1e-10)[0]

    local_uv = np.stack((solve(desired_u), solve(desired_v)), axis=-1)
    return local_uv[cut_corners]


def _estimate_integer_translations(
    corner_uv: np.ndarray,
    faces: np.ndarray,
    adjacency: np.ndarray,
    turns: np.ndarray,
    seams: np.ndarray,
) -> np.ndarray:
    translations = np.zeros((adjacency.shape[0], 2), dtype=np.int64)
    for edge_id in np.flatnonzero(seams):
        source, target = adjacency[edge_id]
        rotation = quarter_turn_matrix(turns[edge_id])
        deltas = []
        for source_corner, target_corner in _shared_corners(
            faces, int(source), int(target)
        ):
            source_uv = corner_uv[source, source_corner]
            target_uv = corner_uv[target, target_corner]
            deltas.append(target_uv - rotation @ source_uv)
        translations[edge_id] = np.rint(np.mean(deltas, axis=0)).astype(np.int64)
    return translations


def _initial_corner_uv_and_translations(
    faces: np.ndarray,
    adjacency: np.ndarray,
    turns: np.ndarray,
    seams: np.ndarray,
    gradients: np.ndarray,
    areas: np.ndarray,
    desired_u: np.ndarray,
    desired_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    _, face_charts = _face_components(
        faces.shape[0], adjacency, blocked=seams
    )
    cut_corners = _cut_corner_classes(faces, adjacency, seams)
    corner_uv = _solve_cut_initialization(
        faces,
        gradients,
        areas,
        desired_u,
        desired_v,
        cut_corners,
        face_charts,
    )
    translations = _estimate_integer_translations(
        corner_uv, faces, adjacency, turns, seams
    )
    cut_vertex_count = int(cut_corners.max()) + 1 if cut_corners.size else 0
    return corner_uv, face_charts, translations, cut_vertex_count


def solve_fixed_transition_parameterization(
    vertices,
    faces,
    desired_gradient_u,
    desired_gradient_v,
    face_adjacency,
    edge_quarter_turns,
    seam_edges,
    *,
    continuity_weight: float = 1e4,
    anchor_weight: float = 1.0,
    atol: float = 1e-10,
    btol: float = 1e-10,
    max_iterations: Optional[int] = None,
    integer_refinement_iterations: int = 5,
) -> FixedTransitionParameterizationResult:
    """Solve UVs with refined integer translations and fixed C4 rotations.

    The relaxed initialization is solved on an actual cut mesh whose seam
    vertices are duplicated.  Integer translations are then alternated with
    sparse continuous solves until they stabilize or the requested update
    limit is reached.  This is still an approximate alternative to a joint
    mixed-integer solver, but it avoids the previous one-shot rounding and
    correctly represents non-separating cuts.
    """
    vertices = _as_numpy(vertices, np.float64)
    faces = _as_numpy(faces, np.int64)
    desired_u = _as_numpy(desired_gradient_u, np.float64)
    desired_v = _as_numpy(desired_gradient_v, np.float64)
    adjacency = _as_numpy(face_adjacency, np.int64)
    turns = _as_numpy(edge_quarter_turns, np.int64).reshape(-1) % 4
    seams = _as_numpy(seam_edges, bool).reshape(-1)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if desired_u.shape != (faces.shape[0], 3) or desired_v.shape != desired_u.shape:
        raise ValueError("desired gradients must have shape [F, 3]")
    if adjacency.ndim != 2 or adjacency.shape[1] != 2:
        raise ValueError("face_adjacency must have shape [E, 2]")
    if turns.shape != (adjacency.shape[0],) or seams.shape != turns.shape:
        raise ValueError("edge transition arrays must have shape [E]")
    if continuity_weight <= 0 or anchor_weight <= 0:
        raise ValueError("constraint weights must be positive")
    if integer_refinement_iterations < 1:
        raise ValueError("integer_refinement_iterations must be positive")
    if not (
        np.isfinite(vertices).all()
        and np.isfinite(desired_u).all()
        and np.isfinite(desired_v).all()
    ):
        raise ValueError("parameterization inputs must be finite")

    gradients, areas, face_normals = _face_gradient_data(vertices, faces)
    initial_uv, face_charts, translations, cut_vertex_count = (
        _initial_corner_uv_and_translations(
            faces,
            adjacency,
            turns,
            seams,
            gradients,
            areas,
            desired_u,
            desired_v,
        )
    )

    face_count = faces.shape[0]
    cut_corners = _cut_corner_classes(faces, adjacency, seams)
    cut_vertex_count = int(cut_corners.max()) + 1 if cut_corners.size else 0
    variable_count = cut_vertex_count * 2
    sqrt_area = np.sqrt(areas)
    rows_u = np.broadcast_to(
        (6 * np.arange(face_count)[:, None, None] + np.arange(3)[None, :, None]),
        (face_count, 3, 3),
    )
    rows_v = rows_u + 3
    cut_ids = np.broadcast_to(cut_corners[:, None, :], (face_count, 3, 3))
    columns_u = 2 * cut_ids
    columns_v = columns_u + 1
    weighted_values = gradients.transpose(0, 2, 1) * sqrt_area[:, None, None]
    base_matrix = sp.coo_matrix(
        (
            np.concatenate((weighted_values.reshape(-1), weighted_values.reshape(-1))),
            (
                np.concatenate((rows_u.reshape(-1), rows_v.reshape(-1))),
                np.concatenate((columns_u.reshape(-1), columns_v.reshape(-1))),
            ),
        ),
        shape=(face_count * 6, variable_count),
    ).tocsr()
    base_rhs = np.empty(face_count * 6, dtype=np.float64)
    base_rhs.reshape(face_count, 6)[:, :3] = desired_u * sqrt_area[:, None]
    base_rhs.reshape(face_count, 6)[:, 3:] = desired_v * sqrt_area[:, None]

    transition_rows: list[int] = []
    transition_columns: list[int] = []
    transition_values: list[float] = []
    transition_edge_ids: list[int] = []
    transition_coordinates: list[int] = []
    constraint_row = 0
    # Non-seam corners already share the same cut-vertex variable and therefore
    # satisfy continuity exactly.  Only true cut edges need a chart-transition
    # equation.  The previous per-face-corner formulation redundantly imposed
    # all adjacency constraints with a huge penalty and was badly conditioned.
    for edge_id in np.flatnonzero(seams):
        source, target = adjacency[edge_id]
        rotation = quarter_turn_matrix(turns[edge_id])
        for source_corner, target_corner in _shared_corners(
            faces, int(source), int(target)
        ):
            source_variable = 2 * cut_corners[int(source), source_corner]
            target_variable = 2 * cut_corners[int(target), target_corner]
            for coordinate in range(2):
                transition_rows.append(constraint_row)
                transition_columns.append(target_variable + coordinate)
                transition_values.append(1.0)
                for source_coordinate in range(2):
                    transition_rows.append(constraint_row)
                    transition_columns.append(source_variable + source_coordinate)
                    transition_values.append(
                        -float(rotation[coordinate, source_coordinate])
                    )
                transition_edge_ids.append(edge_id)
                transition_coordinates.append(coordinate)
                constraint_row += 1

    transition_matrix = sp.coo_matrix(
        (transition_values, (transition_rows, transition_columns)),
        shape=(constraint_row, variable_count),
    ).tocsr()
    transition_edge_ids_array = np.asarray(transition_edge_ids, dtype=np.int64)
    transition_coordinates_array = np.asarray(
        transition_coordinates, dtype=np.int64
    )

    full_component_count, full_components = _face_components(
        face_count, adjacency, blocked=seams
    )
    anchor_rows: list[int] = []
    anchor_columns: list[int] = []
    anchor_values: list[float] = []
    anchor_rhs: list[float] = []
    anchor_row = 0
    for component in range(full_component_count):
        anchor_face = int(np.flatnonzero(full_components == component)[0])
        anchor_uv = initial_uv[anchor_face, 0]
        anchor_variable = 2 * cut_corners[anchor_face, 0]
        for coordinate in range(2):
            anchor_rows.append(anchor_row)
            anchor_columns.append(anchor_variable + coordinate)
            anchor_values.append(1.0)
            anchor_rhs.append(float(anchor_uv[coordinate]))
            anchor_row += 1

    anchor_matrix = sp.coo_matrix(
        (anchor_values, (anchor_rows, anchor_columns)),
        shape=(anchor_row, variable_count),
    ).tocsr()

    solver_iterations = 0
    solver_stop_codes: list[int] = []

    initial_cut_uv = np.zeros((cut_vertex_count, 2), dtype=np.float64)
    initial_counts = np.zeros(cut_vertex_count, dtype=np.int64)
    np.add.at(initial_cut_uv, cut_corners.reshape(-1), initial_uv.reshape(-1, 2))
    np.add.at(initial_counts, cut_corners.reshape(-1), 1)
    initial_cut_uv /= np.maximum(initial_counts[:, None], 1)

    def solve_continuous(current_translations, weight):
        nonlocal solver_iterations
        transition_rhs = current_translations[
            transition_edge_ids_array, transition_coordinates_array
        ].astype(np.float64)
        # ``continuity_weight`` is an energy coefficient, hence its square
        # root scales least-squares rows.  This keeps the intended 1e4 penalty
        # without creating a 1e8 normal-equation condition ratio.
        transition_scale = np.sqrt(float(weight))
        blocks = [base_matrix]
        rhs_blocks = [base_rhs]
        if transition_matrix.shape[0]:
            blocks.append(transition_scale * transition_matrix)
            rhs_blocks.append(transition_scale * transition_rhs)
        blocks.append(anchor_weight * anchor_matrix)
        rhs_blocks.append(anchor_weight * np.asarray(anchor_rhs))
        system = sp.vstack(blocks, format="csr")
        rhs = np.concatenate(rhs_blocks)
        iteration_limit = max_iterations
        if iteration_limit is None:
            iteration_limit = max(1000, min(20000, 4 * variable_count))
        result = lsmr(
            system,
            rhs,
            atol=atol,
            btol=btol,
            conlim=1e12,
            maxiter=iteration_limit,
            x0=initial_cut_uv.reshape(-1),
        )
        solver_iterations += int(result[2])
        solver_stop_codes.append(int(result[1]))
        cut_uv = result[0].reshape(cut_vertex_count, 2)
        return cut_uv[cut_corners]

    corner_uv = initial_uv
    translation_update_count = 0
    solve_count = 0
    translations_changed = False
    for _ in range(integer_refinement_iterations):
        corner_uv = solve_continuous(translations, float(continuity_weight))
        solve_count += 1
        refined = _estimate_integer_translations(
            corner_uv, faces, adjacency, turns, seams
        )
        changed = np.any(refined != translations, axis=1)
        translation_update_count += int(np.count_nonzero(changed))
        translations_changed = bool(changed.any())
        translations = refined
        if not translations_changed:
            break

    # If the final rounding changed an integer, return UVs that satisfy that
    # new transition rather than the stale continuous solution.
    if translations_changed:
        corner_uv = solve_continuous(translations, float(continuity_weight))
        solve_count += 1

    reconstructed_u = np.einsum("fvc,fv->fc", gradients, corner_uv[:, :, 0])
    reconstructed_v = np.einsum("fvc,fv->fc", gradients, corner_uv[:, :, 1])
    residual_u = np.sum(np.square(reconstructed_u - desired_u), axis=-1)
    residual_v = np.sum(np.square(reconstructed_v - desired_v), axis=-1)
    per_face_residual = np.sqrt(residual_u + residual_v)
    rms = float(
        np.sqrt(
            np.sum(areas * np.square(per_face_residual)) / np.sum(areas)
        )
    )

    transition_residual = np.zeros(adjacency.shape[0], dtype=np.float64)
    for edge_id, (source, target) in enumerate(adjacency):
        rotation = quarter_turn_matrix(turns[edge_id])
        translation = translations[edge_id]
        errors = []
        for source_corner, target_corner in _shared_corners(
            faces, int(source), int(target)
        ):
            predicted = rotation @ corner_uv[source, source_corner] + translation
            errors.append(np.linalg.norm(corner_uv[target, target_corner] - predicted))
        transition_residual[edge_id] = max(errors)

    target_rms = float(
        np.sqrt(
            np.sum(
                areas
                * (
                    np.sum(np.square(desired_u), axis=-1)
                    + np.sum(np.square(desired_v), axis=-1)
                )
            )
            / np.sum(areas)
        )
    )
    relative_rms = rms / max(target_rms, 1e-14)
    transition_rms = float(
        np.sqrt(np.mean(np.square(transition_residual)))
        if transition_residual.size
        else 0.0
    )

    orientation = np.einsum(
        "fc,fc->f",
        np.cross(reconstructed_u, reconstructed_v),
        face_normals,
    )
    flipped_face_count = int(np.count_nonzero(orientation <= 0.0))
    return FixedTransitionParameterizationResult(
        corner_uv=corner_uv,
        face_charts=face_charts,
        edge_quarter_turns=turns,
        edge_translations=translations,
        seam_edges=seams,
        reconstructed_gradient_u=reconstructed_u,
        reconstructed_gradient_v=reconstructed_v,
        per_face_residual=per_face_residual,
        transition_residual=transition_residual,
        rms_gradient_residual=rms,
        max_transition_residual=float(
            transition_residual.max() if transition_residual.size else 0.0
        ),
        flipped_face_count=flipped_face_count,
        solver_iterations=solver_iterations,
        cut_vertex_count=cut_vertex_count,
        integer_refinement_iterations=solve_count,
        translation_update_count=translation_update_count,
        relative_gradient_residual=relative_rms,
        rms_transition_residual=transition_rms,
        flipped_face_fraction=(flipped_face_count / max(face_count, 1)),
    )


def save_fixed_transition_parameterization(
    result: FixedTransitionParameterizationResult,
    vertices,
    faces,
    output_dir: str | Path,
) -> Path:
    """Save a per-corner UV mesh and transition diagnostics."""
    vertices = _as_numpy(vertices, np.float64)
    faces = _as_numpy(faces, np.int64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "seamless_parameterization.npz",
        corner_uv=result.corner_uv,
        face_charts=result.face_charts,
        edge_quarter_turns=result.edge_quarter_turns,
        edge_translations=result.edge_translations,
        seam_edges=result.seam_edges,
        per_face_residual=result.per_face_residual,
        transition_residual=result.transition_residual,
        cut_vertex_count=np.asarray(result.cut_vertex_count, dtype=np.int64),
    )
    with (output_dir / "seamless_parameterization.obj").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("# HKTex2Quad fixed-transition seamless parameterization\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for uv in result.corner_uv.reshape(-1, 2):
            handle.write(f"vt {uv[0]:.17g} {uv[1]:.17g}\n")
        for face_id, face in enumerate(faces):
            vertex_ids = face + 1
            uv_ids = 3 * face_id + np.arange(3) + 1
            handle.write(
                "f "
                + " ".join(
                    f"{vertex_id}/{uv_id}"
                    for vertex_id, uv_id in zip(vertex_ids, uv_ids)
                )
                + "\n"
            )

    report = {
        "format": "hktex2quad-fixed-transition-parameterization-v1",
        "solver": (
            "cut-mesh alternating rounded integer transitions and sparse "
            "least squares"
        ),
        "rms_gradient_residual": result.rms_gradient_residual,
        "relative_gradient_residual": result.relative_gradient_residual,
        "rms_transition_residual": result.rms_transition_residual,
        "max_transition_residual": result.max_transition_residual,
        "flipped_face_count": result.flipped_face_count,
        "flipped_face_fraction": result.flipped_face_fraction,
        "solver_iterations": result.solver_iterations,
        "cut_vertex_count": result.cut_vertex_count,
        "integer_refinement_iterations": result.integer_refinement_iterations,
        "translation_update_count": result.translation_update_count,
        "seam_edge_count": int(result.seam_edges.sum()),
        "chart_count": int(result.face_charts.max() + 1),
        "limitation": (
            "Integer translations are optimized by alternating rounding, not "
            "jointly by a true MIQ/CoMISo mixed-integer solve."
        ),
    }
    report_path = output_dir / "seamless_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path
