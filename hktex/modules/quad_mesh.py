"""Integer-grid quad extraction, seam welding, and quality metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "QuadMeshResult",
    "evaluate_quad_mesh",
    "extract_integer_grid_quads",
    "save_quad_mesh",
    "weld_quad_mesh",
]


class QuadMeshResult(NamedTuple):
    vertices: np.ndarray
    quads: np.ndarray
    grid_uv: np.ndarray
    vertex_charts: np.ndarray
    chart_offsets: np.ndarray
    source_triangle: np.ndarray
    metrics: dict


def _as_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _barycentric_2d(point: np.ndarray, triangle: np.ndarray) -> Optional[np.ndarray]:
    edge_0 = triangle[1] - triangle[0]
    edge_1 = triangle[2] - triangle[0]
    relative = point - triangle[0]
    denominator = edge_0[0] * edge_1[1] - edge_0[1] * edge_1[0]
    if abs(denominator) <= 1e-14:
        return None
    b1 = (relative[0] * edge_1[1] - relative[1] * edge_1[0]) / denominator
    b2 = (edge_0[0] * relative[1] - edge_0[1] * relative[0]) / denominator
    return np.array((1.0 - b1 - b2, b1, b2), dtype=np.float64)


def _sample_chart_lattice(
    vertices: np.ndarray,
    faces: np.ndarray,
    corner_uv: np.ndarray,
    chart_faces: np.ndarray,
    offset: np.ndarray,
    *,
    inside_epsilon: float,
) -> tuple[
    dict[tuple[int, int], list[tuple[np.ndarray, int]]],
    dict[tuple[int, int], list[int]],
]:
    # A non-injective UV map can cover the same lattice coordinate on several
    # unrelated parts of the surface. Keep those samples separate here. The
    # old implementation averaged them immediately, creating long edges that
    # jumped between distinct sheets of an overlapping parameterization.
    point_candidates: dict[tuple[int, int], list[tuple[np.ndarray, int]]] = {}
    candidate_cells: dict[tuple[int, int], list[int]] = {}

    for face_id in chart_faces:
        uv_triangle = corner_uv[face_id] + offset
        xyz_triangle = vertices[faces[face_id]]
        minimum = uv_triangle.min(axis=0)
        maximum = uv_triangle.max(axis=0)

        lower = np.ceil(minimum - inside_epsilon).astype(np.int64)
        upper = np.floor(maximum + inside_epsilon).astype(np.int64)
        for i in range(int(lower[0]), int(upper[0]) + 1):
            for j in range(int(lower[1]), int(upper[1]) + 1):
                key = (i, j)
                barycentric = _barycentric_2d(np.array(key), uv_triangle)
                if barycentric is None or np.min(barycentric) < -inside_epsilon:
                    continue
                position = barycentric @ xyz_triangle
                point_candidates.setdefault(key, []).append(
                    (position, int(face_id))
                )

        cell_lower = np.ceil(minimum - 0.5 - inside_epsilon).astype(np.int64)
        cell_upper = np.floor(maximum - 0.5 + inside_epsilon).astype(np.int64)
        for i in range(int(cell_lower[0]), int(cell_upper[0]) + 1):
            for j in range(int(cell_lower[1]), int(cell_upper[1]) + 1):
                center = np.array((i + 0.5, j + 0.5))
                barycentric = _barycentric_2d(center, uv_triangle)
                if barycentric is not None and np.min(barycentric) >= -inside_epsilon:
                    candidate_cells.setdefault((i, j), []).append(int(face_id))

    return point_candidates, candidate_cells


def _face_neighborhoods(
    faces: np.ndarray,
) -> tuple[list[set[int]], list[set[int]]]:
    """Build edge and vertex face neighborhoods for local sheet tracking."""
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    vertex_to_faces: dict[int, list[int]] = {}
    for face_id, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces.setdefault(int(vertex), []).append(face_id)
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_to_faces.setdefault(edge, []).append(face_id)

    edge_neighbors = [set() for _ in range(faces.shape[0])]
    touching_neighbors = [set() for _ in range(faces.shape[0])]
    for incident in edge_to_faces.values():
        for face_id in incident:
            edge_neighbors[face_id].update(incident)
    for incident in vertex_to_faces.values():
        for face_id in incident:
            touching_neighbors[face_id].update(incident)
    for face_id in range(faces.shape[0]):
        edge_neighbors[face_id].discard(face_id)
        touching_neighbors[face_id].discard(face_id)
    return edge_neighbors, touching_neighbors


def _candidate_components(
    face_ids: list[int],
    touching_neighbors: list[set[int]],
) -> list[tuple[int, ...]]:
    remaining = set(face_ids)
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = {root}
        queue = [root]
        while queue:
            current = queue.pop()
            linked = touching_neighbors[current] & remaining
            remaining.difference_update(linked)
            component.update(linked)
            queue.extend(linked)
        components.append(tuple(sorted(component)))
    return components


def _local_point_candidate(
    candidates: list[tuple[np.ndarray, int]],
    source_faces: tuple[int, ...],
    edge_neighbors: list[set[int]],
    *,
    max_face_hops: int,
) -> Optional[tuple[np.ndarray, int, int]]:
    """Find the nearest topological copy of a UV lattice point."""
    by_face: dict[int, list[np.ndarray]] = {}
    for position, face_id in candidates:
        by_face.setdefault(face_id, []).append(position)
    candidate_faces = set(by_face)
    visited = set(source_faces)
    frontier = set(source_faces)
    for distance in range(max_face_hops + 1):
        matches = sorted(frontier & candidate_faces)
        if matches:
            positions = [
                position
                for face_id in matches
                for position in by_face[face_id]
            ]
            return np.mean(positions, axis=0), matches[0], distance
        next_frontier: set[int] = set()
        for face_id in frontier:
            next_frontier.update(edge_neighbors[face_id])
        next_frontier.difference_update(visited)
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier
    return None


def _complete_sheet_cells(
    points: dict[tuple[int, int], list[tuple[np.ndarray, int]]],
    candidates: dict[tuple[int, int], list[int]],
    edge_neighbors: list[set[int]],
    touching_neighbors: list[set[int]],
    *,
    max_corner_face_hops: int,
) -> tuple[list[tuple[tuple[int, int], int, np.ndarray, np.ndarray]], dict]:
    complete = []
    occurrence_count = 0
    rejected = 0
    overlap_keys = 0
    maximum_hops = 0
    for (i, j), center_faces in sorted(candidates.items()):
        components = _candidate_components(center_faces, touching_neighbors)
        occurrence_count += len(components)
        if len(components) > 1:
            overlap_keys += 1
        corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
        for component in components:
            selected = []
            for corner in corners:
                if corner not in points:
                    selected = []
                    break
                match = _local_point_candidate(
                    points[corner],
                    component,
                    edge_neighbors,
                    max_face_hops=max_corner_face_hops,
                )
                if match is None:
                    selected = []
                    break
                selected.append(match)
            if not selected:
                rejected += 1
                continue
            positions = np.asarray([value[0] for value in selected])
            source_faces_array = np.asarray(
                [value[1] for value in selected], dtype=np.int64
            )
            maximum_hops = max(maximum_hops, *(value[2] for value in selected))
            complete.append(
                ((i, j), int(component[0]), positions, source_faces_array)
            )
    diagnostics = {
        "candidate_cell_occurrence_count": occurrence_count,
        "rejected_incomplete_cell_count": rejected,
        "overlapping_uv_cell_key_count": overlap_keys,
        "maximum_corner_face_hops": maximum_hops,
    }
    return complete, diagnostics


def _choose_chart_offset(
    vertices: np.ndarray,
    faces: np.ndarray,
    corner_uv: np.ndarray,
    chart_faces: np.ndarray,
    offset_candidates: np.ndarray,
    inside_epsilon: float,
    touching_neighbors: list[set[int]],
):
    best = None
    for offset in offset_candidates:
        points, candidates = _sample_chart_lattice(
            vertices,
            faces,
            corner_uv,
            chart_faces,
            offset,
            inside_epsilon=inside_epsilon,
        )
        estimated_complete = 0
        for (i, j), center_faces in candidates.items():
            corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            if all(corner in points for corner in corners):
                estimated_complete += len(
                    _candidate_components(center_faces, touching_neighbors)
                )
        score = (estimated_complete, len(points))
        if best is None or score > best[0]:
            best = (score, offset.copy(), points, candidates)
    return best[1], best[2], best[3]


def weld_quad_mesh(
    vertices,
    quads,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weld spatially coincident seam vertices and remove invalid quads."""
    vertices = _as_numpy(vertices, np.float64)
    quads = _as_numpy(quads, np.int64)
    if tolerance < 0:
        raise ValueError("weld tolerance must be non-negative")
    if vertices.shape[0] == 0:
        return vertices.reshape(0, 3), quads.reshape(0, 4), np.empty((0,), dtype=np.int64)

    parent = np.arange(vertices.shape[0], dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    if tolerance > 0 and vertices.shape[0] > 1:
        for first, second in cKDTree(vertices).query_pairs(tolerance):
            union(int(first), int(second))
    roots = np.array([find(index) for index in range(vertices.shape[0])])
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    welded = np.zeros((unique_roots.shape[0], 3), dtype=np.float64)
    counts = np.zeros(unique_roots.shape[0], dtype=np.int64)
    np.add.at(welded, inverse, vertices)
    np.add.at(counts, inverse, 1)
    welded /= counts[:, None]

    remapped = inverse[quads]
    valid = np.array([len(set(quad.tolist())) == 4 for quad in remapped])
    remapped = remapped[valid]
    kept = []
    seen: set[tuple[int, ...]] = set()
    for quad in remapped:
        canonical = tuple(sorted(int(vertex) for vertex in quad))
        if canonical not in seen:
            seen.add(canonical)
            kept.append(quad)
    remapped = np.asarray(kept, dtype=np.int64).reshape(-1, 4)

    if remapped.size:
        used, compact_inverse = np.unique(remapped.reshape(-1), return_inverse=True)
        welded = welded[used]
        remapped = compact_inverse.reshape(-1, 4)
        old_to_compact = np.full(unique_roots.shape[0], -1, dtype=np.int64)
        old_to_compact[used] = np.arange(used.shape[0])
        original_to_welded = old_to_compact[inverse]
    else:
        welded = np.empty((0, 3), dtype=np.float64)
        original_to_welded = np.full(vertices.shape[0], -1, dtype=np.int64)
    return welded, remapped, original_to_welded


def _manifold_quad_mask(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Drop the worst cell in each >2-face edge conflict deterministically."""
    keep = np.ones(quads.shape[0], dtype=bool)
    if quads.shape[0] == 0:
        return keep
    positions = vertices[quads]
    lengths = np.linalg.norm(
        np.roll(positions, -1, axis=1) - positions, axis=-1
    )
    aspect = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-14)

    while True:
        incidence: dict[tuple[int, int], list[int]] = {}
        for quad_id in np.flatnonzero(keep):
            quad = quads[quad_id]
            for first, second in (
                (quad[0], quad[1]),
                (quad[1], quad[2]),
                (quad[2], quad[3]),
                (quad[3], quad[0]),
            ):
                edge = tuple(sorted((int(first), int(second))))
                incidence.setdefault(edge, []).append(int(quad_id))
        conflicts = [values for values in incidence.values() if len(values) > 2]
        if not conflicts:
            break
        conflict_counts: dict[int, int] = {}
        for quad_ids in conflicts:
            for quad_id in quad_ids:
                conflict_counts[quad_id] = conflict_counts.get(quad_id, 0) + 1
        remove = max(
            conflict_counts,
            key=lambda quad_id: (
                conflict_counts[quad_id],
                float(aspect[quad_id]),
                quad_id,
            ),
        )
        keep[remove] = False
    return keep


def evaluate_quad_mesh(vertices, quads) -> dict:
    """Compute basic geometry and quad-quality statistics."""
    vertices = _as_numpy(vertices, np.float64)
    quads = _as_numpy(quads, np.int64).reshape(-1, 4)
    if quads.shape[0] == 0:
        return {
            "vertex_count": int(vertices.shape[0]),
            "quad_count": 0,
            "mean_edge_length": 0.0,
            "mean_aspect_ratio": 0.0,
            "median_aspect_ratio": 0.0,
            "p95_aspect_ratio": 0.0,
            "p99_aspect_ratio": 0.0,
            "max_aspect_ratio": 0.0,
            "mean_orthogonality_error": 0.0,
            "p95_orthogonality_error": 0.0,
            "max_orthogonality_error": 0.0,
            "mean_planarity_error": 0.0,
            "p95_planarity_error": 0.0,
            "max_planarity_error": 0.0,
            "edge_count": 0,
            "boundary_edge_count": 0,
            "nonmanifold_edge_count": 0,
            "connected_component_count": 0,
            "euler_characteristic": 0,
            "extraordinary_vertex_count": 0,
            "valence_histogram": {},
        }

    positions = vertices[quads]
    edges = np.roll(positions, -1, axis=1) - positions
    lengths = np.linalg.norm(edges, axis=-1)
    minimum = lengths.min(axis=1)
    aspect = lengths.max(axis=1) / np.maximum(minimum, 1e-14)
    unit_edges = edges / np.maximum(lengths[..., None], 1e-14)
    orthogonality = np.abs(
        np.sum(unit_edges * np.roll(unit_edges, -1, axis=1), axis=-1)
    )

    triangle_normal = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
    normal_length = np.linalg.norm(triangle_normal, axis=-1)
    unit_normal = triangle_normal / np.maximum(normal_length[:, None], 1e-14)
    fourth_distance = np.abs(
        np.sum((positions[:, 3] - positions[:, 0]) * unit_normal, axis=-1)
    )
    planarity = fourth_distance / np.maximum(lengths.mean(axis=1), 1e-14)

    directed_edges = np.concatenate(
        (
            quads[:, [0, 1]],
            quads[:, [1, 2]],
            quads[:, [2, 3]],
            quads[:, [3, 0]],
        ),
        axis=0,
    )
    undirected_edges = np.sort(directed_edges, axis=1)
    unique_edges, edge_incidence = np.unique(
        undirected_edges, axis=0, return_counts=True
    )
    boundary_edges = unique_edges[edge_incidence == 1]
    boundary_vertices = np.zeros(vertices.shape[0], dtype=bool)
    if boundary_edges.size:
        boundary_vertices[boundary_edges.reshape(-1)] = True
    valence = np.bincount(
        unique_edges.reshape(-1), minlength=vertices.shape[0]
    )
    interior_extraordinary = (~boundary_vertices) & (valence != 4)
    boundary_extraordinary = boundary_vertices & ~np.isin(valence, (2, 3))
    valence_values, valence_counts = np.unique(valence, return_counts=True)

    parent = np.arange(vertices.shape[0], dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for first, second in unique_edges:
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)
    component_count = len({find(index) for index in range(vertices.shape[0])})
    return {
        "vertex_count": int(vertices.shape[0]),
        "quad_count": int(quads.shape[0]),
        "mean_edge_length": float(lengths.mean()),
        "mean_aspect_ratio": float(aspect.mean()),
        "median_aspect_ratio": float(np.median(aspect)),
        "p95_aspect_ratio": float(np.quantile(aspect, 0.95)),
        "p99_aspect_ratio": float(np.quantile(aspect, 0.99)),
        "max_aspect_ratio": float(aspect.max()),
        "mean_orthogonality_error": float(orthogonality.mean()),
        "p95_orthogonality_error": float(np.quantile(orthogonality, 0.95)),
        "max_orthogonality_error": float(orthogonality.max()),
        "mean_planarity_error": float(planarity.mean()),
        "p95_planarity_error": float(np.quantile(planarity, 0.95)),
        "max_planarity_error": float(planarity.max()),
        "edge_count": int(unique_edges.shape[0]),
        "boundary_edge_count": int(np.count_nonzero(edge_incidence == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(edge_incidence > 2)),
        "connected_component_count": int(component_count),
        "euler_characteristic": int(
            vertices.shape[0] - unique_edges.shape[0] + quads.shape[0]
        ),
        "extraordinary_vertex_count": int(
            np.count_nonzero(interior_extraordinary | boundary_extraordinary)
        ),
        "valence_histogram": {
            str(int(value)): int(count)
            for value, count in zip(valence_values, valence_counts)
        },
    }


def extract_integer_grid_quads(
    vertices,
    faces,
    corner_uv,
    face_charts,
    *,
    auto_offset: bool = True,
    offset_resolution: int = 4,
    inside_epsilon: float = 1e-9,
    weld_tolerance: Optional[float] = None,
    max_corner_face_hops: int = 8,
) -> QuadMeshResult:
    """Extract complete integer UV cells without merging overlapping sheets."""
    vertices = _as_numpy(vertices, np.float64)
    faces = _as_numpy(faces, np.int64)
    corner_uv = _as_numpy(corner_uv, np.float64)
    face_charts = _as_numpy(face_charts, np.int64).reshape(-1)
    if corner_uv.shape != (faces.shape[0], 3, 2):
        raise ValueError("corner_uv must have shape [F, 3, 2]")
    if face_charts.shape != (faces.shape[0],):
        raise ValueError("face_charts must have shape [F]")
    if offset_resolution < 1:
        raise ValueError("offset_resolution must be positive")
    if max_corner_face_hops < 0:
        raise ValueError("max_corner_face_hops must be non-negative")

    if auto_offset:
        fractions = np.arange(offset_resolution, dtype=np.float64) / offset_resolution
        offset_candidates = np.array(
            [(first, second) for first in fractions for second in fractions]
        )
    else:
        offset_candidates = np.zeros((1, 2), dtype=np.float64)

    raw_vertices: list[np.ndarray] = []
    raw_uv: list[np.ndarray] = []
    raw_charts: list[int] = []
    raw_source_faces: list[int] = []
    raw_quads: list[list[int]] = []
    chart_count = int(face_charts.max()) + 1 if face_charts.size else 0
    chart_offsets = np.zeros((chart_count, 2), dtype=np.float64)
    edge_neighbors, touching_neighbors = _face_neighborhoods(faces)
    extraction_diagnostics = {
        "candidate_cell_occurrence_count": 0,
        "rejected_incomplete_cell_count": 0,
        "overlapping_uv_cell_key_count": 0,
        "maximum_corner_face_hops": 0,
    }

    for chart in range(chart_count):
        chart_faces = np.flatnonzero(face_charts == chart)
        offset, points, candidates = _choose_chart_offset(
            vertices,
            faces,
            corner_uv,
            chart_faces,
            offset_candidates,
            inside_epsilon,
            touching_neighbors,
        )
        chart_offsets[chart] = offset
        complete, diagnostics = _complete_sheet_cells(
            points,
            candidates,
            edge_neighbors,
            touching_neighbors,
            max_corner_face_hops=max_corner_face_hops,
        )
        for key, value in diagnostics.items():
            if key == "maximum_corner_face_hops":
                extraction_diagnostics[key] = max(
                    extraction_diagnostics[key], value
                )
            else:
                extraction_diagnostics[key] += value

        for (i, j), source_face, positions, corner_source_faces in complete:
            keys = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            quad = []
            for key, position, corner_source in zip(
                keys, positions, corner_source_faces
            ):
                quad.append(len(raw_vertices))
                raw_vertices.append(position)
                raw_uv.append(np.asarray(key, dtype=np.float64) - offset)
                raw_charts.append(chart)
                raw_source_faces.append(int(corner_source))
            quad_normal = np.cross(positions[1] - positions[0], positions[3] - positions[0])
            triangle = vertices[faces[source_face]]
            face_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            if np.dot(quad_normal, face_normal) < 0:
                quad = [quad[0], quad[3], quad[2], quad[1]]
            raw_quads.append(quad)

    raw_vertices_array = np.asarray(raw_vertices, dtype=np.float64).reshape(-1, 3)
    raw_quads_array = np.asarray(raw_quads, dtype=np.int64).reshape(-1, 4)
    if weld_tolerance is None:
        diagonal = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        weld_tolerance = max(diagonal * 1e-7, 1e-12)
    welded_vertices, welded_quads, original_to_welded = weld_quad_mesh(
        raw_vertices_array,
        raw_quads_array,
        tolerance=weld_tolerance,
    )

    if welded_vertices.shape[0]:
        welded_uv = np.zeros((welded_vertices.shape[0], 2), dtype=np.float64)
        welded_charts = np.full(welded_vertices.shape[0], -1, dtype=np.int64)
        welded_sources = np.full(welded_vertices.shape[0], -1, dtype=np.int64)
        counts = np.zeros(welded_vertices.shape[0], dtype=np.int64)
        for original, welded in enumerate(original_to_welded):
            if welded < 0:
                continue
            welded_uv[welded] += raw_uv[original]
            if welded_charts[welded] < 0:
                welded_charts[welded] = raw_charts[original]
                welded_sources[welded] = raw_source_faces[original]
            counts[welded] += 1
        welded_uv /= np.maximum(counts[:, None], 1)
    else:
        welded_uv = np.empty((0, 2), dtype=np.float64)
        welded_charts = np.empty((0,), dtype=np.int64)
        welded_sources = np.empty((0,), dtype=np.int64)

    manifold_keep = _manifold_quad_mask(welded_vertices, welded_quads)
    nonmanifold_conflict_quad_count = int(np.count_nonzero(~manifold_keep))
    if nonmanifold_conflict_quad_count:
        welded_quads = welded_quads[manifold_keep]
        used, compact_inverse = np.unique(
            welded_quads.reshape(-1), return_inverse=True
        )
        welded_vertices = welded_vertices[used]
        welded_uv = welded_uv[used]
        welded_charts = welded_charts[used]
        welded_sources = welded_sources[used]
        welded_quads = compact_inverse.reshape(-1, 4)

    metrics = evaluate_quad_mesh(welded_vertices, welded_quads)
    metrics.update(
        {
            "chart_count": chart_count,
            "weld_tolerance": float(weld_tolerance),
            "raw_vertex_count": int(raw_vertices_array.shape[0]),
            "raw_quad_count": int(raw_quads_array.shape[0]),
            "sheet_aware_extraction": True,
            "max_corner_face_hops": int(max_corner_face_hops),
            "nonmanifold_conflict_quad_count": (
                nonmanifold_conflict_quad_count
            ),
            **extraction_diagnostics,
        }
    )
    return QuadMeshResult(
        vertices=welded_vertices,
        quads=welded_quads,
        grid_uv=welded_uv,
        vertex_charts=welded_charts,
        chart_offsets=chart_offsets,
        source_triangle=welded_sources,
        metrics=metrics,
    )


def save_quad_mesh(
    result: QuadMeshResult,
    output_dir: str | Path,
) -> Path:
    """Save the extracted quad OBJ, arrays, and quality report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "quad_mesh.npz",
        vertices=result.vertices,
        quads=result.quads,
        grid_uv=result.grid_uv,
        vertex_charts=result.vertex_charts,
        chart_offsets=result.chart_offsets,
        source_triangle=result.source_triangle,
    )
    with (output_dir / "quad_mesh.obj").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("# HKTex2Quad integer-grid extraction\n")
        for vertex in result.vertices:
            handle.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for quad in result.quads + 1:
            handle.write("f " + " ".join(str(int(index)) for index in quad) + "\n")
    report_path = output_dir / "quad_report.json"
    report_path.write_text(
        json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8"
    )
    return report_path
