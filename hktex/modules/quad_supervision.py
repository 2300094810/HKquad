"""Quad-derived 4-RoSy labels on independently supplied triangles.

The labels supervise the dense field, not individual kernel parameters.
"""
from pathlib import Path
import numpy as np


def read_quad_obj(path):
    vertices, quads = [], []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "v":
                vertices.append([float(v) for v in parts[1:4]])
            elif parts[0] == "f":
                if len(parts) != 5:
                    raise ValueError("Reference must contain only quadrilateral faces")
                indices = [int(v.split("/")[0]) for v in parts[1:]]
                quads.append([v - 1 if v > 0 else len(vertices) + v for v in indices])
    vertices = np.asarray(vertices, dtype=np.float64)
    quads = np.asarray(quads, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid reference vertices")
    if quads.ndim != 2 or quads.shape[1] != 4 or quads.min() < 0 or quads.max() >= len(vertices):
        raise ValueError("Invalid reference quad indices")
    if any(len(set(q)) != 4 for q in quads):
        raise ValueError("Repeated index within a reference quad")
    return vertices, quads


def _triangulate_reference_for_queries(vertices, quads):
    """Temporary closest-point query surface; never an optimization input."""
    faces = np.stack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), axis=1).reshape(-1, 3)
    triangle = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=-1)
    if np.any(area2 < 1e-14):
        raise ValueError("Reference triangulation has degenerate triangles")
    return faces, np.repeat(np.arange(len(quads)), 2)


def cross_from_edges(edges, tangent_u, tangent_v, minimum_confidence=0.1):
    """Project quad boundary edges; average unit complex fourth powers."""
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be in [0, 1]")
    x = np.einsum("fkj,fj->fk", edges, tangent_u)
    y = np.einsum("fkj,fj->fk", edges, tangent_v)
    length = np.hypot(x, y)
    valid = length > 1e-12
    z = (x + 1j * y) / np.maximum(length, 1e-12)
    mean = ((z ** 4) * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    confidence = np.abs(mean).clip(0, 1)
    confidence[(valid.sum(axis=1) != 4) | (confidence < minimum_confidence)] = 0
    unit = mean / np.maximum(np.abs(mean), 1e-12)
    return np.stack((unit.real, unit.imag), axis=-1), confidence


def load_quad_targets(path, vertices, faces, tangent_u, tangent_v, minimum_confidence=0.1):
    """Load v2 independent-input labels; check geometry and face indices."""
    with np.load(path, allow_pickle=False) as data:
        version = str(data["format"])
        if version != "hktex2quad-quad-supervision-v2":
            raise ValueError("Independent-input v2 labels are required; prepare them with --triangle and --quad")
        raw_vertices = data["vertices"]
        if not np.array_equal(faces, data["faces"]):
            raise ValueError("Quad labels do not match face indices; use data.preserve_topology=true")
        if vertices.shape != raw_vertices.shape:
            raise ValueError("Quad labels do not match vertex count")
        def centered_unit(v):
            return (v - v.mean(axis=0)) / np.linalg.norm(np.ptp(v, axis=0))
        if not np.allclose(centered_unit(vertices), centered_unit(raw_vertices), atol=2e-6):
            raise ValueError("Quad label geometry differs from training mesh")
        edges = data["mapped_edges"]
        valid = data["mapping_valid"]
        if edges.shape != (len(faces), 4, 3) or not np.isfinite(edges).all():
            raise ValueError("Invalid mapped quad edge vectors")
        if valid.shape != (len(faces),) or valid.dtype != np.bool_:
            raise ValueError("Invalid quad mapping mask")
        power, confidence = cross_from_edges(edges, tangent_u, tangent_v, minimum_confidence)
        confidence[~valid] = 0
        return power, confidence




def read_triangle_obj(path):
    """Read the optimization surface without merging or reordering it."""
    vertices, faces = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "v":
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            if len(parts) != 4:
                raise ValueError("Input optimization mesh must contain only triangles")
            indices = [int(x.split("/")[0]) for x in parts[1:]]
            faces.append([x - 1 if x > 0 else len(vertices) + x for x in indices])
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid input vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid input triangle indices")
    triangle_frames(vertices, faces)
    return vertices, faces


def triangle_frames(vertices, faces):
    """Unit frame and area, preserving the supplied face winding."""
    triangles = vertices[faces]
    first = triangles[:, 1] - triangles[:, 0]
    cross = np.cross(first, triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    scale = np.linalg.norm(np.ptp(vertices, axis=0))
    if not np.isfinite(area2).all() or scale <= 0 or np.any(area2 <= scale * scale * 1e-14):
        raise ValueError("Input has degenerate triangles")
    normals = cross / area2[:, None]
    u = first / np.linalg.norm(first, axis=1)[:, None]
    return u, np.cross(normals, u), normals, area2 / 2


def map_quad_directions(vertices, faces, reference_vertices, quads, *,
                        max_distance_ratio=0.01, max_normal_angle=60., chunk_size=1024):
    """Sample reference directions at the independent input's fixed face centers.

    Reference triangulation serves ONLY closest-surface queries. It never replaces
    the optimization surface. Reject distant/oppositely oriented matches; project
    quad boundary edges to the reference tangent plane and minimally rotate them
    to the input tangent plane. Rejected samples receive zero supervision weight.
    """
    import trimesh
    if not np.isfinite(max_distance_ratio) or max_distance_ratio <= 0:
        raise ValueError("max_distance_ratio must be positive")
    if not np.isfinite(max_normal_angle) or not 0 < max_normal_angle < 90:
        raise ValueError("max_normal_angle must be between 0 and 90 degrees")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    u, v, normals, areas = triangle_frames(vertices, faces)
    ref_faces, parent = _triangulate_reference_for_queries(reference_vertices, quads)
    _, _, ref_normals, _ = triangle_frames(reference_vertices, ref_faces)
    reference = trimesh.Trimesh(reference_vertices, ref_faces, process=False)
    centers = vertices[faces].mean(axis=1)
    closest, distances, triangle_ids = [], [], []
    for start in range(0, len(centers), chunk_size):
        point, distance, face_id = trimesh.proximity.closest_point(reference, centers[start:start + chunk_size])
        closest.append(point)
        distances.append(distance)
        triangle_ids.append(face_id)
    closest = np.concatenate(closest)
    distances = np.concatenate(distances)
    triangle_ids = np.concatenate(triangle_ids).astype(np.int64)
    matched_normals = ref_normals[triangle_ids]
    normal_cos = np.einsum("ij,ij->i", matched_normals, normals).clip(-1, 1)
    diagonal = np.linalg.norm(np.ptp(vertices, axis=0))
    close = distances <= max_distance_ratio * diagonal
    aligned = normal_cos >= np.cos(np.deg2rad(max_normal_angle))
    valid = close & aligned & np.isfinite(distances)
    quad_ids = parent[triangle_ids]
    positions = reference_vertices[quads[quad_ids]]
    edges = np.roll(positions, -1, axis=1) - positions
    edges -= np.einsum("fkj,fj->fk", edges, matched_normals)[..., None] * matched_normals[:, None, :]
    # Rodrigues minimal rotation; never rotate an invalid/antiparallel match.
    mapped = np.zeros_like(edges)
    axis = np.cross(matched_normals[valid], normals[valid])[:, None, :]
    first = np.cross(axis, edges[valid])
    mapped[valid] = edges[valid] + first + np.cross(axis, first) / (1 + normal_cos[valid, None, None])
    _, confidence = cross_from_edges(mapped, u, v)
    total_area = areas.sum()
    fraction = lambda mask: float(areas[mask].sum() / total_area)
    report = dict(method="closest reference surface; signed normal gate; minimal normal rotation",
                  max_distance_ratio=float(max_distance_ratio), max_normal_angle=float(max_normal_angle),
                  input_vertex_count=len(vertices), input_face_count=len(faces), reference_quad_count=len(quads),
                  mapping_valid_area_fraction=fraction(valid), reliable_label_area_fraction=fraction(valid & (confidence >= .1)),
                  evaluation_label_area_fraction=fraction(valid & (confidence >= .25)),
                  distance_rejected_area_fraction=fraction(~close), normal_rejected_area_fraction=fraction(~aligned),
                  distance_ratio_mean=float(np.average(distances / diagonal, weights=areas)),
                  distance_ratio_max=float((distances / diagonal).max()),
                  normal_angle_max_degrees=float(np.rad2deg(np.arccos(normal_cos)).max()),
                  reference_triangulation_used_for_queries_only=True)
    arrays = dict(mapped_edges=mapped, mapping_valid=valid, reference_quad_ids=quad_ids,
                  reference_triangle_ids=triangle_ids, closest_reference_points=closest,
                  mapping_distance=distances, mapping_normal_cos=normal_cos)
    return arrays, report
