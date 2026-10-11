"""Local tangent frames and one-edge discrete Levi-Civita transport.

The routines here are suitable for connection-aware smoothness energies on
adjacent mesh faces.  They do not define transport between arbitrary distant
surface points; that operation still requires a path or a connection heat
kernel.
"""

import heapq
import math
from typing import NamedTuple, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = [
    "BoundaryCrossConstraints",
    "FaceConnectionPathTransport",
    "FaceTangentFrames",
    "boundary_cross_constraints",
    "build_face_adjacency",
    "curvature_target_sizes",
    "encode_tangent_vectors",
    "face_adjacency_connection",
    "face_adjacency_edge_coordinates",
    "face_connection_weights",
    "face_tangent_frames",
    "principal_direction_face_guidance",
    "transport_angle_between_frames",
    "vertex_angle_defects",
]


class FaceTangentFrames(NamedTuple):
    tangent_u: Tensor
    tangent_v: Tensor
    normal: Tensor


class BoundaryCrossConstraints(NamedTuple):
    """Cross-field targets induced by boundary-edge tangents."""

    face_indices: Tensor
    target_power: Tensor
    weights: Tensor


class FaceConnectionPathTransport:
    """Reference source-to-query transport on the triangle dual graph.

    The chosen path minimises the sum of adjacent face-centroid distances and
    the one-edge Levi-Civita rotations are accumulated along that path.  This
    makes the otherwise ambiguous path choice explicit and deterministic.  It
    is intended for small-mesh validation; production-scale training should
    replace it with batched tracing, Vector Heat, or a connection kernel.
    """

    def __init__(
        self,
        vertices: Tensor,
        faces: Tensor,
        *,
        face_normals: Optional[Tensor] = None,
        eps: float = 1e-12,
    ):
        frames, adjacency, edge_angles = face_adjacency_connection(
            vertices,
            faces,
            face_normals=face_normals,
            eps=eps,
        )
        self.frames = frames
        self.face_adjacency = adjacency
        self.edge_transport_angles = edge_angles
        self.num_faces = int(faces.shape[0])
        self._cache: dict[tuple[int, int], float] = {}

        centroids = vertices[faces.long()].mean(dim=1)
        if adjacency.numel():
            edge_costs = torch.linalg.vector_norm(
                centroids[adjacency[:, 1]] - centroids[adjacency[:, 0]], dim=-1
            )
        else:
            edge_costs = vertices.new_empty((0,))

        adjacency_cpu = adjacency.detach().cpu().long().tolist()
        angles_cpu = edge_angles.detach().cpu().tolist()
        costs_cpu = edge_costs.detach().cpu().tolist()
        self._neighbors: list[list[tuple[int, float, float]]] = [
            [] for _ in range(self.num_faces)
        ]
        for (source, target), cost, angle in zip(
            adjacency_cpu, costs_cpu, angles_cpu
        ):
            self._neighbors[source].append((target, float(cost), float(angle)))
            self._neighbors[target].append((source, float(cost), -float(angle)))
        for neighbors in self._neighbors:
            neighbors.sort(key=lambda item: item[0])

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _validate_face(self, face: int):
        if face < 0 or face >= self.num_faces:
            raise ValueError(
                f"face index {face} is outside [0, {self.num_faces})"
            )

    def _dijkstra(
        self,
        root: int,
        goals: set[int],
    ) -> dict[int, float]:
        self._validate_face(root)
        for goal in goals:
            self._validate_face(goal)
        if not goals:
            return {}

        distances = [math.inf] * self.num_faces
        accumulated_angles = [0.0] * self.num_faces
        distances[root] = 0.0
        queue: list[tuple[float, int]] = [(0.0, root)]
        remaining = set(goals)
        result: dict[int, float] = {}

        while queue and remaining:
            distance, face = heapq.heappop(queue)
            if distance > distances[face]:
                continue
            if face in remaining:
                result[face] = self._wrap_angle(accumulated_angles[face])
                remaining.remove(face)
            for neighbor, cost, edge_angle in self._neighbors[face]:
                candidate = distance + cost
                if candidate + 1e-15 < distances[neighbor]:
                    distances[neighbor] = candidate
                    accumulated_angles[neighbor] = (
                        accumulated_angles[face] + edge_angle
                    )
                    heapq.heappush(queue, (candidate, neighbor))

        if remaining:
            missing = ", ".join(str(face) for face in sorted(remaining))
            raise ValueError(
                f"no dual-graph path from face {root} to face(s) {missing}"
            )
        return result

    def angles(self, source_faces: Tensor, target_faces: Tensor) -> Tensor:
        """Return ordinary tangent-vector rotations for broadcast face pairs."""
        try:
            sources, targets = torch.broadcast_tensors(source_faces, target_faces)
        except RuntimeError as exc:
            raise ValueError(
                "source_faces and target_faces must be broadcastable"
            ) from exc
        output_device = sources.device
        output_dtype = self.edge_transport_angles.dtype
        sources_cpu = sources.detach().cpu().long().reshape(-1).tolist()
        targets_cpu = targets.detach().cpu().long().reshape(-1).tolist()

        unresolved: set[tuple[int, int]] = set()
        for source, target in zip(sources_cpu, targets_cpu):
            self._validate_face(source)
            self._validate_face(target)
            if source == target:
                self._cache[(source, target)] = 0.0
            elif (source, target) not in self._cache:
                unresolved.add((source, target))

        unique_sources = {source for source, _ in unresolved}
        unique_targets = {target for _, target in unresolved}
        if len(unique_sources) <= len(unique_targets):
            for source in sorted(unique_sources):
                goals = {target for src, target in unresolved if src == source}
                solved = self._dijkstra(source, goals)
                for target, angle in solved.items():
                    self._cache[(source, target)] = angle
                    self._cache[(target, source)] = self._wrap_angle(-angle)
        else:
            for target in sorted(unique_targets):
                goals = {source for source, dst in unresolved if dst == target}
                reverse_solved = self._dijkstra(target, goals)
                for source, reverse_angle in reverse_solved.items():
                    angle = self._wrap_angle(-reverse_angle)
                    self._cache[(source, target)] = angle
                    self._cache[(target, source)] = reverse_angle

        values = [
            self._cache[(source, target)]
            for source, target in zip(sources_cpu, targets_cpu)
        ]
        return torch.tensor(values, dtype=output_dtype, device=output_device).reshape(
            sources.shape
        )


def encode_tangent_vectors(
    vectors: Tensor,
    tangent_u: Tensor,
    tangent_v: Tensor,
    *,
    degree: int = 4,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Encode 3D tangent directions as unit N-RoSy power coefficients."""
    if degree < 1:
        raise ValueError("degree must be positive")
    if vectors.shape != tangent_u.shape or vectors.shape != tangent_v.shape:
        raise ValueError("vectors and tangent frames must have matching shapes")
    if vectors.shape[-1] != 3:
        raise ValueError("vectors and tangent frames must end in dimension 3")

    coordinate_u = (vectors * tangent_u).sum(dim=-1)
    coordinate_v = (vectors * tangent_v).sum(dim=-1)
    tangent_norm = torch.sqrt(coordinate_u.square() + coordinate_v.square())
    valid = tangent_norm > eps
    angle = torch.atan2(coordinate_v, coordinate_u)
    power_angle = degree * angle
    power = torch.stack((torch.cos(power_angle), torch.sin(power_angle)), dim=-1)
    power = torch.where(valid.unsqueeze(-1), power, torch.zeros_like(power))
    return power, valid


def face_tangent_frames(
    vertices: Tensor,
    faces: Tensor,
    *,
    face_normals: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> FaceTangentFrames:
    """Construct deterministic geometric frames independent of curvature lines.

    The longest directed triangle edge is used for the first tangent.  These
    frames are only a coordinate gauge; principal curvature directions should
    be treated as optional field guidance, not as the storage basis itself.
    """
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3]")

    triangles = vertices[faces.long()]
    edge_candidates = torch.stack(
        (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ),
        dim=1,
    )
    edge_lengths_sq = edge_candidates.square().sum(dim=-1)
    longest = edge_lengths_sq.argmax(dim=1)
    tangent_u = edge_candidates[
        torch.arange(faces.shape[0], device=vertices.device), longest
    ]

    geometric_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=-1,
    )
    geometric_norm = torch.linalg.vector_norm(
        geometric_normals, dim=-1, keepdim=True
    )
    if bool((geometric_norm.squeeze(-1) <= eps).any()):
        raise ValueError("Cannot construct tangent frames for degenerate faces")

    if face_normals is None:
        normal = geometric_normals / geometric_norm
    else:
        if face_normals.shape != geometric_normals.shape:
            raise ValueError("face_normals must have shape [F, 3]")
        normal = F.normalize(face_normals, dim=-1, eps=eps)
        if bool((torch.linalg.vector_norm(normal, dim=-1) <= eps).any()):
            raise ValueError("face_normals contains a zero normal")

    tangent_u = tangent_u - (tangent_u * normal).sum(
        dim=-1, keepdim=True
    ) * normal
    tangent_u = F.normalize(tangent_u, dim=-1, eps=eps)
    tangent_v = F.normalize(torch.cross(normal, tangent_u, dim=-1), dim=-1, eps=eps)
    return FaceTangentFrames(tangent_u, tangent_v, normal)


def transport_angle_between_frames(
    source_u: Tensor,
    source_v: Tensor,
    source_normal: Tensor,
    target_u: Tensor,
    target_v: Tensor,
    target_normal: Tensor,
    *,
    eps: float = 1e-12,
    antiparallel_mode: str = "error",
) -> Tensor:
    """Return one-edge tangent-vector transport rotation between face frames.

    Source frames are minimally rotated so that their normals align with target
    normals.  For adjacent consistently oriented triangles, the rotation axis
    is their shared edge and this is the standard polyhedral Levi-Civita
    transport.  Antiparallel normals have no unique minimal rotation axis.
    The default rejects them; ``antiparallel_mode='source-u'`` chooses the
    source frame's first tangent as a deterministic, fully vectorised fallback
    suitable for GPU training.
    """
    tensors = (
        source_u,
        source_v,
        source_normal,
        target_u,
        target_v,
        target_normal,
    )
    if any(t.shape[-1] != 3 for t in tensors):
        raise ValueError("all frame vectors must have final dimension 3")
    try:
        tensors = torch.broadcast_tensors(*tensors)
    except RuntimeError as exc:
        raise ValueError("source and target frames must be broadcastable") from exc
    source_u, _, source_normal, target_u, target_v, target_normal = tensors

    source_normal = F.normalize(source_normal, dim=-1, eps=eps)
    target_normal = F.normalize(target_normal, dim=-1, eps=eps)
    normal_cross = torch.cross(source_normal, target_normal, dim=-1)
    sin_angle = torch.linalg.vector_norm(normal_cross, dim=-1)
    cos_angle = (source_normal * target_normal).sum(dim=-1).clamp(-1.0, 1.0)

    if antiparallel_mode not in ("error", "source-u"):
        raise ValueError("antiparallel_mode must be 'error' or 'source-u'")
    antiparallel = (sin_angle <= eps) & (cos_angle < 0.0)
    if antiparallel_mode == "error" and bool(antiparallel.any()):
        raise ValueError("transport between antiparallel normals is ambiguous")

    axis = normal_cross / sin_angle.clamp_min(eps).unsqueeze(-1)
    if antiparallel_mode == "source-u":
        fallback_axis = F.normalize(source_u, dim=-1, eps=eps)
        axis = torch.where(antiparallel.unsqueeze(-1), fallback_axis, axis)
    rotation_angle = torch.atan2(sin_angle, cos_angle)
    cos_rotation = torch.cos(rotation_angle).unsqueeze(-1)
    sin_rotation = torch.sin(rotation_angle).unsqueeze(-1)
    transported_u = (
        source_u * cos_rotation
        + torch.cross(axis, source_u, dim=-1) * sin_rotation
        + axis
        * (axis * source_u).sum(dim=-1, keepdim=True)
        * (1.0 - cos_rotation)
    )
    parallel_normals = (sin_angle <= eps) & (cos_angle >= 0.0)
    transported_u = torch.where(
        parallel_normals.unsqueeze(-1), source_u, transported_u
    )

    transported_u = transported_u - (
        transported_u * target_normal
    ).sum(dim=-1, keepdim=True) * target_normal
    transported_u = F.normalize(transported_u, dim=-1, eps=eps)

    return torch.atan2(
        (transported_u * target_v).sum(dim=-1),
        (transported_u * target_u).sum(dim=-1),
    )


def build_face_adjacency(faces: Tensor) -> Tensor:
    """Return one oriented face pair for every manifold interior edge."""
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3]")

    faces_cpu = faces.detach().cpu().long()
    owner: dict[tuple[int, int], int] = {}
    adjacency: list[tuple[int, int]] = []
    for face_id, face in enumerate(faces_cpu.tolist()):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (a, b) if a < b else (b, a)
            if edge not in owner:
                owner[edge] = face_id
            else:
                first_face = owner.pop(edge)
                adjacency.append((first_face, face_id))

    # Edges still in owner are boundary edges.  Encountering a third incident
    # face would leave a duplicate pair and must be detected separately.
    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces_cpu.tolist():
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (a, b) if a < b else (b, a)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if any(count > 2 for count in edge_counts.values()):
        raise ValueError("face adjacency requires a manifold triangle mesh")

    if not adjacency:
        return torch.empty((0, 2), dtype=torch.long, device=faces.device)
    return torch.tensor(adjacency, dtype=torch.long, device=faces.device)


def face_adjacency_connection(
    vertices: Tensor,
    faces: Tensor,
    *,
    face_normals: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> tuple[FaceTangentFrames, Tensor, Tensor]:
    """Build face gauges, oriented adjacency, and one-edge transport angles."""
    frames = face_tangent_frames(
        vertices, faces, face_normals=face_normals, eps=eps
    )
    adjacency = build_face_adjacency(faces)
    if adjacency.numel() == 0:
        return frames, adjacency, vertices.new_empty((0,))

    source = adjacency[:, 0]
    target = adjacency[:, 1]
    angles = transport_angle_between_frames(
        frames.tangent_u[source],
        frames.tangent_v[source],
        frames.normal[source],
        frames.tangent_u[target],
        frames.tangent_v[target],
        frames.normal[target],
        eps=eps,
    )
    return frames, adjacency, angles


def face_connection_weights(
    vertices: Tensor,
    faces: Tensor,
    adjacency: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Return normalized primal-edge/dual-edge weights for face adjacency.

    The ratio approximates the area weighting in a face-based Dirichlet
    energy.  It is normalized to mean one so changing mesh scale does not
    silently change the relative loss coefficient.
    """
    if adjacency.ndim != 2 or adjacency.shape[-1] != 2:
        raise ValueError("adjacency must have shape [E, 2]")
    if adjacency.numel() == 0:
        return vertices.new_empty((0,))

    face_vertices = faces.long()
    source_faces = face_vertices[adjacency[:, 0].long()]
    target_faces = face_vertices[adjacency[:, 1].long()]
    shared_mask = (
        source_faces.unsqueeze(-1) == target_faces.unsqueeze(-2)
    ).any(dim=-1)
    if not bool((shared_mask.sum(dim=-1) == 2).all()):
        raise ValueError("every adjacent face pair must share exactly one edge")
    shared_vertices = source_faces[shared_mask].reshape(-1, 2)
    primal_lengths = torch.linalg.vector_norm(
        vertices[shared_vertices[:, 1]] - vertices[shared_vertices[:, 0]], dim=-1
    )

    centroids = vertices[face_vertices].mean(dim=1)
    dual_lengths = torch.linalg.vector_norm(
        centroids[adjacency[:, 1].long()] - centroids[adjacency[:, 0].long()],
        dim=-1,
    )
    if bool(((primal_lengths <= eps) | (dual_lengths <= eps)).any()):
        raise ValueError("cannot weight degenerate primal or dual edges")
    weights = primal_lengths / dual_lengths
    return weights / weights.mean().clamp_min(eps)


def face_adjacency_edge_coordinates(
    vertices: Tensor,
    faces: Tensor,
    adjacency: Tensor,
    frames: FaceTangentFrames,
) -> tuple[Tensor, Tensor]:
    """Express every oriented shared edge in both incident face gauges.

    The returned tensors have shape ``[E, 2]``.  They encode the same ambient
    edge vector in the source and target tangent frames and are therefore the
    geometric data required by a discrete edge-integrability objective.
    """
    if adjacency.ndim != 2 or adjacency.shape[-1] != 2:
        raise ValueError("adjacency must have shape [E, 2]")
    if adjacency.numel() == 0:
        empty = vertices.new_empty((0, 2))
        return empty, empty

    face_vertices = faces.long()
    source_faces = face_vertices[adjacency[:, 0].long()]
    target_faces = face_vertices[adjacency[:, 1].long()]
    shared_mask = (
        source_faces.unsqueeze(-1) == target_faces.unsqueeze(-2)
    ).any(dim=-1)
    if not bool((shared_mask.sum(dim=-1) == 2).all()):
        raise ValueError("every adjacent face pair must share exactly one edge")
    shared_vertices = source_faces[shared_mask].reshape(-1, 2)
    edge_vectors = (
        vertices[shared_vertices[:, 1]] - vertices[shared_vertices[:, 0]]
    )

    source = adjacency[:, 0].long()
    target = adjacency[:, 1].long()

    def project(face_ids: Tensor) -> Tensor:
        return torch.stack(
            (
                (edge_vectors * frames.tangent_u[face_ids]).sum(dim=-1),
                (edge_vectors * frames.tangent_v[face_ids]).sum(dim=-1),
            ),
            dim=-1,
        )

    return project(source), project(target)


def principal_direction_face_guidance(
    faces: Tensor,
    vertex_directions: Tensor,
    frames: FaceTangentFrames,
    *,
    vertex_curvatures: Optional[Tensor] = None,
    degree: int = 4,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Convert sign-ambiguous vertex directions into face N-RoSy targets.

    Vertex directions are encoded in power space before averaging, so the
    arbitrary sign of a principal direction cancels for even ``degree``.
    When signed principal curvatures ``[V, 2]`` are supplied, supervision is
    attenuated in planar and umbilic regions using
    ``|k1-k2| / (|k1|+|k2|)``.  The returned face weight additionally measures
    agreement of the three corner directions.
    """
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if vertex_directions.ndim != 2 or vertex_directions.shape[-1] != 3:
        raise ValueError("vertex_directions must have shape [V, 3]")
    if frames.tangent_u.shape != (faces.shape[0], 3):
        raise ValueError("frames must contain one tangent frame per face")
    if degree < 1:
        raise ValueError("degree must be positive")

    corner_directions = vertex_directions[faces.long()]
    tangent_u = frames.tangent_u.unsqueeze(1).expand_as(corner_directions)
    tangent_v = frames.tangent_v.unsqueeze(1).expand_as(corner_directions)
    corner_power, valid = encode_tangent_vectors(
        corner_directions,
        tangent_u,
        tangent_v,
        degree=degree,
        eps=eps,
    )

    reliability = valid.to(vertex_directions.dtype)
    if vertex_curvatures is not None:
        if vertex_curvatures.shape != (vertex_directions.shape[0], 2):
            raise ValueError("vertex_curvatures must have shape [V, 2]")
        corner_curvatures = vertex_curvatures[faces.long()]
        k1, k2 = corner_curvatures.unbind(dim=-1)
        anisotropy = (k1 - k2).abs() / (k1.abs() + k2.abs()).clamp_min(eps)
        reliability = reliability * anisotropy

    denominator = reliability.sum(dim=1, keepdim=True)
    face_power = (corner_power * reliability.unsqueeze(-1)).sum(dim=1)
    face_power = face_power / denominator.clamp_min(eps)
    face_power = torch.where(
        (denominator > eps), face_power, torch.zeros_like(face_power)
    )
    coherence = torch.linalg.vector_norm(face_power, dim=-1).clamp(max=1.0)
    face_reliability = reliability.mean(dim=1)
    return face_power, coherence * face_reliability


def curvature_target_sizes(
    faces: Tensor,
    vertex_curvatures: Tensor,
    base_size: float,
    *,
    curvature_scale: float = 1.0,
    min_size: float = 1e-4,
    max_size: float = 1e4,
) -> Tensor:
    """Build a positive curvature-adaptive face target edge-size field.

    ``base_size`` is the flat-region edge length.  The dimensionless product
    ``base_size * curvature`` makes the rule invariant to uniform rescaling of
    the input mesh.
    """
    if vertex_curvatures.ndim != 2 or vertex_curvatures.shape[-1] != 2:
        raise ValueError("vertex_curvatures must have shape [V, 2]")
    if base_size <= 0 or curvature_scale < 0:
        raise ValueError("base_size must be positive and curvature_scale non-negative")
    if not 0 < min_size <= max_size:
        raise ValueError("size bounds must be positive and ordered")

    face_curvature = vertex_curvatures[faces.long()].abs().sum(dim=-1).mean(dim=1)
    scaled_curvature = base_size * face_curvature
    sizes = base_size / torch.sqrt(
        1.0 + curvature_scale * scaled_curvature.square()
    )
    return sizes.clamp(min=min_size, max=max_size)


def vertex_angle_defects(
    vertices: Tensor,
    faces: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Compute oriented-mesh interior angle defects ``2*pi - sum(corners)``.

    The returned values provide the lifted Levi-Civita holonomy needed for
    robust singularity indices.  In particular, unlike wrapping a summed
    connection angle, they retain the sign of an angle defect equal to pi.
    Boundary entries are returned as the same algebraic expression but should
    not be used as interior singularity indices.
    """
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3]")

    triangles = vertices[faces.long()]
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 1]
    third = triangles[:, 0] - triangles[:, 2]
    corner_pairs = (
        (first, -third),
        (second, -first),
        (third, -second),
    )
    corner_angles = []
    for left, right in corner_pairs:
        left_norm = torch.linalg.vector_norm(left, dim=-1)
        right_norm = torch.linalg.vector_norm(right, dim=-1)
        if bool(((left_norm <= eps) | (right_norm <= eps)).any()):
            raise ValueError("cannot compute angle defects with zero-length edges")
        cosine = (left * right).sum(dim=-1) / (left_norm * right_norm)
        corner_angles.append(torch.acos(cosine.clamp(-1.0, 1.0)))
    corner_angles = torch.stack(corner_angles, dim=-1)

    angle_sums = vertices.new_zeros((vertices.shape[0],))
    angle_sums.scatter_add_(0, faces.long().reshape(-1), corner_angles.reshape(-1))
    return 2.0 * torch.pi - angle_sums


def boundary_cross_constraints(
    vertices: Tensor,
    faces: Tensor,
    *,
    frames: Optional[FaceTangentFrames] = None,
    degree: int = 4,
    eps: float = 1e-12,
) -> BoundaryCrossConstraints:
    """Create face-field targets aligned with every manifold boundary edge.

    A face incident on two boundary edges contributes two constraints.  This
    is intentional: at a sharp boundary corner the optimiser should see both
    directions rather than silently selecting one.  Edge length is returned
    as the natural integration weight for each constraint.
    """
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if frames is None:
        frames = face_tangent_frames(vertices, faces, eps=eps)
    if frames.tangent_u.shape != (faces.shape[0], 3):
        raise ValueError("frames must contain one tangent frame per face")

    faces_list = faces.detach().cpu().long().tolist()
    edge_owners: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_id, face in enumerate(faces_list):
        for source, target in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = (
                (source, target) if source < target else (target, source)
            )
            edge_owners.setdefault(edge, []).append((face_id, source, target))

    if any(len(owners) > 2 for owners in edge_owners.values()):
        raise ValueError("boundary constraints require a manifold triangle mesh")
    boundary = [owners[0] for owners in edge_owners.values() if len(owners) == 1]
    if not boundary:
        return BoundaryCrossConstraints(
            torch.empty((0,), dtype=torch.long, device=vertices.device),
            vertices.new_empty((0, 2)),
            vertices.new_empty((0,)),
        )

    face_indices = torch.tensor(
        [entry[0] for entry in boundary],
        dtype=torch.long,
        device=vertices.device,
    )
    source_vertices = torch.tensor(
        [entry[1] for entry in boundary],
        dtype=torch.long,
        device=vertices.device,
    )
    target_vertices = torch.tensor(
        [entry[2] for entry in boundary],
        dtype=torch.long,
        device=vertices.device,
    )
    edge_vectors = vertices[target_vertices] - vertices[source_vertices]
    edge_lengths = torch.linalg.vector_norm(edge_vectors, dim=-1)
    target_power, valid = encode_tangent_vectors(
        edge_vectors,
        frames.tangent_u[face_indices],
        frames.tangent_v[face_indices],
        degree=degree,
        eps=eps,
    )
    if not bool(valid.all()):
        raise ValueError("mesh contains a zero-length boundary edge")
    return BoundaryCrossConstraints(face_indices, target_power, edge_lengths)
