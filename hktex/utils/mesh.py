"""Load and normalize triangle geometry while preserving supervision indices."""
import numpy as np
import trimesh
import hktex

__all__ = ["load_mesh"]


def load_mesh(file_path, normalise_size=True, preserve_topology=False):
    loaded = trimesh.load(file_path, process=False)
    def geometry(mesh):
        return trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node in loaded.graph.nodes_geometry:
            transform, name = loaded.graph[node]
            if isinstance(loaded.geometry[name], trimesh.Trimesh):
                current = geometry(loaded.geometry[name])
                current.apply_transform(transform)
                meshes.append(current)
        if not meshes:
            raise ValueError(f"No triangle mesh in {file_path}")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = geometry(loaded)
    else:
        raise ValueError(f"Expected a triangle mesh: {file_path}")
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh must have faces and finite vertices")
    if not preserve_topology:
        trimesh.grouping.merge_vertices(mesh, merge_tex=True, merge_norm=True)
    if normalise_size:
        extent = float(max(mesh.extents))
        if extent <= 0 or mesh.area <= 0:
            raise ValueError("Cannot normalize degenerate mesh")
        translation = -mesh.centroid.copy()
        scale = 2.0 / extent
        mesh.apply_translation(translation)
        mesh.apply_scale(scale)
        hktex.info(f"Scaling mesh by {scale}, and translating by {translation}")
    return mesh
