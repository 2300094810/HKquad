"""Publication-style static diagnostics for HKTex2Quad fields and kernels."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

__all__ = [
    "kernel_visualization_data_from_checkpoint",
    "save_quad_field_visualizations",
]


def _as_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def kernel_visualization_data_from_checkpoint(
    dense_payload: Mapping,
    checkpoint: Mapping,
    *,
    range_enforcement_type: str = "pgd",
    diffusion_time: float = 0.1,
    min_target_size: float = 1e-4,
    max_target_size: float = 1e4,
) -> dict[str, np.ndarray]:
    """Recover display-ready kernel parameters from a saved model state.

    Checkpoints store kernel locations as two barycentric coordinates and may
    store raw parameters before their bounded activations.  This helper makes
    standalone visualization match the values seen by the trained model.
    """
    required = (
        "_kernel_face_ids",
        "_cross_coefficients",
        "_target_log_sizes",
        "_angles",
        "_anisotropies",
        "_thresholds",
        "_sharpnesses",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint is missing kernel keys: {missing}")

    vertices = _as_numpy(dense_payload["vertices"], np.float64)
    faces = _as_numpy(dense_payload["faces"], np.int64)
    face_ids = _as_numpy(checkpoint["_kernel_face_ids"], np.int64).reshape(-1)
    if np.any(face_ids < 0) or np.any(face_ids >= faces.shape[0]):
        raise ValueError("checkpoint kernel face ids are invalid for the dense mesh")

    if "_kernel_locations" in checkpoint:
        locations = _as_numpy(checkpoint["_kernel_locations"], np.float64)
    elif "_kernel_bary_coords" in checkpoint:
        bary_first_two = _as_numpy(
            checkpoint["_kernel_bary_coords"], np.float64
        )
        if bary_first_two.shape != (face_ids.shape[0], 2):
            raise ValueError("_kernel_bary_coords must have shape [K, 2]")
        barycentric = np.concatenate(
            (
                bary_first_two,
                1.0 - bary_first_two.sum(axis=1, keepdims=True),
            ),
            axis=1,
        )
        locations = np.einsum(
            "ki,kij->kj", barycentric, vertices[faces[face_ids]]
        )
    else:
        raise ValueError(
            "checkpoint must contain _kernel_locations or _kernel_bary_coords"
        )

    coefficients = _as_numpy(checkpoint["_cross_coefficients"], np.float64)
    amplitudes = np.linalg.norm(coefficients, axis=1)
    degree = int(dense_payload.get("rosy_degree", 4))
    cross_angles = np.arctan2(coefficients[:, 1], coefficients[:, 0]) / degree
    target_sizes = np.exp(
        np.clip(
            _as_numpy(checkpoint["_target_log_sizes"], np.float64),
            math.log(min_target_size),
            math.log(max_target_size),
        )
    )

    raw_angles = _as_numpy(checkpoint["_angles"], np.float64)
    raw_anisotropies = _as_numpy(checkpoint["_anisotropies"], np.float64)
    raw_thresholds = _as_numpy(checkpoint["_thresholds"], np.float64)
    raw_sharpnesses = _as_numpy(checkpoint["_sharpnesses"], np.float64)
    if range_enforcement_type == "activations":
        support_angles = np.pi * _sigmoid(raw_angles)
        anisotropies = 1.0 + 99.0 * _sigmoid(raw_anisotropies)
        thresholds = 0.3 + (0.7 - 1e-8) * _sigmoid(raw_thresholds + 0.916)
        sharpnesses = 10.0 + 190.0 * _sigmoid(raw_sharpnesses)
    elif range_enforcement_type == "pgd":
        support_angles = raw_angles
        anisotropies = raw_anisotropies
        thresholds = raw_thresholds
        sharpnesses = raw_sharpnesses
    else:
        raise ValueError(
            "range_enforcement_type must be 'pgd' or 'activations'"
        )

    kernel_count = face_ids.shape[0]
    return {
        "locations": locations,
        "face_ids": face_ids,
        "cross_angles": cross_angles,
        "cross_amplitudes": amplitudes,
        "target_sizes": target_sizes,
        "support_angles": support_angles,
        "anisotropies": anisotropies,
        "thresholds": thresholds,
        "sharpnesses": sharpnesses,
        "diffusion_times": np.full(kernel_count, diffusion_time, dtype=np.float64),
    }


def _normalise_vectors(vectors: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _robust_limits(
    values: np.ndarray,
    *,
    lower: float = 0.02,
    upper: float = 0.98,
    positive: bool = False,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if positive:
        values = values[values > 0]
    if values.size == 0:
        return (1e-6, 1.0) if positive else (0.0, 1.0)
    minimum, maximum = np.quantile(values, (lower, upper))
    if not maximum > minimum:
        padding = max(abs(float(minimum)) * 0.05, 1e-6)
        minimum -= padding
        maximum += padding
    if positive:
        minimum = max(float(minimum), np.finfo(np.float64).tiny)
    return float(minimum), float(maximum)


def _spread_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest-point subset for uncluttered surface glyphs."""
    point_count = points.shape[0]
    if count >= point_count:
        return np.arange(point_count, dtype=np.int64)
    if count <= 0 or point_count == 0:
        return np.empty((0,), dtype=np.int64)
    center = points.mean(axis=0)
    first = int(np.argmax(np.sum(np.square(points - center), axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_distance = np.sum(np.square(points - points[first]), axis=1)
    for index in range(1, count):
        current = int(np.argmax(minimum_distance))
        selected[index] = current
        distance = np.sum(np.square(points - points[current]), axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return selected


def _mesh_diagonal(vertices: np.ndarray) -> float:
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return max(diagonal, 1e-12)


def _face_shading(normals: np.ndarray) -> np.ndarray:
    light = _normalise_vectors(np.array([[0.35, -0.45, 0.82]]))[0]
    lambert = np.clip(normals @ light, -0.15, 1.0)
    return 0.64 + 0.36 * (lambert + 0.15) / 1.15


def _draw_mesh(
    axis,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    *,
    values: Optional[np.ndarray] = None,
    cmap="viridis",
    norm: Optional[mcolors.Normalize] = None,
    alpha: float = 1.0,
):
    triangles = vertices[faces]
    shade = _face_shading(normals)[:, None]
    if values is None:
        colors = np.array([0.78, 0.80, 0.83])[None, :] * shade
    else:
        if norm is None:
            lower, upper = _robust_limits(values)
            norm = mcolors.Normalize(lower, upper)
        colors = plt.get_cmap(cmap)(norm(values))[:, :3] * shade
    colors = np.clip(colors, 0.0, 1.0)
    collection = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors=(0.16, 0.18, 0.20, 0.10),
        linewidths=0.12,
        alpha=alpha,
        antialiased=True,
    )
    collection.set_rasterized(True)
    axis.add_collection3d(collection)
    return norm


def _set_3d_view(
    axis,
    vertices: np.ndarray,
    *,
    elevation: float,
    azimuth: float,
    title: str,
):
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) * 0.5
    extent = np.maximum(maximum - minimum, 1e-12)
    half = extent.max() * 0.54
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()
    axis.set_title(title, fontsize=12, fontweight="semibold", pad=3)
    axis.set_facecolor("white")


def _cross_segments(
    centers: np.ndarray,
    first_axes: np.ndarray,
    second_axes: np.ndarray,
    half_lengths: np.ndarray,
    normals: np.ndarray,
    normal_offset: float,
) -> np.ndarray:
    raised = centers + normal_offset * normals
    first = np.stack(
        (
            raised - half_lengths[:, None] * first_axes,
            raised + half_lengths[:, None] * first_axes,
        ),
        axis=1,
    )
    second = np.stack(
        (
            raised - half_lengths[:, None] * second_axes,
            raised + half_lengths[:, None] * second_axes,
        ),
        axis=1,
    )
    return np.stack((first, second), axis=1).reshape(-1, 2, 3)


def _ellipse_segments(
    centers: np.ndarray,
    first_axes: np.ndarray,
    second_axes: np.ndarray,
    anisotropies: np.ndarray,
    radius: float,
    normals: np.ndarray,
    normal_offset: float,
    samples: int = 40,
) -> list[np.ndarray]:
    parameter = np.linspace(0.0, 2.0 * np.pi, samples)
    segments = []
    for center, first, second, anisotropy, normal in zip(
        centers, first_axes, second_axes, anisotropies, normals
    ):
        # This constant-area ellipse is a visual indicator of support
        # orientation/anisotropy, not an extrinsic approximation of a geodesic
        # heat-kernel level set.
        ratio = np.clip(float(anisotropy), 1.0, 200.0) ** 0.25
        major = radius * ratio
        minor = radius / ratio
        raised = center + normal_offset * normal
        curve = (
            raised[None, :]
            + major * np.cos(parameter)[:, None] * first[None, :]
            + minor * np.sin(parameter)[:, None] * second[None, :]
        )
        segments.append(curve)
    return segments


def _dense_geometry(payload: Mapping) -> dict[str, np.ndarray]:
    vertices = _as_numpy(payload["vertices"], np.float64)
    faces = _as_numpy(payload["faces"], np.int64)
    tangent_u = _as_numpy(payload["face_tangent_u"], np.float64)
    tangent_v = _as_numpy(payload["face_tangent_v"], np.float64)
    normals = _as_numpy(payload["face_normals"], np.float64)
    angles = _as_numpy(payload["cross_angle"], np.float64).reshape(-1)
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    direction_u = cosine * tangent_u + sine * tangent_v
    direction_v = -sine * tangent_u + cosine * tangent_v
    return {
        "vertices": vertices,
        "faces": faces,
        "centroids": vertices[faces].mean(axis=1),
        "normals": _normalise_vectors(normals),
        "direction_u": _normalise_vectors(direction_u),
        "direction_v": _normalise_vectors(direction_v),
        "coherence": _as_numpy(payload["field_coherence"], np.float64).reshape(-1),
        "magnitude": _as_numpy(payload["field_magnitude"], np.float64).reshape(-1),
        "coverage": _as_numpy(payload["coverage"], np.float64).reshape(-1),
        "target_size": _as_numpy(payload["target_size"], np.float64).reshape(-1),
        "valid": _as_numpy(payload["field_valid"], bool).reshape(-1),
    }


def _kernel_geometry(
    payload: Mapping, kernel_data: Mapping
) -> dict[str, np.ndarray]:
    tangent_u = _as_numpy(payload["face_tangent_u"], np.float64)
    tangent_v = _as_numpy(payload["face_tangent_v"], np.float64)
    normals = _normalise_vectors(
        _as_numpy(payload["face_normals"], np.float64)
    )
    face_ids = _as_numpy(kernel_data["face_ids"], np.int64).reshape(-1)
    cross_angles = _as_numpy(kernel_data["cross_angles"], np.float64).reshape(-1)
    support_angles = _as_numpy(
        kernel_data["support_angles"], np.float64
    ).reshape(-1)
    face_u = tangent_u[face_ids]
    face_v = tangent_v[face_ids]

    cross_cosine = np.cos(cross_angles)[:, None]
    cross_sine = np.sin(cross_angles)[:, None]
    support_cosine = np.cos(support_angles)[:, None]
    support_sine = np.sin(support_angles)[:, None]
    return {
        "locations": _as_numpy(kernel_data["locations"], np.float64),
        "face_ids": face_ids,
        "normals": normals[face_ids],
        "cross_u": _normalise_vectors(
            cross_cosine * face_u + cross_sine * face_v
        ),
        "cross_v": _normalise_vectors(
            -cross_sine * face_u + cross_cosine * face_v
        ),
        "support_u": _normalise_vectors(
            support_cosine * face_u + support_sine * face_v
        ),
        "support_v": _normalise_vectors(
            -support_sine * face_u + support_cosine * face_v
        ),
        **{
            key: _as_numpy(kernel_data[key], np.float64).reshape(-1)
            for key in (
                "cross_angles",
                "cross_amplitudes",
                "target_sizes",
                "support_angles",
                "anisotropies",
                "thresholds",
                "sharpnesses",
                "diffusion_times",
            )
        },
    }


def _save_kernel_table(kernel: Mapping[str, np.ndarray], filename: Path) -> None:
    columns = (
        "kernel_id",
        "face_id",
        "x",
        "y",
        "z",
        "cross_angle_rad",
        "cross_amplitude",
        "target_size",
        "support_angle_rad",
        "anisotropy",
        "threshold",
        "sharpness",
        "diffusion_time",
    )
    with filename.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index in range(kernel["locations"].shape[0]):
            location = kernel["locations"][index]
            writer.writerow(
                (
                    index,
                    int(kernel["face_ids"][index]),
                    *location.tolist(),
                    kernel["cross_angles"][index],
                    kernel["cross_amplitudes"][index],
                    kernel["target_sizes"][index],
                    kernel["support_angles"][index],
                    kernel["anisotropies"][index],
                    kernel["thresholds"][index],
                    kernel["sharpnesses"][index],
                    kernel["diffusion_times"][index],
                )
            )


def _save_overview(
    dense: Mapping[str, np.ndarray],
    kernel: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    dense_glyph_count: int,
    kernel_glyph_count: int,
    dpi: int,
    elevation: float,
    azimuth: float,
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    vertices = dense["vertices"]
    faces = dense["faces"]
    normals = dense["normals"]
    diagonal = _mesh_diagonal(vertices)
    normal_offset = diagonal * 0.0025

    valid_faces = np.flatnonzero(dense["valid"])
    selected_local = _spread_indices(
        dense["centroids"][valid_faces], min(dense_glyph_count, valid_faces.size)
    )
    selected_faces = valid_faces[selected_local]
    target_median = max(float(np.median(dense["target_size"][valid_faces])), 1e-12)
    dense_lengths = diagonal * 0.013 * np.clip(
        dense["target_size"][selected_faces] / target_median,
        0.55,
        1.65,
    )
    dense_segments = _cross_segments(
        dense["centroids"][selected_faces],
        dense["direction_u"][selected_faces],
        dense["direction_v"][selected_faces],
        dense_lengths,
        normals[selected_faces],
        normal_offset,
    )

    selected_kernels = _spread_indices(
        kernel["locations"],
        min(kernel_glyph_count, kernel["locations"].shape[0]),
    )
    kernel_size_median = max(float(np.median(kernel["target_sizes"])), 1e-12)
    kernel_lengths = diagonal * 0.017 * np.clip(
        kernel["target_sizes"][selected_kernels] / kernel_size_median,
        0.6,
        1.55,
    )
    kernel_segments = _cross_segments(
        kernel["locations"][selected_kernels],
        kernel["cross_u"][selected_kernels],
        kernel["cross_v"][selected_kernels],
        kernel_lengths,
        kernel["normals"][selected_kernels],
        normal_offset * 1.5,
    )

    figure = plt.figure(figsize=(17.0, 5.8), constrained_layout=True)
    coherence_axis = figure.add_subplot(1, 3, 1, projection="3d")
    size_axis = figure.add_subplot(1, 3, 2, projection="3d")
    kernel_axis = figure.add_subplot(1, 3, 3, projection="3d")

    _draw_mesh(coherence_axis, vertices, faces, normals)
    coherence_norm = mcolors.Normalize(0.0, 1.0)
    coherence_colors = plt.get_cmap("viridis")(
        coherence_norm(dense["coherence"][selected_faces])
    )
    coherence_axis.add_collection3d(
        Line3DCollection(
            dense_segments,
            colors=np.repeat(coherence_colors, 2, axis=0),
            linewidths=0.85,
            alpha=0.98,
        )
    )
    figure.colorbar(
        ScalarMappable(norm=coherence_norm, cmap="viridis"),
        ax=coherence_axis,
        fraction=0.032,
        pad=0.01,
        shrink=0.72,
        label="field coherence",
    )
    _set_3d_view(
        coherence_axis,
        vertices,
        elevation=elevation,
        azimuth=azimuth,
        title="(a) Dense 4-RoSy field",
    )

    size_min, size_max = _robust_limits(dense["target_size"], positive=True)
    size_norm = mcolors.LogNorm(size_min, size_max)
    _draw_mesh(
        size_axis,
        vertices,
        faces,
        normals,
        values=dense["target_size"],
        cmap="magma_r",
        norm=size_norm,
    )
    size_axis.add_collection3d(
        Line3DCollection(
            dense_segments,
            colors=(0.03, 0.07, 0.10, 0.74),
            linewidths=0.58,
        )
    )
    figure.colorbar(
        ScalarMappable(norm=size_norm, cmap="magma_r"),
        ax=size_axis,
        fraction=0.032,
        pad=0.01,
        shrink=0.72,
        label="target edge size h",
    )
    _set_3d_view(
        size_axis,
        vertices,
        elevation=elevation,
        azimuth=azimuth + 112.0,
        title="(b) Learned size field",
    )

    _draw_mesh(kernel_axis, vertices, faces, normals, alpha=0.88)
    amplitude_min, amplitude_max = _robust_limits(kernel["cross_amplitudes"])
    amplitude_norm = mcolors.Normalize(amplitude_min, amplitude_max)
    amplitude_colors = plt.get_cmap("plasma")(
        amplitude_norm(kernel["cross_amplitudes"])
    )
    raised_kernel_locations = (
        kernel["locations"] + normal_offset * 1.75 * kernel["normals"]
    )
    target_ratio = np.clip(
        kernel["target_sizes"] / kernel_size_median, 0.5, 1.8
    )
    kernel_axis.scatter(
        raised_kernel_locations[:, 0],
        raised_kernel_locations[:, 1],
        raised_kernel_locations[:, 2],
        s=7.0 + 15.0 * target_ratio,
        c=amplitude_colors,
        edgecolors="white",
        linewidths=0.24,
        alpha=0.96,
        depthshade=True,
    )
    selected_colors = amplitude_colors[selected_kernels]
    kernel_axis.add_collection3d(
        Line3DCollection(
            kernel_segments,
            colors=np.repeat(selected_colors, 2, axis=0),
            linewidths=1.12,
            alpha=0.98,
        )
    )
    support_ellipses = _ellipse_segments(
        kernel["locations"][selected_kernels],
        kernel["support_u"][selected_kernels],
        kernel["support_v"][selected_kernels],
        kernel["anisotropies"][selected_kernels],
        diagonal * 0.0105,
        kernel["normals"][selected_kernels],
        normal_offset,
    )
    for curve, color in zip(support_ellipses, selected_colors):
        kernel_axis.plot(
            curve[:, 0],
            curve[:, 1],
            curve[:, 2],
            color=color,
            linewidth=0.45,
            alpha=0.48,
        )
    figure.colorbar(
        ScalarMappable(norm=amplitude_norm, cmap="plasma"),
        ax=kernel_axis,
        fraction=0.032,
        pad=0.01,
        shrink=0.72,
        label="kernel cross amplitude",
    )
    kernel_axis.legend(
        handles=(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#d24b9f",
                markersize=5,
                label="all kernel centers",
            ),
            Line2D(
                [0],
                [0],
                color="#d24b9f",
                linewidth=1.2,
                label="kernel 4-RoSy cross",
            ),
            Line2D(
                [0],
                [0],
                color="#d24b9f",
                linewidth=0.6,
                alpha=0.55,
                label="support anisotropy indicator",
            ),
        ),
        loc="lower left",
        frameon=True,
        framealpha=0.86,
        fontsize=7.2,
    )
    _set_3d_view(
        kernel_axis,
        vertices,
        elevation=elevation,
        azimuth=azimuth,
        title="(c) Sparse heat-kernel parameters",
    )

    figure.suptitle(
        "HKTex2Quad: learned cross field, size field, and sparse kernel basis",
        fontsize=15,
        fontweight="semibold",
    )
    png_path = output_dir / "field_and_kernels.png"
    pdf_path = output_dir / "field_and_kernels.pdf"
    figure.savefig(png_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path, selected_faces, selected_kernels


def _loss_arrays(loss_history) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not loss_history:
        return np.empty((0,), dtype=np.int64), {}
    keys = (
        "total",
        "direction_smoothness",
        "guidance_alignment",
        "field_strength",
        "size_target",
    )
    available = {
        key: np.asarray(
            [float(row.get(key, np.nan)) for row in loss_history],
            dtype=np.float64,
        )
        for key in keys
        if any(key in row for row in loss_history)
    }
    return np.arange(1, len(loss_history) + 1), available


def _save_diagnostics(
    payload: Mapping,
    dense: Mapping[str, np.ndarray],
    kernel: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    dpi: int,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), constrained_layout=True)
    loss_axis, quality_axis, scatter_axis, support_axis = axes.reshape(-1)

    steps, loss_arrays = _loss_arrays(payload.get("loss_history"))
    colors = ("#1f4e79", "#d95f02", "#1b9e77", "#7570b3", "#666666")
    for (name, values), color in zip(loss_arrays.items(), colors):
        finite = np.isfinite(values) & (values > 0)
        if finite.any():
            loss_axis.plot(
                steps[finite],
                values[finite],
                label=name.replace("_", " "),
                color=color,
                linewidth=1.35,
            )
    if loss_arrays:
        loss_axis.set_yscale("log")
        loss_axis.legend(frameon=False, fontsize=8, ncol=2)
    else:
        loss_axis.text(
            0.5,
            0.5,
            "loss history unavailable",
            ha="center",
            va="center",
            transform=loss_axis.transAxes,
        )
    loss_axis.set_xlabel("optimization step")
    loss_axis.set_ylabel("loss (log scale)")
    loss_axis.set_title("(a) Training objective")
    loss_axis.grid(alpha=0.18, linewidth=0.6)

    bins = np.linspace(0.0, 1.0, 36)
    quality_axis.hist(
        dense["coherence"][dense["valid"]],
        bins=bins,
        color="#2a9d8f",
        alpha=0.78,
        density=True,
        label="coherence",
    )
    magnitude = dense["magnitude"][dense["valid"]]
    magnitude_scale = max(float(np.quantile(magnitude, 0.95)), 1e-12)
    quality_axis.hist(
        np.clip(magnitude / magnitude_scale, 0.0, 1.0),
        bins=bins,
        color="#e76f51",
        alpha=0.55,
        density=True,
        label="magnitude / q95",
    )
    quality_axis.set_xlabel("normalized value")
    quality_axis.set_ylabel("density")
    quality_axis.set_title("(b) Dense-field quality")
    quality_axis.legend(frameon=False, fontsize=8)
    quality_axis.grid(alpha=0.18, linewidth=0.6)

    target_min, target_max = _robust_limits(kernel["target_sizes"], positive=True)
    target_norm = mcolors.LogNorm(target_min, target_max)
    threshold_min, threshold_max = _robust_limits(kernel["thresholds"])
    threshold_span = max(threshold_max - threshold_min, 1e-12)
    marker_sizes = 12.0 + 45.0 * np.clip(
        (kernel["thresholds"] - threshold_min) / threshold_span, 0.0, 1.0
    )
    scatter = scatter_axis.scatter(
        kernel["anisotropies"],
        kernel["cross_amplitudes"],
        c=kernel["target_sizes"],
        cmap="magma_r",
        norm=target_norm,
        s=marker_sizes,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.25,
    )
    scatter_axis.set_xscale("log")
    scatter_axis.set_xlabel("support anisotropy (log scale)")
    scatter_axis.set_ylabel("cross amplitude")
    scatter_axis.set_title("(c) Kernel attributes")
    scatter_axis.grid(alpha=0.18, linewidth=0.6)
    figure.colorbar(scatter, ax=scatter_axis, label="target edge size h")

    support_axis.hist(
        kernel["thresholds"],
        bins=32,
        color="#457b9d",
        alpha=0.78,
        label="threshold",
    )
    sharp_axis = support_axis.twiny()
    sharp_axis.hist(
        kernel["sharpnesses"],
        bins=32,
        histtype="step",
        color="#e63946",
        linewidth=1.35,
        label="sharpness",
    )
    support_axis.set_xlabel("support threshold")
    support_axis.set_ylabel("kernel count")
    sharp_axis.set_xlabel("support sharpness", color="#e63946")
    support_axis.set_title("(d) Support falloff parameters")
    support_axis.grid(alpha=0.18, linewidth=0.6)
    support_axis.legend(loc="upper left", frameon=False, fontsize=8)
    sharp_axis.legend(loc="upper right", frameon=False, fontsize=8)

    singularity_summary = payload.get("singularity_summary", {}) or {}
    subtitle = (
        f"faces={dense['faces'].shape[0]:,}, kernels={kernel['locations'].shape[0]:,}, "
        f"median coherence={np.median(dense['coherence']):.3f}, "
        f"singularities={singularity_summary.get('nonzero_count', 'n/a')}"
    )
    figure.suptitle(
        "HKTex2Quad optimization diagnostics\n" + subtitle,
        fontsize=14,
        fontweight="semibold",
    )
    png_path = output_dir / "training_diagnostics.png"
    pdf_path = output_dir / "training_diagnostics.pdf"
    figure.savefig(png_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def save_quad_field_visualizations(
    dense_payload: Mapping,
    kernel_data: Mapping,
    output_dir: str | Path,
    *,
    dense_glyph_count: int = 700,
    kernel_glyph_count: int = 192,
    dpi: int = 240,
    elevation: float = 24.0,
    azimuth: float = -58.0,
) -> dict[str, Path]:
    """Save paper-ready field/kernel figures and machine-readable diagnostics."""
    if dense_glyph_count <= 0 or kernel_glyph_count <= 0 or dpi <= 0:
        raise ValueError("glyph counts and dpi must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dense = _dense_geometry(dense_payload)
    kernel = _kernel_geometry(dense_payload, kernel_data)
    if kernel["locations"].shape[0] == 0:
        raise ValueError("at least one kernel is required for visualization")

    overview_png, overview_pdf, selected_faces, selected_kernels = _save_overview(
        dense,
        kernel,
        output_dir,
        dense_glyph_count=dense_glyph_count,
        kernel_glyph_count=kernel_glyph_count,
        dpi=dpi,
        elevation=elevation,
        azimuth=azimuth,
    )
    diagnostics_png, diagnostics_pdf = _save_diagnostics(
        dense_payload, dense, kernel, output_dir, dpi=dpi
    )

    table_path = output_dir / "kernel_parameters.csv"
    _save_kernel_table(kernel, table_path)
    arrays_path = output_dir / "field_visualization_data.npz"
    np.savez_compressed(
        arrays_path,
        dense_face_indices=selected_faces,
        dense_centers=dense["centroids"][selected_faces],
        dense_direction_u=dense["direction_u"][selected_faces],
        dense_direction_v=dense["direction_v"][selected_faces],
        dense_coherence=dense["coherence"][selected_faces],
        dense_magnitude=dense["magnitude"][selected_faces],
        dense_target_size=dense["target_size"][selected_faces],
        selected_kernel_indices=selected_kernels,
        **{f"kernel_{key}": value for key, value in kernel.items()},
    )

    report = {
        "format": "hktex2quad-field-visualization-v1",
        "dense_face_count": int(dense["faces"].shape[0]),
        "dense_glyph_count": int(selected_faces.shape[0]),
        "kernel_count": int(kernel["locations"].shape[0]),
        "kernel_glyph_count": int(selected_kernels.shape[0]),
        "encoding": {
            "dense_cross": "two orthogonal lines represent the 4-RoSy cross",
            "dense_cross_color": "field coherence in [0, 1]",
            "dense_cross_length": "target edge size, robustly clipped for legibility",
            "kernel_center": "all learned heat-kernel locations",
            "kernel_color": "cross coefficient amplitude",
            "kernel_cross": "learned kernel 4-RoSy direction",
            "kernel_cross_length": "kernel target edge size, robustly clipped",
            "kernel_ellipse": (
                "schematic constant-area indicator of support angle and "
                "anisotropy; not a geodesic heat-kernel contour"
            ),
            "kernel_marker_size": "kernel target edge size",
        },
        "statistics": {
            "coherence_quantiles": np.quantile(
                dense["coherence"], (0.0, 0.25, 0.5, 0.75, 1.0)
            ).tolist(),
            "magnitude_quantiles": np.quantile(
                dense["magnitude"], (0.0, 0.25, 0.5, 0.75, 1.0)
            ).tolist(),
            "kernel_amplitude_quantiles": np.quantile(
                kernel["cross_amplitudes"], (0.0, 0.25, 0.5, 0.75, 1.0)
            ).tolist(),
            "kernel_anisotropy_quantiles": np.quantile(
                kernel["anisotropies"], (0.0, 0.25, 0.5, 0.75, 1.0)
            ).tolist(),
        },
        "files": {
            "overview_png": overview_png.name,
            "overview_pdf": overview_pdf.name,
            "diagnostics_png": diagnostics_png.name,
            "diagnostics_pdf": diagnostics_pdf.name,
            "kernel_table": table_path.name,
            "arrays": arrays_path.name,
        },
    }
    report_path = output_dir / "visualization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "overview_png": overview_png,
        "overview_pdf": overview_pdf,
        "diagnostics_png": diagnostics_png,
        "diagnostics_pdf": diagnostics_pdf,
        "kernel_table": table_path,
        "arrays": arrays_path,
        "report": report_path,
    }
