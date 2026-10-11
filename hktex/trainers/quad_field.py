"""GPU training loop for the HKTex2Quad cross and target-size fields."""

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import torch
from torch import Tensor

import hktex
from hktex.data import MeshSamplerDataModule
from hktex.modules import (
    HeatKernelCrossFieldKNN,
    build_connection_reference_field,
    boundary_cross_constraints,
    curvature_target_sizes,
    detect_face_field_singularities,
    face_adjacency_connection,
    face_adjacency_edge_coordinates,
    face_connection_weights,
    principal_direction_face_guidance,
    quad_field_objective,
    vertex_angle_defects,
)
from hktex.utils import compute_principal_curvature_data

from .base import BaseTrainer


@hktex.register("trainers.quad-field")
class QuadFieldTrainer(BaseTrainer):
    """Optimise a sparse HKTex representation against geometric field targets.

    Training samples dual edges and faces, then evaluates only the union of
    required face centroids.  Heat/KNN queries, source-to-query transport,
    field aggregation, and all differentiable losses remain on the GPU.
    CPU work is restricted to one-time mesh preprocessing and periodic
    singularity analysis.
    """

    @dataclass
    class Config(BaseTrainer.Config):
        model_type: Optional[str] = "modules.heat-kernel-cross-field-knn"
        use_knn_implementation: bool = True
        eigen_albo_type: str = "modules.eigen-albo-interpolation-knn"
        tracer_type: str = "modules.gpu-geodesic-tracer"

        face_sample_count: int = 4096
        edge_sample_count: int = 4096
        boundary_sample_count: int = 1024
        forward_batch_size: int = 4096

        target_quad_count: int = 10000
        base_target_size: Optional[float] = None
        # Direction-only experiment: prescribe one external size at every face
        # and freeze the kernel size parameters.
        fixed_target_size: bool = False
        fixed_kernel_sharpness: bool = False
        fixed_cross_amplitude: bool = False
        fixed_kernel_anisotropy: bool = False
        fixed_support_angle: bool = False
        fixed_kernel_threshold: bool = False
        curvature_size_scale: float = 4.0
        min_target_size: float = 1e-4
        max_target_size: float = 1e4
        # Principal directions become unstable near umbilics.  Ignore weak
        # curvature guidance there and let the connection energy interpolate
        # the cross field instead of fitting high-frequency noise.
        guidance_min_weight: float = 0.05
        initialize_from_geometry: bool = True
        supervision: str = "geometry"
        # Keep the same geometry initialization in both modes. Only the
        # training target changes: a completed reference, or reliable local
        # principal directions with their original confidence weights.
        geometry_guidance_mode: str = "reference"
        quad_supervision_path: Optional[str] = None
        quad_min_confidence: float = 0.1
        # Select a globally smooth 4-RoSy field in the low spectrum of the
        # connection Laplacian.  Principal curvature only breaks ties between
        # topologically clean candidates; it is not fitted face by face.
        use_connection_reference: bool = True
        reference_eigenvector_count: int = 4
        reference_candidate_count: int = 48
        reference_seed: int = 0

        direction_smoothness_weight: float = 2.0
        guidance_alignment_weight: float = 5.0
        boundary_alignment_weight: float = 2.0
        size_smoothness_weight: float = 0.1
        size_target_weight: float = 1.0
        field_strength_weight: float = 0.2
        integrability_weight: float = 0.5
        direction_tail_weight: float = 0.5
        coherence_weight: float = 0.25
        coverage_weight: float = 0.1
        maximum_direction_jump_degrees: float = 10.0
        minimum_coherence: float = 0.5
        minimum_coverage: float = 0.25

        gradient_clip_norm: Optional[float] = 10.0
        singularity_report_frequency: int = 250

        # Saved automatically beside dense_face_field.pt.
        save_field_visualization: bool = True
        visualization_dense_glyph_count: int = 700
        visualization_kernel_glyph_count: int = 192
        visualization_dpi: int = 240
        visualization_elevation: float = 24.0
        visualization_azimuth: float = -58.0

    cfg: Config
    model: HeatKernelCrossFieldKNN

    def configure(self, datamodule: MeshSamplerDataModule, **kwargs):
        if self.cfg.geometry_guidance_mode not in ("reference", "curvature"):
            raise ValueError("geometry_guidance_mode must be reference or curvature")
        if self.cfg.supervision not in ("geometry", "quad"):
            raise ValueError("supervision must be geometry or quad")
        if self.cfg.supervision == "quad" and not self.cfg.quad_supervision_path:
            raise ValueError("quad supervision requires quad_supervision_path")
        if self.cfg.quad_supervision_path and not Path(self.cfg.quad_supervision_path).is_file():
            raise FileNotFoundError(self.cfg.quad_supervision_path)
        # Accept the flags beside fixed_target_size as well as in model.
        # A true model-level flag is not overridden by the trainer default.
        requested_freezes = {
            name: True for name in (
                "fixed_kernel_sharpness", "fixed_cross_amplitude",
                "fixed_kernel_anisotropy", "fixed_support_angle",
                "fixed_kernel_threshold",
            ) if getattr(self.cfg, name)
        }
        if requested_freezes:
            self.cfg.model = {**dict(self.cfg.model), **requested_freezes}
        super().configure(datamodule, **kwargs)
        if not isinstance(self.model, HeatKernelCrossFieldKNN):
            raise TypeError(
                "QuadFieldTrainer requires modules.heat-kernel-cross-field-knn"
            )
        if self.cfg.face_sample_count <= 0 or self.cfg.forward_batch_size <= 0:
            raise ValueError("face_sample_count and forward_batch_size must be positive")
        if self.cfg.target_quad_count <= 0 and self.cfg.base_target_size is None:
            raise ValueError(
                "target_quad_count must be positive when base_target_size is unset"
            )
        if (
            self.cfg.visualization_dense_glyph_count <= 0
            or self.cfg.visualization_kernel_glyph_count <= 0
            or self.cfg.visualization_dpi <= 0
        ):
            raise ValueError("visualization glyph counts and dpi must be positive")
        quality_values = (
            self.cfg.integrability_weight,
            self.cfg.direction_tail_weight,
            self.cfg.coherence_weight,
            self.cfg.coverage_weight,
            self.cfg.minimum_coherence,
            self.cfg.minimum_coverage,
        )
        if any(value < 0 for value in quality_values):
            raise ValueError("field-quality weights and thresholds must be non-negative")
        if self.cfg.maximum_direction_jump_degrees <= 0:
            raise ValueError("maximum_direction_jump_degrees must be positive")
        if (
            self.cfg.reference_eigenvector_count <= 0
            or self.cfg.reference_candidate_count < 0
        ):
            raise ValueError("connection-reference counts are invalid")

        vertices = self.mesh.verts
        faces = self.mesh.faces.long()
        self.face_centroids = vertices[faces].mean(dim=1)
        (
            self.face_frames,
            self.face_adjacency,
            self.connection_angles,
        ) = face_adjacency_connection(
            vertices,
            faces,
            face_normals=self.mesh.fnorms,
        )
        self.connection_weights = face_connection_weights(
            vertices, faces, self.face_adjacency
        )
        (
            self.edge_source_coordinates,
            self.edge_target_coordinates,
        ) = face_adjacency_edge_coordinates(
            vertices,
            faces,
            self.face_adjacency,
            self.face_frames,
        )
        self.boundary_constraints = boundary_cross_constraints(
            vertices,
            faces,
            frames=self.face_frames,
            degree=self.model.cfg.rosy_degree,
        )
        self.angle_defects = vertex_angle_defects(vertices, faces)

        # EigenAlbo already computes the principal direction used by the
        # anisotropic heat basis.  Curvature values are recovered once here to
        # gate unreliable directions and construct the target size field.
        vertex_directions = self.eigalbo_interp._local_direction.to(
            device=vertices.device, dtype=vertices.dtype
        )
        vertex_curvatures = None
        try:
            _, _, k1, k2 = compute_principal_curvature_data(
                datamodule.mesh.vertices, datamodule.mesh.faces
            )
            vertex_curvatures = torch.stack((k1, k2), dim=-1).to(
                device=vertices.device, dtype=vertices.dtype
            )
        except Exception as exc:
            hktex.warn(
                "Principal curvature values are unavailable; using uniform "
                f"guidance weights and target sizes. Reason: {exc}"
            )

        self.curvature_guidance_power, self.curvature_guidance_weights = (
            principal_direction_face_guidance(
                faces,
                vertex_directions,
                self.face_frames,
                vertex_curvatures=vertex_curvatures,
                degree=self.model.cfg.rosy_degree,
            )
        )
        self.curvature_guidance_weights = torch.where(
            self.curvature_guidance_weights >= self.cfg.guidance_min_weight,
            self.curvature_guidance_weights,
            torch.zeros_like(self.curvature_guidance_weights),
        )
        self.connection_reference = None
        if self.cfg.use_connection_reference:
            self.connection_reference = build_connection_reference_field(
                self.curvature_guidance_power,
                self.curvature_guidance_weights,
                faces,
                self.face_adjacency,
                self.connection_angles,
                self.angle_defects,
                self.connection_weights,
                num_vertices=self.mesh.N_verts,
                degree=self.model.cfg.rosy_degree,
                eigenvector_count=self.cfg.reference_eigenvector_count,
                random_candidate_count=self.cfg.reference_candidate_count,
                seed=self.cfg.reference_seed,
            )
            self.guidance_power = self.connection_reference.power_field
            # Once a globally valid candidate has been selected it is a dense
            # reference, including at curvature umbilics where the original
            # principal direction is unreliable.
            self.guidance_weights = torch.ones_like(
                self.curvature_guidance_weights
            )
            hktex.info(
                "Connection reference: "
                f"lambda0={self.connection_reference.eigenvalue:.6g}, "
                f"singularities={self.connection_reference.nonzero_singularity_count}, "
                f"wrong-sign={self.connection_reference.wrong_sign_singularity_count}, "
                f"high-order={self.connection_reference.high_order_singularity_count}, "
                f"curvature alignment loss={self.connection_reference.alignment_loss:.6g}"
            )
        else:
            self.guidance_power = self.curvature_guidance_power
            self.guidance_weights = self.curvature_guidance_weights

        # Both modes retain the original geometry initialization. Reference
        # labels supplied to geometry mode are exported for evaluation only.
        self.initial_guidance_power = self.guidance_power
        self.initial_guidance_weights = self.guidance_weights
        geometry_source = (
            "connection_reference" if self.cfg.use_connection_reference else "curvature"
        )
        self.initialization_source = (
            geometry_source if self.cfg.initialize_from_geometry else "model_default"
        )
        self.training_guidance_source = geometry_source
        if (
            self.cfg.supervision == "geometry"
            and self.cfg.geometry_guidance_mode == "curvature"
        ):
            self.guidance_power = self.curvature_guidance_power
            self.guidance_weights = self.curvature_guidance_weights
            self.training_guidance_source = "curvature"
        self.quad_guidance_power = None
        self.quad_guidance_weights = None
        if self.cfg.supervision not in ("geometry", "quad"):
            raise ValueError("supervision must be geometry or quad")
        if self.cfg.quad_supervision_path:
            if self.model.cfg.rosy_degree != 4:
                raise ValueError("Quad reference labels require rosy_degree=4")
            from hktex.modules.quad_supervision import load_quad_targets
            power, confidence = load_quad_targets(
                self.cfg.quad_supervision_path,
                vertices.detach().cpu().numpy(), faces.detach().cpu().numpy(),
                self.face_frames.tangent_u.detach().cpu().numpy(),
                self.face_frames.tangent_v.detach().cpu().numpy(),
                self.cfg.quad_min_confidence,
            )
            self.quad_guidance_power = torch.as_tensor(power, device=vertices.device, dtype=vertices.dtype)
            self.quad_guidance_weights = torch.as_tensor(confidence, device=vertices.device, dtype=vertices.dtype)
        if self.cfg.supervision == "quad":
            if self.quad_guidance_power is None:
                raise ValueError("quad supervision requires quad_supervision_path")
            if not bool((self.quad_guidance_weights > 0).any()):
                raise ValueError("No valid quad supervision labels")
            self.guidance_power = self.quad_guidance_power
            self.guidance_weights = self.quad_guidance_weights
            self.training_guidance_source = "quad"
        hktex.info(
            f"Direction supervision: {self.cfg.supervision}; "
            f"training guidance: {self.training_guidance_source}; "
            f"initialization: {self.initialization_source}"
        )

        base_size = self.cfg.base_target_size
        if base_size is None:
            base_size = math.sqrt(
                float(self.mesh.tot_area.item()) / self.cfg.target_quad_count
            )
        self.base_target_size = float(base_size)
        if self.cfg.fixed_target_size:
            minimum = max(self.cfg.min_target_size, self.model.cfg.min_target_size)
            maximum = min(self.cfg.max_target_size, self.model.cfg.max_target_size)
            if not (
                math.isfinite(self.base_target_size)
                and 0 < minimum <= self.base_target_size <= maximum
            ):
                raise ValueError("fixed target size must be finite and within size bounds")
            with torch.no_grad():
                self.model._target_log_sizes.fill_(math.log(self.base_target_size))
            self.model._target_log_sizes.requires_grad_(False)
        if vertex_curvatures is None or self.cfg.fixed_target_size:
            target_sizes = vertices.new_full(
                (faces.shape[0],), self.base_target_size
            )
        else:
            target_sizes = curvature_target_sizes(
                faces,
                vertex_curvatures,
                self.base_target_size,
                curvature_scale=self.cfg.curvature_size_scale,
                min_size=max(
                    self.cfg.min_target_size, self.model.cfg.min_target_size
                ),
                max_size=min(
                    self.cfg.max_target_size, self.model.cfg.max_target_size
                ),
            )
        self.target_log_sizes = target_sizes.log()
        triangle_areas = 0.5 * torch.linalg.vector_norm(
            torch.cross(
                vertices[faces[:, 1]] - vertices[faces[:, 0]],
                vertices[faces[:, 2]] - vertices[faces[:, 0]],
                dim=-1,
            ),
            dim=-1,
        )
        self.target_size_weights = triangle_areas / triangle_areas.mean().clamp_min(
            1e-12
        )
        self.loss_history: list[dict[str, float]] = []
        self.last_singularity_summary: dict[str, float | int] = {}

    @torch.no_grad()
    def data_dependent_initialisation(self, **kwargs):
        """Initialize kernel attributes from their current source faces."""
        if not self.cfg.initialize_from_geometry:
            return
        source_faces = self.model.kernel_face_ids.long()
        guidance = self.initial_guidance_power[source_faces]
        magnitude = torch.linalg.vector_norm(guidance, dim=-1, keepdim=True)
        valid = self.initial_guidance_weights[source_faces] > 0
        unit_guidance = guidance / magnitude.clamp_min(1e-8)
        self.model._cross_coefficients[valid] = unit_guidance[valid]
        self.model._target_log_sizes.copy_(self.target_log_sizes[source_faces])

    def _sample(self, count: int, requested: int) -> Tensor:
        if count == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        if requested <= 0 or requested >= count:
            return torch.arange(count, dtype=torch.long, device=self.device)
        return torch.randint(count, (requested,), device=self.device)

    def _sample_training_faces(self) -> tuple[Tensor, Tensor, Tensor]:
        edge_ids = self._sample(
            self.face_adjacency.shape[0], self.cfg.edge_sample_count
        )
        face_ids = self._sample(self.mesh.N_faces, self.cfg.face_sample_count)
        boundary_ids = self._sample(
            self.boundary_constraints.face_indices.shape[0],
            self.cfg.boundary_sample_count,
        )
        required = [face_ids]
        if edge_ids.numel():
            required.append(self.face_adjacency[edge_ids].reshape(-1))
        if boundary_ids.numel():
            required.append(self.boundary_constraints.face_indices[boundary_ids])
        unique_faces = torch.unique(torch.cat(required), sorted=True)
        return unique_faces, edge_ids, boundary_ids

    def _evaluate_faces(self, face_ids: Tensor):
        outputs = []
        for start in range(0, face_ids.shape[0], self.cfg.forward_batch_size):
            batch_faces = face_ids[start : start + self.cfg.forward_batch_size]
            points_info = self.model.prepare_points(
                self.mesh,
                self.eigalbo_interp,
                batch_faces,
                pts=self.face_centroids[batch_faces],
            )
            output, _, _, _ = self.model.diffuse_cross_field(
                self.eigalbo_interp, points_info
            )
            if self.cfg.fixed_target_size:
                # Size is externally prescribed even where the direction is
                # invalid. Preserve raw direction validity and coverage;
                # neither missing directions nor weak support may change
                # scale to make the integrability objective easier.
                output = output._replace(
                    target_size=output.power_field.new_full(
                        (output.power_field.shape[0], 1), self.base_target_size
                    )
                )
            outputs.append(output)
        return self._concatenate_field_outputs(outputs)

    @staticmethod
    def _concatenate_field_outputs(outputs):
        if not outputs:
            raise ValueError("cannot concatenate an empty face evaluation")
        output_type = type(outputs[0])
        target_sizes = [output.target_size for output in outputs]
        target_size = None
        if all(value is not None for value in target_sizes):
            target_size = torch.cat(target_sizes, dim=0)
        elif not all(value is None for value in target_sizes):
            raise ValueError("face batches disagree on target-size availability")
        return output_type(
            power_field=torch.cat([value.power_field for value in outputs], dim=0),
            unit_power_field=torch.cat(
                [value.unit_power_field for value in outputs], dim=0
            ),
            magnitude=torch.cat([value.magnitude for value in outputs], dim=0),
            coherence=torch.cat([value.coherence for value in outputs], dim=0),
            angle=torch.cat([value.angle for value in outputs], dim=0),
            valid=torch.cat([value.valid for value in outputs], dim=0),
            coverage=torch.cat([value.coverage for value in outputs], dim=0),
            target_size=target_size,
        )

    def _training_objective(self, face_ids, edge_ids, boundary_ids):
        output = self._evaluate_faces(face_ids)
        if output.target_size is None:
            raise RuntimeError("quad-field model did not return a target-size field")

        if edge_ids.numel():
            sampled_edges = self.face_adjacency[edge_ids]
            local_edges = torch.searchsorted(face_ids, sampled_edges)
            connection_angles = self.connection_angles[edge_ids]
            edge_weights = self.connection_weights[edge_ids]
            source_edge_coordinates = self.edge_source_coordinates[edge_ids]
            target_edge_coordinates = self.edge_target_coordinates[edge_ids]
        else:
            local_edges = torch.empty((0, 2), dtype=torch.long, device=self.device)
            connection_angles = self.connection_angles[:0]
            edge_weights = self.connection_weights[:0]
            source_edge_coordinates = self.edge_source_coordinates[:0]
            target_edge_coordinates = self.edge_target_coordinates[:0]

        if boundary_ids.numel():
            boundary_faces = self.boundary_constraints.face_indices[boundary_ids]
            local_boundary = torch.searchsorted(face_ids, boundary_faces)
            boundary_power = self.boundary_constraints.target_power[boundary_ids]
            boundary_weights = self.boundary_constraints.weights[boundary_ids]
            boundary_weights = boundary_weights / boundary_weights.mean().clamp_min(
                1e-12
            )
        else:
            local_boundary = None
            boundary_power = None
            boundary_weights = None

        log_sizes = output.target_size.squeeze(-1).clamp_min(1e-12).log()
        losses = quad_field_objective(
            output.power_field,
            local_edges,
            connection_angles,
            guidance_power=self.guidance_power[face_ids],
            guidance_weights=self.guidance_weights[face_ids],
            boundary_power=boundary_power,
            boundary_indices=local_boundary,
            boundary_weights=boundary_weights,
            log_sizes=log_sizes,
            target_log_sizes=self.target_log_sizes[face_ids],
            target_size_weights=self.target_size_weights[face_ids],
            edge_weights=edge_weights,
            source_edge_coordinates=source_edge_coordinates,
            target_edge_coordinates=target_edge_coordinates,
            coherence=output.coherence,
            coverage=output.coverage,
            direction_smoothness_weight=self.cfg.direction_smoothness_weight,
            guidance_alignment_weight=self.cfg.guidance_alignment_weight,
            boundary_alignment_weight=self.cfg.boundary_alignment_weight,
            size_smoothness_weight=self.cfg.size_smoothness_weight,
            size_target_weight=self.cfg.size_target_weight,
            field_strength_weight=self.cfg.field_strength_weight,
            integrability_weight=self.cfg.integrability_weight,
            direction_tail_weight=self.cfg.direction_tail_weight,
            coherence_weight=self.cfg.coherence_weight,
            coverage_weight=self.cfg.coverage_weight,
            maximum_direction_jump=math.radians(
                self.cfg.maximum_direction_jump_degrees
            ),
            minimum_coherence=self.cfg.minimum_coherence,
            minimum_coverage=self.cfg.minimum_coverage,
            degree=self.model.cfg.rosy_degree,
        )
        return losses

    @torch.no_grad()
    def capture_initial_kernel_parameters(self):
        """Capture common geometry initialization before any optimizer step."""
        raw = self.model.raw_kernel_parameter_snapshot()
        digest = hashlib.sha256()
        for name, value in sorted(raw.items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(value.contiguous().numpy().tobytes())
        self.canonical_initial_kernel_sha256 = digest.hexdigest()
        self.initial_kernel_raw_parameters = raw
        self.initial_kernel_parameters = self.model.kernel_parameter_snapshot(
            raw_cross=True
        )
        if self.model.cfg.fixed_cross_amplitude:
            self.model.capture_fixed_cross_amplitudes()
        hktex.info(
            "Canonical initial kernel SHA256: "
            f"{self.canonical_initial_kernel_sha256}"
        )

    def kernel_freeze_metadata(self) -> dict:
        flags = {"fixed_target_size": self.cfg.fixed_target_size}
        for name in (
            "fixed_kernel_sharpness", "fixed_cross_amplitude",
            "fixed_kernel_anisotropy", "fixed_support_angle",
            "fixed_kernel_threshold",
        ):
            flags[name] = bool(
                getattr(self.cfg, name, False)
                or getattr(self.model.cfg, name, False)
            )
        metadata = {"freeze_flags": flags}
        # Existing lightweight policy fixtures need not provide kernel state.
        if not hasattr(self.model, "kernel_parameter_snapshot"):
            return metadata
        final = self.model.kernel_parameter_snapshot()
        initial = getattr(self, "initial_kernel_parameters", None)
        metadata.update({
            "canonical_initial_kernel_sha256": getattr(
                self, "canonical_initial_kernel_sha256", None
            ),
            "initial_kernel_parameters": initial,
            "initial_kernel_raw_parameters": getattr(
                self, "initial_kernel_raw_parameters", None
            ),
            "final_kernel_parameters": final,
            "kernel_parameter_max_abs_deltas": (
                {name: float((final[name].double() - value.double()).abs().max())
                 for name, value in initial.items()} if initial else None
            ),
            "kernel_gauge_metadata": {
                "cross_coefficients": "current_source_face_tangent_frame",
                "cross_face_transport": True,
                "support_angle": "eigenalbo_support_basis_parameter",
                "fixed_amplitude": "per_kernel_geometry_initialization_norm",
                "freeze_value_policy": "retain_each_kernel_initial_value",
            },
        })
        return metadata

    def optimise(self, n_iter=100, debug_log_dir=None):
        del debug_log_dir
        self.data_dependent_initialisation()
        self.capture_initial_kernel_parameters()
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.initial_model_sha256 = digest.hexdigest()
        hktex.info(f"Initial model SHA256: {self.initial_model_sha256}")
        pbar = tqdm(range(n_iter), desc="HKTex2Quad field")
        for step in pbar:
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)

            face_ids, edge_ids, boundary_ids = self._sample_training_faces()
            self.model.prepare_kernels(self.mesh, self.eigalbo_interp)
            losses = self._training_objective(face_ids, edge_ids, boundary_ids)
            losses.total.backward()

            if self.cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.cfg.gradient_clip_norm
                )

            gauge_snapshot = self.model.capture_kernel_gauge()
            for optimizer in self.optimizers:
                optimizer.step()
            moved_between_faces = self.model.restore_kernel_gauge(gauge_snapshot)
            for scheduler in self.schedulers:
                scheduler.step()
            self.model.post_optimizer_step()
            self.model.reset(self.eigalbo_interp)

            values = {
                name: float(value.detach().item())
                for name, value in losses._asdict().items()
            }
            values["moved_between_faces"] = moved_between_faces
            self.loss_history.append(values)
            pbar.set_postfix(
                loss=f"{values['total']:.5f}", moved=moved_between_faces
            )

            report_frequency = self.cfg.singularity_report_frequency
            if report_frequency > 0 and (step + 1) % report_frequency == 0:
                summary = self.singularity_summary()
                hktex.info(
                    f"Iteration {step + 1}: loss={values['total']:.6g}, "
                    f"valid singularities={summary['nonzero_count']}, "
                    f"index sum={summary['index_sum']:.6g}"
                )

        # Produce the final analysis now so a no-render run still validates the
        # topology signal before writing its checkpoint/export.
        self.last_singularity_summary = self.singularity_summary()
        return None

    @torch.no_grad()
    def evaluate_dense_field(self):
        face_ids = torch.arange(
            self.mesh.N_faces, dtype=torch.long, device=self.device
        )
        self.model.prepare_kernels(self.mesh, self.eigalbo_interp)
        try:
            return self._evaluate_faces(face_ids)
        finally:
            self.model.reset(self.eigalbo_interp)

    @torch.no_grad()
    def singularity_summary(self) -> dict[str, float | int]:
        dense = self.evaluate_dense_field()
        singularities = detect_face_field_singularities(
            dense.power_field,
            self.mesh.faces.long(),
            self.face_adjacency,
            self.connection_angles,
            self.angle_defects,
            num_vertices=self.mesh.N_verts,
            degree=self.model.cfg.rosy_degree,
        )
        nonzero = singularities.valid & (singularities.index.abs() > 1e-8)
        summary = {
            "valid_vertex_count": int(singularities.valid.sum().item()),
            "nonzero_count": int(nonzero.sum().item()),
            "index_sum": float(
                singularities.index[singularities.valid].sum().item()
            ),
            "mean_residual": float(
                singularities.residual[singularities.valid].mean().item()
                if bool(singularities.valid.any())
                else 0.0
            ),
        }
        self.last_singularity_summary = summary
        return summary

    @torch.no_grad()
    def export_dense_field(self, filename: str):
        """Write the stage-5 face-field interchange file for a backend."""
        dense = self.evaluate_dense_field()
        singularities = detect_face_field_singularities(
            dense.power_field,
            self.mesh.faces.long(),
            self.face_adjacency,
            self.connection_angles,
            self.angle_defects,
            num_vertices=self.mesh.N_verts,
            degree=self.model.cfg.rosy_degree,
        )
        payload = {
            "format": "hktex2quad-face-field-v1",
            "supervision": self.cfg.supervision,
            "geometry_guidance_mode": self.cfg.geometry_guidance_mode,
            "training_guidance_source": self.training_guidance_source,
            "initialization_source": self.initialization_source,
            "kernel_count": int(self.model.N_sources),
            "fixed_target_size": self.cfg.fixed_target_size,
            "target_size_source": (
                "fixed_external" if self.cfg.fixed_target_size else "kernel_aggregation"
            ),
            "fixed_target_size_value": (
                self.base_target_size if self.cfg.fixed_target_size else None
            ),
            "initial_model_sha256": getattr(self, "initial_model_sha256", None),
            "rosy_degree": self.model.cfg.rosy_degree,
            "vertices": self.mesh.verts.detach().cpu(),
            "faces": self.mesh.faces.detach().cpu(),
            "face_centroids": self.face_centroids.detach().cpu(),
            "face_tangent_u": self.face_frames.tangent_u.detach().cpu(),
            "face_tangent_v": self.face_frames.tangent_v.detach().cpu(),
            "face_normals": self.face_frames.normal.detach().cpu(),
            "power_field": dense.power_field.detach().cpu(),
            "unit_power_field": dense.unit_power_field.detach().cpu(),
            "cross_angle": dense.angle.detach().cpu(),
            "field_magnitude": dense.magnitude.detach().cpu(),
            "field_coherence": dense.coherence.detach().cpu(),
            "field_valid": dense.valid.detach().cpu(),
            "coverage": dense.coverage.detach().cpu(),
            "target_size": dense.target_size.detach().cpu(),
            "face_adjacency": self.face_adjacency.detach().cpu(),
            "connection_angles": self.connection_angles.detach().cpu(),
            "singularity_index": singularities.index.detach().cpu(),
            "singularity_valid": singularities.valid.detach().cpu(),
            "singularity_residual": singularities.residual.detach().cpu(),
            "loss_history": self.loss_history,
            "singularity_summary": self.last_singularity_summary,
            "curvature_guidance_power": self.curvature_guidance_power.detach().cpu(),
            "curvature_guidance_weights": self.curvature_guidance_weights.detach().cpu(),
            "training_guidance_power": self.guidance_power.detach().cpu(),
            "training_guidance_weights": self.guidance_weights.detach().cpu(),
            "initial_guidance_power": self.initial_guidance_power.detach().cpu(),
            "initial_guidance_weights": self.initial_guidance_weights.detach().cpu(),
        }
        payload.update(self.kernel_freeze_metadata())
        if self.quad_guidance_power is not None:
            payload["quad_reference_power"] = self.quad_guidance_power.detach().cpu()
            payload["quad_reference_confidence"] = self.quad_guidance_weights.detach().cpu()
        if self.connection_reference is not None:
            payload["connection_reference_power"] = (
                self.connection_reference.power_field.detach().cpu()
            )
            payload["connection_reference_singularity_index"] = (
                self.connection_reference.singularity_index.detach().cpu()
            )
            payload["connection_reference_summary"] = {
                key: value
                for key, value in self.connection_reference._asdict().items()
                if not isinstance(value, Tensor)
            }
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(payload, filename)
        if self.cfg.save_field_visualization:
            visualization_dir = Path(filename).parent / "visualizations"
            try:
                paths = self.export_field_visualization(
                    payload, visualization_dir
                )
                hktex.info(
                    "Field visualization saved: "
                    f"{paths['overview_png']}"
                )
            except Exception as exc:
                # A figure should never destroy a completed training run or
                # prevent its checkpoint/dense field from being recovered.
                hktex.warn(f"Field visualization failed: {exc}")
        return filename

    @torch.no_grad()
    def export_field_visualization(self, dense_payload, output_dir):
        """Save paper-ready kernel/field figures and their source arrays."""
        from hktex.utils.quad_field_visualization import (
            save_quad_field_visualizations,
        )

        diffusion_times = self.model.diff_times
        if diffusion_times.ndim == 0:
            diffusion_times = diffusion_times.expand(self.model.N_sources)
        kernel_data = {
            "locations": self.model.kernel_locations.detach().cpu(),
            "face_ids": self.model.kernel_face_ids.detach().cpu(),
            "cross_angles": self.model.cross_angles.detach().cpu(),
            "cross_amplitudes": self.model.cross_amplitudes.detach().cpu(),
            "target_sizes": self.model.target_sizes.detach().cpu(),
            "support_angles": self.model.support_angles.detach().cpu(),
            "anisotropies": self.model.anisotropies.detach().cpu(),
            "thresholds": self.model.thresholds.detach().cpu(),
            "sharpnesses": self.model.sharpnesses.detach().cpu(),
            "diffusion_times": diffusion_times.detach().cpu(),
        }
        return save_quad_field_visualizations(
            dense_payload,
            kernel_data,
            output_dir,
            dense_glyph_count=self.cfg.visualization_dense_glyph_count,
            kernel_glyph_count=self.cfg.visualization_kernel_glyph_count,
            dpi=self.cfg.visualization_dpi,
            elevation=self.cfg.visualization_elevation,
            azimuth=self.cfg.visualization_azimuth,
        )
