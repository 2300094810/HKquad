"""KNN heat-kernel primitives carrying a sparse N-RoSy field.

This HKTex2Quad bridge reuses HKTex spatial supports while keeping support
orientation and cross orientation as independent parameters.  Callers can
provide source-to-query transport explicitly or enable a deterministic face
dual-graph reference implementation for small-mesh validation.
"""

from dataclasses import dataclass
import math

from termcolor import colored
import torch
import torch.nn as nn

import hktex
from hktex.utils.typing import *

from .eigen_albo_knn import EigenAlboInterpolationKNN
from .heat_kernel_support import HeatKernelSupportKNN
from .rosy_field import (
    RosyFieldOutput,
    aggregate_kernel_rosy_field,
    decode_rosy_angle,
    normalize_power_field,
    rotate_power_field,
)
from .surface_connection import (
    FaceConnectionPathTransport,
    face_tangent_frames,
    transport_angle_between_frames,
)
from .utils import PointsInfoKNN

__all__ = ["HeatKernelCrossFieldKNN"]


@hktex.register("modules.heat-kernel-cross-field-knn")
class HeatKernelCrossFieldKNN(HeatKernelSupportKNN):
    """Sparse heat-kernel representation of an N-RoSy field and size field."""

    @dataclass
    class Config(HeatKernelSupportKNN.Config):
        rosy_degree: int = 4
        singularity_epsilon: float = 1e-8
        init_target_size: float = 1.0
        min_target_size: float = 1e-4
        max_target_size: float = 1e4
        transport_mode: str = "gpu-local"  # "gpu-local" | "explicit" | "dual-path"
        allow_untransported: bool = False
        # Freeze at each kernel's own initialization, retaining all kernels.
        fixed_kernel_sharpness: bool = False
        fixed_cross_amplitude: bool = False
        fixed_kernel_anisotropy: bool = False
        fixed_support_angle: bool = False
        fixed_kernel_threshold: bool = False

    cfg: Config

    _cross_coefficients: Float[Tensor, "G 2"]
    _target_log_sizes: Float[Tensor, "G"]
    _path_transport: Optional[FaceConnectionPathTransport]
    _transport_tangent_u: Float[Tensor, "F 3"]
    _transport_tangent_v: Float[Tensor, "F 3"]
    _transport_normal: Float[Tensor, "F 3"]

    def configure(self, mesh, **kwargs):
        if self.cfg.rosy_degree < 1:
            raise ValueError("rosy_degree must be positive")
        if self.cfg.init_target_size <= 0:
            raise ValueError("init_target_size must be positive")
        if not 0 < self.cfg.min_target_size <= self.cfg.max_target_size:
            raise ValueError("target size bounds must be positive and ordered")
        if self.cfg.transport_mode not in ("gpu-local", "explicit", "dual-path"):
            raise ValueError(
                "transport_mode must be 'gpu-local', 'explicit', or 'dual-path'"
            )
        super().configure(mesh, **kwargs)
        self.apply_parameter_freeze()
        self._path_transport = None
        if self.cfg.transport_mode == "dual-path":
            self._path_transport = FaceConnectionPathTransport(
                mesh.verts,
                mesh.faces,
                face_normals=mesh.fnorms,
            )
        # These gauges are also needed to re-express a cross coefficient when
        # a geodesic optimiser moves its kernel centre into another face.
        transport_frames = face_tangent_frames(
            mesh.verts,
            mesh.faces,
            face_normals=mesh.fnorms,
        )
        tangent_u = transport_frames.tangent_u
        tangent_v = transport_frames.tangent_v
        normal = transport_frames.normal
        self._transport_tangent_u = nn.Buffer(
            tangent_u, persistent=False
        )
        self._transport_tangent_v = nn.Buffer(
            tangent_v, persistent=False
        )
        self._transport_normal = nn.Buffer(
            normal, persistent=False
        )

    def _make_splats(self, mesh):
        super()._make_splats(mesh)

        phase = 2.0 * torch.pi * torch.rand(
            self.N_sources, device=self.device, dtype=self._kernel_locations.dtype
        )
        coefficients = torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)
        target_log_sizes = torch.full(
            (self.N_sources,),
            math.log(self.cfg.init_target_size),
            device=self.device,
            dtype=self._kernel_locations.dtype,
        )

        self._cross_coefficients = nn.Parameter(coefficients)
        self._target_log_sizes = nn.Parameter(target_log_sizes)
        self.splat_param_keys = [
            "cross_amplitudes",
            "cross_angles",
            "target_sizes",
            "support_angles",
            "anisotropies",
            "sharpnesses",
            "thresholds",
        ]

    _FREEZE_PARAMETERS = {
        "fixed_kernel_sharpness": "_sharpnesses",
        "fixed_kernel_anisotropy": "_anisotropies",
        "fixed_support_angle": "_angles",
        "fixed_kernel_threshold": "_thresholds",
    }

    def apply_parameter_freeze(self):
        """Exclude frozen support attributes from Adam and gradient clipping."""
        for flag, name in self._FREEZE_PARAMETERS.items():
            if getattr(self.cfg, flag, False):
                parameter = getattr(self, name)
                parameter.requires_grad_(False)
                parameter.grad = None
        if self.cfg.fixed_cross_amplitude:
            self.capture_fixed_cross_amplitudes()

    @torch.no_grad()
    def capture_fixed_cross_amplitudes(self):
        """Capture after geometry initialization, without freezing direction.

        Only enabled runs add persistent buffers; old/default checkpoints retain
        their original state schema.  The reference phase is a finite fallback
        if an optimizer ever drives a latent coefficient exactly to zero.
        """
        coefficients = self._cross_coefficients.detach()
        amplitude = torch.linalg.vector_norm(coefficients, dim=-1)
        eps = self.cfg.singularity_epsilon
        phase = coefficients / amplitude.unsqueeze(-1).clamp_min(eps)
        fallback = torch.zeros_like(phase)
        fallback[:, 0] = 1.0
        phase = torch.where((amplitude > eps).unsqueeze(-1), phase, fallback)
        for name, value in (
            ("_fixed_cross_amplitudes", amplitude),
            ("_fixed_cross_reference_phases", phase),
        ):
            if name in self._buffers:
                self._buffers[name] = value.clone()
            else:
                self.register_buffer(name, value.clone(), persistent=True)

    @torch.no_grad()
    def kernel_parameter_snapshot(self, *, raw_cross=False) -> dict[str, Tensor]:
        """Physical per-kernel values for reproducible ablation verification."""
        coefficients = (
            self._cross_coefficients if raw_cross else self.cross_coefficients
        )
        return {
            "locations": self.kernel_locations.detach().cpu().clone(),
            "face_ids": self.kernel_face_ids.detach().cpu().clone(),
            "cross_coefficients": coefficients.detach().cpu().clone(),
            "cross_amplitudes": torch.linalg.vector_norm(
                coefficients, dim=-1
            ).detach().cpu().clone(),
            "cross_angles": decode_rosy_angle(
                coefficients, degree=self.cfg.rosy_degree
            ).detach().cpu().clone(),
            "target_log_sizes": self._target_log_sizes.detach().cpu().clone(),
            "target_sizes": self.target_sizes.detach().cpu().clone(),
            "support_angles": self.support_angles.detach().cpu().clone(),
            "anisotropies": self.anisotropies.detach().cpu().clone(),
            "thresholds": self.thresholds.detach().cpu().clone(),
            "sharpnesses": self.sharpnesses.detach().cpu().clone(),
        }

    @torch.no_grad()
    def raw_kernel_parameter_snapshot(self) -> dict[str, Tensor]:
        """Original trainable state, excluding freeze metadata and geometry."""
        names = (
            "_kernel_locations", "_kernel_face_ids", "_angles", "_anisotropies",
            "_thresholds", "_sharpnesses", "_cross_coefficients",
            "_target_log_sizes",
        )
        return {
            name: getattr(self, name).detach().cpu().clone() for name in names
        }

    @property
    def support_angles(self) -> Float[Tensor, "G"]:
        """Orientation of the anisotropic heat support, not the cross field."""
        return self.angles

    @property
    def cross_coefficients(self) -> Float[Tensor, "G 2"]:
        """Real-imaginary coefficients, with optionally fixed initial rho.

        Normalize in the forward pass so angular gradients remain available
        while no objective can change the effective per-kernel amplitude.
        """
        if not self.cfg.fixed_cross_amplitude:
            return self._cross_coefficients
        magnitude = torch.linalg.vector_norm(
            self._cross_coefficients, dim=-1, keepdim=True
        )
        phase = self._cross_coefficients / magnitude.clamp_min(
            self.cfg.singularity_epsilon
        )
        phase = torch.where(
            magnitude > self.cfg.singularity_epsilon,
            phase,
            self._fixed_cross_reference_phases,
        )
        return phase * self._fixed_cross_amplitudes.unsqueeze(-1)

    @property
    def cross_phases(self) -> Float[Tensor, "G 2"]:
        phase, _, _ = normalize_power_field(
            self.cross_coefficients, eps=self.cfg.singularity_epsilon
        )
        return phase

    @property
    def cross_amplitudes(self) -> Float[Tensor, "G"]:
        return torch.linalg.vector_norm(self.cross_coefficients, dim=-1)

    @property
    def cross_angles(self) -> Float[Tensor, "G"]:
        return decode_rosy_angle(self.cross_phases, degree=self.cfg.rosy_degree)

    @property
    def target_sizes(self) -> Float[Tensor, "G"]:
        min_log_size = math.log(self.cfg.min_target_size)
        max_log_size = math.log(self.cfg.max_target_size)
        return torch.exp(self._target_log_sizes.clamp(min_log_size, max_log_size))

    def _gpu_local_transport_angles(
        self,
        source_face_ids: Int64[Tensor, "P k"],
        target_face_ids: Int64[Tensor, "P 1"],
    ) -> Float[Tensor, "P k"]:
        """Vectorised local frame transport with no CPU work in the hot path."""
        source_face_ids = source_face_ids.long()
        target_face_ids = target_face_ids.long()
        return transport_angle_between_frames(
            self._transport_tangent_u[source_face_ids],
            self._transport_tangent_v[source_face_ids],
            self._transport_normal[source_face_ids],
            self._transport_tangent_u[target_face_ids],
            self._transport_tangent_v[target_face_ids],
            self._transport_normal[target_face_ids],
            antiparallel_mode="source-u",
        )

    @torch.no_grad()
    def capture_kernel_gauge(self) -> Int64[Tensor, "G"]:
        """Snapshot source-face gauges before movable kernel optimisation."""
        return self.kernel_face_ids.detach().clone().long()

    @torch.no_grad()
    def restore_kernel_gauge(
        self,
        previous_face_ids: Int64[Tensor, "G"],
    ) -> int:
        """Preserve physical crosses after kernel centres cross face boundaries.

        The stored coefficient is expressed in its source face's deterministic
        tangent frame.  A geodesic position update can change that frame even
        if the physical cross should not rotate.  This method parallel
        transports and re-expresses only coefficients whose source face
        changed, returning the number of updated kernels.

        ``dual-path`` uses its reference path solver.  The scalable modes use
        the same vectorised minimal-normal rotation as ``gpu-local``; this is
        accurate for the small local steps expected from GeodesicOpt and keeps
        the optimiser hot path on the GPU.
        """
        if previous_face_ids.shape != self.kernel_face_ids.shape:
            raise ValueError("previous_face_ids must match kernel_face_ids")
        previous_face_ids = previous_face_ids.to(
            device=self.kernel_face_ids.device, dtype=torch.long
        )
        current_face_ids = self.kernel_face_ids.long()
        changed = previous_face_ids != current_face_ids
        count = int(changed.sum().item())
        if count == 0:
            return 0

        source_faces = previous_face_ids[changed]
        target_faces = current_face_ids[changed]
        if self.cfg.transport_mode == "dual-path":
            transport_angles = self._path_transport.angles(
                source_faces, target_faces
            ).to(
                device=self._cross_coefficients.device,
                dtype=self._cross_coefficients.dtype,
            )
        else:
            transport_angles = self._gpu_local_transport_angles(
                source_faces.unsqueeze(-1), target_faces.unsqueeze(-1)
            ).squeeze(-1)

        transported = rotate_power_field(
            self._cross_coefficients[changed],
            transport_angles,
            degree=self.cfg.rosy_degree,
        )
        self._cross_coefficients[changed] = transported
        if self.cfg.fixed_cross_amplitude:
            self._fixed_cross_reference_phases[changed] = rotate_power_field(
                self._fixed_cross_reference_phases[changed],
                transport_angles,
                degree=self.cfg.rosy_degree,
            )
        return count

    def prepare_points(
        self,
        mesh,
        eigalbo_interp: EigenAlboInterpolationKNN,
        face_ids: Int64[Tensor, "P"],
        barys: Optional[Float[Tensor, "P 3"]] = None,
        pts: Optional[Float[Tensor, "P 3"]] = None,
    ) -> PointsInfoKNN:
        """Prepare heat queries and retain face ids for automatic transport."""
        points_info = super().prepare_points(
            mesh,
            eigalbo_interp,
            face_ids,
            barys=barys,
            pts=pts,
        )
        points_info["face_ids"] = face_ids.long()
        return points_info

    def diffuse_cross_field(
        self,
        eigalbo_interp: EigenAlboInterpolationKNN,
        pts_info: PointsInfoKNN,
        *,
        transport_angles: Optional[Float[Tensor, "P K"]] = None,
        query_face_ids: Optional[Int64[Tensor, "P"]] = None,
    ) -> tuple[
        RosyFieldOutput,
        None,
        Int64[Tensor, "k P 1"],
        Float[Tensor, "k P 1"],
    ]:
        """Evaluate the sparse cross and target-size fields at query points.

        Explicit ``transport_angles`` must align with the outer KNN entries in
        ``pts_info``.  In the default ``gpu-local`` mode they can be omitted
        and ``query_face_ids`` drives a fully vectorised source/query frame
        transport for the selected inner K kernels.  ``dual-path`` instead
        uses the slow reference path solver.  Omitting transport entirely
        requires the explicit ``allow_untransported`` smoke-test flag.
        """
        filtered, outer_global_indices = self.evaluate_kernel_weights(
            eigalbo_interp=eigalbo_interp,
            pts_info=pts_info,
        )

        _, outer_k = filtered.shape
        inner_k = self.cfg.knn_inner_k
        if inner_k is None:
            inner_k = outer_k
        inner_k = min(inner_k, outer_k)

        top_weights, top_local_indices = filtered.topk(
            k=inner_k, largest=True, dim=1
        )
        top_global_indices = torch.gather(
            outer_global_indices, dim=1, index=top_local_indices
        )

        top_transport_angles = None
        if transport_angles is not None:
            if transport_angles.shape != filtered.shape:
                raise ValueError(
                    "transport_angles must match the outer KNN heat weights [P, K]"
                )
            top_transport_angles = torch.gather(
                transport_angles, dim=1, index=top_local_indices
            )
        elif self.cfg.transport_mode in ("gpu-local", "dual-path"):
            if query_face_ids is None:
                query_face_ids = pts_info.get("face_ids")
            if query_face_ids is None:
                raise ValueError("query_face_ids is required for automatic transport")
            if (
                query_face_ids.ndim != 1
                or query_face_ids.shape[0] != filtered.shape[0]
            ):
                raise ValueError("query_face_ids must have shape [P]")
            source_face_ids = self.kernel_face_ids[top_global_indices].long()
            target_face_ids = query_face_ids.long().unsqueeze(-1)
            if self.cfg.transport_mode == "gpu-local":
                top_transport_angles = self._gpu_local_transport_angles(
                    source_face_ids, target_face_ids
                )
            else:
                top_transport_angles = self._path_transport.angles(
                    source_face_ids, target_face_ids
                ).to(device=filtered.device, dtype=filtered.dtype)
        elif not self.cfg.allow_untransported:
            raise ValueError(
                "transport_angles is required in explicit mode; set "
                "allow_untransported=True only for a planar/common-frame smoke test"
            )

        output = aggregate_kernel_rosy_field(
            top_weights,
            self.cross_coefficients,
            kernel_indices=top_global_indices,
            transport_angles=top_transport_angles,
            kernel_log_sizes=self._target_log_sizes,
            degree=self.cfg.rosy_degree,
            eps=self.cfg.singularity_epsilon,
        )

        topk_kernel_idxs = top_global_indices.transpose(0, 1).unsqueeze(-1)
        topk_kernel_contribs = top_weights.transpose(0, 1).unsqueeze(-1)
        return output, None, topk_kernel_idxs, topk_kernel_contribs


    @torch.no_grad()
    def post_optimizer_step(self):
        super().post_optimizer_step()
        if self.cfg.fixed_cross_amplitude:
            # Keep saved latent coefficients at the prescribed norm too.
            # Adam still learns their angle through the normalized forward.
            self._cross_coefficients.copy_(self.cross_coefficients)

    @torch.no_grad()
    def clamp_parameters(self):
        for parameter, minimum, maximum in (
            (self._sharpnesses, 10.0, 200.0),
            (self._thresholds, 0.3, 1.0 - 1e-8),
            (self._angles, 0.0, self._angle_scale),
            (self._anisotropies, 1.0, 200.0),
            (self._target_log_sizes,
             math.log(self.cfg.min_target_size), math.log(self.cfg.max_target_size)),
        ):
            if parameter.requires_grad:
                parameter.clamp_(min=minimum, max=maximum)

    @property
    def colored_print_opt_params(self):
        support_angles = torch.rad2deg(self.support_angles).detach().cpu().numpy()
        cross_angles = torch.rad2deg(self.cross_angles).detach().cpu().numpy()
        return (
            colored(f"Support angles: {support_angles}, ", "yellow")
            + colored(f"Cross angles: {cross_angles}, ", "magenta")
            + colored(
                f"Cross amplitudes: {self.cross_amplitudes.detach().cpu().numpy()}, ",
                "red",
            )
            + colored(
                f"Target sizes: {self.target_sizes.detach().cpu().numpy()}, ", "white"
            )
            + colored(f"Anisotropies: {self.anisotropies}", "green")
        )

    def _process_load(self, sd: Dict[str, Union[Tensor, Any]]):
        n_sources_in_ckpt = sd["_cross_coefficients"].shape[0]
        if n_sources_in_ckpt != self.cfg.n_sources:
            self.cfg.n_sources = n_sources_in_ckpt
            mesh = self._mesh
            self.configure(mesh)

        if "_kernel_bary_coords" in sd and "_kernel_face_ids" in sd:
            mesh = self._mesh
            bary_coords = sd["_kernel_bary_coords"]
            bary_coords = torch.cat(
                (bary_coords, 1.0 - bary_coords.sum(dim=1, keepdim=True)), dim=1
            )
            sd["_kernel_locations"] = mesh.barycentric_to_cartesian(
                bary_coords,
                mesh.get_face_vertices(sd["_kernel_face_ids"].long()),
            )
            sd.pop("_kernel_bary_coords")

        # PT files saved before the geometry/texture split use this mesh prefix.
        # Rename it so loading preserves the legacy geometry precision as well.
        sd = {key.replace("_HeatKernelTexture__mesh.", "_mesh."): value
              for key, value in sd.items()}
        self.load_state_dict(sd, strict=False)
        self.float()
        # A legacy checkpoint has no captured amplitude.  Freeze its loaded
        # amplitudes rather than those of the temporary random model.
        if self.cfg.fixed_cross_amplitude and "_fixed_cross_amplitudes" not in sd:
            self.capture_fixed_cross_amplitudes()
