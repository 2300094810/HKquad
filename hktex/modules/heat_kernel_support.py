"""Geometry-only heat-kernel supports shared by the quad direction/size fields."""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import hktex
import hktex.utils as utils
from hktex.utils import BaseModule
from hktex.utils.typing import *
from .mesh import Mesh
from .eigen_albo_knn import EigenAlboInterpolationKNN
from .utils import PointsInfoKNN, KernelInfo

__all__ = ["HeatKernelSupport", "HeatKernelSupportKNN"]


class HeatKernelSupport(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        n_sources: int = 128
        diff_time: float = 1e-2
        mass_type: str = "kde"
        init_min_threshold: float = 0.3
        range_enforcement_type: str = "activations"
        init_kernel_edge_type: str = "uniform"
        power_diffused_diracs: int = 1

    def configure(self, mesh: Mesh, **kwargs):
        super().configure()
        # Keep geometry buffers in PT checkpoints for legacy load compatibility.
        self._mesh = mesh
        self.N_sources = self.cfg.n_sources
        if self.N_sources <= 0:
            raise ValueError("n_sources must be positive")
        self._angle_scale = torch.pi
        if self.cfg.range_enforcement_type == "activations":
            self._thresholds_act = lambda x: 0.3 + (0.7 - 1e-8) * torch.sigmoid(x + 0.916)
            self._angle_act = lambda x: self._angle_scale * torch.sigmoid(x)
            self._anis_act = lambda x: 1.0 + 99.0 * torch.sigmoid(x)
            self._sharpness_act = lambda x: 10.0 + 190.0 * torch.sigmoid(x)
        elif self.cfg.range_enforcement_type == "pgd":
            self._thresholds_act = self._angle_act = self._anis_act = self._sharpness_act = lambda x: x
        else:
            raise ValueError(f"Unknown range enforcement type: {self.cfg.range_enforcement_type}")
        self.kernel_filter_func = utils.rescaled_soft_step
        self._make_splats(mesh)

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        for name, buffer in super().named_buffers(prefix, recurse, remove_duplicate):
            if not name.startswith("_mesh."):
                yield name, buffer

    def _make_splats(self, mesh: Mesh):
        factory_kwargs = {"dtype": torch.float, "device": self.device}
        if self.cfg.range_enforcement_type == "activations":
            # Initialise angles for uniform output in [0, 蟺/2] considering activation
            p = torch.rand(self.N_sources, **factory_kwargs)
            angles = torch.log(p / (1 - p + 1e-7))

            epsilon = 1e-8
            # Initialize Sharpnesses for uniform output in [10.0, 50.0] considering activation
            power = 0.5 if self.cfg.init_kernel_edge_type == "high_skewed" else 1.0
            random_power = torch.rand(self.N_sources, **factory_kwargs) ** power
            uniform_sharpnesses = 10.0 + 190.0 * random_power
            p_sharp = (uniform_sharpnesses - 10.0) / (200.0 - 10.0)
            sharpnesses = torch.log(p_sharp / (1 - p_sharp + epsilon))

            # Initialize Thresholds for uniform output in [min_thresh, 0.9999] considering activation
            power = 0.7 if self.cfg.init_kernel_edge_type == "high_skewed" else 1.0
            random_power = torch.rand(self.N_sources, **factory_kwargs) ** power
            min_thresh = self.cfg.init_min_threshold
            uniform_thresholds = min_thresh + (1.0 - min_thresh - 1e-8) * random_power
            p_thresh = (uniform_thresholds - 0.3) / (0.7 - 1e-8)
            thresholds = torch.log(p_thresh / (1 - p_thresh + epsilon)) - 0.916

            # Initialize Anisotropies for uniform output in a chosen range [1, 100]
            uniform_anisotropies = 1.0 + 99.0 * torch.rand(
                self.N_sources, **factory_kwargs
            )
            p_anisotropy = (uniform_anisotropies - 1.0) / 99.0
            anisotropies = torch.log(p_anisotropy / (1 - p_anisotropy + epsilon))

        elif self.cfg.range_enforcement_type == "pgd":
            angles = torch.rand(self.N_sources, **factory_kwargs) * self._angle_scale

            power = 0.5 if self.cfg.init_kernel_edge_type == "high_skewed" else 1.0
            random_power_sharp = torch.rand(self.N_sources, **factory_kwargs) ** power
            sharpnesses = 10.0 + 40.0 * random_power_sharp

            power = 0.7 if self.cfg.init_kernel_edge_type == "high_skewed" else 1.0
            random_power_thresh = torch.rand(self.N_sources, **factory_kwargs) ** power
            min_thresh = self.cfg.init_min_threshold
            thresholds = min_thresh + (1.0 - min_thresh - 1e-8) * random_power_thresh

            anisotropies = 1.0 + 99.0 * torch.rand(self.N_sources, **factory_kwargs)

        else:
            # This case is handled in the configure method. No need to raise an error
            pass

        # Uniformly sample many points on the mesh surface
        # and then use farthest point sampling to select the kernel locations
        fids, bary = utils.uniform_sampling(
            mesh.verts, mesh.faces, max(3 * self.N_sources, 10_000)
        )
        pos = mesh.barycentric_to_cartesian(bary, mesh.get_face_vertices(fids))
        mask = utils.farthest_point_sampling(pos, self.N_sources)
        kernel_locations = pos[mask]
        kernel_face_ids = fids[mask]

        self._angles = torch.nn.Parameter(angles)
        self._anisotropies = torch.nn.Parameter(anisotropies)
        self._thresholds = torch.nn.Parameter(thresholds)
        self._sharpnesses = torch.nn.Parameter(sharpnesses)
        self._kernel_locations = nn.Parameter(kernel_locations)
        self._kernel_face_ids = nn.Buffer(kernel_face_ids, persistent=True)


    @property
    def angles(self) -> Float[Tensor, "G"]:
        return self._angle_act(self._angles)

    @property
    def anisotropies(self) -> Float[Tensor, "G"]:
        return self._anis_act(self._anisotropies)

    @property
    def diff_times(self) -> Float[Tensor, "G"]:
        return self.cfg.diff_time * torch.ones(self.N_sources, device=self.device)

    @property
    def thresholds(self) -> Float[Tensor, "G"]:
        return self._thresholds_act(self._thresholds)

    @property
    def sharpnesses(self) -> Float[Tensor, "G"]:
        return self._sharpness_act(self._sharpnesses)

    @property
    def kernel_locations(self) -> Float[Tensor, "G 3"]:
        return self._kernel_locations

    @property
    def kernel_face_ids(self) -> Int[Tensor, "G"]:
        return self._kernel_face_ids

    def post_optimizer_step(self):
        if self.cfg.range_enforcement_type == "pgd":
            self.clamp_parameters()

    def save_barycentric_locations(self, barycentric_coords):
        setattr(
            self._kernel_locations,
            "bary_coords",
            barycentric_coords.detach(),
        )

    def save_torch(self, filename, compressed: bool = True):
        if compressed:
            state_dict = {
                k: v.half() if v.is_floating_point() else v
                for k, v in self.state_dict().items()
            }
        else:
            state_dict = self.state_dict()

        if "_kernel_locations" in state_dict and "_kernel_face_ids" in state_dict:
            bary_coords = self._mesh.cartesian_to_barycentric(
                state_dict["_kernel_locations"],
                self._mesh.get_face_vertices(state_dict["_kernel_face_ids"]),
            )

            state_dict["_kernel_bary_coords"] = bary_coords[:, :2]
            state_dict.pop("_kernel_locations")

            if state_dict["_kernel_face_ids"].max().item() < 32_767:
                state_dict["_kernel_face_ids"] = state_dict["_kernel_face_ids"].to(
                    torch.int16
                )
            else:
                state_dict["_kernel_face_ids"] = state_dict["_kernel_face_ids"].to(
                    torch.int32
                )

        drop_keys = ["_error_accumulator", "_hit_accumulator", "_contrib_accumulator"]
        for k in list(state_dict.keys()):
            if k in drop_keys:
                state_dict.pop(k)

        torch.save(state_dict, filename)

    def save_numpy_npz(self, filename, compressed: bool = True):
        np_dict = {}
        for k, v in self.state_dict().items():
            if "__mesh" in k or k.startswith("_mesh."):
                continue
            if compressed and v.is_floating_point():
                np_dict[k] = v.half().detach().cpu().numpy()
            else:
                np_dict[k] = v.detach().cpu().numpy()

        if "_kernel_locations" in np_dict and "_kernel_face_ids" in np_dict:
            face_ids = torch.as_tensor(
                np_dict["_kernel_face_ids"],
                dtype=torch.long,
                device=self._mesh.faces.device,
            )
            face_vertices = self._mesh.get_face_vertices(face_ids)
            bary_coords = self._mesh.cartesian_to_barycentric(
                torch.as_tensor(
                    np_dict["_kernel_locations"],
                    dtype=self._mesh.verts.dtype,
                    device=face_vertices.device,
                ),
                face_vertices,
            )

            np_dict["_kernel_bary_coords"] = bary_coords[:, :2].cpu().numpy()
            np_dict.pop("_kernel_locations")

            if np_dict["_kernel_face_ids"].max().item() < 32767:
                np_dict["_kernel_face_ids"] = np_dict["_kernel_face_ids"].astype(
                    np.int16
                )
            else:
                np_dict["_kernel_face_ids"] = np_dict["_kernel_face_ids"].astype(
                    np.int32
                )

        drop_keys = ["_error_accumulator", "_hit_accumulator", "_contrib_accumulator"]
        for k in list(np_dict.keys()):
            if k in drop_keys:
                np_dict.pop(k)

        np.savez_compressed(filename, **np_dict)

    def load_torch(self, filename: str):
        sd = torch.load(filename, map_location=self.device, weights_only=True)
        self._process_load(sd)

    def load_numpy_npz(self, filename: str):
        if filename.endswith(".pt"):
            filename = filename.replace(".pt", ".npz")

        np_dict = np.load(filename, allow_pickle=False)
        sd = {}
        for k, v in np_dict.items():
            sd[k] = torch.tensor(v, device=self.device)
        self._process_load(sd)


class HeatKernelSupportKNN(HeatKernelSupport):
    @dataclass
    class Config(HeatKernelSupport.Config):
        knn_outer_k: int = 100
        knn_inner_k: Optional[int] = 10


    @property
    def diff_times(self) -> Float[Tensor, "G"]:
        return torch.scalar_tensor(self.cfg.diff_time, device=self.device)

    def reset(
        self,
        eigalbo_knn: EigenAlboInterpolationKNN,
    ):
        eigalbo_knn.reset()

    def prepare_kernels(
        self,
        mesh: Mesh,
        eigalbo_knn: EigenAlboInterpolationKNN,
        save_barycentric: bool = True,
    ) -> KernelInfo:
        kernel_vert_idx = mesh.get_face_vertices(self.kernel_face_ids)
        kernel_barycentric_coords = mesh.cartesian_to_barycentric(
            self.kernel_locations, kernel_vert_idx
        )

        if save_barycentric:
            self.save_barycentric_locations(kernel_barycentric_coords)

        with torch.profiler.record_function("build_kernel_graph"):
            eigalbo_knn.build_kernel_graph(
                kernel_bary=kernel_barycentric_coords,
                kernel_vert_idx=kernel_vert_idx,
                kernel_angles=self.angles,
                kernel_scales=self.anisotropies,
                diffusion_time=self.diff_times,
            )
        return KernelInfo(
            vert_idx=kernel_vert_idx,
            barycentric_coords=kernel_barycentric_coords,
            albo_evecs=None,
            mass=None,
        )

    def prepare_points(
        self,
        mesh: Mesh,
        eigalbo_interp: EigenAlboInterpolationKNN,
        face_ids: Float[Tensor, "P"],
        barys: Float[Tensor, "P 3"] | None = None,
        pts: Float[Tensor, "P 3"] | None = None,
    ) -> PointsInfoKNN:
        pts_tri_vert_idx = mesh.get_face_vertices(face_ids)  # [P, 3]

        if barys is None and pts is not None:
            barys = mesh.cartesian_to_barycentric(pts, pts_tri_vert_idx)
        elif barys is not None and pts is None:
            pass
        else:
            raise ValueError(
                "Either barys or pts must be provided to prepare points for diffusion"
            )

        k_search = min(self.cfg.knn_outer_k, self.N_sources)
        with torch.profiler.record_function("query_points"):
            query_points = eigalbo_interp.query_points(
                barys, pts_tri_vert_idx, k_search
            )

        return PointsInfoKNN(
            albo_evals=query_points["evals"],
            albo_evecs=query_points["evecs"],
            iso_evecs=None,
            mass=None,
            weights=query_points["weights"],
            distances=query_points["distances"],
            indices=query_points["indices"],
        )

    def evaluate_kernel_weights(
        self,
        eigalbo_interp: EigenAlboInterpolationKNN,
        pts_info: PointsInfoKNN,
    ) -> tuple[Float[Tensor, "P K"], Int64[Tensor, "P K"]]:
        """Evaluate filtered heat support without attaching output attributes.

        Direction and target-size reconstruction share the same spatial support.
        """
        pts_evecs: Float[Tensor, "P K E"] = pts_info["albo_evecs"]
        pts_evals: Float[Tensor, "P K E"] = pts_info["albo_evals"]
        pts_weights: Float[Tensor, "P K"] | None = pts_info["weights"]
        pts_indices: Int64[Tensor, "P K"] = pts_info["indices"]

        with torch.profiler.record_function("diffuse_heat"):
            heat_qk, heat_qk_norm = eigalbo_interp.diffuse_heat(
                pts_evals, pts_evecs, pts_indices, weights=pts_weights
            )
        diffused_diracs: Float[Tensor, "P K"] = (
            heat_qk_norm if heat_qk_norm is not None else heat_qk
        )

        if self.cfg.power_diffused_diracs != 1:
            diffused_diracs = diffused_diracs**self.cfg.power_diffused_diracs

        with torch.profiler.record_function("kernel_filter"):
            filtered: Float[Tensor, "P K"] = self.kernel_filter_func(
                diffused_diracs,
                epsilon=self.thresholds[pts_indices],
                sharpness=self.sharpnesses[pts_indices],
            )
        if not torch.isfinite(filtered).all():
            hktex.warn("Non-finite filtered heat weights detected; replacing with zero.")
            filtered = torch.nan_to_num(
                filtered, nan=0.0, posinf=0.0, neginf=0.0
            )
        return filtered, pts_indices
