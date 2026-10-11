"""Construction and checkpoint saving shared by field trainers."""
from dataclasses import dataclass, field
import os
import hktex
from hktex.modules import Mesh
from hktex.utils import BaseObject
from .utils import parse_optimizers_and_schedulers


class BaseTrainer(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        tracer_type: str = "modules.gpu-geodesic-tracer"
        tracer: dict = field(default_factory=dict)
        eigen_albo_type: str = "modules.eigen-albo-interpolation-knn"
        eigen_albo: dict = field(default_factory=dict)
        model_type: str = "modules.heat-kernel-cross-field-knn"
        model: dict = field(default_factory=dict)
        optimizers: list = field(default_factory=list)
        # Accepted only for old HKTex2Quad saved configurations.
        use_knn_implementation: bool = True
        density_controllers: list = field(default_factory=list)

    def configure(self, datamodule, **kwargs):
        super().configure()
        if self.cfg.density_controllers:
            raise ValueError("HKTex2Quad keeps all kernels; density_controllers must be empty")
        self.datamodule = datamodule
        self.mesh = Mesh.from_trimesh(datamodule.mesh, device=self.device)
        model_config = dict(self.cfg.model)
        # Old quad presets inherited inert texture settings. Keep them loadable.
        for name in ("kernel_dim", "out_net", "out_dim", "normalize_colours", "allow_negative_colours"):
            model_config.pop(name, None)
        self.model = hktex.find(self.cfg.model_type)(model_config, self.mesh)
        self.eigalbo_interp = hktex.find(self.cfg.eigen_albo_type)(self.cfg.eigen_albo, self.mesh)
        self.tracer = hktex.find(self.cfg.tracer_type)(self.cfg.tracer, self.mesh)
        self.optimizers, self.schedulers = parse_optimizers_and_schedulers(self.cfg.optimizers, self)

    def save_model(self, filename):
        self.model.save_torch(filename)
        torch_size = os.path.getsize(filename)

        npz_filename = filename.replace(".pt", ".npz")
        self.model.save_numpy_npz(npz_filename)
        npz_size = os.path.getsize(npz_filename)
        return torch_size / 1024, npz_size / 1024
