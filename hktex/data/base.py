"""Triangle mesh input for HKTex2Quad; no appearance observations are loaded."""
from dataclasses import dataclass
import hktex
from hktex.utils import load_mesh


@dataclass
class MeshSamplerDataConfig:
    mesh_path: str = ""
    preserve_topology: bool = True


@hktex.register("data.mesh")
class MeshSamplerDataModule:
    def __init__(self, cfg):
        self.cfg = cfg

    def prepare_data(self):
        self.mesh = load_mesh(self.cfg.mesh_path, preserve_topology=getattr(self.cfg, "preserve_topology", True))

    def setup(self, stage=None):
        if not hasattr(self, "mesh"):
            raise RuntimeError("Call prepare_data before setup")
