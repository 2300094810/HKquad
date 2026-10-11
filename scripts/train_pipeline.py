"""Readable launch sequence: settings -> inputs -> field training -> quad mesh.

This module orchestrates existing algorithms. It does not alter kernel parameters
or field/quad mathematics. GPU dependencies are imported only by the child stage.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from scripts.runtime import run_logged, runtime_environment
from scripts.mesh_output_files import export_mesh_files, write_run_guide

DEFAULT_CONFIG = "configs/hktex2quad_cross_mesh_gpu.yaml"
CLI_FIELDS = {
    "shape": "run.shape", "gpu": "run.gpu", "stage2": "run.stage2",
    "dataset_root": "run.dataset_root", "memory_fraction": "run.memory_fraction",
    "iters": "optim.iters", "kernels": "trainer.model.n_sources",
    "target_quads": "trainer.target_quad_count", "seed": "seed",
    "supervision": "trainer.supervision", "output_root": "exp_root_dir",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an HKTex2Quad field and export a quad mesh.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML settings (quad preset by default)")
    parser.add_argument("--shape", help="prepared model folder, e.g. dente, double_torus or fandisk")
    parser.add_argument("--gpu", help="physical GPU ID; default comes from run.gpu")
    parser.add_argument("--iters", type=int, help="training iterations")
    parser.add_argument("--kernels", type=int, help="total number of kernels")
    parser.add_argument("--target-quads", type=int, help="quad density target")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--supervision", choices=("quad", "geometry"))
    parser.add_argument("--dataset-root", help="directory containing the prepared shape folders")
    parser.add_argument("--data-dir", help="one prepared directory with triangle.obj and quad_targets.npz")
    parser.add_argument("--output-root", help="output directory within this project")
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--stage2", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stage1-only", dest="stage2", action="store_false", default=None,
                        help="train/export the field only; skip MIQ/QEx")
    parser.add_argument("--python", help="override the automatically selected Python interpreter")
    parser.add_argument("--dry-run", action="store_true", help="validate and show settings without training")
    parser.add_argument("--_worker", dest="worker", action="store_true", help=argparse.SUPPRESS)
    args, overrides = parser.parse_known_args(argv)
    for value in overrides:
        if value.startswith("-") or "=" not in value:
            parser.error(f"unrecognized argument: {value}")
    args.overrides = overrides
    return args


def project_path(value, root):
    """Mesh-side spectral caches and all outputs must stay inside HKtex."""
    path = (root / str(value)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Keep data, configs and outputs inside {root}: {path}")
    return path


def load_settings(args, root):
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver("filename"):
        OmegaConf.register_new_resolver("filename", lambda path: Path(path).stem)
    config_path = project_path(args.config, root)
    cfg = OmegaConf.load(config_path)
    # Old presets without a run section remain usable.
    defaults = {"shape": "dente", "gpu": "0", "dataset_root": "datasets/bimdf_cross_mesh",
                "stage2": True, "memory_fraction": 0.35}
    cfg.run = OmegaConf.merge(defaults, cfg.get("run", {}))
    for argument, field in CLI_FIELDS.items():
        value = getattr(args, argument)
        if value is not None:
            OmegaConf.update(cfg, field, value, merge=False)
    if args.data_dir:
        data_dir = project_path(args.data_dir, root)
        cfg.data.mesh_path = str(data_dir / "triangle.obj")
        cfg.trainer.quad_supervision_path = str(data_dir / "quad_targets.npz")
    elif args.shape is not None or args.dataset_root is not None:
        # Keep references lazy so later run.shape/run.dataset_root overrides agree.
        cfg.data.mesh_path = "${run.dataset_root}/${run.shape}/triangle.obj"
        cfg.trainer.quad_supervision_path = "${run.dataset_root}/${run.shape}/quad_targets.npz"
    # Explicit dot overrides have highest precedence, matching the old worker CLI.
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if cfg.trainer.supervision == "geometry":
        cfg.trainer.quad_supervision_path = None
    cfg.data.mesh_path = str(project_path(cfg.data.mesh_path, root))
    cfg.data.preserve_topology = True
    cfg.trainer.eigen_albo.mesh_path = cfg.data.mesh_path
    if cfg.trainer.quad_supervision_path:
        cfg.trainer.quad_supervision_path = str(project_path(cfg.trainer.quad_supervision_path, root))
    cfg.exp_root_dir = str(project_path(cfg.exp_root_dir, root))
    OmegaConf.resolve(cfg)
    validate_settings(cfg, root)
    return cfg, config_path


def backend_paths(root):
    return {name: (root / Path(os.environ.get(f"HKTEX2QUAD_{name.upper()}_EXECUTABLE",
                       root / "cpp" / f"{name}_backend" / "build" / f"hktex2quad_{name}"))).resolve()
            for name in ("miq", "qex")}


def validate_settings(cfg, root):
    for name in (cfg.run.shape, cfg.name):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(name)):
            raise ValueError("Shape and experiment names may contain letters, digits, underscores and hyphens")
    if cfg.trainer.supervision not in ("quad", "geometry"):
        raise ValueError("supervision must be quad or geometry")
    if not isinstance(cfg.run.stage2, bool):
        raise ValueError("run.stage2 must be true or false")
    if not 0 < float(cfg.run.memory_fraction) <= 1:
        raise ValueError("run.memory_fraction must be in (0, 1]")
    for name, value in (("iterations", cfg.optim.iters), ("kernels", cfg.trainer.model.n_sources),
                        ("target quads", cfg.trainer.target_quad_count)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    inputs = [Path(cfg.data.mesh_path)]
    if cfg.trainer.supervision == "quad":
        if not cfg.trainer.quad_supervision_path:
            raise ValueError("Quad supervision requires quad_supervision_path")
        inputs.append(Path(cfg.trainer.quad_supervision_path))
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared input: {path}; see README.md, BiMDF data preparation")
    if cfg.run.stage2:
        for path in backend_paths(root).values():
            if not path.is_file() or not os.access(path, os.X_OK):
                raise FileNotFoundError(f"Missing executable: {path}; use --stage1-only to train the field")


def print_settings(cfg):
    print(f"Python: {sys.executable}\nGPU: {cfg.run.gpu}\nShape: {cfg.run.shape}")
    print(f"Supervision: {cfg.trainer.supervision}\nMesh: {cfg.data.mesh_path}")
    print(f"Iterations: {cfg.optim.iters}\nKernels: {cfg.trainer.model.n_sources}")
    print(f"Target quads: {cfg.trainer.target_quad_count}\nRun MIQ/QEx: {cfg.run.stage2}", flush=True)


def stage_commands(cfg, config_path, run_dir, root):
    field = [sys.executable, str(root / "optimisation.py"), "--config", str(config_path),
             "--cuda-memory-fraction", str(cfg.run.memory_fraction)]
    paths = backend_paths(root)
    quad = [sys.executable, str(root / "scripts/run_hktex2quad_pipeline.py"),
            "--input", str(run_dir / "dense_face_field.pt"),
            "--output-dir", str(run_dir / "quad_pipeline"), "--backend", "miq", "--extractor", "qex",
            "--miq-executable", str(paths["miq"]), "--qex-executable", str(paths["qex"])]
    return field, quad


def run_pipeline(args, root):
    from omegaconf import OmegaConf
    root = root.resolve()
    cfg, source_config = load_settings(args, root)
    print_settings(cfg)
    if args.dry_run:
        print("Settings and inputs validated. No training or output run was created.")
        return 0

    experiment_dir = project_path(Path(cfg.exp_root_dir) / cfg.name, root)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{cfg.run.shape}_{cfg.trainer.supervision}_{datetime.now():%Y%m%d-%H%M%S}_"
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=experiment_dir))
    cfg.tag, cfg.use_timestamp, cfg.timestamp = run_dir.name, False, ""
    config_dir = run_dir / "configs"
    config_dir.mkdir()
    shutil.copyfile(source_config, config_dir / "source.yaml")
    launch_config = config_dir / "launch.yaml"
    OmegaConf.save(cfg, launch_config)
    environment = runtime_environment(run_dir / "runtime", cfg.run.gpu)
    field_command, quad_command = stage_commands(cfg, launch_config, run_dir, root)
    status = {"stage1": "pending", "stage2": "pending" if cfg.run.stage2 else "skipped",
              "commands": {"stage1": field_command, "stage2": quad_command if cfg.run.stage2 else None}}
    def save_status():
        (run_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    print(f"Run directory: {run_dir}", flush=True)
    stages = [("stage1", field_command, run_dir / "dense_face_field.pt")]
    if cfg.run.stage2:
        stages.append(("stage2", quad_command, run_dir / "quad_pipeline/quad/quad_mesh.obj"))
    for name, command, expected_output in stages:
        status[name] = "running"
        save_status()
        try:
            run_logged(command, run_dir / f"{name}_console.log", cwd=root, env=environment)
            if not expected_output.is_file():
                raise RuntimeError(f"{name} finished without expected output: {expected_output}")
        except BaseException as error:
            status[name] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            status["error"] = str(error) or type(error).__name__
            save_status()
            raise
        status[name] = "complete"
        save_status()
    if cfg.run.stage2:
        status["viewer_meshes"] = export_mesh_files(run_dir / "quad_pipeline", run_dir)
        save_status()
    write_run_guide(run_dir, cfg.run.stage2)
    print(f"Dense field: {run_dir / 'dense_face_field.pt'}")
    if cfg.run.stage2:
        print(f"MeshLab FINAL QUAD: {run_dir / 'FINAL_QUAD.obj'}")
        print(f"MeshLab INPUT TRIANGLES: {run_dir / 'INPUT_TRIANGLE.obj'}")
    print(f"Output guide: {run_dir / 'README_OUTPUT_FILES.md'}")
    print(f"Resolved settings: {config_dir / 'parsed.yaml'}")
    return 0
