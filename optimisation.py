"""Stage 1 worker: load a mesh, fit its field, and save results.

For everyday use run ``python train.py``. This worker remains callable directly
for the paired experiment driver and existing dot-override commands.
"""
import argparse
import logging
import os
from pathlib import Path
import shlex
import sys
import time


DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs/hktex2quad_quad_gpu.yaml")


def configure_device(gpu, memory_fraction):
    """Select the device before importing the GPU model and its dependencies."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    import torch
    if memory_fraction is not None:
        if not 0 < memory_fraction <= 1:
            raise ValueError("cuda-memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def configure_logging(cfg, verbose):
    logger = logging.getLogger("hktex")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    if not cfg.optim.save_logs:
        return None
    handler = logging.FileHandler(Path(cfg.trial_dir) / "logs.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return handler


def save_results(trainer, cfg):
    """Keep the checkpoint, dense field and existing visualization contract."""
    import hktex
    output = Path(cfg.trial_dir)
    storage = (None, None)
    if cfg.optim.save_model:
        checkpoint_dir = output / "ckpts"
        checkpoint_dir.mkdir(exist_ok=True)
        storage = trainer.save_model(str(checkpoint_dir / cfg.optim.save_model_name))
        hktex.info(f"Model saved: PT {storage[0]:.2f} KB, NPZ {storage[1]:.2f} KB")
    dense_field = trainer.export_dense_field(str(output / "dense_face_field.pt"))
    hktex.info(f"Dense face field saved: {dense_field}")
    return storage, dense_field


def main(args, extras, render=False):
    """Execute one field fit. ``render`` is retained for old Python callers."""
    n_gpus = configure_device(args.gpu, getattr(args, "cuda_memory_fraction", None))
    import hktex
    from hktex.utils import load_config, save_config_snapshot, seed_everything

    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)
    file_handler = configure_logging(cfg, args.verbose)
    try:
        # Record settings before initialization, so failed runs can be diagnosed.
        save_config_snapshot(str(Path(cfg.trial_dir) / "configs"), cfg, args.config)
        (Path(cfg.trial_dir) / "cmd.txt").write_text(
            shlex.join([sys.executable, *sys.argv]) + "\n" + repr(extras) + "\n", encoding="utf-8")
        seed_everything(cfg.seed)

        # 1. Load geometry. Quad labels require unchanged vertex/face indices.
        data = hktex.find(cfg.data_type)(cfg.data)
        data.prepare_data()
        data.setup("fit")

        # 2. Build kernels, spectral queries, targets and optimizers.
        trainer = hktex.find(cfg.trainer_type)(cfg.trainer, data)

        # 3. Fit the field; its mathematical losses remain in QuadFieldTrainer.
        started = time.perf_counter()
        trainer.optimise(n_iter=cfg.optim.iters)
        hktex.info(f"Optimise took {time.perf_counter() - started:.2f} seconds")

        # 4. Export the same checkpoint/field formats used by MIQ and QEx.
        storage, dense_field = save_results(trainer, cfg)
        return {"optimisation": trainer, "datamodule": data,
                "storage": storage, "dense_field": dense_field}
    finally:
        if file_handler is not None:
            logging.getLogger("hktex").removeHandler(file_handler)
            file_handler.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", default="0", help="used only if CUDA_VISIBLE_DEVICES is unset")
    parser.add_argument("--cuda-memory-fraction", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="legacy no-op; fields are not RGB textures")
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    args, extras = parse_args()
    main(args, extras)
