"""Render an existing HKTex2Quad run without retraining it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from omegaconf import OmegaConf
import torch

from hktex.utils.quad_field_visualization import (
    kernel_visualization_data_from_checkpoint,
    save_quad_field_visualizations,
)


def _resolve_inputs(args):
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir is not None:
        dense_path = (
            Path(args.dense_field)
            if args.dense_field
            else run_dir / "dense_face_field.pt"
        )
        config_path = Path(args.config) if args.config else run_dir / "configs" / "parsed.yaml"
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            candidates = sorted((run_dir / "ckpts").glob("*.pt"))
            if not candidates:
                candidates = sorted((run_dir / "ckpts").glob("*.npz"))
            if not candidates:
                raise FileNotFoundError(f"no checkpoint found below {run_dir / 'ckpts'}")
            checkpoint_path = candidates[0]
        output_dir = Path(args.output_dir) if args.output_dir else run_dir / "visualizations"
    else:
        if not args.dense_field or not args.checkpoint:
            raise ValueError(
                "provide --run-dir, or provide both --dense-field and --checkpoint"
            )
        dense_path = Path(args.dense_field)
        checkpoint_path = Path(args.checkpoint)
        config_path = Path(args.config) if args.config else None
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else dense_path.parent / "visualizations"
        )
    return dense_path.resolve(), checkpoint_path.resolve(), config_path, output_dir.resolve()


def _load_checkpoint(filename: Path) -> dict:
    if filename.suffix.lower() == ".npz":
        with np.load(filename, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    state = torch.load(filename, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    return state


def _visualization_config(config_path: Path | None) -> dict:
    defaults = {
        "range_enforcement_type": "pgd",
        "diffusion_time": 0.1,
        "min_target_size": 1e-4,
        "max_target_size": 1e4,
    }
    if config_path is None or not config_path.is_file():
        print(
            "Warning: parsed config not found; assuming PGD parameters and "
            "default size bounds. Pass --config if this checkpoint used activations."
        )
        return defaults
    config = OmegaConf.load(config_path)
    defaults.update(
        {
            "range_enforcement_type": OmegaConf.select(
                config,
                "trainer.model.range_enforcement_type",
                default=defaults["range_enforcement_type"],
            ),
            "diffusion_time": float(
                OmegaConf.select(
                    config,
                    "trainer.model.diff_time",
                    default=defaults["diffusion_time"],
                )
            ),
            "min_target_size": float(
                OmegaConf.select(
                    config,
                    "trainer.model.min_target_size",
                    default=defaults["min_target_size"],
                )
            ),
            "max_target_size": float(
                OmegaConf.select(
                    config,
                    "trainer.model.max_target_size",
                    default=defaults["max_target_size"],
                )
            ),
        }
    )
    return defaults


def main():
    parser = argparse.ArgumentParser(
        description="Create paper-style 4-RoSy and heat-kernel figures"
    )
    parser.add_argument(
        "--run-dir",
        help="experiment directory containing dense field/config/ckpt",
    )
    parser.add_argument("--dense-field")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", help="parsed.yaml used for parameter activations")
    parser.add_argument("--output-dir")
    parser.add_argument("--dense-glyph-count", type=int, default=700)
    parser.add_argument("--kernel-glyph-count", type=int, default=192)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-58.0)
    args = parser.parse_args()

    dense_path, checkpoint_path, config_path, output_dir = _resolve_inputs(args)
    dense_payload = torch.load(dense_path, map_location="cpu", weights_only=True)
    if not isinstance(dense_payload, dict):
        raise ValueError("dense field must contain a dictionary")
    checkpoint = _load_checkpoint(checkpoint_path)
    config = _visualization_config(config_path)
    kernel_data = kernel_visualization_data_from_checkpoint(
        dense_payload,
        checkpoint,
        **config,
    )
    paths = save_quad_field_visualizations(
        dense_payload,
        kernel_data,
        output_dir,
        dense_glyph_count=args.dense_glyph_count,
        kernel_glyph_count=args.kernel_glyph_count,
        dpi=args.dpi,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )
    print("HKTex2Quad visualizations saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
