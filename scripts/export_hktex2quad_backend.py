"""Convert a dense HKTex2Quad trainer export into a portable backend bundle."""

import argparse
import importlib.util
from pathlib import Path


def _load_bridge_module():
    # Keep this utility usable even when optional HKTex rendering dependencies
    # are not installed: quad_backend itself only needs PyTorch and NumPy.
    module_path = (
        Path(__file__).resolve().parents[1]
        / "hktex"
        / "modules"
        / "quad_backend.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hktex_quad_backend_export", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load backend bridge from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Export HKTex2Quad face crosses and sizes for a quad backend"
    )
    parser.add_argument(
        "--input", required=True, help="path to dense_face_field.pt"
    )
    parser.add_argument(
        "--output-dir", required=True, help="directory for the portable bundle"
    )
    args = parser.parse_args()

    bridge = _load_bridge_module()
    payload = bridge.load_dense_face_field(args.input)
    manifest = bridge.save_backend_bundle(payload, args.output_dir)
    print(f"HKTex2Quad backend bundle written to {manifest}")


if __name__ == "__main__":
    main()

