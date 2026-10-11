"""Map reference quad directions onto a separately supplied triangle mesh."""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np


def prepare_pair(quad, output_dir, source_url="", *, triangle,
                 max_distance_ratio=0.01, max_normal_angle=60., minimum_label_area=0.95):
    """Require an independent input; never create it by splitting the reference."""
    if triangle is None:
        raise ValueError("An independent triangle input is required; quad-only preparation is not supported")
    return prepare_cross_mesh_pair(triangle, quad, output_dir, source_url,
                                   max_distance_ratio, max_normal_angle, minimum_label_area)


def prepare_cross_mesh_pair(triangle, quad, output_dir, source_url="",
                            max_distance_ratio=.01, max_normal_angle=60., minimum_label_area=.95):
    """Preserve the supplied triangle file and precompute only its reference signal."""
    triangle, quad, output = Path(triangle).resolve(), Path(quad).resolve(), Path(output_dir).resolve()
    if not 0 < minimum_label_area <= 1:
        raise ValueError("minimum_label_area must be in (0, 1]")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = dict(input_triangle_sha256=sha(triangle), sha256=sha(quad))
    settings = dict(max_distance_ratio=float(max_distance_ratio), max_normal_angle=float(max_normal_angle),
                    minimum_label_area=float(minimum_label_area))
    if output.exists() and any(output.iterdir()):
        existing = output / "provenance.json"
        if existing.is_file():
            metadata = json.loads(existing.read_text(encoding="utf-8"))
            if (all(metadata.get(k) == h for k, h in hashes.items())
                    and metadata.get("mapping_parameters") == settings
                    and metadata.get("pairing") == "independent input triangles; projected quad direction supervision"
                    and (output / "triangle.obj").is_file() and (output / "quad_targets.npz").is_file()
                    and sha(output / "triangle.obj") == hashes["input_triangle_sha256"]
                    and sha(output / "quad_targets.npz") == metadata.get("labels_sha256")):
                return metadata
        raise FileExistsError(f"Output is not an identical complete cross-mesh pair: {output}")
    spec = importlib.util.spec_from_file_location("quad_labels", Path(__file__).resolve().parents[1] / "hktex/modules/quad_supervision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vertices, faces = module.read_triangle_obj(triangle)
    reference_vertices, quads = module.read_quad_obj(quad)
    arrays, report = module.map_quad_directions(vertices, faces, reference_vertices, quads,
                         max_distance_ratio=max_distance_ratio, max_normal_angle=max_normal_angle)
    if report["reliable_label_area_fraction"] < minimum_label_area:
        raise ValueError(f"Insufficient trustworthy mapped supervision: {json.dumps(report)}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(triangle, output / "triangle.obj")
    np.savez_compressed(output / "quad_targets.npz", format="hktex2quad-quad-supervision-v2",
                        vertices=vertices, faces=faces, **arrays)
    metadata = dict(source=str(quad), triangle_source=str(triangle), source_url=source_url,
                    **hashes, vertex_count=len(vertices), triangle_count=len(faces), quad_count=len(quads),
                    pairing="independent input triangles; projected quad direction supervision",
                    label="quad boundary edges projected and transported to input face tangent planes",
                    mapping_parameters=settings, mapping_report=report,
                    labels_sha256=sha(output / "quad_targets.npz"))
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quad", required=True, help="reference mesh with quad-only OBJ faces")
    parser.add_argument("--triangle", required=True, help="independent input triangle OBJ; preserved exactly")
    parser.add_argument("--max-distance-ratio", type=float, default=.01, help="maximum mapping distance / input bbox diagonal")
    parser.add_argument("--max-normal-angle", type=float, default=60., help="maximum signed normal mismatch in degrees")
    parser.add_argument("--minimum-label-area", type=float, default=.95, help="reject preparation below this reliable area fraction")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-url", default="")
    args = parser.parse_args()
    print(json.dumps(prepare_pair(args.quad, args.output_dir, args.source_url, triangle=args.triangle,
                       max_distance_ratio=args.max_distance_ratio, max_normal_angle=args.max_normal_angle,
                       minimum_label_area=args.minimum_label_area), indent=2))


if __name__ == "__main__":
    main()
