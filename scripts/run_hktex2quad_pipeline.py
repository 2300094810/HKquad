"""Run HKTex2Quad from a dense face field to a quad OBJ."""

import argparse
import importlib.util
import json
from pathlib import Path


def _load_module(name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a quad mesh from an HKTex2Quad dense field"
    )
    parser.add_argument("--input", required=True, help="dense_face_field.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extractor", choices=("qex", "complete-cells"), default=None,
                        help="default: qex for MIQ, complete-cells for approximate")
    parser.add_argument("--qex-executable", default=None)
    parser.add_argument("--qex-timeout", type=float, default=600)
    parser.add_argument(
        "--backend",
        choices=("miq", "approximate"),
        default="miq",
        help=(
            "stage-2 parameterization backend; miq uses the real "
            "libigl/CoMISo mixed-integer solver"
        ),
    )
    parser.add_argument(
        "--miq-executable",
        default=None,
        help=(
            "path to hktex2quad_miq; otherwise use "
            "HKTEX2QUAD_MIQ_EXECUTABLE or the default build directory"
        ),
    )
    parser.add_argument(
        "--miq-gradient-scale",
        type=float,
        default=1.0,
        help="multiply MIQ's inverse target-size field; larger means more quads",
    )
    parser.add_argument("--miq-stiffness", type=float, default=5.0)
    parser.add_argument("--miq-iterations", type=int, default=5)
    parser.add_argument("--miq-local-iterations", type=int, default=5)
    parser.add_argument(
        "--miq-direct-round",
        action="store_true",
        help="greedily round all integers at once (faster, usually lower quality)",
    )
    parser.add_argument(
        "--miq-no-round-singularities",
        action="store_true",
        help="do not constrain singularities to integer UV coordinates",
    )
    parser.add_argument("--miq-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--continuity-weight",
        type=float,
        default=10000.0,
        help="approximate backend only",
    )
    parser.add_argument(
        "--integer-refinement-iterations",
        type=int,
        default=5,
        help="approximate backend only",
    )
    parser.add_argument(
        "--offset-resolution",
        type=int,
        default=8,
        help="approximate backend only; MIQ uses its exact integer lattice",
    )
    parser.add_argument(
        "--max-corner-face-hops",
        type=int,
        default=8,
        help=(
            "maximum topological distance used to match a cell corner on "
            "the same UV sheet"
        ),
    )
    parser.add_argument(
        "--weld-tolerance",
        type=float,
        default=None,
        help="3D seam weld radius; default is 1e-7 of the input bounding-box diagonal",
    )
    return parser.parse_args()


def main():
    args = _parse_arguments()
    bridge = _load_module(
        "hktex_quad_backend_pipeline", "hktex/modules/quad_backend.py"
    )
    seamless = _load_module(
        "hktex_seamless_parameterization_pipeline",
        "hktex/modules/seamless_parameterization.py",
    )
    miq = _load_module(
        "hktex_miq_parameterization_pipeline",
        "hktex/modules/miq_parameterization.py",
    )
    quad_mesh = _load_module(
        "hktex_quad_mesh_pipeline", "hktex/modules/quad_mesh.py"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = bridge.load_dense_face_field(args.input)
    bundle = bridge.make_backend_field_bundle(payload)
    if not bool(bundle.valid.all()):
        invalid_count = int((~bundle.valid).sum().item())
        raise RuntimeError(
            f"dense field has {invalid_count} invalid faces; improve coverage "
            "or field strength before parameterization"
        )
    bridge.save_backend_bundle(payload, output_dir / "backend_bundle")

    if args.backend == "miq":
        parameterization = miq.solve_miq_parameterization(
            bundle.vertices,
            bundle.faces,
            bundle.direction_u,
            bundle.direction_v,
            bundle.target_size,
            bundle.desired_gradient_u,
            bundle.desired_gradient_v,
            bundle.face_adjacency,
            output_dir / "miq_work",
            executable=args.miq_executable,
            gradient_scale=args.miq_gradient_scale,
            stiffness=args.miq_stiffness,
            iterations=args.miq_iterations,
            local_iterations=args.miq_local_iterations,
            direct_round=args.miq_direct_round,
            round_singularities=not args.miq_no_round_singularities,
            timeout=args.miq_timeout,
        )
        miq.save_miq_parameterization(
            parameterization,
            bundle.vertices,
            bundle.faces,
            output_dir / "parameterization",
        )
        stage_6 = {
            "backend": "libigl-comiso-miq",
            "mixed_integer": True,
            "rms_gradient_residual": parameterization.rms_gradient_residual,
            "relative_gradient_residual": (
                parameterization.relative_gradient_residual
            ),
            "flipped_face_count": parameterization.flipped_face_count,
            "flipped_face_fraction": parameterization.flipped_face_fraction,
            "seam_edge_count": int(parameterization.seam_edges.sum()),
            "chart_count": parameterization.chart_count,
            "cut_vertex_count": parameterization.cut_vertex_count,
            "miq_singularity_count": int(parameterization.singular.sum()),
            "gradient_scale": parameterization.gradient_scale,
            "stiffness": parameterization.stiffness,
            "iterations": parameterization.iterations,
            "local_iterations": parameterization.local_iterations,
            "direct_round": parameterization.direct_round,
            "round_singularities": parameterization.round_singularities,
        }
    else:
        parameterization = seamless.solve_fixed_transition_parameterization(
            bundle.vertices,
            bundle.faces,
            bundle.desired_gradient_u,
            bundle.desired_gradient_v,
            bundle.face_adjacency,
            bundle.edge_quarter_turns,
            bundle.seam_edges,
            continuity_weight=args.continuity_weight,
            integer_refinement_iterations=args.integer_refinement_iterations,
        )
        seamless.save_fixed_transition_parameterization(
            parameterization,
            bundle.vertices,
            bundle.faces,
            output_dir / "parameterization",
        )
        stage_6 = {
            "backend": "approximate-alternating-rounding",
            "mixed_integer": False,
            "rms_gradient_residual": parameterization.rms_gradient_residual,
            "relative_gradient_residual": (
                parameterization.relative_gradient_residual
            ),
            "rms_transition_residual": parameterization.rms_transition_residual,
            "max_transition_residual": parameterization.max_transition_residual,
            "flipped_face_count": parameterization.flipped_face_count,
            "flipped_face_fraction": parameterization.flipped_face_fraction,
            "seam_edge_count": int(parameterization.seam_edges.sum()),
            "chart_count": int(parameterization.face_charts.max() + 1),
            "cut_vertex_count": parameterization.cut_vertex_count,
            "integer_refinement_iterations": (
                parameterization.integer_refinement_iterations
            ),
            "translation_update_count": (
                parameterization.translation_update_count
            ),
        }

    extractor = args.extractor or ("qex" if args.backend == "miq" else "complete-cells")
    if extractor == "qex":
        qex = _load_module("qex_pipeline", "hktex/modules/qex_extraction.py")
        quads = qex.extract_qex_quads(
            bundle.vertices, bundle.faces, parameterization.corner_uv,
            output_dir / "qex_work", executable=args.qex_executable, timeout=args.qex_timeout)
    else:
        quads = quad_mesh.extract_integer_grid_quads(
            bundle.vertices, bundle.faces, parameterization.corner_uv,
            parameterization.face_charts, auto_offset=args.backend != "miq",
            offset_resolution=args.offset_resolution, weld_tolerance=args.weld_tolerance,
            max_corner_face_hops=args.max_corner_face_hops)
    quad_report = quad_mesh.save_quad_mesh(quads, output_dir / "quad")
    summary = {
        "format": "hktex2quad-pipeline-report-v2",
        "input": str(Path(args.input).resolve()),
        "stage_6": stage_6,
        "extractor": extractor,
        "stage_7_8": quads.metrics,
        "quad_obj": str((output_dir / "quad" / "quad_mesh.obj").resolve()),
        "limitation": (
            "QEx handles integer-grid seams; poor fields or folded maps can still yield poor quads."
            if extractor == "qex" else
            "Complete-cell extraction can leave gaps along seams and boundaries."
        ),
    }
    summary_path = output_dir / "pipeline_report.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Backend: {stage_6['backend']}\n"
        f"Quad mesh: {output_dir / 'quad' / 'quad_mesh.obj'}\n"
        f"Quads: {quads.metrics['quad_count']}\n"
        f"Parameterization RMS: {parameterization.rms_gradient_residual:.8g}\n"
        f"Report: {summary_path}\n"
        f"Detailed quad report: {quad_report}"
    )


if __name__ == "__main__":
    main()
