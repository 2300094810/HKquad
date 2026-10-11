"""Run libQEx in a separate process, preserving MIQ's per-corner seam UVs."""
import importlib.util
import os
from pathlib import Path
import subprocess
import numpy as np


def extract_qex_quads(vertices, faces, corner_uv, work_dir, executable=None, timeout=600):
    def array(x, dtype):
        return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=dtype)
    vertices, faces, uv = array(vertices, float), array(faces, np.int64), array(corner_uv, float)
    if uv.shape != (len(faces), 3, 2) or not np.isfinite(uv).all():
        raise ValueError("QEx requires finite per-triangle-corner UV coordinates")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid QEx vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid QEx faces")
    root = Path(__file__).resolve().parents[2]
    default = root / "cpp/qex_backend/build" / ("hktex2quad_qex.exe" if os.name == "nt" else "hktex2quad_qex")
    exe = Path(executable or os.environ.get("HKTEX2QUAD_QEX_EXECUTABLE", default)).resolve()
    if not exe.is_file():
        raise FileNotFoundError(f"Build the QEx backend first: {exe}; see cpp/qex_backend")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    source, destination = work / "qex_input.txt", work / "qex_output.obj"
    with source.open("w", encoding="ascii") as f:
        f.write(f"HKTEX2QUAD_QEX_V1\n{len(vertices)} {len(faces)}\n")
        np.savetxt(f, vertices, fmt="%.17g")
        np.savetxt(f, faces, fmt="%d")
        np.savetxt(f, uv.reshape(-1, 6), fmt="%.17g")
    proc = subprocess.run([str(exe), str(source.resolve()), str(destination.resolve())],
                          capture_output=True, text=True, timeout=timeout)
    (work / "qex.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"QEx exited {proc.returncode}; inspect {work / 'qex.log'}")
    vv, qq = [], []
    with destination.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "v":
                vv.append([float(v) for v in parts[1:4]])
            elif parts and parts[0] == "f":
                if len(parts) != 5:
                    raise RuntimeError("QEx returned a non-quad face")
                qq.append([int(i) - 1 for i in parts[1:]])
    vv, qq = np.asarray(vv).reshape(-1, 3), np.asarray(qq, dtype=np.int64).reshape(-1, 4)
    if not len(qq) or not np.isfinite(vv).all():
        raise RuntimeError("QEx returned empty or nonfinite geometry")
    if qq.min() < 0 or qq.max() >= len(vv) or any(len(set(q)) != 4 for q in qq):
        raise RuntimeError("QEx returned invalid face indices")
    spec = importlib.util.spec_from_file_location("qex_quad_metrics", Path(__file__).with_name("quad_mesh.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = module.evaluate_quad_mesh(vv, qq)
    metrics["extractor"] = "libQEx"
    # This CLI does not export optional grid/source provenance. Unknown
    # values are explicit sentinels, never fabricated charts/source indices.
    return module.QuadMeshResult(vv, qq, np.full((len(vv), 2), np.nan),
                                 np.full(len(vv), -1), np.empty((0, 2)),
                                 np.full(len(vv), -1), metrics)
