"""Batch-map reference quads onto mandatory independent triangle inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_quad_supervision import prepare_pair


def prepare_dataset(source_dir, output_dir, pattern="**/flow/quad_smooth_flow.obj", limit=None, source_url="", triangle_relative=None):
    if not triangle_relative:
        raise ValueError("triangle_relative is required; missing triangle inputs cannot be generated from quads")
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output == source or source in output.parents:
        raise ValueError("Keep prepared data outside the raw source directory")
    candidates = sorted(p for p in source.glob(pattern) if p.is_file())
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError(f"No files match {pattern} in {source}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in candidates:
        relative = path.relative_to(source).as_posix()
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", str(Path(relative).with_suffix(""))).strip("-")
        shape_id = stem[:110] + "-" + hashlib.sha256(relative.encode()).hexdigest()[:8]
        record = dict(shape_id=shape_id, source=relative)
        try:
            triangle = (path.parent / triangle_relative).resolve()
            if not triangle.is_relative_to(source):
                raise ValueError("Triangle input must stay within the raw source directory")
            metadata = prepare_pair(path, output / shape_id, source_url, triangle=triangle)
            record.update(status="ready", quad_count=metadata["quad_count"], triangle_count=metadata["triangle_count"])
            print(f"READY {shape_id}: {metadata['quad_count']} quads", flush=True)
        except (ValueError, OSError) as error:
            record.update(status="skipped", reason=str(error))
            print(f"SKIP {relative}: {error}", flush=True)
        records.append(record)
    manifest = dict(source_dir=str(source), pattern=pattern, triangle_relative=triangle_relative, records=records,
                    ready_count=sum(r["status"] == "ready" for r in records),
                    skipped_count=sum(r["status"] == "skipped" for r in records))
    # The manifest describes this invocation, including every rejected candidate.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern", default="**/flow/quad_smooth_flow.obj", help="reference quad glob; BiMDF flow variant by default")
    parser.add_argument("--triangle-relative", required=True, help="independent triangle path relative to each quad parent, e.g. ../mesh_rem.obj")
    parser.add_argument("--limit", type=int, help="optional limit on candidates inspected")
    parser.add_argument("--source-url", default="")
    args = parser.parse_args()
    manifest = prepare_dataset(args.source_dir, args.output_dir, args.pattern, args.limit, args.source_url, args.triangle_relative)
    print(f"Ready: {manifest['ready_count']}; skipped: {manifest['skipped_count']}")
    if not manifest["ready_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
