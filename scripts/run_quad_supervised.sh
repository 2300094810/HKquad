#!/usr/bin/env bash
# Compatibility for previous GPU=... SHAPE=... commands; new use: python train.py.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
args=(--python "$PYTHON")
# Forward only explicitly set variables; defaults now live in the YAML.
for pair in GPU:gpu SHAPE:shape CONFIG:config DATASET_ROOT:dataset-root DATA_DIR:data-dir \
            OUTPUT_ROOT:output-root MEMORY_FRACTION:memory-fraction ITERS:iters \
            KERNELS:kernels TARGET_QUADS:target-quads SEED:seed; do
  variable="${pair%%:*}"
  option="${pair#*:}"
  [[ -z "${!variable:-}" ]] || args+=("--$option" "${!variable}")
done
if [[ -n "${RUN_STAGE2:-}" ]]; then
  case "$RUN_STAGE2" in
    0) args+=(--stage1-only) ;;
    1) args+=(--stage2) ;;
    *) echo 'RUN_STAGE2 must be 0 or 1.' >&2; exit 1 ;;
  esac
fi
[[ -z "${DIFF_TIME:-}" ]] || args+=("trainer.model.diff_time=$DIFF_TIME")
[[ -z "${KNN_OUTER:-}" ]] || args+=("trainer.model.knn_outer_k=$KNN_OUTER")
[[ -z "${KNN_INNER:-}" ]] || args+=("trainer.model.knn_inner_k=$KNN_INNER")
exec "$PYTHON" "$REPO/train.py" "${args[@]}" "$@"
