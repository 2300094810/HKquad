"""HKTex2Quad: run training and quad extraction with ``python train.py``."""
import os
from pathlib import Path
import sys

# Bootstrap uses only the standard library, even before conda activate.
sys.dont_write_bytecode = True
from scripts.runtime import choose_python, runtime_environment
from scripts.train_pipeline import parse_args, run_pipeline

ROOT = Path(__file__).resolve().parent


def main():
    args = parse_args()
    try:
        if not args.worker:
            python = choose_python(args.python)
            environment = runtime_environment(ROOT / ".runtime" / "launcher")
            # Start a clean interpreter before importing CUDA-dependent modules.
            os.execve(python, [python, str(ROOT / "train.py"), *sys.argv[1:], "--_worker"], environment)
        return run_pipeline(args, ROOT)
    except KeyboardInterrupt:
        print("Training interrupted; see this run's logs and run_status.json.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Cannot complete training: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
