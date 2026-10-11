"""Interpreter selection, local caches and logged child processes. No GPU imports."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

CACHE_DIRECTORIES = {
    "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
    "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "matplotlib",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton",
    "TORCH_EXTENSIONS_DIR": "extensions", "CUDA_CACHE_PATH": "cuda",
}


def choose_python(requested=None):
    """Use an explicit interpreter, HKTEX_PYTHON, or the active environment."""
    requested = requested or os.environ.get("HKTEX_PYTHON")
    if requested:
        selected = shutil.which(requested)
        if selected is None:
            raise FileNotFoundError(f"Python interpreter not found: {requested}")
        return str(Path(selected).absolute())
    return str(Path(sys.executable).absolute())


def runtime_environment(directory, gpu=None):
    """Put all caches in the given project/run directory before starting Python."""
    directory = Path(directory).resolve()
    environment = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        environment.pop(key, None)
    for key, name in CACHE_DIRECTORIES.items():
        cache = directory / name
        cache.mkdir(parents=True, exist_ok=True)
        environment[key] = str(cache)
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def run_logged(command, log_path, *, cwd, env):
    """Print and save output; raise on failure and stop our child tree on Ctrl+C."""
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1, start_new_session=os.name == "posix")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except BaseException:
            # Each stage has its own group, including MIQ/QEx descendants.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if returncode:
        raise RuntimeError(f"Stage exited with code {returncode}; log: {log_path}")
