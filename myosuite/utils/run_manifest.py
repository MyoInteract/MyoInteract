"""Write reproducibility metadata without requiring a tracking service."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


RUNTIME_PACKAGES = (
    "MyoSuite", "numpy", "scipy", "jax", "jaxlib", "mujoco", "mujoco-mjx",
    "playground", "brax", "flax", "optax", "orbax-checkpoint", "gradio",
)


def checkpoint_digest(path):
    """Hash checkpoint filenames and bytes so policy identity survives relocation."""
    root = Path(path)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    for file in files:
        name = file.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(file.stat().st_size.to_bytes(8, "big"))
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_revision(source):
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(source), *args], capture_output=True, text=True,
            check=False, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        # A wheel installed inside another project's venv is not that project's
        # source checkout. Never attribute its code to an unrelated parent repo.
        top_level = git("rev-parse", "--show-toplevel")
        if not top_level or Path(top_level).resolve() != Path(source).resolve():
            return {"commit": None, "dirty": None}
        revision = git("rev-parse", "HEAD")
        return {"commit": revision, "dirty": bool(git("status", "--porcelain")) if revision else None}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def create_run_manifest(config, *, checkpoint_path=None, backend=None, devices=()):
    """Snapshot effective config, code, dependencies, seeds, and restore semantics."""
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    # Round-trip to detach the snapshot from the caller's mutable configuration.
    effective = json.loads(json.dumps(config, allow_nan=False))
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"), allow_nan=False)
    versions = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    run = effective.get("run", {})
    if run.get("play_only") or effective.get("env", {}).get("eval_mode"):
        mode = "evaluation"
    elif checkpoint_path and str(checkpoint_path) != "None":
        mode = "parameter_warm_start"
    else:
        mode = "fresh_training"
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "source": _git_revision(Path(__file__).resolve().parents[2]),
        "config": effective,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "restore": {
            "mode": mode,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": checkpoint_digest(checkpoint_path) if checkpoint_path else None,
            "optimizer_state_restored": False,
            "training_step_restored": False,
        },
        "seeds": {key: run.get(key) for key in ("seed", "eval_seed")},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": versions,
            "backend": backend,
            "devices": [str(device) for device in devices],
            "environment": {key: os.environ[key] for key in (
                "MUJOCO_GL", "XLA_FLAGS", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE",
            ) if key in os.environ},
        },
    }


def write_run_manifest(directory, manifest):
    """Atomically replace the manifest without leaving a partial JSON document."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
