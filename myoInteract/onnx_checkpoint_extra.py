# Copyright (c) myoInteract Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""myointeract-specific ONNX checkpoint helpers not part of MyoSuite's public API.

Everything myosuite-generic (is_onnx_checkpoint_name, onnx_checkpoint_sort_key,
get_wandb_onnx_checkpoint_path, read_onnx_checkpoint_metadata) is imported
from myosuite.utils.onnx_checkpoint directly. Only the two functions
below, which are myointeract-config-shape-specific, live here:
- extract_wandb_hydra_config: walks the hydra "env" config schema this
  runs of the Gradio UI embed in W&B.
- download_all_wandb_onnx_checkpoints: also downloads myointeract_config.json
  sidecar files (a myointeract convention, not a generic MyoSuite one).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from myosuite.utils.onnx_checkpoint import (
    is_onnx_checkpoint_name,
    onnx_checkpoint_sort_key,
)


def download_all_wandb_onnx_checkpoints(
    log_path: Path,
    run_path: str | Path,
) -> tuple[list[Path], list[bool]]:
    """Download **all** ONNX checkpoints from a W&B run.

    Files land in ``log_path/wandb_checkpoints/{run_id}/``.  Returns
    ``(paths, was_cached)`` where both lists are sorted by
    :func:`onnx_checkpoint_sort_key` (``model_final.onnx`` last).
    ``was_cached[i]`` is ``True`` when the file already existed locally.
    """
    import wandb

    run_id = str(run_path).split("/")[-1]
    download_dir = log_path / "wandb_checkpoints" / run_id
    api = wandb.Api()
    wandb_run = api.run(str(run_path))
    files = sorted(
        [f.name for f in wandb_run.files() if is_onnx_checkpoint_name(f.name)],
        key=onnx_checkpoint_sort_key,
    )
    if not files:
        raise FileNotFoundError(f"No ONNX checkpoints found in W&B run {run_path}.")
    download_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    was_cached: list[bool] = []
    for fname in files:
        p = download_dir / fname
        cached = p.exists()
        if not cached:
            wandb_run.file(fname).download(str(download_dir), replace=True)
        paths.append(p)
        was_cached.append(cached)

    # Also download myointeract_config.json if the run has one (uploaded by
    # launch_training so all Python overrides are
    # preserved).
    _cfg_name = "myointeract_config.json"
    _cfg_local = download_dir / _cfg_name
    if not _cfg_local.exists():
        try:
            _run_files = {f.name for f in wandb_run.files()}
            if _cfg_name in _run_files:
                wandb_run.file(_cfg_name).download(str(download_dir), replace=True)
        except Exception:
            pass

    return paths, was_cached


def extract_wandb_hydra_config(wandb_config: dict) -> dict | None:
    """Extract the myointeract hydra config embedded in a W&B run config dict.

    Runs of the Gradio UI store the hydra env config as a YAML literal string at
    ``config["env_cfg"]["scene"]["entities"][<entity_name>]["base_config"]``.
    This function traverses that path, finds the first non-empty string value,
    parses it with ``yaml.safe_load``, and returns the resulting dict (which
    has a top-level ``"env"`` key matching the myointeract hydra config format).

    Returns ``None`` when no parseable ``base_config`` is found.
    """
    import yaml as _yaml

    env_cfg = wandb_config.get("env_cfg", {})
    if not isinstance(env_cfg, dict):
        return None
    scene = env_cfg.get("scene", {})
    if not isinstance(scene, dict):
        return None
    entities = scene.get("entities", {})
    if not isinstance(entities, dict):
        return None
    for entity_val in entities.values():
        if not isinstance(entity_val, dict):
            continue
        base_config_str = entity_val.get("base_config", None)
        if isinstance(base_config_str, str) and base_config_str.strip():
            try:
                parsed: Any = _yaml.safe_load(base_config_str)
                if isinstance(parsed, dict) and "env" in parsed:
                    return parsed
            except Exception:
                continue
    return None
