"""Identify remote W&B runs consistently before looking up cached checkpoints."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


def normalize_wandb_run_path(run_path: str | Path) -> str:
    """Return ``entity/project/run_id`` from a run path or a W&B run URL."""
    value = str(run_path).strip().replace("\\", "/")
    if "://" in value:
        url = urlparse(value)
        if url.scheme != "https" or url.netloc not in {"wandb.ai", "www.wandb.ai"}:
            raise ValueError("Expected a https://wandb.ai run URL.")
        value = unquote(url.path)
    parts = value.strip("/").split("/")
    if len(parts) == 4 and parts[2] == "runs":
        del parts[2]
    if len(parts) != 3 or any(
        not part or part in {".", ".."} or any(ord(c) < 32 for c in part)
        for part in parts
    ):
        raise ValueError("Expected a W&B run path: entity/project/run_id.")
    return "/".join(parts)


def wandb_checkpoint_dir(log_path: Path, run_path: str | Path) -> Path:
    """Cache under the full run identity, independent of project/run name syntax.

    The fixed-length digest avoids filename restrictions and long paths on Windows.
    Old run-ID-only folders have no reliable provenance and are not reused here.
    """
    identity = normalize_wandb_run_path(run_path)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Path(log_path) / "wandb_checkpoints" / digest


def resolve_wandb_run_input(
    value: str, default_entity: str, default_project: str
) -> str | None:
    """Resolve remote UI input while leaving local files/directories to the caller."""
    value = value.strip()
    if (
        not value
        or value.endswith((".onnx", ".pt"))
        or Path(value).is_absolute()
        or re.match(r"^[A-Za-z]:[/\\]", value)
        or value.startswith(("./", "../", "~/", ".\\", "..\\"))
        or Path(value).exists()
    ):
        return None
    if "://" in value:
        return normalize_wandb_run_path(value)
    if "/" not in value and "\\" not in value:
        return normalize_wandb_run_path(f"{default_entity}/{default_project}/{value}")
    try:
        return normalize_wandb_run_path(value)
    except ValueError:
        return None
