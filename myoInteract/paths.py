"""Where myoInteract keeps its runs, logs and user presets."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def is_repo_checkout() -> bool:
    """True when running from a source checkout (not a pip install into site-packages)."""
    return (REPO_ROOT / "tutorials" / "myointeract").is_dir()


def data_root() -> Path:
    """Root of the persistent data (``logs/``, Studio runs, user presets).

    ``$MYOINTERACT_HOME`` if set; the repository root in a source checkout; otherwise
    ``./myoInteract_data``, so a pip install never writes into site-packages.
    """
    if "MYOINTERACT_HOME" in os.environ:
        return Path(os.environ["MYOINTERACT_HOME"]).resolve()
    if is_repo_checkout():
        return REPO_ROOT
    return Path.cwd() / "myoInteract_data"
