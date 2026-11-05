"""Resource and output paths that do not depend on the working directory."""

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def bundled_path(path):
    """Resolve a legacy ``myosuite/...`` model path inside the installed package."""
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    parts = value.parts
    if parts and parts[0] == "myosuite":
        return PACKAGE_ROOT.joinpath(*parts[1:]).resolve()
    # A user-supplied custom path retains ordinary cwd-relative semantics.
    return value.resolve()


def get_data_dir():
    override = os.environ.get("MYOINTERACT_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "myointeract"


def get_logs_dir():
    override = os.environ.get("MYOINTERACT_LOG_DIR")
    return Path(override).expanduser().resolve() if override else get_data_dir() / "logs"


def get_user_config_dir():
    override = os.environ.get("MYOINTERACT_CONFIG_DIR")
    return Path(override).expanduser().resolve() if override else get_data_dir() / "gradio_configs"


def get_bundled_config_dir():
    installed = PACKAGE_ROOT / "data" / "gradio_configs"
    source = PACKAGE_ROOT.parent / "gradio_configs"
    return installed if installed.is_dir() or not source.is_dir() else source


def get_bundled_checkpoint_dir():
    installed = PACKAGE_ROOT / "data" / "tracked_checkpoints" / "universal"
    source = PACKAGE_ROOT.parent / "tracked_checkpoints" / "universal"
    return installed if installed.is_dir() or not source.is_dir() else source
