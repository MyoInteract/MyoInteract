"""Unit tests: myointeract_config.json round-trip.

Verifies that:
1. A ConfigDict with Python overrides (as used in a training script or
   launch_training) is correctly serialised to myointeract_config.json via
   config.to_dict() + json.dump().
2. The JSON, when loaded back by gradio_ui.py's load_policy_clicked code path
   (json.loads → training_config_to_labelled_dict → gradio_json_to_overrides +
   extra_hydra_overrides), reproduces the overridden values in the override
   strings that are ultimately fed to load_config_interactive().
3. The override strings, when applied by load_config_interactive(), produce a
   config that matches the original on every overridden key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ml_collections import ConfigDict

# ---------------------------------------------------------------------------
# Helpers under test
# ---------------------------------------------------------------------------
from myoInteract.myointeract_hydra_cli import (
    extra_hydra_overrides,
    gradio_json_to_overrides,
    load_config_interactive,
    training_config_to_labelled_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_CFG_PATH = (
    Path(__file__).resolve().parents[1] / "myo_config/tasks/pointing_anthro.json"
)


def _make_overridden_config() -> ConfigDict:
    """Reproduce the Python overrides from a training script."""
    with open(_BASE_CFG_PATH) as f:
        config = ConfigDict(json.load(f))
    config.env.vision.enabled = False
    config.env.task_config.omni_keys = [
        "target_pos",
        "target_size",
        "phase_progress",
        "dwell_fraction",
    ]
    config.env.muscle_config.fatigue_enabled = True
    config.env.muscle_config.fatigue_reset_random = True
    config.env.task_config.reset_type = "None"
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_to_dict_preserves_overrides():
    """config.to_dict() keeps every Python override intact."""
    config = _make_overridden_config()
    d = config.to_dict()

    assert d["env"]["vision"]["enabled"] is False
    assert d["env"]["task_config"]["omni_keys"] == [
        "target_pos",
        "target_size",
        "phase_progress",
        "dwell_fraction",
    ]
    assert d["env"]["muscle_config"]["fatigue_enabled"] is True
    assert d["env"]["muscle_config"]["fatigue_reset_random"] is True
    assert d["env"]["task_config"]["reset_type"] == "None"


def test_json_roundtrip_preserves_overrides(tmp_path):
    """Serialising to JSON and reading it back keeps every override."""
    config = _make_overridden_config()
    cfg_file = tmp_path / "myointeract_config.json"
    with open(cfg_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    loaded = json.loads(cfg_file.read_text())

    assert loaded["env"]["vision"]["enabled"] is False
    assert loaded["env"]["task_config"]["omni_keys"] == [
        "target_pos",
        "target_size",
        "phase_progress",
        "dwell_fraction",
    ]
    assert loaded["env"]["muscle_config"]["fatigue_enabled"] is True
    assert loaded["env"]["muscle_config"]["fatigue_reset_random"] is True
    assert loaded["env"]["task_config"]["reset_type"] == "None"


def test_extra_hydra_overrides_contains_fatigue_fields():
    """extra_hydra_overrides emits overrides for non-UI fatigue fields."""
    config = _make_overridden_config()
    config_dict = config.to_dict()

    overrides = extra_hydra_overrides(config_dict)
    override_str = "\n".join(overrides)

    assert "env.muscle_config.fatigue_reset_random=True" in override_str
    assert "env.muscle_config.fatigue_enabled=True" in override_str


def test_full_roundtrip_via_gradio_overrides():
    """The combined override list (gradio_json + extra) reconstructs overridden values.

    This mirrors exactly what load_policy_clicked does when loading a
    myointeract_config.json sidecar: json.loads → training_config_to_labelled_dict
    → gradio_json_to_overrides + extra_hydra_overrides → load_config_interactive.

    Requires the full mjlab / Hydra stack; skipped when not available.
    """
    pytest.importorskip("mjlab")

    config = _make_overridden_config()
    config_dict = config.to_dict()

    labelled = training_config_to_labelled_dict(config_dict)
    overrides = gradio_json_to_overrides(labelled) + extra_hydra_overrides(config_dict)

    # Check that the key overrides are present somewhere in the list.
    override_str = "\n".join(overrides)
    assert "env.muscle_config.fatigue_reset_random=True" in override_str
    assert "env.task_config.reset_type=None" in override_str
    # network_factory sub-keys must NOT appear — they are Hydra group overrides
    # and cannot be applied with dot notation.
    assert "rl.network_factory." not in override_str

    # Reconstruct via load_config_interactive and compare.
    reloaded = load_config_interactive(overrides)

    assert reloaded.env.muscle_config.fatigue_enabled is True
    assert reloaded.env.muscle_config.fatigue_reset_random is True
    assert reloaded.env.task_config.reset_type == "None"
    assert reloaded.env.vision.enabled is False
    assert list(reloaded.env.task_config.omni_keys) == [
        "target_pos",
        "target_size",
        "phase_progress",
        "dwell_fraction",
    ]
