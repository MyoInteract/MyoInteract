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
import math
import ast
import os
from pathlib import Path

import pytest
from ml_collections import ConfigDict

# ---------------------------------------------------------------------------
# Helpers under test
# ---------------------------------------------------------------------------
from myoInteract.myointeract_hydra_cli import (
    button_geometry,
    extra_hydra_overrides,
    gradio_json_to_overrides,
    load_config_interactive,
    restore_button_geometry,
    training_config_to_labelled_dict,
)


def _rebuild_target(target: dict) -> dict:
    config = {
        "env": {"task_config": {"targets": {"num_targets": 1, "target_0": target}}}
    }
    labelled = training_config_to_labelled_dict(config)
    overrides = gradio_json_to_overrides(labelled) + extra_hydra_overrides(config)
    return load_config_interactive(overrides).env.task_config.targets.target_0.to_dict()


def test_fixed_sphere_position_survives_training_config_conversion() -> None:
    target = {
        "name": "pointing_target",
        "position": [0.45, -0.12, 0.07],
        "size": [0.03, 0.05],
    }
    reloaded = _rebuild_target(target)
    assert reloaded["position"] == [target["position"], target["position"]]
    assert reloaded["size"] == target["size"]


@pytest.mark.parametrize("ranged", [False, True])
@pytest.mark.parametrize("fixed_position", [False, True])
def test_button_rotation_and_depth_survive_training_config_conversion(
    ranged: bool, fixed_position: bool
) -> None:
    target = {
        "name": "button_target",
        "position": [0.4, -0.1, 0.2]
        if fixed_position
        else [[0.4, -0.1, 0.2], [0.45, -0.05, 0.25]],
        "size": [[0.03, 0.04, 0.003], [0.04, 0.05, 0.006]]
        if ranged
        else [0.03, 0.04, 0.003],
        "euler": [0.25, -0.4, 0.6],
        "min_touch_force": 2.5,
    }
    reloaded = _rebuild_target(target)
    assert reloaded["euler"] == pytest.approx(target["euler"])
    assert reloaded["position"] == (
        [target["position"], target["position"]]
        if fixed_position
        else target["position"]
    )
    assert reloaded["size"] == (
        target["size"] if ranged else [target["size"], target["size"]]
    )
    assert reloaded["min_touch_force"] == target["min_touch_force"]


def test_button_widget_edits_preserve_geometry_without_widgets() -> None:
    stored = {"euler": [0.25, -0.4, 0.6], "box_thickness": [0.006, 0.012]}
    live = {
        "orientation_angle": 30.0,
        "box_size_x": [0.1, 0.15],
        "box_size_y": [0.2, 0.25],
    }
    box = restore_button_geometry(live, stored)
    euler, depth = button_geometry(box)
    assert euler == pytest.approx([0.25, -math.pi / 6, 0.6])
    assert depth == [0.003, 0.006]
    assert box["box_size_x"] == live["box_size_x"]
    assert box["box_size_y"] == live["box_size_y"]
    assert "euler" not in live
    assert stored["euler"] == [0.25, -0.4, 0.6]
    assert button_geometry(restore_button_geometry(live, {})) == (
        [0, -math.pi / 6, 0],
        [0.01, 0.01],
    )


@pytest.mark.parametrize("studio_import", [False, True])
def test_gradio_form_keeps_hidden_geometry_out_of_widgets_and_preserves_resaves(
    tmp_path: Path,
    studio_import: bool,
) -> None:
    # Load the real form adapter classes without importing Gradio's GPU viewer
    # setup. Their serialization and argument parsing need no UI or simulator.
    source = Path(__file__).resolve().parents[1] / "myoInteract/scripts/ui.py"
    classes = {
        "ConfigSaver",
        "BoxParameters",
        "SphereParameters",
        "BMParameters",
        "TaskParameters",
        "ObservationSpace",
        "RewardFunction",
        "RLParameters",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"))
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in classes
        ],
        type_ignores=[],
    )
    namespace = {
        "os": os,
        "math": math,
        "CONFIG_PATH": tmp_path,
        "MAX_TARGETS": 10,
        "restore_button_geometry": restore_button_geometry,
        "button_geometry": button_geometry,
        "hex_to_rgb": lambda _value: [0.8, 0.1, 0.1],
    }
    exec(compile(module, str(source), "exec"), namespace)
    saver = namespace["ConfigSaver"]()
    boxes = namespace["BoxParameters"]
    boxes.sanity_check_args = staticmethod(lambda *_args: True)
    labelled = training_config_to_labelled_dict(
        {
            "env": {
                "task_config": {
                    "targets": {
                        "num_targets": 1,
                        "target_0": {
                            "name": "button_target",
                            "position": [[0.4, -0.1, 0.2], [0.4, -0.1, 0.2]],
                            "size": [[0.03, 0.04, 0.003], [0.04, 0.05, 0.006]],
                            "euler": [0.25, -0.4, 0.6],
                        },
                    }
                }
            },
        }
    )
    if studio_import:
        # Studio retains its legacy Y-angle control alongside the authoritative
        # rotation of a screen-mounted button. The latter takes precedence.
        labelled["target_1"]["box"]["orientation_angle"] = 90
    flat = saver.from_labelled_dict(labelled)
    assert len(flat) == 189
    expected_box = {
        **labelled["target_1"]["box"],
        "orientation_angle": 0.4 * 180 / math.pi,
    }
    assert boxes.get_my_args(0, flat) == [expected_box[key] for key in boxes.fields()]
    assert namespace["BMParameters"].get_my_args(flat) == list(labelled["bm"].values())
    saved = saver.to_labelled_dict(flat)
    assert saved["target_1"]["box"]["euler"] == pytest.approx([0.25, -0.4, 0.6])
    assert saved["target_1"]["box"]["box_thickness"] == [0.006, 0.012]

    # Playing immediately after import must use the same pose as re-saving.
    valid, overrides = boxes.parse_values(0, flat, saver.button_geometry.get(0))
    assert valid
    values = {
        key: json.loads(value)
        for key, value in (item.split("=", 1) for item in overrides)
    }
    assert values["env.task_config.targets.target_0.euler"] == pytest.approx(
        [0.25, -0.4, 0.6]
    )

    box_start = namespace["RLParameters"].num_elements + 1 + namespace["MAX_TARGETS"]
    flat[box_start + boxes.fields().index("orientation_angle")] = 30.0
    flat[box_start + boxes.fields().index("box_size_x")] = [0.1, 0.15]
    valid, overrides = boxes.parse_values(0, flat, saver.button_geometry.get(0))
    assert valid
    values = {
        key: json.loads(value)
        for key, value in (item.split("=", 1) for item in overrides)
    }
    assert values["env.task_config.targets.target_0.euler"] == pytest.approx(
        [0.25, -math.pi / 6, 0.6]
    )
    assert values["env.task_config.targets.target_0.size"] == [
        [0.05, 0.04, 0.003],
        [0.075, 0.05, 0.006],
    ]
    saved = saver.to_labelled_dict(flat)
    assert saved["target_1"]["box"]["euler"] == pytest.approx([0.25, -math.pi / 6, 0.6])

    # Loading an ordinary preset clears the previous policy's hidden geometry.
    del labelled["target_1"]["box"]["euler"]
    del labelled["target_1"]["box"]["box_thickness"]
    flat = saver.from_labelled_dict(labelled)
    assert saver.button_geometry[0] == {}
    assert "euler" not in saver.to_labelled_dict(flat)["target_1"]["box"]


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
