"""Checkpoint continuation must select its sensing recipe before registration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from myoInteract.myointeract_hydra_cli import (
    gradio_json_to_overrides,
    load_config_interactive,
    training_task_overrides,
)


@pytest.fixture
def vision_preset() -> dict:
    return {
        "obs": {"obs": [True] * 5, "fatigue": [], "vision": [True, False], "omni": []}
    }


@pytest.mark.parametrize("version", [None, 2])
@pytest.mark.parametrize("equals", [False, True])
def test_explicit_continuation_uses_saved_legacy_or_new_vision_recipe(
    tmp_path: Path, vision_preset: dict, version: int | None, equals: bool
):
    source = load_config_interactive(
        [
            "env=universal",
            "vision=enabled",
            "vision.vision_mode=depth",
            "vision.render_width=64",
            "vision.depth_cutoff=2.0",
        ]
    ).to_dict()
    if version is None:
        for value in (source["vision"], source["env"]["vision"]):
            value.pop("recipe_version", None)
    (tmp_path / "myointeract_config.json").write_text(json.dumps(source))
    checkpoint = tmp_path / "checkpoint with spaces.onnx"
    args = (
        [f"--resume-checkpoint={checkpoint}"]
        if equals
        else ["--resume-checkpoint", str(checkpoint)]
    )
    restored = load_config_interactive(training_task_overrides(vision_preset, args))
    assert restored.vision.recipe_version == (1 if version is None else version)
    assert restored.vision.vision_mode == "depth"
    assert restored.vision.render_width == 64
    assert restored.vision.depth_cutoff == 2.0


@pytest.mark.parametrize(
    "override",
    [
        "vision.recipe_version=2",
        "env.vision.recipe_version=2",
        "+env.vision.recipe_version=2",
        "++vision.recipe_version=2",
    ],
)
def test_explicit_preset_recipe_wins_without_checkpoint_metadata(
    vision_preset, override
):
    preset = copy.deepcopy(vision_preset)
    preset["extra_overrides"] = [override]
    overrides = training_task_overrides(preset, ["--agent.resume", "True"])
    assert load_config_interactive(overrides).vision.recipe_version == 2
    assert "vision.recipe_version=2" in overrides


def test_missing_vision_checkpoint_metadata_requires_an_explicit_recipe(
    tmp_path, vision_preset
):
    with pytest.raises(ValueError, match="myointeract_config.json.*extra_overrides"):
        training_task_overrides(
            vision_preset, ["--resume-checkpoint", str(tmp_path / "model.onnx")]
        )


@pytest.mark.parametrize(
    "args,enabled",
    [
        (["--agent.resume", "True"], True),
        (["--agent.resume=True"], True),
        (["--agent.resume", "False"], False),
        (["--agent.resume=False"], False),
        (["--agent.resume=True", "--agent.resume", "False"], False),
    ],
)
def test_auto_resume_requires_an_unambiguous_vision_recipe(
    vision_preset, args, enabled
):
    if enabled:
        with pytest.raises(ValueError, match="auto-resume.*--resume-checkpoint"):
            training_task_overrides(vision_preset, args)
    else:
        assert training_task_overrides(vision_preset, args) == gradio_json_to_overrides(
            vision_preset
        )


def test_nonvision_continuation_needs_no_vision_recipe(tmp_path):
    preset = {"obs": {"obs": [True] * 5, "fatigue": [], "vision": [], "omni": []}}
    assert training_task_overrides(
        preset, ["--agent.resume=True"]
    ) == gradio_json_to_overrides(preset)
    assert training_task_overrides(
        preset, ["--resume-checkpoint", str(tmp_path / "model.onnx")]
    ) == gradio_json_to_overrides(preset)
