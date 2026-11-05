import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from myosuite.envs.myo.myouser import gradio_ui as ui


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "wandb", SimpleNamespace(api=SimpleNamespace(default_entity="test")))
    monkeypatch.setenv("MYOINTERACT_CONFIG_DIR", str(tmp_path / "presets"))
    state = ui.RunState()
    demo = ui.get_ui("test", state)
    callbacks = {event.fn.__name__: event.fn for event in demo.fns.values() if event.fn}
    source = Path(__file__).resolve().parents[1] / "gradio_configs" / "public-display-4-buttons.json"
    values = json.loads(source.read_text())
    yield state, callbacks, values
    demo.close()


def test_invalid_biomechanics_and_rl_leave_the_last_valid_run_intact(workflow):
    state, callbacks, values = workflow
    callbacks["run_training"]("study.v2", *values)
    before = state.cfg_overrides.copy()
    assert before
    assert "env.muscle_config.noise_params.sigdepnoise_type=null" in before
    assert "rl.load_checkpoint_path=null" in before
    invalid = values.copy()
    invalid[160] = None
    assert callbacks["run_training"]("study.v2", *invalid).startswith("Error:")
    assert state.cfg_overrides == before
    invalid = values.copy()
    invalid[4] = 0
    assert callbacks["run_training"]("study.v2", *invalid).startswith("Error:")
    assert state.cfg_overrides == before


def test_render_preserves_seed_and_resample_renders_the_returned_seed(workflow, monkeypatch):
    state, callbacks, values = workflow
    seen = []

    def render(config):
        seen.append(config.env.task_config.target_init_seed)
        return [("head", np.zeros((2, 2, 3), dtype=np.uint8)), ("side", np.zeros((2, 2, 3), dtype=np.uint8))]

    monkeypatch.setattr(ui, "render_configuration", render)
    original_seed = values[8]
    assert len(callbacks["render_environment"]("study", *values)) == 3
    assert seen == [original_seed]
    assert values[8] == original_seed
    assert not state.cfg_overrides
    result = callbacks["resample_environment"]("study", *values)
    assert len(result) == 4
    assert result[0]["value"] == original_seed + 1
    assert seen == [original_seed, result[0]["value"]]
    assert values[8] == original_seed


def test_missing_or_incomplete_checkpoint_directories_are_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "CHECKPOINT_PATH", tmp_path / "absent")
    monkeypatch.setattr(ui, "LOGS_PATH", tmp_path / "logs")
    assert ui.get_available_checkpoints() == ["None"]
    (tmp_path / "logs" / "new-run").mkdir(parents=True)
    assert ui.get_available_checkpoint_numbers("new-run") == ["None"]
    checkpoints = tmp_path / "logs" / "new-run" / "checkpoints"
    for number in ("100", "2", "20"):
        (checkpoints / number).mkdir(parents=True)
    assert ui.get_available_checkpoint_numbers("new-run") == ["2", "20", "100"]


def test_reload_preserves_explicit_backend_and_xla_settings(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_triton_gemm_any=False")
    importlib.reload(ui)
    importlib.reload(ui)
    assert os.environ["MUJOCO_GL"] == "glfw"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_triton_gemm_any=False"


def test_malformed_color_is_a_configuration_error_without_replacing_run(workflow):
    state, callbacks, values = workflow
    callbacks["run_training"]("study", *values)
    before = state.cfg_overrides.copy()
    values[27] = "rgba"
    with pytest.warns(UserWarning, match="RGB must be a valid"):
        result = callbacks["run_training"]("study", *values)
    assert result.startswith("Error:")
    assert state.cfg_overrides == before
