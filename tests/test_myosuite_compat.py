"""Smoke tests against the installed MyoSuite 3 and mjlab: registration, env build and the CLI."""

import subprocess
import sys

import pytest

pytest.importorskip("mjlab")


def test_myo_train_builds_its_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "myoInteract.scripts.train", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-600:]
    assert "--agent.max-iterations" in result.stdout


def test_muscle_action_is_myosuites_with_noise_before_fatigue() -> None:
    """The universal task uses MyoSuite's action term: sigmoid -> motor noise -> fatigue."""
    import json
    from pathlib import Path

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoAction

    from myoInteract.myointeract_hydra_cli import (
        gradio_json_to_overrides,
        load_config_interactive,
    )
    from myoInteract.register_mjlab_myointeract_tasks import (
        register_mjlab_myointeract_task,
    )

    preset = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "myo_config/gradio_tasks/pointing.json"
        ).read_text()
    )
    preset["bm"].update({"sigdepnoise_enabled": True, "fatigue_enabled": True})
    register_mjlab_myointeract_task(
        load_config_interactive(gradio_json_to_overrides(preset))
    )
    cfg = load_env_cfg("myoInteractUniversal-v0")
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    term = env.action_manager.get_term("muscles")
    assert isinstance(term, MyoAction)
    assert term.stage_names[:2] == ("noise", "fatigue")
    assert (
        term.cfg.motor_noise.signal_dependent_std == 0.103
        and term.cfg.motor_noise.constant_std == 0.0
    )
    env.reset()
    env.step(
        torch.full((2, env.action_manager.total_action_dim), 3.0)
    )  # beyond [-1, 1]: not clipped
    assert term._fatigue is not None and float(term._fatigue.MA.max()) > 0.0
    env.close()
