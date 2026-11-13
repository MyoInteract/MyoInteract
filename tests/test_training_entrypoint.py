"""CLI lifecycle plumbing, using the native config types and a controlled runner."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg

from myoInteract.scripts import train


def _exercise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stop: bool = False,
    fail: bool = False,
    timing_error: bool = False,
) -> tuple[dict, object, object]:
    source = tmp_path / "source.onnx"
    source.write_bytes(b"source unchanged")
    preset = Path(train._DEFAULT_CFG)
    stop_file = tmp_path / "stop.requested"
    if stop:
        stop_file.touch()
    args = train.TrainConfig(
        env=ManagerBasedRlEnvCfg(
            decimation=25, scene=SceneCfg(num_envs=2), episode_length_s=4.0
        ),
        agent=RslRlOnPolicyRunnerCfg(
            max_iterations=5, num_steps_per_env=4, seed=19, logger="tensorboard"
        ),
        log_root=str(tmp_path / "output"),
        resume_checkpoint=str(source),
        stop_file=str(stop_file),
    )
    created = {}

    class Env:
        closed = False

        def __init__(self, cfg: object, **_kwargs: object) -> None:
            created["env"] = self
            self.cfg = cfg

        def close(self) -> None:
            self.closed = True

    class Runner:
        def __init__(
            self,
            _env: object,
            _cfg: dict,
            log_dir: str,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            created["runner"] = self
            self.output = Path(log_dir)
            self.current_learning_iteration = 0
            self.updates = []
            self.logger = SimpleNamespace(
                log=lambda **_kwargs: None, stop_logging_writer=lambda: None
            )

        def load(self, _path: str) -> None:
            self.current_learning_iteration = 2  # three completed native updates

        def learn(self, num_learning_iterations: int, **_kwargs: object) -> None:
            if fail:
                raise RuntimeError("learner failed")
            start = self.current_learning_iteration
            for it in range(start, start + num_learning_iterations):
                self.current_learning_iteration = it
                self.updates.append(it)
                self.logger.log(it=it)
            self.save(str(self.output / f"model_{self.current_learning_iteration}.pt"))

        def save(self, path: str) -> None:
            self._last_checkpoint = Path(path).with_suffix(".onnx")
            self._last_checkpoint.write_bytes(
                f"iteration {self.current_learning_iteration}".encode()
            )

    monkeypatch.setattr(
        sys, "argv", ["myo_train", "--config", str(preset), "--agent.seed", "19"]
    )
    monkeypatch.setattr(train, "register_mjlab_myointeract_task", lambda _cfg: None)
    monkeypatch.setattr(
        train.MjlabTrainConfig, "from_task", staticmethod(lambda _task: args)
    )
    monkeypatch.setattr(train.tyro, "cli", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(train, "configure_torch_backends", lambda: None)
    monkeypatch.setattr(train, "ManagerBasedRlEnv", Env)
    monkeypatch.setattr(train, "RslRlVecEnvWrapper", lambda env, **_kwargs: env)
    monkeypatch.setattr(train, "TrainingRunner", Runner)
    if timing_error:
        args.env.decimation = 10
        with pytest.raises(ValueError, match="captured task/action recipe"):
            train.main()
        assert not created and not (tmp_path / "output").exists()
        return {}, args.env, args
    elif fail:
        with pytest.raises(RuntimeError, match="learner failed"):
            train.main()
    else:
        train.main()
    files = list((tmp_path / "output").glob("*/*/lifecycle.json"))
    assert len(files) == 1
    assert source.read_bytes() == b"source unchanged"
    return json.loads(files[0].read_text()), created["env"], created["runner"]


def test_entrypoint_resumes_only_remaining_updates_and_writes_rebuild_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, env, runner = _exercise(tmp_path, monkeypatch)
    assert runner.updates == [3, 4]
    assert outcome["completed_iterations"] == 5 and outcome["status"] == "completed"
    assert env.closed and env.cfg.seed == 19
    assert (runner.output / "model_final.onnx").read_bytes() == b"iteration 4"
    assert (runner.output / "original_hydra_config.json").is_file()
    assert (
        json.loads((runner.output / "myointeract_config.json").read_text())["rl"][
            "num_timesteps"
        ]
        == 40
    )
    assert json.loads((runner.output / "agent_config.json").read_text())["seed"] == 19


def test_stop_before_next_update_keeps_the_restored_completed_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, env, runner = _exercise(tmp_path, monkeypatch, stop=True)
    assert runner.updates == [] and outcome["status"] == "stopped"
    assert outcome["completed_iterations"] == 3 and env.closed
    assert (runner.output / "model_stopped.onnx").read_bytes() == b"iteration 2"
    assert not (runner.output / "model_final.onnx").exists()


def test_failed_training_closes_env_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, env, runner = _exercise(tmp_path, monkeypatch, fail=True)
    assert env.closed and outcome["status"] == "failed"
    assert "learner failed" in outcome["error"]
    assert not (runner.output / "model_final.onnx").exists()


def test_native_timing_override_is_rejected_before_physics_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _exercise(tmp_path, monkeypatch, timing_error=True)


def test_registered_and_saved_training_recipe_are_identical(tmp_path, monkeypatch):
    from myoInteract.myointeract_hydra_cli import gradio_json_to_overrides

    monkeypatch.setattr(
        train,
        "training_task_overrides",
        lambda raw, argv: gradio_json_to_overrides(raw)
        + ["vision=enabled", "vision.recipe_version=1"],
    )
    compose = train.load_config_interactive
    selected = []

    def capture(overrides):
        config = compose(overrides)
        selected.append(config.vision.recipe_version)
        return config

    monkeypatch.setattr(train, "load_config_interactive", capture)
    _, _, runner = _exercise(tmp_path, monkeypatch)
    saved = json.loads((runner.output / "original_hydra_config.json").read_text())
    resolved = json.loads((runner.output / "myointeract_config.json").read_text())
    assert "vision.recipe_version=1" in saved
    assert resolved["vision"]["recipe_version"] == 1
    assert selected == [1, 1]
