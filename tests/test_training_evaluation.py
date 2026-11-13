"""Evaluation latches first-episode metrics before native manual resets."""
# ruff: noqa: E402

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myoInteract.studio import worker


class _ManualEnv:
    num_envs = 2
    max_episode_length = 3
    step_dt = 0.1

    def __init__(self) -> None:
        self.age = torch.zeros(2, dtype=torch.long)
        self.pending = torch.zeros(2, dtype=torch.bool)
        self.muscles = SimpleNamespace(
            processed_action=torch.zeros(2, 1),
            _fatigue=SimpleNamespace(MF=torch.zeros(2, 1)),
        )
        self.success = torch.zeros(2, dtype=torch.bool)
        self.action_manager = SimpleNamespace(get_term=lambda _name: self.muscles)
        self.termination_manager = SimpleNamespace(get_term=lambda _name: self.success)
        # targets reached this episode: one per step, of two
        self.robot = SimpleNamespace(
            num_targets=2, current_target_id=torch.zeros(2, 1, dtype=torch.long)
        )
        self.scene = {"universal_robot": self.robot}
        self.closed = False

    def reset(
        self, *, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        ids = slice(None) if env_ids is None else env_ids
        self.age[ids] = 0
        self.pending[ids] = False
        self.muscles.processed_action[ids] = 0
        self.muscles._fatigue.MF[ids] = 0
        self.robot.current_target_id[ids] = 0
        return self.get_observations(), {}

    def get_observations(self) -> torch.Tensor:
        return self.age.float().unsqueeze(-1)

    def step(
        self, _action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        assert (
            not self.pending.any()
        ), "native manual-reset environments must reset before another step"
        self.age += 1
        self.muscles.processed_action[:] = 0.8
        self.muscles._fatigue.MF[:] = self.age.unsqueeze(-1).float() / 10
        self.robot.current_target_id[:, 0] = self.age.clamp(max=2)
        done = self.age >= torch.tensor([1, 3])
        self.pending[:] = done
        self.success[:] = done
        return self.get_observations(), torch.zeros(2), done.long(), {}

    def close(self) -> None:
        self.closed = True


def test_first_episode_terminal_effort_and_fatigue_survive_other_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _ManualEnv()
    config = SimpleNamespace(vision=SimpleNamespace(enabled=False))

    def build(_overrides: list, episodes: int, _device: str, **kwargs: object):
        assert episodes == 2 and kwargs == {"auto_reset": False, "seed": 7}
        return env, config

    monkeypatch.setattr(worker, "build_env", build)
    monkeypatch.setattr(
        worker.torch.cuda, "is_available", lambda: False
    )  # the fake env lives on the CPU
    monkeypatch.setattr(worker, "run_overrides", lambda _path: [])
    monkeypatch.setattr(worker, "wrap", lambda value: value)
    monkeypatch.setattr(
        worker, "load_policy", lambda *_args: lambda obs: torch.zeros(len(obs), 1)
    )
    args = SimpleNamespace(
        log_dir=tmp_path,
        episodes=2,
        checkpoint="model_final.onnx",
        seed=7,
        video_seconds=0,
        width=32,
        height=32,
        out=tmp_path,
    )
    result = worker.evaluate(args)
    assert result["success_rate"] == 1.0
    assert result["time_to_success_s"] == pytest.approx(0.2)
    assert result["mean_excitation"] == pytest.approx(0.8)
    # Env 0 has been reset twice while waiting for env 1; its first terminal
    # fatigue was 0.1. Reading the final live tensor would lose that episode.
    assert result["final_mf"] == pytest.approx(0.2)
    # Env 0 reached one target in its first episode, env 1 both.
    assert result["targets_reached"] == pytest.approx([1.0, 0.5])
    episodes = result["per_episode"]
    assert episodes["targets"] == [1, 2]
    assert episodes["time_s"] == pytest.approx([0.1, 0.3])
    assert episodes["final_mf"] == pytest.approx([0.1, 0.3])
    assert sum(episodes["excitation"]) / 2 == pytest.approx(0.8)
    assert env.closed


def test_evaluation_closes_environment_if_policy_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _ManualEnv()
    monkeypatch.setattr(worker, "build_env", lambda *_args, **_kwargs: (env, None))
    monkeypatch.setattr(worker, "run_overrides", lambda _path: [])
    monkeypatch.setattr(worker, "wrap", lambda value: value)

    def fail(*_args: object) -> None:
        raise ValueError("incompatible policy")

    monkeypatch.setattr(worker, "load_policy", fail)
    args = SimpleNamespace(
        log_dir=tmp_path, episodes=2, checkpoint="bad.onnx", seed=0, video_seconds=0
    )
    with pytest.raises(ValueError, match="incompatible policy"):
        worker.evaluate(args)
    assert env.closed


def test_renderer_transfers_only_environment_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from myoInteract.mjlab_offscreen_renderer import _sync_render_data

    transfers = []
    original_cpu = torch.Tensor.cpu

    def cpu(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        transfers.append(tuple(value.shape))
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    shapes = {
        "xpos": (2, 3),
        "xquat": (2, 4),
        "xmat": (2, 3, 3),
        "geom_xpos": (3, 3),
        "geom_xmat": (3, 3, 3),
        "site_xpos": (1, 3),
        "site_xmat": (1, 3, 3),
    }
    values = {
        name: torch.arange(16 * np.prod(shape), dtype=torch.float32).reshape(16, *shape)
        for name, shape in shapes.items()
    }
    render_values = {
        name: np.zeros((shape[0], 9) if name.endswith("xmat") else shape)
        for name, shape in shapes.items()
    }
    rendered = SimpleNamespace(**render_values)
    model = SimpleNamespace(nsite=1, ntendon=0, geom_rgba=np.zeros((3, 4)))
    sim = SimpleNamespace(
        data=SimpleNamespace(**values),
        mj_model=model,
        model=SimpleNamespace(geom_rgba=torch.zeros(16, 3, 4)),
    )
    assert _sync_render_data(sim, rendered) is rendered
    assert all(shape[0] != 16 for shape in transfers)
    assert rendered.xpos.tolist() == values["xpos"][0].numpy().tolist()
