"""Small real RSL PPO update and native ONNX/optimizer continuation on CPU."""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from myosuite.core.muscle_conditions import TorchFatigueState
from tensordict import TensorDict

from myoInteract.scripts.train import TrainingRunner
from myoInteract.studio.worker import load_weights
from myoInteract.training_lifecycle import (
    BoundaryLogger,
    TrainingStopped,
    remaining_iterations,
)


class _Env:
    """A CPU tensor environment; the learner, optimizer and ONNX IO are native."""

    num_envs = 2
    num_actions = 1
    max_episode_length = 10
    device = "cpu"

    def __init__(self) -> None:
        self.cfg = {}
        self.unwrapped = self
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.obs = torch.tensor([[0.1, -0.2], [-0.3, 0.4]])

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"proprioception": self.obs.clone()}, batch_size=[self.num_envs]
        )

    def step(self, actions: torch.Tensor):
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self.obs += 0.01
        return (
            self.get_observations(),
            -actions.square().sum(-1),
            torch.zeros(self.num_envs, dtype=torch.long),
            {},
        )


def _config() -> dict:
    return asdict(
        RslRlOnPolicyRunnerCfg(
            actor=RslRlModelCfg(
                hidden_dims=(8,),
                obs_normalization=True,
                distribution_cfg={
                    "class_name": "GaussianDistribution",
                    "init_std": 0.3,
                    "std_type": "scalar",
                },
            ),
            critic=RslRlModelCfg(hidden_dims=(8,), obs_normalization=True),
            algorithm=RslRlPpoAlgorithmCfg(
                num_learning_epochs=1, num_mini_batches=1, schedule="fixed"
            ),
            obs_groups={"actor": ("proprioception",), "critic": ("proprioception",)},
            num_steps_per_env=4,
            max_iterations=3,
            save_interval=10,
            logger="tensorboard",
            upload_model=False,
        )
    )


def test_native_optimizer_continuation_advances_the_next_iteration(
    tmp_path: Path,
) -> None:
    torch.manual_seed(4)
    runner = TrainingRunner(_Env(), _config(), str(tmp_path / "first"), "cpu")
    initial = {
        key: value.clone() for key, value in runner.alg.actor.state_dict().items()
    }
    boundaries = []

    def stop_after_update(completed: int) -> None:
        boundaries.append(completed)
        raise TrainingStopped()

    runner.logger = BoundaryLogger(runner.logger, stop_after_update)
    with pytest.raises(TrainingStopped):
        runner.learn(num_learning_iterations=3)
    assert boundaries == [1] and runner.current_learning_iteration == 0
    assert any(
        not torch.equal(initial[key], value)
        for key, value in runner.alg.actor.state_dict().items()
    )
    source = tmp_path / "first" / "model_stopped.onnx"
    source_fatigue = TorchFatigueState(num_envs=2, n_muscles=1, device="cpu")
    source_fatigue.MF[:] = 0.4
    source_fatigue.MR[:] = 0.6
    source_term = SimpleNamespace(_fatigue=source_fatigue)
    runner.env.action_manager = SimpleNamespace(
        _terms={"muscles": source_term}, get_term=lambda _name: source_term
    )
    runner.save(str(source.with_suffix(".pt")))
    runner.logger.stop_logging_writer()

    inference = TrainingRunner(_Env(), _config(), None, "cpu")
    inference.env.common_step_counter = 17
    critic_before = {
        key: value.clone() for key, value in inference.alg.critic.state_dict().items()
    }
    policy = load_weights(inference, source, "cpu")
    assert inference.env.common_step_counter == 17
    assert not inference.alg.optimizer.state_dict()["state"]
    assert all(
        torch.equal(value, inference.alg.critic.state_dict()[key])
        for key, value in critic_before.items()
    )
    with torch.no_grad():
        observations = inference.env.get_observations()
        actions = policy(observations)
        expected = runner.get_inference_policy(device="cpu")(observations)
        assert torch.isfinite(actions).all() and torch.equal(actions, expected)

    destination = _Env()
    destination.num_envs = 3
    destination.episode_length_buf = torch.zeros(3, dtype=torch.long)
    destination.obs = torch.tensor([[0.1, -0.2], [-0.3, 0.4], [0.5, -0.6]])
    destination_fatigue = TorchFatigueState(num_envs=3, n_muscles=1, device="cpu")
    destination_term = SimpleNamespace(_fatigue=destination_fatigue)
    destination.action_manager = SimpleNamespace(
        _terms={"muscles": destination_term}, get_term=lambda _name: destination_term
    )
    restored = TrainingRunner(destination, _config(), str(tmp_path / "second"), "cpu")
    restored.load(str(source))
    assert (
        destination_fatigue.MF.shape == (3, 1)
        and torch.count_nonzero(destination_fatigue.MF) == 0
    )
    assert restored.current_learning_iteration == 0
    for key, value in runner.alg.actor.state_dict().items():
        assert torch.equal(value, restored.alg.actor.state_dict()[key])
    # Adam moments and update counts come from RSL's own checkpoint payload.
    saved_opt = runner.alg.optimizer.state_dict()["state"]
    restored_opt = restored.alg.optimizer.state_dict()["state"]
    assert saved_opt.keys() == restored_opt.keys() and saved_opt
    for parameter, state in saved_opt.items():
        for key, value in state.items():
            assert torch.equal(value, restored_opt[parameter][key])
    completed = restored.current_learning_iteration + 1
    restored.current_learning_iteration = completed
    after = []
    restored.logger = BoundaryLogger(restored.logger, after.append)
    restored.learn(num_learning_iterations=remaining_iterations(3, completed))
    restored.logger.stop_logging_writer()
    assert after == [2, 3]
    assert restored.current_learning_iteration == 2
    assert source.is_file() and (tmp_path / "second" / "model_2.onnx").is_file()
