"""Sequence state through real mjlab reset, termination, reward and observation hooks."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

import mujoco  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.envs.mdp import dr, events  # noqa: E402
from mjlab.managers import EventTermCfg, RewardTermCfg  # noqa: E402
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjlab.scene import SceneCfg  # noqa: E402
from mjlab.sim import MujocoCfg, SimulationCfg  # noqa: E402

from myoInteract.register_mjlab_myointeract_tasks import (  # noqa: E402
    SequentialTaskLogic,
    SequentialTaskSuccess,
    TaskEntityCfg,
    _dwell_fraction,
    _next_target_pos,
    _phase_bonus,
    _phase_progress,
    _phase_successfully_completed,
    _sequential_distance_reward,
    _sequential_task_time_out,
)


@pytest.fixture
def make_env() -> Iterator:
    opened = []

    def build(
        dwell_steps: tuple[int, ...],
        *,
        button: bool = False,
        max_steps: int = 10,
        auto_reset: bool = False,
    ) -> ManagerBasedRlEnv:
        targets = {}
        elements = []
        sensors = []
        for i, dwell in enumerate(dwell_steps):
            is_button = button and i == len(dwell_steps) - 1
            targets[f"target_{i}"] = SimpleNamespace(
                name="button_target" if is_button else "pointing_target",
                dwell_duration=dwell * 0.01,
                min_touch_force=1.0 if is_button else 0.0,
            )
            shape = (
                'type="box" size="0.1 0.1 0.1"'
                if is_button
                else 'type="sphere" size="0.1"'
            )
            collision = 1 if is_button else 0
            elements.append(f"""
                <body name="body_target_{i}" pos="{i} 0 0">
                  <geom name="geom_target_{i}" {shape} contype="{collision}" conaffinity="{collision}"/>
                  <site name="site_target_{i}" type="box" size="0.11 0.11 0.11"/>
                </body>""")
            sensors.append(f'<touch name="sensor_target_{i}" site="site_target_{i}"/>')
        xml = f"""<mujoco><worldbody>
            <body name="finger">
              <joint name="finger_joint" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.01" mass="0.1"/>
              <site name="fingertip"/>
            </body>{"".join(elements)}
            </worldbody><sensor>{"".join(sensors)}</sensor></mujoco>"""
        base = SimpleNamespace(
            env=SimpleNamespace(
                ctrl_dt=0.01,
                task_config=SimpleNamespace(
                    targets=SimpleNamespace(num_targets=len(dwell_steps), **targets),
                ),
            )
        )
        selection = SceneEntityCfg("universal_robot", site_names=["fingertip"])
        cfg = ManagerBasedRlEnvCfg(
            scene=SceneCfg(
                num_envs=2,
                entities={
                    "universal_robot": TaskEntityCfg(
                        spec_fn=lambda: mujoco.MjSpec.from_string(xml),
                        base_config=base,
                    )
                },
            ),
            sim=SimulationCfg(
                mujoco=MujocoCfg(timestep=0.01, gravity=(0, 0, 0)), njmax=32, nconmax=16
            ),
            decimation=1,
            episode_length_s=max_steps * 0.01,
            auto_reset=auto_reset,
            scale_rewards_by_dt=False,
            events={
                "reset_scene": EventTermCfg(
                    func=events.reset_scene_to_default, mode="reset"
                ),
                "task_logic": EventTermCfg(
                    func=SequentialTaskLogic,
                    mode="reset",
                    params={"asset_cfg": selection},
                ),
                "target_size": EventTermCfg(
                    func=dr.geom_size,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "universal_robot", geom_names=["geom_target_.*"]
                        ),
                        "ranges": {
                            f"geom_target_{i}": {axis: (0.1, 0.1) for axis in range(3)}
                            for i in range(len(dwell_steps))
                        },
                        "operation": "abs",
                    },
                ),
            },
            observations={
                "task": ObservationGroupCfg(
                    terms={
                        "phase": ObservationTermCfg(
                            func=_phase_progress, params={"asset_cfg": selection}
                        ),
                        "dwell": ObservationTermCfg(
                            func=_dwell_fraction, params={"asset_cfg": selection}
                        ),
                        "target": ObservationTermCfg(
                            func=_next_target_pos, params={"asset_cfg": selection}
                        ),
                    }
                )
            },
            rewards={
                "distance": RewardTermCfg(
                    func=_sequential_distance_reward,
                    weight=1,
                    params={"asset_cfg": selection},
                ),
                "phase": RewardTermCfg(
                    func=_phase_bonus, weight=10, params={"asset_cfg": selection}
                ),
                "done": RewardTermCfg(
                    func=_phase_successfully_completed,
                    weight=100,
                    params={
                        "asset_cfg": selection,
                        "phase_id": len(dwell_steps) - 1,
                    },
                ),
            },
            terminations={
                "time_out": TerminationTermCfg(
                    func=_sequential_task_time_out, time_out=True
                ),
                "episode_success": TerminationTermCfg(func=SequentialTaskSuccess),
            },
        )
        env = ManagerBasedRlEnv(cfg, device="cpu")
        opened.append(env)
        env.reset()
        return env

    yield build
    for env in opened:
        env.close()


def _step(env: ManagerBasedRlEnv):
    return env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))


@pytest.mark.parametrize("auto_reset", [False, True])
def test_success_and_rewards_use_the_current_step_at_the_timeout_boundary(
    make_env, tmp_path, auto_reset
) -> None:
    env = make_env((1,), max_steps=1, auto_reset=auto_reset)
    obs, reward, terminated, truncated, extras = _step(env)
    assert terminated.tolist() == [True, True]
    assert truncated.tolist() == [False, False]
    assert env.termination_manager.get_term("episode_success").tolist() == [True, True]
    torch.testing.assert_close(reward, torch.full((2,), 110.0))
    torch.testing.assert_close(
        obs["task"][:, 0], torch.full((2,), -1.0 if auto_reset else 1.0)
    )
    if auto_reset:
        from myoInteract.studio.runs import metrics  # noqa: PLC0415

        log = extras["log"]
        assert log["Episode_Termination/episode_success"] == 2
        assert log["Episode_Termination/time_out"] == 0
        scalars = {key: [(0, float(value))] for key, value in log.items()}
        assert metrics(tmp_path, scalars, env.step_dt)["points"][0]["success"] == 1.0


@pytest.mark.filterwarnings(
    "ignore:Initializing zero-element tensors is a no-op:UserWarning"
)
def test_native_ppo_bootstraps_only_unfinished_deadline_episodes(make_env) -> None:
    from mjlab.rl import RslRlVecEnvWrapper  # noqa: PLC0415
    from rsl_rl.algorithms import PPO  # noqa: PLC0415
    from rsl_rl.models import MLPModel  # noqa: PLC0415
    from rsl_rl.storage import RolloutStorage  # noqa: PLC0415

    env = make_env((1,), max_steps=1, auto_reset=True)
    wrapped = RslRlVecEnvWrapper(env)
    asset = env.scene["universal_robot"]
    asset.write_joint_state_to_sim(torch.tensor([[0.0], [0.5]]), torch.zeros(2, 1))
    env.sim.forward()
    observations = wrapped.get_observations()
    groups = {"actor": ["task"], "critic": ["task"]}
    # This physics fixture has no actuators; native RSL supports an empty action vector.
    actor = MLPModel(
        observations,
        groups,
        "actor",
        wrapped.num_actions,
        hidden_dims=(4,),
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.3,
            "std_type": "scalar",
        },
    )
    critic = MLPModel(observations, groups, "critic", 1, hidden_dims=(4,))
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        list(critic.parameters())[-1].fill_(2.0)
    storage = RolloutStorage("rl", 2, 1, observations, [wrapped.num_actions])
    ppo = PPO(actor, critic, storage, gamma=0.9)
    with torch.inference_mode():
        actions = ppo.act(observations)
        observations, reward, dones, extras = wrapped.step(actions)
        assert dones.tolist() == [1, 1]
        assert extras["time_outs"].tolist() == [False, True]
        assert extras["log"]["Episode_Termination/episode_success"] == 1
        assert extras["log"]["Episode_Termination/time_out"] == 1
        ppo.process_env_step(observations, reward, dones, extras)
    torch.testing.assert_close(
        storage.rewards[0, :, 0], reward + torch.tensor([0.0, 1.8])
    )


def test_phase_transition_reward_and_observation_refer_to_the_next_target(
    make_env,
) -> None:
    env = make_env((1, 3))
    obs, reward, terminated, _, _ = _step(env)
    asset = env.scene["universal_robot"]
    assert not terminated.any()
    torch.testing.assert_close(reward, torch.full((2,), 9.0))
    torch.testing.assert_close(asset.distance_to_target, torch.ones(2, 1))
    torch.testing.assert_close(
        obs["task"][:, :2], torch.tensor([[0.0, -1.0], [0.0, -1.0]])
    )
    torch.testing.assert_close(
        obs["task"][:, 2:], torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    # Native consumers may re-read termination/rewards without advancing time.
    env.termination_manager.compute()
    assert asset.current_target_id.tolist() == [[1], [1]]
    assert asset.steps_inside_target.tolist() == [[0], [0]]


def test_partial_reset_restores_first_dwell_and_clears_only_reset_slots(
    make_env,
) -> None:
    env = make_env((1, 0), button=True)
    _step(env)
    asset = env.scene["universal_robot"]
    assert asset.current_target_id.tolist() == [[1], [1]]
    env.reset(env_ids=torch.tensor([0]))
    assert asset.current_target_id.tolist() == [[0], [1]]
    assert asset.target_dwell_steps_batch.tolist() == [[1.0], [0.0]]
    assert asset.phase_successfully_completed.tolist() == [[False], [True]]
    # Move the reset fingertip outside target 0: the preceding button's zero
    # threshold must not complete it on this episode's first control step.
    asset.write_joint_state_to_sim(torch.full((2, 1), 0.5), torch.zeros(2, 1))
    env.sim.forward()
    _step(env)
    assert asset.current_target_id[0, 0].item() == 0
    assert not asset.phase_successfully_completed[0, 0].item()


def test_mixed_target_touch_completion_and_autoreset_restore_first_phase(
    make_env,
) -> None:
    env = make_env((1, 0), button=True, auto_reset=True)
    _step(env)
    asset = env.scene["universal_robot"]
    asset.write_joint_state_to_sim(torch.full((2, 1), 0.91), torch.zeros(2, 1))
    env.sim.forward()
    _, reward, terminated, _, _ = _step(env)
    assert terminated.tolist() == [True, True]
    assert torch.all(reward > 100)
    assert asset.current_target_id.tolist() == [[0], [0]]
    assert asset.target_dwell_steps_batch.tolist() == [[1.0], [1.0]]
    assert asset.phase_successfully_completed.tolist() == [[False], [False]]


def test_zero_dwell_sphere_requires_entry(make_env) -> None:
    env = make_env((0,))
    asset = env.scene["universal_robot"]
    asset.write_joint_state_to_sim(torch.ones(2, 1), torch.zeros(2, 1))
    env.sim.forward()
    assert not _step(env)[2].any()


def test_exponential_distance_reward_is_one_scalar_per_environment() -> None:
    asset = SimpleNamespace(
        num_targets=1,
        distance_to_target=torch.tensor([[1.0], [2.0]]),
        inside_target=torch.tensor([[False], [True]]),
    )
    env = SimpleNamespace(scene={"universal_robot": asset}, num_envs=2)
    reward = _sequential_distance_reward(env, exponential_distance_reward=True)
    assert reward.shape == (2,)
    np.testing.assert_allclose(reward.numpy(), [np.expm1(-10) / 10, 0], atol=1e-7)
