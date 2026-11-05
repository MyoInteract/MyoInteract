from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

from myosuite.envs.myo.myouser import myouser_universal as task_module
from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
from myosuite.envs.myo.myouser.myouser_universal import MyoUserUniversal, PointingTargetClass


@pytest.fixture(scope="module")
def environment():
    config = load_config_interactive([
        "env=universal", "wandb=disabled", "env/task_config/targets=single",
    ])
    return MyoUserUniversal(config.env)


@pytest.fixture(scope="module")
def reset_state(environment):
    return jax.jit(environment.reset)(jax.random.PRNGKey(3))


def test_acceleration_reward_counts_every_independent_joint_once(environment):
    count = len(environment._independent_dofs)
    assert count == 5  # The shipped arm has sparse full-model DOF indices.
    for index in range(count):
        acceleration = jp.zeros(count).at[index].set(100.)
        cost = environment.get_jac_effort_costs({
            "last_ctrl": jp.zeros(environment.action_size), "qacc": acceleration,
        })
        assert float(cost) == pytest.approx(0.0667)


@pytest.mark.parametrize("target_count", [1, 3])
def test_evaluation_includes_successful_final_target(target_count):
    env = object.__new__(MyoUserUniversal)
    env.target_objs = [None] * target_count
    info = {}
    metrics = {f"target_{i}_completed": 1. for i in range(target_count - 1)}
    metrics["target_final_completed"] = 1.
    for index in range(target_count):
        info[f"target_{index}/target_size"] = 0.1
        info[f"target_{index}/target_pos"] = jp.zeros(3)
    successful = SimpleNamespace(info=info, metrics=metrics)
    assert all(result["success"] for result in env.eval_metrics([[successful]]))
    successful.metrics = {**metrics, "target_final_completed": 0.}
    assert not env.eval_metrics([[successful]])[-1]["success"]


def test_reset_observation_does_not_accumulate_dwell(environment, reset_state):
    info = dict(reset_state.info)
    info["target_0/target_pos"] = reset_state.data.site_xpos[environment.ee_pos_id]
    info["target_0/target_size"] = jp.array(0.1)
    observation = environment.get_obs_dict(reset_state.data, info)
    assert float(observation["steps_inside_target"]) == 0.
    assert float(observation["phase_completed"]) == 0.
    assert float(observation["dwell_fraction"]) == -1.


@pytest.mark.parametrize("first_target_type", ["sphere", "button"])
def test_transition_returns_new_target_and_resets_dwell(environment, reset_state, monkeypatch, first_target_type):
    weights = environment._config.task_config.weighted_reward_keys
    targets = [PointingTargetClass(
        i, 2, [[-1, -1, -1], [1, 1, 1]], [0.01, 0.15], 3, jp.zeros(3), weights,
        enable_extra_dist=False,
    ) for i in range(2)]
    data = reset_state.data
    if first_target_type == "button":
        config = task_module.ButtonTarget(position=[0, 0, 0])
        fields = ("position", "geom_size", "site_size", "site_pos", "geom_margin", "euler", "min_touch_force")
        targets[0] = task_module.ButtonTargetClass(
            0, 2, **{key: jp.asarray(getattr(config, key)) for key in fields},
            weighted_reward_keys=weights, enable_extra_dist=False,
        )
        targets[0].sensor_id = 0
        data = data.replace(sensordata=jp.array([2.]))
    monkeypatch.setattr(environment, "target_objs", targets)
    monkeypatch.setattr(environment, "num_targets", 2)
    monkeypatch.setattr(environment, "get_ctrl", lambda state, action, rng: state.data.ctrl)
    monkeypatch.setattr(task_module.mjx_env, "step", lambda model, data, ctrl, n_substeps: data)
    fingertip = reset_state.data.site_xpos[environment.ee_pos_id]
    info = dict(reset_state.info)
    info.update({
        "target_0/target_pos": fingertip + jp.array([0.1, 0, 0]),
        "target_0/target_size": jp.array(0.15),
        "target_1/target_pos": fingertip,
        "target_1/target_size": jp.array(0.01),
        "steps_inside_target": jp.array(2.),
        "num_phases": jp.array(2),
    })
    state = reset_state.replace(info=info, data=data)
    step = jax.jit(environment.step)
    state = step(state, jp.zeros(environment.action_size))
    assert int(state.info["phase"]) == 1
    assert float(state.info["steps_inside_target"]) == 0.
    np.testing.assert_allclose(state.obs["proprioception"][-6:-3], fingertip)
    assert float(state.obs["proprioception"][-3]) == pytest.approx(0.01)
    assert float(state.obs["proprioception"][-2]) == 0.
    assert float(state.obs["proprioception"][-1]) == -1.
    state = step(state, jp.zeros(environment.action_size))
    assert not float(state.done)
    assert float(state.info["steps_inside_target"]) == 1.
    state = step(state, jp.zeros(environment.action_size))
    assert not float(state.done)
    state = step(state, jp.zeros(environment.action_size))
    assert float(state.done) == 1.


def test_visual_target_inputs_match_episode_without_mutating_physics(environment, reset_state):
    target = environment.target_objs[0]
    info = dict(reset_state.info)
    info["target_0/target_pos"] = jp.array([0.1, 0.2, 0.3])
    info["target_0/target_size"] = jp.array(0.07)
    original_size = np.array(environment.mjx_model.geom_size)
    original_positions = np.array(reset_state.data.geom_xpos)
    render_data, render_model = environment.task_render_inputs(reset_state.data, info)
    np.testing.assert_allclose(render_data.geom_xpos[target.target_geom_id], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(render_model.geom_size[target.target_geom_id], [0.07] * 3)
    np.testing.assert_array_equal(environment.mjx_model.geom_size, original_size)
    np.testing.assert_array_equal(reset_state.data.geom_xpos, original_positions)


@pytest.mark.parametrize("mode", ["rgb", "rgbd", "rgb+depth", "rgbd_only", "depth", "depth_only", "depth_w_aux_task"])
def test_vision_reset_step_and_auxiliary_targets_use_fresh_scene(environment, reset_state, monkeypatch, mode):
    target_id = environment.target_objs[0].target_geom_id

    class Renderer:
        def init(self, data, model):
            value = (model.geom_size[target_id, 0] * 255).astype(jp.uint8)
            rgb = jp.full((1, 4, 4, 4), value, dtype=jp.uint8)
            depth = jp.full((1, 4, 4, 1), jp.linalg.norm(data.geom_xpos[target_id]))
            return jp.array(True), rgb, depth

        def render(self, token, data):
            raise AssertionError("Dynamic target sizes/colors require refreshed model buffers.")

    monkeypatch.setattr(environment, "vision", True)
    monkeypatch.setattr(environment, "vision_mode", mode, raising=False)
    monkeypatch.setattr(environment, "batch_renderer", Renderer(), raising=False)
    monkeypatch.setattr(environment, "_reset_bm_model", lambda rng: reset_state.data)
    monkeypatch.setattr(task_module.mjx_env, "step", lambda model, data, ctrl, n_substeps: data)
    state = jax.jit(lambda rng: environment.reset(rng))(jax.random.PRNGKey(11))
    assert "render_token" in state.info
    state = jax.jit(lambda state, action: environment.step(state, action))(state, jp.zeros(environment.action_size))
    key = "pixels/view_0" if mode in {"rgb", "rgbd", "rgbd_only"} else "pixels/depth"
    assert key in state.obs
    assert np.isfinite(state.obs[key]).all()
    if mode == "rgb":
        expected = float((state.info["target_0/target_size"] * 255).astype(jp.uint8)) / 255
        np.testing.assert_allclose(state.obs[key], expected)
    else:
        depth = state.obs[key][..., -1:] if mode in {"rgbd", "rgbd_only"} else state.obs[key]
        np.testing.assert_allclose(depth, np.linalg.norm(state.info["target_0/target_pos"]), rtol=1e-6)
    if mode == "depth_w_aux_task":
        np.testing.assert_allclose(state.obs["vision_aux_targets"][:3], state.info["target_pos"])
        assert float(state.obs["vision_aux_targets"][3]) == pytest.approx(float(state.info["target_size"]))
