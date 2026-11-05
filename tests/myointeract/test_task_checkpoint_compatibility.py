import copy
import json
from pathlib import Path

import jax
import jax.numpy as jp
from ml_collections import ConfigDict
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from myosuite.envs.myo.myouser import myouser_universal as task_module
from myosuite.envs.myo.myouser.myouser_universal import MyoUserUniversal, UniversalTaskConfig


CHECKPOINTS = Path(__file__).resolve().parents[2] / 'tracked_checkpoints' / 'universal'


@pytest.mark.parametrize('name', ['ar', 'mobile-typing-5c-words', 'public-display-4-buttons'])
def test_stored_configs_construct_reset_and_step_without_mutation(name, monkeypatch):
    path = CHECKPOINTS / name / 'checkpoints' / 'config.json'
    original_bytes = path.read_bytes()
    source = ConfigDict(json.loads(original_bytes)['env'])
    original_config = copy.deepcopy(source.to_dict())
    env = MyoUserUniversal(source)
    defaults = UniversalTaskConfig()
    assert env._config.task_config.enable_extra_dist == defaults.enable_extra_dist
    assert env._config.task_config.exponential_reach_reward == defaults.exponential_reach_reward
    expected_weight = original_config['task_config']['weighted_reward_keys'].get('jac_effort', 0.)
    assert env._config.task_config.weighted_reward_keys.jac_effort == expected_weight
    # _setup adds recorded omniscient features once, only to the environment copy.
    expected_obs = original_config['task_config']['obs_keys'] + original_config['task_config']['omni_keys']
    assert list(env.obs_keys) == expected_obs
    assert source.to_dict() == original_config
    physical_data = mujoco.MjData(env.mj_model)
    mujoco.mj_forward(env.mj_model, physical_data)
    reset_data = mjx.put_data(env.mj_model, physical_data)
    monkeypatch.setattr(env, '_reset_bm_model', lambda rng: reset_data)
    state = jax.jit(env.reset)(jax.random.PRNGKey(23))
    assert state.obs['proprioception'].shape == ((50 if name == 'ar' else 49),)
    # Exercise real observation/reward/phase logic with stable physics here; the
    # benchmark smoke test exercises full MJX dynamics for all three tasks.
    monkeypatch.setattr(task_module.mjx_env, 'step', lambda model, data, ctrl, n_substeps: data)
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    assert np.isfinite(state.reward)
    assert np.isfinite(state.obs['proprioception']).all()
    assert list(env._config.task_config.obs_keys) == original_config['task_config']['obs_keys']
    replay = MyoUserUniversal(env._config)
    replay_obs, _ = replay.get_obs_vec(state.data, dict(state.info))
    assert list(replay.obs_keys) == expected_obs
    assert replay_obs['proprioception'].shape == state.obs['proprioception'].shape
    assert source.to_dict() == original_config
    assert path.read_bytes() == original_bytes


def test_explicit_legacy_flags_and_observations_are_preserved(monkeypatch):
    path = CHECKPOINTS / 'ar' / 'checkpoints' / 'config.json'
    config = ConfigDict(json.loads(path.read_text())['env'])
    config.task_config.enable_extra_dist = False
    config.task_config.exponential_reach_reward = True
    config.task_config.obs_keys = ['qpos']
    config.task_config.omni_keys = []
    config.lock()
    original = config.to_dict()
    # Inspect constructor normalization without allocating another physical model.
    monkeypatch.setattr(task_module.MyoUserBase, '__init__', lambda self, normalized, overrides: setattr(self, '_config', normalized))
    env = MyoUserUniversal(config)
    assert not env._config.task_config.enable_extra_dist
    assert env._config.task_config.exponential_reach_reward
    assert list(env._config.task_config.obs_keys) == ['qpos']
    assert list(env._config.task_config.omni_keys) == []
    assert config.to_dict() == original
