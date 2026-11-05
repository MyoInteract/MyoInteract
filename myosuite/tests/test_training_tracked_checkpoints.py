"""Restore real bundled policies on CPU and verify legacy feature evidence."""

import copy
import json

import jax
import jax.numpy as jp
import numpy as np
import pytest
import mujoco
from mujoco import mjx
from brax.training.acme import running_statistics
from ml_collections import ConfigDict

from myosuite.envs.myo.myouser.myouser_universal import MyoUserUniversal
from myosuite.envs.myo.myouser.paths import get_bundled_checkpoint_dir
from myosuite.train.myouser.custom_ppo import checkpoint
from myosuite.train.myouser.custom_ppo import networks_vision_unified as networks
from myosuite.train.utils.train import (
    adapt_legacy_checkpoint_observations, resolve_checkpoint_path,
)


@pytest.mark.parametrize('run_name', ['ar', 'mobile-typing-5c-words', 'public-display-4-buttons'])
def test_real_tracked_policy_restores_with_its_verified_observation_schema(run_name, monkeypatch):
    root = get_bundled_checkpoint_dir() / run_name / 'checkpoints'
    config_path = root / 'config.json'
    source_bytes = config_path.read_bytes()
    original = ConfigDict(json.loads(source_bytes))
    config = copy.deepcopy(original)
    params = checkpoint.load(resolve_checkpoint_path(root))
    env = MyoUserUniversal(config.env)
    if run_name != 'ar':
        # These independent statistics prove the same five values were observed
        # twice; width alone would not justify changing the observation schema.
        for field in ('mean', 'std', 'summed_variance'):
            values = np.asarray(getattr(params[0], field))
            np.testing.assert_array_equal(values[44:49], values[49:54])
    migration = adapt_legacy_checkpoint_observations(env, config.env, params)
    if run_name == 'ar':
        assert migration is None
        expected_width = 50
    else:
        assert migration['kind'] == 'legacy_duplicated_omni_keys'
        assert migration['duplicated_keys'] == ['target_pos', 'target_size', 'phase']
        assert env.obs_keys[-6:] == ['target_pos', 'target_size', 'phase'] * 2
        assert config.env.task_config.obs_keys[-3:] == ['target_pos', 'target_size', 'phase']
        assert env._config.checkpoint_observation_migrations[0]['kind'] == migration['kind']
        expected_width = 54

    for name in ('policy_network', 'value_network'):
        layers = params[1]['params'][name]['layers']
        params[1]['params'][name]['layers'] = {int(key): value for key, value in layers.items()}
    policy_layers = params[1]['params']['policy_network']['layers']
    assert policy_layers[0]['kernel'].shape[0] == expected_width
    assert params[0].mean.shape == (expected_width,)
    network = networks.custom_network_factory(
        {'proprioception': (expected_width,)}, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        vision=False, **config.rl.network_factory,
    )
    policy = networks.make_inference_function_ppo_networks_unified(network)(params, deterministic=True)
    # Produce the actual environment observation from CPU MuJoCo forward data;
    # no full MJX dynamics rollout or expensive physics compilation is needed.
    cpu_data = mujoco.MjData(env.mj_model)
    mujoco.mj_forward(env.mj_model, cpu_data)
    reset_data = mjx.put_data(env.mj_model, cpu_data)
    monkeypatch.setattr(env, '_reset_bm_model', lambda key: reset_data)
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs['proprioception'].shape == (expected_width,)
    if migration:
        np.testing.assert_array_equal(
            state.obs['proprioception'][44:49], state.obs['proprioception'][49:54]
        )
    action, _ = jax.jit(policy)(state.obs, jax.random.PRNGKey(0))
    assert action.shape == (26,)
    assert np.isfinite(np.asarray(action)).all()
    assert config_path.read_bytes() == source_bytes
    assert original.to_dict() == json.loads(source_bytes)


def test_legacy_migration_rejects_width_without_matching_statistics():
    from types import SimpleNamespace

    env_cfg = ConfigDict({'task_config': {
        'obs_keys': ['qpos', 'qvel', 'qacc', 'ee_pos', 'act'],
        'omni_keys': ['target_pos', 'target_size', 'phase'],
    }})
    env = SimpleNamespace(_independent_qpos=list(range(6)), _independent_dofs=list(range(6)),
                          _na=23, obs_keys=list(env_cfg.task_config.obs_keys) + list(env_cfg.task_config.omni_keys))
    normalizer = running_statistics.init_state(jp.zeros(54))
    normalizer = normalizer.replace(count=jp.array(10.), mean=jp.arange(54.),
                                    summed_variance=jp.ones(54))
    with pytest.raises(ValueError, match='no verified legacy'):
        adapt_legacy_checkpoint_observations(env, env_cfg, (normalizer, {}))
