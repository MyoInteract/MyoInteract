"""Exercise project PPO/network/checkpoint modules, rather than upstream copies."""

import functools

import jax
import numpy as np
import pytest

from myosuite.tests.test_training_wrappers import ToyEnv
from myosuite.train.myouser.custom_ppo import checkpoint
from myosuite.train.myouser.custom_ppo import networks_vision_unified as networks
from myosuite.train.myouser.custom_ppo import train as ppo
from myosuite.train.utils.wrapper import wrap_myosuite_training


def network_factory():
    return functools.partial(
        networks.custom_network_factory,
        policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
    )


def train_toy(**overrides):
    kwargs = dict(
        environment=ToyEnv(end_after=3), num_timesteps=64, num_envs=4,
        episode_length=3, batch_size=4, num_minibatches=1,
        unroll_length=2, num_updates_per_batch=1, num_evals=0,
        num_checkpoints=0, normalize_observations=True,
        network_factory=network_factory(), wrap_env_fn=wrap_myosuite_training,
        seed=4,
    )
    kwargs.update(overrides)
    return ppo.train(**kwargs)


def test_project_ppo_performs_an_update_with_finite_params():
    make_policy, params, _ = train_toy()
    leaves = jax.tree_util.tree_leaves(params)
    assert leaves
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
    observation = ToyEnv().reset(jax.random.PRNGKey(0)).obs
    action, _ = make_policy(params, deterministic=True)(observation, jax.random.PRNGKey(1))
    assert action.shape == (2,)
    assert np.isfinite(np.asarray(action)).all()


def test_project_checkpoint_roundtrip_preserves_deterministic_policy(tmp_path):
    make_policy, params, _ = train_toy()
    config = checkpoint.network_config(
        observation_size={'proprioception': (3,)}, action_size=2,
        normalize_observations=True, network_factory=network_factory(),
    )
    checkpoint.save(tmp_path, 64, params, config)
    restore_path = tmp_path / '000000000064'
    restored_policy, restored_params, _ = train_toy(
        num_timesteps=0, restore_checkpoint_path=restore_path,
    )
    observation = ToyEnv().reset(jax.random.PRNGKey(10)).obs
    key = jax.random.PRNGKey(20)
    original_action = make_policy(params, deterministic=True)(observation, key)[0]
    restored_action = restored_policy(restored_params, deterministic=True)(observation, key)[0]
    np.testing.assert_allclose(original_action, restored_action, rtol=0, atol=0)


def test_checkpoint_restore_rejects_an_unknown_observation_width(monkeypatch):
    _, params, _ = train_toy(num_timesteps=0)
    unknown_normalizer = params[0].replace(mean=jax.numpy.zeros(4))
    monkeypatch.setattr(checkpoint, 'load', lambda path: (unknown_normalizer, params[1]))
    with pytest.raises(ValueError, match='Checkpoint observation shape does not match'):
        train_toy(num_timesteps=0, restore_checkpoint_path='unknown-shape')
