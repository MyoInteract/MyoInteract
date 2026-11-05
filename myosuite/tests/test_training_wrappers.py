"""Regression tests for the wrappers actually used by MyoInteract PPO."""

import jax
import jax.numpy as jp
import numpy as np
import pytest
import mujoco
from mujoco import mjx
from jax.interpreters import batching
from mujoco_playground._src.mjx_env import State

from myosuite.train.utils.wrapper import (
    EvalVmapWrapper, _maybe_wrap_env_for_evaluation, wrap_myosuite_training,
)


class ToyEnv:
    """Small pure JAX environment with observable episode state."""

    action_size = 2
    non_accumulation_metrics = ['distance']
    vision = False

    def __init__(self, end_after=2):
        self.end_after = end_after

    @property
    def unwrapped(self):
        return self

    def reset(self, rng):
        rng, target_key = jax.random.split(rng)
        target = jax.random.uniform(target_key)
        return State(
            data=jp.array(0.),
            obs={'proprioception': jp.array([0., 0., target])},
            reward=jp.array(0.),
            done=jp.array(0.),
            metrics={'cost': jp.array(0.), 'distance': jp.array(0.)},
            info={'rng': rng, 'phase': jp.array(0), 'target': target},
        )

    def eval_reset(self, rng, eval_id):
        return self.reset(jax.random.fold_in(rng, eval_id))

    def auto_reset(self, state):
        return self.reset(state.info['rng'])

    def step(self, state, action):
        del action
        count = state.data + 1
        done = (count >= self.end_after).astype(jp.float32)
        info = dict(state.info)
        info['phase'] = info['phase'] + 1
        obs = {'proprioception': jp.array([count, info['phase'], info['target']])}
        return state.replace(
            data=count, obs=obs, reward=jp.where(done, 7., 1.), done=done,
            metrics={'cost': jp.array(2.), 'distance': count}, info=info,
        )


def test_autoreset_preserves_terminal_reward_and_fresh_episode_info():
    env = wrap_myosuite_training(ToyEnv(), episode_length=10)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
    original_targets = state.info['target']
    original_rng = state.info['rng']
    step = jax.jit(env.step)
    state = step(state, jp.zeros((2, 2)))
    state = step(state, jp.zeros((2, 2)))
    np.testing.assert_array_equal(state.reward, [7., 7.])
    np.testing.assert_array_equal(state.done, [1., 1.])
    np.testing.assert_array_equal(state.data, [0., 0.])
    np.testing.assert_array_equal(state.info['phase'], [0, 0])
    np.testing.assert_array_equal(state.obs['proprioception'][:, 2], state.info['target'])
    assert not np.array_equal(original_targets, state.info['target'])
    assert not np.array_equal(original_rng, state.info['rng'])
    np.testing.assert_array_equal(state.info['episode_metrics']['sum_reward'], [8., 8.])
    np.testing.assert_array_equal(state.info['episode_metrics']['length'], [2., 2.])
    np.testing.assert_array_equal(state.info['episode_metrics']['cost'], [4., 4.])
    np.testing.assert_array_equal(state.info['episode_metrics']['distance'], [2., 2.])

    state = step(state, jp.zeros((2, 2)))
    np.testing.assert_array_equal(state.info['steps'], [1., 1.])
    np.testing.assert_array_equal(state.info['episode_metrics']['sum_reward'], [1., 1.])
    np.testing.assert_array_equal(state.info['episode_metrics']['length'], [1., 1.])
    np.testing.assert_array_equal(state.info['episode_metrics']['cost'], [2., 2.])
    np.testing.assert_array_equal(state.info['episode_metrics']['distance'], [1., 1.])


def test_timeout_keeps_truncation_and_resets_next_episode_counters():
    env = wrap_myosuite_training(ToyEnv(end_after=99), episode_length=2)
    state = env.reset(jax.random.split(jax.random.PRNGKey(1), 1))
    step = jax.jit(env.step)
    for _ in range(2):
        state = step(state, jp.zeros((1, 2)))
    np.testing.assert_array_equal(state.reward, [1.])
    np.testing.assert_array_equal(state.info['truncation'], [1.])
    np.testing.assert_array_equal(state.info['episode_done'], [1.])
    np.testing.assert_array_equal(state.info['episode_metrics']['length'], [2.])
    state = step(state, jp.zeros((1, 2)))
    np.testing.assert_array_equal(state.info['truncation'], [0.])
    np.testing.assert_array_equal(state.info['episode_metrics']['length'], [1.])


@pytest.mark.parametrize('end_after, reward, length', [(1, 7., 1.), (2, 8., 2.)])
def test_action_repeat_stops_at_the_first_terminal_transition(end_after, reward, length):
    env = wrap_myosuite_training(ToyEnv(end_after=end_after), episode_length=10, action_repeat=3)
    state = env.reset(jax.random.split(jax.random.PRNGKey(4), 1))
    step = jax.jit(env.step)
    for _ in range(2):
        state = step(state, jp.zeros((1, 2)))
        np.testing.assert_array_equal(state.reward, [reward])
        np.testing.assert_array_equal(state.done, [1.])
        np.testing.assert_array_equal(state.info['truncation'], [0.])
        np.testing.assert_array_equal(state.info['episode_metrics']['length'], [length])
        np.testing.assert_array_equal(state.info['episode_metrics']['cost'], [2 * length])
        np.testing.assert_array_equal(state.info['episode_metrics']['sum_reward'], [reward])
        np.testing.assert_array_equal(state.info['episode_metrics']['distance'], [length])


def test_action_repeat_masks_each_world_independently():
    env = wrap_myosuite_training(ToyEnv(end_after=2), episode_length=10, action_repeat=3)
    state = env.reset(jax.random.split(jax.random.PRNGKey(5), 2))
    state = state.replace(data=jp.array([0., 1.]))
    state = jax.jit(env.step)(state, jp.zeros((2, 2)))
    np.testing.assert_array_equal(state.reward, [8., 7.])
    np.testing.assert_array_equal(state.info['episode_metrics']['length'], [2., 1.])
    np.testing.assert_array_equal(state.info['episode_metrics']['cost'], [4., 2.])


def test_action_repeat_truncates_a_partial_repeat_at_the_time_limit():
    env = wrap_myosuite_training(ToyEnv(end_after=99), episode_length=2, action_repeat=3)
    state = env.reset(jax.random.split(jax.random.PRNGKey(6), 1))
    state = jax.jit(env.step)(state, jp.zeros((1, 2)))
    np.testing.assert_array_equal(state.reward, [2.])
    np.testing.assert_array_equal(state.info['truncation'], [1.])
    np.testing.assert_array_equal(state.info['episode_metrics']['length'], [2.])
    np.testing.assert_array_equal(state.info['episode_metrics']['cost'], [4.])
    np.testing.assert_array_equal(state.info['episode_metrics']['distance'], [2.])


@pytest.mark.parametrize('count', [1, 2, 5])
@pytest.mark.parametrize('predefined', [True, False])
def test_eval_reset_has_a_key_for_every_episode(count, predefined):
    env = EvalVmapWrapper(ToyEnv(), n_randomizations=2, predefined_evals=predefined)
    reset = jax.jit(env.eval_reset)
    ids = jp.arange(count, dtype=jp.int32)
    first = reset(jax.random.PRNGKey(3), ids)
    second = reset(jax.random.PRNGKey(3), ids)
    assert first.reward.shape == (count,)
    np.testing.assert_array_equal(first.info['target'], second.info['target'])
    if count > 1:
        assert len(np.unique(np.asarray(first.info['target']))) == count


def test_vision_wrapping_does_not_leak_a_mapped_model_into_autoreset_or_evaluation():
    # Madrona's real init primitive expects these operand buffers to include
    # every world.  Test its batching contract without requiring a CUDA build.
    fields = ('geom_rgba', 'geom_matid', 'geom_size', 'light_pos', 'light_dir',
              'light_type', 'light_castshadow', 'light_cutoff')
    renderer_init = jax.core.Primitive('test_madrona_model_buffers')
    renderer_init.def_impl(lambda *args: jp.array(0.))
    renderer_init.def_abstract_eval(lambda *args: jax.core.ShapedArray((), jp.float32))
    mapped_calls = []

    def check_buffers(values, axes):
        assert all(axis == 0 for axis in axes)
        assert all(value.shape[0] == 2 for value in values)
        mapped_calls.append(axes)
        return jp.zeros(2), 0

    batching.primitive_batchers[renderer_init] = check_buffers

    class ModelEnv(ToyEnv):
        def __init__(self):
            super().__init__()
            model = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
            )
            self._mjx_model = mjx.put_model(model)

        @property
        def mjx_model(self):
            return self._mjx_model

        def reset(self, rng):
            state = super().reset(rng)
            token = renderer_init.bind(*(getattr(self.mjx_model, field) for field in fields))
            return state.replace(obs={
                'proprioception': state.obs['proprioception'] + self.mjx_model.geom_size[0, 0] + token
            })

        def step(self, state, action):
            state = super().step(state, action)
            token = renderer_init.bind(*(getattr(self.mjx_model, field) for field in fields))
            return state.replace(obs={
                'proprioception': state.obs['proprioception'] + self.mjx_model.geom_size[0, 0] + token
            })

        def prepare_eval_rollout(self, rng):
            return None

    raw = ModelEnv()
    original_model = raw.mjx_model
    env = wrap_myosuite_training(raw, vision=True, num_vision_envs=2, episode_length=3)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(7), 2))
    assert raw.mjx_model is original_model
    step = jax.jit(env.step)
    for _ in range(2):
        state = step(state, jp.zeros((2, 2)))
    assert raw.mjx_model is original_model
    np.testing.assert_array_equal(state.done, [1., 1.])
    # A separate evaluation trace must read concrete model arrays, rather than
    # the BatchTracer installed by the training wrapper's earlier vmap.
    eval_env, _ = _maybe_wrap_env_for_evaluation(env.vision_env, seed=8)
    eval_state = jax.jit(eval_env.eval_reset)(jax.random.PRNGKey(8), jp.arange(2))
    assert np.isfinite(np.asarray(eval_state.obs['proprioception'])).all()
    eval_step = jax.jit(eval_env.step)
    for _ in range(2):
        eval_state = eval_step(eval_state, jp.zeros((2, 2)))
    np.testing.assert_array_equal(eval_state.data, [2., 2.])
    np.testing.assert_array_equal(eval_state.reward, [7., 7.])
    assert raw.mjx_model is original_model
    assert len(mapped_calls) >= 4
