# Modified from code in Mujoco Playground to suit this project
#
#
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrappers for MuJoCo Playground environments."""

import functools
from absl import logging
from typing import Any, Callable, Optional, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.wrappers.training import VmapWrapper, DomainRandomizationVmapWrapper
from brax.base import System
from brax import base

from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper, BraxDomainRandomizationVmapWrapper

# class BraxDomainRandomizationVmapWrapper(Wrapper):
#   """Brax wrapper for domain randomization."""

#   def __init__(
#       self,
#       env: mjx_env.MjxEnv,
#       randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
#   ):
#     super().__init__(env)
#     self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

#   def _env_fn(self, mjx_model: mjx.Model) -> mjx_env.MjxEnv:
#     env = self.env
#     env.unwrapped._mjx_model = mjx_model
#     return env

#   def reset(self, rng: jax.Array) -> mjx_env.State:
#     def reset(mjx_model, rng):
#       env = self._env_fn(mjx_model=mjx_model)
#       return env.reset(rng)

#     state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
#     return state

#   def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
#     def step(mjx_model, s, a):
#       env = self._env_fn(mjx_model=mjx_model)
#       return env.step(s, a)

#     res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
#         self._mjx_model_v, state, action
#     )
#     return res


def _identity_vision_randomization_fn(
    mjx_model: mjx.Model, num_worlds: int
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary fields for the Madrona memory buffer copy."""
  in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_type': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })
  mjx_model = mjx_model.tree_replace({
      'geom_rgba': jp.repeat(
          jp.expand_dims(mjx_model.geom_rgba, 0), num_worlds, axis=0
      ),
      'geom_matid': jp.repeat(
          jp.expand_dims(mjx_model.geom_matid, 0), num_worlds, axis=0
      ),
      'geom_size': jp.repeat(
          jp.expand_dims(mjx_model.geom_size, 0), num_worlds, axis=0
      ),
      'light_pos': jp.repeat(
          jp.expand_dims(mjx_model.light_pos, 0), num_worlds, axis=0
      ),
      'light_dir': jp.repeat(
          jp.expand_dims(mjx_model.light_dir, 0), num_worlds, axis=0
      ),
      'light_type': jp.repeat(
          jp.expand_dims(mjx_model.light_type, 0), num_worlds, axis=0
      ),
      'light_castshadow': jp.repeat(
          jp.expand_dims(mjx_model.light_castshadow, 0), num_worlds, axis=0
      ),
      'light_cutoff': jp.repeat(
          jp.expand_dims(mjx_model.light_cutoff, 0), num_worlds, axis=0
      ),
  })
  return mjx_model, in_axes


def _supplement_vision_randomization_fn(
    mjx_model: mjx.Model,
    randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    num_worlds: int,
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary missing fields for the Madrona memory buffer copy."""
  mjx_model, in_axes = randomization_fn(mjx_model)

  required_fields = [
      'geom_rgba',
      'geom_matid',
      'geom_size',
      'light_pos',
      'light_dir',
      'light_type',
      'light_castshadow',
      'light_cutoff',
  ]

  for field in required_fields:
    if getattr(in_axes, field) is None:
      in_axes = in_axes.tree_replace({field: 0})
      val = getattr(mjx_model, field)
      mjx_model = mjx_model.tree_replace({
          field: jp.repeat(jp.expand_dims(val, 0), num_worlds, axis=0),
      })
  return mjx_model, in_axes


class MadronaWrapper:
  """Wraps a MuJoCo Playground to be used in Brax with Madrona."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      num_worlds: int,
      randomization_fn: Optional[
          Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
      ] = None,
  ):
    if not randomization_fn:
      randomization_fn = functools.partial(
          _identity_vision_randomization_fn, num_worlds=num_worlds
      )
    else:
      randomization_fn = functools.partial(
          _supplement_vision_randomization_fn,
          randomization_fn=randomization_fn,
          num_worlds=num_worlds,
      )
    self._env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    self.num_worlds = num_worlds

    # For user-made DR functions, ensure that the output model includes the
    # needed in_axes and has the correct shape for madrona initialization.
    required_fields = [
        'geom_rgba',
        'geom_matid',
        'geom_size',
        'light_pos',
        'light_dir',
        'light_type',
        'light_castshadow',
        'light_cutoff',
    ]
    for field in required_fields:
      assert hasattr(self._env._in_axes, field), f'{field} not in in_axes'
      assert (
          getattr(self._env._mjx_model_v, field).shape[0] == num_worlds
      ), f'{field} shape does not match num_worlds'

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    state = self._call_with_model_restored(self._env.reset, rng)
    state.info['_madrona_world_id'] = jp.arange(rng.shape[0], dtype=jp.int32)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    return self._call_with_model_restored(self._env.step, state, action)

  def _call_with_model_restored(self, function, *args):
    # Playground installs a per-world model on the raw environment while
    # tracing its vmap.  Leaving that BatchTracer there leaks it into a later
    # autoreset or evaluation trace.  The compiled operations already capture
    # the mapped model; restore the Python object when that scope has ended.
    raw_env = self._env.unwrapped
    model = raw_env._mjx_model
    try:
      return function(*args)
    finally:
      raw_env._mjx_model = model

  def auto_reset(self, state: State) -> State:
    """Reset one mapped world using the same model as its training steps."""
    world_id = state.info['_madrona_world_id']
    model = jax.tree_util.tree_map(
        lambda value, axis: value[world_id] if axis == 0 else value,
        self._env._mjx_model_v, self._env._in_axes,
        is_leaf=lambda value: value is None,
    )

    def reset():
      raw_env = self._env.unwrapped
      raw_env._mjx_model = model
      next_state = raw_env.auto_reset(state)
      next_state.info['_madrona_world_id'] = world_id
      return next_state

    return self._call_with_model_restored(reset)

  def eval_reset(self, rng: jax.Array, eval_id: jax.Array,
                 predefined_evals: bool = True) -> State:
    """Map reset keys and all renderer model buffers over the fixed batch."""
    def reset(model, key, index):
      raw_env = self._env.unwrapped
      raw_env._mjx_model = model
      if predefined_evals:
        return raw_env.eval_reset(key, index)
      return raw_env.reset(key)

    def mapped_reset():
      return jax.vmap(reset, in_axes=[self._env._in_axes, 0, 0])(
          self._env._mjx_model_v, rng, eval_id
      )

    state = self._call_with_model_restored(mapped_reset)
    state.info['_madrona_world_id'] = jp.arange(rng.shape[0], dtype=jp.int32)
    return state

  def __getattr__(self, name):
    """Delegate attribute access to the wrapped instance."""
    return getattr(self._env.unwrapped, name)

class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    try: 
      non_accumulation_metrics = env.non_accumulation_metrics
      print(f"Found non-accumulation metrics: {non_accumulation_metrics}")
    except AttributeError:
      print("No non-accumulation metrics found")
      non_accumulation_metrics = []
    self.non_accumulation_metrics = non_accumulation_metrics

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jp.zeros(rng.shape[:-1])
    episode_metrics = dict()
    episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
    episode_metrics['length'] = jp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def step(self, state: State, action: jax.Array) -> State:
    if self.action_repeat == 1:
      # Keep the normal training path free of repeat masking/tree selections.
      state = self.env.step(state, action)
      rewards = jp.expand_dims(state.reward, 0)
      steps_taken = jp.ones_like(state.done)
      accumulated_metrics = state.metrics
    else:
      initial_steps = state.info['steps']

      def f(carry, _):
        current, active, count = carry
        active = active & (initial_steps + count < self.episode_length)
        candidate = self.env.step(current, action)

        def select(new, old):
          mask = active.reshape(active.shape + (1,) * (new.ndim - active.ndim))
          return jp.where(mask, new, old)

        # Freeze each world at its first terminal state.  Later repeated
        # actions contribute neither rewards, metrics nor episode steps.
        next_state = jax.tree_util.tree_map(select, candidate, current)
        reward = jp.where(active, candidate.reward, jp.zeros_like(candidate.reward))
        metrics = jax.tree_util.tree_map(
            lambda value: select(value, jp.zeros_like(value)), candidate.metrics
        )
        count += active.astype(jp.int32)
        active = active & (next_state.done == 0)
        return (next_state, active, count), (reward, metrics)

      (state, _, steps_taken), (rewards, metric_steps) = jax.lax.scan(
          f,
          (state, jp.ones_like(state.done, dtype=jp.bool_),
           jp.zeros_like(state.done, dtype=jp.int32)),
          (), self.action_repeat,
      )
      accumulated_metrics = jax.tree_util.tree_map(
          lambda values: jp.sum(values, axis=0), metric_steps
      )
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + steps_taken
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps

    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    # Clear the previous episode before adding this transition.  Clearing after
    # the addition discards the first reward and step of every later episode.
    episode_metrics = jax.tree_util.tree_map(
        lambda value: jp.where(prev_done, jp.zeros_like(value), value),
        state.info['episode_metrics'],
    )
    episode_metrics['sum_reward'] += jp.sum(rewards, axis=0)
    episode_metrics['length'] += steps_taken
    for metric_name in state.metrics.keys():
      if (metric_name != 'reward') and (metric_name not in self.non_accumulation_metrics):
        episode_metrics[metric_name] += accumulated_metrics[metric_name]
      if metric_name in self.non_accumulation_metrics:
        episode_metrics[metric_name] = state.metrics[metric_name]
    state.info['episode_metrics'] = episode_metrics
    state.info['episode_done'] = done
    return state.replace(done=done)


def wrap_myosuite_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    if vision:
      env = MadronaWrapper(env, num_worlds=num_vision_envs, randomization_fn=randomization_fn)
      vision_env = env
    elif randomization_fn is None:
      env = VmapWrapper(env)
    else:
      env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    # env = AutoResetWrapper(env)
    env = BetterAutoResetWrapper(env)
    if vision:
      # Reuse this renderer/model mapping for evaluation, without training's
      # episode/autoreset wrappers replacing terminal physical states.
      env.vision_env = vision_env
    return env


class BetterAutoResetWrapper(Wrapper):
    """Reset episode state while retaining the completed transition for PPO."""
    
    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        
        def return_state_as_it_is(current_state):
          return current_state

        def reset_state(current_state):
          new_state = self.env.auto_reset(current_state)
          # Environment fields (targets, phase, controller history and RNG)
          # belong to the fresh episode.  Only wrapper bookkeeping describes
          # the terminal transition and is consumed by the metrics logger.
          info = dict(new_state.info)
          for key in ('steps', 'truncation', 'episode_done', 'episode_metrics'):
            if key in current_state.info:
              info[key] = current_state.info[key]
          return new_state.replace(
              reward=current_state.reward,
              done=current_state.done,
              info=info,
              metrics=current_state.metrics,
          )


        def compute_next_state(current_state):
          return jax.lax.cond(current_state.done, reset_state, return_state_as_it_is, current_state)

        vmapped_compute_next_state = jax.vmap(compute_next_state)
        next_state = vmapped_compute_next_state(state)
        return next_state

class AutoResetWrapper(Wrapper):
    """Automatically "resets" Brax envs that are done, without clearing info state."""

    ## AutoReset Wrapper required to implement adaptive target curriculum; checks if episode is completed and calls reset inside this function;
    ## WARNING: Due to the following lines, applying the default Brax AutoResetWrapper has no effect to this env!

    def step(self, state: mjx_env.State, action: jax.Array):
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        # ####################################################################################
        rng = state.info["rng"]

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        state_after_reset = jax.vmap(self.env.auto_reset)(rng, state.info)
        # state_after_reset = self.env.reset(rng)  #, state.info)

        for k in state.info:
            state_after_reset.info[k] = state_after_reset.info.get(k, state.info[k])

        data = jax.tree.map(
            where_done, state_after_reset.data, state.data  # state.pipeline_state
        )
        # obs_dict = jax.tree.map(where_done, obs_dict_after_reset, obs_dict)
        obs = jax.tree.map(where_done, state_after_reset.obs, state.obs)  # state.obs)
        info = jax.tree.map(where_done, state_after_reset.info, state.info)

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=state.reward,
            done=state.done,
            metrics=state.metrics,
            info=info,
        )

class EvalVmapWrapper(Wrapper):
  """Vectorizes Brax env for evaluation runs, using eval_reset instead of reset as entrypoint."""

  def __init__(self, env: Env, n_randomizations: Optional[int] = None, predefined_evals: Optional[bool] = True):
    super().__init__(env)
    # self.batch_size = batch_size
    self.n_randomizations = n_randomizations
    self.predefined_evals = predefined_evals
    self.eval_wrapped = True

  def eval_reset(self, rng: jax.Array, eval_id: jax.Array) -> State:
    batch_size = len(eval_id)
    if batch_size < 1:
      raise ValueError('Evaluation requires at least one episode')
    rng = jax.random.split(rng, batch_size)
    if self.predefined_evals:
      if self.n_randomizations is not None:
        ## only allow for eval_id values between 0 and (self.n_randomizations - 1) in this case
        eval_id = eval_id % self.n_randomizations
      return jax.vmap(self.env.eval_reset)(rng, eval_id)
    else:
      return jax.vmap(self.env.reset)(rng)

  def step(self, state: State, action: jax.Array) -> State:
    return jax.vmap(self.env.step)(state, action)


class MadronaEvalWrapper(Wrapper):
  """Evaluation adapter retaining fixed-world renderer model buffers."""

  def __init__(self, env, n_randomizations=None, predefined_evals=True):
    super().__init__(env)
    self.n_randomizations = n_randomizations
    self.predefined_evals = predefined_evals
    self.eval_wrapped = True

  def eval_reset(self, rng: jax.Array, eval_id: jax.Array) -> State:
    if len(eval_id) != self.env.num_worlds:
      raise ValueError(f'Madrona evaluation requires {self.env.num_worlds} reset keys')
    if self.n_randomizations is not None:
      eval_id = eval_id % self.n_randomizations
    return self.env.eval_reset(
        jax.random.split(rng, len(eval_id)), eval_id,
        predefined_evals=self.predefined_evals,
    )
  

def _maybe_wrap_env_for_evaluation(eval_env, seed):
    rng = jax.random.PRNGKey(seed)
    predefined_evals = True
    try:
      n_randomizations = eval_env.prepare_eval_rollout(rng)
    except AssertionError:
      print("ERROR: No evaluations defined for this task! Will use random initialisations for evaluation instead...")
      n_randomizations = 1
      predefined_evals = False
    
    if not hasattr(eval_env, "eval_wrapped"):
      wrapper_type = MadronaEvalWrapper if isinstance(eval_env, MadronaWrapper) else EvalVmapWrapper
      eval_env = wrapper_type(eval_env, n_randomizations=n_randomizations, predefined_evals=predefined_evals)
      assert hasattr(eval_env, "eval_wrapped")
    
    return eval_env, n_randomizations
