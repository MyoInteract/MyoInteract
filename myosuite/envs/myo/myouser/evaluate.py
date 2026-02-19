import os
import jax
from ml_collections import ConfigDict
import json
import jax.numpy as jp

from myosuite.train.utils.wrapper import _maybe_wrap_env_for_evaluation

def find_first_one(arr):
# Returns the index of first 1, or len(arr) if no 1 exists
    return jp.argmax(jp.concatenate([arr, jp.array([1])]))

def evaluate_non_vision(eval_env, jit_inference_fn, jit_reset, jit_step, seed=123, n_episodes=1, ep_length=None, reset_info_kwargs={}):
    eval_key = jax.random.PRNGKey(seed)
    eval_key, reset_keys = jax.random.split(eval_key)
    state = jit_reset(reset_keys, eval_id=jp.arange(n_episodes, dtype=jp.int32), **reset_info_kwargs)
    unvmap = lambda x, i : jax.tree.map(lambda x: x[i], x)
    dones = jp.zeros(n_episodes)
    rollouts = {i: [] for i in range(n_episodes)}
    for ii in range(ep_length):
        eval_key, key = jax.random.split(eval_key)
        ctrl, _ = jit_inference_fn(state.obs, jax.random.split(key, n_episodes))
        state = jit_step(state, ctrl)
        for i in range(n_episodes):
            if not dones[i]:
                stated_unvampped = unvmap(state, i)
                rollouts[i].append(stated_unvampped)
        dones = jp.logical_or(dones, state.done)
        if dones.all():
            break
    rollout = []
    for i in range(n_episodes):
        rollout.append(rollouts[i])
    return rollout, "rollout"

def evaluate_vision(eval_env, jit_inference_fn, jit_reset, jit_step, seed=123, n_episodes=1, ep_length=None):
    eval_key = jax.random.PRNGKey(seed)
    eval_key, reset_keys = jax.random.split(eval_key)
    # reset_keys, reset_prepare_keys = jax.random.split(reset_keys)
    # _n_episodes = eval_env.prepare_eval_rollout(reset_prepare_keys)
    # if _n_episodes is not None:
    #     ## Override n_episodes with enforced value from eval_env
    #     n_episodes = _n_episodes
    ##TODO: combine _n_episodes with num_worlds??
    num_worlds = eval_env._config.vision.num_worlds
    state = jit_reset(jax.random.split(reset_keys, num_worlds))
    pixel_key = [key for key in state.obs.keys() if 'pixels' in key]
    assert len(pixel_key) == 1, "Only one pixel key is supported"
    pixel_key = pixel_key[0]
    unvmap_upto = lambda x, i : jax.tree.map(lambda x: x[:i], x)
    unvmap = lambda x, i : jax.tree.map(lambda x: x[i], x)
    extract_states = unvmap_upto(state, n_episodes)
    dones = jp.zeros(n_episodes)
    videos = {i: [] for i in range(n_episodes)}
    rollouts = {i: [] for i in range(n_episodes)}
    for ii in range(ep_length):
        eval_key, key = jax.random.split(eval_key)
        ctrl, _ = jit_inference_fn(state.obs, jax.random.split(key, num_worlds))
        state = jit_step(state, ctrl)
        extract_states = unvmap_upto(state, n_episodes)
        for i in range(n_episodes):
            if not dones[i]:
                videos[i].append(extract_states.obs[pixel_key][i])
                stated_unvampped = unvmap(state, i)
                rollouts[i].append(stated_unvampped)
        dones = jp.logical_or(dones, extract_states.done)
        if dones.all():
            break
    videos_all = []
    rollout = []
    for i in range(n_episodes):
        videos_all.append(videos[i])
        rollout.append(rollouts[i])
    return (rollout, videos_all), "videos"


#TODO: pass rng instead of seed, to avoid correlation between rng used for _maybe_wrap_env_for_evaluation and for further evaluation reset/step calls (re-generated using same seed)
def evaluate_policy(checkpoint_path=None, env_name=None,
                    eval_env=None, jit_inference_fn=None, jit_reset=None, jit_step=None,
                    seed=123, n_episodes=1,
                    ep_length=None,
                    log_MT_only=False):
    """
    Generate an evaluation trajectory from a stored checkpoint policy.

    You can either call this method by directly passing the checkpoint, env, jitted policy, etc. (useful if checkpoint was already loaded in advance, e.g., in a Jupyter notebook file), 
    or let this method load the env and policy from scratch by passing only checkpoint_path and env_name (takes ~1min).
    """

    if checkpoint_path is None:
        assert eval_env is not None, "If no checkpoint path is provided, env must be passed directly as 'eval_env'"
        assert jit_inference_fn is not None, "If no checkpoint path is provided, policy must be passed directly as 'jit_inference_fn'"
        assert jit_reset is not None, "If no checkpoint path is provided, jitted eval_reset function(!) must be passed directly as 'jit_reset'"
        assert jit_step is not None, "If no checkpoint path is provided, jitted step function must be passed directly as 'jit_step'"
    else:
        assert env_name is not None, "If checkpoint path is provided, env name must also be passed as 'env_name'"
        from myosuite.train.utils.train import train_or_load_checkpoint
        _cfg_path = os.path.join(checkpoint_path, "config.json") if checkpoint_path.rstrip("/").endswith("checkpoints") else os.path.join(checkpoint_path, "../config.json") 
        with open(_cfg_path, "r") as f:
            config = ConfigDict(json.load(f))
        eval_env, make_inference_fn, params = train_or_load_checkpoint(env_name, config, eval_mode=True, checkpoint_path=checkpoint_path, seed=config.run.seed)
        eval_env, n_randomizations = _maybe_wrap_env_for_evaluation(eval_env=eval_env, seed=seed)
        # n_episodes = 1 * n_randomizations
        # logging.info(f"Environment allows for {n_randomizations} randomized evaluation episodes.")
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        jit_reset = jax.jit(eval_env.eval_reset)
        jit_step = jax.jit(eval_env.step)

    if ep_length is None:
        ep_length = int(eval_env._config.task_config.max_duration / eval_env._config.ctrl_dt)

    if eval_env.vision:
        return evaluate_vision(eval_env, jit_inference_fn, jit_reset, jit_step, seed, n_episodes, ep_length)
    else:
        return evaluate_non_vision(eval_env, jit_inference_fn, jit_reset, jit_step, seed, n_episodes, ep_length)