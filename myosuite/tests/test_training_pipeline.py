"""Checkpoint, optional logging, and fixed-batch evaluation contracts."""

import copy
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest
from ml_collections import ConfigDict

from myosuite.envs.myo.myouser.evaluate import evaluate_policy
from myosuite.train.utils import train as training
from myosuite.train.utils.wrapper import EvalVmapWrapper
from myosuite.tests.test_training_wrappers import ToyEnv


def minimal_config(vision=False):
    return ConfigDict({
        'env': {'muscle_config': {'noise_params': {
            'sigdepnoise_type': 'None', 'constantnoise_type': 'None',
        }}, 'vision': {'enabled': vision, 'num_worlds': 4}},
        'vision': {'enabled': vision},
        'rl': {'num_timesteps': 100, 'num_envs': 4, 'episode_length': 2,
               'action_repeat': 1, 'network_factory': {}, 'num_checkpoints': 1},
        'run': {'seed': 0, 'eval_seed': 1, 'eval_episodes': 2},
        'wandb': {'enabled': False},
    })


@pytest.mark.parametrize('individual_step', [False, True])
@pytest.mark.parametrize('vision', [False, True])
def test_loading_keeps_checkpoint_files_and_caller_config_unchanged(
        tmp_path, monkeypatch, individual_step, vision):
    root = tmp_path / 'checkpoints'
    (root / '20').mkdir(parents=True)
    (root / '3').mkdir()
    (root / 'metadata').mkdir()
    config = minimal_config(vision=vision)
    original = copy.deepcopy(config.to_dict())
    (root / 'config.json').write_text(config.to_json(), encoding='utf-8')
    (root / '20' / 'marker').write_bytes(b'original checkpoint')
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    path = root / '20' if individual_step else root
    raw_env = SimpleNamespace(enable_eval_mode=lambda: None)
    wrapped_env = SimpleNamespace(unwrapped=raw_env, vision_env=raw_env)
    monkeypatch.setattr(training.registry, 'load', lambda *a, **kw: raw_env)
    monkeypatch.setattr(training, 'wrap_myosuite_training', lambda *a, **kw: wrapped_env)
    calls = []

    def trainer(**kwargs):
        calls.append(kwargs)
        return 'policy', 'params', {}

    monkeypatch.setattr(training.ppo, 'train', trainer)
    result = training.train_or_load_checkpoint(
        'fixture', config, eval_mode=True, checkpoint_path=path,
    )
    assert result == (raw_env, 'policy', 'params')
    assert config.to_dict() == original
    assert calls[0]['num_timesteps'] == 0
    assert calls[0]['restore_checkpoint_path'].name == '20'
    assert calls[0]['network_factory'].keywords['vision'] is vision
    assert training.checkpoint_config_path(path) == root / 'config.json'
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert after == before


def test_checkpoint_resolver_ignores_metadata_and_reports_empty_roots(tmp_path):
    root = tmp_path / 'checkpoints'
    (root / 'metadata').mkdir(parents=True)
    with pytest.raises(ValueError, match='No numbered checkpoints'):
        training.resolve_checkpoint_path(root)
    with pytest.raises(FileNotFoundError, match='does not exist'):
        training.resolve_checkpoint_path(root / 'missing')
    (root / '10').mkdir()
    (root / '2').mkdir()
    assert training.resolve_checkpoint_path(root).name == '10'


def test_checkpoint_only_logger_writes_locally_without_wandb(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('W&B must not be called when disabled')

    for name in ('log', 'Image', 'Artifact', 'init'):
        monkeypatch.setattr(training.wandb, name, forbidden)
    saves = []
    monkeypatch.setattr(training.ocp, 'PyTreeCheckpointer', lambda: SimpleNamespace(
        save=lambda path, params, **kwargs: saves.append((path, params)),
    ))
    logger = training.ProgressEvalLogger(
        logdir=tmp_path, eval_env=None,
        log_wandb=False, log_wandb_checkpoints=True,
    )
    metrics = logger.progress_eval_run(10, None, {'weights': jp.arange(3.)})
    assert metrics == {}
    assert saves[0][0] == tmp_path / 'checkpoints' / '10'
    np.testing.assert_array_equal(saves[0][1]['weights'], jp.arange(3.))


def test_vision_training_uses_checkpoint_only_callback_and_one_env(tmp_path, monkeypatch):
    config = minimal_config(vision=True)
    config.rl.num_timesteps = 8
    raw_env = SimpleNamespace(enable_eval_mode=lambda: None)
    wrapped_env = SimpleNamespace(unwrapped=raw_env, vision_env=raw_env)
    loads = []
    saves = []
    monkeypatch.setattr(training.ocp, 'PyTreeCheckpointer', lambda: SimpleNamespace(
        save=lambda path, params, **kwargs: saves.append((path, params)),
    ))
    monkeypatch.setattr(training.registry, 'load', lambda *a, **kw: loads.append(kw) or raw_env)
    monkeypatch.setattr(training, 'wrap_myosuite_training', lambda *a, **kw: wrapped_env)

    def trainer(**kwargs):
        logger = kwargs['policy_params_fn_checkpoints'].__self__
        assert logger.eval_env is None
        assert logger.n_episodes == 0
        assert kwargs['eval_env'] is None
        assert kwargs['network_factory'].keywords['vision'] is True
        kwargs['policy_params_fn_checkpoints'](8, None, {'weights': jp.arange(2.)})
        return 'policy', 'params', {}

    monkeypatch.setattr(training.ppo, 'train', trainer)
    assert training.train_or_load_checkpoint('fixture', config, logdir=tmp_path)[0] is raw_env
    assert len(loads) == 1
    assert saves[0][0] == tmp_path / 'checkpoints' / '8'
    np.testing.assert_array_equal(saves[0][1]['weights'], jp.arange(2.))


class ToyVisionEnv(ToyEnv):
    vision = True
    _config = ConfigDict({'vision': {'num_worlds': 4}})

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(obs=dict(state.obs, **{'pixels/view_0': jp.zeros((2, 2, 3))}))

    def step(self, state, action):
        state = super().step(state, action)
        pixels = jp.ones((2, 2, 3)) * state.data
        return state.replace(obs=dict(state.obs, **{'pixels/view_0': pixels}))


def test_vision_evaluation_resets_full_batch_and_keeps_terminal_states():
    env = EvalVmapWrapper(ToyVisionEnv())
    policy = jax.jit(lambda obs, keys: (jp.zeros((obs['proprioception'].shape[0], 2)), {}))
    (rollouts, videos), kind = evaluate_policy(
        eval_env=env, jit_inference_fn=policy, jit_reset=jax.jit(env.eval_reset),
        jit_step=jax.jit(env.step), n_episodes=2, ep_length=3,
    )
    assert kind == 'videos'
    assert len(rollouts) == len(videos) == 2
    assert [len(rollout) for rollout in rollouts] == [2, 2]
    np.testing.assert_array_equal([rollout[-1].reward for rollout in rollouts], [7., 7.])
    np.testing.assert_array_equal([rollout[-1].data for rollout in rollouts], [2., 2.])
    (many_rollouts, many_videos), _ = evaluate_policy(
        eval_env=env, jit_inference_fn=policy, jit_reset=jax.jit(env.eval_reset),
        jit_step=jax.jit(env.step), n_episodes=5, ep_length=3,
    )
    assert len(many_rollouts) == len(many_videos) == 5
    assert float(many_rollouts[0][0].info['target']) != float(many_rollouts[4][0].info['target'])


@pytest.mark.parametrize('vision', [False, True])
def test_reset_prefix_is_optional_and_leaves_transitions_unchanged(vision):
    env = EvalVmapWrapper(ToyVisionEnv() if vision else ToyEnv())
    policy = jax.jit(lambda obs, keys: (jp.zeros((obs['proprioception'].shape[0], 2)), {}))
    kwargs = dict(
        eval_env=env, jit_inference_fn=policy, jit_reset=jax.jit(env.eval_reset),
        jit_step=jax.jit(env.step), n_episodes=1, ep_length=3,
    )
    default, _ = evaluate_policy(**kwargs)
    prefixed, _ = evaluate_policy(**kwargs, include_reset_state=True)
    if vision:
        default, default_videos = default
        prefixed, prefixed_videos = prefixed
        assert len(default_videos[0]) == len(prefixed_videos[0])
    assert len(prefixed[0]) == len(default[0]) + 1
    assert float(prefixed[0][0].data) == 0.
    np.testing.assert_array_equal(
        [state.data for state in default[0]],
        [state.data for state in prefixed[0][1:]],
    )
    assert int(prefixed[0][0].info['phase']) == 0
