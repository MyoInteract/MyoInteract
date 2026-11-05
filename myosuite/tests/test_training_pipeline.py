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


def _capture_pipeline(monkeypatch):
    raw_env = SimpleNamespace(enable_eval_mode=lambda: None)
    monkeypatch.setattr(training.registry, 'load', lambda *a, **kw: raw_env)
    monkeypatch.setattr(training, 'wrap_myosuite_training', lambda *a, **kw: SimpleNamespace(unwrapped=raw_env, vision_env=raw_env))
    monkeypatch.setattr(training, 'ProgressEvalLogger', lambda **kwargs: SimpleNamespace(progress_eval_run=lambda *a, **kw: None))
    calls = []
    monkeypatch.setattr(training.ppo, 'train', lambda **kwargs: calls.append(kwargs) or ('policy', 'params', {}))
    return calls


@pytest.mark.parametrize('steps,checkpoints,expected', [(100, 2, True), (100, 0, False), (0, 1, False)])
def test_full_state_snapshot_output_is_automatic_only_for_training(tmp_path, monkeypatch, steps, checkpoints, expected):
    config = minimal_config()
    config.rl.num_timesteps = steps
    config.rl.num_checkpoints = checkpoints
    original = copy.deepcopy(config.to_dict())
    calls = _capture_pipeline(monkeypatch)
    training.train_or_load_checkpoint('fixture', config, logdir=tmp_path)
    assert calls[0]['save_training_state_path'] == (tmp_path / 'training_state' if expected else None)
    assert calls[0]['resume_training_path'] is None
    assert calls[0]['resume_metadata']['environment_name'] == 'fixture'
    assert calls[0]['resume_metadata']['environment']['muscle_config']['noise_params']['sigdepnoise_type'] is None
    assert config.to_dict() == original


def test_fresh_multihost_training_keeps_policy_only_checkpoints(tmp_path, monkeypatch):
    config = minimal_config()
    calls = _capture_pipeline(monkeypatch)
    monkeypatch.setattr(training.jax, 'process_count', lambda: 2)
    training.train_or_load_checkpoint('fixture', config, logdir=tmp_path)
    assert calls[0]['save_training_state_path'] is None
    assert calls[0]['resume_training_path'] is None
    assert calls[0]['policy_params_fn_checkpoints'] is not None


def test_exact_resume_forwards_full_state_without_policy_loading_or_legacy_adapter(tmp_path, monkeypatch):
    config = minimal_config()
    config.rl.resume_training_path = str(tmp_path / 'source-run')
    original = copy.deepcopy(config.to_dict())
    source = tmp_path / 'source-run' / 'training_state' / '000000000020'
    source.mkdir(parents=True)
    marker = source / 'marker'
    marker.write_bytes(b'complete training state')
    resolved = []
    monkeypatch.setattr(training, 'resolve_training_resume_path', lambda path: resolved.append(path) or source)
    calls = _capture_pipeline(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError('Exact resume must not load policy-only checkpoints or migrate legacy observations')

    monkeypatch.setattr(training, 'adapt_legacy_checkpoint_observations', forbidden)
    monkeypatch.setattr(training.ppo_checkpoint, 'load', forbidden)
    output = tmp_path / 'continued-run'
    training.train_or_load_checkpoint('fixture', config, logdir=output)
    assert resolved == [str(tmp_path / 'source-run')]
    assert calls[0]['resume_training_path'] == source
    assert calls[0]['restore_checkpoint_path'] is None
    assert calls[0]['save_training_state_path'] == output / 'training_state'
    assert config.to_dict() == original
    assert marker.read_bytes() == b'complete training state'
    assert list(source.iterdir()) == [marker]


@pytest.mark.parametrize('updates,eval_mode,match', [
    ({'run': {'play_only': True}}, False, 'evaluation mode'),
    ({'rl': {'load_checkpoint_path': 'policy/checkpoints/20'}}, False, 'not both'),
    ({'rl': {'num_timesteps': 0}}, False, 'positive'),
    ({'vision': {'enabled': True}}, False, 'without vision'),
    ({}, True, 'evaluation mode'),
])
def test_invalid_resume_modes_fail_before_environment_or_output(tmp_path, monkeypatch, updates, eval_mode, match):
    config = minimal_config()
    config.rl.resume_training_path = 'source/training_state'
    for section, values in updates.items():
        config[section].update(values)
    monkeypatch.setattr(training.registry, 'load', lambda *a, **kw: pytest.fail('Environment allocated before resume validation'))
    with pytest.raises(ValueError, match=match):
        training.train_or_load_checkpoint('fixture', config, logdir=tmp_path / 'output', eval_mode=eval_mode)
    assert not (tmp_path / 'output').exists()


def test_historical_policy_directory_is_not_an_exact_resume(tmp_path, monkeypatch):
    config = minimal_config()
    policy = tmp_path / 'checkpoints' / '20'
    policy.mkdir(parents=True)
    config.rl.resume_training_path = str(policy)
    monkeypatch.setattr(training.registry, 'load', lambda *a, **kw: pytest.fail('Policy-only resume must fail before environment allocation'))
    with pytest.raises(ValueError, match='(?i)(policy|warm|training.state)'):
        training.train_or_load_checkpoint('fixture', config, logdir=tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_resume_context_detects_changed_model_assets_with_identical_path():
    import mujoco
    config = minimal_config()
    first = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    same = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    changed = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".2"/></worldbody></mujoco>')
    def context(model):
        return training.training_resume_metadata('fixture', config.env, SimpleNamespace(mj_model=model), vision=False, domain_randomization=False, network_config={})
    assert context(first)['compiled_model_sha256'] == context(same)['compiled_model_sha256']
    assert context(first)['compiled_model_sha256'] != context(changed)['compiled_model_sha256']


def test_hydra_exposes_distinct_resume_option_and_rejects_conflicts():
    from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
    config = load_config_interactive(['rl.resume_training_path=source/training_state'], cfg_only=True)
    assert config.rl.resume_training_path == 'source/training_state'
    assert config.rl.load_checkpoint_path is None
    with pytest.raises(ValueError, match='not both'):
        load_config_interactive(['rl.resume_training_path=source/training_state', 'rl.load_checkpoint_path=policy/checkpoints/20'], cfg_only=True)
    with pytest.raises(ValueError, match='evaluation mode'):
        load_config_interactive(['rl.resume_training_path=source/training_state', 'run.play_only=True'], cfg_only=True)


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
