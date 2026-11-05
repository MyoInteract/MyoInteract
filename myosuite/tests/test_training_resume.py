"""Exact continuation checks using real PPO updates and full-state checkpoint IO."""

import copy
import hashlib
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest
import mujoco
from mujoco import mjx

from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
from myosuite.envs.myo.myouser.myouser_universal import MyoUserUniversal
from myosuite.tests.test_training_custom_ppo import network_factory, train_toy
from myosuite.tests.test_training_wrappers import ToyEnv
from myosuite.train.myouser.custom_ppo import checkpoint, resume, train as ppo


class ResumeToyEnv(ToyEnv):
    """Stochastic transitions also depend on actions, so carry/RNG loss matters."""

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(info={**state.info, 'signal': jp.array(0.)})

    def step(self, state, action):
        key, noise_key = jax.random.split(state.info['rng'])
        next_state = super().step(state, action)
        noise = 0.03 * jax.random.normal(noise_key)
        signal = state.info['signal'] + noise + 0.05 * jp.sum(action)
        info = {**next_state.info, 'rng': key, 'signal': signal}
        observation = next_state.obs['proprioception'].at[2].add(signal)
        return next_state.replace(
            obs={'proprioception': observation}, info=info,
            reward=next_state.reward + noise + 0.1 * jp.sum(action),
        )


class WideResumeToyEnv(ResumeToyEnv):
    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(obs={'proprioception': jp.append(state.obs['proprioception'], 0.)})

    def step(self, state, action):
        narrow_state = state.replace(obs={'proprioception': state.obs['proprioception'][:3]})
        state = super().step(narrow_state, action)
        return state.replace(obs={'proprioception': jp.append(state.obs['proprioception'], 0.)})


CONTEXT = {
    'environment': {'name': 'ResumeToyEnv', 'task': {'end_after': 3}},
    'vision': False, 'domain_randomization': False,
}


def _train(**overrides):
    arguments = {
        'environment': ResumeToyEnv(end_after=3), 'num_timesteps': 64,
        'num_checkpoints': 2, 'num_evals': 3,
        'log_training_metrics': True, 'training_metrics_steps': 8,
        'resume_metadata': copy.deepcopy(CONTEXT),
    }
    arguments.update(overrides)
    return train_toy(**arguments)


def _tree_equal(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    actual_leaves = jax.tree_util.tree_flatten_with_path(actual)[0]
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert actual_leaves
    for (path, actual_leaf), expected_leaf in zip(actual_leaves, expected_leaves):
        actual_array, expected_array = np.asarray(actual_leaf), np.asarray(expected_leaf)
        assert actual_array.dtype == expected_array.dtype, jax.tree_util.keystr(path)
        if actual_array.dtype.kind in 'fc':
            assert np.isfinite(actual_array).all(), jax.tree_util.keystr(path)
        np.testing.assert_array_equal(
            actual_array, expected_array, err_msg=jax.tree_util.keystr(path),
        )


def _hashes(path):
    return {
        str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in path.rglob('*') if file.is_file()
    }


class InterruptedRun(Exception):
    pass


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    results = {}
    original_save = resume.save
    captured = {}

    def capture_save(root, step, state, metadata):
        # Keep a typed restoration template while exercising the real writer.
        result = original_save(root, step, state, metadata)
        captured[(Path(root).resolve(), int(step))] = copy.deepcopy(state)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(resume, 'save', capture_save)
        for resets in (0, 2):
            root = tmp_path_factory.mktemp(f'exact_resume_resets_{resets}')
            baseline = root / 'baseline' / 'training_state'
            interrupted = root / 'interrupted' / 'training_state'
            continued = root / 'continued' / 'training_state'
            baseline_evaluations, baseline_checkpoints = [], []
            split_evaluations, split_checkpoints = [], []
            resumed_evaluations, resumed_checkpoints = [], []
            baseline_progress, split_progress, resumed_progress = [], [], []

            def record_evaluation(target):
                def callback(step, *unused):
                    target.append(int(step))
                    return {}
                return callback

            def stop_after_first_snapshot(step, *unused):
                split_checkpoints.append(int(step))
                raise InterruptedRun

            def record_progress(target):
                def callback(step, metrics):
                    target.append(int(step))
                return callback

            baseline_result = _train(
                num_resets_per_eval=resets, save_training_state_path=baseline,
                policy_params_fn=record_evaluation(baseline_evaluations),
                policy_params_fn_checkpoints=lambda step, *unused: baseline_checkpoints.append(int(step)),
                progress_fn=record_progress(baseline_progress),
            )
            with pytest.raises(InterruptedRun):
                _train(
                    num_resets_per_eval=resets, save_training_state_path=interrupted,
                    policy_params_fn=record_evaluation(split_evaluations),
                    policy_params_fn_checkpoints=stop_after_first_snapshot,
                    progress_fn=record_progress(split_progress),
                )
            first_snapshot = resume.resolve(interrupted)
            assert first_snapshot.name == '000000000032'
            assert (first_snapshot / 'COMPLETE').is_file()
            original_hashes = _hashes(interrupted)
            resumed_result = _train(
                num_resets_per_eval=resets, resume_training_path=interrupted,
                save_training_state_path=continued,
                policy_params_fn=record_evaluation(resumed_evaluations),
                policy_params_fn_checkpoints=lambda step, *unused: resumed_checkpoints.append(int(step)),
                progress_fn=record_progress(resumed_progress),
            )
            template = captured[(baseline.resolve(), 64)]
            baseline_state, baseline_metadata = resume.load(baseline, template, expected_metadata=None)
            resumed_state, resumed_metadata = resume.load(continued, template, expected_metadata=None)
            results[resets] = {
                'root': root, 'baseline': baseline, 'interrupted': interrupted,
                'continued': continued, 'template': template,
                'baseline_state': baseline_state, 'resumed_state': resumed_state,
                'baseline_metadata': baseline_metadata, 'resumed_metadata': resumed_metadata,
                'baseline_result': baseline_result, 'resumed_result': resumed_result,
                'baseline_evaluations': baseline_evaluations,
                'baseline_checkpoints': baseline_checkpoints,
                'split_evaluations': split_evaluations, 'split_checkpoints': split_checkpoints,
                'resumed_evaluations': resumed_evaluations,
                'resumed_checkpoints': resumed_checkpoints,
                'baseline_progress': baseline_progress, 'split_progress': split_progress,
                'resumed_progress': resumed_progress,
                'original_hashes': original_hashes,
            }
    return results


@pytest.mark.parametrize('resets', [0, 2])
def test_interrupted_resume_exactly_matches_uninterrupted_training(runs, resets):
    run = runs[resets]
    expected, actual = dict(run['baseline_state']), dict(run['resumed_state'])
    assert float(actual.pop('training_walltime')) >= 0
    assert float(expected.pop('training_walltime')) >= 0
    assert int(actual['next_iteration']) == 2
    _tree_equal(actual, expected)
    _tree_equal(run['resumed_result'][1], run['baseline_result'][1])
    assert np.asarray(actual['training_state'].normalizer_params.count).min() > 0
    assert run['resumed_metadata']['metrics_logger'] == run['baseline_metadata']['metrics_logger']
    assert run['resumed_metadata']['metrics_logger']['num_steps'] == 64
    assert run['resumed_metadata']['metrics_logger']['buffers']
    assert _hashes(run['interrupted']) == run['original_hashes']


@pytest.mark.parametrize('resets', [0, 2])
def test_resume_continues_original_callback_schedule_without_replaying_initial_work(runs, resets):
    run = runs[resets]
    assert run['baseline_evaluations'] == [0, 32, 64]
    assert run['baseline_checkpoints'] == [32, 64]
    assert run['split_evaluations'] == [0]
    assert run['split_checkpoints'] == [32]
    assert run['resumed_evaluations'] == [64]
    assert run['resumed_checkpoints'] == [64]
    assert run['baseline_progress'] == list(range(8, 65, 8))
    assert run['split_progress'] + run['resumed_progress'] == run['baseline_progress']


@pytest.mark.parametrize('resets', [0, 2])
def test_completed_snapshot_does_not_update_save_or_repeat_callbacks(runs, resets):
    run = runs[resets]
    evaluations, checkpoints = [], []
    destination = run['root'] / 'completed_noop' / 'training_state'
    _, params, metrics = _train(
        num_resets_per_eval=resets, resume_training_path=run['continued'],
        save_training_state_path=destination,
        policy_params_fn=lambda *args: evaluations.append(args[0]),
        policy_params_fn_checkpoints=lambda *args: checkpoints.append(args[0]),
    )
    _tree_equal(params, run['baseline_result'][1])
    assert metrics == {}
    assert evaluations == checkpoints == []
    assert not list(destination.glob('*/COMPLETE'))


@pytest.mark.parametrize('changes', [
    {'num_timesteps': 128}, {'seed': 5}, {'learning_rate': 0.002},
    {'num_checkpoints': 1}, {'num_evals': 2}, {'num_resets_per_eval': 1},
    {'unroll_length': 4}, {'environment': WideResumeToyEnv(end_after=3)},
    {'resume_metadata': {**CONTEXT, 'environment': {'name': 'different_task'}}},
])
def test_exact_resume_rejects_changed_training_contract(runs, changes):
    with pytest.raises(ValueError, match='(?i)resume|mismatch|compatible|metadata|shape|contract'):
        _train(resume_training_path=runs[0]['interrupted'], **changes)


def test_inference_only_checkpoint_is_rejected_as_exact_resume(runs):
    run = runs[0]
    root = run['root'] / 'inference_only'
    config = checkpoint.network_config(
        observation_size={'proprioception': (3,)}, action_size=2,
        normalize_observations=True, network_factory=network_factory(),
    )
    checkpoint.save(root, 64, run['baseline_result'][1], config)
    with pytest.raises(ValueError, match='(?i)full training.state|training_state|resume'):
        _train(resume_training_path=root / '000000000064')


def test_exact_resume_rejects_madrona_before_training(runs):
    with pytest.raises(ValueError, match='(?i)madrona|vision'):
        _train(resume_training_path=runs[0]['interrupted'], madrona_backend=True)


def test_exact_resume_rejects_multiple_processes(runs, monkeypatch):
    monkeypatch.setattr(jax, 'process_count', lambda: 2)
    with pytest.raises(ValueError, match='(?i)process|host|distributed'):
        _train(resume_training_path=runs[0]['interrupted'])


@pytest.fixture(scope='module')
def actual_environments():
    config = load_config_interactive([
        'env=universal', 'wandb=disabled', 'env/task_config/targets=single',
    ])
    return MyoUserUniversal(config.env), MyoUserUniversal(config.env)


def test_real_environment_signature_is_stable_and_identifies_model_changes(actual_environments):
    first, second = actual_environments
    original = ppo._resume_environment_signature(first)
    assert original == ppo._resume_environment_signature(second)
    assert len(original['model_sha256']) == 64
    gravity = second.mj_model.opt.gravity.copy()
    try:
        second.mj_model.opt.gravity[0] += 0.01
        changed = ppo._resume_environment_signature(second)
        assert changed['model_sha256'] != original['model_sha256']
    finally:
        second.mj_model.opt.gravity[:] = gravity


@pytest.mark.parametrize('change', ['reward', 'dwell'])
def test_real_environment_signature_checks_task_configuration_beyond_mjb(actual_environments, change):
    original_env, _ = actual_environments
    config = copy.deepcopy(original_env._config)
    with config.unlocked():
        if change == 'reward':
            config.task_config.weighted_reward_keys.reach += 0.5
        else:
            config.task_config.targets.target_0.dwell_duration += 0.05
    changed_env = MyoUserUniversal(config)
    original = ppo._resume_environment_signature(original_env)
    changed = ppo._resume_environment_signature(changed_env)
    assert original['model_sha256'] == changed['model_sha256']
    assert original != changed


def _factory_with_defaults(default, keyword):
    def factory(argument=default, *, factor=keyword):
        return argument * factor
    return factory


def _factory_with_closure(setting):
    def factory():
        return setting
    return factory


@pytest.mark.parametrize('arguments', [(2., 1.), (1., 2.)])
def test_callable_signature_identifies_defaults_with_identical_source(arguments):
    first, changed = _factory_with_defaults(1., 1.), _factory_with_defaults(*arguments)
    assert first.__qualname__ == changed.__qualname__
    assert first.__code__ is changed.__code__
    assert ppo._resume_value(first) != ppo._resume_value(changed)


@pytest.mark.parametrize('settings', [(1., 2.), (np.array([1., 2.]), np.array([1., 3.]))])
def test_callable_signature_identifies_closures_with_identical_source(settings):
    first, changed = map(_factory_with_closure, settings)
    assert first.__qualname__ == changed.__qualname__
    assert first.__code__ is changed.__code__
    assert ppo._resume_value(first) != ppo._resume_value(changed)


def test_dynamic_callable_signature_identifies_changed_bytecode_without_source():
    first_namespace, changed_namespace = {'__name__': __name__}, {'__name__': __name__}
    exec(compile('def factory(value):\n    return value + 1\n', '<resume_factory>', 'exec'), first_namespace)
    exec(compile('def factory(value):\n    return value + 2\n', '<resume_factory>', 'exec'), changed_namespace)
    first = ppo._resume_value(first_namespace['factory'])
    changed = ppo._resume_value(changed_namespace['factory'])
    assert first['name'] == changed['name']
    assert first['bytecode_sha256'] != changed['bytecode_sha256']


def test_full_state_storage_roundtrips_actual_mjx_environment_data(actual_environments, tmp_path, monkeypatch):
    env, _ = actual_environments
    cpu_data = mujoco.MjData(env.mj_model)
    mujoco.mj_forward(env.mj_model, cpu_data)
    reset_data = mjx.put_data(env.mj_model, cpu_data)
    monkeypatch.setattr(env, '_reset_bm_model', lambda rng: reset_data)
    state = env.reset(jax.random.PRNGKey(13))
    mapped_state = jax.tree_util.tree_map(lambda value: jp.asarray(value)[None, None], state)
    payload = {
        'environment_state': mapped_state,
        'local_key': jax.random.PRNGKey(17),
        'reset_keys': jax.random.split(jax.random.PRNGKey(19), 1)[None],
        'next_iteration': np.asarray(1, dtype=np.int64),
        'training_walltime': np.asarray(0.5, dtype=np.float64),
    }
    metadata = {'case': 'actual_mjx_environment_data'}
    resume.save(tmp_path / 'training_state', 32, payload, metadata)
    restored, restored_metadata = resume.load(tmp_path, payload, expected_metadata=metadata)
    assert restored_metadata == metadata
    assert type(restored['environment_state'].data) is type(mapped_state.data)
    _tree_equal(restored, payload)
