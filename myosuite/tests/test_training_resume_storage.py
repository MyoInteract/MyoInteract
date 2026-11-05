"""Full-state storage round trips and failure boundaries using real Orbax I/O."""

import hashlib
import json
from pathlib import Path

import flax.struct
import jax
import jax.numpy as jp
import numpy as np
import optax
import pytest
import mujoco
from mujoco import mjx
from brax.training import types
from brax.training.acme import running_statistics
from mujoco_playground._src.mjx_env import State

from myosuite.train.myouser.custom_ppo import resume


@flax.struct.dataclass
class Learner:
    params: object
    optimizer_state: object
    normalizer_params: object
    env_steps: object


def full_state():
    params = {0: {'kernel': jp.arange(6, dtype=jp.float32).reshape(2, 3)},
              1: {'bias': jp.array([0.5, -0.5, 1.], dtype=jp.float32)}}
    optimizer_state = optax.adam(3e-4).init(params)
    learner = Learner(params, optimizer_state,
                      running_statistics.init_state(jp.zeros(3)),
                      types.UInt64(hi=jp.uint32(1), lo=jp.uint32(123)))
    environment = State(
        data={'qpos': jp.arange(8, dtype=jp.float32).reshape(2, 4)},
        obs={'proprioception': jp.ones((2, 3))},
        reward=jp.array([1., 2.]), done=jp.array([0., 1.]),
        metrics={'success': jp.array([False, True])},
        info={'rng': jax.random.split(jax.random.PRNGKey(4), 2),
              'phase': jp.array([1, 2], dtype=jp.int32)},
    )
    return {'training_state': learner, 'environment_state': environment,
            'local_key': jax.random.key(3),
            'reset_keys': jax.random.split(jax.random.PRNGKey(5), 2),
            'next_iteration': np.int64(7), 'training_walltime': np.float64(12.5),
            'python_values': (123, True, 0.25),
            'weak_array': jp.stack([jp.asarray(1.), jp.asarray(2.)])}


def assert_same_tree(expected, restored):
    assert jax.tree_util.tree_structure(expected) == jax.tree_util.tree_structure(restored)
    for left, right in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(restored)):
        assert type(left) is type(right)
        if isinstance(left, jax.Array):
            assert left.dtype == right.dtype
            assert left.shape == right.shape
            if jax.dtypes.issubdtype(left.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(jax.random.key_data(left), jax.random.key_data(right))
                continue
            assert left.weak_type == right.weak_type
        np.testing.assert_array_equal(left, right)


def test_typed_full_state_roundtrip_with_real_optimizer_environment_and_rng(tmp_path):
    state = full_state()
    metadata = {'training': {'batch_size': 4}, 'environment': {'name': 'toy'},
                'logging': {'buffer': [1, 2]}}
    root = tmp_path / 'run' / 'training_state'
    snapshot = resume.save(root, 64, state, metadata)
    assert snapshot.name == '000000000064'
    assert resume.resolve(root) == resume.resolve(root.parent) == snapshot
    assert resume.resolve(snapshot) == snapshot
    envelope = resume.read_metadata(root.parent)
    assert envelope['format_version'] == resume.FORMAT_VERSION
    assert envelope['step'] == 64
    assert envelope['metadata'] == metadata
    restored, returned_metadata = resume.load(
        root, state, {'training': {'batch_size': 4}, 'environment': {'name': 'toy'}}
    )
    assert_same_tree(state, restored)
    assert returned_metadata == metadata
    assert isinstance(restored['training_state'], Learner)
    assert type(restored['training_state'].optimizer_state[0]) is type(state['training_state'].optimizer_state[0])
    assert set(restored['training_state'].params) == {0, 1}
    metadata['logging']['buffer'].append(3)
    assert resume.read_metadata(snapshot)['metadata']['logging']['buffer'] == [1, 2]
    before = (snapshot / 'metadata.json').read_bytes()
    with pytest.raises(FileExistsError, match='not be overwritten'):
        resume.save(root, 64, state, {})
    assert (snapshot / 'metadata.json').read_bytes() == before


def test_real_mjx_data_and_its_static_pytree_metadata_roundtrip(tmp_path):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    cpu_data = mujoco.MjData(model)
    mujoco.mj_forward(model, cpu_data)
    state = {'physics': mjx.put_data(model, cpu_data), 'iteration': np.int64(1)}
    snapshot = resume.save(tmp_path / 'training_state', 1, state, {'model': 'free-sphere'})
    restored, _ = resume.load(snapshot, state, {'model': 'free-sphere'})
    assert_same_tree(state, restored)


def test_resolver_ignores_partial_snapshots_but_rejects_exact_partial_and_policy(tmp_path):
    root = tmp_path / 'training_state'
    state = full_state()
    complete = resume.save(root, 8, state, {})
    (root / '000000000999').mkdir()
    (root / '.000000001000.tmp-aborted').mkdir()
    assert resume.resolve(root) == complete
    with pytest.raises(ValueError, match='Incomplete or corrupt'):
        resume.resolve(root / '000000000999')
    policy = tmp_path / 'checkpoints' / '000000001000'
    policy.mkdir(parents=True)
    (policy / '_METADATA').write_text('{}')
    with pytest.raises(ValueError, match='full training-state snapshot'):
        resume.resolve(policy)
    with pytest.raises(FileNotFoundError, match='does not exist'):
        resume.resolve(tmp_path / 'missing')


def test_metadata_and_template_mismatch_rejected_before_array_restore(tmp_path, monkeypatch):
    state = full_state()
    snapshot = resume.save(tmp_path / 'training_state', 16, state,
                           {'environment': {'ctrl_dt': 0.05, 'seed': 3}})

    def forbidden(*args, **kwargs):
        raise AssertionError('array restore must not run before validation')

    monkeypatch.setattr(resume.ocp, 'PyTreeCheckpointer', forbidden)
    with pytest.raises(ValueError, match=r'metadata.environment.ctrl_dt'):
        resume.load(snapshot, state, {'environment': {'ctrl_dt': 0.01, 'seed': 3}})
    with pytest.raises(ValueError, match=r'metadata.environment.seed'):
        resume.load(snapshot, state, {'environment': {'ctrl_dt': 0.05}})
    changed = dict(state, reset_keys=jp.zeros((3, 2), dtype=jp.uint32))
    with pytest.raises(ValueError, match='template structure/shape/dtype'):
        resume.load(snapshot, changed, {})
    changed = dict(state, python_values=[123, True, 0.25])
    with pytest.raises(ValueError, match='template structure/shape/dtype'):
        resume.load(snapshot, changed, {})


def test_corrupt_payload_metadata_and_unknown_versions_fail_actionably(tmp_path):
    state = full_state()
    first = resume.save(tmp_path / 'training_state', 1, state, {})
    payload_file = next(path for path in (first / 'state').rglob('*') if path.is_file())
    payload_file.write_bytes(payload_file.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='payload checksum'):
        resume.load(first, state, {})
    second = resume.save(tmp_path / 'training_state', 2, state, {})
    metadata_path = second / 'metadata.json'
    envelope = json.loads(metadata_path.read_text())
    envelope['format_version'] = 999
    metadata_path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='Unsupported.*version'):
        resume.read_metadata(second)
    # A corrupt selected complete snapshot must never fall back to an older one.
    with pytest.raises(ValueError, match='Unsupported.*version'):
        resume.resolve(tmp_path / 'training_state')


def test_failed_write_never_publishes_a_complete_snapshot(tmp_path, monkeypatch):
    class FailedCheckpointer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def save(self, path, **kwargs):
            Path(path).mkdir()
            raise OSError('disk full')

    monkeypatch.setattr(resume.ocp, 'PyTreeCheckpointer', FailedCheckpointer)
    root = tmp_path / 'training_state'
    with pytest.raises(OSError, match='disk full'):
        resume.save(root, 8, full_state(), {})
    assert not (root / '000000000008').exists()
    assert not (root / '.000000000008.lock').exists()
    with pytest.raises(ValueError, match='No complete'):
        resume.resolve(root)


@pytest.mark.parametrize('step', [-1, True, 10**12, 1.5])
def test_invalid_steps_and_non_json_metadata_rejected(tmp_path, step):
    with pytest.raises(ValueError, match='step must be an integer'):
        resume.save(tmp_path / 'training_state', step, full_state(), {})
    with pytest.raises(ValueError, match='finite JSON'):
        resume.save(tmp_path / 'training_state', 1, full_state(), {'bad': float('nan')})
