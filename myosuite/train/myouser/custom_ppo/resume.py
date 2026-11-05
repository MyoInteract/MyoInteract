"""Versioned, atomic full-training-state storage, separate from policy exports.

The payload contains numeric arrays only. A caller-supplied typed template
reconstructs Flax dataclasses, optimizer namedtuples, integer dict keys, and
environment state without pickling Python objects or importing saved classes.
"""

from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

import jax
import numpy as np
from orbax import checkpoint as ocp


FORMAT_VERSION = 1
_FORMAT = 'myointeract.training_state'
_METADATA = 'metadata.json'
_COMPLETE = 'COMPLETE'
_PAYLOAD = 'state'
_STEP = re.compile(r'^[0-9]{12}$')


def _json_copy(value):
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError('Training-state metadata must be finite JSON data') from error


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def _write(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _payload_digest(path):
    digest = hashlib.sha256()
    files = sorted(path.rglob('*'), key=lambda item: item.relative_to(path).as_posix())
    for file in files:
        if file.is_symlink():
            raise ValueError(f'Training-state payload must not contain symlinks: {file}')
        if not file.is_file():
            continue
        name = file.relative_to(path).as_posix().encode('utf-8')
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        digest.update(file.stat().st_size.to_bytes(8, 'big'))
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def _qualified(value):
    return f'{value.__module__}.{value.__qualname__}'


def _static_schema(value):
    """Describe PyTree static auxiliary data without executing saved objects."""
    if isinstance(value, Enum):
        return {'enum': _qualified(type(value)), 'value': _static_schema(value.value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {'type': _qualified(type(value)), 'value': _json_copy(value)}
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        array = np.asarray(jax.device_get(value))
        return {'array_shape': list(array.shape), 'array_dtype': str(array.dtype),
                'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, (tuple, list)):
        return {'type': _qualified(type(value)), 'items': [_static_schema(item) for item in value]}
    if isinstance(value, dict):
        return {'type': 'builtins.dict',
                'items': [[_static_schema(key), _static_schema(item)] for key, item in value.items()]}
    if isinstance(value, type):
        return {'class': _qualified(value)}
    # MJX static auxiliary values include NumPy dtypes and immutable enum-like
    # objects. Their repr is data for comparison only; it is never evaluated.
    representation = repr(value)
    if ' at 0x' in representation:
        raise ValueError(f'Cannot persist unstable PyTree static metadata: {_qualified(type(value))}')
    return {'type': _qualified(type(value)), 'repr': representation}


def _tree_schema(definition):
    node = definition.node_data()
    if node is None:
        return {'node': 'leaf'}
    node_type, auxiliary = node
    return {'node': _qualified(node_type), 'auxiliary': _static_schema(auxiliary),
            'children': [_tree_schema(child) for child in definition.children()]}


def _is_key(value):
    return isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)


def _leaf(value, key, path, *, copy_array):
    if _is_key(value):
        physical = jax.random.key_data(value)
        kind = 'jax_prng'
    elif isinstance(value, jax.Array):
        physical, kind = value, 'jax_array'
    elif isinstance(value, np.ndarray):
        physical, kind = value, 'numpy_array'
    elif isinstance(value, np.generic):
        physical, kind = value, 'numpy_scalar'
    elif type(value) in (bool, int, float, complex):
        physical, kind = value, 'python_' + type(value).__name__
    else:
        raise ValueError(f'Unsupported training-state leaf at {path}: {type(value).__name__}; use numeric arrays/scalars')
    array = np.asarray(jax.device_get(physical))
    if array.dtype.kind not in 'biufc' and not (
            array.dtype.name == 'bfloat16' or array.dtype.name.startswith('float8')):
        raise ValueError(f'Unsupported training-state dtype at {path}: {array.dtype}')
    entry = {'key': key, 'path': path, 'shape': list(np.shape(value)),
             'dtype': str(getattr(value, 'dtype', array.dtype)), 'kind': kind,
             'stored_shape': list(array.shape), 'stored_dtype': str(array.dtype)}
    if isinstance(value, jax.Array):
        entry['weak_type'] = bool(getattr(value, 'weak_type', False))
    if kind == 'jax_prng':
        entry['prng_impl'] = str(jax.random.key_impl(value))
    if array.size == 0:
        # Orbax 0.11 rejects zero-sized arrays, but MJX state routinely contains
        # empty actuator/contact buffers. Their exact shape and dtype live in
        # the validated schema; a single zero byte is the storage placeholder.
        entry['empty_array'] = True
        return np.asarray(0, dtype=np.uint8), entry
    return (np.array(array, copy=True) if copy_array else array), entry


def _flatten(state, *, copy_arrays):
    paths_and_values, definition = jax.tree_util.tree_flatten_with_path(state)
    payload, schema = {}, []
    for index, (path, value) in enumerate(paths_and_values):
        key = f'leaf_{index:08d}'
        array, entry = _leaf(value, key, jax.tree_util.keystr(path), copy_array=copy_arrays)
        payload[key] = array
        schema.append(entry)
    return payload, schema, _tree_schema(definition), definition


def _read_snapshot(path):
    try:
        metadata_bytes = (path / _METADATA).read_bytes()
        complete = json.loads((path / _COMPLETE).read_text(encoding='utf-8'))
        envelope = json.loads(metadata_bytes)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f'Incomplete or corrupt full training-state snapshot at {path}: missing/invalid metadata or COMPLETE marker') from error
    if not isinstance(envelope, dict) or not isinstance(complete, dict):
        raise ValueError(f'Invalid training-state metadata at {path}')
    if envelope.get('format') != _FORMAT or envelope.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'Unsupported training-state snapshot format/version at {path}; expected version {FORMAT_VERSION}')
    if (type(envelope.get('step')) is not int or not _STEP.fullmatch(path.name)
            or envelope['step'] != int(path.name)):
        raise ValueError(f'Training-state snapshot step does not match its directory at {path}')
    if complete.get('metadata_sha256') != hashlib.sha256(metadata_bytes).hexdigest():
        raise ValueError(f'Corrupt training-state metadata checksum at {path}')
    if (not isinstance(envelope.get('metadata'), dict)
            or not isinstance(envelope.get('state_schema'), list)
            or not isinstance(envelope.get('tree_schema'), dict)
            or not isinstance(envelope.get('payload_sha256'), str)
            or not (path / _PAYLOAD).is_dir()):
        raise ValueError(f'Incomplete full training-state snapshot at {path}')
    # Reject non-finite values even if a hand-written JSON file bypassed save().
    _json_copy(envelope)
    return envelope


def resolve(path):
    """Resolve a run, training_state root, or exact complete snapshot directory.

    Incomplete numeric directories and temporary writes are ignored when finding
    the latest snapshot. A selected complete-but-corrupt snapshot fails loudly;
    it never silently falls back to an older training state.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Training-state directory does not exist: {path}')
    if _STEP.fullmatch(path.name):
        _read_snapshot(path)
        return path
    if (path / 'training_state').is_dir():
        path = path / 'training_state'
    candidates = [child for child in path.iterdir()
                  if child.is_dir() and _STEP.fullmatch(child.name) and (child / _COMPLETE).is_file()]
    if not candidates:
        raise ValueError(f'No complete full training-state snapshot found in {path}; policy-only checkpoints cannot resume training')
    selected = max(candidates, key=lambda child: int(child.name))
    _read_snapshot(selected)
    return selected


def read_metadata(path):
    """Read the version/step/user metadata/schema envelope without loading arrays."""
    return _read_snapshot(resolve(path))


def save(root, step: int, state: Any, metadata: dict):
    """Atomically publish a new snapshot under an explicit training_state root."""
    if type(step) is not int or not 0 <= step < 10**12:
        raise ValueError('Training-state step must be an integer between 0 and 999999999999')
    user_metadata = _json_copy(metadata)
    if not isinstance(user_metadata, dict):
        raise ValueError('Training-state metadata must be a JSON object')
    payload, schema, tree_schema, _ = _flatten(state, copy_arrays=True)
    if not payload:
        raise ValueError('Cannot save an empty full training-state snapshot')
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f'{step:012d}'
    lock = root / f'.{step:012d}.lock'
    if destination.exists():
        raise FileExistsError(f'Training-state snapshot already exists and will not be overwritten: {destination}')
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f'Training-state save is in progress or has a stale lock: {lock}') from error
    os.close(descriptor)
    temporary = root / f'.{step:012d}.tmp-{uuid.uuid4().hex}'
    try:
        temporary.mkdir()
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(temporary / _PAYLOAD, args=ocp.args.PyTreeSave(item=payload))
        envelope = {'format': _FORMAT, 'format_version': FORMAT_VERSION, 'step': step,
                    'metadata': user_metadata, 'state_schema': schema, 'tree_schema': tree_schema,
                    'payload_sha256': _payload_digest(temporary / _PAYLOAD)}
        encoded = _json_bytes(envelope)
        _write(temporary / _METADATA, encoded)
        _write(temporary / _COMPLETE, _json_bytes({'metadata_sha256': hashlib.sha256(encoded).hexdigest()}))
        if destination.exists():
            raise FileExistsError(f'Training-state snapshot already exists and will not be overwritten: {destination}')
        temporary.rename(destination)
    finally:
        lock.unlink()
    return destination


def _difference(expected, saved, path):
    if type(expected) is not type(saved):
        return path
    if isinstance(expected, dict):
        for key in sorted(expected.keys() | saved.keys()):
            if key not in expected or key not in saved:
                return f'{path}.{key}'
            difference = _difference(expected[key], saved[key], f'{path}.{key}')
            if difference:
                return difference
    elif isinstance(expected, list):
        if len(expected) != len(saved):
            return path
        for index, (left, right) in enumerate(zip(expected, saved)):
            difference = _difference(left, right, f'{path}[{index}]')
            if difference:
                return difference
    elif expected != saved:
        return path
    return None


def _restore_leaf(array, template, entry):
    if entry.get('empty_array'):
        if array.shape != () or array.dtype != np.dtype('uint8') or array.item() != 0:
            raise ValueError(f'Corrupt training-state empty-array placeholder at {entry["path"]}')
        array = np.empty(entry['stored_shape'], dtype=entry['stored_dtype'])
    if list(array.shape) != entry['stored_shape'] or str(array.dtype) != entry['stored_dtype']:
        raise ValueError(f'Corrupt training-state array shape/dtype at {entry["path"]}')
    kind = entry['kind']
    if kind == 'jax_prng':
        return jax.random.wrap_key_data(jax.device_put(array, template.sharding),
                                        impl=jax.random.key_impl(template))
    if kind == 'jax_array':
        restored = jax.device_put(array, template.sharding)
        if entry['weak_type']:
            # JAX has no public setter for weak_type. The pinned runtime's
            # conversion preserves this numerical/type property without a cast.
            from jax._src.lax.lax import _convert_element_type
            restored = _convert_element_type(restored, restored.dtype, weak_type=True,
                                              sharding=template.sharding)
        return restored
    if kind == 'numpy_array':
        return array
    if kind == 'numpy_scalar':
        return array[()]
    return type(template)(array.item())


def load(path, template: Any, expected_metadata: dict | None = None):
    """Validate and restore an exact typed state using the caller's template."""
    snapshot = resolve(path)
    envelope = _read_snapshot(snapshot)
    expected = _json_copy({} if expected_metadata is None else expected_metadata)
    if not isinstance(expected, dict):
        raise ValueError('Expected resume metadata must be a JSON object')
    for key, value in expected.items():
        difference = (f'metadata.{key}' if key not in envelope['metadata']
                      else _difference(value, envelope['metadata'][key], f'metadata.{key}'))
        if difference:
            raise ValueError(f'Resume metadata mismatch at {difference} in {snapshot}')
    _, schema, tree_schema, definition = _flatten(template, copy_arrays=False)
    if schema != envelope['state_schema'] or tree_schema != envelope['tree_schema']:
        raise ValueError(f'Resume state template structure/shape/dtype mismatch at {snapshot}; exact training state is required')
    if _payload_digest(snapshot / _PAYLOAD) != envelope['payload_sha256']:
        raise ValueError(f'Corrupt training-state payload checksum at {snapshot}')
    restore_args = {entry['key']: ocp.RestoreArgs(restore_type=np.ndarray) for entry in schema}
    try:
        with ocp.PyTreeCheckpointer() as checkpointer:
            payload = checkpointer.restore(snapshot / _PAYLOAD,
                args=ocp.args.PyTreeRestore(restore_args=restore_args))
        leaves = jax.tree_util.tree_leaves(template)
        if set(payload) != {entry['key'] for entry in schema}:
            raise ValueError('Unexpected or missing payload leaves')
        restored = [_restore_leaf(np.asarray(payload[entry['key']]), leaf, entry)
                    for leaf, entry in zip(leaves, schema)]
    except Exception as error:
        raise ValueError(f'Cannot restore full training-state payload at {snapshot}: {error}') from error
    return jax.tree_util.tree_unflatten(definition, restored), envelope['metadata']
