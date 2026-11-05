"""Run manifests preserve effective configuration and disclose restore semantics."""

import json
import subprocess
import pytest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from myosuite.utils.run_manifest import _git_revision, checkpoint_digest, create_run_manifest, update_manifest_config, write_run_manifest


def test_manifest_detaches_config_and_identifies_warm_start(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "must-not-be-recorded")
    cfg = {"env": {"eval_mode": False}, "rl": {"num_timesteps": 64}, "run": {"seed": 3, "eval_seed": 9}}
    manifest = create_run_manifest(cfg, checkpoint_path="saved/checkpoints/64", backend="cpu", devices=["cpu:0"])
    cfg["rl"]["num_timesteps"] = 0
    assert manifest["config"]["rl"]["num_timesteps"] == 64
    assert manifest["restore"]["mode"] == "parameter_warm_start"
    assert not manifest["restore"]["optimizer_state_restored"]
    assert manifest["seeds"] == {"seed": 3, "eval_seed": 9}
    path = write_run_manifest(tmp_path, manifest)
    saved = json.loads(path.read_text())
    assert saved["config_sha256"] == manifest["config_sha256"]
    assert "must-not-be-recorded" not in path.read_text()
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_digest_tracks_configuration_and_evaluation_mode():
    cfg = {"env": {}, "rl": {"num_timesteps": 16}, "run": {"play_only": True}}
    first = create_run_manifest(cfg, checkpoint_path="source")
    assert first["restore"]["mode"] == "evaluation"
    cfg["rl"]["num_timesteps"] = 32
    assert create_run_manifest(cfg)["config_sha256"] != first["config_sha256"]


def test_exact_resume_manifest_distinguishes_requested_from_verified_restoration(tmp_path):
    source = tmp_path / 'training_state' / '000000000064'
    source.mkdir(parents=True)
    (source / 'state').write_bytes(b'full optimizer rng and environment state')
    cfg = {'env': {'eval_mode': False}, 'rl': {'num_timesteps': 128, 'resume_training_path': str(source)}, 'run': {'seed': 0}}
    manifest = create_run_manifest(cfg)
    restore = manifest['restore']
    assert restore['mode'] == 'exact_training_resume'
    assert restore['resume_training_path'] == str(source)
    assert restore['source_training_step'] == 64
    assert restore['resume_training_sha256'] == checkpoint_digest(source)
    assert restore['checkpoint_path'] is None
    assert not restore['restoration_verified']
    assert not any(restore[key] for key in ('optimizer_state_restored', 'training_step_restored', 'rng_state_restored', 'environment_state_restored'))
    assert manifest['config']['rl']['num_timesteps'] == 128


def test_exact_resume_manifest_rejects_ambiguous_restore_modes():
    config = {'env': {}, 'rl': {'num_timesteps': 100, 'resume_training_path': 'training_state/64'}, 'run': {}}
    with pytest.raises(ValueError, match='warm start'):
        create_run_manifest(config, checkpoint_path='policy/64')
    config['run']['play_only'] = True
    with pytest.raises(ValueError, match='evaluation mode'):
        create_run_manifest(config)


@pytest.mark.parametrize('trainer_fails', [False, True])
def test_training_orchestrator_verifies_resume_only_after_success(tmp_path, monkeypatch, trainer_fails):
    from types import SimpleNamespace
    import numpy as np
    from myosuite.envs.myo.myouser import train_jax_ppo as runner
    from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
    from myosuite.train.myouser.custom_ppo import resume

    source = tmp_path / 'source' / 'training_state' / '000000000064'
    source.mkdir(parents=True)
    (source / 'marker').write_bytes(b'original state')
    config = load_config_interactive(['wandb=disabled', 'rl.resume_training_path=source', 'rl.num_timesteps=128'], cfg_only=True)
    monkeypatch.setattr(runner, 'get_logs_dir', lambda: tmp_path / 'logs')
    monkeypatch.setattr(runner, 'resolve_training_resume_path', lambda path: source)
    monkeypatch.setattr(resume, 'read_metadata', lambda path: {'step': 64})
    monkeypatch.setattr(runner, 'ProgressLogger', lambda **kw: SimpleNamespace(times=[], progress=lambda *a, **k: None))
    calls = []

    def trainer(name, effective, **kwargs):
        calls.append(kwargs)
        if trainer_fails:
            raise ValueError('Resume compatibility mismatch')
        env = SimpleNamespace(_config=effective.env, dt=.05, eval_reset=lambda *a, **kw: None, step=lambda *a, **kw: None)
        return env, lambda *a, **kw: (lambda obs, rng: None), {'weights': np.zeros(1)}

    monkeypatch.setattr(runner, 'train_or_load_checkpoint', trainer)
    monkeypatch.setattr(runner, '_maybe_wrap_env_for_evaluation', lambda eval_env, seed: (eval_env, None))
    state = SimpleNamespace(reward=np.asarray(0.), data=SimpleNamespace(qpos=np.zeros(1), ctrl=np.zeros(1)))
    monkeypatch.setattr(runner, 'evaluate_policy', lambda **kwargs: ([[state]], 'rollout'))
    monkeypatch.setattr(runner, 'render_traj', lambda *a, **kw: [np.zeros((2, 2, 3), dtype=np.uint8)])
    monkeypatch.setattr(runner.media, 'write_video', lambda *a, **kw: None)
    if trainer_fails:
        with pytest.raises(ValueError, match='compatibility'):
            runner.train(config)
    else:
        runner.train(config)
    saved = list((tmp_path / 'logs').glob('*/run_manifest.json'))
    assert len(saved) == 1
    manifest = json.loads(saved[0].read_text())
    assert calls[0]['resume_training_path'] == source
    assert manifest['restore']['mode'] == 'exact_training_resume'
    assert manifest['restore']['source_training_step'] == 64
    assert manifest['restore']['restoration_verified'] is not trainer_fails
    for key in ('optimizer_state_restored', 'training_step_restored', 'rng_state_restored', 'environment_state_restored'):
        assert manifest['restore'][key] is not trainer_fails
    assert manifest['status'] == ('started' if trainer_fails else 'completed')
    assert (source / 'marker').read_bytes() == b'original state'


def test_manifest_records_effective_environment_migrations():
    cfg = {"env": {"task_config": {"obs_keys": ["qpos"]}}, "run": {}, "rl": {}}
    manifest = create_run_manifest(cfg)
    original_digest = manifest["config_sha256"]
    original_creation = manifest["created_at"]
    cfg["env"]["task_config"]["obs_keys"].append("target_pos")
    update_manifest_config(manifest, cfg)
    assert manifest["config"]["env"]["task_config"]["obs_keys"] == ["qpos", "target_pos"]
    assert manifest["config_sha256"] != original_digest
    assert manifest["created_at"] == original_creation
    cfg["env"]["task_config"]["obs_keys"].clear()
    assert manifest["config"]["env"]["task_config"]["obs_keys"] == ["qpos", "target_pos"]


def test_wheel_does_not_borrow_enclosing_project_revision(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    installed = tmp_path / "venv" / "site-packages"
    installed.mkdir(parents=True)
    assert _git_revision(installed) == {"commit": None, "dirty": None}


def test_checkpoint_identity_depends_on_content_and_not_location(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / "weights").write_bytes(b"original policy")
    assert checkpoint_digest(first) == checkpoint_digest(second)
    (second / "weights").write_bytes(b"different policy")
    assert checkpoint_digest(first) != checkpoint_digest(second)


def test_release_update_preserves_dynamic_version_expression(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / ".github" / "release_utils.py"
    spec = spec_from_file_location("myointeract_release_utils", source)
    release = module_from_spec(spec)
    spec.loader.exec_module(release)
    version_file = tmp_path / "myosuite" / "version.py"
    version_file.parent.mkdir()
    version_file.write_text('__version_tuple__ = (2, 8, 3)\n__version__ = ".".join(str(x) for x in __version_tuple__)\n')
    monkeypatch.chdir(tmp_path)
    release.update_version((2, 8, 4))
    namespace = {}
    exec(version_file.read_text(), namespace)
    assert namespace["__version__"] == "2.8.4"
    assert release.find_version(version_file) == "2.8.4"
