"""Run manifests preserve effective configuration and disclose restore semantics."""

import json
import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from myosuite.utils.run_manifest import _git_revision, checkpoint_digest, create_run_manifest, write_run_manifest


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
