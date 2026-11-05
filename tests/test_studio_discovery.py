"""Run discovery remains usable when optional metadata is incomplete."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from myoInteract.studio import runs


def _external(root: Path, name: str) -> Path:
    output = root / runs.LOG_ROOT / "experiment" / name
    output.mkdir(parents=True)
    (output / "model_final.onnx").write_bytes(b"")
    return output


def _studio(manager: runs.RunManager, rid: str) -> Path:
    folder = manager.dir / rid
    folder.mkdir()
    (folder / "run.json").write_text(json.dumps({"id": rid, "pid": None}))
    (folder / "overrides.json").write_text("[]")
    (folder / "resolved.json").write_text("{}")
    return folder


@pytest.mark.parametrize("source", ["studio", "logs"])
@pytest.mark.parametrize(
    "contents", ["{", "[]", '{"env":null}', '{"env":{"vision":[1]}}']
)
def test_corrupt_optional_sidecars_do_not_hide_other_runs(
    tmp_path: Path, source: str, contents: str
) -> None:
    manager = runs.RunManager(tmp_path)
    if source == "studio":
        rid = "20261105-120000_broken"
        _studio(manager, rid)
        output = _external(tmp_path, f"time_{rid}")
    else:
        rid = "experiment/broken"
        output = _external(tmp_path, "broken")
    (output / "myointeract_config.json").write_text(contents)
    _external(tmp_path, "good")
    assert manager.get(rid, with_metrics=False)["summary"] is None
    listed = {row["id"]: row for row in manager.list()}
    assert rid in listed and "experiment/good" in listed
    assert listed[rid]["checkpoints"] == ["model_final.onnx"]


def test_invalid_control_timing_does_not_break_training_curves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _external(tmp_path, "broken-timing")
    (output / "myointeract_config.json").write_text('{"env":{"ctrl_dt":"bad"}}')
    (output / "events.out.tfevents.test").touch()
    monkeypatch.setattr(
        runs,
        "_Scalars",
        lambda _path: SimpleNamespace(
            read=lambda: {
                "Train/mean_reward": [(0, 1.5)],
                "Train/mean_episode_length": [(0, 10)],
            }
        ),
    )
    result = runs.RunManager(tmp_path).get("experiment/broken-timing")
    assert result["metrics"]["points"][0]["reward"] == 1.5
    assert result["metrics"]["points"][0]["episode_s"] is None


@pytest.mark.parametrize("contents", [None, "{", "{}"])
def test_missing_or_corrupt_studio_inputs_are_not_published_as_sidecars(
    tmp_path: Path, contents: str | None
) -> None:
    manager = runs.RunManager(tmp_path)
    rid = "20261105-120000_inputs"
    folder = _studio(manager, rid)
    output = _external(tmp_path, f"time_{rid}")
    source = folder / "overrides.json"
    if contents is None:
        source.unlink()
    else:
        source.write_text(contents)
    assert manager.get(rid)["evaluable"] is False
    assert not (output / "original_hydra_config.json").exists()
    assert any(row["id"] == rid for row in manager.list())


def test_bulk_queries_scan_training_outputs_once_and_refresh_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = runs.RunManager(tmp_path)
    ids = [f"20261105-120000_run_{i}" for i in range(4)]
    for rid in ids:
        _studio(manager, rid)
        _external(tmp_path, f"time_{rid}")
    original = Path.glob
    scans = []

    def glob(path, pattern, *args, **kwargs):
        if path == tmp_path / runs.LOG_ROOT and pattern.startswith("*/*"):
            scans.append(pattern)
        return original(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(runs, "_same_process", lambda _meta: True)
    assert manager.active() == sorted(ids)
    assert len(scans) == 1
    scans.clear()
    listed = [row for row in manager.list() if row["source"] != "examples"]
    assert {row["id"] for row in listed} == set(ids)
    assert len(scans) == 1
    _external(tmp_path, "new-cli-run")
    assert any(row["id"] == "experiment/new-cli-run" for row in manager.list())


def test_discovery_keeps_lexically_latest_matching_studio_output(
    tmp_path: Path,
) -> None:
    manager = runs.RunManager(tmp_path)
    rid = "20261105-120000_with_underscores"
    _studio(manager, rid)
    earlier = _external(tmp_path, f"a_{rid}")
    latest = _external(tmp_path, f"z_{rid}")
    earlier.touch()
    row = next(row for row in manager.list() if row["id"] == rid)
    assert row["log_dir"] == str(latest)


def test_bulk_queries_ignore_output_links_outside_the_log_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = runs.RunManager(tmp_path)
    rid = "20261105-120000_linked"
    _studio(manager, rid)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model_final.onnx").touch()
    parent = tmp_path / runs.LOG_ROOT / "experiment"
    parent.mkdir(parents=True)
    try:
        (parent / f"time_{rid}").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"This system cannot create links: {exc}")
    monkeypatch.setattr(runs, "_same_process", lambda _meta: True)
    assert manager.active() == [rid]
    listed = [row for row in manager.list() if row["source"] != "examples"]
    assert len(listed) == 1 and listed[0]["log_dir"] is None
    with pytest.raises(KeyError):
        manager.get(f"experiment/time_{rid}")
