"""Lifecycle contracts that do not require running GPU physics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import psutil

from myoInteract.studio.runs import JobManager, RunManager, checkpoints
from myoInteract import training_lifecycle
from myoInteract.training_lifecycle import (
    BoundaryLogger,
    StopRequest,
    TrainingStopped,
    copy_checkpoint,
    remaining_iterations,
    save_onnx_atomic,
    write_json,
)


@pytest.mark.parametrize("target,completed,result", [(5, 0, 5), (5, 3, 2), (5, 5, 0)])
def test_budget_counts_completed_updates(
    target: int, completed: int, result: int
) -> None:
    assert remaining_iterations(target, completed) == result


@pytest.mark.parametrize(
    "target,completed", [(2, 3), (-1, 0), (True, 0), (3, -1), (3.0, 0)]
)
def test_invalid_budget_is_rejected(target: int, completed: int) -> None:
    with pytest.raises(ValueError):
        remaining_iterations(target, completed)


def test_stop_is_observed_after_native_update_logging(tmp_path: Path) -> None:
    events = []
    request = StopRequest(tmp_path / "stop")
    native = SimpleNamespace(log=lambda **kw: events.append(("native", kw["it"])))

    def completed(count: int) -> None:
        events.append(("completed", count))
        request.check()

    logger = BoundaryLogger(native, completed)
    logger.log(it=0)
    request.path.touch()
    with pytest.raises(TrainingStopped):
        logger.log(it=1)
    assert events == [("native", 0), ("completed", 1), ("native", 1), ("completed", 2)]


def test_logger_flushes_tensorboard_once() -> None:
    events = []
    native = SimpleNamespace(
        writer=SimpleNamespace(close=lambda: events.append("flush")),
        stop_logging_writer=lambda: events.append("stop"),
    )
    logger = BoundaryLogger(native, lambda _count: None)
    logger.stop_logging_writer()
    logger.stop_logging_writer()
    assert events == ["stop", "flush"]


def test_atomic_checkpoint_never_publishes_an_incomplete_bundle(tmp_path: Path) -> None:
    final = tmp_path / "model_3.onnx"
    final.write_bytes(b"previous complete checkpoint")

    def failing(path: str, _infos: object) -> None:
        Path(path).with_suffix(".onnx").write_bytes(b"incomplete")
        assert checkpoints(tmp_path) == [final.name]
        assert final.read_bytes() == b"previous complete checkpoint"
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        save_onnx_atomic(failing, str(final.with_suffix(".pt")))
    assert final.read_bytes() == b"previous complete checkpoint"
    assert sorted(p.name for p in tmp_path.iterdir()) == [final.name]

    def success(path: str, infos: object) -> None:
        assert infos == {"iterations": 4}
        Path(path).with_suffix(".onnx").write_bytes(b"complete")

    save_onnx_atomic(success, str(final.with_suffix(".pt")), {"iterations": 4})
    copy_checkpoint(final, tmp_path / "model_final.onnx")
    assert (
        final.read_bytes()
        == (tmp_path / "model_final.onnx").read_bytes()
        == b"complete"
    )


def test_json_replacement_is_complete_and_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.json"
    write_json(path, {"status": "running"})
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"status": "running"}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kind", ["json", "copy", "native"])
@pytest.mark.parametrize("failures", [2, 5])
def test_publication_retries_transient_permissions_and_preserves_old_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, failures: int
) -> None:
    destination = tmp_path / (
        "lifecycle.json" if kind == "json" else "model_final.onnx"
    )
    destination.write_bytes(b"previous complete file")
    source = tmp_path / "source.bin"
    source.write_bytes(b"new checkpoint")
    original = os.replace
    attempts = []
    waits = []

    def replace(pending: Path, final: Path) -> None:
        attempts.append(True)
        if len(attempts) <= failures:
            raise PermissionError("poll reader holds destination")
        original(pending, final)

    monkeypatch.setattr(training_lifecycle.os, "replace", replace)
    monkeypatch.setattr(training_lifecycle, "sleep", waits.append)

    def publish() -> None:
        if kind == "json":
            write_json(destination, {"status": "completed"})
        elif kind == "copy":
            copy_checkpoint(source, destination)
        else:

            def save(path: str, _infos: object) -> None:
                Path(path).with_suffix(".onnx").write_bytes(b"new checkpoint")

            save_onnx_atomic(save, str(destination.with_suffix(".pt")))

    if failures == 5:
        with pytest.raises(PermissionError):
            publish()
        assert destination.read_bytes() == b"previous complete file"
        assert len(attempts) == 5 and len(waits) == 4
    else:
        publish()
        assert destination.read_bytes() != b"previous complete file"
        assert len(attempts) == 3 and waits == [0.02, 0.04]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [destination.name, source.name]
    )


def test_restart_counts_detached_training_and_stop_is_cooperative(
    tmp_path: Path,
) -> None:
    manager = RunManager(tmp_path)
    rid = "detached"
    rdir = manager.dir / rid
    rdir.mkdir()
    write_json(
        rdir / "run.json",
        {
            "id": rid,
            "name": rid,
            "pid": os.getpid(),
            "process_started": psutil.Process().create_time(),
        },
    )
    assert manager.active() == [rid]
    # A restarted manager has no Popen handle. Stopping it must request a safe
    # boundary, never signal an arbitrary persisted PID (our own PID here).
    result = manager.stop(rid)
    assert result["status"] == "stopping"
    assert (rdir / "stop.requested").is_file()


@pytest.mark.parametrize("started", [None, -1.0])
def test_reused_or_legacy_pid_does_not_claim_gpu_capacity(
    tmp_path: Path, started: float | None
) -> None:
    manager = RunManager(tmp_path)
    rid = "unowned"
    rdir = manager.dir / rid
    rdir.mkdir()
    write_json(
        rdir / "run.json",
        {"id": rid, "name": rid, "pid": os.getpid(), "process_started": started},
    )
    assert manager.active() == []
    assert manager.stop(rid)["status"] == "unknown"
    assert not (rdir / "stop.requested").exists()


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_invalid_historical_metadata_does_not_break_admission(
    tmp_path: Path, contents: str, caplog: pytest.LogCaptureFixture
) -> None:
    manager = RunManager(tmp_path)
    rdir = manager.dir / "broken"
    rdir.mkdir()
    (rdir / "run.json").write_text(contents)
    assert manager.active() == []
    assert not [record for record in manager.list() if record["source"] != "examples"]
    with pytest.raises(KeyError, match="invalid run metadata"):
        manager.get("broken")
    assert "Ignoring invalid run metadata" in caplog.text


def test_stopping_an_already_finished_run_preserves_status(tmp_path: Path) -> None:
    manager = RunManager(tmp_path)
    rid = "finished"
    rdir = manager.dir / rid
    rdir.mkdir()
    write_json(rdir / "run.json", {"id": rid, "name": rid, "pid": None})
    output = tmp_path / "logs/rsl_rl/task" / f"timestamp_{rid}"
    output.mkdir(parents=True)
    write_json(output / "lifecycle.json", {"status": "completed"})
    (output / "model_final.onnx").write_bytes(b"complete")
    (rdir / "overrides.json").write_text("[]")
    (rdir / "resolved.json").write_text("{}")
    assert manager.stop(rid)["status"] == "finished"
    assert not (rdir / "stop.requested").exists()


def test_worker_failure_wins_over_stale_ready_marker(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    jid = "stream-crashed"
    jdir = manager.dir / jid
    jdir.mkdir()
    write_json(jdir / "result.json", {"port": 8080})
    (jdir / "worker.log").write_text("RuntimeError: simulation failed")
    manager._procs[jid] = SimpleNamespace(poll=lambda: 1)
    result = manager.get(jid)
    assert result["status"] == "failed" and result["exit_code"] == 1
    assert "simulation failed" in result["log"]


def test_dead_warm_worker_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = JobManager(tmp_path)
    manager._warm_enabled = True
    manager._warm = SimpleNamespace(poll=lambda: 1)
    replacements = []
    monkeypatch.setattr(manager, "_start_warm", lambda: replacements.append(True))
    assert manager._take_warm(["preview"], tmp_path / "log") is None
    assert replacements == [True]
