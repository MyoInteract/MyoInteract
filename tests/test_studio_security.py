"""Studio security boundaries without starting simulation or training workers."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from myoInteract.studio import server  # noqa: E402
from myoInteract.studio import runs as runs_module  # noqa: E402
from myoInteract.studio.runs import JobManager, RunManager  # noqa: E402


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server, "_device", lambda: {"device": "test", "cuda": None})
    return TestClient(server.create_app(tmp_path / "data", allow_origins=[]))


def _outside_run(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run.json").write_text(json.dumps({"id": "outside", "pid": None}))
    (outside / "train.log").write_text("private outside log")
    return outside


@pytest.mark.parametrize("method", ["_meta", "log_path", "log_dir", "get", "stop"])
def test_run_operations_reject_paths_outside_the_data_root(
    tmp_path: Path, method: str
) -> None:
    outside = _outside_run(tmp_path)
    manager = RunManager(tmp_path / "data")
    before = (outside / "run.json").read_bytes()
    for rid in ("../../../../outside", str(outside)):
        with pytest.raises(KeyError):
            getattr(manager, method)(rid)
    assert (outside / "run.json").read_bytes() == before


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_encoded_run_traversal_is_rejected_by_http_routes(
    client: TestClient,
    tmp_path: Path,
    separator: str,
) -> None:
    outside = _outside_run(tmp_path)
    rid = quote(separator.join([".."] * 4 + ["outside"]), safe="")
    assert client.get(f"/api/runs/{rid}/log").status_code == 404
    assert client.get(f"/api/runs/{rid}").status_code == 404
    assert client.delete(f"/api/runs/{rid}").status_code == 404
    assert "stopped" not in json.loads((outside / "run.json").read_text())


def test_normal_studio_run_and_external_training_run_remain_accessible(
    tmp_path: Path,
) -> None:
    manager = RunManager(tmp_path)
    rid = "20261023-120000_example"
    run = manager.dir / rid
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"id": rid, "pid": None}))
    (run / "train.log").write_text("training log")
    assert manager.log_path(rid).read_text() == "training log"
    assert manager.get(rid)["source"] == "studio"
    output = tmp_path / "logs" / "rsl_rl" / "experiment" / "existing_run"
    output.mkdir(parents=True)
    (output / "model_final.onnx").write_bytes(b"")
    assert manager.log_dir("experiment/existing_run") == output.resolve()
    assert manager.get("experiment/existing_run")["source"] == "logs"


def test_job_files_cannot_escape_the_requested_job(tmp_path: Path) -> None:
    jobs = JobManager(tmp_path / "data")
    first, second = jobs.dir / "first", jobs.dir / "second"
    first.mkdir()
    second.mkdir()
    (first / "preview.png").write_bytes(b"preview")
    (second / "worker.log").write_text("other job")
    assert jobs.path("first", "preview.png").read_bytes() == b"preview"
    with pytest.raises(KeyError):
        jobs.path("first", "../second/worker.log")
    outside = _outside_run(tmp_path)
    with pytest.raises(KeyError):
        jobs.path(str(outside), "train.log")
    with pytest.raises(KeyError):
        jobs.get(str(outside))


def test_malformed_run_paths_are_rejected_as_unknown_runs(client: TestClient) -> None:
    assert client.get("/api/runs/%00/log").status_code == 404


def test_linked_logs_and_job_files_cannot_escape_their_directory(
    tmp_path: Path,
) -> None:
    outside = _outside_run(tmp_path)
    manager = RunManager(tmp_path / "data")
    run = manager.dir / "linked"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"id": "linked", "pid": None}))
    try:
        (run / "train.log").symlink_to(outside / "train.log")
    except OSError as exc:
        pytest.skip(f"This system does not permit creating file links: {exc}")
    with pytest.raises(KeyError):
        manager.log_path("linked")
    with pytest.raises(KeyError):
        manager.get("linked")
    jobs = JobManager(tmp_path / "data")
    job = jobs.dir / "linked"
    job.mkdir()
    (job / "result.json").symlink_to(outside / "run.json")
    with pytest.raises(KeyError):
        jobs.path("linked", "result.json")


@pytest.mark.parametrize(
    "path, method", [("/api/presets", "get"), ("/api/viewer", "delete")]
)
def test_forwarding_headers_do_not_bypass_hosted_page_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    method: str,
) -> None:
    monkeypatch.setattr(server, "_device", lambda: {})
    origin = "https://studio.example.org"
    client = TestClient(
        server.create_app(tmp_path, allow_origins=[origin], pairing_code="secret")
    )
    spoofed = {"Origin": origin, "X-Forwarded-Host": "studio.example.org"}
    request = getattr(client, method)
    assert request(path, headers=spoofed).status_code == 401
    accepted = request(path, headers={**spoofed, "X-Studio-Token": "secret"})
    assert accepted.status_code == 200
    assert accepted.headers["access-control-allow-origin"] == origin
    assert request(path, headers={"Origin": "http://testserver"}).status_code == 200


def test_explicit_proxy_mode_remains_usable_without_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_device", lambda: {})
    client = TestClient(
        server.create_app(tmp_path, allow_origins=[], pairing_code=None)
    )
    assert (
        client.get(
            "/api/presets",
            headers={
                "Origin": "https://proxy.example.org",
                "X-Forwarded-Host": "proxy.example.org",
            },
        ).status_code
        == 200
    )


def test_concurrent_training_starts_respect_the_capacity_limit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolving = threading.Barrier(2)
    spawned = []

    def resolve(preset: dict) -> tuple[list[str], dict]:
        resolving.wait(timeout=10)
        return [], {"env": {"ctrl_dt": 0.05}}

    def spawn(*args, **kwargs):
        proc = SimpleNamespace(pid=100_000_000, poll=lambda: None, returncode=None)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(server, "resolve_preset", resolve)
    monkeypatch.setattr(runs_module, "_spawn", spawn)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda name: client.post(
                    "/api/runs", json={"preset": {}, "name": name}
                ),
                ["first", "second"],
            )
        )
    assert sorted(response.status_code for response in responses) == [200, 409]
    assert len(spawned) == 1
