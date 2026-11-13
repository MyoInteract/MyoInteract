"""Studio worker ownership and bounded metric extraction without simulation."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("psutil")

from myoInteract.studio import runs  # noqa: E402


def test_concurrent_jobs_keep_only_one_idle_warm_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff_started = threading.Event()
    finish_handoff = threading.Event()
    spawned = []
    terminated = []

    class Pipe:
        used = False

        def write(self, _value: bytes) -> None:
            self.used = True
            handoff_started.set()
            assert finish_handoff.wait(timeout=5)

        def close(self) -> None:
            pass

    def spawn(cmd, *_args, **_kwargs):
        proc = SimpleNamespace(
            pid=len(spawned) + 1,
            poll=lambda: None,
            stdin=Pipe(),
            warm=cmd[-1] == "warm",
        )
        spawned.append(proc)
        return proc

    monkeypatch.setattr(runs, "_spawn", spawn)
    monkeypatch.setattr(runs, "_terminate", lambda proc, _pid: terminated.append(proc))
    manager = runs.JobManager(tmp_path, warm=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(manager.start, "preview", [])
        assert handoff_started.wait(timeout=5)
        second = pool.submit(manager.start, "evaluate", [])
        try:
            second.result(timeout=0.2)
        except TimeoutError:
            pass  # A serialized handoff waits for the first request.
        finally:
            finish_handoff.set()
        first.result(timeout=5)
        second.result(timeout=5)

    # Every idle worker must remain owned so shutdown can terminate it.
    warm = [proc for proc in spawned if proc.warm and not proc.stdin.used]
    assert warm == [manager._warm]
    manager.close()
    assert terminated == warm


def test_closed_manager_does_not_restart_a_warm_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = []

    def spawn(*_args, **_kwargs):
        proc = SimpleNamespace(pid=len(spawned) + 1, poll=lambda: None)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(runs, "_spawn", spawn)
    monkeypatch.setattr(runs, "_terminate", lambda *_args: None)
    manager = runs.JobManager(tmp_path, warm=True)
    manager.close()
    assert manager._take_warm([], tmp_path / "worker.log") is None
    assert len(spawned) == 1


def test_scalar_reader_extracts_only_studios_curve_tags() -> None:
    used = [
        "Episode_Termination/episode_success",
        "Episode_Termination/time_out",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Perf/total_fps",
    ]
    read = []

    def scalars(tag):
        read.append(tag)
        return [SimpleNamespace(step=3, value=0.5)]

    reader = runs._Scalars.__new__(runs._Scalars)
    reader.acc = SimpleNamespace(
        Reload=lambda: None,
        Tags=lambda: {
            "scalars": [*used, "Policy/mean_noise_std", "Loss/value_function"]
        },
        Scalars=scalars,
    )
    assert reader.read() == {tag: [(3, 0.5)] for tag in used}
    assert set(read) == set(used)
