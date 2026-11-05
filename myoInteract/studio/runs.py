"""Training runs and background jobs started by Studio, each one a subprocess."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from myoInteract.studio.examples import example_dir, list_examples
from myoInteract.studio.wandb_account import run_url
from myoInteract.training_lifecycle import write_json

LOG_ROOT = Path("logs") / "rsl_rl"
EXAMPLE_PREFIX = "examples/"  # run ids of the bundled example policies (read-only)
_CKPT = re.compile(r"model_(\d+|final|stopped)\.onnx$")
_LOGGER = logging.getLogger(__name__)


def _contained_path(root: Path, *parts: str | Path) -> Path:
    """Resolve a path strictly below ``root``, rejecting traversal and linked escapes."""
    base = root.resolve()
    try:
        path = base.joinpath(*parts).resolve()
    except (OSError, ValueError) as exc:
        raise KeyError(str(parts)) from exc
    if base not in path.parents:
        raise KeyError(str(path))
    return path


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:40] or "run"


def _alive(pid: int | None) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def _process_started(pid: int) -> float | None:
    """Read an OS process identity without sending a signal."""
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _same_process(meta: dict) -> bool:
    """A persisted PID is owned only when its creation time still matches."""
    started = meta.get("process_started")
    pid = meta.get("pid")
    if (
        not isinstance(started, (int, float))
        or isinstance(started, bool)
        or not _alive(pid)
    ):
        return False
    return _process_started(pid) == started


def _spawn(
    cmd: list[str],
    cwd: Path,
    log: Path,
    env: dict[str, str] | None = None,
    stdin: int | None = None,
) -> subprocess.Popen:
    """Start ``cmd`` detached from the server's process group, stdout/stderr to ``log``."""
    kwargs: dict[str, Any] = {"start_new_session": True} if os.name == "posix" else {}
    # Jobs run the server's own myoInteract: with a checkout as working directory, `python -m`
    # would otherwise import that checkout's copy (PYTHONSAFEPATH keeps the cwd off sys.path).
    code = str(Path(__file__).resolve().parents[2])
    path = os.pathsep.join(p for p in (code, os.environ.get("PYTHONPATH", "")) if p)
    base = {**os.environ, "PYTHONPATH": path, "PYTHONSAFEPATH": "1", "PYTHONUTF8": "1"}
    with log.open("ab") as out:
        return subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=stdin,
            stdout=out,
            stderr=subprocess.STDOUT,
            env={**base, **(env or {})},
            **kwargs,
        )


def _terminate(proc: subprocess.Popen | None, pid: int | None) -> None:
    """Stop and reap an owned viewer/worker; never signal a persisted PID alone."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return  # It exited between poll() and the signal.
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=5)


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 16_000))
        text = f.read().decode(errors="replace").replace("\r", "\n")
    return "\n".join([line for line in text.splitlines() if line.strip()][-lines:])


def checkpoints(log_dir: Path) -> list[str]:
    """``model_*.onnx`` of a run, by iteration (``model_final`` last)."""
    found = [p.name for p in log_dir.glob("model_*.onnx") if _CKPT.search(p.name)]
    return sorted(
        found,
        key=lambda n: (2, 0)
        if "final" in n
        else ((1, 0) if "stopped" in n else (0, int(_CKPT.search(n)[1]))),
    )


def _saved_status(log_dir: Path) -> str | None:
    """Read an atomic trainer outcome, including CLI runs discovered by Studio."""
    lifecycle = _contained_path(log_dir, "lifecycle.json")
    if lifecycle.is_file():
        try:
            outcome = json.loads(lifecycle.read_text())
        except (OSError, ValueError) as error:
            _LOGGER.warning("Invalid lifecycle metadata in %s: %s", lifecycle, error)
            return None
        if not isinstance(outcome, dict):
            _LOGGER.warning(
                "Invalid lifecycle metadata in %s: expected a JSON object", lifecycle
            )
            return None
        status = outcome.get("status")
        if status in ("completed", "stopped", "failed"):
            return "finished" if status == "completed" else status
    return None


def summary(log_dir: Path) -> dict[str, Any] | None:
    """Short description of a run's task (model, targets, fatigue, vision), from its config sidecar."""
    path = log_dir / "myointeract_config.json"
    if not path.exists():
        return None
    try:
        cfg = json.loads(path.read_text())
        env = cfg["env"]
        vision = cfg.get("vision") or env.get("vision") or {}
        return {
            "model": Path(env["model_path"]).stem.replace("_universal_myointeract", ""),
            "targets": env["task_config"]["targets"]["num_targets"],
            "fatigue": bool(env["muscle_config"].get("fatigue_enabled", False)),
            "vision": bool(vision.get("enabled", False)),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _yaml_int(path: Path, key: str) -> int | None:
    if not path.exists():
        return None
    match = re.search(rf"^\s*{key}:\s*(\d+)\s*$", path.read_text(), re.M)
    return int(match[1]) if match else None


class _Scalars:
    """Cached tensorboard reader of one run directory."""

    def __init__(self, log_dir: Path):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )  # noqa: PLC0415

        self.acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})

    def read(self) -> dict[str, list[tuple[int, float]]]:
        self.acc.Reload()
        tags = set(self.acc.Tags().get("scalars", []))
        return {t: [(e.step, e.value) for e in self.acc.Scalars(t)] for t in tags}


def metrics(
    log_dir: Path, scalars: dict[str, list[tuple[int, float]]], ctrl_dt: float | None
) -> dict[str, Any]:
    """Training curves from disjoint success and failure-timeout counts.

    Current tasks exclude terminal successes from ``time_out``. Older logs with
    overlapping aggregate counts cannot reveal the overlap and are left as recorded.
    """
    steps_per_iter = (
        _yaml_int(log_dir / "params" / "agent.yaml", "num_steps_per_env") or 0
    ) * (_yaml_int(log_dir / "params" / "env.yaml", "num_envs") or 0)
    succ = dict(scalars.get("Episode_Termination/episode_success", []))
    tout = dict(scalars.get("Episode_Termination/time_out", []))
    reward = dict(scalars.get("Train/mean_reward", []))
    length = dict(scalars.get("Train/mean_episode_length", []))
    fps = dict(scalars.get("Perf/total_fps", []))
    rows = []
    for it in sorted(set(reward) | set(succ)):
        s, t = succ.get(it, 0.0), tout.get(it, 0.0)
        rows.append(
            {
                "iteration": it,
                "steps": (it + 1) * steps_per_iter if steps_per_iter else None,
                "success": s / (s + t) if s + t > 0 else None,
                "reward": reward.get(it),
                "episode_s": length[it] * ctrl_dt if ctrl_dt and it in length else None,
                "fps": fps.get(it),
            }
        )
    return {"steps_per_iteration": steps_per_iter or None, "points": rows}


class RunManager:
    """Starts ``myo_train`` from a preset and reports status, curves and checkpoints.

    Studio's own bookkeeping lives in ``<root>/logs/studio/runs/<id>/`` (preset, overrides, log); the
    training output goes where ``myo_train`` puts it, ``<root>/logs/rsl_rl/<experiment>/<time>_<id>``.
    Runs started elsewhere (UI, CLI) are listed too, so they can be evaluated, and so are the
    bundled example policies (``examples/<name>``, see ``studio.examples``).
    """

    def __init__(self, root: Path):
        self.root = root
        self.dir = root / "logs" / "studio" / "runs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}
        self._scalars: dict[str, _Scalars] = {}

    # ------------------------------------------------------------------ start / stop
    def active(self) -> list[str]:
        active = {rid for rid, p in list(self._procs.items()) if p.poll() is None}
        # Detached training keeps running across a Studio restart. Continue to
        # account for those jobs when enforcing the GPU capacity limit.
        for path in self.dir.glob("*/run.json"):
            if path.parent.name in self._procs:
                continue
            try:
                meta = self._meta(path.parent.name)
            except KeyError as error:
                _LOGGER.warning(
                    "Ignoring invalid run metadata for %s: %s", path.parent.name, error
                )
                continue
            output = self.log_dir(path.parent.name)
            if output and _saved_status(output):
                continue
            if _same_process(meta):
                active.add(path.parent.name)
        return sorted(active)

    def start(
        self,
        preset: dict,
        name: str,
        overrides: list[str],
        resolved: dict,
        env: dict[str, str] | None = None,
        wandb: dict | None = None,
    ) -> dict:
        """Start ``myo_train`` on ``preset``; ``env`` (e.g. a W&B login) goes to its environment only."""
        rid = f"{datetime.now():%Y%m%d-%H%M%S}_{_slug(name)}_{uuid.uuid4().hex[:8]}"
        rdir = self.dir / rid
        rdir.mkdir(parents=True)
        (rdir / "preset.json").write_text(json.dumps(preset, indent=1))
        (rdir / "overrides.json").write_text(json.dumps(overrides, indent=1))
        (rdir / "resolved.json").write_text(json.dumps(resolved, indent=1))
        cmd = [
            sys.executable,
            "-m",
            "myoInteract.scripts.train",
            "--config",
            str(rdir / "preset.json"),
            "--agent.run-name",
            rid,
            "--stop-file",
            str(rdir / "stop.requested"),
        ]
        proc = _spawn(
            cmd,
            self.root,
            rdir / "train.log",
            {"MYOINTERACT_LOGGER": "tensorboard", **(env or {})},
        )
        self._procs[rid] = proc
        meta = {
            "id": rid,
            "name": name,
            "pid": proc.pid,
            "process_started": _process_started(proc.pid),
            "started": time.time(),
            "ctrl_dt": resolved.get("env", {}).get("ctrl_dt"),
        }
        if wandb:
            meta["wandb"] = wandb  # project and entity, never the key
        write_json(rdir / "run.json", meta)
        return self.get(rid)

    def stop(self, rid: str) -> dict:
        meta = self._meta(rid)
        proc = self._procs.get(rid)
        output = self.log_dir(rid)
        terminal = _saved_status(output) if output else None
        running = (
            proc.poll() is None
            if proc is not None
            else (_same_process(meta) and not terminal)
        )
        if running:
            rdir = self._studio_dir(rid)
            write_json(
                _contained_path(rdir, "stop.requested"), {"requested": time.time()}
            )
            meta["stop_requested"] = True
            write_json(_contained_path(rdir, "run.json"), meta)
        return self.get(rid)

    # ------------------------------------------------------------------ queries
    def _studio_dir(self, rid: str) -> Path | None:
        """A direct Studio run directory, or None for other/unknown run IDs."""
        try:
            path = _contained_path(self.dir, rid)
        except KeyError:
            return None
        return path if path.parent == self.dir.resolve() and path.is_dir() else None

    def _meta(self, rid: str) -> dict:
        rdir = self._studio_dir(rid)
        if rdir is None:
            raise KeyError(rid)
        path = _contained_path(rdir, "run.json")
        if not path.is_file():
            raise KeyError(rid)
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise KeyError(f"invalid run metadata for {rid}: {error}") from error
        if not isinstance(meta, dict):
            raise KeyError(f"invalid run metadata for {rid}: expected a JSON object")
        return meta

    def log_path(self, rid: str) -> Path | None:
        """Training log of a Studio run; None for runs started elsewhere and for examples."""
        rdir = self._studio_dir(rid)
        if rdir is not None:
            path = _contained_path(rdir, "train.log")
            return path if path.is_file() else None
        self.log_dir(rid)  # raises KeyError for unknown runs
        return None

    def log_dir(self, rid: str) -> Path | None:
        """Training output directory of a Studio run, an example, or a run id ``<experiment>/<dir>``."""
        if rid.startswith(EXAMPLE_PREFIX) and example_dir(rid[len(EXAMPLE_PREFIX) :]):
            return example_dir(rid[len(EXAMPLE_PREFIX) :])
        rdir = self._studio_dir(rid)
        log_root = (self.root / LOG_ROOT).resolve()
        if rdir is not None:
            hits = sorted(
                path.resolve()
                for path in log_root.glob(f"*/*_{rdir.name}")
                if log_root in path.resolve().parents and path.is_dir()
            )
            return hits[-1] if hits else None
        path = _contained_path(log_root, rid)
        if path.is_dir():
            return path
        raise KeyError(rid)

    def _status(self, rid: str, meta: dict, log_dir: Path | None) -> str:
        proc = self._procs.get(rid)
        if proc is None and log_dir:
            saved = _saved_status(log_dir)
            if saved:
                return saved
        running = proc.poll() is None if proc is not None else _same_process(meta)
        if running:
            if meta.get("stop_requested"):
                return "stopping"
            return "running" if log_dir else "starting"
        if proc is not None and proc.returncode not in (None, 0):
            return "failed"
        status = _saved_status(log_dir) if log_dir else None
        if status:
            return status
        if meta.get("stopped"):
            return "stopped"
        if log_dir and (log_dir / "model_final.onnx").exists():
            return "finished"
        if proc is None and (
            meta.get("process_started") is None or _alive(meta.get("pid"))
        ):
            return "unknown"
        return "failed"

    def _sidecars(self, rid: str, log_dir: Path) -> None:
        """Config files the Gradio UI and the evaluator read from a run directory."""
        rdir = self._studio_dir(rid)
        if rdir is None:
            raise KeyError(rid)
        if not (log_dir / "original_hydra_config.json").exists():
            source = _contained_path(rdir, "overrides.json")
            _contained_path(log_dir, "original_hydra_config.json").write_text(
                source.read_text()
            )
        if not (log_dir / "myointeract_config.json").exists():
            source = _contained_path(rdir, "resolved.json")
            _contained_path(log_dir, "myointeract_config.json").write_text(
                source.read_text()
            )

    def get(self, rid: str, with_metrics: bool = True) -> dict:
        rdir = self._studio_dir(rid)
        if rdir is not None:
            meta = self._meta(rid)
            log_dir = self.log_dir(rid)
            if log_dir:
                self._sidecars(rid, log_dir)
            train_log = _contained_path(rdir, "train.log")
            out = {
                **meta,
                "source": "studio",
                "status": self._status(rid, meta, log_dir),
                "log": _tail(train_log),
            }
            if meta.get("wandb") and train_log.exists():
                with train_log.open("rb") as f:  # wandb prints the link when it starts
                    out["wandb_url"] = run_url(f.read(1 << 16).decode(errors="replace"))
            ctrl_dt = meta.get("ctrl_dt")
        elif rid.startswith(EXAMPLE_PREFIX) and example_dir(rid[len(EXAMPLE_PREFIX) :]):
            log_dir = self.log_dir(rid)
            example = json.loads((log_dir / "example.json").read_text())
            out = {
                "id": rid,
                "name": example["title"],
                "source": "examples",
                "status": "finished",
                "example": example,
            }
            ctrl_dt = None
        else:
            log_dir = self.log_dir(rid)
            out = {
                "id": rid,
                "name": rid,
                "source": "logs",
                "status": _saved_status(log_dir)
                or (
                    "finished" if (log_dir / "model_final.onnx").exists() else "unknown"
                ),
            }
            ctrl_dt = None
            cfg = log_dir / "myointeract_config.json"
            if cfg.exists():
                ctrl_dt = json.loads(cfg.read_text()).get("env", {}).get("ctrl_dt")
        out["log_dir"] = str(log_dir) if log_dir else None
        out["checkpoints"] = checkpoints(log_dir) if log_dir else []
        now = time.time()
        out["checkpoint_age_s"] = {
            c: round(now - (log_dir / c).stat().st_mtime) for c in out["checkpoints"]
        }
        out["summary"] = summary(log_dir) if log_dir else None
        out["evaluable"] = bool(
            log_dir and ((log_dir / "original_hydra_config.json").exists())
        )
        if with_metrics and log_dir and any(log_dir.glob("events.out.tfevents.*")):
            key = str(log_dir)
            if key not in self._scalars:
                self._scalars[key] = _Scalars(log_dir)
            out["metrics"] = metrics(log_dir, self._scalars[key].read(), ctrl_dt)
        return out

    def list(self) -> list[dict]:
        runs = []
        for path in sorted(self.dir.iterdir(), reverse=True):
            if (path / "run.json").exists():
                try:
                    runs.append(self.get(path.name, with_metrics=False))
                except KeyError as error:
                    _LOGGER.warning(
                        "Ignoring invalid run metadata for %s: %s", path.name, error
                    )
        mine = {r["log_dir"] for r in runs}
        for d in sorted(
            (self.root / LOG_ROOT).glob("*/*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:50]:
            if d.is_dir() and str(d) not in mine and any(d.glob("model_*.onnx")):
                runs.append(self.get(f"{d.parent.name}/{d.name}", with_metrics=False))
        runs += [
            self.get(EXAMPLE_PREFIX + e["name"], with_metrics=False)
            for e in list_examples()
        ]
        return runs


class JobManager:
    """Previews, evaluations and live views: ``python -m myoInteract.studio.worker <kind> ...`` per job.

    With ``warm=True`` one worker always waits with its imports done (``worker warm``, about 10 s of
    mjlab and task imports) and takes the next job from stdin; a new one starts as soon as it is used.
    """

    def __init__(self, root: Path, warm: bool = False):
        self.root = root
        self.dir = root / "logs" / "studio" / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}
        self._stopped: set[str] = set()  # jobs ended on request, not by failing
        self._warm_enabled = warm
        self._warm: subprocess.Popen | None = None
        if warm:
            self._start_warm()

    def _start_warm(self) -> None:
        cmd = [sys.executable, "-m", "myoInteract.studio.worker", "warm"]
        self._warm = _spawn(
            cmd, self.root, self.dir / "warm-worker.log", stdin=subprocess.PIPE
        )

    def _take_warm(self, argv: list[str], log: Path) -> subprocess.Popen | None:
        """Hand a job to the waiting worker (and start the next one); None if there is none."""
        proc, self._warm = self._warm, None
        if proc is None or proc.poll() is not None:
            if self._warm_enabled:
                self._start_warm()
            return None
        try:
            proc.stdin.write(
                (json.dumps({"argv": argv, "log": str(log)}) + "\n").encode()
            )
            proc.stdin.close()
        except OSError:
            return None
        finally:
            if self._warm_enabled:
                self._start_warm()
        return proc

    def close(self) -> None:
        """Stop the waiting worker (the server shuts down)."""
        if self._warm is not None:
            _terminate(self._warm, self._warm.pid)
            self._warm = None

    def start(
        self, kind: str, args: list[str], inputs: dict[str, Any] | None = None
    ) -> dict:
        jid = f"{kind}-{uuid.uuid4().hex[:10]}"
        jdir = self.dir / jid
        jdir.mkdir(parents=True)
        for name, data in (inputs or {}).items():
            (jdir / name).write_text(json.dumps(data, indent=1))
        args = [a.replace("{dir}", str(jdir)) for a in args]
        argv = [kind, *args, "--out", str(jdir)]
        proc = (
            self._take_warm(argv, jdir / "worker.log") if self._warm_enabled else None
        )
        self._procs[jid] = proc or _spawn(
            [sys.executable, "-m", "myoInteract.studio.worker", *argv],
            self.root,
            jdir / "worker.log",
        )
        return self.get(jid)

    def stop(self, jid: str) -> None:
        proc = self._procs.get(jid)
        if proc is not None and proc.poll() is None:
            self._stopped.add(jid)
        _terminate(proc, proc.pid if proc is not None else None)

    def path(self, jid: str, name: str) -> Path:
        jdir = _contained_path(self.dir, jid)
        path = _contained_path(jdir, name)
        if not path.is_file():
            raise KeyError(name)
        return path

    def get(self, jid: str) -> dict:
        jdir = _contained_path(self.dir, jid)
        if not jdir.is_dir():
            raise KeyError(jid)
        proc = self._procs.get(jid)
        result = _contained_path(jdir, "result.json")
        returncode = proc.poll() if proc is not None else None
        if jid in self._stopped and returncode is not None:
            status, data = "stopped", None
        elif returncode not in (None, 0):
            status, data = "failed", None
        elif result.exists():
            status, data = "done", json.loads(result.read_text())
        elif proc is not None and proc.poll() is None:
            status, data = "running", None
        else:
            status, data = "failed", None
        alive = proc is not None and proc.poll() is None
        log = _tail(_contained_path(jdir, "worker.log"), 15) if status != "done" else ""
        return {
            "id": jid,
            "status": status,
            "alive": alive,
            "result": data,
            "log": log,
            "exit_code": returncode,
        }
