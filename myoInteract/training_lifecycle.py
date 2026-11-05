"""Small lifecycle adapters around the native mjlab/RSL-RL training loop.

RSL checkpoints already contain the models, normalizers and optimizer. Their
``iter`` field is the zero-based index of the last completed update, rather than
the number of updates completed. They do not contain simulator or RNG state.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from time import sleep
from typing import Any

from typing_extensions import Self


class TrainingStopped(Exception):
    """A cooperative stop reached a completed learner update."""


def _publish(pending: Path, destination: Path) -> None:
    """Bound retries when a Windows poll reader briefly holds the old file open."""
    delays = (0.02, 0.04, 0.08, 0.16)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(pending, destination)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            sleep(delays[attempt])


def remaining_iterations(target: int, completed: int) -> int:
    """Return the updates remaining in an absolute training budget.

    Args:
        target: Desired total number of completed updates.
        completed: Number already completed, including a restored update.

    Raises:
        ValueError: If either count is invalid or the target precedes the source.
    """
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v < 0
        for v in (target, completed)
    ):
        raise ValueError("Training iteration counts must be non-negative integers.")
    if target < completed:
        raise ValueError(
            f"Target {target} iterations precedes the checkpoint's {completed} completed updates. "
            "Increase --agent.max-iterations or use --additional-iterations."
        )
    return target - completed


def write_json(path: Path, data: Any) -> None:
    """Publish a JSON sidecar atomically in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def save_onnx_atomic(save: Callable[..., None], path: str, infos: Any = None) -> None:
    """Publish the native ONNX bundle only after its complete export succeeds.

    ``save`` is the inherited OnnxCheckpointingMjlabRunner.save method. Its native
    optimizer serialization and ONNX bundling are deliberately reused unchanged.
    """
    destination = Path(path).with_suffix(".onnx")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-", dir=destination.parent
    ) as temporary:
        native = Path(temporary) / Path(path).name
        save(str(native), infos)
        _publish(native.with_suffix(".onnx"), destination)


def copy_checkpoint(source: Path, destination: Path) -> None:
    """Publish a final alias without repeating ONNX export or optimizer encoding."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-", dir=destination.parent
    ) as temporary:
        pending = Path(temporary) / destination.name
        shutil.copyfile(source, pending)
        _publish(pending, destination)


class StopRequest:
    """Remember signals or a Studio stop file until an update boundary is reached."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.requested = False
        self._handlers: dict[int, Any] = {}

    def __enter__(self) -> Self:
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                self._handlers[number] = signal.getsignal(number)
                signal.signal(number, self._signal)
        return self

    def _signal(self, _number: int, _frame: Any) -> None:
        self.requested = True

    def check(self) -> None:
        """Raise only when the caller has reached a safe learner boundary."""
        if self.requested or (self.path is not None and self.path.exists()):
            raise TrainingStopped(
                "Stop requested; the last learner update is complete."
            )

    def __exit__(self, *_args: object) -> None:
        for number, handler in self._handlers.items():
            signal.signal(number, handler)


class BoundaryLogger:
    """Delegate native logging, then notify the trainer after each complete update.

    RSL-RL calls ``logger.log`` after ``alg.update`` and assigning its iteration
    counter. Delegation avoids maintaining a second copy of its rollout loop.
    """

    def __init__(self, logger: Any, on_update: Callable[[int], None]) -> None:
        self.wrapped = logger
        self.on_update = on_update
        self.closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def log(self, **values: Any) -> None:
        """Log the native update and report its completed-update count."""
        self.wrapped.log(**values)
        self.on_update(int(values["it"]) + 1)

    def stop_logging_writer(self) -> None:
        """Close the native writer once, including an interrupted learning loop."""
        if not self.closed:
            self.wrapped.stop_logging_writer()
            writer = getattr(self.wrapped, "writer", None)
            # Native RSL stops its LogWriter subclasses, but its tensorboard
            # SummaryWriter is a separate type and otherwise remains unflushed.
            if writer is not None and not callable(getattr(writer, "stop", None)):
                close = getattr(writer, "close", None)
                if close is not None:
                    close()
            self.closed = True
