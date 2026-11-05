"""Weights & Biases for Studio's training runs: the account the backend logs to.

The page sends an API key once; the backend checks it with W&B, keeps it in memory for the runs it
starts (``WANDB_API_KEY`` in their environment) and never sends it back. "Remember" writes it with
``wandb login`` (``~/.netrc``) instead. Without a key from the page, a login that already exists
on the machine (``WANDB_API_KEY`` or ``~/.netrc``) is used.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

_RUN_URL = re.compile(r"https://wandb\.ai/[\w.-]+/[\w.-]+/runs/[\w-]+")
MACHINE_LOGIN_TTL_S = 60.0  # how long a check of the machine's own login is reused


class WandbError(ValueError):
    """W&B is not installed, or it refused the key."""


def run_url(log: str) -> str | None:
    """Last W&B run page printed in a training log (``wandb: 🚀 View run ... at: <url>``)."""
    hits = _RUN_URL.findall(log)
    return hits[-1] if hits else None


class WandbAccount:
    """The W&B login of this backend, shared by every run it starts."""

    def __init__(self) -> None:
        self._key: str | None = None
        self._viewer: dict[str, Any] | None = None
        self._machine: tuple[float, dict[str, Any]] | None = (
            None  # (when, status) of the machine's own login
        )
        self._lock = threading.Lock()

    @staticmethod
    def _lookup(api_key: str | None) -> dict[str, Any]:
        """User, default entity and teams of a key (``None``: the machine's own login)."""
        try:
            import wandb  # noqa: PLC0415
        except ImportError as exc:
            raise WandbError(
                "wandb is not installed on the backend: pip install wandb"
            ) from exc
        try:
            viewer = wandb.Api(api_key=api_key, timeout=20).viewer
            user = viewer.username
        except (
            Exception
        ) as exc:  # wandb raises its own error types (auth, network, usage)
            raise WandbError(f"W&B did not accept the key: {exc}") from exc
        if not user:
            raise WandbError("W&B did not accept the key.")
        teams = list(dict.fromkeys([user, *(viewer.teams or [])]))
        return {"user": user, "entity": viewer.entity or user, "teams": teams}

    def status(self) -> dict[str, Any]:
        """Login state for the page; never the key itself."""
        with self._lock:
            if self._viewer:
                return {
                    "installed": True,
                    "connected": True,
                    "source": "studio",
                    **self._viewer,
                }
            cached = self._machine
        if cached and time.monotonic() - cached[0] < MACHINE_LOGIN_TTL_S:
            return cached[1]
        try:
            info = self._lookup(None)
            source = "env" if os.environ.get("WANDB_API_KEY") else "machine"
            out = {"installed": True, "connected": True, "source": source, **info}
        except WandbError as exc:
            out = {"installed": "not installed" not in str(exc), "connected": False}
        with self._lock:
            self._machine = (time.monotonic(), out)
        return out

    def login(self, api_key: str, remember: bool = False) -> dict[str, Any]:
        """Check ``api_key`` with W&B and use it for the runs started from now on.

        Args:
            api_key: Key from https://wandb.ai/authorize.
            remember: Also write it to ``~/.netrc`` (``wandb login``), so it outlives this server.

        Raises:
            WandbError: wandb is missing or the key was refused.
        """
        key = api_key.strip()
        if not key:
            raise WandbError("Paste an API key first.")
        info = self._lookup(key)
        if remember:
            import wandb  # noqa: PLC0415

            wandb.login(key=key, relogin=True, verify=False)
        with self._lock:
            self._key, self._viewer = key, info
        return {
            "installed": True,
            "connected": True,
            "source": "studio",
            "remembered": remember,
            **info,
        }

    def logout(self) -> dict[str, Any]:
        """Forget the key the page sent (a key in ``~/.netrc`` stays; ``wandb logout`` removes it)."""
        with self._lock:
            self._key, self._viewer, self._machine = None, None, None
        return self.status()

    def run_env(self, project: str, entity: str | None) -> dict[str, str]:
        """Environment of a training run that logs to W&B (rsl_rl's ``WandbLogWriter``)."""
        env = {"MYOINTERACT_LOGGER": "wandb", "WANDB_PROJECT": project}
        if entity:
            env["WANDB_USERNAME"] = entity  # rsl_rl reads the entity from this variable
        with self._lock:
            if self._key:
                env["WANDB_API_KEY"] = self._key
        return env
