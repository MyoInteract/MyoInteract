"""Out-of-distribution evaluation: a run's policy on the setup in the editor.

The run keeps what its policy depends on (body model, timing, observations, vision);
everything else, mainly the targets, comes from the editor.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Any

# Resolved config entries that set the policy's inputs, its actions or their timing.
POLICY_KEYS = (
    "env.env_name",
    "env.model_path",
    "env.mj_impl",
    "env.ctrl_dt",
    "env.sim_dt",
    "env.task_config.obs_keys",
    "env.task_config.omni_keys",
    "env.task_config.reach_settings",
    "env.muscle_config.control_type",
    "env.muscle_config.fatigue_obs_keys",
    "env.vision",
    "vision",
)
# Not part of the task an evaluation sees (rewards and training-only settings).
IGNORED_KEYS = (
    "env.task_config.weighted_reward_keys",
    "env.eval_mode",
)


def _diff(old: Any, new: Any, key: str = "") -> list[tuple[str, Any, Any]]:
    """Differing entries of two config trees; a part only one side has is one entry."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(old.keys() | new.keys(), key=str):
            out += _diff(old.get(k), new.get(k), f"{key}.{k}" if key else str(k))
        return out
    if (
        key.endswith(".rgb") and old and new
    ):  # colours round-trip through the page's 0-255 values
        old, new = [round(c, 2) for c in old], [round(c, 2) for c in new]
        if old == new:
            return []
    return [] if old == new else [(key, old, new)]


def _under(key: str, roots: tuple[str, ...]) -> bool:
    return any(key == r or key.startswith(r + ".") for r in roots)


def _short(key: str) -> str:
    for head in ("env.task_config.", "env.muscle_config.", "env."):
        if key.startswith(head):
            return key[len(head) :]
    return key


def task_changes(trained: dict, setup: dict) -> list[str]:
    """What the editor's setup changes against the run's, as readable lines.

    Args:
        trained: Resolved config the run was trained with.
        setup: Resolved config of the setup in the editor.

    Returns:
        One line per changed task setting, e.g. ``"targets.target_0.size: [0.01, 0.01] -> [0.005, 0.005]"``.

    Raises:
        ValueError: The setup changes what the policy depends on (listed in the message).
    """
    differ = _diff(
        {k: trained.get(k) for k in ("env", "vision")},
        {k: setup.get(k) for k in ("env", "vision")},
    )
    fixed = [d for d in differ if _under(d[0], POLICY_KEYS)]
    if fixed:

        def show(key: str, value: Any) -> str:  # model files by name
            return repr(
                PurePath(value).name if key == "env.model_path" and value else value
            )

        lines = "; ".join(
            f"{_short(k)}: {show(k, old)} in the run, {show(k, new)} here"
            for k, old, new in fixed
        )
        raise ValueError(
            "The setup in the editor changes what this policy depends on, so it cannot run on it: "
            + lines
        )

    def line(key: str, old: Any, new: Any) -> str:
        if old is None and isinstance(new, dict):
            return f"{_short(key)}: added"
        if new is None and isinstance(old, dict):
            return f"{_short(key)}: removed"
        return f"{_short(key)}: {old!r} -> {new!r}"

    return [line(*d) for d in differ if not _under(d[0], IGNORED_KEYS)]
