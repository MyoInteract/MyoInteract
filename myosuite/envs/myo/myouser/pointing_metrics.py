# Licensed under the Apache License, Version 2.0; see the LICENSE file in the
# root directory of this repository.

"""Movement metrics of MyoUser evaluation rollouts (sequential pointing and buttons).

Turns the rollouts of :func:`myosuite.envs.myo.myouser.evaluate.evaluate_policy` (one
list of ``mujoco_playground.State`` per episode) into one pointing trial per target and
summarises them with :mod:`myosuite.utils.movement_metrics`. Pass
``include_reset_state=True`` to ``evaluate_policy`` so that the first target's trial
starts at the reset.

Segmentation follows ``MyoUserUniversal``: ``info["phase"]`` after a step is the target
pursued next, so target ``k`` is active from the sample at which it became the phase
(the selection of target ``k - 1``, or the start of the episode for ``k = 0``) to the
sample at which it is selected (the phase changes, or the episode ends with ``done``,
which the env sets when the last target is selected). "Inside" is the env's own test
(``info["steps_inside_target"] > 0``: sphere distance or button press), and dwell times
come from the env's target objects.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from myosuite.utils import movement_metrics as mm

_TARGET_POS_KEY = re.compile(r"target_(\d+)/target_pos")


@dataclass(frozen=True)
class EpisodeArrays:
    """One evaluation episode as NumPy arrays (:func:`episode_arrays`).

    Attributes:
        time: Simulation time of each sample ``(N,)`` (s); 0 for a reset state.
        ee_pos: End-effector (fingertip) position ``(N, 3)``.
        phase: ``info["phase"]`` ``(N,)``, the target pursued after each sample.
        inside: Inside the target that was active at each sample ``(N,)``.
        done: Termination flag ``(N,)``.
        act: Muscle activations ``(N, na)``.
        target_pos: Target centres ``(K, 3)``.
        target_size: Target radii ``(K,)`` (half-widths for buttons).
    """

    time: np.ndarray
    ee_pos: np.ndarray
    phase: np.ndarray
    inside: np.ndarray
    done: np.ndarray
    act: np.ndarray
    target_pos: np.ndarray
    target_size: np.ndarray


def episode_arrays(episode: Sequence[Any]) -> EpisodeArrays:
    """Stack the states of one episode (one ``evaluate_policy`` rollout) into arrays.

    Args:
        episode: The episode's states, each with unbatched ``info``, ``data`` and ``done``.

    Returns:
        The episode's arrays.
    """
    info = episode[0].info
    n_targets = 1 + max(
        (int(m.group(1)) for m in map(_TARGET_POS_KEY.fullmatch, info) if m), default=-1
    )
    if not n_targets:
        raise ValueError("no 'target_<k>/target_pos' in state.info: not a MyoUser task")

    def stack(get: Any) -> np.ndarray:
        return np.stack([np.asarray(get(state)) for state in episode])

    return EpisodeArrays(
        time=stack(lambda s: s.data.time).astype(float),
        ee_pos=stack(lambda s: s.info["ee_pos"]).astype(float),
        phase=stack(lambda s: s.info["phase"]).astype(int),
        inside=stack(lambda s: s.info["steps_inside_target"]) > 0,
        done=stack(lambda s: s.done).astype(bool),
        act=stack(lambda s: s.data.act).astype(float),
        target_pos=np.stack(
            [
                np.asarray(info[f"target_{k}/target_pos"], float)
                for k in range(n_targets)
            ]
        ),
        target_size=np.array(
            [
                float(np.asarray(info[f"target_{k}/target_size"]))
                for k in range(n_targets)
            ]
        ),
    )


def target_segments(
    phase: ArrayLike, done: ArrayLike, n_targets: int
) -> list[tuple[int, int, bool]]:
    """Sample ranges of the targets of one episode.

    Args:
        phase: ``info["phase"]`` after each sample ``(N,)``.
        done: Termination flag ``(N,)``.
        n_targets: Number of targets.

    Returns:
        ``(start, end, selected)`` per activated target, in order: the inclusive sample
        range from its activation to its selection, or to the last sample.
    """
    phase = np.asarray(phase, dtype=int)
    done = np.asarray(done, dtype=bool)
    n = len(phase)
    pursued = np.concatenate([[0], phase[:-1]])  # target pursued during each sample
    selection = (phase != pursued) | done
    segments: list[tuple[int, int, bool]] = []
    start = 0
    for k in range(n_targets):
        first = start + 1 if k else 0  # the activation sample selected target k - 1
        hits = np.flatnonzero(selection[first:])
        if hits.size == 0:
            if first < n:
                segments.append((start, n - 1, False))
            break
        end = first + int(hits[0])
        segments.append((start, end, True))
        if done[end]:
            break
        start = end
    return segments


def episode_trials(
    episode: EpisodeArrays,
    dt: float,
    dwell_times: Sequence[float | None] | None = None,
    *,
    onset_fraction: float = mm.ONSET_FRACTION,
    savgol: mm.SavgolParams | None = None,
) -> list[dict[str, float]]:
    """Pointing-trial metrics of each activated target of one episode.

    Args:
        episode: Arrays of :func:`episode_arrays`.
        dt: Control interval (s).
        dwell_times: Dwell per target (s), or ``None`` (reached and held).
        onset_fraction: Speed threshold of the movement onset and offset.
        savgol: Optional Savitzky-Golay smoothing of the derivatives.

    Returns:
        :func:`myosuite.utils.movement_metrics.pointing_trial_metrics` per target.
    """
    segments = target_segments(episode.phase, episode.done, len(episode.target_pos))
    trials = []
    for k, (i0, i1, selected) in enumerate(segments):
        window = slice(i0, i1 + 1)
        inside = episode.inside[window].copy()
        if k:
            inside[0] = False  # the activation sample is inside the previous target
        trials.append(
            mm.pointing_trial_metrics(
                episode.ee_pos[window],
                episode.target_pos[k],
                episode.target_size[k],
                dt,
                selected=selected,
                inside=inside,
                dwell_time=None if dwell_times is None else dwell_times[k],
                activations=episode.act[window],
                onset_fraction=onset_fraction,
                savgol=savgol,
            )
        )
    return trials


def rollout_trials(
    rollouts: Sequence[Sequence[Any]],
    env: Any = None,
    *,
    dt: float | None = None,
    dwell_times: Sequence[float | None] | None = None,
    onset_fraction: float = mm.ONSET_FRACTION,
    savgol: mm.SavgolParams | None = None,
) -> list[list[dict[str, float]]]:
    """Per-target trial metrics of evaluation rollouts, ``trials[episode][target]``.

    Args:
        rollouts: ``evaluate_policy`` rollouts, one list of states per episode.
        env: The evaluation env (wrapped or not); supplies ``dt`` and the dwell times.
        dt: Control interval (s); default ``env.dt``.
        dwell_times: Dwell per target (s); default from ``env.target_objs``
            (``dwell_steps * dt``, one step for buttons), else reached and held.
        onset_fraction: Speed threshold of the movement onset and offset.
        savgol: Optional Savitzky-Golay smoothing of the derivatives.

    Returns:
        One list of trial metrics per episode.
    """
    if dt is None:
        if env is None:
            raise ValueError("pass env or dt")
        dt = float(env.dt)
    if dwell_times is None and env is not None and hasattr(env, "target_objs"):
        dwell_times = [getattr(t, "dwell_steps", 1) * dt for t in env.target_objs]
    episodes = [episode_arrays(episode) for episode in rollouts]
    if any(episode.time[0] > 0.5 * dt for episode in episodes):
        warnings.warn(
            "rollouts do not start at the reset state, so the first target's trial "
            "starts one step late; pass include_reset_state=True to evaluate_policy",
            stacklevel=2,
        )
    return [
        episode_trials(
            episode, dt, dwell_times, onset_fraction=onset_fraction, savgol=savgol
        )
        for episode in episodes
    ]


def summarize_trials(
    trials: Sequence[Sequence[Mapping[str, float]]],
    *,
    conditions: Sequence[Hashable] | None = None,
    n_bins: int = 3,
    time_key: str = "trial_time",
) -> dict[str, float]:
    """Scalar summary of :func:`rollout_trials`, e.g. for logging.

    Keys: ``trials`` (count); the means over all trials of
    :func:`myosuite.utils.movement_metrics.mean_metrics` (``success`` is the selection
    rate of the activated targets); ``target_<k>/trial_time``,
    ``target_<k>/index_of_difficulty`` and ``target_<k>/success``, the means over the
    episodes that activated target ``k``; and
    :func:`myosuite.utils.movement_metrics.fitts_summary`.

    Args:
        trials: ``trials[episode][target]`` of :func:`rollout_trials`.
        conditions: Optional Fitts' law condition label per trial (episode-major order).
        n_bins: Number of ID groups without ``conditions``.
        time_key: Per-trial movement time of the Fitts' law summary.

    Returns:
        Metric name to value; empty without trials.
    """
    flat = [trial for episode in trials for trial in episode]
    if not flat:
        return {}
    out = {"trials": float(len(flat)), **mm.mean_metrics(flat)}
    for k in range(max(len(episode) for episode in trials)):
        means = mm.mean_metrics([episode[k] for episode in trials if len(episode) > k])
        for key in ("trial_time", "index_of_difficulty", "success"):
            out[f"target_{k}/{key}"] = means[key]
    out.update(
        mm.fitts_summary(flat, conditions=conditions, n_bins=n_bins, time_key=time_key)
    )
    return out


def rollout_movement_metrics(
    rollouts: Sequence[Sequence[Any]], env: Any = None, **kwargs: Any
) -> dict[str, float]:
    """:func:`summarize_trials` of :func:`rollout_trials` in one call.

    Args:
        rollouts: ``evaluate_policy`` rollouts, one list of states per episode.
        env: The evaluation env.
        **kwargs: Keyword arguments of :func:`rollout_trials` and
            :func:`summarize_trials`.

    Returns:
        Metric name to value.
    """
    summary_keys = ("conditions", "n_bins", "time_key")
    summary_kwargs = {k: kwargs.pop(k) for k in summary_keys if k in kwargs}
    return summarize_trials(rollout_trials(rollouts, env, **kwargs), **summary_kwargs)
