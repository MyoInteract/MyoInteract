"""Movement metrics of MyoUser evaluation rollouts (myosuite.envs.myo.myouser.pointing_metrics).

A synthetic two-target episode that follows MyoUserUniversal's phase/dwell rules, and a
short real MyoUserUniversal rollout through evaluate_policy (CPU; jit takes ~2 min).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from myosuite.envs.myo.myouser import pointing_metrics as pm
from myosuite.utils import movement_metrics as mm

DT = 0.05


@pytest.mark.parametrize(
    "phase, done, n_targets, expected",
    [
        ([0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1], 2, [(0, 2, True), (2, 5, True)]),
        ([0, 0, 0], [0, 0, 0], 1, [(0, 2, False)]),
        ([0, 0, 0, 0], [0, 0, 0, 1], 1, [(0, 3, True)]),
        ([0, 1, 1, 1], [0, 0, 0, 0], 3, [(0, 1, True), (1, 3, False)]),
        ([0, 0, 1], [0, 0, 0], 2, [(0, 2, True)]),  # truncated at a selection
    ],
)
def test_target_segments(phase, done, n_targets, expected):
    assert pm.target_segments(phase, done, n_targets) == expected


def _synthetic_episode(dwell_steps: int = 3, include_reset: bool = True):
    """Two sequential sphere targets scored with MyoUserUniversal's rules."""
    targets = np.array([[0.2, 0.0, 0.0], [0.2, 0.15, 0.0]])
    radii = np.array([0.02, 0.03])
    t = np.arange(0.0, 2.0 + 1e-9, DT)
    ee = (
        mm.minimum_jerk(np.zeros(3), targets[0], 0.5, t).position
        + mm.minimum_jerk(np.zeros(3), targets[1] - targets[0], 0.5, t - 0.8).position
    )
    target_info = {}
    for k in range(2):
        target_info[f"target_{k}/target_pos"] = targets[k]
        target_info[f"target_{k}/target_size"] = radii[k]
    states, phase, counter = [], 0, 0
    for time, pos in zip(t, ee):
        # The dwell counter of the active target; it carries over a phase change.
        inside = np.linalg.norm(pos - targets[phase]) < radii[phase]
        counter = counter + 1 if inside else 0
        selected = counter >= dwell_steps
        done = selected and phase == 1
        next_phase = (0 if phase == 1 else phase + 1) if selected else phase
        states.append(
            SimpleNamespace(
                info={
                    "ee_pos": pos,
                    "phase": next_phase,
                    "steps_inside_target": float(counter),
                    **target_info,
                },
                data=SimpleNamespace(time=time, act=np.full(4, 0.5)),
                done=float(done),
            )
        )
        phase = next_phase
        if done:
            break
    return (states if include_reset else states[1:]), targets, ee


def test_rollout_trials_on_a_synthetic_two_target_episode():
    dwell = 3
    states, targets, ee = _synthetic_episode(dwell)
    phase = np.array([s.info["phase"] for s in states])
    e0, e1 = int(np.flatnonzero(phase == 1)[0]), len(states) - 1
    ((first, second),) = pm.rollout_trials(
        [states], dt=DT, dwell_times=[dwell * DT] * 2
    )
    assert first["success"] == second["success"] == 1.0
    assert first["trial_time"] == pytest.approx(e0 * DT)
    assert second["trial_time"] == pytest.approx((e1 - e0) * DT)
    assert first["time_to_acquire"] == pytest.approx((e0 - dwell + 1) * DT)
    assert first["distance"] == pytest.approx(np.linalg.norm(targets[0]))
    assert second["distance"] == pytest.approx(np.linalg.norm(targets[1] - ee[e0]))
    assert second["target_entries"] == 1.0  # its first sample is inside target 0 only
    assert first["effort"] == pytest.approx(0.25)
    assert first["width"] == pytest.approx(0.04)
    summary = pm.summarize_trials([[first, second]] * 2)
    assert summary["trials"] == 4.0 and summary["success"] == 1.0
    assert summary["target_1/trial_time"] == pytest.approx(second["trial_time"])
    assert summary["fitts_trials"] == 4.0
    for key in ("fitts_throughput", "fitts_slope", "ldlj", "sparc", "peak_speed"):
        assert key in summary


def test_rollout_trials_warns_without_the_reset_state():
    states, _, _ = _synthetic_episode(include_reset=False)
    with pytest.warns(UserWarning, match="include_reset_state"):
        trials = pm.rollout_trials([states], dt=DT)
    assert len(trials[0]) == 2


# ── Real MyoUserUniversal rollout ────────────────────────────────────────────


@pytest.fixture(scope="module")
def real_rollouts():
    jax = pytest.importorskip("jax")
    pytest.importorskip("mujoco_playground")
    import jax.numpy as jp
    from mujoco_playground import registry

    from myosuite.envs.myo.myouser.evaluate import evaluate_policy
    from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
    from myosuite.train.utils.wrapper import _maybe_wrap_env_for_evaluation

    # One pointing target large enough to contain the fingertip from the reset on.
    config = load_config_interactive(
        [
            "env=universal",
            "env/task_config/targets=single",
            "wandb=disabled",
            "env.task_config.targets.target_0.size=[2.0,2.0]",
        ]
    )
    env = registry.load("MyoUserUniversal", config=config.env)
    eval_env, _ = _maybe_wrap_env_for_evaluation(eval_env=env, seed=0)
    n = 2
    kwargs = dict(
        eval_env=eval_env,
        jit_inference_fn=lambda obs, keys: (jp.zeros((n, env.action_size)), {}),
        jit_reset=jax.jit(eval_env.eval_reset),
        jit_step=jax.jit(eval_env.step),
        seed=0,
        n_episodes=n,
        ep_length=8,
    )
    with_reset = evaluate_policy(**kwargs, include_reset_state=True)[0]
    default = evaluate_policy(**kwargs)[0]
    return env, eval_env, with_reset, default


def test_include_reset_state_only_prepends_the_reset(real_rollouts):
    _, _, with_reset, default = real_rollouts
    for full, steps in zip(with_reset, default):
        assert float(full[0].data.time) == 0.0
        assert len(full) == len(steps) + 1
        for a, b in zip(full[1:], steps):
            np.testing.assert_array_equal(
                np.asarray(a.data.qpos), np.asarray(b.data.qpos)
            )
            np.testing.assert_array_equal(
                np.asarray(a.info["ee_pos"]), np.asarray(b.info["ee_pos"])
            )


def test_rollout_trials_on_a_real_myouser_rollout(real_rollouts):
    env, eval_env, with_reset, _ = real_rollouts
    dwell_steps = env.target_objs[0].dwell_steps
    trials = pm.rollout_trials(with_reset, eval_env)
    for episode, (trial,) in zip(with_reset, trials):
        # The env selects the target after dwell_steps samples inside, the reset included.
        assert float(episode[-1].done) == 1.0
        assert float(episode[-1].metrics["target_final_completed"]) == 1.0
        assert len(episode) == dwell_steps
        assert trial["success"] == 1.0
        assert trial["trial_time"] == pytest.approx((dwell_steps - 1) * env.dt)
        assert trial["time_to_target"] == 0.0 and trial["time_to_acquire"] == 0.0
        start = np.asarray(episode[0].info["ee_pos"])
        target = np.asarray(episode[0].info["target_0/target_pos"])
        assert trial["distance"] == pytest.approx(
            np.linalg.norm(target - start), rel=1e-6
        )
        assert trial["width"] == pytest.approx(4.0)
        assert trial["index_of_difficulty"] == pytest.approx(
            math.log2(trial["distance"] / 4.0 + 1.0), rel=1e-6
        )
        assert np.isfinite(trial["effort"])
    summary = pm.rollout_movement_metrics(with_reset, eval_env)
    assert summary["trials"] == 2.0 and summary["success"] == 1.0
    assert summary["target_0/trial_time"] == pytest.approx((dwell_steps - 1) * env.dt)
    assert summary["fitts_trials"] == 2.0 and math.isnan(summary["fitts_slope"])
