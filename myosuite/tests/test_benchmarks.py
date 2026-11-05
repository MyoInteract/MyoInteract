"""Benchmark comparisons detect changed rewards without comparing unlike tasks."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from myosuite.benchmarks.regression import compare_reports, summarize_rollouts


def state(time, reward, success=False):
    return SimpleNamespace(data=SimpleNamespace(time=time), reward=reward,
                           metrics={"distance_reward": reward}, info={"task_completed": success})


def test_benchmark_retains_terminal_rewards_and_success():
    summary = summarize_rollouts([[state(0, 0), state(0.05, -1), state(0.1, 7, True)],
                                  [state(0.05, -2)]])
    assert summary["success_rate"] == 0.5
    assert summary["mean_reward"] == 2
    assert summary["episodes"][0]["reward"] == 6
    assert summary["episodes"][0]["steps"] == 2


def test_benchmark_rejects_inconsistent_reward_components():
    broken = state(0.05, 7)
    broken.metrics["distance_reward"] = 0
    with pytest.raises(ValueError, match="components"):
        summarize_rollouts([[broken]])


def test_comparison_detects_reward_regressions_and_requires_same_protocol():
    reference = {
        "protocol": {"policy": "zero", "seeds": [0], "episodes_per_seed": 1, "max_steps": 2},
        "cases": [{"task": "ar", "checkpoint": None, "config_sha256": "same", "summary": summarize_rollouts([[state(0.05, -2)]])}],
    }
    actual = deepcopy(reference)
    assert compare_reports(actual, reference) == []
    actual["cases"][0]["summary"]["mean_reward"] = 0
    assert compare_reports(actual, reference)
    actual["protocol"]["seeds"] = [1]
    with pytest.raises(ValueError, match="protocol"):
        compare_reports(actual, reference)
    actual = deepcopy(reference)
    actual["cases"][0]["config_sha256"] = "different"
    with pytest.raises(ValueError, match="configuration"):
        compare_reports(actual, reference)
    actual = deepcopy(reference)
    actual["cases"][0]["checkpoint_sha256"] = "changed policy bytes"
    with pytest.raises(ValueError, match="content"):
        compare_reports(actual, reference)
    actual = deepcopy(reference)
    actual["cases"][0]["summary"]["mean_duration_seconds"] = 1
    assert "duration" in compare_reports(actual, reference)[0]
