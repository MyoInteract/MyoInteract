"""Out-of-distribution evaluation: a run's policy on the setup in the editor."""

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from myoInteract.studio.examples import example_dir  # noqa: E402
from myoInteract.studio.ood import task_changes  # noqa: E402
from myoInteract.studio.server import create_app  # noqa: E402

TRAINED = {
    "env": {
        "model_path": "arm.xml",
        "ctrl_dt": 0.05,
        "task_config": {
            "obs_keys": ["qpos"],
            "max_duration": 4.0,
            "targets": {"num_targets": 1, "target_0": {"size": [0.01, 0.01]}},
            "weighted_reward_keys": {"distance": 1.0},
        },
    },
    "vision": {"enabled": False},
}


def _setup(**edits) -> dict:
    setup = copy.deepcopy(TRAINED)
    for path, value in edits.items():
        node = setup
        *parents, leaf = path.split("__")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return setup


def test_task_changes_lists_the_task_settings_that_differ() -> None:
    assert task_changes(TRAINED, copy.deepcopy(TRAINED)) == []
    setup = _setup(env__task_config__max_duration=6.0)
    setup["env"]["task_config"]["targets"]["target_0"]["size"] = [0.005, 0.005]
    setup["env"]["task_config"]["weighted_reward_keys"]["distance"] = (
        2.0  # not evaluated
    )
    assert task_changes(TRAINED, setup) == [
        "max_duration: 4.0 -> 6.0",
        "targets.target_0.size: [0.01, 0.01] -> [0.005, 0.005]",
    ]


@pytest.mark.parametrize(
    "edits",
    [
        {"env__model_path": "hand.xml"},
        {"env__ctrl_dt": 0.02},
        {"env__task_config__obs_keys": ["qpos", "qvel"]},
        {"vision": {"enabled": True}},
    ],
)
def test_task_changes_refuses_what_the_policy_depends_on(edits: dict) -> None:
    with pytest.raises(ValueError, match="cannot run on it"):
        task_changes(TRAINED, _setup(**edits))


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list:
    import myoInteract.studio.runs as runs_module

    calls: list = []

    def start(self, kind: str, args: list[str], inputs: dict | None = None) -> dict:
        calls.append((kind, args, inputs))
        return {"id": f"job{len(calls)}", "alive": True}

    monkeypatch.setattr(runs_module.JobManager, "start", start)
    return calls


def test_evaluation_and_viewer_on_the_editors_setup(
    tmp_path: Path, started: list
) -> None:
    client = TestClient(create_app(tmp_path, allow_origins=[], viewer_port=1))
    preset = json.loads((example_dir("pointing") / "preset.json").read_text())
    body = {"run": "examples/pointing", "checkpoint": "model_final.onnx"}

    assert client.post("/api/evaluations", json=body).status_code == 200
    assert "--task" not in started[-1][1] and started[-1][2] is None

    preset["target_1"]["sphere"]["size_range"] = [0.005, 0.005]
    preset["task"]["max_duration"] = 6
    r = client.post("/api/evaluations", json={**body, "preset": preset})
    assert r.status_code == 200, r.text
    kind, args, inputs = started[-1]
    assert kind == "evaluate" and args[-2:] == ["--task", "{dir}/task.json"]
    assert inputs["task.json"]["changes"] == [
        "max_duration: 4.0 -> 6.0",
        "targets.target_0.size: [0.01, 0.01] -> [0.005, 0.005]",
    ]
    assert "env.task_config.max_duration=6" in " ".join(
        inputs["task.json"]["overrides"]
    )

    r = client.post("/api/viewer", json={**body, "preset": preset})
    assert r.status_code == 200, r.text
    assert started[-1][0] == "stream" and "--task" in started[-1][1]

    preset["bm"]["ctrl_dt"] = 0.02
    r = client.post("/api/evaluations", json={**body, "preset": preset})
    assert r.status_code == 422 and "ctrl_dt" in r.json()["detail"]


def test_added_and_removed_targets_are_one_line_each() -> None:
    setup = copy.deepcopy(TRAINED)
    targets = setup["env"]["task_config"]["targets"]
    targets["num_targets"] = 2
    targets["target_1"] = {"size": [0.01, 0.01]}
    assert task_changes(TRAINED, setup) == [
        "targets.num_targets: 1 -> 2",
        "targets.target_1: added",
    ]
    assert task_changes(setup, TRAINED)[1] == "targets.target_1: removed"


def test_colour_round_off_is_no_change() -> None:
    trained = _setup(env__task_config__targets__target_0={"rgb": [0.8, 0.1, 0.1]})
    setup = _setup(env__task_config__targets__target_0={"rgb": [0.8, 0.10196, 0.10196]})
    assert task_changes(trained, setup) == []
