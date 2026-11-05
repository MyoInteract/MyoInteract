"""Posture retention and successful-episode commits without GPU physics."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def task_code() -> dict:
    if importlib.util.find_spec("mjlab") is not None:
        from myoInteract import register_mjlab_myointeract_tasks

        return vars(register_mjlab_myointeract_tasks)
    # The buffers and reset adapter only require torch. Execute their actual
    # source when the optional native training stack is unavailable (Windows).
    source = (
        Path(__file__).resolve().parents[1]
        / "myoInteract/register_mjlab_myointeract_tasks.py"
    )
    names = {"VisitedSuccessBuffer", "PostureBuffer", "VisitedSuccessReset"}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in names
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "ManagerTermBase": object}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def _buffer(code: dict, kind: str, capacity: int, device: str = "cpu"):
    if kind == "visited":
        return code["VisitedSuccessBuffer"](
            2, capacity, [], torch.zeros(2), device=device
        )
    return code["PostureBuffer"](2, 3, 4, capacity, device=device)


@pytest.mark.parametrize("kind", ["visited", "posture"])
@pytest.mark.parametrize("capacity", [1, 3, 7])
def test_buffer_batches_match_sequential_insertion_across_wraps_and_overflow(
    task_code: dict, kind: str, capacity: int, device: str
) -> None:
    buffer = _buffer(task_code, kind, capacity, device)
    expected = torch.zeros(capacity, device=device)
    pointer = 1 % capacity if kind == "visited" else 0
    size = 1 if kind == "visited" else 0
    next_row = 1
    for n in [0, 1, 2, 11, 4, 0, 23, 3]:
        rows = torch.arange(next_row, next_row + n, dtype=torch.float32, device=device)
        qpos = rows[:, None].expand(-1, 2)
        if kind == "visited":
            buffer.add(qpos, rows + 300)
        else:
            buffer.add(
                qpos,
                (rows + 100)[:, None].expand(-1, 3),
                (rows + 200)[:, None].expand(-1, 4),
            )
        for row in rows:
            expected[pointer] = row
            pointer = (pointer + 1) % capacity
            size = min(capacity, size + 1)
        next_row += n
        assert buffer.size == size
        assert buffer._ptr == pointer
        torch.testing.assert_close(buffer._qpos[:, 0], expected)
        torch.testing.assert_close(buffer._qpos[:, 1], expected)
        valid = expected != 0
        if kind == "visited":
            torch.testing.assert_close(
                buffer._relative_time[valid], expected[valid] + 300
            )
        else:
            torch.testing.assert_close(
                buffer._qvel[valid], (expected[valid] + 100)[:, None].expand(-1, 3)
            )
            torch.testing.assert_close(
                buffer._act[valid], (expected[valid] + 200)[:, None].expand(-1, 4)
            )


@pytest.mark.parametrize("kind", ["visited", "posture"])
@pytest.mark.parametrize("capacity", [0, -1])
def test_buffer_capacity_must_be_positive(task_code: dict, kind: str, capacity: int):
    with pytest.raises(ValueError, match="capacity must be positive"):
        _buffer(task_code, kind, capacity)


@pytest.mark.parametrize("kind", ["visited", "posture"])
def test_oversized_batches_write_each_destination_only_once(
    task_code: dict, kind: str, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = _buffer(task_code, kind, 3, device)
    destinations = []
    original = torch.Tensor.__setitem__

    def record(tensor, index, value):
        if tensor is buffer._qpos:
            destinations.append(index.clone())
        return original(tensor, index, value)

    monkeypatch.setattr(torch.Tensor, "__setitem__", record)
    rows = torch.arange(11, dtype=torch.float32, device=device)
    qpos = rows[:, None].expand(-1, 2)
    if kind == "visited":
        buffer.add(qpos, rows)
    else:
        buffer.add(qpos, rows[:, None].expand(-1, 3), rows[:, None].expand(-1, 4))
    assert len(destinations) == 1
    assert destinations[0].numel() == 3
    assert torch.unique(destinations[0]).numel() == 3
    torch.testing.assert_close(
        buffer._qpos[:, 0].sort().values, torch.tensor([8.0, 9.0, 10.0], device=device)
    )


def test_single_slot_visited_buffer_starts_with_a_sampleable_seed(
    task_code: dict, device: str
):
    buffer = task_code["VisitedSuccessBuffer"](
        2, 1, ["muscle"], torch.tensor([2.0, 3.0]), device=device
    )
    assert buffer.size == 1
    assert buffer._ptr == 0
    torch.testing.assert_close(
        buffer.sample(3), torch.tensor([[2.0, 3.0]], device=device).expand(3, -1)
    )
    assert buffer.sample(0) is None
    buffer.add(
        torch.tensor([[4.0, 5.0], [6.0, 7.0]], device=device),
        torch.tensor([0.5, 0.75], device=device),
    )
    torch.testing.assert_close(
        buffer.sample(1), torch.tensor([[6.0, 7.0]], device=device)
    )
    assert buffer._relative_time.tolist() == [0.75]


def test_posture_buffer_preserves_selected_environment_order(
    task_code: dict, device: str
):
    buffer = task_code["PostureBuffer"](1, 1, 1, 3, device=device)
    rows = torch.arange(6, dtype=torch.float32, device=device)[:, None]
    buffer.add(rows, rows + 10, rows + 20, torch.tensor([5, 2, 4, 0], device=device))
    assert buffer._ptr == 1
    assert buffer.size == 3
    torch.testing.assert_close(
        buffer._qpos[:, 0], torch.tensor([0.0, 2.0, 4.0], device=device)
    )
    torch.testing.assert_close(buffer._qvel, buffer._qpos + 10)
    torch.testing.assert_close(buffer._act, buffer._qpos + 20)


@pytest.mark.parametrize("all_failed", [False, True])
@pytest.mark.parametrize("selection", ["subset", "all", "empty"])
def test_success_reset_batches_only_valid_rows_in_environment_then_snapshot_order(
    task_code: dict,
    monkeypatch: pytest.MonkeyPatch,
    all_failed: bool,
    device: str,
    selection: str,
):
    monkeypatch.setitem(task_code, "_UNIVERSAL_ENTITY_NAME", "universal_robot")
    monkeypatch.setitem(task_code, "_validate_initpose001_model", lambda *_args: None)
    n, snapshots, nq = 6, 4, 17
    buffer = task_code["VisitedSuccessBuffer"](
        nq, 4, [], torch.zeros(nq), device=device
    )
    tags = torch.arange(n * snapshots, dtype=torch.float32, device=device).reshape(
        n, snapshots
    )
    staged = tags[:, :, None].expand(-1, -1, nq).clone()
    counts = torch.tensor([3, 4, 0, 2, 4, 1], device=device)
    success = (
        torch.zeros(n, dtype=torch.bool, device=device)
        if all_failed
        else torch.tensor([True, False, True, True, False, True], device=device)
    )
    writes = []
    asset = SimpleNamespace(
        data=SimpleNamespace(
            data=SimpleNamespace(act=torch.zeros(n, 26, device=device))
        ),
        visited_success_buffer=buffer,
        visited_success_staging_count=counts.clone(),
        visited_success_staging_qpos=staged,
        visited_success_staging_rel_time=tags + 0.5,
        write_joint_state_to_sim=lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    env = SimpleNamespace(
        scene={"universal_robot": asset},
        num_envs=n,
        device=device,
        termination_manager=SimpleNamespace(get_term=lambda _name: success),
    )
    reset = task_code["VisitedSuccessReset"].__new__(task_code["VisitedSuccessReset"])
    reset._asset_cfg = SimpleNamespace(name="universal_robot", joint_ids=slice(None))
    env_ids = torch.tensor(
        [3, 1, 2, 5, 0]
        if selection == "subset"
        else list(range(n))
        if selection == "all"
        else [],
        device=device,
        dtype=torch.long,
    )
    reset(env, None if selection == "all" else env_ids)
    expected = torch.zeros(4, device=device)
    expected_time = torch.zeros(4, device=device)
    pointer, size = 1, 1
    for env_id in env_ids.tolist():
        if success[env_id]:
            for slot in range(int(counts[env_id])):
                expected[pointer] = tags[env_id, slot]
                expected_time[pointer] = tags[env_id, slot] + 0.5
                pointer = (pointer + 1) % 4
                size = min(4, size + 1)
    assert buffer.size == size
    assert buffer._ptr == pointer
    torch.testing.assert_close(buffer._qpos, expected[:, None].expand(-1, nq))
    torch.testing.assert_close(buffer._relative_time, expected_time)
    expected_counts = counts.clone()
    expected_counts[env_ids] = 0
    torch.testing.assert_close(asset.visited_success_staging_count, expected_counts)
    assert len(writes) == (0 if selection == "empty" else 1)
    if writes:
        torch.testing.assert_close(writes[0][1]["env_ids"], env_ids)
