"""Play routing and argument preservation without opening an interactive viewer."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def entrypoint() -> dict:
    if importlib.util.find_spec("mjlab") is not None:
        from myoInteract.scripts import train

        return vars(train)
    # Execute the actual entry-point source in a CPU Studio install, where the
    # optional training imports are unavailable. Native CI imports the module.
    source = Path(__file__).resolve().parents[1] / "myoInteract/scripts/train.py"
    functions = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in {"main", "_pop_play_flag"}
    ]
    namespace = {"os": os, "sys": sys, "json": json, "Path": Path}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


def _setup(
    entrypoint: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> list:
    preset = tmp_path / "preset.json"
    preset.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["myo_train.exe", "--config", str(preset), *args])
    monkeypatch.setitem(entrypoint, "gradio_json_to_overrides", lambda _raw: [])
    monkeypatch.setitem(
        entrypoint,
        "load_config_interactive",
        lambda _: SimpleNamespace(run=SimpleNamespace(seed=14)),
    )
    monkeypatch.setitem(entrypoint, "register_mjlab_myointeract_task", lambda _: None)
    calls = []

    def cli(_config, *, args, **_kwargs):
        calls.append(("args", list(args)))
        return SimpleNamespace(viewer="native" if "--viewer=native" in args else "auto")

    monkeypatch.setitem(entrypoint, "tyro", SimpleNamespace(cli=cli))
    monkeypatch.setitem(entrypoint, "mjlab", SimpleNamespace(TYRO_FLAGS=()))
    play = ModuleType("myosuite.viz.mjlab_play")
    play.PlayConfig = object
    play.run_play = lambda task, cfg: calls.append(("play", task, cfg.viewer))
    monkeypatch.setitem(sys.modules, play.__name__, play)
    base = ModuleType("mjlab.viewer.base")
    base.BaseViewer = type("BaseViewer", (), {"_execute_step": lambda _self: True})
    viewer = ModuleType("mjlab.viewer.viser.viewer")
    viewer.ViserPlayViewer = type(
        "ViserPlayViewer",
        (),
        {
            "reset_environment": lambda _self: None,
            "_update_status_display": lambda _self: None,
        },
    )
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, viewer.__name__, viewer)
    return calls


@pytest.mark.parametrize("flag", [["--play"], ["--play", "True"], ["--play=true"]])
def test_play_forwards_every_viewer_argument_and_selects_the_registered_task(
    entrypoint: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: list[str]
) -> None:
    options = ["--checkpoint-file", "a checkpoint/model_final.onnx", "--viewer=native"]
    calls = _setup(entrypoint, tmp_path, monkeypatch, [*flag, *options])
    entrypoint["main"]()
    assert calls == [
        ("args", options),
        ("play", "myoInteractUniversal-v0", "native"),
    ]
    assert sys.argv == ["myo_train", *options]


def test_bare_play_at_end_of_command_routes_without_popping_a_missing_value(
    entrypoint: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _setup(entrypoint, tmp_path, monkeypatch, ["--play"])
    entrypoint["main"]()
    assert calls == [("args", []), ("play", "myoInteractUniversal-v0", "auto")]


@pytest.mark.parametrize("flag", [["--play", "False"], ["--play=false"]])
def test_explicit_false_preserves_training_options_and_does_not_open_a_viewer(
    entrypoint: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: list[str]
) -> None:
    options = ["--agent.max-iterations", "12"]
    calls = _setup(entrypoint, tmp_path, monkeypatch, [*flag, *options])

    class TrainingSelected(Exception):
        pass

    def defaults(task):
        assert task == "myoInteractUniversal-v0"
        raise TrainingSelected

    monkeypatch.setitem(
        entrypoint, "MjlabTrainConfig", SimpleNamespace(from_task=defaults)
    )
    with pytest.raises(TrainingSelected):
        entrypoint["main"]()
    assert sys.argv[1:] == options
    assert calls == []


def test_invalid_explicit_play_value_is_reported_before_task_registration(
    entrypoint: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _setup(entrypoint, tmp_path, monkeypatch, ["--play=maybe"])
    with pytest.raises(ValueError, match="true or false"):
        entrypoint["main"]()
    assert calls == []


def test_play_uses_the_native_play_config_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mjlab")
    from myoInteract.scripts import train
    from myosuite.viz import mjlab_play

    calls = []
    monkeypatch.setattr(train, "register_mjlab_myointeract_task", lambda _: None)
    monkeypatch.setattr(
        mjlab_play, "run_play", lambda task, cfg: calls.append((task, cfg))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "myo_train",
            "--play",
            "--viewer=native",
            "--checkpoint-file",
            "saved policy.onnx",
        ],
    )
    train.main()
    assert len(calls) == 1
    task, cfg = calls[0]
    assert task == "myoInteractUniversal-v0"
    assert isinstance(cfg, mjlab_play.PlayConfig)
    assert cfg.viewer == "native" and cfg.checkpoint_file == "saved policy.onnx"
