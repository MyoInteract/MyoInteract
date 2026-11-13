"""Remote run identity and Gradio cache lookup without a W&B connection."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from myoInteract.wandb_cache import (
    normalize_wandb_run_path,
    resolve_wandb_run_input,
    wandb_checkpoint_dir,
)


@pytest.mark.parametrize(
    "value",
    [
        "team/project/run-id",
        "team/project/runs/run-id/",
        "team\\project\\run-id",
        "https://wandb.ai/team/project/runs/run-id?workspace=user#overview",
    ],
)
def test_run_path_and_url_forms_share_a_cache(value: str, tmp_path: Path) -> None:
    assert normalize_wandb_run_path(value) == "team/project/run-id"
    assert wandb_checkpoint_dir(tmp_path, value) == wandb_checkpoint_dir(
        tmp_path, "team/project/run-id"
    )


def test_cache_identity_includes_entity_and_project(tmp_path: Path) -> None:
    inputs = ["team/a/same-id", "team/b/same-id", "other/a/same-id"]
    folders = [wandb_checkpoint_dir(tmp_path, run) for run in inputs]
    assert len(set(folders)) == len(inputs)
    assert all(folder.parent == tmp_path / "wandb_checkpoints" for folder in folders)
    assert all(len(folder.name) == 64 for folder in folders)


@pytest.mark.parametrize(
    "value",
    ["id", "team/project", "team/../id", "team//id", "https://example.com/t/p/r"],
)
def test_incomplete_or_invalid_remote_identity_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_wandb_run_path(value)


def test_remote_input_defaults_and_local_paths_remain_distinct(tmp_path: Path) -> None:
    local = tmp_path / "model_final.onnx"
    local.write_bytes(b"local policy")
    for value in [
        str(local),
        str(tmp_path),
        "C:/runs/policy",
        "./runs/policy",
        "run/model.pt",
    ]:
        assert resolve_wandb_run_input(value, "team", "default") is None
    assert resolve_wandb_run_input("id", "team", "default") == "team/default/id"
    assert (
        resolve_wandb_run_input("team/other/runs/id", "team", "default")
        == "team/other/id"
    )


def _ui_cache_callbacks(tmp_path: Path) -> dict:
    source = Path(__file__).resolve().parents[1] / "myoInteract/scripts/ui.py"
    names = {
        "_resolve_local_run_dir",
        "load_policy_input_changed",
        "_list_onnx_files_in_dir",
        "_onnx_dir_listing_sort_key",
        "_run_checkpoint_dir",
        "checkpoint_path_from_run_filename",
    }
    functions = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    logs = tmp_path / "logs"
    tracked = tmp_path / "tracked"
    logs.mkdir()
    tracked.mkdir()
    namespace = {
        "os": os,
        "Path": Path,
        "gr": SimpleNamespace(update=lambda **kwargs: kwargs),
        "LOG_DIR": str(logs),
        "CHECKPOINT_PATH": str(tracked),
        "WANDB_DEFAULT_ENTITY": "team",
        "WANDB_DEFAULT_PROJECT": "a",
        "get_available_checkpoints": lambda **_kwargs: ["None"],
        "resolve_wandb_run_input": resolve_wandb_run_input,
        "wandb_checkpoint_dir": wandb_checkpoint_dir,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


def test_gradio_dropdown_uses_the_full_run_identity_and_selected_cache_path(
    tmp_path: Path,
) -> None:
    ui = _ui_cache_callbacks(tmp_path)
    for run in ["team/a/id", "team/b/id"]:
        folder = wandb_checkpoint_dir(Path(ui["LOG_DIR"]), run)
        folder.mkdir(parents=True)
        (folder / "model_final.onnx").write_text(run)
    values = []
    for value in ["id", "https://wandb.ai/team/b/runs/id?workspace=user"]:
        run, file = ui["load_policy_input_changed"](value)
        values.append(run["value"])
        checkpoint = ui["checkpoint_path_from_run_filename"](
            run["value"], file["value"]
        )
        assert Path(checkpoint).read_text() == (
            "team/a/id" if value == "id" else "team/b/id"
        )
    assert values[0] != values[1]


def test_gradio_does_not_auto_select_legacy_unscoped_cache(tmp_path: Path) -> None:
    ui = _ui_cache_callbacks(tmp_path)
    legacy = Path(ui["LOG_DIR"]) / "wandb_checkpoints/id"
    legacy.mkdir(parents=True)
    (legacy / "model_final.onnx").write_bytes(b"unknown origin")
    assert ui["load_policy_input_changed"]("team/a/id") == ({}, {})


def test_gradio_preserves_an_external_local_directory_selection(tmp_path: Path) -> None:
    ui = _ui_cache_callbacks(tmp_path)
    external = tmp_path / "outside the log root"
    external.mkdir()
    (external / "model_final.onnx").write_bytes(b"local policy")
    run, file = ui["load_policy_input_changed"](str(external))
    checkpoint = ui["checkpoint_path_from_run_filename"](run["value"], file["value"])
    assert Path(checkpoint) == external / "model_final.onnx"
