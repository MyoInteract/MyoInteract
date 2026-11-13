"""W&B download cache isolation with real checkpoint-name helpers and a fake API."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from myoInteract.onnx_checkpoint_extra import download_all_wandb_onnx_checkpoints  # noqa: E402


def test_same_run_id_in_different_projects_caches_policy_and_config_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads = []

    class Run:
        def __init__(self, name):
            self.name = name

        def files(self):
            return [
                SimpleNamespace(name=name)
                for name in [
                    "model_2.onnx",
                    "model_final.onnx",
                    "myointeract_config.json",
                ]
            ]

        def file(self, name):
            def download(root, *, replace):
                downloads.append((self.name, name))
                (Path(root) / name).write_text(self.name, encoding="utf-8")

            return SimpleNamespace(download=download)

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Api=lambda: SimpleNamespace(run=Run))
    )
    legacy = tmp_path / "wandb_checkpoints/id"
    legacy.mkdir(parents=True)
    (legacy / "model_final.onnx").write_bytes(b"unscoped policy")
    first, first_cached = download_all_wandb_onnx_checkpoints(tmp_path, "team/a/id")
    second, second_cached = download_all_wandb_onnx_checkpoints(
        tmp_path, "https://wandb.ai/team/b/runs/id?workspace=user"
    )
    assert first[0].parent != second[0].parent
    assert first_cached == second_cached == [False, False]
    for paths, expected in [(first, "team/a/id"), (second, "team/b/id")]:
        assert [path.name for path in paths] == ["model_2.onnx", "model_final.onnx"]
        assert all(path.read_text() == expected for path in paths)
        assert (paths[0].parent / "myointeract_config.json").read_text() == expected
    cached, was_cached = download_all_wandb_onnx_checkpoints(
        tmp_path, Path("team/b/id")
    )
    assert cached == second and was_cached == [True, True]
    assert len(downloads) == 6
