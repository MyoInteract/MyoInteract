"""Example policies of Studio's Evaluate step: one trained run per bundled task example.

Each example is a folder ``examples/<name>/`` that works as a run directory (``model_final.onnx``,
``original_hydra_config.json``, ``myointeract_config.json``) and also holds what the page shows
without a backend: ``example.json`` (title, stored evaluation, replay header), ``preset.json`` (the
task, to open in the editor) and ``replay.bin`` (a recorded live feed, see ``worker.live_frame``).

    python -m myoInteract.studio.examples --run logs/rsl_rl/myo_universal/<run> --name pointing
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXAMPLES = Path(__file__).parent / "examples"
PRESETS = Path(__file__).resolve().parents[2] / "myo_config" / "gradio_tasks"
# What a page without a backend needs (the hosted site ships these, not the checkpoints).
PAGE_FILES = ("example.json", "preset.json", "replay.bin")
# Written by studio.browser for examples the page can simulate itself.
SIM_FILES = ("sim.json", "sim_model.xml", "sim_policy.bin")


def example_dir(name: str, root: Path = EXAMPLES) -> Path | None:
    """Folder of the example ``name``, or None."""
    path = root / name
    return path if path.parent == root and (path / "example.json").is_file() else None


def list_examples(root: Path = EXAMPLES) -> list[dict]:
    """``example.json`` of every example, by name."""
    return [json.loads(p.read_text()) for p in sorted(root.glob("*/example.json"))]


def build(
    run: Path,
    name: str,
    title: str,
    out: Path = EXAMPLES,
    episodes: int = 1024,
    seconds: float = 20.0,
) -> dict:
    """Bundle a finished run as the example ``name``: checkpoint, configs, evaluation and replay."""
    from myoInteract.studio.runs import _yaml_int, summary  # noqa: PLC0415
    from myoInteract.studio.server import resolve_preset  # noqa: PLC0415
    from myoInteract.studio.worker import record  # noqa: PLC0415

    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "model_final.onnx", dest / "model_final.onnx")
    shutil.copy(PRESETS / f"{name}.json", dest / "preset.json")
    # The sidecars Studio writes for its own runs: the task is rebuilt from them.
    overrides, resolved = resolve_preset(json.loads((dest / "preset.json").read_text()))
    (dest / "original_hydra_config.json").write_text(json.dumps(overrides, indent=1))
    (dest / "myointeract_config.json").write_text(json.dumps(resolved, indent=1))
    # The evaluation runs as its own process, like a Studio job (one task registration per process).
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable,
            "-m",
            "myoInteract.studio.worker",
            "evaluate",
            "--log-dir",
            str(dest),
            "--episodes",
            str(episodes),
            "--video-seconds",
            "0",
            "--out",
            tmp,
        ]
        subprocess.run(cmd, check=True)
        evaluation = json.loads((Path(tmp) / "result.json").read_text())
    hello, frames = record(dest, "model_final.onnx", seconds)
    (dest / "replay.bin").write_bytes(frames)
    floats = (
        7 * len(hello["bodies"])
        + 3 * len(hello["targets"])
        + len(hello.get("muscles", []))
        + len(hello.get("joints", []))
    )
    steps = (
        (_yaml_int(run / "params" / "agent.yaml", "max_iterations") or 0)
        * (_yaml_int(run / "params" / "agent.yaml", "num_steps_per_env") or 0)
        * (_yaml_int(run / "params" / "env.yaml", "num_envs") or 0)
    )
    example = {
        "name": name,
        "title": title,
        "summary": summary(dest),
        "training_steps": steps or None,
        "evaluation": {k: v for k, v in evaluation.items() if k != "video"},
        "replay": {
            "file": "replay.bin",
            "floats": floats,
            "frames": len(frames) // (4 * floats),
            "hello": hello,
        },
    }
    (dest / "example.json").write_text(json.dumps(example, indent=1) + "\n")
    return example


def refresh_hello(name: str, root: Path = EXAMPLES) -> dict:
    """Rewrite an example's replay header from its task (the frames stay as recorded)."""
    import torch  # noqa: PLC0415

    from myoInteract.studio.worker import (
        build_env,
        live_hello,
        paint_targets,
        run_overrides,
    )  # noqa: PLC0415

    folder = root / name
    example = json.loads((folder / "example.json").read_text())
    env, config = build_env(
        run_overrides(folder), 1, "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    hello = live_hello(env, config, paint_targets(env, config))
    env.close()
    if hello["bodies"] != example["replay"]["hello"]["bodies"] or len(
        hello["targets"]
    ) != len(example["replay"]["hello"]["targets"]):
        raise ValueError(
            f"{name}: the task no longer matches its recording; build the example again"
        )
    for key in (
        "muscles",
        "joints",
    ):  # recorded before frames carried muscle activations or joint angles
        if key not in example["replay"]["hello"]:
            hello.pop(key)
    example["replay"]["hello"] = hello
    (folder / "example.json").write_text(json.dumps(example, indent=1) + "\n")
    return example


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, help="finished training run (log directory)")
    p.add_argument(
        "--refresh-hello",
        action="store_true",
        help="only rewrite the replay header of an existing example",
    )
    p.add_argument(
        "--name",
        required=True,
        help="bundled task example it was trained on, e.g. pointing",
    )
    p.add_argument("--title", help="shown in Studio (default: the name)")
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument(
        "--seconds", type=float, default=20.0, help="length of the recorded rollout"
    )
    args = p.parse_args()
    if args.refresh_hello:
        print(
            json.dumps(refresh_hello(args.name)["replay"]["hello"]["targets"], indent=1)
        )
        return
    if args.run is None:
        p.error("--run is needed to build an example")
    example = build(
        args.run,
        args.name,
        args.title or args.name,
        episodes=args.episodes,
        seconds=args.seconds,
    )
    print(json.dumps(example["evaluation"], indent=1))


if __name__ == "__main__":
    main()
