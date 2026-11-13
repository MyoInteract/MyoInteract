import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig as MjlabTrainConfig
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from myosuite.integrations.musclemimic.mjlab_policy_runner import (
    OnnxCheckpointingMjlabRunner,
)
from myosuite.utils.onnx_checkpoint import extract_checkpoint_from_onnx

from myoInteract.myointeract_hydra_cli import (
    training_task_overrides,
    load_config_interactive,
)
from myoInteract.register_mjlab_myointeract_tasks import (
    register_mjlab_myointeract_task,
)
from myoInteract.training_lifecycle import (
    BoundaryLogger,
    StopRequest,
    TrainingStopped,
    copy_checkpoint,
    remaining_iterations,
    save_onnx_atomic,
    write_json,
)


@dataclass(frozen=True)
class TrainConfig(MjlabTrainConfig):
    """Native training options plus explicit continuation and cooperative stop."""

    resume_checkpoint: str | None = None
    """Explicit local ONNX bundle. Restores optimizer; starts fresh simulation/RNG."""
    additional_iterations: int | None = None
    """Explicit extension beyond the source checkpoint instead of an absolute target."""
    stop_file: str | None = None
    """Studio requests a checkpointed stop by creating this file."""


class TrainingRunner(OnnxCheckpointingMjlabRunner):
    """Keep the native checkpoint payload and publish its ONNX file atomically."""

    def save(self, path: str, infos: object = None) -> None:
        upload = self.cfg.get("upload_model", False)
        self.cfg["upload_model"] = False
        try:
            save_onnx_atomic(super().save, path, infos)
        finally:
            self.cfg["upload_model"] = upload
        if upload:
            self.logger.save_model(
                str(Path(path).with_suffix(".onnx")), self.current_learning_iteration
            )
        self._last_checkpoint = Path(path).with_suffix(".onnx")

    def load(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Restore the native learner while keeping the new simulator state.

        Fatigue from the source episode cannot be restored coherently without
        that episode's pose, task state and RNG. Skip the ONNX fatigue sidecar;
        the native learner and global curriculum counter still restore normally.
        """
        kwargs.setdefault("map_location", self.device)
        if Path(path).suffix == ".onnx":
            native, _metadata, temporary = extract_checkpoint_from_onnx(path)
            try:
                return super().load(str(native), **kwargs)
            finally:
                if temporary is not None:
                    temporary.cleanup()
        return super().load(path, **kwargs)


# ---------------------------------------------------------------------------
# Config selection: --config <path> strips the flag before tyro sees argv.
# Falls back to the bundled public-display-4-buttons config.
# ---------------------------------------------------------------------------
_DEFAULT_CFG = (
    Path(__file__).resolve().parents[2]
    / "myo_config/gradio_tasks/public-display-4-buttons.json"
)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
def _pop_play_flag(argv: list[str]) -> bool:
    """Consume the play switch without consuming the next viewer option.

    Keep accepting the legacy ``--play True/False`` form as well as a bare
    switch. Explicit false values leave the remaining training arguments intact.
    """
    play = False
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "--play":
            argv.pop(index)
            play = True
            if index < len(argv) and argv[index].lower() in {"true", "false"}:
                play = argv.pop(index).lower() == "true"
        elif argument.startswith("--play="):
            value = argv.pop(index).split("=", 1)[1].lower()
            if value not in {"true", "false"}:
                raise ValueError("--play expects true or false when given a value.")
            play = value == "true"
        else:
            index += 1
    return play


def main() -> None:
    """Console entry point ``myo_train`` (``--play`` opens the viewer instead)."""
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="backslashreplace")
    play = _pop_play_flag(sys.argv)
    if "--config" in sys.argv:
        index = sys.argv.index("--config")
        if index + 1 == len(sys.argv):
            raise ValueError("--config requires a JSON preset path.")
        cfg_path = Path(sys.argv.pop(index + 1))
        sys.argv.pop(index)
    else:
        cfg_path = _DEFAULT_CFG
    with cfg_path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    recipe_overrides = training_task_overrides(raw, sys.argv[1:])
    config = load_config_interactive(recipe_overrides)
    # The seed also participates in the task's model recipe, which is captured
    # during registration (before tyro can construct its environment defaults).
    # Resolve this one native option early so saved/evaluated task recipes agree.
    for index, value in enumerate(sys.argv[1:], start=1):
        if value == "--agent.seed":
            if index + 1 == len(sys.argv):
                raise ValueError("--agent.seed requires an integer.")
            config.run.seed = int(sys.argv[index + 1])
        elif value.startswith("--agent.seed="):
            config.run.seed = int(value.split("=", 1)[1])
    register_mjlab_myointeract_task(config)
    if play:
        from myosuite.viz.mjlab_play import PlayConfig, run_play

        sys.argv[0] = sys.argv[0].removesuffix(".exe")
        args = tyro.cli(PlayConfig, args=sys.argv[1:], config=mjlab.TYRO_FLAGS)
        # Patch the viser viewer to accumulate a "Total play time" counter
        # across episode resets (step_count resets to 0 on each reset).
        if args.viewer in {"auto", "viser"}:
            from mjlab.viewer.base import BaseViewer
            from mjlab.viewer.viser.viewer import ViserPlayViewer

            _orig_execute = BaseViewer._execute_step
            _orig_viser_reset = ViserPlayViewer.reset_environment
            _orig_update_status = ViserPlayViewer._update_status_display

            def _execute_with_global_count(self) -> bool:
                result = _orig_execute(self)
                if result:
                    self._global_step_count = getattr(self, "_global_step_count", 0) + 1
                return result

            def _viser_reset_preserve_global(self) -> None:
                saved = getattr(self, "_global_step_count", 0)
                _orig_viser_reset(self)
                self._global_step_count = saved

            def _update_status_with_play_time(self) -> None:
                _orig_update_status(self)
                global_steps = getattr(self, "_global_step_count", 0)
                total_time = global_steps * self.env.unwrapped.step_dt
                self._status_html.content = self._status_html.content.replace(
                    "</div>",
                    f"<br/><strong>Total play time:</strong> {total_time:.1f}s<br/>\n      </div>",
                    1,
                )

            BaseViewer._execute_step = _execute_with_global_count  # type: ignore[method-assign]
            ViserPlayViewer.reset_environment = _viser_reset_preserve_global  # type: ignore[method-assign]
            ViserPlayViewer._update_status_display = _update_status_with_play_time  # type: ignore[method-assign]

        run_play("myoInteractUniversal-v0", args)

    else:
        # This script only ever trains myoInteractUniversal-v0 (registered above), so skip
        # mjlab.scripts.train's generic two-stage "pick a task name, defer the rest" CLI:
        # with tyro==1.0.8 its first stage fails on any extra argument (e.g. --agent.resume).
        chosen_task = "myoInteractUniversal-v0"

        defaults = MjlabTrainConfig.from_task(chosen_task)
        defaults.agent.seed = int(config.run.seed)
        args = tyro.cli(
            TrainConfig,
            args=sys.argv[1:],
            default=TrainConfig(
                env=defaults.env,
                agent=defaults.agent,
            ),
            config=mjlab.TYRO_FLAGS,
        )

        configure_torch_backends()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if args.additional_iterations is not None and args.additional_iterations < 0:
            raise ValueError("--additional-iterations must be non-negative.")
        if args.additional_iterations is not None and not (
            args.agent.resume or args.resume_checkpoint
        ):
            raise ValueError("--additional-iterations requires a resume checkpoint.")
        remaining_iterations(args.agent.max_iterations, 0)
        if (
            min(
                args.env.scene.num_envs,
                args.agent.num_steps_per_env,
                args.agent.save_interval,
            )
            < 1
        ):
            raise ValueError(
                "Environment count, rollout length and save interval must all be positive."
            )
        if args.agent.max_iterations == 0 and not (
            args.agent.resume or args.resume_checkpoint
        ):
            raise ValueError(
                "The training budget contains no complete update; increase total steps or --agent.max-iterations."
            )
        recipe_decimation = int(round(config.env.ctrl_dt / config.env.sim_dt))
        if not math.isclose(
            recipe_decimation * config.env.sim_dt,
            config.env.ctrl_dt,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Task ctrl_dt must be an integer multiple of sim_dt.")
        if (
            args.env.decimation != recipe_decimation
            or not math.isclose(
                args.env.sim.mujoco.timestep, config.env.sim_dt, rel_tol=1e-9
            )
            or not math.isclose(
                args.env.episode_length_s,
                config.env.task_config.max_duration,
                rel_tol=1e-9,
            )
        ):
            raise ValueError(
                "Native environment timing overrides would disagree with the captured task/action recipe. "
                "Set env.ctrl_dt, env.sim_dt and env.task_config.max_duration in the --config preset's "
                "extra_overrides instead."
            )
        args.env.seed = args.agent.seed
        log_root = Path(args.log_root) / args.agent.experiment_name

        # Resolve resume checkpoint BEFORE creating log_dir to avoid the new
        # directory being picked up as the "latest previous run".
        resume_path = None
        if args.resume_checkpoint:
            resume_path = Path(args.resume_checkpoint).expanduser().resolve(strict=True)
        elif args.agent.resume:
            ckpt_pattern = args.agent.load_checkpoint.replace(".pt", ".onnx")
            resume_path = get_checkpoint_path(
                log_root, args.agent.load_run, ckpt_pattern
            )
            print(f"[INFO] Resuming from checkpoint: {resume_path}")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        run_name = (
            f"{timestamp}_{args.agent.run_name}" if args.agent.run_name else timestamp
        )
        log_dir = log_root / run_name
        log_dir.mkdir(parents=True, exist_ok=False)
        dump_yaml(log_dir / "params" / "env.yaml", asdict(args.env))
        dump_yaml(log_dir / "params" / "agent.yaml", asdict(args.agent))
        # Keep task reconstruction and the actual CLI-selected network architecture
        # with every run, including runs started without Studio.
        overrides = list(recipe_overrides)
        overrides.extend(
            [
                f"rl.num_envs={args.env.scene.num_envs}",
                f"rl.unroll_length={args.agent.num_steps_per_env}",
                f"run.seed={args.agent.seed}",
                f"env.sim_dt={args.env.sim.mujoco.timestep}",
                f"env.ctrl_dt={args.env.decimation * args.env.sim.mujoco.timestep}",
                f"env.task_config.max_duration={args.env.episode_length_s}",
            ]
        )
        effective = load_config_interactive(overrides)
        write_json(log_dir / "original_hydra_config.json", overrides)
        write_json(log_dir / "myointeract_config.json", effective.to_dict())
        write_json(log_dir / "agent_config.json", asdict(args.agent))
        lifecycle = {
            "status": "starting",
            "completed_iterations": 0,
            "target_iterations": args.agent.max_iterations,
            "resume_checkpoint": str(resume_path) if resume_path else None,
            "resume_semantics": "optimizer continuation; fresh simulator, task state, fatigue and RNG; global curriculum counter restored"
            if resume_path
            else None,
        }
        write_json(log_dir / "lifecycle.json", lifecycle)
        env = None
        logger = None
        try:
            env = ManagerBasedRlEnv(cfg=args.env, device=device)
            vec_env = RslRlVecEnvWrapper(env, clip_actions=args.agent.clip_actions)
            runner = TrainingRunner(
                vec_env, asdict(args.agent), str(log_dir), device, task_id=chosen_task
            )
            completed = 0
            if resume_path is not None:
                runner.load(str(resume_path))
                completed = int(runner.current_learning_iteration) + 1
                # RSL saves the last completed index and otherwise repeats that
                # index when loading. The next rollout belongs to the next index.
                runner.current_learning_iteration = completed
            target = (
                completed + args.additional_iterations
                if args.additional_iterations is not None
                else args.agent.max_iterations
            )
            count = remaining_iterations(target, completed)
            args.agent.max_iterations = target
            effective.rl.num_timesteps = (
                target * args.env.scene.num_envs * args.agent.num_steps_per_env
            )
            overrides.append(f"rl.num_timesteps={effective.rl.num_timesteps}")
            write_json(log_dir / "original_hydra_config.json", overrides)
            write_json(log_dir / "myointeract_config.json", effective.to_dict())
            write_json(log_dir / "agent_config.json", asdict(args.agent))
            dump_yaml(log_dir / "params" / "agent.yaml", asdict(args.agent))
            lifecycle.update(
                status="running",
                completed_iterations=completed,
                target_iterations=target,
            )
            write_json(log_dir / "lifecycle.json", lifecycle)
            with StopRequest(Path(args.stop_file) if args.stop_file else None) as stop:

                def on_update(value: int) -> None:
                    lifecycle["completed_iterations"] = value
                    stop.check()

                logger = BoundaryLogger(runner.logger, on_update)
                runner.logger = logger
                try:
                    stop.check()
                    if count:
                        runner.learn(
                            num_learning_iterations=count, init_at_random_ep_len=True
                        )
                    lifecycle["status"] = "completed"
                except TrainingStopped:
                    lifecycle["status"] = "stopped"
                # Keep a restored checkpoint's last-completed index when no
                # additional update ran (RSL's in-memory counter was advanced).
                if lifecycle["completed_iterations"]:
                    runner.current_learning_iteration = (
                        lifecycle["completed_iterations"] - 1
                    )
                    filename = (
                        "model_final.pt"
                        if lifecycle["status"] == "completed"
                        else "model_stopped.pt"
                    )
                    if lifecycle["status"] == "completed" and count:
                        copy_checkpoint(
                            runner._last_checkpoint,
                            log_dir / Path(filename).with_suffix(".onnx"),
                        )
                    elif (
                        lifecycle["status"] == "completed"
                        and resume_path is not None
                        and resume_path.suffix == ".onnx"
                    ):
                        copy_checkpoint(
                            resume_path, log_dir / Path(filename).with_suffix(".onnx")
                        )
                    else:
                        runner.save(str(log_dir / filename))
            write_json(log_dir / "lifecycle.json", lifecycle)
        except BaseException as error:
            lifecycle.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(log_dir / "lifecycle.json", lifecycle)
            raise
        finally:
            try:
                try:
                    if logger is not None:
                        logger.stop_logging_writer()
                finally:
                    if env is not None:
                        env.close()
            except BaseException as error:
                lifecycle.update(
                    status="failed", error=f"{type(error).__name__}: {error}"
                )
                write_json(log_dir / "lifecycle.json", lifecycle)
                raise


if __name__ == "__main__":
    main()
