"""Studio jobs run in their own process (GPU, Hydra and task registration stay out of the server).

    python -m myoInteract.studio.worker preview --preset preset.json --out <dir>
    python -m myoInteract.studio.worker evaluate --log-dir <run> --checkpoint model_final.onnx \\
        --episodes 64 --out <dir>
    python -m myoInteract.studio.worker viewer (--preset preset.json | --log-dir <run> --checkpoint ...) \\
        --port 8080 --out <dir>
    python -m myoInteract.studio.worker stream (--preset preset.json | --log-dir <run> ...) --port 8080 --out <dir>

``preview`` and ``evaluate`` write ``<dir>/result.json`` last, with images and videos next to it.
``viewer`` (viser) and ``stream`` (Studio's own scene) write ``result.json`` once they listen and
then keep running until stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from myoInteract.mjlab_offscreen_renderer import (
    _prepare_mjlab_offscreen_renderer,
    _sync_render_data,
)
from myoInteract.training_lifecycle import write_json

TASK_ID = "myoInteractUniversal-v0"
ZERO_EXCITATION = -3.0  # raw action with sigmoid(5 * (a - 0.5)) ~ 0


def build_env(
    overrides: list[str],
    num_envs: int,
    device: str,
    *,
    auto_reset: bool = True,
    seed: int | None = None,
):
    """Compose the config, register the task and build ``num_envs`` environments."""
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg  # noqa: PLC0415

    from myoInteract.myointeract_hydra_cli import load_config_interactive  # noqa: PLC0415
    from myoInteract.register_mjlab_myointeract_tasks import (
        register_mjlab_myointeract_task,
    )  # noqa: PLC0415

    config = load_config_interactive(overrides)
    register_mjlab_myointeract_task(config)
    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env_cfg.auto_reset = auto_reset
    if seed is not None:
        env_cfg.seed = seed
    return ManagerBasedRlEnv(cfg=env_cfg, device=device), config


def run_overrides(log_dir: Path) -> list[str]:
    """The Hydra overrides a run was trained with."""
    from myoInteract.myointeract_hydra_cli import checkpoint_task_overrides

    return checkpoint_task_overrides(
        json.loads((log_dir / "original_hydra_config.json").read_text())
    )


def task_overrides(args) -> tuple[list[str], list[str] | None]:
    """Overrides of the task a run's policy plays, and what they change against its training.

    The run's own task, or with ``--task`` the setup from the editor (out of distribution).
    """
    if getattr(args, "task", None) is None:
        return run_overrides(args.log_dir), None
    task = json.loads(args.task.read_text())
    from myoInteract.myointeract_hydra_cli import checkpoint_task_overrides

    return checkpoint_task_overrides(task["overrides"]), task["changes"]


def paint_targets(env, config) -> list[int]:
    """Target colours and mean sizes on the CPU model (offscreen renderer and viser draw from it).

    Returns the target geom ids.
    """
    model = env.sim.mj_model
    targets = config.env.task_config.targets
    try:
        ids = env.scene["universal_robot"].target_size_ids
    except (KeyError, AttributeError):
        return []
    geoms = [int(g) for g in ids[: targets.num_targets]]
    for i, gid in enumerate(geoms):
        t = getattr(targets, f"target_{i}", None) or targets.target_0
        model.geom_rgba[gid] = [*list(t.rgb)[:3], 1.0]
        lo, hi = t.size
        if isinstance(lo, (int, float)):
            model.geom_size[gid] = [(lo + hi) / 2] * 3
        else:
            model.geom_size[gid, :3] = [(a + b) / 2 for a, b in zip(lo, hi)]
    return geoms


def make_runner(vec_env, log_dir: Path, device: str):
    """The training runner for a run's policies (MLP and vision policies alike)."""
    from mjlab.tasks.registry import load_rl_cfg  # noqa: PLC0415
    from myosuite.integrations.musclemimic.mjlab_policy_runner import (
        OnnxCheckpointingMjlabRunner,
    )  # noqa: PLC0415

    saved = log_dir / "agent_config.json"
    cfg = (
        json.loads(saved.read_text())
        if saved.is_file()
        else asdict(load_rl_cfg(TASK_ID))
    )
    return OnnxCheckpointingMjlabRunner(vec_env, cfg, None, device, task_id=TASK_ID)


def load_weights(runner, onnx: Path, device: str):
    """Load a checkpoint's network weights into *runner* and return its inference policy.

    Weights only: ``load_onnx`` would also restore the run's per-env fatigue state, which does not
    fit an env with a different number of environments.
    """
    from myosuite.utils.onnx_checkpoint import extract_checkpoint_from_onnx  # noqa: PLC0415

    native, _, tmp = extract_checkpoint_from_onnx(onnx)
    counter = runner.env.unwrapped.common_step_counter
    try:
        runner.load(str(native), load_cfg={"actor": True}, map_location=device)
    finally:
        # mjlab restores this field even for actor-only loads. Evaluation and
        # follow mode must keep their own environment/curriculum timeline.
        runner.env.unwrapped.common_step_counter = counter
        if tmp is not None:
            tmp.cleanup()
    return runner.get_inference_policy(device=device)


def load_policy(vec_env, log_dir: Path, checkpoint: str, device: str):
    """Inference policy of one checkpoint of a run."""
    try:
        return load_weights(
            make_runner(vec_env, log_dir, device), log_dir / checkpoint, device
        )
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
        raise ValueError(
            "This policy does not fit the task: its observations or actions have a different size."
        ) from exc


class FollowingPolicy:
    """Policy that switches to the newest checkpoint of a run as training saves it (follow mode).

    Relaxed muscles until the first checkpoint exists. A background thread checks every few seconds;
    the viewer keeps running (one instance for the whole training run).
    """

    def __init__(
        self,
        vec_env,
        log_dir: Path,
        device: str,
        zero_policy,
        on_switch=None,
        poll_s: float = 4.0,
    ):
        self.runner, self.log_dir, self.device = (
            make_runner(vec_env, log_dir, device),
            log_dir,
            device,
        )
        self.current: str | None = None
        self._policy, self._lock = zero_policy, threading.Lock()
        self._on_switch, self._poll_s = on_switch, poll_s
        self._try_poll()
        threading.Thread(target=self._loop, daemon=True).start()

    def __call__(self, obs):
        with self._lock:
            return self._policy(obs)

    def _poll_once(self) -> None:
        from myoInteract.studio.runs import checkpoints  # noqa: PLC0415

        found = checkpoints(self.log_dir)
        if not found or found[-1] == self.current:
            return
        newest = found[-1]
        with self._lock:
            self._policy = load_weights(self.runner, self.log_dir / newest, self.device)
            self.current = newest
        print(f"[viewer] following: {newest}", flush=True)
        if self._on_switch is not None:
            self._on_switch(newest)

    def _loop(self) -> None:
        while True:
            time.sleep(self._poll_s)
            self._try_poll()

    def _try_poll(self) -> None:
        try:
            self._poll_once()
        except (
            Exception
        ) as err:  # existing/native writers may still publish non-atomically
            print(
                f"[viewer] could not load the newest checkpoint yet: {err}", flush=True
            )


def wrap(env):
    from mjlab.rl import RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_rl_cfg  # noqa: PLC0415

    return RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK_ID).clip_actions)


class Camera:
    """Offscreen renderer of environment 0, framed on the arm and its targets."""

    def __init__(self, env, config, width: int, height: int):
        self.env = env
        self.renderer, self.data, self.cam = _prepare_mjlab_offscreen_renderer(
            env.sim, env.cfg, width=width, height=height
        )
        self.cam.elevation = -10.0
        self.cam.azimuth = 160.0
        self.target_geoms = paint_targets(env, config)
        self._framed = False

    def frame(self) -> np.ndarray:
        data = _sync_render_data(self.env.sim, self.data)
        if not self._framed:  # box around the arm's bodies and the targets
            bodies = data.xpos[1:]
            bodies = bodies[
                np.linalg.norm(bodies - np.median(bodies, axis=0), axis=1) < 1.0
            ]
            pts = np.concatenate([bodies, data.geom_xpos[self.target_geoms]])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            self.cam.lookat[:] = (lo + hi) / 2
            self.cam.distance = 0.25 + 1.1 * float(np.max(hi - lo))
            self._framed = True
        self.renderer.update_scene(data, camera=self.cam)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def preview(args) -> dict:
    """Four reset states of the configured task, tiled 2 x 2 (targets are re-drawn every reset)."""
    import imageio.v3 as iio  # noqa: PLC0415

    from myoInteract.myointeract_hydra_cli import gradio_json_to_overrides  # noqa: PLC0415

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env, config = build_env(
        gradio_json_to_overrides(json.loads(args.preset.read_text())), 1, device
    )
    cam = Camera(env, config, args.width, args.height)
    tiles = []
    for seed in range(4):
        torch.manual_seed(seed)
        env.reset()
        env.sim.forward()
        tiles.append(cam.frame())
    cam.close()
    env.close()
    grid = np.concatenate(
        [np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0
    )
    iio.imwrite(args.out / "preview.png", grid)
    return {
        "image": "preview.png",
        "num_targets": int(config.env.task_config.targets.num_targets),
    }


@torch.no_grad()
def evaluate(args) -> dict:
    """Deterministic policy on fresh episodes: success, targets reached, time to success, effort,
    fatigue and, with ``--video-seconds`` > 0, a video (Studio shows the live 3D viewer instead)."""
    import imageio  # noqa: PLC0415

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.episodes < 1 or args.video_seconds < 0:
        raise ValueError(
            "Evaluation needs positive episodes and non-negative video seconds."
        )
    overrides, changes = task_overrides(args)
    env, config = build_env(
        overrides,
        args.episodes,
        device,
        auto_reset=False,
        seed=args.seed,
    )
    cam = None
    try:
        vec_env = wrap(env)
        policy = load_policy(vec_env, args.log_dir, args.checkpoint, device)
        muscles = env.action_manager.get_term("muscles")
        fatigue = getattr(muscles, "_fatigue", None)
        robot = env.scene[
            "universal_robot"
        ]  # holds the count of targets reached this episode
        ctrl_dt = float(env.step_dt)
        n = env.num_envs
        episode_steps = env.max_episode_length + 2
        video_steps = int(round(args.video_seconds / ctrl_dt))
        cam = Camera(env, config, args.width, args.height) if video_steps else None

        torch.manual_seed(args.seed)
        obs, _ = vec_env.reset()
        done = torch.zeros(n, dtype=torch.bool, device=device)
        success = torch.zeros(n, dtype=torch.bool, device=device)
        t_done = torch.zeros(n, device=device)
        effort = torch.zeros(n, device=device)
        final_mf = torch.zeros(n, device=device)
        reached = torch.zeros(n, device=device)
        frames = []
        for step in range(max(episode_steps, video_steps)):
            obs, _, dones, _ = vec_env.step(policy(obs))
            live = ~done
            # Manual reset exposes the terminal action/fatigue before action
            # manager.reset clears them. Latch each first episode independently.
            effort += live.float() * muscles.processed_action.mean(dim=-1)
            if fatigue is not None:
                final_mf = torch.where(live, fatigue.MF.mean(dim=-1), final_mf)
            reached = torch.where(live, robot.current_target_id[:, 0].float(), reached)
            ended = live & dones.bool()
            success |= (
                ended & env.termination_manager.get_term("episode_success").bool()
            )
            t_done = torch.where(
                ended, torch.full_like(t_done, (step + 1) * ctrl_dt), t_done
            )
            done |= ended
            if step < video_steps:
                frames.append(cam.frame())
            if step + 1 >= video_steps and done.all():
                break
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids):
                env.reset(env_ids=reset_ids)
                obs = vec_env.get_observations()
        steps_taken = torch.where(
            done, t_done / ctrl_dt, torch.full_like(t_done, float(episode_steps))
        )
        result = {
            "checkpoint": args.checkpoint,
            "episodes": n,
            "vision": bool(getattr(config.vision, "enabled", False)),
            "success_rate": float(success.float().mean()),
            "time_to_success_s": float(t_done[success].mean())
            if success.any()
            else None,
            "mean_excitation": float((effort / steps_taken).mean()),
            "final_mf": float(final_mf.mean()) if fatigue is not None else None,
            # share of episodes that reached target k (the targets are visited in order)
            "targets_reached": [
                float((reached > k).float().mean()) for k in range(robot.num_targets)
            ],
            # per episode, for the distributions behind the summary tiles
            "per_episode": {
                "targets": [int(x) for x in reached.tolist()],
                "time_s": [
                    round(t, 3) if ok else None
                    for t, ok in zip(t_done.tolist(), success.tolist())
                ],
                "excitation": [round(x, 4) for x in (effort / steps_taken).tolist()],
                "final_mf": [round(x, 4) for x in final_mf.tolist()]
                if fatigue is not None
                else None,
            },
            # the editor's setup instead of the run's own: what it changes
            "task_changes": changes,
            "video": "rollout.mp4" if frames else None,
        }
        if frames:
            imageio.mimsave(
                args.out / "rollout.mp4",
                frames,
                fps=round(1 / ctrl_dt),
                macro_block_size=8,
            )
        return result
    finally:
        if cam is not None:
            cam.close()
        env.close()


def _task_env(args, device: str):
    """The single-environment task of a viewer job: a run's task or a preset's."""
    from myoInteract.myointeract_hydra_cli import gradio_json_to_overrides  # noqa: PLC0415

    if args.log_dir is not None:
        overrides, _ = task_overrides(args)
    else:
        overrides = gradio_json_to_overrides(json.loads(args.preset.read_text()))
    return build_env(overrides, 1, device)


def _viewer_policy(args, vec_env, device: str, set_status):
    """What a viewer job plays: the newest checkpoint (follow), one checkpoint, or relaxed muscles."""
    n_act = vec_env.num_actions

    def zero_policy(obs):
        return torch.full((obs.batch_size[0], n_act), ZERO_EXCITATION, device=device)

    if args.log_dir is not None and args.follow:
        set_status(
            "Following training · waiting for the first checkpoint (relaxed muscles)"
        )
        return FollowingPolicy(
            vec_env,
            args.log_dir,
            device,
            zero_policy,
            on_switch=lambda name: set_status(
                f"Following training · newest checkpoint {name}"
            ),
        )
    if args.log_dir is not None:
        set_status(f"Checkpoint {args.checkpoint}")
        return load_policy(vec_env, args.log_dir, args.checkpoint, device)
    set_status("Task preview · relaxed muscles")
    return zero_policy


def _policy_label(args) -> str | None:
    return "follow" if args.follow else (args.checkpoint if args.log_dir else None)


def viewer(args) -> None:
    """Interactive viser viewer of a task (zero excitation) or of a checkpoint's policy; runs until killed."""
    import viser  # noqa: PLC0415

    from myoInteract.scripts.viser_ext import CameraRestoringViserViewer  # noqa: PLC0415

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env, config = _task_env(args, device)
    env.cfg.viewer.distance = 2.5
    env.cfg.viewer.elevation = 45.0
    env.cfg.viewer.azimuth = 135.0
    paint_targets(env, config)
    vec_env = wrap(env)
    server = viser.ViserServer(host=args.host, port=args.port)
    status = server.gui.add_markdown("")
    policy = _viewer_policy(
        args, vec_env, device, lambda text: setattr(status, "content", text)
    )
    write_json(
        args.out / "result.json",
        {"port": server.get_port(), "policy": _policy_label(args)},
    )
    CameraRestoringViserViewer(
        vec_env, policy, viser_server=server, record_name="studio"
    ).run()


def live_hello(env, config, target_ids: list[int]) -> dict:
    """First message of a live feed (and header of a recorded one): model, bodies and targets."""
    from myoInteract.myointeract_hydra_cli import BM_MODEL_MAP  # noqa: PLC0415
    from myoInteract.studio.scene import body_short_name  # noqa: PLC0415

    mj_model = env.sim.mj_model
    rest = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, rest)

    def target(g: int) -> dict:
        """Colour and shape of target geom ``g``: a sphere by its radius, a button as its box."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rest.geom_xmat[g])
        box = mj_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        return {
            "rgba": [round(float(c), 3) for c in mj_model.geom_rgba[g]],
            "radius": float(mj_model.geom_size[g, 0]),
            "shape": "box" if box else "sphere",
            "size": [float(s) for s in mj_model.geom_size[g]],  # half extents of a box
            "quat": [round(float(q), 6) for q in quat],
        }

    return {
        "type": "hello",
        "model": next(
            (
                k
                for k, (path, *_) in BM_MODEL_MAP.items()
                if path == config.env.model_path
            ),
            None,
        ),
        "bodies": [
            body_short_name(mj_model.body(b).name) for b in range(mj_model.nbody)
        ],
        "targets": [target(g) for g in target_ids],
        "muscles": [
            body_short_name(mj_model.actuator(a).name) for a in muscle_acts(mj_model)
        ],
        "joints": [
            {
                "name": body_short_name(mj_model.joint(j).name),
                "range": [float(r) for r in mj_model.jnt_range[j]],
            }
            for j in shown_joints(mj_model)
        ],
        "ctrl_dt": float(env.step_dt),
    }


def muscle_acts(mj_model) -> list[int]:
    """Actuators with an activation state (the muscles), in actuator order."""
    return [a for a in range(mj_model.nu) if mj_model.actuator_actadr[a] >= 0]


def shown_joints(mj_model) -> list[int]:
    """Joints the live feed reports: limited hinge and slide joints of the body, not driven by another
    joint (joint equality constraints) and not on a mocap body."""
    driven = {
        int(mj_model.eq_obj1id[e])
        for e in range(mj_model.neq)
        if mj_model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT
    }
    kinds = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    return [
        j
        for j in range(mj_model.njnt)
        if mj_model.jnt_type[j] in kinds
        and mj_model.jnt_limited[j]
        and j not in driven
        and mj_model.body_mocapid[mj_model.jnt_bodyid[j]] < 0
    ]


def frame_addresses(mj_model) -> tuple[list[int], list[int]]:
    """Addresses of the muscle activations (in ``act``) and joint angles (in ``qpos``) of a live frame."""
    act_adr = [int(mj_model.actuator_actadr[a]) for a in muscle_acts(mj_model)]
    qpos_adr = [int(mj_model.jnt_qposadr[j]) for j in shown_joints(mj_model)]
    return act_adr, qpos_adr


def live_frame(
    data,
    target_ids: list[int],
    act_adr: list[int] | None = None,
    qpos_adr: list[int] | None = None,
) -> bytes:
    """One float32 frame of environment 0: body positions and orientations, target positions, muscle
    activations, joint angles."""
    parts = [data.xpos[0], data.xquat[0]]
    if target_ids:
        parts.append(data.geom_xpos[0, target_ids])
    if act_adr:
        parts.append(data.act[0, act_adr])
    if qpos_adr:
        parts.append(data.qpos[0, qpos_adr])
    return torch.cat([q.reshape(-1) for q in parts]).float().cpu().numpy().tobytes()


def record(
    log_dir: Path, checkpoint: str, seconds: float, seed: int = 0
) -> tuple[dict, bytes]:
    """A recorded live feed of one checkpoint: its hello message and ``seconds`` of frames."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env, config = build_env(run_overrides(log_dir), 1, device)
    target_ids = paint_targets(env, config)
    vec_env = wrap(env)
    policy = load_policy(vec_env, log_dir, checkpoint, device)
    act_adr, qpos_adr = frame_addresses(env.sim.mj_model)
    torch.manual_seed(seed)
    obs, _ = vec_env.reset()
    frames = []
    with torch.no_grad():
        for _ in range(int(round(seconds / float(env.step_dt)))):
            obs, _, _, _ = vec_env.step(policy(obs))
            frames.append(live_frame(env.sim.data, target_ids, act_adr, qpos_adr))
    hello = live_hello(env, config, target_ids)
    env.close()
    return hello, b"".join(frames)


def stream(args) -> None:
    """Live feed for Studio's own 3D scene: environment 0 over a websocket; runs until killed.

    The first message (JSON, ``type: hello``) names the bodies and the targets. Every further
    binary message is one float32 frame: per body its position (3) and orientation (w, x, y, z),
    then per target its position (3), then per muscle its activation, then per joint its angle.
    Status changes come as JSON (``type: status``). The simulation runs in real time while a
    browser is connected and pauses otherwise.
    """
    import asyncio  # noqa: PLC0415

    import websockets  # noqa: PLC0415
    from websockets.asyncio.server import serve  # noqa: PLC0415

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env, config = _task_env(args, device)
    target_ids = paint_targets(env, config)
    vec_env = wrap(env)
    state = {"status": "", "frame": None, "seq": 0, "error": None}
    policy = _viewer_policy(
        args, vec_env, device, lambda text: state.__setitem__("status", text)
    )
    ctrl_dt = float(env.step_dt)
    hello = live_hello(env, config, target_ids)
    act_adr, qpos_adr = frame_addresses(env.sim.mj_model)
    watching = threading.Event()  # set while a browser is connected

    def simulate() -> None:
        try:
            data = env.sim.data
            obs = vec_env.get_observations()
            while True:
                watching.wait()
                t0 = time.monotonic()
                with torch.no_grad():
                    obs, _, _, _ = vec_env.step(policy(obs))
                state["frame"] = live_frame(data, target_ids, act_adr, qpos_adr)
                state["seq"] += 1
                time.sleep(max(0.0, ctrl_dt - (time.monotonic() - t0)))
        except Exception as error:
            state["error"] = error

    threading.Thread(target=simulate, daemon=True).start()
    clients: set = set()

    async def handler(ws) -> None:
        clients.add(ws)
        watching.set()
        try:
            status = state["status"]
            await ws.send(json.dumps({**hello, "status": status}))
            seq = -1
            while True:
                if state["status"] != status:
                    status = state["status"]
                    await ws.send(json.dumps({"type": "status", "status": status}))
                if state["seq"] != seq and state["frame"] is not None:
                    seq = state["seq"]
                    await ws.send(state["frame"])
                await asyncio.sleep(1 / 60)
        except websockets.ConnectionClosed:
            pass  # the browser left
        finally:
            clients.discard(ws)
            if not clients:
                watching.clear()

    async def run() -> None:
        async with serve(handler, args.host, args.port, max_size=None):
            result = {
                "port": args.port,
                "policy": _policy_label(args),
                "kind": "stream",
            }
            write_json(args.out / "result.json", result)
            while state["error"] is None:
                await asyncio.sleep(0.1)
            raise RuntimeError("The live simulation failed.") from state["error"]

    asyncio.run(run())


def warm() -> None:
    """Do the slow imports now, then run the job that arrives on stdin as one JSON line.

    ``{"argv": [kind, ...], "log": path}``: the job's arguments as for the command line and its log.
    Importing mjlab loads every installed task package (about 10 s), so Studio keeps one of these ready.
    """
    import os  # noqa: PLC0415

    import mjlab  # noqa: F401, PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: F401, PLC0415
    from mjlab.rl import RslRlVecEnvWrapper  # noqa: F401, PLC0415

    from myoInteract.myointeract_hydra_cli import (  # noqa: F401, PLC0415
        gradio_json_to_overrides,
        load_config_interactive,
    )
    from myoInteract.register_mjlab_myointeract_tasks import (  # noqa: F401, PLC0415
        register_mjlab_myointeract_task,
    )

    print("warm worker ready", flush=True)
    line = sys.stdin.readline()
    if not line.strip():
        return  # the server stopped without a job
    job = json.loads(line)
    log = os.open(job["log"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    main(job["argv"])


def main(argv: list[str] | None = None) -> None:
    if (argv if argv is not None else sys.argv[1:])[:1] == ["warm"]:
        warm()
        return
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="kind", required=True)
    pv = sub.add_parser("preview")
    pv.add_argument("--preset", type=Path, required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--log-dir", type=Path, required=True)
    ev.add_argument("--checkpoint", default="model_final.onnx")
    ev.add_argument("--episodes", type=int, default=64)
    ev.add_argument("--video-seconds", type=float, default=8.0)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--task", type=Path, help="the editor's setup (shifted_task)")
    vws = [sub.add_parser("viewer"), sub.add_parser("stream")]
    for vw in vws:
        vw.add_argument("--preset", type=Path)
        vw.add_argument("--log-dir", type=Path)
        vw.add_argument("--checkpoint", default="model_final.onnx")
        vw.add_argument("--task", type=Path, help="the editor's setup (shifted_task)")
        vw.add_argument(
            "--follow",
            action="store_true",
            help="load each new checkpoint of the run as training saves it",
        )
        vw.add_argument("--host", default="127.0.0.1")
        vw.add_argument("--port", type=int, default=8080)
    for sp in (pv, ev, *vws):
        sp.add_argument("--out", type=Path, required=True)
        sp.add_argument("--width", type=int, default=480)
        sp.add_argument("--height", type=int, default=360)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.kind in ("viewer", "stream"):
        (viewer if args.kind == "viewer" else stream)(args)
        return
    result = preview(args) if args.kind == "preview" else evaluate(args)
    write_json(args.out / "result.json", result)


if __name__ == "__main__":
    main()
