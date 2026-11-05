import os

import gradio as gr
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.envs import ManagerBasedRlEnv
import numpy as np
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

if gr.NO_RELOAD:
    import json
    import threading
    from dataclasses import asdict, dataclass, field
    from pathlib import Path

    import torch
    import viser
    from myoInteract.scripts.range_slider import RangeSlider
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer.viser.viewer import ViserPlayViewer
    from myoInteract.paths import REPO_ROOT, data_root
    from myoInteract.myointeract_hydra_cli import (
        BM_MODEL_MAP,
        checkpoint_config_path,
        load_config_interactive,
        training_config_to_labelled_dict,
        extra_hydra_overrides,
    )
    from myoInteract.register_mjlab_myointeract_tasks import (
        _vision_camera_recipe,
        _universal_spec_fn,
        _get_myoInteract_root as _get_myosuite_root,
        _compute_target_origin,
        register_mjlab_myointeract_task,
    )
    import time as _time
    from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago
    from myosuite.utils.onnx_checkpoint import (
        is_onnx_checkpoint_name,
        onnx_checkpoint_sort_key,
    )
    from myoInteract.onnx_checkpoint_extra import (
        download_all_wandb_onnx_checkpoints,
        extract_wandb_hydra_config,
    )
    from myosuite.integrations.musclemimic.mjlab_policy_runner import (
        OnnxCheckpointingMjlabRunner,
    )
    from myoInteract.scripts.viser_ext import (
        CameraRestoringViserViewer,
        PollingViserViewer,
    )

    # Persistent data dirs (checkpoints, logs). In a repo checkout they stay in
    # tutorials/myointeract/ and logs/; a pip install writes to $MYOINTERACT_HOME or
    # ./myoInteract_data instead of into site-packages.
    myosuite_path = data_root()
    if myosuite_path == REPO_ROOT and "MYOINTERACT_HOME" not in os.environ:
        myointeract_path = REPO_ROOT / "tutorials" / "myointeract" / "_sentinel"
    else:
        myointeract_path = myosuite_path / "_sentinel"

    _viser_preview_server = viser.ViserServer(port=7861)
    _viser_preview_url = f"http://localhost:{_viser_preview_server.get_port()}"
    # A public viser tunnel is only needed where localhost is not reachable from the browser
    # (Colab). Elsewhere it stays off, because the URL is open to anyone who finds it;
    # MYOINTERACT_VISER_SHARE=1 / 0 forces it on / off.
    _share = os.environ.get(
        "MYOINTERACT_VISER_SHARE", "1" if os.environ.get("COLAB_RELEASE_TAG") else "0"
    )
    if _share == "1":
        try:
            _viser_preview_url = _viser_preview_server.request_share_url()
        except Exception:
            pass

    _render_state: dict = {
        "env": None,
        "vec_env": None,
        "viewer": None,
        "stop_event": None,
        "thread": None,
        "struct_key": None,  # (model_path, num_targets, target_shape_tuple)
        "policy": None,  # OnnxPolicy instance, or None for zero-action
        "checkpoint_manager": None,  # CheckpointManager for viser checkpoint tab
    }

    def _render_struct_key(config):
        """Return a hashable key that captures the structural aspects of the config.

        A key change means the MuJoCo model must be rebuilt from scratch.
        Non-structural changes (colors, sizes, position ranges) can be applied
        to the running env without restarting.

        All BMParameters fields require a rebuild:
          model_path  — different XML skeleton
          ctrl_dt     — changes env decimation
          reset_type  — changes events dict
          sigdepnoise/constantnoise type — changes action-term noise config
          fatigue_enabled — changes whether TorchFatigueState is created
          mj_impl     — changes MuJoCo backend
        Observation space parameters also require a rebuild as they change
        the observation vector dimension:
          obs_keys, fatigue_obs_keys, omni_keys — individual obs terms
          vision enabled/mode — adds camera observations
        """
        targets = config.env.task_config.targets
        num_t = targets.num_targets
        _t0 = targets.target_0
        shapes = tuple(
            (getattr(targets, f"target_{i}", None) or _t0).name for i in range(num_t)
        )
        noise = config.env.muscle_config.noise_params
        obs_keys = tuple(config.env.task_config.obs_keys)
        fatigue_obs_keys = tuple(config.env.muscle_config.fatigue_obs_keys)
        omni_keys = tuple(config.env.task_config.omni_keys)
        try:
            vision_cfg = config.vision
            if getattr(vision_cfg, "enabled", False):
                recipe = _vision_camera_recipe(vision_cfg)
                vision_key = (
                    True,
                    recipe["data_types"],
                    recipe["width"],
                    recipe["height"],
                    recipe["depth_min"],
                    recipe["depth_cutoff"],
                    recipe["enabled_geom_groups"],
                )
            else:
                vision_key = (False,)
        except AttributeError:
            vision_key = (False,)
        return (
            config.env.model_path,
            config.env.ctrl_dt,
            config.env.task_config.reset_type,
            noise.sigdepnoise_type,
            noise.constantnoise_type,
            config.env.muscle_config.fatigue_enabled,
            config.env.mj_impl,
            num_t,
            shapes,
            obs_keys,
            fatigue_obs_keys,
            omni_keys,
            vision_key,
        )

    _UNIVERSAL_ENTITY = "universal_robot"

    def _update_model_geoms(env, config) -> None:
        """Update mj_model geom rgba/size and event term position/size ranges.

        Architecture notes
        ------------------
        Viser (mjviser) bakes colors and sizes into trimesh vertex colors when
        ViserPlayViewer.setup() calls create_primitive_mesh() / _resolve_flat_rgba().
        This reads mj_model.geom_rgba and mj_model.geom_size — NOT the WARP batched
        model (env.sim.model). There is no per-frame color/size sync from WARP.

        dr.geom_rgba / dr.geom_size only update env.sim.model (WARP tensors,
        initialized from mj_model via mjwarp.put_model() but independent thereafter).
        Calling those DR callbacks would not affect viser.

        Correct fix: update mj_model directly, then restart ViserPlayViewer so
        setup() re-reads mj_model and rebuilds the scene handles with new values.

        Geom IDs come from asset.target_size_ids (the same global IDs used by the
        TaskEntity and DR terms) to avoid any entity-namespace ambiguity that
        mujoco.mj_name2id would have.
        """
        mj_model = env.sim.mj_model  # standard mujoco.MjModel — what viser renders from
        targets = config.env.task_config.targets
        num_active = targets.num_targets
        _t0_cfg = targets.target_0
        # Per-model origin (ref_site's world position at neutral pose), matching
        # what register_mjlab_myointeract_task() actually used to build the
        # target_pos_dr event term's ranges for this model. Must NOT be a
        # hardcoded constant: different bm_models use different ref_site values
        # (e.g. myoarm/myoarm_v2 use "R.Shoulder_marker", not MoBL's "humphant").
        target_origin = _compute_target_origin(config)

        try:
            asset = env.scene[_UNIVERSAL_ENTITY]
            geom_ids_tensor = (
                asset.target_size_ids
            )  # shape [num_targets], global geom IDs
        except Exception:
            geom_ids_tensor = None

        pos_ranges: dict = {}
        size_ranges: dict = {}

        for i in range(num_active):
            target_cfg = getattr(targets, f"target_{i}", None) or _t0_cfg

            # Update mj_model so the restarted ViserPlayViewer sees new colors/sizes.
            if geom_ids_tensor is not None and i < len(geom_ids_tensor):
                geom_id = int(geom_ids_tensor[i].item())
                rgb = list(target_cfg.rgb)
                mj_model.geom_rgba[geom_id] = [rgb[0], rgb[1], rgb[2], 1.0]

                size_range = target_cfg.size
                if isinstance(size_range[0], (int, float)):
                    # Sphere: scalar [min_r, max_r] → uniform radius
                    avg_r = float((size_range[0] + size_range[1]) / 2)
                    mj_model.geom_size[geom_id] = [avg_r, avg_r, avg_r]
                else:
                    # Box: [[min_x,min_y,min_z], [max_x,max_y,max_z]] → per-axis half-extents
                    avg_size = [
                        float((size_range[0][k] + size_range[1][k]) / 2)
                        for k in range(3)
                    ]
                    mj_model.geom_size[geom_id, :3] = avg_size

            # Update event term params so env.reset() samples new position/size ranges.
            pos_ranges[f"body_target_{i}"] = {
                xyz: (
                    target_cfg.position[0][xyz] + target_origin[xyz],
                    target_cfg.position[1][xyz] + target_origin[xyz],
                )
                for xyz in range(3)
            }
            s = target_cfg.size
            size_ranges[f"geom_target_{i}"] = {
                xyz: (
                    s[0] if isinstance(s[0], (int, float)) else s[0][xyz],
                    s[1] if isinstance(s[1], (int, float)) else s[1][xyz],
                )
                for xyz in range(3)
            }

        try:
            env.event_manager.get_term_cfg("target_pos_dr").params["ranges"] = (
                pos_ranges
            )
        except Exception:
            pass
        try:
            env.event_manager.get_term_cfg("target_size_dr").params["ranges"] = (
                size_ranges
            )
        except Exception:
            pass

    def _build_policy_and_ckpt_manager(
        checkpoint_run: str,
        checkpoint_file: str,
        device: str,
        run_state: "RunState | None" = None,
    ):
        """Build (policy, CheckpointManager) from dropdown selections.

        Uses OnnxCheckpointingMjlabRunner (PyTorch actor) exactly as play_notebook_viser.py.
        The runner is initialised from the current gradio vec_env; if obs_dim does not
        match the checkpoint, load_onnx raises and (None, None) is returned.

        Returns (None, None) when no checkpoint is selected or loading fails.
        """
        ckpt_path = checkpoint_path_from_run_filename(checkpoint_run, checkpoint_file)
        if (
            (not ckpt_path or ckpt_path == "None")
            and run_state is not None
            and run_state.loaded_checkpoint_path
        ):
            ckpt_path = run_state.loaded_checkpoint_path
        if not ckpt_path or ckpt_path == "None":
            return None, None
        p = Path(ckpt_path)
        if p.suffix != ".onnx" or not p.exists():
            print(f"[render] Checkpoint not found or not .onnx: {p}")
            return None, None
        vec_env = _render_state.get("vec_env")
        if vec_env is None:
            print("[render] No vec_env available; cannot load checkpoint.")
            return None, None
        try:
            rl_cfg = load_rl_cfg("myoInteractUniversal-v0")
            runner = OnnxCheckpointingMjlabRunner(
                vec_env,
                asdict(rl_cfg),
                str(p.parent),
                device,
                task_id="myoInteractUniversal-v0",
            )
            runner.load_onnx(p)
            policy = runner.get_inference_policy(device=device)
            print(f"[render] Loaded checkpoint from {p.name}")

            ckpt_dir = p.parent

            def _reload_policy(path: str):
                runner.load_onnx(Path(path))
                return runner.get_inference_policy(device=device)

            def _fetch_local() -> list[tuple[str, str]]:
                now = _time.time()
                entries: list[tuple[str, str]] = []
                for f in ckpt_dir.glob("*.onnx"):
                    if not is_onnx_checkpoint_name(f.name):
                        continue
                    ago = format_time_ago(int(now - f.stat().st_mtime))
                    entries.append((f.name, ago))
                entries.sort(key=lambda x: onnx_checkpoint_sort_key(x[0]))
                return entries

            ckpt_manager = CheckpointManager(
                current_name=p.name,
                fetch_available=_fetch_local,
                load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
            )
            return policy, ckpt_manager
        except Exception as exc:
            import traceback

            print(f"[render] Could not load checkpoint from {p}: {exc}")
            traceback.print_exc()
            if isinstance(exc, RuntimeError) and "size mismatch" in str(exc):
                gr.Warning(
                    "The selected checkpoint is incompatible with the current observation "
                    "space settings (observation dimension mismatch). Please match the "
                    "settings to those used during training, or select a different checkpoint."
                )
            return None, None

    class _StopViewer(BaseException):
        """Raised from the policy callback to cleanly unwind ViserPlayViewer.run()."""

    class _StoppablePolicy:
        """Wraps any callable policy and raises _StopViewer when stop_event is set."""

        def __init__(self, inner, stop_event: threading.Event) -> None:
            self._inner = inner
            self._stop = stop_event

        def __call__(self, obs):
            if self._stop.is_set():
                raise _StopViewer
            return self._inner(obs)

        def reset(self) -> None:
            if hasattr(self._inner, "reset"):
                self._inner.reset()

    class _ZeroActionInner:
        """Stateless zero-action policy (no stop logic — that lives in _StoppablePolicy)."""

        def __init__(
            self, num_actions: int, device: str, action_mode: str = "sigmoid"
        ) -> None:
            self._n = num_actions
            self._dev = device
            # In sigmoid mode process_actions applies sigma(5*(a-0.5)), so raw=0
            # maps to ~7.6% excitation, not zero.  Use a strongly negative value
            # so that sigma(5*(-3-0.5)) < 3e-8 — effectively zero excitation.
            self._fill = -3.0 if action_mode == "sigmoid" else 0.0

        def __call__(self, obs):
            n = (
                next(iter(obs.values())).shape[0]
                if hasattr(obs, "values")
                else obs.shape[0]
            )
            return torch.full((n, self._n), self._fill, device=self._dev)

        def reset(self) -> None:
            pass

    def _run_viewer(viewer: ViserPlayViewer, stop_event: threading.Event) -> None:
        try:
            viewer.run()
        except _StopViewer:
            pass
        except Exception:
            # Expected when the env/viewer was torn down externally (e.g. a new
            # "Update Preview" call closed this env mid-loop) -- but a genuine crash
            # during viewer.run() (e.g. viser scene setup choking on an unexpected
            # obs/asset shape) looks identical from here and previously vanished with
            # zero diagnostics: the browser tab would just sit on a half-built scene
            # forever with no error anywhere. Printing lets this be told apart from a
            # deliberate stop without changing the "don't crash the thread" behavior.
            if not stop_event.is_set():
                import traceback

                print("[render] Viewer thread terminated by an unexpected exception:")
                traceback.print_exc()


sphere_ranges = {
    "x": (0.0, 0.75),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.75),
    "size": (0.001, 0.15),  # radius
}

INIT_ELEMENTS = 2
MAX_TARGETS = 15  # maximum number of targets supported by the Gradio UI


parent_path = myointeract_path.parent
CHECKPOINT_PATH = os.path.join(parent_path, "tracked_checkpoints/universal/")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
LOG_DIR = os.path.join(myosuite_path, "logs/rsl_rl/myo_universal")
os.makedirs(LOG_DIR, exist_ok=True)

# Default W&B entity and project used when only a bare run ID is given.
WANDB_DEFAULT_ENTITY = "hci-biomechsims"
WANDB_DEFAULT_PROJECT = "mjlab"


@dataclass
class RunState:
    wandb_run_name: str = ""
    cfg_overrides: list[str] = field(default_factory=lambda: [])
    log_dir: str = ""  # set by training cell; used by render_training_result()
    is_training: bool = (
        False  # set True during runner.learn(); blocks render operations
    )
    loaded_checkpoint_path: str = ""  # absolute ONNX path from Load/Resume section
    # Non-UI overrides from the loaded policy (e.g. fatigue_reset_random).
    # Prepended before UI-derived overrides so the UI always wins for overlapping fields.
    loaded_policy_extra_overrides: list[str] = field(default_factory=lambda: [])


def checkpoints_diff_folder(show_training_ckpts: bool = False):
    tracked_checkpoints = os.listdir(CHECKPOINT_PATH)
    if show_training_ckpts:
        tracked_checkpoints += os.listdir(LOG_DIR)
    return tracked_checkpoints


def get_available_checkpoints(show_training_ckpts: bool = False):
    tracked_checkpoints = checkpoints_diff_folder(
        show_training_ckpts=show_training_ckpts
    )
    checkpoints = tracked_checkpoints + ["None"]
    return checkpoints


def is_number(s: str):
    if ("." not in s) and s.isdigit():
        return True
    return False


def _run_checkpoint_dir(checkpoint_run: str) -> str | None:
    """Return the directory containing .onnx files for a run.

    Searches CHECKPOINT_PATH first, then LOG_DIR (for own training runs).
    Returns None when neither contains a matching directory.
    """
    for base_dir in (CHECKPOINT_PATH, LOG_DIR):
        candidate = os.path.join(base_dir, checkpoint_run)
        if os.path.isdir(candidate):
            return candidate
    return None


def get_available_checkpoint_numbers(checkpoint_run, show_training_ckpts):
    if checkpoint_run == "None":
        return ["None"]
    # Resolve directory: accept absolute paths directly, else search base dirs.
    # The tracked_checkpoints membership check is intentionally skipped so that
    # own training runs (in LOG_DIR) appear regardless of show_training_ckpts.
    if os.path.isdir(checkpoint_run):
        checkpoint_path = checkpoint_run
    else:
        checkpoint_path = _run_checkpoint_dir(checkpoint_run)
    if checkpoint_path is None:
        return ["None"]
    return _list_onnx_files_in_dir(checkpoint_path)


def checkpoint_path_from_run_filename(checkpoint_run, checkpoint_filename):
    if checkpoint_run == "None" or checkpoint_filename == "None":
        return "None"
    # Accept absolute paths directly, then search base dirs.
    # The tracked_checkpoints membership check is skipped so that own training
    # runs and wandb_checkpoints sub-dirs resolve regardless of show_training_ckpts.
    if os.path.isdir(checkpoint_run):
        ckpt_dir = checkpoint_run
    else:
        ckpt_dir = _run_checkpoint_dir(checkpoint_run)
    if ckpt_dir is None:
        return "None"
    return os.path.join(ckpt_dir, checkpoint_filename)


def update_checkpoint_dirs(show_training_ckpts):
    choices = get_available_checkpoints(show_training_ckpts)
    return gr.update(choices=choices, value=choices[0])


def _onnx_dir_listing_sort_key(filename: str) -> float:
    """Sort key for _list_onnx_files_in_dir: any "*_final.*" file first (e.g.
    "model_final.onnx"),
    then descending by trailing numeric step. Falls back to -inf (sorts last) for any
    other filename shape instead of raising, so one odd file can't crash the dropdown."""
    stem = os.path.splitext(filename)[0]
    if stem.endswith("_final"):
        return float("inf")
    try:
        return float(stem.split("_")[-1])
    except ValueError:
        return float("-inf")


def _list_onnx_files_in_dir(dir_path: str) -> list[str]:
    """List ONNX/PT checkpoint files in dir_path, *_final first then descending step."""
    if not os.path.isdir(dir_path):
        return ["None"]
    files = [
        f
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
        and os.path.splitext(f)[1] in (".onnx", ".pt")
    ]
    files.sort(key=_onnx_dir_listing_sort_key, reverse=True)
    return files or ["None"]


def _resolve_local_run_dir(val: str) -> tuple[str | None, str | None]:
    """Resolve val to (run_name, abs_dir_path) for a local checkpoint directory.

    Accepts:
    - Absolute path: /full/path/to/run_dir
    - Bare name: 2026-06-25_19-17-45  (looked up in LOG_DIR then CHECKPOINT_PATH)
    Returns (None, None) when the directory cannot be resolved.
    """
    val = val.strip()
    run_name = Path(val).name  # works for both absolute paths and bare names
    if os.path.isdir(val):
        return run_name, os.path.abspath(val)
    for base_dir in (LOG_DIR, CHECKPOINT_PATH):
        candidate = os.path.join(base_dir, run_name)
        if os.path.isdir(candidate):
            return run_name, candidate
    return None, None


def _dedup_overrides(overrides: list) -> list:
    """Deduplicate Hydra overrides, keeping the last occurrence of each key.

    Group-append overrides (starting with '+') are never deduplicated because
    each one adds a distinct config node (e.g. different target indices).
    """
    seen: dict[str, int] = {}  # key → index in result list
    result: list = list(overrides)
    for i, o in enumerate(overrides):
        if o.startswith("+") or "=" not in o:
            continue
        key = o.split("=", 1)[0]
        if key in seen:
            result[seen[key]] = None  # mark earlier duplicate for removal
        seen[key] = i
    return [o for o in result if o is not None]


def load_policy_input_changed(val):
    """Populate the run/file dropdowns when a local directory path is typed
    or when a wandb URL/run-ID whose download dir already exists is entered."""
    val = (val or "").strip()
    if not val or val.endswith(".onnx") or val.endswith(".pt"):
        return gr.update(), gr.update()

    # ── Wandb URL / run-ID ────────────────────────────────────────────────
    run_id = None
    if val.startswith("https://wandb.ai/"):
        # e.g. https://wandb.ai/hci-biomechsims/mjlab/runs/lgd335xy
        parts = val[len("https://wandb.ai/") :].replace("/runs/", "/").split("/")
        run_id = parts[-1] if parts else None
    elif not val.startswith("/") and "/" in val:
        # entity/project/run_id or entity/project/runs/run_id
        parts = val.replace("/runs/", "/").split("/")
        run_id = parts[-1] if parts else None
    elif (
        not val.startswith("/")
        and os.sep not in val
        and not _resolve_local_run_dir(val)[1]
    ):
        # Bare run ID
        run_id = val

    if run_id:
        # Only populate if the download dir already exists locally.
        rel_run = os.path.join("wandb_checkpoints", run_id)
        _, resolved_dir = _resolve_local_run_dir(rel_run)
        if resolved_dir is None:
            return gr.update(), gr.update()
        onnx_files = _list_onnx_files_in_dir(resolved_dir)
        run_choices = get_available_checkpoints(show_training_ckpts=True)
        if rel_run not in run_choices:
            run_choices = [rel_run] + [c for c in run_choices if c != "None"] + ["None"]
        return (
            gr.update(choices=run_choices, value=rel_run),
            gr.update(choices=onnx_files, value=onnx_files[0]),
        )

    # ── Local directory path ──────────────────────────────────────────────
    # Skip only if there is no path separator AND the value is not a known
    # local run directory.  Bare names like "2026-06-25_19-17-45_my-run" have
    # no '/' but may exist inside LOG_DIR or CHECKPOINT_PATH, so check both.
    if (
        "/" not in val
        and os.sep not in val
        and not os.path.isdir(val)
        and not _resolve_local_run_dir(val)[1]
    ):
        return gr.update(), gr.update()

    run_name, resolved_dir = _resolve_local_run_dir(val)
    if resolved_dir is None:
        return gr.update(), gr.update()

    onnx_files = _list_onnx_files_in_dir(resolved_dir)
    run_choices = get_available_checkpoints(show_training_ckpts=True)
    if run_name not in run_choices:
        # Prepend the run so the dropdown reflects it even if it lives outside LOG_DIR.
        non_none = [c for c in run_choices if c != "None"]
        run_choices = [run_name] + non_none + ["None"]

    best_file = onnx_files[0]  # model_final first, else highest step
    return (
        gr.update(choices=run_choices, value=run_name),
        gr.update(choices=onnx_files, value=best_file),
    )


def update_checkpoint_numbers(checkpoint_run, show_training_ckpts):
    choices = get_available_checkpoint_numbers(checkpoint_run, show_training_ckpts)
    return gr.update(choices=choices, value=choices[0])


def extract_rgb(rgba):
    rgba = rgba.split("rgba(")[1].strip(")")
    r, g, b, a = rgba.split(",")
    r = r.strip()
    g = g.strip()
    b = b.strip()
    rgb = [float(x.strip()) / 255.0 for x in [r, g, b]]
    return rgb


def hex_to_rgb(hex_color):
    if "rgba" in hex_color:
        return extract_rgb(hex_color)
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return rgb


def init_target_rgb(i):
    target_rgb = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.8, 0.8, 0.1],
        [0.4, 0.4, 0.4],
        [0.4, 0.1, 0.4],
        [0.1, 0.4, 0.4],
        [0.4, 0.4, 0.1],
        [0.1, 0.1, 0.1],
        [0.8, 0.8, 0.8],
    ]
    rgbs = [el * 255 for el in target_rgb[i % len(target_rgb)]]
    return f"rgba({rgbs[0]},{rgbs[1]},{rgbs[2]},1)"


def sanity_check_float(value, name):
    try:
        float(value)
    except Exception:
        gr.Warning(f"{name} must be a float, got {value}")
        return False
    return True


def sanity_check_bool(value, name):
    try:
        bool(value)
    except Exception:
        gr.Warning(f"{name} must be a boolean, got {value}")
        return False
    return True


def sanity_check_choices(value, name, choices):
    try:
        assert value in choices
    except Exception:
        gr.Warning(
            f"{name} must be a string and one of the following: {choices} but got {value}"
        )
        return False
    return True


def sanity_check_rgb(rgb, name):
    len_rgb = len(rgb)
    if len_rgb != 3:
        gr.Warning(f"RGB {name} must be a list of 3 floats, got {rgb}")
        return False
    for i, el in enumerate(rgb):
        if not sanity_check_float(el, f" RGB {name} element {i}"):
            return False
    return True


def sanity_check_int(value, name):
    try:
        int(value)
    except Exception:
        gr.Warning(f"{name} must be an integer, got {value}")
        return False
    return True


def sanity_check_number_targets(num_targets, name):
    try:
        n = int(num_targets)
    except Exception:
        gr.Warning(f"Number of targets must be an integer, got {num_targets}")
        return False
    if n < 1:
        gr.Warning(f"Number of targets must be at least 1, got {n}")
        return False
    if n > MAX_TARGETS:
        gr.Warning(
            f"{n} targets selected — only target 1 settings are used (all {n} targets copy from it). "
            f"Only target 1 is shown in the GUI."
        )
    return True


def sanity_check_float_array(array, name):
    try:
        to_array = np.array(array)
        for i, el in enumerate(to_array):
            assert sanity_check_float(el, f"{name} element {i}")
    except Exception as e:
        print(e)
        gr.Warning(f"{name} must be an array of floats, got {array}")
        return False
    return True


class BMParameters:
    num_elements: int = 7
    reset_type_choices = (
        "zero",
        "epsilon_uniform",
        "range_uniform",
        "initpose001",
        "initpose001_epsilon_uniform",
        "None",
    )
    mj_impl_choices = ("mjlab",)
    bm_model_choices = (
        "MoBL_Arms_Index",
        "MoBL_Arms_Index_v2.1",
        "MoBL_Arms_Index_v2.2",
        "MoBL_Arms_Hand",
        "MoBL_Arms_Hand_v2.1",
        "MyoArm_nohand",
        "MyoArm",
        "MyoArm_v2",
        "MyoArm_reach",
        "MyoArm_pointing",
    )

    @staticmethod
    def fields():
        return [
            "bm_model",
            "ctrl_dt",
            "reset_type",
            "sigdepnoise_enabled",
            "constantnoise_enabled",
            "fatigue_enabled",
            "mj_impl_type",
        ]

    @staticmethod
    def get_parameters():
        with gr.Row():
            bm_model = gr.Dropdown(
                label="Biomechanical Model",
                choices=BMParameters.bm_model_choices,
                interactive=True,
                value="MoBL_Arms_Index",
            )
            ctrl_dt = gr.Number(
                label="Control Timestep (s)",
                value=0.05,
                minimum=0.01,
                maximum=0.5,
                step=0.01,
                interactive=True,
            )
            reset_type = gr.Dropdown(
                label="Define how to reset the body pose",
                choices=BMParameters.reset_type_choices,
                interactive=True,
                value="epsilon_uniform",
            )
            sigdepnoise_enabled = gr.Checkbox(
                label="Signal-dependent motor noise", value=False, interactive=True
            )
            constantnoise_enabled = gr.Checkbox(
                label="Constant motor noise", value=False, interactive=True
            )
            fatigue_enabled = gr.Checkbox(
                label="Muscle Fatigue", value=False, interactive=True
            )
            mj_impl_type = gr.Dropdown(
                label="MuJoCo Backend",
                choices=BMParameters.mj_impl_choices,
                interactive=True,
                value="mjlab",
                visible="hidden",
            )
        return (
            bm_model,
            ctrl_dt,
            reset_type,
            sigdepnoise_enabled,
            constantnoise_enabled,
            fatigue_enabled,
            mj_impl_type,
        )

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = MAX_TARGETS
        box_number = MAX_TARGETS * BoxParameters.num_elements
        sphere_number = MAX_TARGETS * SphereParameters.num_elements
        bm_start = rl_number + num_targets + radio_number + box_number + sphere_number
        bm_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + BMParameters.num_elements
        )
        return all_args[bm_start:bm_end]

    @staticmethod
    def sanity_check_args(
        bm_model,
        ctrl_dt,
        reset_type,
        sigdepnoise_enabled,
        constantnoise_enabled,
        fatigue_enabled,
        mj_impl_type,
    ):
        sanity_check = sanity_check_choices(
            bm_model, "Biomechanical Model", BMParameters.bm_model_choices
        )
        sanity_check &= sanity_check_float(ctrl_dt, "Ctrl Timestep (s)")
        sanity_check &= sanity_check_choices(
            reset_type, "Reset type", BMParameters.reset_type_choices
        )
        sanity_check &= sanity_check_bool(
            sigdepnoise_enabled, "Signal-dependent motor noise"
        )
        sanity_check &= sanity_check_bool(constantnoise_enabled, "Constant motor noise")
        sanity_check &= sanity_check_bool(fatigue_enabled, "Muscle Fatigue")
        sanity_check &= sanity_check_choices(
            mj_impl_type, "MuJoCo Implementation", BMParameters.mj_impl_choices
        )
        return sanity_check

    @classmethod
    def parse_values(cls, all_args):
        (
            bm_model,
            ctrl_dt,
            reset_type,
            sigdepnoise_enabled,
            constantnoise_enabled,
            fatigue_enabled,
            mj_impl_type,
        ) = cls.get_my_args(all_args)
        sanity_check = cls.sanity_check_args(
            bm_model,
            ctrl_dt,
            reset_type,
            sigdepnoise_enabled,
            constantnoise_enabled,
            fatigue_enabled,
            mj_impl_type,
        )
        if not sanity_check:
            return False, []
        overrides = []
        ## TODO: move logic (argument to config attribute mappings) to separate function
        overrides.append(f"env.ctrl_dt={float(ctrl_dt)}")
        overrides.append(f"env.task_config.reset_type={reset_type}")
        sigdepnoise_type = "white" if sigdepnoise_enabled else "None"
        constantnoise_type = "white" if constantnoise_enabled else "None"
        overrides.append(
            f"env.muscle_config.noise_params.sigdepnoise_type={sigdepnoise_type}"
        )
        overrides.append(
            f"env.muscle_config.noise_params.constantnoise_type={constantnoise_type}"
        )
        overrides.append(f"env.muscle_config.fatigue_enabled={fatigue_enabled}")
        overrides.append(f"env.mj_impl={mj_impl_type}")
        if bm_model not in BM_MODEL_MAP:
            raise ValueError(f"Invalid biomechanical model: {bm_model}")
        bm_model_path, ee_site, ref_site = BM_MODEL_MAP[bm_model]
        overrides.append(f"env.model_path={bm_model_path}")
        overrides.append(f"env.task_config.reach_settings.ref_site={ref_site}")
        overrides.append(f"env.task_config.reach_settings.ee_site={ee_site}")
        return True, overrides


class BoxParameters:
    num_elements: int = 8

    @staticmethod
    def init_button_positions(i):
        button_positions = [
            [0.42, 0.02, 0.01],
            [0.42, 0.32, 0.01],
            [0.42, 0.02, -0.29],
            [0.42, 0.32, -0.29],
            [0.5, 0.0, -0.05],
            [0.5, 0.07, -0.05],
            [0.41, -0.07, 0.05],
            [0.41, 0.07, 0.05],
            [0.41, -0.07, -0.25],
            [0.41, 0.0, -0.25],
        ]
        return button_positions[i % len(button_positions)]

    @staticmethod
    def fields():
        return [
            "box_position_x",
            "box_position_y",
            "box_position_z",
            "box_size_x",
            "box_size_y",
            "min_touch_force",
            "orientation_angle",
            "rgb",
        ]

    @staticmethod
    def get_parameters(i):
        # Boxes option row
        button_position = BoxParameters.init_button_positions(i)
        button_rgb = init_target_rgb(i)
        with gr.Row(visible=False) as box_row:
            with gr.Accordion(label=f"Box {i+1} Settings", open=False):
                with gr.Row():
                    box_position_x = RangeSlider(
                        label="Depth Position",
                        minimum=0.0,
                        maximum=0.75,
                        value=(button_position[0], button_position[0]),
                        step=0.001,
                        interactive=True,
                    )
                    box_position_y = RangeSlider(
                        label="Horizontal Position",
                        minimum=-0.5,
                        maximum=0.5,
                        value=(button_position[1], button_position[1]),
                        step=0.001,
                        interactive=True,
                    )
                    box_position_z = RangeSlider(
                        label="Vertical Position",
                        minimum=-0.5,
                        maximum=0.75,
                        value=(button_position[2], button_position[2]),
                        step=0.001,
                        interactive=True,
                    )
                with gr.Row():
                    box_size_x_slider = RangeSlider(
                        label="Width",
                        minimum=0.005,
                        maximum=0.06,
                        value=(0.05, 0.05),
                        step=0.001,
                        interactive=True,
                    )
                    box_size_y_slider = RangeSlider(
                        label="Height",
                        minimum=0.005,
                        maximum=0.06,
                        value=(0.05, 0.05),
                        step=0.001,
                        interactive=True,
                    )
                with gr.Row():
                    min_touch_force = gr.Slider(
                        label="Minimum Touch Force",
                        info="Minimum force required to register a touch",
                        minimum=0.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.1,
                        interactive=True,
                    )
                    orientation_angle = gr.Slider(
                        label="Orientation Angle",
                        minimum=0.0,
                        maximum=180.0,
                        value=45,
                        step=1.0,
                        interactive=True,
                    )
                    rgb_btn = gr.ColorPicker(
                        label="RGB", value=button_rgb, interactive=True
                    )
        return box_row, (
            box_position_x,
            box_position_y,
            box_position_z,
            box_size_x_slider,
            box_size_y_slider,
            min_touch_force,
            orientation_angle,
            rgb_btn,
        )

    @staticmethod
    def get_my_args(i, all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = MAX_TARGETS
        box_start = rl_number + radio_number + num_targets
        box_end = box_start + MAX_TARGETS * BoxParameters.num_elements
        box_args = all_args[box_start:box_end]
        start_index = i * BoxParameters.num_elements
        end_index = start_index + BoxParameters.num_elements
        return box_args[start_index:end_index]

    @staticmethod
    def sanity_check_args(
        i,
        box_position_x,
        box_position_y,
        box_position_z,
        box_size_x_slider,
        box_size_y_slider,
        min_touch_force,
        orientation_angle,
        rgb,
    ):
        sanity_check = sanity_check_float(
            box_position_x[0], f"Box {i+1} Position X Range Start"
        )
        sanity_check &= sanity_check_float(
            box_position_x[1], f"Box {i+1} Position X Range End"
        )
        sanity_check &= sanity_check_float(
            box_position_y[0], f"Box {i+1} Position Y Range Start"
        )
        sanity_check &= sanity_check_float(
            box_position_y[1], f"Box {i+1} Position Y Range End"
        )
        sanity_check &= sanity_check_float(
            box_position_z[0], f"Box {i+1} Position Z Range Start"
        )
        sanity_check &= sanity_check_float(
            box_position_z[1], f"Box {i+1} Position Z Range End"
        )
        sanity_check &= sanity_check_float(
            box_size_x_slider[0], f"Box {i+1} Size X Range Start"
        )
        sanity_check &= sanity_check_float(
            box_size_x_slider[1], f"Box {i+1} Size X Range End"
        )
        sanity_check &= sanity_check_float(
            box_size_y_slider[0], f"Box {i+1} Size Y Range Start"
        )
        sanity_check &= sanity_check_float(
            box_size_y_slider[1], f"Box {i+1} Size Y Range End"
        )
        sanity_check &= sanity_check_float(
            min_touch_force, f"Box {i+1} Min Touch Force"
        )
        sanity_check &= sanity_check_float(
            orientation_angle, f"Box {i+1} Orientation Angle"
        )
        sanity_check &= sanity_check_rgb(rgb, f"Box {i+1} RGB")
        return sanity_check

    @classmethod
    def parse_values(cls, i, all_args):
        (
            box_position_x,
            box_position_y,
            box_position_z,
            box_size_x_slider,
            box_size_y_slider,
            min_touch_force,
            orientation_angle,
            rgb_btn,
        ) = cls.get_my_args(i, all_args)
        try:
            rgb = hex_to_rgb(rgb_btn)
        except Exception:
            gr.Warning(f"Box {i+1} RGB must be a valid hex color code, got {rgb_btn}")
            return False, []
        sanity_check = cls.sanity_check_args(
            i,
            box_position_x,
            box_position_y,
            box_position_z,
            box_size_x_slider,
            box_size_y_slider,
            min_touch_force,
            orientation_angle,
            rgb,
        )
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(
            f"env.task_config.targets.target_{i}.position=[[{box_position_x[0]},{box_position_y[0]},{box_position_z[0]}],[{box_position_x[1]},{box_position_y[1]},{box_position_z[1]}]]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.size=[[{0.5*box_size_x_slider[0]},{0.5*box_size_y_slider[0]},0.01],[{0.5*box_size_x_slider[1]},{0.5*box_size_y_slider[1]},0.01]]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.min_touch_force={min_touch_force}"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.euler=[0,{-orientation_angle*np.pi/180},0]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]"
        )
        return True, overrides


class SphereParameters:
    num_elements: int = 6

    @staticmethod
    def fields():
        return ["x_range", "y_range", "z_range", "size_range", "dwell_duration", "rgb"]

    @staticmethod
    def get_parameters(i, dwell_duration_min=0.0):
        # Sphere option row
        sphere_rgb = init_target_rgb(i)
        with gr.Row(visible=True) as sphere_row:
            with gr.Accordion(label=f"Sphere {i+1} Settings", open=False):
                with gr.Row():
                    gr.Markdown(
                        "#### The coordinates for the sphere targets are randomly sampled from a range, please choose them below"
                    )
                with gr.Row():
                    x_slider = RangeSlider(
                        label="Depth Range",
                        minimum=sphere_ranges["x"][0],
                        maximum=sphere_ranges["x"][1],
                        value=(sphere_ranges["x"][0], sphere_ranges["x"][1]),
                        step=0.001,
                        interactive=True,
                    )
                    y_slider = RangeSlider(
                        label="Horizontal Range",
                        minimum=sphere_ranges["y"][0],
                        maximum=sphere_ranges["y"][1],
                        value=(sphere_ranges["y"][0], sphere_ranges["y"][1]),
                        step=0.001,
                        interactive=True,
                    )
                    z_slider = RangeSlider(
                        label="Vertical Range",
                        minimum=sphere_ranges["z"][0],
                        maximum=sphere_ranges["z"][1],
                        value=(sphere_ranges["z"][0], sphere_ranges["z"][1]),
                        step=0.001,
                        interactive=True,
                    )
                with gr.Row():
                    size_slider = RangeSlider(
                        label="Size Range (Sphere Radius)",
                        minimum=sphere_ranges["size"][0],
                        maximum=sphere_ranges["size"][1],
                        value=(sphere_ranges["size"][0], sphere_ranges["size"][1]),
                        step=0.001,
                        interactive=True,
                    )
                    dwell_duration = gr.Number(
                        label="Dwell Duration",
                        value=0.25,
                        minimum=dwell_duration_min,
                        maximum=60.0,
                        step=0.05,
                        interactive=True,
                    )
                    color_picker = gr.ColorPicker(
                        label="RGB", value=sphere_rgb, interactive=True
                    )
        return sphere_row, (
            x_slider,
            y_slider,
            z_slider,
            size_slider,
            dwell_duration,
            color_picker,
        )

    def get_my_args(i, all_args):
        rl_number = RLParameters.num_elements
        radio_number = MAX_TARGETS
        box_number = MAX_TARGETS * BoxParameters.num_elements
        num_targets = 1
        sphere_start = rl_number + radio_number + box_number + num_targets
        sphere_args = all_args[sphere_start:]
        start_index = i * SphereParameters.num_elements
        end_index = start_index + SphereParameters.num_elements
        return sphere_args[start_index:end_index]

    @staticmethod
    def sanity_check_args(
        i, x_range, y_range, z_range, size_range, dwell_duration, rgb
    ):
        sanity_check = sanity_check_float(x_range[0], f"Sphere {i+1} X Range Start")
        sanity_check &= sanity_check_float(x_range[1], f"Sphere {i+1} X Range End")
        sanity_check &= sanity_check_float(y_range[0], f"Sphere {i+1} Y Range Start")
        sanity_check &= sanity_check_float(y_range[1], f"Sphere {i+1} Y Range End")
        sanity_check &= sanity_check_float(z_range[0], f"Sphere {i+1} Z Range Start")
        sanity_check &= sanity_check_float(z_range[1], f"Sphere {i+1} Z Range End")
        sanity_check &= sanity_check_float(
            size_range[0], f"Sphere {i+1} Size Range Start"
        )
        sanity_check &= sanity_check_float(
            size_range[1], f"Sphere {i+1} Size Range End"
        )
        sanity_check &= sanity_check_float(
            dwell_duration, f"Sphere {i+1} Dwell Duration"
        )
        sanity_check &= sanity_check_rgb(rgb, f"Sphere {i+1} RGB")
        return sanity_check

    @classmethod
    def parse_values(cls, i, all_args):
        x_range, y_range, z_range, size_range, dwell_duration, rgb = cls.get_my_args(
            i, all_args
        )
        try:
            rgb = hex_to_rgb(rgb)
        except Exception:
            gr.Warning(f"Sphere {i+1} RGB must be a valid hex color code, got {rgb}")
            return False, []
        sanity_check = cls.sanity_check_args(
            i, x_range, y_range, z_range, size_range, dwell_duration, rgb
        )
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(
            f"env.task_config.targets.target_{i}.position=[[{x_range[0]},{y_range[0]},{z_range[0]}],[{x_range[1]},{y_range[1]},{z_range[1]}]]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.size=[{size_range[0]},{size_range[1]}]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.dwell_duration={dwell_duration}"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]"
        )
        return True, overrides


class TaskParameters:
    num_elements: int = 2

    @staticmethod
    def fields():
        return ["max_duration", "reject_oor"]

    @staticmethod
    def get_parameters(ctrl_dt=0.05):
        with gr.Row():
            max_duration = gr.Number(
                label="Maximum Episode Duration (s)",
                value=4.0,
                minimum=0.5,
                maximum=60.0 * 60,
                step=ctrl_dt,
                interactive=True,
            )
            reject_oor = gr.Checkbox(
                label="Reject out-of-reach targets",
                value=False,
                info="Resample targets whose distance from the shoulder exceeds the arm-length proxy (0.65 m).",
                interactive=True,
            )
        return (max_duration, reject_oor)

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = MAX_TARGETS
        box_number = MAX_TARGETS * BoxParameters.num_elements
        sphere_number = MAX_TARGETS * SphereParameters.num_elements
        bm_number = BMParameters.num_elements
        task_start = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
        )
        task_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + TaskParameters.num_elements
        )
        return all_args[task_start:task_end]

    @classmethod
    def parse_values(cls, all_args):
        (max_duration, _reject_oor) = cls.get_my_args(all_args)
        # reject_oor is applied per-target inside args_to_cfg_overrides.
        sanity_check = sanity_check_float(max_duration, "Maximum Episode Duration (s)")
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(f"env.task_config.max_duration={float(max_duration)}")
        return True, overrides


class ObservationSpace:
    num_elements: int = 4

    possible_obs_keys = ["qpos", "qvel", "qacc", "ee_pos", "act"]
    possible_fatigue_obs_keys = ["MA", "MR", "MF"]
    possible_vision_keys = ["rgb", "depth"]
    possible_omni_keys = [
        "target_pos",
        "target_size",
        "phase_progress",
        "dwell_fraction",
    ]

    @classmethod
    def fields(cls):
        return [
            *cls.possible_obs_keys,
            *cls.possible_fatigue_obs_keys,
            *cls.possible_vision_keys,
            *cls.possible_omni_keys,
        ]

    @staticmethod
    def get_parameters():
        obs_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_obs_keys:
                obs_key = gr.Checkbox(label=k, value=True, interactive=True)
                obs_keys.append(obs_key)

        fatigue_obs_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_fatigue_obs_keys:
                fatigue_obs_key = gr.Checkbox(label=k, value=False, interactive=False)
                fatigue_obs_keys.append(fatigue_obs_key)

        vision_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_vision_keys:
                vision_key = gr.Checkbox(label=k, value=False, interactive=True)
                vision_keys.append(vision_key)

        omni_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_omni_keys:
                omni_key = gr.Checkbox(label=k, value=True, interactive=True)
                omni_keys.append(omni_key)

        return obs_keys, fatigue_obs_keys, vision_keys, omni_keys

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = MAX_TARGETS
        box_number = MAX_TARGETS * BoxParameters.num_elements
        sphere_number = MAX_TARGETS * SphereParameters.num_elements
        bm_number = BMParameters.num_elements
        task_number = TaskParameters.num_elements
        obs_start = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
        )
        obs_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + len(ObservationSpace.possible_obs_keys)
        )
        fatigue_obs_start = obs_end
        fatigue_obs_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + len(ObservationSpace.possible_obs_keys)
            + len(ObservationSpace.possible_fatigue_obs_keys)
        )
        vision_start = fatigue_obs_end
        vision_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + len(ObservationSpace.possible_obs_keys)
            + len(ObservationSpace.possible_fatigue_obs_keys)
            + len(ObservationSpace.possible_vision_keys)
        )
        omni_start = vision_end
        omni_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + len(ObservationSpace.possible_obs_keys)
            + len(ObservationSpace.possible_fatigue_obs_keys)
            + len(ObservationSpace.possible_vision_keys)
            + len(ObservationSpace.possible_omni_keys)
        )
        return (
            all_args[obs_start:obs_end],
            all_args[fatigue_obs_start:fatigue_obs_end],
            all_args[vision_start:vision_end],
            all_args[omni_start:omni_end],
        )

    @classmethod
    def parse_values(cls, all_args):
        obs_keys, fatigue_obs_keys, vision_keys, omni_keys = cls.get_my_args(all_args)
        obs_keys_selected = [
            ObservationSpace.possible_obs_keys[id] for id, k in enumerate(obs_keys) if k
        ]  # [k.label for k in obs_keys if k.value]
        fatigue_obs_keys_selected = [
            ObservationSpace.possible_fatigue_obs_keys[id]
            for id, k in enumerate(fatigue_obs_keys)
            if k
        ]  # [k.label for k in fatigue_obs_keys if k.value]
        vision_keys_selected = [
            ObservationSpace.possible_vision_keys[id]
            for id, k in enumerate(vision_keys)
            if k
        ]
        vision_enabled = len(vision_keys_selected) > 0
        vision_enabled_flag = "enabled" if vision_enabled else "disabled"
        vision_mode = None
        if "rgb" in vision_keys_selected:
            if "depth" in vision_keys_selected:
                vision_mode = "rgbd"
            else:
                vision_mode = "rgb"
        elif "depth" in vision_keys_selected:
            vision_mode = "depth"
        omni_keys_selected = [
            ObservationSpace.possible_omni_keys[id]
            for id, k in enumerate(omni_keys)
            if k
        ]  # [k.label for k in omni_keys if k.value]
        overrides = []
        overrides.append(f"env.task_config.obs_keys={obs_keys_selected}")
        overrides.append(
            f"env.muscle_config.fatigue_obs_keys={fatigue_obs_keys_selected}"
        )
        overrides.append(f"vision={vision_enabled_flag}")
        if vision_enabled:
            overrides.append(f"vision.vision_mode={vision_mode}")
        overrides.append(f"env.task_config.omni_keys={omni_keys_selected}")
        return True, overrides


class RewardFunction:
    num_elements: int = 1

    weights_default_min_max_step = {
        "distance": (1, 0, 10, 0.1),
        "subtask_bonus": (0, 0, 10, 0.5),
        "completion_bonus": (10, 0, 50, 1),
        "neural_effort": (0, 0, 1, 0.01),
        "body_contact_penalty": (0, 0, 0.01, 0.0001),
    }
    keys = [
        "distance",
        "subtask_bonus",
        "completion_bonus",
        "neural_effort",
        "body_contact_penalty",
    ]

    @staticmethod
    def fields():
        return [*RewardFunction.keys]

    @staticmethod
    def get_parameters():
        reward_weights = []
        with gr.Row():
            for k, (
                value,
                minimum,
                maximum,
                step,
            ) in RewardFunction.weights_default_min_max_step.items():
                reward_weight = gr.Number(
                    label=k,
                    value=value,
                    minimum=minimum,
                    maximum=maximum,
                    step=step,
                    interactive=True,
                )
                reward_weights.append(reward_weight)

        return (reward_weights,)

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = MAX_TARGETS
        box_number = MAX_TARGETS * BoxParameters.num_elements
        sphere_number = MAX_TARGETS * SphereParameters.num_elements
        bm_number = BMParameters.num_elements
        task_number = TaskParameters.num_elements
        obs_number = (
            len(ObservationSpace.possible_obs_keys)
            + len(ObservationSpace.possible_fatigue_obs_keys)
            + len(ObservationSpace.possible_vision_keys)
            + len(ObservationSpace.possible_omni_keys)
        )
        reward_start = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + obs_number
        )
        reward_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + obs_number
            + len(RewardFunction.weights_default_min_max_step)
        )
        return all_args[reward_start:reward_end]

    @staticmethod
    def sanity_check_args(reward_weights):
        sanity_check = True
        for i, k in enumerate(reward_weights):
            key = RewardFunction.keys[i]
            sanity_check &= sanity_check_float(k, f"Reward Weight {key}")
        return sanity_check

    @classmethod
    def parse_values(cls, all_args):
        reward_weights = cls.get_my_args(all_args)
        sanity_check = cls.sanity_check_args(reward_weights)
        if not sanity_check:
            return False, []
        overrides = []
        for i, k in enumerate(reward_weights):
            key = RewardFunction.keys[i]
            rename_keys = {
                "distance": "distance",
                "subtask_bonus": "phase_bonus",
                "completion_bonus": "done",
                "body_contact_penalty": "torso_contact_penalty",
            }
            key = rename_keys.get(key, key)
            overrides.append(f"env.task_config.weighted_reward_keys.{key}={k}")
        return True, overrides


class RLParameters:
    num_elements: int = 10
    num_mds: int = 1

    @staticmethod
    def fields():
        return [
            "Number of Total Training Steps",
            "Number of Checkpoints During/After Training",
            "Number of Evaluations During/After Training",
            "Batch Size",
            "Number of Parallel Environments",
            "Number of Minibatches",
            "Select Experiment from Which to Load Checkpoints",
            "Select Checkpoint Number",
            "Show Checkpoints from Own Training Runs",
            "Target Initial Seed",
        ]

    @staticmethod
    def get_parameters():
        with gr.Row():
            num_timesteps = gr.Number(
                label="Number of Total Training Steps",
                value=15000000,
                minimum=0,
                maximum=1000000000,
                step=100_000,
                interactive=True,
            )
            checkpoint_interval = gr.Number(
                label="Checkpoint Interval (in Number of Training Iterations)",
                value=30,
                minimum=1,
                maximum=1000,
                step=10,
                interactive=True,
            )
        md_rl_note = gr.Markdown(
            "<span style='font-size: 1em;'>"
            "<b><span style='color:red'>Note:</span></b> If GPU memory errors occur, try with a smaller <i>number of parallel environments</i>."
            "</span>",
            elem_id="hint-text",
        )
        with gr.Row():
            num_envs = gr.Number(
                label="Number of Parallel Environments",
                value=1024,
                minimum=0,
                maximum=4096,
                interactive=True,
            )
            unroll_length = gr.Number(
                label="Unroll Length",
                value=10,
                minimum=0,
                maximum=200,
                interactive=True,
            )
            num_minibatches = gr.Number(
                label="Number of Minibatches",
                value=8,
                minimum=1,
                maximum=40,
                interactive=True,
            )
            num_epochs_per_update = gr.Number(
                label="Number of Epochs Per Update",
                value=8,
                minimum=1,
                maximum=50,
                interactive=True,
            )
        target_init_seed = gr.Number(
            label="Target Initial Seed",
            value=0,
            minimum=0,
            maximum=1000000,
            interactive=True,
            visible=False,
        )
        # Checkpoint dropdowns are now created in the "Load / Resume from Checkpoint"
        # section of get_ui() and injected back into run_inputs at positions 6-8.
        return (
            num_timesteps,
            checkpoint_interval,
            num_envs,
            unroll_length,
            num_minibatches,
            num_epochs_per_update,
            target_init_seed,
            md_rl_note,
        )

    @staticmethod
    def get_my_args(all_args):
        return all_args[: RLParameters.num_elements]

    @staticmethod
    def sanity_check_args(
        num_targets,
        num_timesteps,
        checkpoint_interval,
        num_envs,
        unroll_length,
        num_minibatches,
        num_epochs_per_update,
        select_checkpoint_run,
        select_checkpoint_file,
        show_training_ckpts,
        target_init_seed,
    ):
        sanity_check = sanity_check_number_targets(num_targets, "Number of Targets")
        sanity_check &= sanity_check_int(
            num_timesteps, "Number of Total Training Steps"
        )
        sanity_check &= sanity_check_int(
            checkpoint_interval,
            "Checkpoint Interval (in Number of Training Iterations)",
        )
        sanity_check &= sanity_check_int(num_envs, "Number of Parallel Environments")
        sanity_check &= sanity_check_int(unroll_length, "Unroll Length")
        sanity_check &= sanity_check_int(num_minibatches, "Number of Minibatches")
        sanity_check &= sanity_check_int(
            num_epochs_per_update, "Number of Epochs Per Update"
        )
        # Checkpoint dropdowns allow arbitrary values (allow_custom_value=True) so that
        # wandb download paths and custom local dirs can be set without failing validation.
        sanity_check &= sanity_check_bool(
            show_training_ckpts, "Show Checkpoints from Own Training Runs"
        )
        sanity_check &= sanity_check_int(target_init_seed, "Target Initial Seed")
        return sanity_check

    @classmethod
    def parse_values(cls, all_args):
        (
            num_timesteps,
            checkpoint_interval,
            num_envs,
            unroll_length,
            num_minibatches,
            num_epochs_per_update,
            select_checkpoint_run,
            select_checkpoint_file,
            show_training_ckpts,
            target_init_seed,
        ) = cls.get_my_args(all_args)
        overrides = []
        num_targets = cls.get_number_targets(all_args)
        sanity_check = cls.sanity_check_args(
            num_targets,
            num_timesteps,
            checkpoint_interval,
            num_envs,
            unroll_length,
            num_minibatches,
            num_epochs_per_update,
            select_checkpoint_run,
            select_checkpoint_file,
            show_training_ckpts,
            target_init_seed,
        )
        if not sanity_check:
            return False, []
        to_text = [
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
        ]
        _n = int(num_targets)
        if _n <= MAX_TARGETS:
            overrides.append(f"env/task_config/targets={to_text[_n]}")
        else:
            # No Hydra target config exists for N>MAX_TARGETS.
            # Use single-target base; auto-copy in add_task_relevant_geoms() replicates target_0.
            overrides.append("env/task_config/targets=one")
            overrides.append(f"env.task_config.targets.num_targets={_n}")
        overrides.append(f"rl.num_timesteps={int(num_timesteps)}")
        overrides.append(f"rl.save_interval={int(checkpoint_interval)}")
        overrides.append(f"rl.num_envs={int(num_envs)}")
        overrides.append(f"rl.unroll_length={int(unroll_length)}")
        overrides.append(f"rl.num_minibatches={int(num_minibatches)}")
        overrides.append(f"rl.num_epochs_per_update={int(num_epochs_per_update)}")
        exact_checkpoint_path = checkpoint_path_from_run_filename(
            select_checkpoint_run, select_checkpoint_file
        )
        overrides.append(f"rl.load_checkpoint_path={exact_checkpoint_path}")
        overrides.append(f"env.task_config.target_init_seed={int(target_init_seed)}")
        return True, overrides

    @classmethod
    def get_number_targets(cls, all_args):
        return all_args[RLParameters.num_elements]

    @classmethod
    def get_target_init_seed(cls, all_args):
        return all_args[RLParameters.num_elements - 1]


def update_dwell_duration(dwell_duration, ctrl_dt):
    return gr.update(minimum=max(ctrl_dt, 0))


# Gradio-authored UI configs (ConfigSaver's own labelled-dict shape: top-level keys
# rl/num_targets/target_N/bm/task/obs/reward) live in myo_config/gradio_tasks/, kept
# separate from myo_config/tasks/ -- the latter holds mjlab/Hydra training configs
# (top-level keys env/rl/run/vision/wandb, loaded directly by scripts like
# the training scripts via ConfigDict(json.load(...)), never through ConfigSaver).
# The two shapes both happen to have a top-level "rl" key with different sub-schemas,
# so they are not safely distinguishable by content alone -- keeping them in separate
# directories is what actually prevents an mjlab config from being unusable when picked
# from this UI's own "load saved config" dropdown (ConfigSaver.load_config below still
# auto-converts via training_config_to_labelled_dict as a defense-in-depth fallback, in
# case an mjlab-shaped file ends up in this folder anyway).
CONFIG_PATH = Path(__file__).resolve().parents[2] / "myo_config" / "gradio_tasks"


class ConfigSaver:
    def check_folder(self):
        if not os.path.exists(CONFIG_PATH):
            os.makedirs(CONFIG_PATH)
        files = os.listdir(CONFIG_PATH)
        json_files = [file for file in files if file.endswith(".json")]
        file_names = [file.split(".")[0] for file in json_files]
        return file_names

    def __init__(self):
        self.my_configs = self.check_folder()
        self.run_inputs = None

    def to_labelled_dict(self, data):
        labelled_dict = {}
        rl_dict = dict(zip(RLParameters.fields(), RLParameters.get_my_args(data)))
        labelled_dict["rl"] = rl_dict
        labelled_dict["num_targets"] = data[RLParameters.num_elements]
        radio_start = RLParameters.num_elements + 1
        radio_end = radio_start + MAX_TARGETS
        [f"target_{i+1}_radio" for i in range(MAX_TARGETS)]
        radio_values = data[radio_start:radio_end]
        # labelled_dict.update(zip(radio_keys, radio_values))
        for i in range(MAX_TARGETS):
            item_dict = {}

            item_dict["radio"] = radio_values[i]
            item_dict["box"] = dict(
                zip(BoxParameters.fields(), BoxParameters.get_my_args(i, data))
            )
            item_dict["sphere"] = dict(
                zip(SphereParameters.fields(), SphereParameters.get_my_args(i, data))
            )
            labelled_dict[f"target_{i+1}"] = item_dict
        labelled_dict["bm"] = dict(
            zip(BMParameters.fields(), BMParameters.get_my_args(data))
        )
        labelled_dict["task"] = dict(
            zip(TaskParameters.fields(), TaskParameters.get_my_args(data))
        )
        labelled_dict["obs"] = dict(
            zip(ObservationSpace.fields(), ObservationSpace.get_my_args(data))
        )
        labelled_dict["reward"] = dict(
            zip(RewardFunction.fields(), RewardFunction.get_my_args(data))
        )
        return labelled_dict

    @staticmethod
    def _default_target(i):
        """Return a default target dict for index i, matching the UI's initial widget values."""
        bp = BoxParameters.init_button_positions(i)
        rgb = init_target_rgb(i)
        return {
            "radio": "Sphere",
            "box": {
                "box_position_x": (bp[0], bp[0]),
                "box_position_y": (bp[1], bp[1]),
                "box_position_z": (bp[2], bp[2]),
                "box_size_x": (0.05, 0.05),
                "box_size_y": (0.05, 0.05),
                "min_touch_force": 1.0,
                "orientation_angle": 45,
                "rgb": rgb,
            },
            "sphere": {
                "x_range": (sphere_ranges["x"][0], sphere_ranges["x"][1]),
                "y_range": (sphere_ranges["y"][0], sphere_ranges["y"][1]),
                "z_range": (sphere_ranges["z"][0], sphere_ranges["z"][1]),
                "size_range": (sphere_ranges["size"][0], sphere_ranges["size"][1]),
                "dwell_duration": 0.25,
                "rgb": rgb,
            },
        }

    def from_labelled_dict(self, labelled_data):
        data = []
        data.extend(labelled_data["rl"].values())
        data.append(labelled_data["num_targets"])
        targets = [
            labelled_data.get(f"target_{i+1}") or self._default_target(i)
            for i in range(MAX_TARGETS)
        ]
        for t in targets:
            data.append(t["radio"])
        for i, t in enumerate(targets):
            box = t.get("box") or self._default_target(i)["box"]
            data.extend(box.values())
        for i, t in enumerate(targets):
            sphere = t.get("sphere") or self._default_target(i)["sphere"]
            data.extend(sphere.values())
        data.extend(labelled_data["bm"].values())
        _task = labelled_data.get("task", {})
        _task_defaults = {"max_duration": 4.0, "reject_oor": False}
        data.extend(_task.get(k, _task_defaults[k]) for k in TaskParameters.fields())
        obs_values = labelled_data["obs"].values()
        data.extend(sum(obs_values, []))
        reward_defaults = {
            k: v[0] for k, v in RewardFunction.weights_default_min_max_step.items()
        }
        data.extend(
            labelled_data["reward"].get(k, reward_defaults[k])
            for k in RewardFunction.fields()
        )
        return data

    def add_config(self, config_name, data):
        labelled_dict = self.to_labelled_dict(data)
        # Settings without a UI control (e.g. the reward terms action_rate, pain_penalty, pronation,
        # button_force) that came with the loaded preset or checkpoint stay in the saved preset.
        extra = getattr(
            getattr(self, "run_state", None), "loaded_policy_extra_overrides", []
        )
        if extra:
            labelled_dict["extra_overrides"] = list(extra)
        with open(CONFIG_PATH / f"{config_name}.json", "w") as f:
            json.dump(labelled_dict, f, indent=4)

    def config_save_clicked(self, config_name_input, *args):
        if config_name_input == "":
            gr.Warning("Please enter a name for the configuration.")
        else:
            self.add_config(config_name_input, args)
            if config_name_input not in self.my_configs:
                self.my_configs.append(config_name_input)
        return gr.update(choices=self.my_configs, value=config_name_input)

    def available_configs(self):
        return self.my_configs

    def load_config(self, config_name):
        with open(CONFIG_PATH / f"{config_name}.json") as f:
            raw_data = json.load(f)
        # Defense-in-depth: CONFIG_PATH is scoped to gradio-shaped configs only, but
        # an mjlab/Hydra training config (top-level "env" key) landing here anyway
        # would otherwise KeyError inside from_labelled_dict -- auto-convert instead
        # of crashing, the same way the "Load / Resume from Checkpoint" flow already
        # does for training configs bundled with a run.
        if "env" in raw_data:
            labelled_data = training_config_to_labelled_dict(raw_data)
            if getattr(self, "run_state", None) is not None:
                self.run_state.loaded_policy_extra_overrides = extra_hydra_overrides(
                    raw_data
                )
        else:
            labelled_data = raw_data
            # A preset's settings without a UI control apply to training like a checkpoint's do.
            if getattr(self, "run_state", None) is not None:
                self.run_state.loaded_policy_extra_overrides = list(
                    raw_data.get("extra_overrides", [])
                )
        data = self.from_labelled_dict(labelled_data)
        return tuple([gr.update(value=k) for k in data])


def load_test_environment():
    task_id = "myoInteractUniversal-v0"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    play_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda", render_mode="rgb_array")

    return play_env


def get_ui(project_name, run_state=RunState(), use_legacy_rendering=False):
    # Fix box handlers properly
    with gr.Blocks() as demo:
        pageview = gr.Radio(
            choices=["Simple", "Advanced"],
            label="Show Advanced Options",
            value="Advanced",
            interactive=True,
        )

        gr.Markdown("**Weights & Biases URL:**")
        gr.Textbox(
            value="https://wandb.ai/home",
            label="Results Dashboard",
            interactive=False,
            buttons=["copy"],
            info="Once your run starts, you can view the training progress for your projects here. Click to copy the URL to go to the wandb project and monitor training progress",
        )

        gr.Markdown("### Pre-saved configurations")
        config_saver = ConfigSaver()
        with gr.Row():
            pre_saved_configs = gr.Dropdown(
                label="Pre-saved configurations",
                choices=config_saver.available_configs(),
                value="public-display-4-buttons",
                interactive=True,
                visible=True,
            )

        # ── Load / Resume from Checkpoint ─────────────────────────────────────
        with gr.Accordion("Load / Resume from Checkpoint", open=False):
            gr.Markdown(
                "Paste a **W&B run URL** or a **local ONNX path**, *or* pick a local run "
                "from the dropdowns below.  \n"
                "Optionally check **Also load training config** to populate all UI fields "
                "from the policy's training settings.  \n"
                "Click **Update Preview** afterwards to render the loaded policy."
            )
            load_policy_input = gr.Textbox(
                label="W&B Run URL / Run ID / Local Path",
                placeholder=(
                    "https://wandb.ai/entity/project/runs/RUN_ID  |  RUN_ID  "
                    "|  /abs/path/to/run_dir  |  run_dir"
                ),
                interactive=True,
            )
            with gr.Row():
                show_training_ckpts = gr.Checkbox(
                    label="Show own training runs",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                select_checkpoint_run = gr.Dropdown(
                    label="Select Local Experiment",
                    choices=get_available_checkpoints(False),
                    value="None",
                    interactive=True,
                    allow_custom_value=True,
                )
                select_checkpoint_file = gr.Dropdown(
                    label="Select Checkpoint File",
                    choices=get_available_checkpoint_numbers("None", False),
                    value="None",
                    interactive=True,
                    allow_custom_value=True,
                )
            load_config_too = gr.Checkbox(
                label="Also load training config into UI (overrides current UI settings)",
                value=True,
                interactive=True,
            )
            load_policy_button = gr.Button("Load Checkpoint")
            load_policy_status = gr.Textbox(label="Status", interactive=False, lines=2)

        md1 = gr.Markdown("### Biomechanical Model Parameters")
        bm_params = BMParameters.get_parameters()
        (
            bm_model,
            ctrl_dt,
            reset_type,
            sigdepnoise_enabled,
            constantnoise_enabled,
            fatigue_enabled,
            mj_impl_type,
        ) = bm_params

        gr.Markdown("### Task Parameters")
        gr.Markdown("#### Target Setup")
        num_elements = gr.Number(
            label="Number of Targets",
            value=INIT_ELEMENTS,
            minimum=0,
            maximum=500,
            precision=0,
            interactive=True,
        )

        # Create MAX_TARGETS sets of elements (maximum possible), first INIT_ELEMENTS visible by default
        dynamic_rows = []
        radios = []
        dwell_durations = []
        box_rows = []
        sphere_rows = []

        # Store all components for easy access
        all_components = {"boxes": [], "spheres": []}

        # Create variable that sums up all selected dwell times
        gr.Number(
            label="total Dwell Time",
            value=0,
            minimum=0,
            step=0.01,
            interactive=False,
            visible=False,
        )

        for i in range(MAX_TARGETS):
            # Main row with radio selection
            with gr.Row(visible=(i < INIT_ELEMENTS)) as main_row:
                with gr.Column():
                    radio = gr.Radio(
                        choices=["Box", "Sphere"],
                        label=f"Target {i+1} Type",
                        value="Sphere",
                        interactive=True,
                    )
                    box_row, box_params = BoxParameters.get_parameters(i)
                    (
                        box_position_x,
                        box_position_y,
                        box_position_z,
                        box_size_x_slider,
                        box_size_y_slider,
                        min_touch_force,
                        orientation_angle,
                        rgb_btn,
                    ) = box_params
                    # Store box components
                    all_components["boxes"].append(
                        {
                            key: value
                            for key, value in zip(BoxParameters.fields(), box_params)
                        }
                    )

                    sphere_row, sphere_params = SphereParameters.get_parameters(
                        i, dwell_duration_min=ctrl_dt.value
                    )
                    (
                        x_slider,
                        y_slider,
                        z_slider,
                        size_slider,
                        dwell_duration,
                        color_picker,
                    ) = sphere_params
                    ctrl_dt.change(
                        fn=update_dwell_duration,
                        inputs=(dwell_duration, ctrl_dt),
                        outputs=dwell_duration,
                        preprocess=False,
                    )
                    # Store sphere components
                    all_components["spheres"].append(
                        {
                            key: value
                            for key, value in zip(
                                SphereParameters.fields(), sphere_params
                            )
                        }
                    )

            # Store references
            dynamic_rows.append(main_row)
            radios.append(radio)
            dwell_durations.append(dwell_duration)
            box_rows.append(box_row)
            sphere_rows.append(sphere_row)

        md22 = gr.Markdown("#### Other Task Parameters")
        task_params = TaskParameters.get_parameters(ctrl_dt=ctrl_dt.value)
        (max_duration, reject_oor_checkbox) = task_params

        md23 = gr.Markdown("#### Observation Space")
        obs_keys, fatigue_obs_keys, vision_keys, omni_keys = (
            ObservationSpace.get_parameters()
        )

        gr.Markdown("#### Reward Weights")

        def get_max_dist(num_elements, *radios_and_box_and_sphere_positions):
            ee_pos0 = np.array([0.0, -0.27, 0.37])  # TODO: infer from model!

            _num_targets_max = int(
                len(radios_and_box_and_sphere_positions) // (1 + 2 * 3)
            )
            n_total = int(num_elements)
            num_configurable = min(n_total, _num_targets_max)

            radios = radios_and_box_and_sphere_positions[:num_configurable]
            box_positions_x = radios_and_box_and_sphere_positions[
                _num_targets_max : 2 * _num_targets_max
            ]
            box_positions_y = radios_and_box_and_sphere_positions[
                2 * _num_targets_max : 3 * _num_targets_max
            ]
            box_positions_z = radios_and_box_and_sphere_positions[
                3 * _num_targets_max : 4 * _num_targets_max
            ]
            sphere_positions_x = radios_and_box_and_sphere_positions[
                4 * _num_targets_max : 5 * _num_targets_max
            ]
            sphere_positions_y = radios_and_box_and_sphere_positions[
                5 * _num_targets_max : 6 * _num_targets_max
            ]
            sphere_positions_z = radios_and_box_and_sphere_positions[
                6 * _num_targets_max : 7 * _num_targets_max
            ]

            def _max_dist_between_ranges(min_a, max_a, min_b, max_b):
                """Upper-bound Euclidean distance between two axis-aligned bounding boxes."""
                # Enumerate all 8 corners of each box; max distance is at a corner pair.
                corners_a = np.array(
                    [
                        [
                            min_a[0] if not (i & 4) else max_a[0],
                            min_a[1] if not (i & 2) else max_a[1],
                            min_a[2] if not (i & 1) else max_a[2],
                        ]
                        for i in range(8)
                    ]
                )
                corners_b = np.array(
                    [
                        [
                            min_b[0] if not (j & 4) else max_b[0],
                            min_b[1] if not (j & 2) else max_b[1],
                            min_b[2] if not (j & 1) else max_b[2],
                        ]
                        for j in range(8)
                    ]
                )
                diffs = corners_a[:, None, :] - corners_b[None, :, :]  # (8, 8, 3)
                return float(np.linalg.norm(diffs, axis=-1).max())

            # Build position ranges (min, max) for each configurable target.
            target_ranges: list[tuple[np.ndarray, np.ndarray]] = []
            for target_id in range(num_configurable):
                if radios[target_id] == "Box":
                    box_x = box_positions_x[target_id]
                    box_y = box_positions_y[target_id]
                    box_z = box_positions_z[target_id]
                    sanity_check = sanity_check_float_array(
                        box_x, f"Box {target_id+1} Range Position (Depth)"
                    )
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(
                        box_y, f"Box {target_id+1} Range Position (Horizontal)"
                    )
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(
                        box_z, f"Box {target_id+1} Range Position (Vertical)"
                    )
                    if not sanity_check:
                        return False, 0
                    target_ranges.append(
                        (
                            np.array([box_x[0], box_y[0], box_z[0]]),
                            np.array([box_x[1], box_y[1], box_z[1]]),
                        )
                    )
                elif radios[target_id] == "Sphere":
                    sphere_x = sphere_positions_x[target_id]
                    sphere_y = sphere_positions_y[target_id]
                    sphere_z = sphere_positions_z[target_id]
                    sanity_check = sanity_check_float_array(
                        sphere_x, f"Sphere {target_id+1} Range Position (Depth)"
                    )
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(
                        sphere_y, f"Sphere {target_id+1} Range Position (Horizontal)"
                    )
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(
                        sphere_z, f"Sphere {target_id+1} Range Position (Vertical)"
                    )
                    if not sanity_check:
                        return False, 0
                    target_ranges.append(
                        (
                            np.array([sphere_x[0], sphere_y[0], sphere_z[0]]),
                            np.array([sphere_x[1], sphere_y[1], sphere_z[1]]),
                        )
                    )
                else:
                    raise NotImplementedError()

            # Targets beyond the configurable ones copy target_0's range.
            if target_ranges:
                t0_range = target_ranges[0]
                target_ranges.extend([t0_range] * (n_total - num_configurable))

            # Sum upper-bound distances: ee_pos0 → target_0 → ... → target_{N-1}.
            total_max = 0.0
            prev_min = prev_max = ee_pos0
            for cur_min, cur_max in target_ranges:
                total_max += _max_dist_between_ranges(
                    prev_min, prev_max, cur_min, cur_max
                )
                prev_min, prev_max = cur_min, cur_max

            return True, total_max

        def update_max_dist(num_elements, *radios_and_box_and_sphere_positions):
            sanity_check, max_dist = get_max_dist(
                num_elements, *radios_and_box_and_sphere_positions
            )
            if not sanity_check:
                return gr.skip()
            return gr.update(value=max_dist)

        radios_and_box_and_sphere_positions = (
            radios
            + [b["box_position_x"] for b in all_components["boxes"]]
            + [b["box_position_y"] for b in all_components["boxes"]]
            + [b["box_position_z"] for b in all_components["boxes"]]
            + [s["x_range"] for s in all_components["spheres"]]
            + [s["y_range"] for s in all_components["spheres"]]
            + [s["z_range"] for s in all_components["spheres"]]
        )
        max_dist = gr.Number(
            label="Maximum Path Length",
            value=get_max_dist(
                num_elements.value,
                *[r.value for r in radios_and_box_and_sphere_positions],
            )[1],
            interactive=False,
            visible=False,
        )
        for k in radios_and_box_and_sphere_positions + [num_elements]:
            k.change(
                fn=update_max_dist,
                inputs=(num_elements, *radios_and_box_and_sphere_positions),
                outputs=max_dist,
                preprocess=False,
            )

        num_ctrls = 26  # TODO: infer from chosen MuJoCo model

        def reward_fct_view(
            num_elements,
            distance,
            phase_bonus,
            done,
            neural_effort,
            body_contact_penalty,
            max_dist,
        ):
            return f"""<span style='font-size: 1em;'>The following **Reward** will be provided at each time step *n*, depending on the current target *i*:
            <div align="center">
            $r_n =$ <span title="min: {-1*distance*max_dist:.4g};
            max: {0}">${-1*distance} \\cdot (\\text{{distance to current target }} i + \\sum_{{j=i+1}}^{{{num_elements}}}\\text{{dist(target }}j\\text{{, target }} j-1\\text{{)}})$</span>

            <span title="min: {0};
            max: {1*phase_bonus}">$+ {1*phase_bonus} \\cdot (\\text{{current target }} i \\text{{ successfully hit for the first time}})$</span>

            <span title="min: {0};
            max: {1*done}">$+ {1*done} \\cdot (\\text{{task successfully completed}})$</span>

            <span title="min: {-1*neural_effort*num_ctrls};
            max: {0}">$- {1*neural_effort} \\cdot (\\text{{squared control effort costs}})$</span>

            <span title="max: {0}">$- {1*body_contact_penalty} \\cdot (\\text{{total contact force between the torso and other body parts}})$</span>
            </div></span>"""

        (
            _distance_d,
            _phase_bonus_d,
            _done_d,
            _neural_effort_d,
            _body_contact_penalty_d,
        ) = (
            RewardFunction.weights_default_min_max_step["distance"][0],
            RewardFunction.weights_default_min_max_step["subtask_bonus"][0],
            RewardFunction.weights_default_min_max_step["completion_bonus"][0],
            RewardFunction.weights_default_min_max_step["neural_effort"][0],
            RewardFunction.weights_default_min_max_step["body_contact_penalty"][0],
        )
        reward_function_text = gr.Markdown(
            reward_fct_view(
                num_elements=num_elements.value,
                distance=_distance_d,
                phase_bonus=_phase_bonus_d,
                done=_done_d,
                neural_effort=_neural_effort_d,
                body_contact_penalty=_body_contact_penalty_d,
                max_dist=max_dist.value,
            ),
            elem_id="reward-function",
            line_breaks=True,
            latex_delimiters=[{"left": "$", "right": "$", "display": False}],
        )

        def update_reward_fct_view(
            num_elements,
            distance,
            phase_bonus,
            done,
            neural_effort,
            body_contact_penalty,
            max_dist,
        ):
            return gr.update(
                value=reward_fct_view(
                    num_elements=num_elements,
                    distance=distance,
                    phase_bonus=phase_bonus,
                    done=done,
                    neural_effort=neural_effort,
                    body_contact_penalty=body_contact_penalty,
                    max_dist=max_dist,
                )
            )

        (reward_weights,) = RewardFunction.get_parameters()
        weighted_reward_keys_gr = {k.label: k for k in reward_weights}
        for k in reward_weights + [num_elements, max_dist]:
            k.change(
                update_reward_fct_view,
                [
                    num_elements,
                    weighted_reward_keys_gr["distance"],
                    weighted_reward_keys_gr["subtask_bonus"],
                    weighted_reward_keys_gr["completion_bonus"],
                    weighted_reward_keys_gr["neural_effort"],
                    weighted_reward_keys_gr["body_contact_penalty"],
                    max_dist,
                ],
                reward_function_text,
            )

        md3 = gr.Markdown("### RL Parameters")
        rl_params_and_mds = RLParameters.get_parameters()
        # get_parameters() now returns 7 non-checkpoint RL params + 1 markdown note.
        # Checkpoint dropdowns (select_checkpoint_run/file, show_training_ckpts) live in
        # the "Load / Resume" accordion above and are injected into run_inputs at positions
        # 6-8 to preserve the layout that _build_policy_and_ckpt_manager() and
        # args_to_cfg_overrides() depend on (RLParameters.num_elements stays 10).
        _n_rl_no_ckpts = RLParameters.num_elements - 3  # 7
        rl_params = rl_params_and_mds[:_n_rl_no_ckpts]
        md_rl_notes = rl_params_and_mds[-RLParameters.num_mds :]
        num_timesteps = rl_params[0]
        target_init_seed = rl_params[-1]  # last of the 7 non-checkpoint params

        gr.Markdown("### Preview of the environment")
        render_button = gr.Button("Update Preview", variant="primary", size="lg")
        load_training_button = gr.Button(
            "Show Training Policy",
            variant="secondary",
            size="lg",
        )
        with gr.Row(visible=False) as env_view_row:
            if use_legacy_rendering:
                env_view_1 = gr.Image(label="Environment View", interactive=False)
                env_view_2 = gr.Image(label="Environment View", interactive=False)
            else:
                env_view = gr.HTML()

        gr.Markdown("### Save current configuration")
        with gr.Row():
            config_name_input = gr.Textbox(
                label="Configuration Name", value="", interactive=True
            )
            save_config_button = gr.Button(
                "Save/Update Configuration", variant="primary", size="lg"
            )
        # Add Run button and output
        gr.Markdown("### Send to Training")
        gr.Markdown("""This sets up the current configuration and sends it to the training pipeline.

                    Make sure to run the next notebook cell to start training with this configuration.""")
        with gr.Row():
            run_button = gr.Button("Setup Training", variant="primary", size="lg")

        output_text = gr.Textbox(
            label="Configuration Output",
            lines=20,
            max_lines=30,
            interactive=False,
            buttons=["copy"],
        )

        def args_to_cfg_overrides(run_name, *args):
            """Print all configuration details"""
            # Extract values from args
            num_targets = args[RLParameters.num_elements]
            radio_start = RLParameters.num_elements + 1
            radio_end = radio_start + MAX_TARGETS
            radio_values = args[radio_start:radio_end]

            cfg_overrides = [
                "env=universal",
                "run.using_gradio=True",
                f"wandb.project={project_name}",
                f"wandb.name={run_name}",
            ]
            all_sanity_checks = True
            sanity_check, bm_overrides = BMParameters.parse_values(args)
            cfg_overrides.extend(bm_overrides)
            sanity_check, task_overrides = TaskParameters.parse_values(args)
            all_sanity_checks &= sanity_check
            cfg_overrides.extend(task_overrides)
            sanity_check, obs_overrides = ObservationSpace.parse_values(args)
            all_sanity_checks &= sanity_check
            cfg_overrides.extend(obs_overrides)
            sanity_check, rl_overrides = RLParameters.parse_values(args)
            all_sanity_checks &= sanity_check
            # If a policy was loaded via the Load/Resume section and the dropdowns
            # still show "None", use run_state.loaded_checkpoint_path as fallback.
            if run_state.loaded_checkpoint_path:
                rl_overrides = [
                    o
                    if not o.startswith("rl.load_checkpoint_path=None")
                    else f"rl.load_checkpoint_path={run_state.loaded_checkpoint_path}"
                    for o in rl_overrides
                ]
            sanity_check, reward_overrides = RewardFunction.parse_values(args)
            all_sanity_checks &= sanity_check
            cfg_overrides.extend(reward_overrides)
            cfg_overrides.extend(rl_overrides)

            # When N > MAX_TARGETS only target_0 is shown/configurable in the GUI.
            # The backend auto-copies target_0 config for all N targets.
            # For N <= MAX_TARGETS all N targets are configurable.
            num_configurable = 1 if int(num_targets) > MAX_TARGETS else int(num_targets)
            _max_dur, reject_oor = TaskParameters.get_my_args(args)
            for i in range(num_configurable):
                target_type = radio_values[i]
                cfg_overrides.append(
                    f"+env/task_config/targets/target_{i}={target_type.lower()}"
                )

                if target_type == "Box":
                    sanity_check, overrides = BoxParameters.parse_values(i, args)
                    all_sanity_checks &= sanity_check
                    if not sanity_check:
                        return False, []
                    cfg_overrides.extend(overrides)

                else:  # Sphere
                    sanity_check, overrides = SphereParameters.parse_values(i, args)
                    all_sanity_checks &= sanity_check
                    if not sanity_check:
                        return False, []
                    cfg_overrides.extend(overrides)

                cfg_overrides.append(
                    f"+env.task_config.targets.target_{i}.reject_oor={bool(reject_oor)}"
                )

            return all_sanity_checks, cfg_overrides

        def run_training(run_name, *args):
            if run_name == "":
                gr.Warning("Please enter a name for the configuration.")
                text = "Error: You must enter a name for the configuration."
                return text
            sanity_check, cfg_overrides = args_to_cfg_overrides(run_name, *args)
            if not sanity_check:
                gr.Warning("Invalid configuration.")
                return "Error: Invalid configuration."
            # cfg = load_config_interactive(cfg_overrides, cfg_only=True)
            gr.Info("Go back to the notebook and run the next cell to start training!")
            run_state.cfg_overrides.clear()
            run_state.cfg_overrides.extend(
                _dedup_overrides(
                    run_state.loaded_policy_extra_overrides + cfg_overrides
                )
            )
            run_state.wandb_run_name = run_name
            # train(cfg)
            text = "Go to the next notebook in the cell and run the training!\n"
            text += "\n".join(run_state.cfg_overrides)
            return text

        def _render_environment_legacy(run_name, *args):
            import mujoco

            target_init_seed = RLParameters.get_target_init_seed(args)
            next_seed = target_init_seed + 1
            sanity_check, cfg_overrides = args_to_cfg_overrides(run_name, *args)
            if not sanity_check:
                gr.Warning("Invalid configuration.")
                return gr.skip()
            config = load_config_interactive(cfg_overrides)

            xml_file = _get_myosuite_root() / config.env.model_path
            if not xml_file.exists():
                gr.Warning(
                    f"Model XML not found in myosuite assets: {config.env.model_path}"
                )
                return gr.skip()

            spec = _universal_spec_fn(xml_file, config)
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            imgs = []
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                for cam in ["head_on_camera", "side_camera"]:
                    renderer.update_scene(data, camera=cam)
                    imgs.append(renderer.render())

            return (
                gr.update(value=next_seed),
                gr.update(visible=True),
                gr.update(value=imgs[0]),
                gr.update(value=imgs[1]),
            )

        def _capture_camera_state():
            """Return {position, look_at} for the first connected viser client, or None."""
            clients = _viser_preview_server.get_clients()
            if not clients:
                return None
            client = next(iter(clients.values()))
            try:
                return {
                    "position": client.camera.position.copy(),
                    "look_at": client.camera.look_at.copy(),
                }
            except Exception:
                return None

        def _stop_current_viewer() -> None:
            """Stop the running viewer thread and properly close the viewer."""
            if _render_state["stop_event"] is not None:
                _render_state["stop_event"].set()
            if _render_state["thread"] is not None:
                _render_state["thread"].join(timeout=5.0)
            if _render_state["viewer"] is not None:
                try:
                    _render_state["viewer"].close()
                except Exception:
                    pass
                _render_state["viewer"] = None
            # Reset both 3D scene geometry and GUI control panel so no old
            # elements bleed into the new viewer instance.
            _viser_preview_server.scene.reset()
            import concurrent.futures as _cf

            try:
                with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                    _ex.submit(_viser_preview_server.gui.reset).result(timeout=5)
            except (RuntimeError, _cf.TimeoutError):
                # RuntimeError: viser raises this when a GuiTabHandle tries to
                # update its already-removed parent GuiTabGroupHandle during the
                # reset() traversal — the reset still completes successfully.
                # TimeoutError: gui.reset() can block indefinitely on a hung
                # viser websocket; abandon it after 5 s to unblock training.
                pass

        def _start_viewer(
            vec_env,
            stop_event,
            camera_state=None,
            poll=False,
            log_dir=None,
            record_name="",
        ) -> None:
            """Create and start a viewer thread.

            camera_state: dict with 'position' and 'look_at' arrays captured
            from the connected client before teardown, or None on first render.
            Uses _render_state['policy'] if an OnnxPolicy was loaded for this
            render, otherwise falls back to zero actions.
            poll: use PollingViserViewer (background checkpoint polling + modal
            notifications) instead of CameraRestoringViserViewer.
            log_dir: if provided (poll=True path), PollingViserViewer adds a
            "Training Metrics" panel that reads from the TFEvents file every 30s.
            record_name: prefix used for video filenames when the in-viewer
            recording button is used (viser_videos/{record_name}-{ts}.mp4).
            """
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            _action_mode = "sigmoid"
            try:
                _muscles = vec_env.unwrapped.action_manager.get_term("muscles")
                if hasattr(_muscles, "cfg") and hasattr(_muscles.cfg, "action_mode"):
                    _action_mode = _muscles.cfg.action_mode
            except Exception:
                pass
            inner = _render_state["policy"] or _ZeroActionInner(
                vec_env.num_actions, device, _action_mode
            )
            policy = _StoppablePolicy(inner, stop_event)
            viewer_cls = PollingViserViewer if poll else CameraRestoringViserViewer
            extra_kwargs = {"log_dir": log_dir} if poll and log_dir is not None else {}
            viewer = viewer_cls(
                vec_env,
                policy,
                viser_server=_viser_preview_server,
                post_setup_camera=camera_state,
                checkpoint_manager=_render_state.get("checkpoint_manager"),
                record_name=record_name,
                **extra_kwargs,
            )
            thread = threading.Thread(
                target=_run_viewer, args=(viewer, stop_event), daemon=True
            )
            _render_state["viewer"] = viewer
            _render_state["stop_event"] = stop_event
            _render_state["thread"] = thread
            thread.start()

        def render_training_result(log_dir=None) -> str:
            """Render the latest checkpoint from a local training log_dir.

            Designed to be called from the notebook after training completes,
            but also works during training as long as the render env already
            exists (guaranteed by prepare_for_training()).
            Falls back to run_state.log_dir when log_dir is not given.
            Returns the viser preview URL for embedding in an IFrame.
            """
            _log_dir = Path(log_dir or run_state.log_dir)
            if not _log_dir.exists():
                print(f"[render] log_dir not found: {_log_dir}")
                return _viser_preview_url

            # Prefer model_final.onnx; otherwise the highest-numbered checkpoint.
            # The runner saves intermediate checkpoints to log_dir/checkpoints/,
            # so scan both the root and that subdirectory.
            final_onnx = _log_dir / "model_final.onnx"
            scan_dirs = [d for d in [_log_dir, _log_dir / "checkpoints"] if d.is_dir()]
            onnx_files = sorted(
                (
                    f
                    for d in scan_dirs
                    for f in d.glob("*.onnx")
                    if is_onnx_checkpoint_name(f.name)
                ),
                key=lambda f: onnx_checkpoint_sort_key(f.name),
            )
            checkpoint = (
                final_onnx
                if final_onnx.exists()
                else (onnx_files[-1] if onnx_files else None)
            )
            if checkpoint is None:
                print(f"[render] No .onnx checkpoint found in {_log_dir}")
                return _viser_preview_url

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # cfg_overrides is authoritative here: run_training() always rebuilds it
            # from the current UI state (with loaded_policy_extra_overrides prepended)
            # before launch_training() is called, so it matches what was actually
            # trained. A stale snapshot taken at checkpoint-load time could preview a
            # different config than the one trained, e.g. if a checkpoint's config was
            # loaded and the UI was edited before starting training.
            config = load_config_interactive(run_state.cfg_overrides)
            new_key = _render_struct_key(config)

            _stop_current_viewer()

            needs_rebuild = (
                _render_state["struct_key"] != new_key or _render_state["env"] is None
            )
            if needs_rebuild and run_state.is_training:
                # Cannot create/destroy envs while training holds the CUDA allocator.
                if _render_state["env"] is None:
                    try:
                        gr.Info(
                            "Render environment not ready. "
                            "Call prepare_for_training() before starting training."
                        )
                    except Exception:
                        print(
                            "[render] Render env not ready — call prepare_for_training() first."
                        )
                    return _viser_preview_url
                # Struct changed during training: reuse existing env with current geoms.
                needs_rebuild = False

            if needs_rebuild:
                if _render_state["env"] is not None:
                    try:
                        _render_state["env"].close()
                    except Exception:
                        pass
                    _render_state["env"] = None
                    _render_state["vec_env"] = None

                register_mjlab_myointeract_task(config)
                env_cfg = load_env_cfg("myoInteractUniversal-v0")
                clip_actions = load_rl_cfg("myoInteractUniversal-v0").clip_actions
                env_cfg.scene.num_envs = 1
                env_cfg.viewer.distance = 2.5
                env_cfg.viewer.elevation = 45.0
                env_cfg.viewer.azimuth = 135.0
                env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
                vec_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
                _update_model_geoms(env, config)
                _render_state["env"] = env
                _render_state["vec_env"] = vec_env
                _render_state["struct_key"] = new_key
            else:
                _update_model_geoms(_render_state["env"], config)
                # max_duration isn't part of _render_struct_key (it doesn't change the
                # model/obs/action shape), so it never triggers a rebuild -- but
                # ManagerBasedRlEnv.max_episode_length is a live property reading
                # self.cfg.episode_length_s (mjlab/envs/manager_based_rl_env.py), so
                # writing it into the already-built env's cfg is enough to make a
                # changed "Maximum Episode Duration (s)" take effect without a rebuild.
                _render_state[
                    "env"
                ].cfg.episode_length_s = config.env.task_config.max_duration

            vec_env = _render_state["vec_env"]

            # Load policy via OnnxCheckpointingMjlabRunner — identical to play_notebook_viser.py.
            rl_cfg_local = load_rl_cfg("myoInteractUniversal-v0")
            runner = OnnxCheckpointingMjlabRunner(
                vec_env,
                asdict(rl_cfg_local),
                str(checkpoint.parent),
                device,
                task_id="myoInteractUniversal-v0",
            )
            try:
                runner.load_onnx(checkpoint)
            except RuntimeError as exc:
                if "size mismatch" in str(exc):
                    gr.Warning(
                        "The checkpoint is incompatible with the current observation "
                        "space settings (observation dimension mismatch). Please match "
                        "the settings to those used during training, or select a different checkpoint."
                    )
                    return _viser_preview_url
                raise
            policy = runner.get_inference_policy(device=device)
            print(f"[render] Loaded checkpoint: {checkpoint.name} from {_log_dir.name}")

            ckpt_dir = checkpoint.parent

            def _reload(path: str):
                runner.load_onnx(Path(path))
                return runner.get_inference_policy(device=device)

            def _fetch_ckpts() -> list:
                now = _time.time()
                entries = []
                for f in ckpt_dir.glob("*.onnx"):
                    if not is_onnx_checkpoint_name(f.name):
                        continue
                    entries.append(
                        (f.name, format_time_ago(int(now - f.stat().st_mtime)))
                    )
                entries.sort(key=lambda x: onnx_checkpoint_sort_key(x[0]))
                return entries

            ckpt_manager = CheckpointManager(
                current_name=checkpoint.name,
                fetch_available=_fetch_ckpts,
                load_checkpoint=lambda name: _reload(str(ckpt_dir / name)),
            )

            _render_state["policy"] = policy
            _render_state["checkpoint_manager"] = ckpt_manager

            stop_event = threading.Event()
            _start_viewer(
                vec_env,
                stop_event,
                poll=True,
                log_dir=str(_log_dir),
                record_name=Path(_log_dir).name,
            )

            return _viser_preview_url

        def render_environment(run_name, *args):
            target_init_seed = RLParameters.get_target_init_seed(args)
            next_seed = target_init_seed + 1
            sanity_check, ui_overrides = args_to_cfg_overrides(run_name, *args)
            if not sanity_check:
                gr.Warning("Invalid configuration.")
                return gr.skip()

            # Prepend non-UI overrides from the loaded policy (e.g. fatigue_reset_random)
            # before the UI overrides so the UI always wins for overlapping fields.
            # Dedup ensures no contradicting entries after the merge.
            cfg_overrides = _dedup_overrides(
                run_state.loaded_policy_extra_overrides + ui_overrides
            )
            config = load_config_interactive(cfg_overrides)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # Load checkpoint policy directly from the dropdown selections,
            # bypassing config so the values are always current at call time.
            # args[6] = select_checkpoint_run, args[7] = select_checkpoint_file
            # (RLParameters fields 6 and 7 within run_inputs).
            checkpoint_run = args[6]
            checkpoint_file = args[7]
            ckpt_path = checkpoint_path_from_run_filename(
                checkpoint_run, checkpoint_file
            )
            if (
                not ckpt_path or ckpt_path == "None"
            ) and run_state.loaded_checkpoint_path:
                ckpt_path = run_state.loaded_checkpoint_path
            new_key = _render_struct_key(config)
            structural_change = (
                _render_state["struct_key"] != new_key or _render_state["env"] is None
            )

            # Capture camera before teardown so we can restore it afterwards.
            camera_state = _capture_camera_state()
            _stop_current_viewer()

            if structural_change and run_state.is_training:
                # Cannot create/destroy envs while training holds the CUDA allocator.
                if _render_state["env"] is None:
                    gr.Warning(
                        "Render environment not ready. "
                        "Call prepare_for_training() before starting training."
                    )
                    return gr.skip()
                # Config changed during training: reuse existing env, update geoms only.
                gr.Warning(
                    "Some of the requested changes can only be applied after training. "
                    "The current rendering may not fully reflect the new configuration."
                )
                structural_change = False

            if structural_change:
                # Full restart: rebuild env from new registration.
                if _render_state["env"] is not None:
                    try:
                        _render_state["env"].close()
                    except Exception:
                        pass
                    _render_state["env"] = None
                    _render_state["vec_env"] = None

                register_mjlab_myointeract_task(config)
                env_cfg = load_env_cfg("myoInteractUniversal-v0")
                clip_actions = load_rl_cfg("myoInteractUniversal-v0").clip_actions
                env_cfg.scene.num_envs = 1
                # Place the camera closer to the hand workspace (only used when
                # no prior camera state exists, i.e. the very first render).
                env_cfg.viewer.distance = 2.5
                env_cfg.viewer.elevation = 45.0
                env_cfg.viewer.azimuth = 135.0
                env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
                vec_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
                # Set mj_model colors/sizes to the configured values so the viewer
                # bakes them correctly.  The spec builds the model with a random DR
                # sample; _update_model_geoms overwrites that with config averages.
                _update_model_geoms(env, config)
                _render_state["env"] = env
                _render_state["vec_env"] = vec_env
                _render_state["struct_key"] = new_key
            else:
                # Fast path: non-structural change (same model, same targets).
                # Update mj_model colors/sizes (what viser bakes at setup() time)
                # and event term params (what env.reset() samples for positions).
                # Restarting the viewer is enough to pick up color/size changes;
                # no env rebuild needed (~1 s vs ~30 s).
                _update_model_geoms(_render_state["env"], config)
                # max_duration isn't part of _render_struct_key (it doesn't change the
                # model/obs/action shape), so it never triggers a rebuild -- but
                # ManagerBasedRlEnv.max_episode_length is a live property reading
                # self.cfg.episode_length_s (mjlab/envs/manager_based_rl_env.py), so
                # writing it into the already-built env's cfg is enough to make a
                # changed "Maximum Episode Duration (s)" take effect without a rebuild.
                _render_state[
                    "env"
                ].cfg.episode_length_s = config.env.task_config.max_duration
                try:
                    _render_state["env"].reset()
                except Exception:
                    pass

            policy, ckpt_manager = _build_policy_and_ckpt_manager(
                checkpoint_run, checkpoint_file, device, run_state
            )
            _render_state["policy"] = policy
            _render_state["checkpoint_manager"] = ckpt_manager

            stop_event = threading.Event()
            _start_viewer(
                _render_state["vec_env"],
                stop_event,
                camera_state=camera_state,
                record_name=run_name,
            )

            iframe_html = (
                f'<iframe src="{_viser_preview_url}" '
                f'width="100%" height="520px" frameborder="0" '
                f'allow="cross-origin-isolated"></iframe>'
            )
            return (
                gr.update(value=next_seed),
                gr.update(visible=True),
                gr.update(value=iframe_html),
            )

        def update_dynamic_elements(num):
            """Show/hide dynamic rows based on the number input.

            When num > MAX_TARGETS only the first row is shown — all extra targets
            copy from target 1, so there is nothing to configure for them individually.
            """
            n = int(num) if num is not None else 0
            num_visible = 1 if n > MAX_TARGETS else max(0, min(n, MAX_TARGETS))
            updates = []
            for i in range(MAX_TARGETS):
                updates.append(gr.update(visible=i < num_visible))
            return updates

        def update_fatigue_obs_keys_interactive(fatigue_enabled, *fatigue_obs_keys):
            """Update interactability of fatigue observation keys based on fatigue_enabled"""
            updates = []
            for k in fatigue_obs_keys:
                if fatigue_enabled:
                    updates.append(gr.update(interactive=True))
                else:
                    updates.append(gr.update(value=False, interactive=False))
            return updates

        def update_omni_key_presets(vision_key, *omni_keys):
            """Update/Override default values of omni keys if vision_key is enabled"""
            updates = []
            for k in omni_keys:
                updates.append(gr.update(value=False) if vision_key else gr.update())
            return updates

        def update_num_timesteps(
            bm_model,
            num_elements,
            ctrl_dt,
            fatigue_enabled,
            *dwell_durations_and_radios,
        ):
            _num_targets_max = int(len(dwell_durations_and_radios) // 2)
            dwell_durations = dwell_durations_and_radios[:num_elements]
            radios = dwell_durations_and_radios[
                _num_targets_max : _num_targets_max + num_elements
            ]
            total_dwell_duration = sum(
                map(
                    lambda x, y: max(0, x - ctrl_dt) * (y == "Sphere"),
                    dwell_durations,
                    radios,
                )
            )

            target_value = (
                5_000_000
                + (
                    bm_model
                    in ("MoBL_Arms_Hand", "MoBL_Arms_Hand_v2.1", "MyoArm", "MyoArm_v2")
                )
                * 3_000_000  # number to be validated
                + (
                    bm_model
                    in (
                        "MyoArm_nohand",
                        "MyoArm",
                        "MyoArm_v2",
                        "MyoArm_reach",
                        "MyoArm_pointing",
                    )
                )
                * 10_000_000  # number to be validated
                + (fatigue_enabled) * 20_000_000  # number to be validated
                + max((num_elements - 1), 0) * 3_000_000
                + int(total_dwell_duration // 0.3) * 1_000_000
            )
            return gr.update(value=target_value)

        def toggle_interface_type(radio_value):
            """Show appropriate interface based on radio selection"""
            if radio_value == "Box":
                return gr.update(visible=True), gr.update(visible=False)
            else:  # Sphere
                return gr.update(visible=False), gr.update(visible=True)

        def update_pageview(pageview, *advanced_options):
            visible = pageview == "Advanced"
            return [gr.update(visible=visible) for opt in advanced_options]

        advanced_options_and_markdowns = [
            *bm_params,
            *task_params,
            *obs_keys,
            *fatigue_obs_keys,
            *vision_keys,
            *omni_keys,
            *rl_params,
            *md_rl_notes,
        ] + [md1, md22, md23, md3]
        pageview.change(
            fn=update_pageview,
            inputs=(pageview, *advanced_options_and_markdowns),
            outputs=advanced_options_and_markdowns,
        )
        # also call this method once at the very beginning
        demo.load(
            fn=update_pageview,
            inputs=(pageview, *advanced_options_and_markdowns),
            outputs=advanced_options_and_markdowns,
        )

        # Event handler for dynamic elements
        num_elements.change(
            update_dynamic_elements, inputs=num_elements, outputs=dynamic_rows
        )

        # Event handler for fatigue boolean to fatigue obs keys
        fatigue_enabled.change(
            update_fatigue_obs_keys_interactive,
            inputs=(fatigue_enabled, *fatigue_obs_keys),
            outputs=fatigue_obs_keys,
            preprocess=False,
        )

        # Event handler for vision checkbox to omni keys
        for vision_key in vision_keys:
            vision_key.change(
                update_omni_key_presets,
                inputs=(vision_key, *omni_keys),
                outputs=omni_keys,
                preprocess=False,
            )

        # Event handler for biomechanical model to suggested number of training steps
        bm_model.change(
            update_num_timesteps,
            inputs=(
                bm_model,
                num_elements,
                ctrl_dt,
                fatigue_enabled,
                *dwell_durations,
                *radios,
            ),
            outputs=num_timesteps,
            preprocess=False,
        )

        # Event handler for fatigue (enabled/disabled) to suggested number of training steps
        fatigue_enabled.change(
            update_num_timesteps,
            inputs=(
                bm_model,
                num_elements,
                ctrl_dt,
                fatigue_enabled,
                *dwell_durations,
                *radios,
            ),
            outputs=num_timesteps,
            preprocess=False,
        )

        # Event handler for number of targets to num of training steps
        num_elements.change(
            update_num_timesteps,
            inputs=(
                bm_model,
                num_elements,
                ctrl_dt,
                fatigue_enabled,
                *dwell_durations,
                *radios,
            ),
            outputs=num_timesteps,
            preprocess=False,
        )

        # Event handlers for each radio button to control interface type and num of training steps
        for i in range(MAX_TARGETS):
            radios[i].change(
                toggle_interface_type,
                inputs=radios[i],
                outputs=[box_rows[i], sphere_rows[i]],
            )
            radios[i].change(
                update_num_timesteps,
                inputs=(
                    bm_model,
                    num_elements,
                    ctrl_dt,
                    fatigue_enabled,
                    *dwell_durations,
                    *radios,
                ),
                outputs=num_timesteps,
                preprocess=False,
            )

        # Event handler for each dwell duration to num of training steps
        for i in range(MAX_TARGETS):
            dwell_durations[i].change(
                update_num_timesteps,
                inputs=(
                    bm_model,
                    num_elements,
                    ctrl_dt,
                    fatigue_enabled,
                    *dwell_durations,
                    *radios,
                ),
                outputs=num_timesteps,
                preprocess=False,
            )

        # Prepare inputs for run button.
        # rl_params has 7 items (indices 0-5 are RL hyper-params, index 6 is target_init_seed).
        # Inject checkpoint dropdowns at positions 6, 7, 8 so that RLParameters.num_elements=10
        # and the args[6]/args[7] reads in render_environment() remain correct.
        run_inputs = (
            list(rl_params[:6])
            + [select_checkpoint_run, select_checkpoint_file, show_training_ckpts]
            + [rl_params[6]]  # target_init_seed at position 9
            + [num_elements]
        )
        run_inputs.extend(radios)

        # Add all box components
        for i in range(MAX_TARGETS):
            for key in BoxParameters.fields():
                run_inputs.append(all_components["boxes"][i][key])

        # Add all sphere components
        for i in range(MAX_TARGETS):
            for key in SphereParameters.fields():
                run_inputs.append(all_components["spheres"][i][key])

        run_inputs.extend(bm_params)
        run_inputs.extend(task_params)
        run_inputs.extend(obs_keys)
        run_inputs.extend(fatigue_obs_keys)
        run_inputs.extend(vision_keys)
        run_inputs.extend(omni_keys)
        run_inputs.extend(reward_weights)

        # Wire checkpoint dropdown change handlers (moved from RLParameters.get_parameters).
        def _on_checkpoint_run_changed(run_val, show_ckpts):
            # Explicit dropdown selection always wins: clear the programmatically loaded
            # checkpoint so the None/None choice is not silently overridden by the fallback
            # in args_to_cfg_overrides.
            run_state.loaded_checkpoint_path = ""
            return update_checkpoint_numbers(run_val, show_ckpts)

        select_checkpoint_run.change(
            _on_checkpoint_run_changed,
            inputs=[select_checkpoint_run, show_training_ckpts],
            outputs=select_checkpoint_file,
        )
        show_training_ckpts.change(
            update_checkpoint_dirs,
            inputs=show_training_ckpts,
            outputs=select_checkpoint_run,
        )
        # Auto-populate dropdowns when a local directory path is typed in the textbox.
        load_policy_input.change(
            load_policy_input_changed,
            inputs=[load_policy_input],
            outputs=[select_checkpoint_run, select_checkpoint_file],
        )

        # ── Load / Resume callback ─────────────────────────────────────────
        # Positions of the checkpoint dropdowns inside run_inputs.
        _run_inp_idx = run_inputs.index(select_checkpoint_run)
        _file_inp_idx = run_inputs.index(select_checkpoint_file)

        def _inject_ckpt_dropdown_updates(
            updates: list, ckpt_path: "Path | None"
        ) -> list:
            """Overwrite select_checkpoint_run/file positions with resolved values."""
            if ckpt_path is None:
                return updates
            _ckpt_dir = str(ckpt_path.parent)
            try:
                _rel = os.path.relpath(_ckpt_dir, LOG_DIR)
            except ValueError:
                _rel = _ckpt_dir
            _files = _list_onnx_files_in_dir(_ckpt_dir)
            _run_choices = get_available_checkpoints(show_training_ckpts=True)
            if _rel not in _run_choices:
                _run_choices = (
                    [_rel] + [c for c in _run_choices if c != "None"] + ["None"]
                )
            updates[_run_inp_idx] = gr.update(choices=_run_choices, value=_rel)
            updates[_file_inp_idx] = gr.update(choices=_files, value=ckpt_path.name)
            return updates

        def load_policy_clicked(wandb_or_path, ckpt_run, ckpt_file, also_load_config):
            """Resolve a checkpoint from a W&B URL, local ONNX path, or local dropdown.

            When *also_load_config* is True the training config is fetched (from wandb
            or a sidecar JSON) and all UI fields are populated to match.
            Returns gr.update() tuples for every component in run_inputs plus a status string.
            """
            wandb_or_path = (wandb_or_path or "").strip()
            ckpt_path = None
            config_dict = None
            status_lines = []

            try:
                if wandb_or_path:
                    # ── W&B URL or run path ───────────────────────────────
                    if wandb_or_path.startswith("https://wandb.ai/"):
                        # Full URL: https://wandb.ai/entity/project/runs/RUN_ID
                        run_path = wandb_or_path[len("https://wandb.ai/") :]
                        run_path = run_path.replace("/runs/", "/")
                    elif not wandb_or_path.startswith("/") and "/" in wandb_or_path:
                        # Already entity/project/run_id format
                        run_path = wandb_or_path
                    elif (
                        not wandb_or_path.startswith("/")
                        and os.sep not in wandb_or_path
                        and not _resolve_local_run_dir(wandb_or_path)[1]
                    ):
                        # Bare run ID — expand with defaults
                        run_path = f"{WANDB_DEFAULT_ENTITY}/{WANDB_DEFAULT_PROJECT}/{wandb_or_path}"
                    else:
                        run_path = None

                    if run_path:
                        import wandb as _wandb

                        status_lines.append(
                            f"Downloading checkpoints from W&B run {run_path} …"
                        )
                        api = _wandb.Api()
                        wandb_run = api.run(run_path)
                        paths, cached_list = download_all_wandb_onnx_checkpoints(
                            Path(LOG_DIR), run_path
                        )
                        # Respect the user's dropdown selection; fall back to model_final.
                        if ckpt_file and ckpt_file != "None":
                            _candidate = paths[0].parent / ckpt_file
                            ckpt_path = _candidate if _candidate.exists() else paths[-1]
                        else:
                            ckpt_path = paths[
                                -1
                            ]  # sorted: model_final last, else highest step
                        new_count = sum(1 for c in cached_list if not c)
                        status_lines[-1] += (
                            f" {new_count} new / {len(paths)} total."
                            if new_count
                            else f" {len(paths)} cached."
                        )
                        if also_load_config:
                            try:
                                # Prefer the explicitly saved myointeract_config.json
                                # (downloaded alongside ONNX files above) since it
                                # captures all Python overrides applied at training time.
                                _downloaded_cfg = checkpoint_config_path(
                                    ckpt_path.parent
                                )
                                if _downloaded_cfg.exists():
                                    config_dict = json.loads(
                                        _downloaded_cfg.read_text()
                                    )
                                    status_lines.append(
                                        "Loaded training config from myointeract_config.json."
                                    )
                                else:
                                    # Fallback: extract from wandb base_config YAML string.
                                    hydra_cfg = extract_wandb_hydra_config(
                                        dict(wandb_run.config)
                                    )
                                    if hydra_cfg is not None:
                                        config_dict = hydra_cfg
                                        status_lines.append(
                                            "Fetched training config from W&B (base_config)."
                                        )
                                    else:
                                        config_dict = dict(wandb_run.config)
                                        status_lines.append(
                                            "W&B config has no base_config; using raw run config "
                                            "(some fields may fall back to defaults)."
                                        )
                            except Exception as exc:
                                status_lines.append(
                                    f"Could not fetch config from W&B: {exc}"
                                )
                    else:
                        # Local path — could be a directory or a direct ONNX file.
                        _input_path = Path(wandb_or_path)
                        if _input_path.is_dir():
                            # Resolve to the selected (or best) checkpoint in the directory.
                            _files = _list_onnx_files_in_dir(str(_input_path))
                            if _files == ["None"]:
                                return tuple(gr.update() for _ in run_inputs) + (
                                    f"No ONNX checkpoints found in {_input_path}",
                                )
                            if (
                                ckpt_file
                                and ckpt_file != "None"
                                and (_input_path / ckpt_file).exists()
                            ):
                                ckpt_path = _input_path / ckpt_file
                            else:
                                ckpt_path = _input_path / _files[0]
                            status_lines.append(
                                f"Resolved to {ckpt_path.name} from directory."
                            )
                        else:
                            # Try bare run-name → resolve via LOG_DIR / CHECKPOINT_PATH.
                            _, _resolved = _resolve_local_run_dir(wandb_or_path)
                            if _resolved:
                                _files = _list_onnx_files_in_dir(_resolved)
                                if (
                                    ckpt_file
                                    and ckpt_file != "None"
                                    and (Path(_resolved) / ckpt_file).exists()
                                ):
                                    ckpt_path = Path(_resolved) / ckpt_file
                                else:
                                    ckpt_path = Path(_resolved) / (
                                        _files[0] if _files != ["None"] else ""
                                    )
                            else:
                                ckpt_path = _input_path
                        if not ckpt_path.exists():
                            return tuple(gr.update() for _ in run_inputs) + (
                                f"File not found: {ckpt_path}",
                            )
                        if also_load_config:
                            sidecar = ckpt_path.with_suffix(".json")
                            myointeract_cfg = checkpoint_config_path(ckpt_path.parent)
                            if sidecar.exists():
                                config_dict = json.loads(sidecar.read_text())
                                status_lines.append(
                                    f"Loaded config from {sidecar.name}."
                                )
                            elif myointeract_cfg.exists():
                                config_dict = json.loads(myointeract_cfg.read_text())
                                status_lines.append(
                                    "Loaded config from myointeract_config.json."
                                )
                            else:
                                status_lines.append(
                                    "No config JSON found — only checkpoint loaded."
                                )
                elif ckpt_run and ckpt_run != "None":
                    # Local dropdown selection
                    raw = checkpoint_path_from_run_filename(ckpt_run, ckpt_file)
                    if raw and raw != "None":
                        ckpt_path = Path(raw)
                    if also_load_config and ckpt_path:
                        sidecar = ckpt_path.with_suffix(".json")
                        myointeract_cfg = checkpoint_config_path(ckpt_path.parent)
                        if sidecar.exists():
                            config_dict = json.loads(sidecar.read_text())
                            status_lines.append(f"Loaded config from {sidecar.name}.")
                        elif myointeract_cfg.exists():
                            config_dict = json.loads(myointeract_cfg.read_text())
                            status_lines.append(
                                "Loaded config from myointeract_config.json."
                            )
                        else:
                            status_lines.append(
                                "No config JSON found — only checkpoint loaded."
                            )
                else:
                    return tuple(gr.update() for _ in run_inputs) + (
                        "Nothing to load — provide a URL, path, or select a run.",
                    )
            except Exception as exc:
                return tuple(gr.update() for _ in run_inputs) + (f"Error: {exc}",)

            if ckpt_path is None:
                return tuple(gr.update() for _ in run_inputs) + (
                    "\n".join(status_lines) or "No checkpoint resolved.",
                )

            run_state.loaded_checkpoint_path = str(ckpt_path)
            status_lines.insert(0, f"Checkpoint: {ckpt_path.name}")

            if also_load_config and config_dict:
                try:
                    labelled = training_config_to_labelled_dict(config_dict)
                    flat = config_saver.from_labelled_dict(labelled)
                    _extra = _dedup_overrides(extra_hydra_overrides(config_dict))
                    # Store non-UI overrides separately so render_environment() and
                    # run_training() can prepend them before the live UI overrides.
                    run_state.loaded_policy_extra_overrides = _extra
                    status_lines.append("UI populated from training config.")
                    _updates = list(gr.update(value=v) for v in flat)
                    _inject_ckpt_dropdown_updates(_updates, ckpt_path)
                    return tuple(_updates) + ("\n".join(status_lines),)
                except Exception as exc:
                    status_lines.append(f"Config load error: {exc}")

            _updates = [gr.update() for _ in run_inputs]
            _inject_ckpt_dropdown_updates(_updates, ckpt_path)
            return tuple(_updates) + ("\n".join(status_lines),)

        load_policy_button.click(
            load_policy_clicked,
            inputs=[
                load_policy_input,
                select_checkpoint_run,
                select_checkpoint_file,
                load_config_too,
            ],
            outputs=list(run_inputs) + [load_policy_status],
        )

        # Run button event
        run_or_save_inputs = [config_name_input] + run_inputs
        run_button.click(run_training, inputs=run_or_save_inputs, outputs=output_text)

        config_saver.run_inputs = run_inputs
        config_saver.run_state = run_state

        pre_saved_configs.change(
            config_saver.load_config, inputs=pre_saved_configs, outputs=run_inputs
        )

        # Event handler for suggested config name (also load once at startup)
        def _update_suggest_config_name(config_name):
            return gr.update(value=config_name) if config_name else gr.update()

        pre_saved_configs.change(
            _update_suggest_config_name,
            inputs=pre_saved_configs,
            outputs=config_name_input,
        )
        demo.load(
            _update_suggest_config_name,
            inputs=pre_saved_configs,
            outputs=config_name_input,
        )

        save_config_button.click(
            config_saver.config_save_clicked,
            inputs=run_or_save_inputs,
            outputs=pre_saved_configs,
        )

        if use_legacy_rendering:
            render_button.click(
                _render_environment_legacy,
                inputs=run_or_save_inputs,
                outputs=[target_init_seed, env_view_row, env_view_1, env_view_2],
            )
            # Load saved config then render on startup.
            demo.load(
                config_saver.load_config, inputs=pre_saved_configs, outputs=run_inputs
            )
        else:
            render_button.click(
                render_environment,
                inputs=run_or_save_inputs,
                outputs=[target_init_seed, env_view_row, env_view],
            )

            def _load_training_handler():
                if not run_state.log_dir:
                    gr.Warning("No training run found. Run the training cell first.")
                    return gr.skip(), gr.skip()
                url = render_training_result()
                iframe_html = (
                    f'<iframe src="{url}" width="100%" height="520px" '
                    f'frameborder="0" allow="cross-origin-isolated"></iframe>'
                )
                return gr.update(visible=True), gr.update(value=iframe_html)

            load_training_button.click(
                _load_training_handler,
                inputs=[],
                outputs=[env_view_row, env_view],
            )

            # Load saved config first, then immediately render with the updated values.
            demo.load(
                config_saver.load_config,
                inputs=pre_saved_configs,
                outputs=run_inputs,
            ).then(
                render_environment,
                inputs=run_or_save_inputs,
                outputs=[target_init_seed, env_view_row, env_view],
            )

    # Expose helpers so the notebook can trigger rendering after training.
    run_state.render_training_result = render_training_result
    run_state._viser_preview_url = _viser_preview_url

    def _prepare_for_training() -> None:
        """Prepare the gradio side for training.

        - Stops the viser viewer thread (no CUDA allocation/deallocation during training).
        - Ensures the render env is initialised now while CUDA is quiet, so the GUI
          buttons (Update Preview, Show Training Policy) remain functional at any point
          during training without needing to create a new env under load.
        - Calls torch.cuda.empty_cache() to defragment the allocator before training starts.

        IMPORTANT: the render env is intentionally kept alive so that viewer restarts
        during training only update the policy/checkpoint — they never touch env creation.
        """
        _stop_current_viewer()

        # Ensure the render env exists and matches the current cfg_overrides before training
        # starts (while CUDA is quiet).  If cfg changed since the last render, close the stale
        # env and rebuild so struct_key is consistent throughout training.
        _pre_overrides = run_state.cfg_overrides
        if _pre_overrides:
            try:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                config = load_config_interactive(_pre_overrides)
                new_key = _render_struct_key(config)
                if (
                    _render_state["env"] is not None
                    and _render_state["struct_key"] != new_key
                ):
                    try:
                        _render_state["env"].close()
                    except Exception:
                        pass
                    _render_state["env"] = None
                    _render_state["vec_env"] = None
                    print(
                        "[prepare] Config changed — rebuilding render env for training."
                    )
                if _render_state["env"] is None:
                    register_mjlab_myointeract_task(config)
                    env_cfg = load_env_cfg("myoInteractUniversal-v0")
                    rl_cfg_local = load_rl_cfg("myoInteractUniversal-v0")
                    env_cfg.scene.num_envs = 1
                    env_cfg.viewer.distance = 2.5
                    env_cfg.viewer.elevation = 45.0
                    env_cfg.viewer.azimuth = 135.0
                    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
                    vec_env = RslRlVecEnvWrapper(
                        env, clip_actions=rl_cfg_local.clip_actions
                    )
                    _update_model_geoms(env, config)
                    _render_state["env"] = env
                    _render_state["vec_env"] = vec_env
                    _render_state["struct_key"] = new_key
                    print("[prepare] Render env initialised for training.")
                else:
                    print("[prepare] Render env already up-to-date for training.")
            except Exception as exc:
                print(f"[prepare] Could not pre-create render env: {exc}")

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    run_state.prepare_for_training = _prepare_for_training

    return demo


def launch_training(state):
    # Stop any running viewer and release the render env so the training process
    # gets a clean, fully-defragmented CUDA allocator.  This is what prevents the
    # RuntimeError on first launch that previously required a retry.
    if hasattr(state, "prepare_for_training"):
        state.prepare_for_training()

    config = load_config_interactive(state.cfg_overrides)
    register_mjlab_myointeract_task(config)

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print(
            "[WARNING] CUDA is not available. Running on CPU may be very slow for training."
        )

    env_cfg = load_env_cfg("myoInteractUniversal-v0")
    rl_cfg = load_rl_cfg("myoInteractUniversal-v0")

    resume_path = None
    if (
        config.rl.load_checkpoint_path is not None
        and config.rl.load_checkpoint_path != "None"
    ):
        # OnnxCheckpointingMjlabRunner saves .onnx; translate the default
        # 'model_.*.pt' regex (and any explicit .pt name) before scanning.
        resume_path = config.rl.load_checkpoint_path.replace(".pt", ".onnx")
        print(f"[INFO] Resuming from checkpoint: {resume_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(LOG_DIR) / f"{timestamp}_{state.wandb_run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    state.log_dir = str(log_dir)

    # Save the fully resolved myointeract config (all overrides applied) immediately
    # so it is available for local checkpoint loading.  The notebook calls
    # wandb.init() before launch_training(), so wandb.run is already active
    # here and we can upload before training starts (not after).
    with open(log_dir / "myointeract_config.json", "w") as _cfg_f:
        json.dump(config.to_dict(), _cfg_f, indent=2)

    # Also save the raw Hydra override strings that produced `config` -- the UI has
    # no single source JSON file the way the training scripts does (config is
    # composed from UI widget state via state.cfg_overrides), so this list IS the
    # original/pristine specification of the run, before Hydra resolution.
    with open(log_dir / "original_hydra_config.json", "w") as _overrides_f:
        json.dump(state.cfg_overrides, _overrides_f, indent=2)

    try:
        import wandb as _wandb

        if _wandb.run is not None:
            _wandb.save(
                str(log_dir / "myointeract_config.json"),
                base_path=str(log_dir),
                policy="now",
            )
    except Exception as _e:
        print(f"[INFO] Could not upload myointeract_config.json to wandb: {_e}")
    try:
        import wandb as _wandb

        if _wandb.run is not None:
            _wandb.save(
                str(log_dir / "original_hydra_config.json"),
                base_path=str(log_dir),
                policy="now",
            )
    except Exception as _e:
        print(f"[INFO] Could not upload original_hydra_config.json to wandb: {_e}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = OnnxCheckpointingMjlabRunner(
        vec_env, asdict(rl_cfg), str(log_dir), device, task_id="myoInteractUniversal-v0"
    )
    state.is_training = True
    try:
        if resume_path is not None:
            runner.load(str(resume_path))  # delegates to load_onnx() for .onnx files
        runner.learn(
            num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True
        )
        runner.save(str(log_dir / "model_final.pt"))
    finally:
        state.is_training = False
        env.close()


def main() -> None:
    """Console entry point ``myo_play``: launch the interactive UI."""
    demo = get_ui("myoInteract")
    demo.launch(share=True)


if __name__ == "__main__":
    main()
