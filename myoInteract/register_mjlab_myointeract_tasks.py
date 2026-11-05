# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Register MyoSuite tasks with mjlab's task registry so make_env(..., backend="mjlab") works.

When mjlab loads this package via the mjlab.tasks entry point, this module is not
auto-imported; the entry point targets myosuite.envs.myo.mjlab (the parent __init__.py).
We call register_mjlab_tasks() from there so that env ids like myoElbowPose1D6MFixed-v0
appear in list_tasks() and can be created via load_env_cfg + ManagerBasedRlEnv.
"""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv
from mjlab.envs.mdp import action_rate_l2 as mdp_action_rate_l2
from mjlab.envs.mdp import Entity, dataclass
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.envs.mdp import dr, events as event_fns
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.managers import EventTermCfg, ManagerTermBase, MetricsTermCfg, RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.sensor import CameraSensorCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task, _REGISTRY as _MJLAB_REGISTRY
import mujoco
import numpy as np
import torch

from myoInteract.body_models import (
    _MYOARM_NJMAX,
    _load_myoarm_r_based_spec,
    _myoarm_r_variant_for,
    base_spec,  # noqa: F401 - preserve the existing public import path
    compile_base_model,
)

from myoInteract.myointeract_utils import (
    add_button_to_spec,
    restrict_button_presses,
    add_sphere_to_spec,
    add_torso_contact_sensor_to_spec,
    TORSO_CONTACT_SENSOR_NAMES,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoActionCfg
from myosuite.terms.base_action import MotorNoiseCfg
from myosuite.integrations.musclemimic.mjlab_policy_runner import (
    OnnxCheckpointingMjlabRunner,
)
from myoInteract.myointeract_configs import (
    Config,
    BaseEnvConfig,
)


_UNIVERSAL_ENTITY_NAME = "universal_robot"
_UNIVERSAL_ENTITY_CFG = SceneEntityCfg(_UNIVERSAL_ENTITY_NAME)
_OBS_PROPRIO_ENTITY_CFG = SceneEntityCfg(
    _UNIVERSAL_ENTITY_NAME, site_names=["fingertip"]
)

MAX_TRIAL_TIME = 4.0  # seconds


def _muscles_fatigue(env: Any) -> Any:
    """Return the TorchFatigueState from the 'muscles' action term, or None.

    Use this in metrics and obs terms instead of accessing non-existent
    entity.data fields (e.g. ``tendon_muscle_fatigue``).  The fatigue object
    lives on the action term instance and exposes ``.MF``, ``.MA``, ``.MR``
    tensors of shape ``(num_envs, n_muscles)``.
    """
    return env.action_manager.get_term("muscles")._fatigue


_MOBL_ARMS_MUSCLE_NAMES = (
    "DELT1",
    "DELT2",
    "DELT3",
    "SUPSP",
    "INFSP",
    "SUBSC",
    "TMIN",
    "TMAJ",
    "PECM1",
    "PECM2",
    "PECM3",
    "LAT1",
    "LAT2",
    "LAT3",
    "CORB",
    "TRIlong",
    "TRIlat",
    "TRImed",
    "ANC",
    "SUP",
    "BIClong",
    "BICshort",
    "BRA",
    "BRD",
    "PT",
    "PQ",
)


def _fatigue_reset_vec(config) -> tuple[float, ...] | None:
    vec = config.env.muscle_config.get("fatigue_reset_vec")
    return None if vec is None else tuple(float(v) for v in vec)


def motor_noise_from_params(noise_params) -> MotorNoiseCfg:
    """MyoSuite ``MotorNoiseCfg`` from the config's ``noise_params`` (a noise type of "white" switches a term on)."""
    return MotorNoiseCfg(
        signal_dependent_std=noise_params.sigdepnoise_level
        if noise_params.sigdepnoise_type == "white"
        else 0.0,
        constant_std=noise_params.constantnoise_level
        if noise_params.constantnoise_type == "white"
        else 0.0,
    )


def _reset_joints_range_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: tuple,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset qpos uniformly within soft joint limits; reset qvel uniformly in velocity_range."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    soft_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]  # (B, nj, 2)

    lo_pos = soft_limits[..., 0]
    hi_pos = soft_limits[..., 1]
    joint_pos = lo_pos + torch.rand_like(lo_pos) * (hi_pos - lo_pos)

    lo_vel, hi_vel = velocity_range
    joint_vel = lo_vel + torch.rand_like(joint_pos) * (hi_vel - lo_vel)

    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids
    )


# Fixed "ready to interact" pose for the mobl_arms_index v2.1/v2.2 17-joint arm chain
# (all 17 joints, including the coupling/"phantom" shoulder-rhythm joints -- NOT just
# _independent_joint_names). Mirrors the "initpose001" MJCF <key> in
# mobl_arms_index_v2.1/v2.2_universal_myointeract.xml; keep both in sync. The mjwarp reset
# path mjlab uses does not read MJCF keyframes, so reset_type="initpose001" below
# applies this constant directly rather than looking up the keyframe at runtime.
_INITPOSE001_QPOS = (
    -6.92e-5,
    -0.293,
    0.124,
    -0.124,
    0.293,
    -0.0593,
    0.479,
    0.215,
    -0.215,
    -0.479,
    0.0593,
    0.373,
    1.21,
    -0.373,
    0.134,
    2.16,
    -1.04,
)
# Initial muscle activation for reset_type="initpose001": zero everywhere except
# the rotator-cuff stabilizers (SUPSP=3, INFSP=4, SUBSC=5) and elbow flexors
# (BIClong=20, BICshort=21, BRA=22, BRD=23), matching mobl_arms_index v2.1/v2.2's
# 26-muscle actuator definition order (DELT1/2/3, SUPSP, INFSP, SUBSC, TMIN, TMAJ,
# PECM1/2/3, LAT1/2/3, CORB, TRIlong/lat/med, ANC, SUP, BIClong, BICshort, BRA, BRD, PT, PQ).
_INITPOSE001_ACT = tuple(
    {3: 0.875, 4: 0.30, 5: 0.75, 20: 0.25, 21: 0.25, 22: 0.25, 23: 0.75}.get(i, 0.0)
    for i in range(26)
)


def _validate_initpose001_model(asset, n_act: int) -> None:
    n_joints = asset.data.joint_pos.shape[1]
    if n_joints != len(_INITPOSE001_QPOS):
        raise ValueError(
            f"reset_type='initpose001*' expects a {len(_INITPOSE001_QPOS)}-joint "
            f"model (mobl_arms_index v2.1/v2.2), got {n_joints} joints."
        )
    if n_act != len(_INITPOSE001_ACT):
        raise ValueError(
            f"reset_type='initpose001*' expects a {len(_INITPOSE001_ACT)}-muscle "
            f"model (mobl_arms_index v2.1/v2.2), got {n_act} actuators."
        )


def _reset_joints_initpose001(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset qpos to _INITPOSE001_QPOS and act to _INITPOSE001_ACT.

    Unlike the other reset_joints branches, asset_cfg here must be UNFILTERED (all
    joints, not just _independent_joint_names): the fixed pose specifies every joint
    in the chain, including the coupling/"phantom" shoulder-rhythm joints that the
    other reset_type branches leave untouched.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    _validate_initpose001_model(asset, asset.data.data.act.shape[1])

    n = len(env_ids)
    joint_pos = (
        torch.tensor(_INITPOSE001_QPOS, device=env.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(n, -1)
    )
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids, joint_ids=asset_cfg.joint_ids
    )

    act = (
        torch.tensor(_INITPOSE001_ACT, device=env.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(n, -1)
    )
    asset.data.data.act[env_ids] = act


def _reset_joints_initpose001_epsilon_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    qpos_epsilon: float = 0.1,
    act_epsilon: float = 0.1,
) -> None:
    """Like _reset_joints_initpose001, but sampling qpos/act uniformly in an
    epsilon-ball around _INITPOSE001_QPOS/_ACT instead of the exact fixed values --
    mirrors "epsilon_uniform"'s own +/-0.1 semantics, just centered on the
    initpose001 pose instead of default_joint_pos/zero-act. qpos is clamped to the
    entity's soft joint limits (matching epsilon_uniform's own clamp behavior); act
    is clamped to the valid [0, 1] muscle-activation range.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    _validate_initpose001_model(asset, asset.data.data.act.shape[1])

    n = len(env_ids)
    base_qpos = (
        torch.tensor(_INITPOSE001_QPOS, device=env.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(n, -1)
    )
    qpos_noise = (torch.rand_like(base_qpos) * 2.0 - 1.0) * qpos_epsilon
    soft_limits = asset.data.soft_joint_pos_limits[env_ids][
        :, asset_cfg.joint_ids
    ]  # (n, nj, 2)
    joint_pos = torch.clamp(
        base_qpos + qpos_noise, min=soft_limits[..., 0], max=soft_limits[..., 1]
    )
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids, joint_ids=asset_cfg.joint_ids
    )

    base_act = (
        torch.tensor(_INITPOSE001_ACT, device=env.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(n, -1)
    )
    act_noise = (torch.rand_like(base_act) * 2.0 - 1.0) * act_epsilon
    asset.data.data.act[env_ids] = torch.clamp(base_act + act_noise, min=0.0, max=1.0)


class VisitedSuccessBuffer:
    """Fixed-size circular buffer of qpos snapshots visited during *successful*
    episodes only (reset_type="visited_success").

    Unlike PostureBuffer (which captures unconditionally on a
    timer), rows are staged per-episode by VisitedSuccessCapture and only
    committed here -- by VisitedSuccessReset -- once the owning episode's
    episode_success termination has fired; episodes that time out or never
    reach the target are discarded, so the buffer only ever contains qpos
    states genuinely visited on the way to success.

    Each row also stores `relative_time` (the row's capture time as a fraction
    of task_config.max_duration, i.e. how far into its episode it occurred).
    `muscle_names` is stored once as buffer-level metadata (not per-row) so
    downstream consumers can sanity-check the buffer against the actuator set
    it was collected under.

    Seeded at construction with a single row equal to the fixed initpose001
    qpos (relative_time=0.0) so `sample` never returns empty before the first
    successful episode has been recorded.
    """

    def __init__(
        self,
        nq: int,
        buffer_size: int,
        muscle_names: tuple[str, ...] | list[str],
        init_qpos: torch.Tensor,
        device: str = "cpu",
        sampling: str = "uniform",
    ) -> None:
        if buffer_size < 1:
            raise ValueError("Visited-success buffer capacity must be positive.")
        self._qpos = torch.zeros(buffer_size, nq, device=device)
        self._relative_time = torch.zeros(buffer_size, device=device)
        self._ptr = 0
        self._full = False
        self.sampling = sampling
        self.muscle_names: list[str] = list(muscle_names)

        self._qpos[0] = init_qpos.to(device=device, dtype=torch.float32)
        self._relative_time[0] = 0.0
        self._ptr = 1 % buffer_size
        self._full = buffer_size == 1

    def add(self, qpos: torch.Tensor, relative_time: torch.Tensor) -> None:
        """Append `qpos` rows (n, nq) with matching `relative_time` rows (n,)."""
        n = qpos.shape[0]
        if n == 0:
            return
        capacity = self._qpos.shape[0]
        # Keep only the newest capacity rows. Duplicate destinations in advanced
        # indexing have undefined write order, including on CUDA.
        skip = max(0, n - capacity)
        idxs = (
            torch.arange(self._ptr + skip, self._ptr + n, device=qpos.device) % capacity
        )
        self._qpos[idxs] = qpos[skip:]
        self._relative_time[idxs] = relative_time[skip:]
        if self._ptr + n >= capacity and not self._full:
            self._full = True
        self._ptr = (self._ptr + n) % capacity

    @property
    def size(self) -> int:
        """Number of valid snapshots currently stored."""
        return self._qpos.shape[0] if self._full else self._ptr

    def sample(self, n: int) -> torch.Tensor | None:
        """Sample n rows -> qpos (n, nq), or None if `n == 0`.

        Never returns None due to an empty buffer: slot 0 is always the
        initpose001 seed row (see class docstring).
        """
        if self.size == 0 or n == 0:
            return None
        if self.sampling == "uniform":
            idx = torch.randint(0, self.size, (n,), device=self._qpos.device)
        else:
            raise ValueError(
                f"Unknown VisitedSuccessBuffer sampling strategy: {self.sampling!r}"
            )
        return self._qpos[idx]


class VisitedSuccessCapture(ManagerTermBase):
    """Interval event that stages temporally-equidistant qpos snapshots over the
    course of the CURRENT episode, for reset_type="visited_success".

    Registered with ``mode="interval"`` and (deliberately) ``is_global_time``
    left at its default ``False``: each env gets its own timer that restarts at
    that env's own episode reset (see EventManager.reset), so
    ``capture_interval_s = max_duration / n_snapshots_per_episode`` ticks land
    at evenly-spaced points within THIS episode regardless of when it started.
    An episode that ends early (e.g. an early success) simply accumulates fewer
    than `n_snapshots_per_episode` staged rows -- VisitedSuccessReset commits
    however many were staged, not a fixed count.

    Staged rows live on the scene entity (`asset.visited_success_staging_*`)
    rather than on `self` so :class:`VisitedSuccessReset` can read/clear them
    without a direct reference. Also creates the shared `asset.visited_success_buffer`
    here (once), lazily.
    """

    def __init__(self, cfg, env) -> None:
        self._env = env
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        self._asset = asset
        p = cfg.params
        n_snapshots = p["n_snapshots_per_episode"]
        self._n_snapshots = n_snapshots
        self._max_duration = p["max_duration"]
        device = env.device

        nq = asset.data.joint_pos.shape[1]
        _validate_initpose001_model(asset, asset.data.data.act.shape[1])
        asset.visited_success_buffer = VisitedSuccessBuffer(
            nq=nq,
            buffer_size=p["buffer_size"],
            muscle_names=p["muscle_names"],
            init_qpos=torch.tensor(_INITPOSE001_QPOS, dtype=torch.float32),
            device=str(device),
            sampling=p["sampling"],
        )
        asset.visited_success_staging_qpos = torch.zeros(
            env.num_envs, n_snapshots, nq, device=device
        )
        asset.visited_success_staging_rel_time = torch.zeros(
            env.num_envs, n_snapshots, device=device
        )
        asset.visited_success_staging_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=device
        )

    def __call__(self, env, env_ids, **_) -> None:
        asset = self._asset
        count = asset.visited_success_staging_count[env_ids]
        slot = count.clamp(max=self._n_snapshots - 1)
        asset.visited_success_staging_qpos[env_ids, slot] = asset.data.joint_pos[
            env_ids
        ]
        elapsed_s = env.episode_length_buf[env_ids].to(torch.float32) * env.step_dt
        asset.visited_success_staging_rel_time[env_ids, slot] = (
            elapsed_s / self._max_duration
        )
        asset.visited_success_staging_count[env_ids] = (count + 1).clamp(
            max=self._n_snapshots
        )


class VisitedSuccessReset(ManagerTermBase):
    """Reset-mode event implementing reset_type="visited_success":

    1. Commits the ending episode's staged temporally-equidistant qpos rows
       (from :class:`VisitedSuccessCapture`) into the shared
       `asset.visited_success_buffer` -- but ONLY for envs whose
       `episode_success` termination just fired this step. Envs that timed out
       or otherwise failed have their staged rows discarded: the buffer only
       ever contains qpos states genuinely visited on the way to a successful
       episode. Safe to read `episode_success` here: this event runs (from
       `ManagerBasedRlEnv._reset_idx`) after `sim.reset()` but before
       `termination_manager.reset()`, so the just-ended episode's termination
       result is still live.
    2. Resets each env's staging counter for the upcoming episode.
    3. Draws the new episode's initial qpos by sampling uniformly from the
       buffer (never empty -- seeded with the initpose001 pose, see
       VisitedSuccessBuffer) -- or, if `visited_success_buffer_enabled=False`
       (no VisitedSuccessCapture registered, so `asset.visited_success_buffer`
       does not exist), always falls back to the fixed initpose001 qpos.

    Same model restriction as _reset_joints_initpose001, and uses the same
    UNFILTERED asset_cfg for the same reason: the buffer's rows (and the
    initpose001 fallback) span every joint in the chain, not just
    _independent_joint_names.
    """

    def __init__(self, cfg, env) -> None:
        self._env = env
        self._asset_cfg = cfg.params.get(
            "asset_cfg", SceneEntityCfg(_UNIVERSAL_ENTITY_NAME)
        )

    def __call__(self, env, env_ids, **_) -> None:
        asset = env.scene[self._asset_cfg.name]
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(dtype=torch.long)
        n = len(env_ids)
        if n == 0:
            return

        _validate_initpose001_model(asset, asset.data.data.act.shape[1])
        buf: VisitedSuccessBuffer | None = getattr(
            asset, "visited_success_buffer", None
        )

        if buf is not None:
            success = env.termination_manager.get_term("episode_success")[
                env_ids
            ].bool()
            counts = asset.visited_success_staging_count[env_ids]
            slots = torch.arange(
                asset.visited_success_staging_qpos.shape[1], device=env.device
            )
            # Boolean indexing preserves env-major, then snapshot order, matching
            # the former per-environment loop without one GPU scalar read per env.
            staged = success[:, None] & (slots[None, :] < counts[:, None])
            buf.add(
                asset.visited_success_staging_qpos[env_ids][staged],
                asset.visited_success_staging_rel_time[env_ids][staged],
            )
            asset.visited_success_staging_count[env_ids] = 0
            sampled = buf.sample(n)
        else:
            sampled = None

        if sampled is None:
            qpos = (
                torch.tensor(_INITPOSE001_QPOS, device=env.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(n, -1)
                .clone()
            )
        else:
            qpos = sampled
        qvel = torch.zeros_like(qpos)
        asset.write_joint_state_to_sim(
            qpos, qvel, env_ids=env_ids, joint_ids=self._asset_cfg.joint_ids
        )


_muscle_names_cache: dict[str, tuple[str, ...]] = {}


def _get_muscle_names(config) -> tuple[str, ...]:
    """Return the actuator names for the model in config.env.model_path.

    Results are cached by path so the model is compiled at most once per path.
    """
    key = config.env.model_path
    if key not in _muscle_names_cache:
        variant = _myoarm_r_variant_for(key)
        if variant is not None:
            m = _load_myoarm_r_based_spec(variant).compile()
        else:
            xml_file = _get_myoInteract_root() / key
            m = mujoco.MjModel.from_xml_path(str(xml_file))
        _muscle_names_cache[key] = tuple(m.actuator(i).name for i in range(m.nu))
    return _muscle_names_cache[key]


def _compute_target_origin(config) -> np.ndarray:
    """Return absolute target coordinate origin: ref_site world pos + target_origin_rel.

    Compiles the base model (without task geoms) to read the neutral-pose site position.
    The hardcoded fallback [-0.0068, -0.1747, 1.0257] corresponds to ref_site='humphant'.
    """
    model = compile_base_model(config.env.model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ref_site = config.env.task_config.reach_settings.ref_site
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ref_site)
    return data.site_xpos[site_id].copy() + np.array(
        config.env.task_config.reach_settings.target_origin_rel
    )


def _get_myoInteract_root() -> Path:
    """Return the repo root that contains the myoInteract package.

    Uses __file__ rather than epath.resource_path() to avoid the
    MultiplexedPath that importlib_resources returns when several editable
    installs expose the same package name simultaneously (Python 3.13
    pathlib rejects it with TypeError: argument should be str, not MultiplexedPath).

    ``config.env.model_path`` values are repo-root-relative
    (e.g. ``myo_user/envs/myo/assets/arm/...``).

    __file__ path:  myoInteract/register_mjlab_myointeract_tasks.py
    parents[1]:     repo root
    """
    return Path(__file__).resolve().parents[1]


def _torso_contact_penalty_enabled(config: BaseEnvConfig) -> bool:
    return (
        config.env.task_config.weighted_reward_keys.get("torso_contact_penalty", 0.0)
        != 0.0
    )


def _universal_spec_fn(xml_file: Path, config: BaseEnvConfig) -> mujoco.MjSpec:
    seed = getattr(config.run, "seed", None)
    variant = _myoarm_r_variant_for(config.env.model_path)
    if variant is not None:
        spec = _load_myoarm_r_based_spec(variant)
    else:
        spec = mujoco.MjSpec.from_file(str(xml_file))
    spec = add_task_relevant_geoms(spec, config, seed=seed)
    if _torso_contact_penalty_enabled(config):
        spec = add_torso_contact_sensor_to_spec(spec)
    return spec


def add_task_relevant_geoms(
    spec: mujoco.MjSpec, config: BaseEnvConfig, seed: int | None = None
) -> mujoco.MjSpec:
    """
    Add task-specific targets to the MuJoCo model.

    Parameters
    ----------
    spec : mujoco.MjSpec
        The model specification to modify.

    Returns
    -------
    mujoco.MjSpec
        The updated specification including task-specific geometry elements.
    """
    task_config = config.env.task_config
    targets = task_config.targets
    num_targets = targets.num_targets
    _t0 = targets.target_0
    targets = [getattr(targets, f"target_{i}", None) or _t0 for i in range(num_targets)]
    len(targets)
    # target_objs = []
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, task_config.reach_settings.ref_site
    )
    target_coordinates_origin = data.site_xpos[site_id].copy() + np.array(
        task_config.reach_settings.target_origin_rel
    )

    for i, target_cfg in enumerate(targets):
        if target_cfg.name == "pointing_target":
            ##TODO: unify add_sphere_to_spec and add_button_to_spec
            # target_obj = PointingTargetClass(
            #     phase_number=i,
            #     total_phases=total_phases,
            #     target_pos_range=target.position,
            #     target_radius_range=target.size,
            #     target_coordinates_origin=target_coordinates_origin,
            #     dwell_steps=int(target.dwell_duration / config.env.ctrl_dt),
            #     weighted_reward_keys=task_config.weighted_reward_keys,
            #     show_all_targets=task_config.show_all_targets,
            #     enable_extra_dist=task_config.enable_extra_dist,
            #     use_vision=False,
            # )
            spec = add_sphere_to_spec(
                spec,
                target_cfg,
                target_id=i,
                target_coordinates_origin=target_coordinates_origin,
                seed=seed,
            )
        elif target_cfg.name == "button_target":
            # target_obj = ButtonTargetClass(
            #     phase_number=i,
            #     total_phases=total_phases,
            #     position=(np.array(target_cfg.position) + target_coordinates_origin).tolist(),
            #     geom_size=target_cfg.size,
            #     site_size=target_cfg.site_size,
            #     site_pos=target_cfg.site_pos,
            #     geom_margin=target_cfg.geom_margin,
            #     euler=target_cfg.euler,
            #     min_touch_force=target_cfg.min_touch_force,
            #     weighted_reward_keys=task_config.weighted_reward_keys,
            #     show_all_targets=task_config.show_all_targets,
            #     enable_extra_dist=task_config.enable_extra_dist,
            #     use_vision=False,
            # )
            spec = add_button_to_spec(
                spec,
                target_cfg,
                target_id=i,
                target_coordinates_origin=target_coordinates_origin,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported target type: {target_cfg['name']}")
    if any(t.name == "button_target" for t in targets):
        restrict_button_presses(spec, task_config.reach_settings.ee_site)
    return spec


_independent_joint_names = [
    "elv_angle",
    "shoulder_elv",
    "shoulder_rot",
    "elbow_flexion",
    "pro_sup",
]


def _time(env) -> torch.Tensor:
    data = env.scene[_UNIVERSAL_ENTITY_NAME].data.data
    return data.time


def _qpos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids

    jnt_range = asset.data.joint_pos_limits[:, jnt_ids, :]
    qpos = asset.data.joint_pos[:, jnt_ids]  # asset.data.joint_pos[:, jnt_ids]

    qpos = (qpos - jnt_range[..., 0]) / (jnt_range[..., 1] - jnt_range[..., 0])
    qpos = (qpos - 0.5) * 2
    return qpos


def _qvel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids

    qvel = asset.data.joint_vel[:, jnt_ids]
    return qvel


def _qacc(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids

    qacc = asset.data.joint_acc[:, jnt_ids]
    return qacc


def _act(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    """Muscle activation state, filtered to asset_cfg.actuator_ids when set.

    Backward-compatible: every existing caller's asset_cfg only sets
    joint_names/site_names, never actuator_names, so asset_cfg.actuator_ids stays
    the manager's default slice(None) (identity slice, returns every actuator
    unchanged) -- filtering only actually activates for a SceneEntityCfg that
    explicitly sets actuator_names.
    """
    asset = env.scene[asset_cfg.name]
    act = (asset.data.data.act - 0.5) * 2
    return act[:, asset_cfg.actuator_ids]


def _ctrl(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # ctrl = asset.data.data.ctrl
    ctrl = asset.data.tendon_effort_target

    return ctrl


def _sites_pos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # Use asset_cfg.site_names[0] (the site name baked into this term's params at
    # build time), NOT the module-level `ee_pos_name` global: that global is
    # reassigned on every register_mjlab_myointeract_task() call (e.g. from an
    # unrelated render/preview for a different bm_model in the same process), so
    # reading it here would silently pick up whichever model's ee_site was
    # registered *last*, not the one this env was actually built with.
    # (Not asset_cfg.site_ids: this function is also called directly during
    # event-term __init__, before the observation manager resolves site_ids,
    # and resolved ids can additionally collapse to slice(None).)
    sites_pos = asset.data.site_pos_w[
        :, asset.find_sites(asset_cfg.site_names[0])[0][0]
    ]
    return sites_pos


def _sites_pos_z(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    """Height (z) of the ee_pos site — used by the fingertip_height metric."""
    return _sites_pos(env, asset_cfg)[:, 2]


def _next_target_pos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # target_pos = torch.stack([asset.data.body_com_pos_w[:, asset.target_pos_ids[i]] for i in range(asset.num_targets)], dim=1)
    target_pos = asset.data.body_com_pos_w[
        torch.arange(env.num_envs),
        asset.target_pos_ids[
            asset.current_target_id[:, 0].clip(0, asset.num_targets - 1)
        ],
    ]
    return target_pos


def _next_target_size(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target_size = asset.data.model.geom_size[
        torch.arange(env.num_envs),
        asset.target_size_ids[
            asset.current_target_id[:, 0].clip(0, asset.num_targets - 1)
        ],
    ]
    return target_size


def _phase_progress(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    phase_progress = asset.current_target_id[:, 0] / asset.num_targets
    phase_progress = -1.0 + 2.0 * phase_progress.reshape(-1, 1)
    return phase_progress


def _dwell_fraction(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    dwell_fraction = (
        (asset.target_dwell_steps_batch > 0)
        * asset.steps_inside_target
        / asset.target_dwell_steps_batch.clip(1, np.inf)
    )  # avoid nan for targets with 0 dwell steps
    dwell_fraction = -1.0 + 2.0 * dwell_fraction
    return dwell_fraction


def _fatigue_MR(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    env.scene[asset_cfg.name]
    fatigue = _muscles_fatigue(env)
    return fatigue.MR


def _fatigue_MF(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    env.scene[asset_cfg.name]
    fatigue = _muscles_fatigue(env)
    return fatigue.MF


def _fatigue_MA(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    env.scene[asset_cfg.name]
    fatigue = _muscles_fatigue(env)
    return fatigue.MA


# def _vision_obs(env: ManagerBasedRlEnv,
#             asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG) -> "torch.Tensor":
#     asset = env.scene[asset_cfg.name]

#     # Access at runtime.
#     data = env.scene["fixed_eye"].data
#     data.rgb      # [B, 120, 160, 3] uint8
#     data.depth    # [B, 120, 160, 1] float32

#     vision_obs = torch.cat([data.rgb / 255.0, data.depth], dim=-1)
#     return vision_obs


def _sequential_distance_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG,
    exponential_distance_reward=False,
    distance_metric=10.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    distance_to_target = asset.distance_to_target[:, 0]

    ## add remaining distances for future targets
    if asset.num_targets > 1:
        distances_total = (
            distance_to_target
            + asset.between_target_distances.flip(dims=(1,))
            .cumsum(dim=1)
            .flip(dims=(1,))[
                torch.arange(env.num_envs),
                asset.current_target_id[:, 0].clip(0, asset.num_targets - 1),
            ]
        )
    else:
        distances_total = distance_to_target

    if exponential_distance_reward:
        distance_reward = torch.where(
            asset.inside_target[:, 0],
            torch.zeros_like(distances_total),
            torch.expm1(-distances_total * distance_metric) / distance_metric,
        )
    else:
        distance_reward = -1.0 * distances_total

    return distance_reward


def _neural_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # ctrl = asset.data.data.ctrl
    ctrl = asset.data.tendon_effort_target

    ctrl_magnitude = torch.linalg.vector_norm(ctrl, dim=-1)
    neural_effort = -1.0 * (ctrl_magnitude**2)
    return neural_effort


def _jac_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    # ctrl = asset.data.data.ctrl
    ctrl = asset.data.tendon_effort_target
    r_effort = 0.00198 * torch.linalg.vector_norm(ctrl, dim=-1) ** 2
    r_jacc = 6.67e-6 * torch.linalg.vector_norm(_qacc(env, asset_cfg), dim=-1) ** 2
    effort_cost = -(r_effort + r_jacc)
    return effort_cost


def _elbow_velocity_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    """Negative reward proportional to squared linear velocity of the elbow.

    Uses the "ulna" body as the elbow reference: mobl_arms_index_body.xml defines
    the elbow_flexion joint at that body's own local origin (pos="0 0 0"), so the
    ulna body's world-frame velocity IS the elbow's velocity -- chosen over any
    single muscle-wrap-point site in that file (e.g. "BRA-P3", the closest such
    site) since those are arbitrarily-named tendon attachment points, not
    anatomical landmarks, and would be a more fragile/obscure dependency for the
    same result. Mirrors _jac_effort's self-negating convention (function returns
    a negative value, config weight stays positive).

    Assumes a MoBL_Arms_Index-family model (the only bm_model choice in ui.py with
    a body literally named "ulna"); like torso_contact_penalty, this is opt-in
    (weight 0 by default) and simply irrelevant for other skeletons -- it would
    raise if enabled against a model with no "ulna" body.
    """
    asset = env.scene[asset_cfg.name]
    elbow_body_id = asset.find_bodies("ulna")[0][0]
    elbow_pos = asset.data.body_com_pos_w[:, elbow_body_id, :]
    ref_pos = torch.tensor(
        [0.00557765, -0.1143197, 1.3650374],
        device=elbow_pos.device,
        dtype=elbow_pos.dtype,
    )
    return -torch.linalg.vector_norm(elbow_pos - ref_pos, dim=-1)


# --------------------------------------------------------------------------- #
# pain_penalty: soft (comfort) joint-angle limits, independently of the
# mechanical/passive qpos range the joint actually allows.
#
# Draft soft-limit buffers, one comfort-informed number per joint per side --
# the *gap* (in degrees) between the soft (comfort) limit and the joint's
# real, model-defined hard limit (mobl_arms_index_body.xml / myoarm_body.xml
# define essentially the same ranges for these 5 joints, within ~5 degrees):
#
#   elv_angle      hard [-90, 130] deg   soft [-30, 120] deg  (buffers: 60 / 10)
#   shoulder_elv   hard [  0, 180] deg   soft [None, 100] deg (buffers: -- / 80)
#   shoulder_rot   hard [-90, 120] deg   soft [-45,  80] deg  (buffers: 45 / 40)
#   elbow_flexion  hard [  0, 130] deg   soft [None, 120] deg (buffers: -- / 10)
#   pro_sup        hard [-90,  90] deg   soft [-70,  70] deg  (buffers: 20 / 20)
#
# Reasoning (ergonomics/shoulder-biomechanics informed, not empirically
# fitted -- a starting point to tune against actual training runs):
#   - shoulder_elv (thoracohumeral elevation, 0=arm at side, 180=overhead):
#     comfortable/low-fatigue up to roughly shoulder height (~90-100 deg);
#     sustained postures above that get uncomfortable quickly toward
#     full overhead. No low-side soft limit -- resting at the side is fine.
#   - elv_angle (plane of elevation: negative = reaching behind the torso,
#     positive = reaching forward/up): reaching behind the body is
#     uncomfortable well before the mechanical stop, so a large buffer
#     (60 deg) is used on the low/extension side; forward reach stays
#     comfortable almost to the mechanical limit (small 10 deg buffer).
#   - shoulder_rot (external(+)/internal(-) humeral rotation): moderately
#     uncomfortable toward both ends; also coupled to shoulder_elv below.
#   - elbow_flexion (0=extended, 130=fully flexed): comfortable through
#     almost the full range; only sustained near-max flexion (small 10 deg
#     buffer) is penalised.
#   - pro_sup (forearm rotation, symmetric): moderately uncomfortable near
#     both full pronation and full supination.
#
# Each per-joint term is a smooth (quadratic) penalty of the qpos distance
# past the soft limit, normalised by the *remaining buffer to that joint's own
# hard mechanical limit at the current risk level*, so a joint held exactly at
# its hard limit always contributes exactly 1.0 -- making the terms comparable
# across joints (and, for shoulder_rot, across elevation/plane risk levels).
#
# Shoulder coupling: elv_angle/shoulder_elv/shoulder_rot are NOT independent
# for comfort -- they are three sequential hinges approximating a single
# ball-and-socket glenohumeral joint, not three unrelated DOFs. In particular:
#   - shoulder_rot away from neutral is far more uncomfortable (subacromial
#     impingement risk) when the arm is also elevated (shoulder_elv).
#   - That risk is strongly *plane*-dependent: it is highest with the arm
#     abducted (elv_angle near 0, the lateral plane -- the classic "painful
#     arc" position) and meaningfully lower, though not zero, in the forward
#     flexion plane (elv_angle toward 90 deg) or in extension (already
#     penalised heavily on its own by elv_angle's large low-side buffer).
# This is modelled by narrowing shoulder_rot's OWN soft-limit bounds (not a
# separate additive term -- that double-counts the same joint and, worse,
# doesn't reuse the hard-limit-relative normalisation above) as a function of
# elevation risk x plane risk, floored so no plane is ever fully risk-free.
# --------------------------------------------------------------------------- #

_PAIN_SOFT_LIMIT_BUFFERS_DEG: dict = {
    # joint_name: (buffer_lo_deg, buffer_hi_deg); None = no soft limit on that side.
    # shoulder_rot's baseline (risk=0) bounds are used directly here but its
    # actual per-step bounds are computed dynamically -- see
    # _shoulder_rot_soft_bounds_deg -- so this entry is not consulted for it.
    "elv_angle": (60.0, 10.0),
    "shoulder_elv": (None, 80.0),
    "shoulder_rot": (45.0, 40.0),
    "elbow_flexion": (None, 10.0),
    "pro_sup": (20.0, 20.0),
}
_PAIN_SOFT_LIMIT_BUFFERS: dict = {
    name: (
        None if lo_deg is None else math.radians(lo_deg),
        None if hi_deg is None else math.radians(hi_deg),
    )
    for name, (lo_deg, hi_deg) in _PAIN_SOFT_LIMIT_BUFFERS_DEG.items()
}

# shoulder_rot baseline (risk=0) soft bounds -- degrees, matches the
# independent-only design (buffers 45/40 above: hard [-90,120] -> [-45,80]).
_SHOULDER_ROT_BASELINE_LO_DEG = -45.0
_SHOULDER_ROT_BASELINE_HI_DEG = 80.0
# At maximum coupling risk, shoulder_rot's soft bounds shrink to this fraction
# of the baseline bounds above (e.g. 0.4 -> [-18, 32] deg).
_SHOULDER_COUPLING_MIN_FREE_FRACTION = 0.4
# Elevation risk: 0 below ELV_START, ramps linearly to 1 at ELV_FULL.
_SHOULDER_COUPLING_ELV_START_DEG = 60.0
_SHOULDER_COUPLING_ELV_FULL_DEG = 150.0
# Plane risk: peaks (1.0) at PLANE_CENTER_DEG (the abduction/lateral plane,
# elv_angle near 0), decays with Gaussian half-width PLANE_HALF_WIDTH_DEG
# toward flexion/extension planes, floored at PLANE_FLOOR (impingement risk
# from elevation+rotation is reduced but not eliminated in other planes).
_SHOULDER_COUPLING_PLANE_CENTER_DEG = 15.0
_SHOULDER_COUPLING_PLANE_HALF_WIDTH_DEG = 50.0
_SHOULDER_COUPLING_PLANE_FLOOR = 0.3


def _shoulder_rot_soft_bounds_deg(
    shoulder_elv_deg: torch.Tensor, elv_angle_deg: torch.Tensor
):
    """shoulder_rot's (soft_lo_deg, soft_hi_deg), narrowed by elevation x plane risk."""
    elv_risk = torch.clamp(
        (shoulder_elv_deg - _SHOULDER_COUPLING_ELV_START_DEG)
        / (_SHOULDER_COUPLING_ELV_FULL_DEG - _SHOULDER_COUPLING_ELV_START_DEG),
        min=0.0,
        max=1.0,
    )
    plane_risk = _SHOULDER_COUPLING_PLANE_FLOOR + (
        1.0 - _SHOULDER_COUPLING_PLANE_FLOOR
    ) * torch.exp(
        -(
            (
                (elv_angle_deg - _SHOULDER_COUPLING_PLANE_CENTER_DEG)
                / _SHOULDER_COUPLING_PLANE_HALF_WIDTH_DEG
            )
            ** 2
        )
    )
    risk = elv_risk * plane_risk
    shrink = 1.0 - risk * (1.0 - _SHOULDER_COUPLING_MIN_FREE_FRACTION)
    return (
        _SHOULDER_ROT_BASELINE_LO_DEG * shrink,
        _SHOULDER_ROT_BASELINE_HI_DEG * shrink,
    )


def _pain_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    """Negative reward for holding an independent-DOF joint past its soft
    (comfort) angle limit -- see the module-level draft/reasoning above.

    Only the 5 independent DOFs (asset_cfg.joint_names, always
    ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion', 'pro_sup']
    for this task -- see _independent_joint_names) are penalised; muscle-only
    DOFs (fingers, wrist, etc.) are out of scope, matching the rest of the
    qpos-based observation/reward terms in this module. Mirrors
    _elbow_velocity_penalty's self-negating convention (function returns a
    negative value, config weight stays positive). Uses the *model's own*
    joint_pos_limits (not hardcoded hard limits) so the soft buffers stay
    correct across the MoBL_Arms/myoarm model families, whose hard ranges for
    these 5 joints differ by only a few degrees.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    qpos = asset.data.joint_pos[:, joint_ids]
    limits = asset.data.joint_pos_limits[:, joint_ids, :]

    q_by_name = {name: qpos[:, i] for i, name in enumerate(_independent_joint_names)}
    rot_soft_lo_deg, rot_soft_hi_deg = _shoulder_rot_soft_bounds_deg(
        torch.rad2deg(q_by_name["shoulder_elv"]), torch.rad2deg(q_by_name["elv_angle"])
    )

    total = torch.zeros(qpos.shape[0], device=qpos.device, dtype=qpos.dtype)
    for i, name in enumerate(_independent_joint_names):
        q = qpos[:, i]
        hard_lo, hard_hi = limits[:, i, 0], limits[:, i, 1]
        if name == "shoulder_rot":
            # Coupling-adjusted bounds (radians), computed above -- everything
            # past this point is identical to the generic per-joint case.
            soft_lo, soft_hi = (
                torch.deg2rad(rot_soft_lo_deg),
                torch.deg2rad(rot_soft_hi_deg),
            )
            v_hi = torch.clamp(q - soft_hi, min=0.0)
            total = total + (v_hi / torch.clamp(hard_hi - soft_hi, min=1e-4)) ** 2
            v_lo = torch.clamp(soft_lo - q, min=0.0)
            total = total + (v_lo / torch.clamp(soft_lo - hard_lo, min=1e-4)) ** 2
            continue
        lo_buf, hi_buf = _PAIN_SOFT_LIMIT_BUFFERS[name]
        if hi_buf is not None:
            v_hi = torch.clamp(q - (hard_hi - hi_buf), min=0.0)
            total = total + (v_hi / hi_buf) ** 2
        if lo_buf is not None:
            v_lo = torch.clamp((hard_lo + lo_buf) - q, min=0.0)
            total = total + (v_lo / lo_buf) ** 2

    # Curriculum coupling (CurriculumConfig.pain_relax_factor): CurriculumState
    # maintains a per-env scale in [0, 1] that relaxes this penalty for envs
    # currently assigned to harder/under-performing target compartments, letting
    # them explore more extreme (less human-like) postures while mastered
    # compartments stay constrained to comfortable ones. Absent (curriculum
    # disabled, or pain_relax_factor=0) it's a no-op -- scale stays 1.0 everywhere.
    scale = getattr(asset, "curriculum_pain_scale", None)
    if scale is not None:
        total = total * scale

    return -total


def _torso_contact_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    """Negative reward proportional to total torso-vs-limb contact force.

    Only meaningful when the model was built with add_torso_contact_sensor_to_spec
    (gated in _universal_spec_fn on weighted_reward_keys["torso_contact_penalty"]
    being set/nonzero) -- the sensor_torso_contact_coll{0..3} sensors this reads
    don't exist otherwise. Mirrors _jac_effort's self-negating convention (function
    returns a negative value, config weight stays positive).
    """
    sensor_sum = torch.cat(
        [
            env.scene[f"{asset_cfg.name}/{name}"].data
            for name in TORSO_CONTACT_SENSOR_NAMES
        ],
        dim=1,
    ).sum(dim=1)
    return -sensor_sum


# Forearm rotation people point, press and type with: palm down. pro_sup is positive towards
# pronation; being less pronated than this costs, more pronated is free.
_PRONATED_DEG = 60.0


def _pronation_reward(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    """Negative reward for a forearm that is less pronated than ``_PRONATED_DEG``.

    Quadratic in the shortfall, scaled so that full supination (-90 degrees) costs 1.
    """
    asset = env.scene[asset_cfg.name]
    pro_sup = asset.data.joint_pos[:, asset_cfg.joint_ids][
        :, _independent_joint_names.index("pro_sup")
    ]
    shortfall = torch.clamp(
        math.radians(_PRONATED_DEG) - pro_sup, min=0.0
    ) / math.radians(_PRONATED_DEG + 90.0)
    return -(shortfall**2)


# Force above which a button press starts to cost (N): a deliberate press, not a strike.
_GENTLE_PRESS_N = 5.0


def _button_force_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    """Negative reward for pressing the current button harder than ``_GENTLE_PRESS_N``.

    Grows with the logarithm of the excess force, so a 900 N strike costs a few times a 50 N one
    instead of swamping every other term. Zero for sphere targets (their touch force is 0).
    """
    force = env.scene[asset_cfg.name].button_touch_force[:, 0]
    return -torch.log1p(torch.clamp(force - _GENTLE_PRESS_N, min=0.0) / _GENTLE_PRESS_N)


def _phase_bonus(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    phase_bonus = asset.phase_successfully_completed[:, 0]
    return phase_bonus


def _phase_successfully_completed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG,
    phase_id: int = 0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    current_target_id = asset.current_target_id[:, 0]
    phase_successfully_completed = current_target_id > phase_id
    return phase_successfully_completed


@dataclass
class TaskEntityCfg(EntityCfg):
    base_config: BaseEnvConfig | None = None

    def build(self) -> TaskEntity:
        """Build task entity instance from this config."""
        return TaskEntity(self)


class TaskEntity(Entity):
    def __init__(self, cfg: TaskEntityCfg):
        super().__init__(cfg)
        base_config = self.cfg.base_config

        self.task_config = base_config.env.task_config
        self.ctrl_dt = base_config.env.ctrl_dt
        self.num_targets = self.task_config.targets.num_targets

        self.target_pos_ids = torch.tensor(
            [
                self.find_bodies(f"body_target_{i}")[0][0]
                for i in range(self.num_targets)
            ],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.target_size_ids = torch.tensor(
            [
                self.find_geoms(f"geom_target_{i}")[0][0]
                for i in range(self.num_targets)
            ],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


# --------------------------------------------------------------------------- #
# Adaptive per-compartment target curriculum + initial-state generalization
# (myosuite/core/myointeract_configs.py::CurriculumConfig).
# --------------------------------------------------------------------------- #


def _curriculum_enabled(config: BaseEnvConfig) -> bool:
    curr_cfg = config.env.get("curriculum_config", None)
    return bool(curr_cfg.enabled) if curr_cfg is not None else False


def _build_target_compartments(
    box_lo: np.ndarray, box_hi: np.ndarray, n_x: int, n_z: int
):
    """Partition the target box into n_x * n_z sub-boxes, gridded over x (index 0,
    forward reach) and z (index 2, elevation); y (index 1, lateral) is left untiled
    -- difficulty correlates most with reach distance and elevation, and this keeps
    the compartment count small (n_x*n_z instead of n_x*n_y*n_z). A single
    swappable function so the partition scheme can be upgraded later (non-uniform
    bins, gridding y too, etc.) without touching callers.

    Returns:
        (N, 3) lo, (N, 3) hi absolute sub-box bounds, N = n_x * n_z.
    """
    x_edges = np.linspace(box_lo[0], box_hi[0], n_x + 1)
    z_edges = np.linspace(box_lo[2], box_hi[2], n_z + 1)
    lo = np.zeros((n_x * n_z, 3), dtype=np.float32)
    hi = np.zeros((n_x * n_z, 3), dtype=np.float32)
    idx = 0
    for ix in range(n_x):
        for iz in range(n_z):
            lo[idx] = [x_edges[ix], box_lo[1], z_edges[iz]]
            hi[idx] = [x_edges[ix + 1], box_hi[1], z_edges[iz + 1]]
            idx += 1
    return lo, hi


class PostureBuffer:
    """Fixed-size circular buffer of (qpos, qvel, act) posture snapshots.

    Add/size/sample-by-index contract over a 3-tensor row. The caller writes sampled
    results via write_joint_state_to_sim + direct act assignment.
    """

    def __init__(
        self,
        nq: int,
        nv: int,
        na: int,
        buffer_size: int,
        device: str = "cpu",
        sampling: str = "uniform",
    ) -> None:
        if buffer_size < 1:
            raise ValueError("Posture buffer capacity must be positive.")
        self._qpos = torch.zeros(buffer_size, nq, device=device)
        self._qvel = torch.zeros(buffer_size, nv, device=device)
        self._act = torch.zeros(buffer_size, na, device=device)
        self._ptr = 0
        self._full = False
        self.sampling = sampling

    def add(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        act: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        rows_qpos = qpos if env_ids is None else qpos[env_ids]
        rows_qvel = qvel if env_ids is None else qvel[env_ids]
        rows_act = act if env_ids is None else act[env_ids]
        n = rows_qpos.shape[0]
        if n == 0:
            return
        capacity = self._qpos.shape[0]
        skip = max(0, n - capacity)
        idxs = (
            torch.arange(self._ptr + skip, self._ptr + n, device=rows_qpos.device)
            % capacity
        )
        self._qpos[idxs] = rows_qpos[skip:]
        self._qvel[idxs] = rows_qvel[skip:]
        self._act[idxs] = rows_act[skip:]
        if self._ptr + n >= capacity and not self._full:
            self._full = True
        self._ptr = (self._ptr + n) % capacity

    @property
    def size(self) -> int:
        return self._qpos.shape[0] if self._full else self._ptr

    def sample(self, n: int):
        """Sample n rows -> (qpos, qvel, act) or None if the buffer is empty."""
        if self.size == 0 or n == 0:
            return None
        idx = torch.randint(0, self.size, (n,), device=self._qpos.device)
        return self._qpos[idx], self._qvel[idx], self._act[idx]


# Initial-state branch codes used by CurriculumState/CurriculumInitialState.
_INIT_BRANCH_READY = 0
_INIT_BRANCH_POSTURE = 1
_INIT_BRANCH_PERSIST = 2


class CurriculumState(ManagerTermBase):
    """CurriculumTermCfg term (mode: compute) driving the adaptive per-compartment
    target curriculum and the initial-state generalization schedule.

    Runs once per reset batch BEFORE sim.reset()/event_manager.apply(mode="reset")
    (see ManagerBasedRlEnv._reset_idx) -- the only lifecycle point where the
    about-to-be-overwritten live qpos/qvel/act can be captured for the "persist"
    initial-state branch, and where the per-compartment success EMA can be updated
    from the termination that just fired for these env_ids.

    All persistent per-env/per-compartment state lives on the scene entity
    (asset.curriculum_*, asset._ema_*, asset._init_branch, asset.posture_buffer)
    so CurriculumTargetSync/PostureBufferCapture/CurriculumInitialState -- separate
    manager terms with no other shared-state channel -- can read it. Restricted to
    the single-target case (see CurriculumTargetSync).
    """

    def __init__(self, cfg, env) -> None:
        self._env = env
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        self._asset = asset
        p = cfg.params
        self._cfg = p
        device = env.device

        box_lo = np.asarray(p["box_lo"], dtype=np.float32)
        box_hi = np.asarray(p["box_hi"], dtype=np.float32)
        lo, hi = _build_target_compartments(box_lo, box_hi, p["n_x"], p["n_z"])
        n_compartments = lo.shape[0]
        self._n_compartments = n_compartments
        asset.curriculum_compart_lo = torch.tensor(lo, device=device)
        asset.curriculum_compart_hi = torch.tensor(hi, device=device)

        asset._ema_train = torch.full((n_compartments,), 0.5, device=device)
        asset._ema_eval = torch.full((n_compartments,), 0.5, device=device)

        num_envs = env.num_envs
        n_eval = max(1, round(p["eval_frac"] * num_envs)) if p["eval_frac"] > 0 else 0
        eval_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        eval_mask[:n_eval] = True
        asset._eval_mask = eval_mask
        # Round-robin, fixed for the entity's lifetime: eval envs cycle
        # deterministically through every compartment so eval EMAs stay unbiased by
        # the resampling weights that drive training-env compartment assignment.
        asset._eval_cycle_ptr = torch.zeros((), dtype=torch.long, device=device)

        asset.curriculum_bucket = torch.randint(
            0, n_compartments, (num_envs,), device=device
        )
        if n_eval > 0:
            eval_ids = torch.nonzero(eval_mask, as_tuple=True)[0]
            asset.curriculum_bucket[eval_ids] = (
                torch.arange(n_eval, device=device) % n_compartments
            )

        na = asset.data.data.act.shape[1]
        nq = asset.data.joint_pos.shape[1]
        nv = asset.data.joint_vel.shape[1]
        asset.posture_buffer = PostureBuffer(
            nq=nq,
            nv=nv,
            na=na,
            buffer_size=p["posture_buffer_size"],
            device=str(device),
        )
        asset._persist_qpos = asset.data.joint_pos.clone()
        asset._persist_qvel = asset.data.joint_vel.clone()
        asset._persist_act = asset.data.data.act.clone()
        asset._init_branch = torch.full(
            (num_envs,), _INIT_BRANCH_READY, dtype=torch.long, device=device
        )
        asset.curriculum_pain_scale = torch.ones(num_envs, device=device)
        self._p_ready_mean = p["ready_base_prob"]

    def __call__(self, env, env_ids, **_):
        asset = self._asset
        p = self._cfg
        device = env.device
        num_envs = env.num_envs

        if isinstance(env_ids, slice):
            env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        else:
            env_ids = env_ids.to(dtype=torch.long)
        n = len(env_ids)
        if n == 0:
            return self._log_dict()

        # 1) Snapshot live state for the "persist" branch BEFORE sim.reset wipes it.
        asset._persist_qpos[env_ids] = asset.data.joint_pos[env_ids]
        asset._persist_qvel[env_ids] = asset.data.joint_vel[env_ids]
        asset._persist_act[env_ids] = asset.data.data.act[env_ids]

        # 2) Update per-compartment success EMA from the episode that just ended.
        success = env.termination_manager.get_term("episode_success")[env_ids].float()
        prev_bucket = asset.curriculum_bucket[env_ids]
        is_eval = asset._eval_mask[env_ids]
        self._scatter_ema(
            asset._ema_train, prev_bucket[~is_eval], success[~is_eval], p["ema_decay"]
        )
        self._scatter_ema(
            asset._ema_eval, prev_bucket[is_eval], success[is_eval], p["ema_decay"]
        )

        # 3) Reassign compartments for the upcoming episode. Eval envs keep a fixed
        # round-robin cycle (independent of learned weights, for an unbiased
        # readout); training envs resample weighted toward under-performing ones.
        new_bucket = prev_bucket.clone()
        train_ids_local = torch.nonzero(~is_eval, as_tuple=True)[0]
        if len(train_ids_local) > 0:
            difficulty = 1.0 - asset._ema_train
            weights = p["weight_floor"] + (1.0 - p["weight_floor"]) * difficulty
            weights = weights * (1.0 + p["difficulty_boost"] * difficulty)
            weights = weights / weights.sum()
            sampled = torch.multinomial(weights, len(train_ids_local), replacement=True)
            new_bucket[train_ids_local] = sampled
        eval_ids_local = torch.nonzero(is_eval, as_tuple=True)[0]
        if len(eval_ids_local) > 0:
            start = int(asset._eval_cycle_ptr.item())
            cyc = (
                start + torch.arange(len(eval_ids_local), device=device)
            ) % self._n_compartments
            new_bucket[eval_ids_local] = cyc
            asset._eval_cycle_ptr += len(eval_ids_local)
        asset.curriculum_bucket[env_ids] = new_bucket

        # 4) Initial-state branch: ready (deterministic initpose001) vs.
        # posture-buffer vs. persist (previous episode's end state), with a hard
        # floor on p(ready) so the policy never fully forgets the fixed-pose start.
        total_env_steps = env.common_step_counter * num_envs
        decay = min(1.0, total_env_steps / max(1, p["ready_decay_horizon_steps"]))
        p_ready = p["ready_base_prob"] - decay * (
            p["ready_base_prob"] - p["ready_floor_prob"]
        )
        p_ready = max(p["ready_floor_prob"], p_ready)
        draw = torch.rand(n, device=device)
        is_ready = draw < p_ready
        branch = torch.full((n,), _INIT_BRANCH_PERSIST, dtype=torch.long, device=device)
        posture_draw = torch.rand(n, device=device)
        is_posture = (~is_ready) & (posture_draw < p["posture_split"])
        branch[is_posture] = _INIT_BRANCH_POSTURE
        branch[is_ready] = _INIT_BRANCH_READY
        if asset.posture_buffer.size == 0:
            # Buffer still empty (e.g. early training): fall back to ready.
            branch[branch == _INIT_BRANCH_POSTURE] = _INIT_BRANCH_READY
        asset._init_branch[env_ids] = branch
        self._p_ready_mean = p_ready

        # 5) Pain-penalty coupling: relax the soft-limit penalty for envs assigned
        # to harder (lower-EMA) compartments, proportional to pain_relax_factor.
        bucket_difficulty = 1.0 - asset._ema_train[new_bucket]
        asset.curriculum_pain_scale[env_ids] = (
            1.0 - p["pain_relax_factor"] * bucket_difficulty
        )

        return self._log_dict()

    @staticmethod
    def _scatter_ema(
        ema: torch.Tensor, buckets: torch.Tensor, values: torch.Tensor, decay: float
    ) -> None:
        if len(buckets) == 0:
            return
        # Multiple env_ids in this reset batch can share the same bucket; update
        # per distinct bucket present (a plain scatter_ would be last-write-wins,
        # not an averaged update) -- the number of distinct buckets is small
        # (<= n_x*n_z), so this loop is cheap regardless of num_envs.
        for b in torch.unique(buckets):
            mask = buckets == b
            mean_val = values[mask].mean()
            ema[b] = decay * ema[b] + (1.0 - decay) * mean_val

    def _log_dict(self) -> dict:
        asset = self._asset
        log = {}
        for i in range(self._n_compartments):
            log[f"bucket_{i}_train_success"] = asset._ema_train[i]
            log[f"bucket_{i}_eval_success"] = asset._ema_eval[i]
        log["p_ready_mean"] = self._p_ready_mean
        return log


class CurriculumTargetSync(ManagerTermBase):
    """Replaces "target_pos_dr" when the adaptive curriculum is enabled: draws each
    just-reset env's target position uniformly within its assigned compartment
    sub-box (asset.curriculum_bucket, maintained by CurriculumState), instead of
    the full configured target range. Carries the class-level
    model_fields/recompute attributes that writing body_pos at reset requires.

    Restricted to the single-target case (asset.num_targets == 1): the compartment
    curriculum only covers pointing tasks with one target sphere; multi-target
    sequential tasks are out of scope for this pass.
    """

    model_fields = ("body_pos",)
    recompute = RecomputeLevel.set_const_0

    def __init__(self, cfg, env) -> None:
        self._env = env

    def __call__(
        self, env, env_ids, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG, **_
    ) -> None:
        asset = env.scene[asset_cfg.name]
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(dtype=torch.long)
        bucket = asset.curriculum_bucket[env_ids]
        lo = asset.curriculum_compart_lo[bucket]
        hi = asset.curriculum_compart_hi[bucket]
        u = torch.rand_like(lo)
        target_pos = (lo + u * (hi - lo)).unsqueeze(
            1
        )  # (n, 1, 3), absolute world coords

        # Translate entity-local target body ids to global model body indices.
        body_indices = asset.indexing.body_ids[asset.target_pos_ids[:1]].to(
            dtype=torch.long
        )
        env_grid, body_grid = torch.meshgrid(env_ids, body_indices, indexing="ij")
        env.sim.model.body_pos[env_grid, body_grid] = target_pos


class PostureBufferCapture(ManagerTermBase):
    """Interval event that snapshots the current qpos/qvel/act into the curriculum's
    posture buffer (asset.posture_buffer, owned/created by CurriculumState).
    Only a fraction of live envs is
    sampled at each tick (capture_fraction) rather than all envs, to keep the
    buffer's coverage broad without every env contributing every tick.

    Reads asset.posture_buffer lazily inside __call__ (not __init__) since
    event_manager terms are constructed before curriculum_manager (see
    ManagerBasedRlEnv.load_managers order) -- the buffer does not exist yet at this
    term's own __init__ time.
    """

    def __init__(self, cfg, env) -> None:
        self._env = env
        self._asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        self._fraction = cfg.params.get("capture_fraction", 0.25)

    def __call__(self, env, env_ids, **_) -> None:
        buf: PostureBuffer | None = getattr(self._asset, "posture_buffer", None)
        if buf is None:
            return
        asset = self._asset
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        n_sample = max(1, int(round(len(env_ids) * self._fraction)))
        if n_sample < len(env_ids):
            perm = torch.randperm(len(env_ids), device=env_ids.device)[:n_sample]
            env_ids = env_ids[perm]
        buf.add(
            asset.data.joint_pos, asset.data.joint_vel, asset.data.data.act, env_ids
        )


class CurriculumInitialState(ManagerTermBase):
    """Reset-mode event (runs after sim.reset, see ManagerBasedRlEnv._reset_idx)
    that writes the actual initial qpos/qvel/act per env, according to the branch
    drawn by CurriculumState (asset._init_branch) for this reset batch: the
    deterministic initpose001 pose, a resampled posture-buffer snapshot, or the
    pre-reset "persist" snapshot of the previous episode's end state. Uses the same
    unfiltered write primitives as _reset_joints_initpose001 (full joint set,
    direct act assignment) since the fixed pose spans every joint in the chain, not
    just the independent DOFs.

    Model-restricted to the 17-joint/26-muscle mobl_arms_index v2.1/v2.2 family,
    same guard as _reset_joints_initpose001 (initpose001 constants only make sense
    for that model).
    """

    def __init__(self, cfg, env) -> None:
        self._env = env
        self._asset_cfg = cfg.params.get(
            "asset_cfg", SceneEntityCfg(_UNIVERSAL_ENTITY_NAME)
        )

    def __call__(self, env, env_ids, **_) -> None:
        asset = env.scene[self._asset_cfg.name]
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(dtype=torch.long)
        _validate_initpose001_model(asset, asset.data.data.act.shape[1])
        n = len(env_ids)

        branch = asset._init_branch[env_ids]
        qpos = (
            torch.tensor(_INITPOSE001_QPOS, device=env.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(n, -1)
            .clone()
        )
        qvel = torch.zeros_like(qpos)
        act = (
            torch.tensor(_INITPOSE001_ACT, device=env.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(n, -1)
            .clone()
        )

        is_persist = branch == _INIT_BRANCH_PERSIST
        if is_persist.any():
            qpos[is_persist] = asset._persist_qpos[env_ids][is_persist]
            qvel[is_persist] = asset._persist_qvel[env_ids][is_persist]
            act[is_persist] = asset._persist_act[env_ids][is_persist]

        is_posture = branch == _INIT_BRANCH_POSTURE
        if is_posture.any():
            sampled = asset.posture_buffer.sample(int(is_posture.sum().item()))
            if sampled is not None:
                s_qpos, s_qvel, s_act = sampled
                qpos[is_posture] = s_qpos
                qvel[is_posture] = s_qvel
                act[is_posture] = s_act
            # else: buffer emptied out from under us (race with CurriculumState's
            # own empty-buffer fallback) -- leave the initpose001 default in place.

        asset.write_joint_state_to_sim(
            qpos, qvel, env_ids=env_ids, joint_ids=self._asset_cfg.joint_ids
        )
        asset.data.data.act[env_ids] = act


def _curriculum_difficulty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG,
) -> torch.Tensor:
    """Current-episode compartment difficulty (1 - train success EMA) per env, 0
    when the curriculum is disabled (asset.curriculum_bucket doesn't exist) --
    exposed as an observation so a HeteroscedasticGaussianDistribution can learn to
    raise exploration std for harder/under-performing compartments
    (CurriculumConfig.exploration_coupling).
    """
    asset = env.scene[asset_cfg.name]
    bucket = getattr(asset, "curriculum_bucket", None)
    if bucket is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    difficulty = 1.0 - asset._ema_train[bucket]
    return difficulty.unsqueeze(-1)


class SequentialTaskLogic(ManagerTermBase):
    """Shared sequence state, initialized before observation manager construction.

    Reset events clear episode state. ``SequentialTaskSuccess`` advances it during
    native termination computation, before rewards and metrics consume the result.
    """

    permitted_target_types = [
        "pointing",
        "button",
    ]  # TODO: check if this is sufficient for all target types defined in the config

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        self.asset = asset
        asset.sequential_task_logic = self
        self._last_updated_step = env.common_step_counter
        self._extra_distances_dirty = True
        # Read the ee-site asset_cfg from this event term's params (baked at
        # build time by _make_universal_env_cfg) rather than relying on
        # _sites_pos's own default argument, which is bound once at module
        # import time and never reflects the model this env was actually
        # built with.
        self._asset_cfg = cfg.params.get("asset_cfg", _OBS_PROPRIO_ENTITY_CFG)

        # TaskEntity.__init__ built these before the env (and its device) existed,
        # guessing "cuda" whenever it's available even for a CPU-device env.
        asset.target_pos_ids = asset.target_pos_ids.to(env.device)
        asset.target_size_ids = asset.target_size_ids.to(env.device)

        asset.current_target_id = torch.tensor([[0]] * env.num_envs, device=env.device)
        asset.steps_inside_target = torch.tensor(
            [[0]] * env.num_envs, device=env.device
        )
        asset.between_target_distances = torch.zeros(
            (env.num_envs, asset.num_targets), device=asset.current_target_id.device
        )

        _tcfg = asset.task_config.targets
        _t0_cfg = _tcfg.target_0
        asset.target_types = [
            (getattr(_tcfg, f"target_{i}", None) or _t0_cfg).name.split("_")[0]
            for i in range(asset.num_targets)
        ]  # TODO: map to integer indices (enum) and use for success constraints etc., or remove from asset and directly use target_cfg in the relevant methods
        asset.target_dwell_steps = torch.tensor(
            [
                [
                    getattr(
                        getattr(_tcfg, f"target_{i}", None) or _t0_cfg,
                        "dwell_duration",
                        0.0,
                    )
                    / asset.ctrl_dt
                    for i in range(asset.num_targets)
                ]
            ]
            * env.num_envs,
            device=env.device,
        )
        asset.target_dwell_steps_batch = asset.target_dwell_steps[
            torch.arange(env.num_envs, device=env.device),
            asset.current_target_id[:, 0].clip(0, asset.num_targets - 1),
        ].reshape(-1, 1)
        asset.target_min_touch_force = torch.tensor(
            [
                [
                    getattr(
                        getattr(_tcfg, f"target_{i}", None) or _t0_cfg,
                        "min_touch_force",
                        0.0,
                    )
                    for i in range(asset.num_targets)
                ]
            ]
            * env.num_envs,
            device=env.device,
        )

        # Track whether any target requires a button press; pure-pointing configs skip sensor lookups.
        self._has_button_targets = any(t == "button" for t in asset.target_types)
        self._pointing_targets = torch.tensor(
            [kind == "pointing" for kind in asset.target_types],
            dtype=torch.bool,
            device=env.device,
        )

        # Event terms are constructed before model fields are expanded per world.
        # Geometry is refreshed after reset/physics, rather than indexed here.
        asset.inside_target = torch.zeros(
            env.num_envs, 1, dtype=torch.bool, device=env.device
        )
        asset.distance_to_target = torch.zeros(env.num_envs, 1, device=env.device)
        asset.target_size = torch.zeros(env.num_envs, 1, device=env.device)
        asset.button_clicked = torch.zeros_like(asset.inside_target)
        asset.button_touch_force = torch.zeros_like(asset.distance_to_target)
        asset.phase_successfully_completed = torch.zeros(
            (env.num_envs, 1), dtype=torch.bool, device=asset.current_target_id.device
        )

    def _compute_extra_distances(self, asset, env_ids=None):
        if asset.num_targets <= 1:
            return
        if env_ids is None:
            env_ids = slice(None)
        # Vectorised: index all target body positions at once, diff consecutive pairs.
        pos = asset.data.body_com_pos_w[env_ids][
            :, asset.target_pos_ids, :
        ]  # (E, N, 3)
        diffs = pos[:, 1:, :] - pos[:, :-1, :]  # (E, N-1, 3)
        asset.between_target_distances[env_ids, : asset.num_targets - 1] = (
            torch.linalg.vector_norm(diffs, dim=-1)  # (E, N-1)
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        asset = self.asset
        if env_ids is None:
            env_ids = slice(None)
        asset.current_target_id[env_ids] = 0
        asset.steps_inside_target[env_ids] = 0
        asset.target_dwell_steps_batch[env_ids] = asset.target_dwell_steps[
            env_ids, 0
        ].reshape(-1, 1)
        asset.phase_successfully_completed[env_ids] = False
        asset.inside_target[env_ids] = False
        asset.distance_to_target[env_ids] = 0
        asset.button_clicked[env_ids] = False
        asset.button_touch_force[env_ids] = 0
        asset.between_target_distances[env_ids] = 0
        asset.target_size[env_ids] = asset.data.model.geom_size[
            env_ids, asset.target_size_ids[0], 0
        ].reshape(-1, 1)
        # Reset events run before forward kinematics. Read the newly placed
        # targets at the next termination check, after physics has refreshed them.
        self._extra_distances_dirty = True
        self._last_updated_step = self._env.common_step_counter

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
    ) -> None:
        """Reset-event hook; advancing a task is handled by the termination term."""
        self.reset(env_ids)

    def advance(self, env: ManagerBasedRlEnv) -> None:
        """Advance at most one phase once per native control step."""
        if self._last_updated_step == env.common_step_counter:
            return
        self._last_updated_step = env.common_step_counter
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        env_ids = torch.arange(env.num_envs, device=env.device)
        if self._extra_distances_dirty:
            self._compute_extra_distances(asset)
            self._extra_distances_dirty = False

        (
            asset.inside_target[env_ids],
            asset.distance_to_target[env_ids],
            target_pos,
            asset.target_size[env_ids],
        ) = self._inside_target(env, env_ids)
        asset.steps_inside_target[env_ids] += asset.inside_target[env_ids]

        asset.button_clicked[env_ids], asset.button_touch_force[env_ids] = (
            self._button_clicked(env, env_ids)
        )

        asset.phase_successfully_completed[env_ids] = (
            (
                asset.steps_inside_target[env_ids]
                >= asset.target_dwell_steps_batch[env_ids]
            )
            & asset.button_clicked[env_ids]
            & (asset.current_target_id[env_ids] < asset.num_targets)
        )
        # A zero-duration sphere still requires entering it; buttons instead
        # require their touch sensor, including rectangular pads wider than x.
        pointing = self._pointing_targets[
            asset.current_target_id[:, 0].clamp(max=asset.num_targets - 1)
        ].reshape(-1, 1)
        asset.phase_successfully_completed &= ~pointing | asset.inside_target
        asset.steps_inside_target[env_ids] *= ~asset.phase_successfully_completed[
            env_ids
        ]  # reset if target completed

        asset.current_target_id[env_ids] += asset.phase_successfully_completed[env_ids]
        asset.target_dwell_steps_batch[env_ids] = asset.target_dwell_steps[
            env_ids,
            asset.current_target_id[env_ids, 0].clip(0, asset.num_targets - 1),
        ].reshape(-1, 1)
        # Rewards and observations now describe the same next target. The force
        # cache above deliberately retains this step's actual button contact.
        (
            asset.inside_target[env_ids],
            asset.distance_to_target[env_ids],
            _,
            asset.target_size[env_ids],
        ) = self._inside_target(env, env_ids)
        asset.distance_to_target *= asset.current_target_id < asset.num_targets

    def _inside_target(self, env, env_ids=None) -> torch.Tensor:
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        model = asset.data.model
        if env_ids is None:
            current_target_id = asset.current_target_id[:, 0].clip(
                0, asset.num_targets - 1
            )
            env_ids = torch.arange(env.num_envs, device=asset.current_target_id.device)
        else:
            current_target_id = asset.current_target_id[env_ids, 0].clip(
                0, asset.num_targets - 1
            )

        target_pos_id = asset.target_pos_ids[current_target_id]
        target_size_id = asset.target_size_ids[current_target_id]

        ee_pos = _sites_pos(env, self._asset_cfg)[env_ids]
        target_pos = asset.data.body_com_pos_w[env_ids, target_pos_id]
        target_size = model.geom_size[
            env_ids, target_size_id
        ][
            :, [0]
        ]  # assuming target_size is a radius for a spherical target; adjust if using different target shapes

        distance_to_target = torch.linalg.vector_norm(
            ee_pos - target_pos, dim=-1, keepdim=True
        )
        inside_target = distance_to_target < target_size
        return inside_target, distance_to_target, target_pos, target_size

    def _button_clicked(self, env, env_ids=None) -> torch.Tensor:
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=asset.current_target_id.device)

        # Pointing targets never require a force threshold — skip the O(N) sensor loop entirely.
        if not self._has_button_targets:
            n = len(env_ids)
            return (
                torch.ones(
                    n, 1, dtype=torch.bool, device=asset.current_target_id.device
                ),
                torch.zeros(n, 1, device=asset.current_target_id.device),
            )

        current_target_id = asset.current_target_id[env_ids, 0].clip(
            0, asset.num_targets - 1
        )
        sensor_data_complete = torch.cat(
            [
                env.scene[f"{_UNIVERSAL_ENTITY_NAME}/sensor_target_{i}"].data
                for i in range(asset.num_targets)
            ],
            dim=1,
        )  # WARNING: asserts that each sensor produces (batch_size, 1) sensordata; in case of vector sensordata, use torch.cat(..., dim=2)
        button_touch_force = sensor_data_complete[env_ids, current_target_id].reshape(
            -1, 1
        )
        button_clicked = (
            button_touch_force
            >= asset.target_min_touch_force[env_ids, current_target_id].reshape(-1, 1)
        )  # TODO: check if sensor_force is the right way to detect button press for all target types

        return button_clicked, button_touch_force


class SequentialTaskSuccess(ManagerTermBase):
    """Native termination hook for the sequence state shared with reset events."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._logic = env.scene[_UNIVERSAL_ENTITY_NAME].sequential_task_logic

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        self._logic.advance(env)
        asset = self._logic.asset
        return asset.current_target_id[:, 0] >= asset.num_targets


def _sequential_task_time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Bootstrap only unfinished sequences at the deadline, never terminal wins."""
    asset = env.scene[_UNIVERSAL_ENTITY_NAME]
    # This term may run before episode_success. Its guarded update is shared with
    # SequentialTaskSuccess, so both consume this step's result exactly once.
    asset.sequential_task_logic.advance(env)
    completed = asset.current_target_id[:, 0] >= asset.num_targets
    return mdp_terminations.time_out(env) & ~completed


def _make_universal_env_cfg(config: Config, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Minimal ManagerBasedRlEnvCfg for myoInteractUniversal-v0 (1 env, CPU/GPU)."""

    # Extract observation keys
    # Fatigue observations only exist with fatigue enabled (the fatigue state is None otherwise).
    fatigue_obs_keys = (
        config.env.muscle_config.fatigue_obs_keys
        if config.env.muscle_config.fatigue_enabled
        else []
    )
    obs_keys = (
        config.env.task_config.obs_keys + fatigue_obs_keys
    )  # TODO: merge fatigue_obs_keys into obs_keys (in gradio class, config files, etc.)
    omni_keys = config.env.task_config.omni_keys
    vision_enabled = config.env.vision.enabled

    _muscle_names = _get_muscle_names(config)

    # Wrap all muscle actuators defined in the XML so that mjlab marks the
    # entity as actuated and exposes valid ctrl indices for the arm muscles.
    articulation = EntityArticulationInfoCfg(
        actuators=(
            XmlActuatorCfg(
                target_names_expr=tuple(f"{name}_tendon" for name in _muscle_names),
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )

    # entity_cfg = EntityCfg(
    #     spec_fn=functools.partial(_universal_spec_fn, xml_file=_get_myoInteract_root() / config.env.model_path, config=config),
    #     articulation=articulation,
    # )
    entity_cfg = TaskEntityCfg(
        spec_fn=functools.partial(
            _universal_spec_fn,
            xml_file=_get_myoInteract_root() / config.env.model_path,
            config=config,
        ),
        articulation=articulation,
        base_config=config,
    )

    # Use a task-specific entity name to avoid collisions with other tasks.
    universal_entity_name = _UNIVERSAL_ENTITY_NAME

    scene_cfg = SceneCfg(
        num_envs=16 if play else config.rl.get("num_envs", 4096),
        entities={universal_entity_name: entity_cfg},
    )

    # Add vision based on an existing camera from the XML file
    if vision_enabled:
        camera_names = ["universal_robot/fixed-eye"]
        cam_kwargs = {
            "universal_robot/fixed-eye": {
                "height": 128,  # 500, #180, #120,
                "width": 128,  # 500, #240, #160,
            },
        }
        cam_types = (
            ["rgb", "depth"] if vision_enabled else []
        )  # TODO: infer from config
        shared_cam_kwargs = dict(
            data_types=cam_types,
            enabled_geom_groups=(0, 3),
            use_shadows=False,
            use_textures=True,
        )

        cam_terms = {}
        for cam_name in camera_names:
            cam_cfg = CameraSensorCfg(
                name=cam_name.split("/")[-1],
                camera_name=cam_name,
                **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
                **shared_cam_kwargs,
            )
            scene_cfg.sensors = (scene_cfg.sensors or ()) + (cam_cfg,)
            param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
            for cam_type in cam_types:
                if cam_type == "depth":
                    _param_kwargs = param_kwargs.copy()
                    _param_kwargs["cutoff_distance"] = 0.5
                    func = manipulation_mdp.camera_depth
                else:
                    _param_kwargs = param_kwargs
                    func = manipulation_mdp.camera_rgb
                cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
                    func=func, params=_param_kwargs
                )

    num_targets = config.env.task_config.targets.num_targets
    _t0 = config.env.task_config.targets.target_0
    targets = [
        getattr(config.env.task_config.targets, f"target_{i}", None) or _t0
        for i in range(num_targets)
    ]
    _targets_have_margin = any(getattr(t, "geom_margin", 0.0) > 0.0 for t in targets)
    target_origin_rel = _compute_target_origin(config)

    global _OBS_PROPRIO_ENTITY_CFG
    ee_pos_name = config.env.task_config.reach_settings.ee_site
    _OBS_PROPRIO_ENTITY_CFG = SceneEntityCfg(
        _UNIVERSAL_ENTITY_NAME,
        joint_names=_independent_joint_names,
        site_names=[ee_pos_name],
    )

    # Full observation vector matching MyoUniversalEnv.
    # Total: (nq-2) + nv + 2 + 4 + 2 + 1 + 6 + 1 + 80 + 80 + 80 + 80 = 403+ dims.
    # Each term function receives `env` and returns a (N, d) torch.Tensor.
    _obs_terms_complete = {
        "time": ObservationTermCfg(func=_time),
        "qpos": ObservationTermCfg(
            func=_qpos, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "qvel": ObservationTermCfg(
            func=_qvel, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "qacc": ObservationTermCfg(
            func=_qacc, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "act": ObservationTermCfg(
            func=_act, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        # "last_ctrl": ObservationTermCfg(func=_ctrl),
        "ee_pos": ObservationTermCfg(
            func=_sites_pos, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        # "sensor_data": ObservationTermCfg(func=_sensor_data),
        # "task_obs": ObservationTermCfg(func=SequentialTaskObservation, params={"task_config": config.env.task_config, "ctrl_dt": config.env.ctrl_dt}),
        "target_pos": ObservationTermCfg(
            func=_next_target_pos, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "target_size": ObservationTermCfg(
            func=_next_target_size, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "phase_progress": ObservationTermCfg(
            func=_phase_progress, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "dwell_fraction": ObservationTermCfg(
            func=_dwell_fraction, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "MR": ObservationTermCfg(
            func=_fatigue_MR, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "MF": ObservationTermCfg(
            func=_fatigue_MF, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "MA": ObservationTermCfg(
            func=_fatigue_MA, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "curriculum_difficulty": ObservationTermCfg(
            func=_curriculum_difficulty, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        # "vision_rgb": ObservationTermCfg(func=manipulation_mdp.camera_rgb, params={"sensor_name": "fixed_eye"}),
        # "vision_depth": ObservationTermCfg(func=manipulation_mdp.camera_depth, params={"sensor_name": "fixed_eye", "cutoff_distance": 0.5}),
    }

    if vision_enabled:
        observations = {
            "proprioception": ObservationGroupCfg(
                terms={k: v for k, v in _obs_terms_complete.items() if k in obs_keys},
            ),
            "vision_mono": ObservationGroupCfg(
                # terms = {"rgb": ObservationTermCfg(func=manipulation_mdp.camera_rgb, params={"sensor_name": "fixed_eye"}),
                #         "depth": ObservationTermCfg(func=manipulation_mdp.camera_depth, params={"sensor_name": "fixed_eye", "cutoff_distance": 0.5})},
                terms=cam_terms,
                enable_corruption=False,
                concatenate_dim=0,
                ## To stack histories of vision:  #TODO: fix
                # history_length=2, flatten_history_dim=False, concatenate_history_dim=True #TODO: infer from config
            ),
        }
    else:
        observations = {
            "proprioception": ObservationGroupCfg(
                terms={
                    k: v
                    for k, v in _obs_terms_complete.items()
                    if k in obs_keys + omni_keys
                },
                concatenate_terms=True,
            ),
        }

    noise_params = config.env.muscle_config.noise_params
    # Read new visited_success-buffer fields with defaults, same
    # old-config-compat rationale as the fatigue-buffer fields above.
    _vs_buffer_enabled = getattr(
        config.env.task_config, "visited_success_buffer_enabled", True
    )
    _vs_buffer_size = getattr(
        config.env.task_config, "visited_success_buffer_size", 100
    )
    _vs_snapshots_per_episode = getattr(
        config.env.task_config, "visited_success_snapshots_per_episode", 5
    )
    _vs_buffer_sampling = getattr(
        config.env.task_config, "visited_success_buffer_sampling", "uniform"
    )
    actions = {
        # MyoSuite's muscle action: sigmoid -> motor noise -> fatigue, one action per muscle in
        # MuJoCo actuator order (all actuators of these models are muscles).
        "muscles": MyoActionCfg(
            entity_name=universal_entity_name,
            # myoInteract policies are trained on unclipped actions (clip_actions=None) and use
            # values beyond [-1, 1], e.g. -3 for zero excitation: do not clip before the sigmoid.
            action_range=(-math.inf, math.inf),
            muscle_fatigue=config.env.muscle_config.fatigue_enabled,
            fatigue_reset_vec=_fatigue_reset_vec(config),
            fatigue_reset_random=bool(
                config.env.muscle_config.get("fatigue_reset_random", False)
            ),
            motor_noise=motor_noise_from_params(noise_params),
        ),
    }

    rewards = {
        "distance": RewardTermCfg(
            # func=_distance_reward,
            func=_sequential_distance_reward,
            params={
                "asset_cfg": _OBS_PROPRIO_ENTITY_CFG,
                "exponential_distance_reward": config.env.task_config.exponential_distance_reward,
                "distance_metric": config.env.task_config.distance_metric,
            },
            weight=config.env.task_config.weighted_reward_keys.get("distance", 0.0),
        ),
        "neural_effort": RewardTermCfg(
            func=_neural_effort,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get(
                "neural_effort", 0.0
            ),
        ),
        "jac_effort": RewardTermCfg(
            func=_jac_effort,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get("jac_effort", 0.0),
        ),
        "elbow_velocity_penalty": RewardTermCfg(
            func=_elbow_velocity_penalty,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get(
                "elbow_velocity_penalty", 0.0
            ),
        ),
        "pain_penalty": RewardTermCfg(
            func=_pain_penalty,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get("pain_penalty", 0.0),
        ),
        # Squared change of the actions between control steps (penalises chattering commands).
        "action_rate": RewardTermCfg(
            func=lambda env: -mdp_action_rate_l2(env),
            weight=config.env.task_config.weighted_reward_keys.get("action_rate", 0.0),
        ),
        "pronation": RewardTermCfg(
            func=_pronation_reward,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get("pronation", 0.0),
        ),
        "button_force": RewardTermCfg(
            func=_button_force_penalty,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get("button_force", 0.0),
        ),
        "fatigue_mean": RewardTermCfg(
            func=lambda *args, **kwargs: -_fatigue_MF(*args, **kwargs).mean(dim=-1),
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get("fatigue_mean", 0.0),
        ),
        "fatigue_max": RewardTermCfg(
            func=lambda *args, **kwargs: -_fatigue_MF(*args, **kwargs).max(dim=-1)[0],
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get("fatigue_max", 0.0),
        ),
        "phase_bonus": RewardTermCfg(
            func=_phase_bonus,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get("phase_bonus", 0.0),
        ),
        "done": RewardTermCfg(
            func=_phase_successfully_completed,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG, "phase_id": num_targets - 1},
            weight=config.env.task_config.weighted_reward_keys.get("done", 0.0),
        ),
    }
    if _torso_contact_penalty_enabled(config):
        # Conditional (unlike the terms above): depends on sensors that only exist
        # in the compiled model when add_torso_contact_sensor_to_spec was actually
        # called in _universal_spec_fn, gated on this same condition.
        rewards["torso_contact_penalty"] = RewardTermCfg(
            func=_torso_contact_penalty,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get(
                "torso_contact_penalty", 0.0
            ),
        )
    # Build reset_joints event based on reset_type config:
    #   epsilon_uniform              → reset_joints_by_offset with ±0.05 rad/rad·s⁻¹ (matches base.py)
    #   zero                         → reset_joints_by_offset with zero offset (fixed default pose)
    #   range_uniform                → uniform sample across full joint range, ±0.05 rad·s⁻¹ vel
    #   initpose001                  → fixed "ready to interact" pose + muscle activation
    #                                  (mobl_arms_index v2.1/v2.2 only; see _INITPOSE001_QPOS)
    #   initpose001_epsilon_uniform  → epsilon-ball around initpose001's qpos+act, not
    #                                  default_joint_pos/zero-act
    #   visited_success              → sample qpos uniformly from a buffer of states
    #                                  visited near the end of successful episodes
    #                                  (mobl_arms_index v2.1/v2.2 only; see
    #                                  VisitedSuccessBuffer/VisitedSuccessReset)
    #   None                         → no joint reset across episodes
    _reset_type = config.env.task_config.reset_type
    _joint_reset_events: dict = {}
    if _reset_type == "epsilon_uniform":
        _joint_reset_events["reset_joints"] = EventTermCfg(
            func=event_fns.reset_joints_by_offset,
            params={
                "asset_cfg": _OBS_PROPRIO_ENTITY_CFG,
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.1, 0.1),
            },
            mode="reset",
        )
    elif _reset_type == "zero":
        _joint_reset_events["reset_joints"] = EventTermCfg(
            func=event_fns.reset_joints_by_offset,
            params={
                "asset_cfg": _OBS_PROPRIO_ENTITY_CFG,
                "position_range": (0, 0),
                "velocity_range": (0, 0),
            },
            mode="reset",
        )
    elif _reset_type == "range_uniform":
        _joint_reset_events["reset_joints"] = EventTermCfg(
            func=_reset_joints_range_uniform,
            params={
                "asset_cfg": _OBS_PROPRIO_ENTITY_CFG,
                "velocity_range": (-0.1, 0.1),
            },
            mode="reset",
        )
    elif _reset_type == "initpose001":
        # Deliberately NOT _OBS_PROPRIO_ENTITY_CFG (filtered to _independent_joint_names
        # above) -- the fixed pose covers every joint in the chain.
        _joint_reset_events["reset_joints"] = EventTermCfg(
            func=_reset_joints_initpose001,
            params={"asset_cfg": SceneEntityCfg(_UNIVERSAL_ENTITY_NAME)},
            mode="reset",
        )
    elif _reset_type == "initpose001_epsilon_uniform":
        _joint_reset_events["reset_joints"] = EventTermCfg(
            func=_reset_joints_initpose001_epsilon_uniform,
            params={"asset_cfg": SceneEntityCfg(_UNIVERSAL_ENTITY_NAME)},
            mode="reset",
        )
    elif _reset_type == "visited_success":
        # Deliberately NOT _OBS_PROPRIO_ENTITY_CFG, same reasoning as initpose001
        # above -- the buffer's rows (and its initpose001 fallback) cover every
        # joint in the chain.
        _joint_reset_events["reset_joints"] = EventTermCfg(
            func=VisitedSuccessReset,
            params={"asset_cfg": SceneEntityCfg(_UNIVERSAL_ENTITY_NAME)},
            mode="reset",
        )

    # reset_type == "None" or None: no reset_joints event

    # Adaptive curriculum (CurriculumConfig): when enabled, initial-state placement
    # is driven entirely by CurriculumInitialState (deterministic initpose001 /
    # posture-buffer / persist branches, see its docstring), overriding whatever
    # reset_joints event the reset_type chain above selected.
    _curriculum_active = _curriculum_enabled(config)
    _curr_cfg = config.env.curriculum_config if _curriculum_active else None
    if _curriculum_active:
        _joint_reset_events["reset_joints"] = EventTermCfg(
            func=CurriculumInitialState,
            params={"asset_cfg": SceneEntityCfg(_UNIVERSAL_ENTITY_NAME)},
            mode="reset",
        )

    if _curriculum_active and num_targets != 1:
        raise ValueError(
            "CurriculumConfig.enabled=True currently only supports single-target "
            f"tasks (CurriculumTargetSync/_build_target_compartments), got {num_targets} targets."
        )
    events = {
        **_joint_reset_events,
        # Initialize sequence state before observation shape inference. Native
        # termination computation advances it before rewards/metrics each step.
        "task_logic_update": EventTermCfg(
            func=SequentialTaskLogic,
            mode="reset",
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
        ),
        # Target location update
        "target_pos_dr": (
            EventTermCfg(mode="reset", func=CurriculumTargetSync, params={})
            if _curriculum_active
            else EventTermCfg(
                mode="reset",
                func=dr.body_pos,
                params={
                    "asset_cfg": SceneEntityCfg(
                        _UNIVERSAL_ENTITY_NAME, body_names=["body_target_.*"]
                    ),
                    # "ranges": {".*knee.*": (0.5, 1.5), ".*hip.*": (0.8, 1.2)},
                    "ranges": {
                        f"body_target_{i}": {
                            xyz: (
                                targets[i].position[0][xyz] + target_origin_rel[xyz],
                                targets[i].position[1][xyz] + target_origin_rel[xyz],
                            )
                            for xyz in range(3)
                        }
                        for i in range(num_targets)
                    },
                    "operation": "abs",
                    # "ranges": {f"body_target_{i}": {xyz: (0, 0) for xyz in range(3)} for i in range(num_targets)},
                },
            )
        ),
        # Target size update
        "target_size_dr": EventTermCfg(
            mode="reset",
            func=dr.geom_size,
            params={
                "asset_cfg": SceneEntityCfg(
                    _UNIVERSAL_ENTITY_NAME, geom_names=["geom_target_.*"]
                ),
                # "axes": [0],
                "ranges": {
                    f"geom_target_{i}": {
                        xyz: (
                            targets[i].size[0]
                            if isinstance(targets[i].size[0], (int, float))
                            else targets[i].size[0][xyz],
                            targets[i].size[1]
                            if isinstance(targets[i].size[1], (int, float))
                            else targets[i].size[1][xyz],
                        )
                        for xyz in range(3)
                    }
                    for i in range(num_targets)
                },
                "operation": "abs",
            },
        ),
    }
    if _curriculum_active:
        # is_global_time=True: a shared timer immune to individual envs' episode
        # boundaries.
        events["posture_buffer_capture"] = EventTermCfg(
            func=PostureBufferCapture,
            mode="interval",
            is_global_time=True,
            params={"capture_fraction": _curr_cfg.posture_capture_fraction},
            interval_range_s=(
                _curr_cfg.posture_capture_interval_s,
                _curr_cfg.posture_capture_interval_s,
            ),
        )

    if _reset_type == "visited_success" and _vs_buffer_enabled:
        # is_global_time left at its default False (unlike the fatigue-buffer
        # capture above): each env's timer must restart at that env's own
        # episode reset, so ticks land at temporally-equidistant points WITHIN
        # each episode -- see VisitedSuccessCapture's docstring.
        events["visited_success_capture"] = EventTermCfg(
            func=VisitedSuccessCapture,
            mode="interval",
            params={
                "n_snapshots_per_episode": _vs_snapshots_per_episode,
                "max_duration": config.env.task_config.max_duration,
                "buffer_size": _vs_buffer_size,
                "sampling": _vs_buffer_sampling,
                "muscle_names": _muscle_names,
            },
            interval_range_s=(
                config.env.task_config.max_duration / max(1, _vs_snapshots_per_episode),
                config.env.task_config.max_duration / max(1, _vs_snapshots_per_episode),
            ),
        )

    terminations = {
        "time_out": TerminationTermCfg(
            func=_sequential_task_time_out,
            time_out=True,
        ),
        "episode_success": TerminationTermCfg(
            func=SequentialTaskSuccess,
        ),
    }

    metrics = {
        **{
            f"phase_{i}_success": MetricsTermCfg(
                reduce="last",
                func=_phase_successfully_completed,
                params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG, "phase_id": i},
            )
            for i in (
                range(num_targets)
                if num_targets <= 20
                else range(0, num_targets, num_targets // 20)
            )
        },
        "inside_target": MetricsTermCfg(
            func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].inside_target[:, 0]
        ),
        # "button_touch_force": MetricsTermCfg(func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].button_touch_force[:, 0]),
        # "distance_to_next_target": MetricsTermCfg(func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].distance_to_target[:, 0]),
        "total_initial_distance": MetricsTermCfg(
            func=lambda env: env.scene[
                _UNIVERSAL_ENTITY_NAME
            ].between_target_distances.sum(dim=-1)
        ),
        "target_size": MetricsTermCfg(
            func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].target_size[:, 0]
        ),
        "current_target_id": MetricsTermCfg(
            func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].current_target_id[:, 0]
        ),
        "fingertip_height": MetricsTermCfg(
            func=_sites_pos_z, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "control_peaks": MetricsTermCfg(
            func=lambda env: torch.abs(
                env.scene[_UNIVERSAL_ENTITY_NAME].data.tendon_effort_target - 0.5
            ).mean(dim=-1),
        ),
        "average_muscle_fatigue": MetricsTermCfg(
            func=lambda env: (
                fatigue.MF.mean(dim=-1)
                if (fatigue := _muscles_fatigue(env)) is not None
                else torch.zeros(env.num_envs, device=env.device)
            )
        ),
        "max_muscle_fatigue": MetricsTermCfg(
            func=lambda env: (
                fatigue.MF.max(dim=-1).values
                if (fatigue := _muscles_fatigue(env)) is not None
                else torch.zeros(env.num_envs, device=env.device)
            )
        ),
    }

    curriculum = {}
    if _curriculum_active:
        # Absolute world-space bounds of target_0's configured position range,
        # same origin-shift as target_pos_dr's own "ranges" above -- the
        # compartment grid tiles exactly this box.
        box_lo = [
            targets[0].position[0][xyz] + target_origin_rel[xyz] for xyz in range(3)
        ]
        box_hi = [
            targets[0].position[1][xyz] + target_origin_rel[xyz] for xyz in range(3)
        ]
        curriculum["compartments"] = CurriculumTermCfg(
            func=CurriculumState,
            params={
                "box_lo": box_lo,
                "box_hi": box_hi,
                "n_x": _curr_cfg.n_x,
                "n_z": _curr_cfg.n_z,
                "ema_decay": _curr_cfg.ema_decay,
                "weight_floor": _curr_cfg.weight_floor,
                "eval_frac": _curr_cfg.eval_frac,
                "ready_base_prob": _curr_cfg.ready_base_prob,
                "ready_floor_prob": _curr_cfg.ready_floor_prob,
                "ready_decay_horizon_steps": _curr_cfg.ready_decay_horizon_steps,
                "difficulty_boost": _curr_cfg.difficulty_boost,
                "posture_split": _curr_cfg.posture_split,
                "posture_buffer_size": _curr_cfg.posture_buffer_size,
                "pain_relax_factor": _curr_cfg.pain_relax_factor,
            },
        )

    return ManagerBasedRlEnvCfg(
        decimation=int(round(config.env.ctrl_dt / config.env.sim_dt)),
        scene=scene_cfg,
        observations=observations,
        actions=actions,
        events=events,
        sim=SimulationCfg(
            # Constraint rows per world: MuJoCo Warp's default is too small for the MyoArm models
            # (equality-coupled shoulder joints, limits, contacts), whose overflow returns NaN.
            njmax=config.env.get("njmax", None)
            or (
                _MYOARM_NJMAX if _myoarm_r_variant_for(config.env.model_path) else None
            ),
            mujoco=MujocoCfg(
                timestep=config.env.sim_dt,
                # MuJoCo Warp only supports multi-contact CCD for geoms without margin,
                # and button targets carry a small contact margin.
                disableflags=("multiccd",) if _targets_have_margin else (),
            ),
        ),
        episode_length_s=config.env.task_config.max_duration,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
    )


def _universal_ppo_runner_cfg(config: Config) -> RslRlOnPolicyRunnerCfg:
    """Minimal PPO runner config for benchmarking."""

    vision_enabled = config.env.vision.enabled

    model_class = "MLPModel"
    if vision_enabled:
        model_class = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"

    num_steps_per_env = config.rl.get("unroll_length", 10)
    max_iterations = config.rl.num_timesteps // (config.rl.num_envs * num_steps_per_env)

    # Exploration/curriculum coupling (CurriculumConfig.exploration_coupling): swap
    # the default global, state-independent std for a state-dependent std head, so
    # the policy can learn to raise exploration for harder/under-performing
    # compartments (see the "curriculum_difficulty" observation term) rather than
    # using a single fixed std everywhere. No rsl_rl internals touched -- this is
    # a stock sibling distribution class, just a config swap.
    _curr_cfg = config.env.get("curriculum_config", None)
    _exploration_coupling = (
        bool(_curr_cfg.exploration_coupling) if _curr_cfg is not None else False
    )
    distribution_cfg = (
        {
            "class_name": "HeteroscedasticGaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }
        if _exploration_coupling
        else {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }
    )

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128, 128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=distribution_cfg,
            cnn_cfg={
                "output_channels": [16, 64, 64, 128],
                "kernel_size": [8, 4, 3, 3],
                "stride": [4, 2, 1, 1],
                "dilation": [1, 1, 1, 1],
                "padding": "none",
                "norm": ["layer", "layer", "layer", "layer"],
                "activation": "elu",
                "max_pool": [False, False, False, False],
                "global_pool": "avg",
                "flatten": True,
                "spatial_softmax": True,
                "spatial_softmax_temperature": 0.2,
            }
            if vision_enabled
            else None,
            class_name=model_class,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256, 256, 256),
            activation="elu",
            obs_normalization=True,
            cnn_cfg={
                "output_channels": [16, 64, 64, 128],
                "kernel_size": [8, 4, 3, 3],
                "stride": [4, 2, 1, 1],
                "dilation": [1, 1, 1, 1],
                "padding": "none",
                "norm": ["layer", "layer", "layer", "layer"],
                "activation": "elu",
                "max_pool": [False, False, False, False],
                "global_pool": "avg",
                "flatten": True,
                "spatial_softmax": True,
                "spatial_softmax_temperature": 0.2,
            }
            if vision_enabled
            else None,
            class_name=model_class,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=config.rl.get("num_epochs_per_update", 8),
            num_mini_batches=config.rl.get("num_minibatches", 8),
            learning_rate=config.rl.get("learning_rate", 3e-4),
            schedule="adaptive",
            gamma=config.rl.get("discounting", 0.97),
            lam=0.95,
            entropy_coef=config.rl.get("entropy_cost", 0.001),
            desired_kl=config.rl.get("desired_kl", 0.01),
            max_grad_norm=config.rl.get("max_grad_norm", 1.0),
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=config.rl.get("clipping_epsilon", 0.3),
        ),
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
        obs_groups={
            "actor": ("proprioception", "vision_mono")
            if vision_enabled
            else ("proprioception",),
            "critic": ("proprioception", "vision_mono")
            if vision_enabled
            else ("proprioception",),
        },
        save_interval=config.rl.get("save_interval", 30),
        experiment_name=config.rl.get("experiment_name", "myo_universal"),
        # Local tensorboard logs by default; "wandb" needs a prior wandb.init() / login.
        logger=os.environ.get("MYOINTERACT_LOGGER", "tensorboard"),
        wandb_project=os.environ.get(
            "WANDB_PROJECT", "mjlab"
        ),  # Studio sets the project of its W&B runs
    )


def register_mjlab_myointeract_task(config: Config) -> None:
    """Register MyoInteract env ids with mjlab.tasks.registry. Idempotent (replaces existing entry)."""

    # --- Universal Task ---
    xml_file = _get_myoInteract_root() / config.env.model_path
    # The MyoArm models are composed from myo_sim by name; their model_path is a sentinel, not a file.
    if xml_file.exists() or _myoarm_r_variant_for(config.env.model_path) is not None:
        # try:
        env_cfg = _make_universal_env_cfg(config, play=False)
        play_cfg = _make_universal_env_cfg(config, play=True)
        rl_cfg = _universal_ppo_runner_cfg(config)
        _MJLAB_REGISTRY.pop("myoInteractUniversal-v0", None)
        register_mjlab_task(
            task_id="myoInteractUniversal-v0",
            env_cfg=env_cfg,
            play_env_cfg=play_cfg,
            rl_cfg=rl_cfg,
            runner_cls=OnnxCheckpointingMjlabRunner,
        )
    else:
        raise FileNotFoundError(
            f"Model XML file not found at {xml_file}. Please check the path in the config and ensure the file exists."
        )

    # When reset_type is None, sim.reset() would still overwrite qpos with qpos0
    # (via mjwarp.reset_data).  Patch ManagerBasedRlEnv.__init__ so that every env
    # created from this task config gets an instance-level sim.reset override that
    # saves and restores qpos around the physics reset.  The guard on
    # "universal_robot" scopes the patch to this task only.  An idempotency flag
    # prevents double-chaining if register_mjlab_myointeract_task is called more than once.
    if config.env.task_config.reset_type in ("None", None):
        if not getattr(ManagerBasedRlEnv.__init__, "_preserve_qpos_patched", False):
            _orig_env_init = ManagerBasedRlEnv.__init__

            def _env_init_preserve_qpos(self, *args, _orig=_orig_env_init, **kwargs):
                _orig(self, *args, **kwargs)
                if "universal_robot" not in self.scene._entities:
                    return
                _orig_sim_reset = self.sim.reset

                def _sim_reset_preserve_qpos(
                    env_ids=None, *, _env=self, _orig_reset=_orig_sim_reset
                ):
                    entity = _env.scene["universal_robot"]
                    ids = slice(None) if env_ids is None else env_ids
                    saved_qpos = entity.data.joint_pos[ids].clone()
                    _orig_reset(env_ids)
                    entity.write_joint_state_to_sim(
                        saved_qpos,
                        torch.zeros_like(saved_qpos),
                        env_ids=env_ids,
                    )

                self.sim.reset = _sim_reset_preserve_qpos

            _env_init_preserve_qpos._preserve_qpos_patched = True
            ManagerBasedRlEnv.__init__ = _env_init_preserve_qpos
