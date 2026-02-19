""" =================================================
# Copyright (c) User-in-the-Box 2024; Facebook, Inc. and its affiliates
Authors  :: Florian Fischer (fjf33@cam.ac.uk); Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

from typing import Any, Callable, Dict, List, Sequence, Optional, Union
import abc
import tqdm

# from myosuite.utils import gym
import mujoco
import numpy as np

import jax
from jax import numpy as jp
from mujoco import mjx

from mujoco_playground import State
from mujoco_playground._src import mjx_env
from ml_collections import config_dict
from myosuite.envs.myo.fatigue import CumulativeFatigue
from typing import Union
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple
from omegaconf import MISSING, OmegaConf
from copy import deepcopy

OmegaConf.register_new_resolver("int_divide", lambda x, y: int(x / y))

@dataclass
class NoiseParams:
    sigdepnoise_type: Union[str, None] = None
    sigdepnoise_level: float = 0.103
    constantnoise_type: Union[str, None] = None
    constantnoise_level: float = 0.185

@dataclass
class MuscleConfig:
    muscle_condition: Union[str, None] = None
    sex: Union[str, None] = None
    control_type: str = "default"
    noise_params: NoiseParams = field(default_factory=lambda: NoiseParams())

@dataclass
class BaseEnvConfig:
    env_name: str = MISSING
    model_path: str = MISSING
    ctrl_dt: float = 0.002 * 25
    sim_dt: float = 0.002
    muscle_config: MuscleConfig = field(default_factory=lambda: MuscleConfig())
    eval_mode: bool = False
    
def get_default_config():
    return config_dict.create(
        model_path="myosuite/simhive/uitb_sim/mobl_arms_index_eepos_pointing.xml",
        ctrl_dt=0.002 * 25,  # Each control step is 25 physics steps
        sim_dt=0.002,
        vision_mode='',
        vision=config_dict.create(
            # vision_mode="rgbd",
            gpu_id=0,
            render_batch_size=1024,
            num_worlds=1024,
            render_width=120,
            render_height=120,
            enabled_geom_groups=[0, 1, 2],
            enabled_cameras=[0],
            use_rasterizer=False,
        ),
        muscle_config=config_dict.create(
            muscle_condition=None,
            sex=None,
            control_type="default",   #"relative"
            noise_params=config_dict.create(
                sigdepnoise_type=None,
                sigdepnoise_level=0.103,
                constantnoise_type=None,
                constantnoise_level=0.185,
            ),
        ),
        task_config=config_dict.create(
            max_trials=1,  # num of trials per episode
            reset_type="range_uniform",
        ),
        eval_mode=False
        # episode_length=80,
    )

ALLOWED_VISION_MODES = ("rgb", "depth", "rgbd", "rgb+depth", "rgbd_only", "depth_only", "depth_w_aux_task")
ALLOWED_MUSCLE_CONDITIONS = ("sarcopenia", "fatigue", "reafferentation", None)
ALLOWED_RESET_TYPES = ("zero", "epsilon_uniform", "range_uniform", None)


class MyoUserBase(mjx_env.MjxEnv):
    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def action_size(self) -> int:
        return self._nu

    @property
    def xml_path(self) -> str:
        return self._config.model_path
    
    def __init__(
        self,
        config: config_dict.ConfigDict = get_default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(deepcopy(config), config_overrides)
        self.eval_mode = self._config.eval_mode

        self._prepare_mjx_model()
        self._prepare_env()
        self._prepare_vision()   

    def preprocess_spec(self, spec:mujoco.MjSpec):
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.conaffinity = 0
                geom.contype = 0
                print(f"Disabled contacts for cylinder geom named \"{geom.name}\"")
        return spec
    
    def modify_mj_model(self, mj_model):
        """Allows task specific modifications to the mujoco model before it is compiled!"""
        return mj_model

    def _prepare_mjx_model(self):
        spec = mujoco.MjSpec.from_file(self.xml_path)
        spec = self.preprocess_spec(spec)

        mj_model = spec.compile()

        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        # mj_model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        mj_model.opt.disableflags = mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP
        mj_model.opt.timestep = self._config.sim_dt
        mj_model = self.modify_mj_model(mj_model)
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(self._mj_model)

    def _prepare_env(self):
        self._prepare_bm_model()
        self._initialize_muscle_conditions()
        self._setup()

        # Do a forward step so stuff like geom and body positions are calculated
        # [using MjData rather than mjx.Data, to reduce computational overheat]
        _data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, _data)

        self._prepare_after_init(_data)

    def _prepare_bm_model(self):
        # Total number of actuators
        self._nu = self._mjx_model.nu

        # Number of muscle actuators
        self._na = self._mjx_model.na

        # Number of motor actuators
        self._nm = self._nu - self._na
        # self._motor_act = jp.zeros((self._nm,))
        # self._motor_alpha = 0.9 * jp.ones(1)

        # Get actuator names (muscle and motor)
        self._actuator_names = [
            mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self._mjx_model.nu)
        ]
        self._muscle_actuator_names = set(
            np.array(self._actuator_names)[
                self._mjx_model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON
            ]
        )  # model.actuator_dyntype==mujoco.mjtDyn.mjDYN_MUSCLE
        self._motor_actuator_names = (
            set(self._actuator_names) - self._muscle_actuator_names
        )

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._muscle_actuator_names = sorted(
            self._muscle_actuator_names, key=self._actuator_names.index
        )
        self._motor_actuator_names = sorted(
            self._motor_actuator_names, key=self._actuator_names.index
        )

        # Find actuator indices in the simulation
        self._muscle_actuators = jp.array(
            [
                mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                for actuator_name in self._muscle_actuator_names
            ],
            dtype=jp.int32,
        )
        self._motor_actuators = jp.array(
            [
                mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                for actuator_name in self._motor_actuator_names
            ],
            dtype=jp.int32,
        )

        # Get joint names (dependent and independent)
        self._joint_names = [
            mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self._mjx_model.njnt)
        ]
        self._dependent_joint_names = (
            {
                self._joint_names[idx]
                for idx in np.unique(
                    self._mjx_model.eq_obj1id[self._mjx_model.eq_active0.astype(bool)]
                )
            }
            if self._mjx_model.eq_obj1id is not None
            else set()
        )
        self._independent_joint_names = (
            set(self._joint_names) - self._dependent_joint_names
        )

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._dependent_joint_names = sorted(
            self._dependent_joint_names, key=self._joint_names.index
        )
        self._independent_joint_names = sorted(
            self._independent_joint_names, key=self._joint_names.index
        )

        # Find dependent and independent joint indices in the simulation
        self._dependent_joints = jp.array([
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in self._dependent_joint_names
        ], dtype=jp.int32)
        self._independent_joints = jp.array([
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in self._independent_joint_names
        ], dtype=jp.int32)

        # If there are 'free' type of joints, we'll need to be more careful with which dof corresponds to
        # which joint, for both qpos and qvel/qacc. There should be exactly one dof per independent/dependent joint.
        def get_dofs(joint_indices):
            qpos = jp.array([], dtype=jp.int32)
            dofs = jp.array([], dtype=jp.int32)
            for joint_idx in joint_indices:
                if self._mjx_model.jnt_type[joint_idx] not in [
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ]:
                    raise NotImplementedError(
                        f"Only 'hinge' and 'slide' joints are supported, joint "
                        f"{self._joint_names[joint_idx]} is of type {mujoco.mjtJoint(self._mjx_model.jnt_type[joint_idx]).name}"
                    )
                qpos = jp.append(qpos, self._mjx_model.jnt_qposadr[joint_idx])
                dofs = jp.append(dofs, self._mjx_model.jnt_dofadr[joint_idx])
            return qpos, dofs

        self._dependent_qpos, self._dependent_dofs = get_dofs(self._dependent_joints)
        self._independent_qpos, self._independent_dofs = get_dofs(
            self._independent_joints
        )

    def _initialize_muscle_conditions(self):
        # initialize muscle properties and conditions

        self.muscle_condition = self._config.muscle_config.muscle_condition
        self.sex = self._config.muscle_config.sex
        self.control_type = self._config.muscle_config.control_type
        self.muscle_noise_params = self._config.muscle_config.noise_params

        ## valid muscle conditions: 
        valid_muscle_conditions = ALLOWED_MUSCLE_CONDITIONS
        assert self.muscle_condition in valid_muscle_conditions, f"Invalid muscle condition '{self.muscle_condition} (valid conditions are {valid_muscle_conditions})."

        if self.muscle_condition == "sarcopenia":
            for mus_idx in range(self._mjx_model.actuator_gainprm.shape[0]):
                self._mjx_model.actuator_gainprm[mus_idx, 2] = (
                    0.5 * self._mjx_model.actuator_gainprm[mus_idx, 2].copy()
                )

        # for muscle fatigue we used the 3CC-r model
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self._mj_model,
                frame_skip=self.n_substeps,
                sex=self.sex,
                seed=self.get_input_seed(),
            )

        # Tendon transfer to redirect EIP --> EPL
        # https://www.assh.org/handcare/condition/tendon-transfer-surgery
        elif self.muscle_condition == "reafferentation":
            self.EPLpos = self._mj_model.actuator_name2id("EPL")
            self.EIPpos = self._mj_model.actuator_name2id("EIP")
    
    def _setup(self):
        """Task specific setup"""
        self.max_trials = self._config.task_config.max_trials
        self.reset_type = self._config.task_config.reset_type

        ## valid reset types: 
        valid_reset_types = ALLOWED_RESET_TYPES
        assert self.reset_type in valid_reset_types, f"Invalid reset type '{self.reset_type} (valid types are {valid_reset_types})."
    
    def _prepare_after_init(self, data):
        """Task specific after init"""
        pass
    
    def _prepare_vision(self):
        if not self._config.vision.enabled:
            self.vision = False
            return
        self.vision = True
        from madrona_mjx.renderer import BatchRenderer

        self.vision_mode = self._config.vision.vision_mode
        assert (
            self.vision_mode in ALLOWED_VISION_MODES
        ), f"Invalid vision mode: {self.vision_mode} (allowed modes: {ALLOWED_VISION_MODES})"
        enabled_cameras = self._config.vision.enabled_cameras
        assert len(enabled_cameras) == 1, "Only one camera is supported for now"
        if self._mjx_model.ncam > 1:
            print(f"Ensuring that all cameras have the same fovy as the chosen camera: {enabled_cameras[0]}")
            cam_fovy = self._mjx_model.cam_fovy
            print(f"Initial cam_fovy: {cam_fovy}")
            relevant_cam_fovy = cam_fovy[enabled_cameras[0]]
            print(f"Camera: {enabled_cameras[0]}, relevant_cam_fovy: {relevant_cam_fovy}")
            cam_fovy = cam_fovy * 0 + relevant_cam_fovy
            self._mjx_model = self._mjx_model.replace(cam_fovy=cam_fovy)
            print(f"Final cam_fovy: {self._mjx_model.cam_fovy}")
        self.batch_renderer = BatchRenderer(
            m=self._mjx_model,
            gpu_id=self._config.vision.gpu_id,
            num_worlds=self._config.vision.num_worlds,
            batch_render_view_width=self._config.vision.render_width,
            batch_render_view_height=self._config.vision.render_height,
            enabled_geom_groups=np.asarray(self._config.vision.enabled_geom_groups),
            enabled_cameras=np.asarray(self._config.vision.enabled_cameras),
            use_rasterizer=False,
            viz_gpu_hdls=None,
            add_cam_debug_geo=False,
        )
    
    def enable_eval_mode(self):
        # TODO: eval wrapper should call this function at initialization
        self.eval_mode = True

    def disable_eval_mode(self):
        self.eval_mode = False

    @abc.abstractmethod    
    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""

    @abc.abstractmethod
    def reset(self, rng, **kwargs):
        """Reset function. Should call at least self._reset_bm_model."""

    @abc.abstractmethod
    def auto_reset(self, rng, info_before_reset, **kwargs):
        """Reset function wrapper called by AutoResetWrapper.
        Can pass information from previous info to reset function."""
    
    def eval_reset(self, rng, eval_id, **kwargs):
        """Reset function wrapper called by evaluate_policy."""
        return self.reset(rng, **kwargs)
    
    def prepare_eval_rollout(self, rng, **kwargs):
        """Function that can be used to define random parameters to be used across multiple evaluation rollouts/resets.
        May return the number of evaluation episodes that should be rolled out (before this method should be called again)."""
        return None

    def _reset_bm_model(self, rng):
        # TODO: do not store anything in self in this function, as its values should mostly be discarded after it is called (no permanent env changes!)

        # Sample random initial values for motor activation
        # rng, rng1 = jax.random.split(rng, 2)
        # self._motor_act = jax.random.uniform(
        #     rng1,
        #     shape=(self._nm,),
        #     minval=jp.zeros((self._nm,)),
        #     maxval=jp.ones((self._nm,)),
        # )

        # Reset qpos/qvel/act
        if self.reset_type == "zero":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng)
        elif self.reset_type == "epsilon_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng)
            data = mjx_env.init(self.mjx_model, qpos=reset_qpos, qvel=reset_qvel, act=reset_act)
            reset_qpos, reset_qvel, reset_act = self._reset_epsilon_uniform(rng, data)
        elif self.reset_type == "range_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng)
            data = mjx_env.init(self.mjx_model, qpos=reset_qpos, qvel=reset_qvel, act=reset_act)
            reset_qpos, reset_qvel, reset_act = self._reset_range_uniform(rng, data)
        else:
            reset_qpos, reset_qvel, reset_act = None, None, None

        data = mjx_env.init(self.mjx_model, qpos=reset_qpos, qvel=reset_qvel, act=reset_act)

        # Reset muscle fatigue state
        if self.muscle_condition == "fatigue":
            self.muscle_fatigue.reset()

        # Reset accumulative noise
        zero = jp.zeros(1)
        self._sigdepnoise_acc = zero
        self._constantnoise_acc = zero

        return data

    def _reset_zero(self, rng):
        """Resets the biomechanical model."""

        # Set joint angles and velocities to zero
        rng, rng1 = jax.random.split(rng, 2)
        nqi = len(self._independent_qpos)
        qpos = jp.zeros((nqi,))
        qvel = jp.zeros((nqi,))
        reset_qpos = jp.zeros((self._mj_model.nq,))
        reset_qvel = jp.zeros((self._mj_model.nv,))

        # Randomly sample act within unit interval
        reset_act = jax.random.uniform(
            rng1,
            shape=self._na,
            minval=jp.zeros((self._na,)),
            maxval=jp.ones((self._na,)),
        )

        zero = jp.zeros(1)

        # Set qpos and qvel
        reset_qpos = reset_qpos.at[self._dependent_qpos].set(zero)
        reset_qpos = reset_qpos.at[self._independent_qpos].set(qpos)
        reset_qvel = reset_qvel.at[self._dependent_dofs].set(zero)
        reset_qvel = reset_qvel.at[self._independent_dofs].set(qvel)

        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel, reset_act

    def _reset_epsilon_uniform(self, rng, data):
        """Resets the biomechanical model."""

        # Randomly sample qpos and qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        qpos = jax.random.uniform(
            rng1,
            shape=nqi,
            minval=jp.ones((nqi,)) * -0.05,
            maxval=jp.ones((nqi,)) * 0.05,
        )
        qvel = jax.random.uniform(
            rng2,
            shape=nqi,
            minval=jp.ones((nqi,)) * -0.05,
            maxval=jp.ones((nqi,)) * 0.05,
        )
        reset_qpos = jp.zeros((self._mj_model.nq,))
        reset_qvel = jp.zeros((self._mj_model.nv,))
        reset_act = jax.random.uniform(
            rng3,
            shape=self._na,
            minval=jp.zeros((self._na,)),
            maxval=jp.ones((self._na,)),
        )

        zero = jp.zeros(1)

        # Set qpos and qvel
        reset_qpos = reset_qpos.at[self._dependent_qpos].set(zero)
        reset_qpos = reset_qpos.at[self._independent_qpos].set(qpos)
        reset_qvel = reset_qvel.at[self._dependent_dofs].set(zero)
        reset_qvel = reset_qvel.at[self._independent_dofs].set(qvel)

        # Ensure that constraints are initially satisfied
        reset_qpos = self.ensure_dependent_joint_angles(data, reset_qpos)

        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel, reset_act

    def _reset_range_uniform(self, rng, data):
        """Resets the biomechanical model."""

        # Randomly sample qpos within joint range, qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        jnt_range = self._mj_model.jnt_range[self._independent_joints]
        qpos = jax.random.uniform(
            rng1, shape=(nqi,), minval=jnt_range[:, 0], maxval=jnt_range[:, 1]
        )
        qvel = jax.random.uniform(
            rng2,
            shape=(nqi,),
            minval=jp.ones((nqi,)) * -0.05,
            maxval=jp.ones((nqi,)) * 0.05,
        )
        reset_qpos = jp.zeros((self._mj_model.nq,))
        reset_qvel = jp.zeros((self._mj_model.nv,))
        reset_act = jax.random.uniform(
            rng3,
            shape=self._na,
            minval=jp.zeros((self._na,)),
            maxval=jp.ones((self._na,)),
        )

        # Set qpos and qvel
        reset_qpos = reset_qpos.at[self._independent_qpos].set(qpos)
        # reset_qpos[self._dependent_qpos] = 0
        reset_qvel = reset_qvel.at[self._independent_dofs].set(qvel)
        # reset_qvel[self._dependent_dofs] = 0
        
        # Ensure that constraints are initially satisfied
        reset_qpos = self.ensure_dependent_joint_angles(data, reset_qpos)

        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel, reset_act

    def ensure_dependent_joint_angles(self, data, reset_qpos):
        """ Adjusts virtual joints according to active joint constraints. """
        _joint_constraints = self.mjx_model.eq_type == 2
        _active_eq_constraints = data.eq_active == 1

        if False:  #default MuJoCo joint equality constraint
            eq_dep, eq_indep, poly_coefs = jp.array(self.mjx_model.eq_obj1id), \
                jp.array(self.mjx_model.eq_obj2id), \
                jp.array(self.mjx_model.eq_data[:, 4::-1])
            
            reset_qpos_new = jp.where(jp.array([jp.any((eq_dep == i) & _joint_constraints & _active_eq_constraints) for i in range(self.mjx_model.njnt)]), 
                                    jp.array([jp.polyval(poly_coefs[jp.argwhere(eq_dep == i, size=1).flatten(), :].flatten(), reset_qpos[eq_indep[jp.argwhere(eq_dep == i, size=1).flatten()]]) for i in range(self.mjx_model.njnt)]).flatten(),
                                    reset_qpos)
        else:  #patch that allows for two joint dependencies, but restricts constraints to polynomials of order three 
            # NEW: the equality constraint now ensures joint(joint2).qpos=polycoef[0] + polycoef[1]*joint(joint1).qpos + polycoef[2]*(joint(joint1).qpos**2) + polycoef[3]*(joint(polycoef[4]).qpos)*joint(joint1).qpos
            # BEFORE (MuJoCo default): 
            eq_dep, eq_indep, poly_coefs = jp.array(self.mjx_model.eq_obj1id), \
                jp.array(self.mjx_model.eq_obj2id), \
                jp.array(self.mjx_model.eq_data[:, 2::-1])
            eq_indep2 = jp.array(self.mjx_model.eq_data[:, 4], dtype=jp.int32)
            linear_coef_indep2 = jp.array(self.mjx_model.eq_data[:, 3] * (eq_indep2 > 0))

            reset_qpos_new = jp.where(jp.array([jp.any((eq_dep == i) & _joint_constraints & _active_eq_constraints) for i in range(self.mjx_model.njnt)]), 
                                    jp.array([jp.polyval(poly_coefs[jp.argwhere(eq_dep == i, size=1).flatten(), :].flatten(), reset_qpos[eq_indep[jp.argwhere(eq_dep == i, size=1).flatten()]]) + jp.dot(linear_coef_indep2[jp.argwhere(eq_dep == i, size=1).flatten()].flatten(), reset_qpos[eq_indep2[jp.argwhere(eq_dep == i, size=1).flatten()]]) for i in range(self.mjx_model.njnt)]).flatten(),
                                    reset_qpos)
        
        return reset_qpos_new
    
    def get_ctrl(self, state: State, action: jp.ndarray, rng: jp.ndarray):
        new_ctrl = action.copy()
        
        if self.control_type == "relative":
            _selected_motor_control = jp.clip(state.data.act[self._motor_actuators] + action[:self._nm], 0, 1)
            _selected_muscle_control = jp.clip(state.data.act[self._muscle_actuators] + action[self._nm:], 0, 1)
        elif self.control_type == "default":
            _selected_motor_control = jp.clip(action[:self._nm], 0, 1)
            _selected_muscle_control = jp.clip(action[self._nm:], 0, 1)
        else:
            raise NotImplementedError(f"Control type {self.control_type} is not valid; valid types are 'relative' and 'default'")

        if self.muscle_noise_params.sigdepnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.muscle_noise_params.sigdepnoise_type == "white":
                _added_noise = self.muscle_noise_params.sigdepnoise_level*_selected_muscle_control*_noise
                _selected_muscle_control += _added_noise
            elif self.muscle_noise_params.sigdepnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.muscle_noise_params.sigdepnoise_level*_selected_muscle_control*_noise
            elif self.muscle_noise_params.sigdepnoise_type == "red":
                # self._sigdepnoise_acc *= 1 - 0.1
                self._sigdepnoise_acc += self.muscle_noise_params.sigdepnoise_level*_selected_muscle_control*_noise
                _selected_muscle_control += self._sigdepnoise_acc
            else:
                raise NotImplementedError(f"{self.muscle_noise_params.sigdepnoise_type}")
        
        if self.muscle_noise_params.constantnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.muscle_noise_params.constantnoise_type == "white":
                _selected_muscle_control += self.muscle_noise_params.constantnoise_level*_noise
            elif self.muscle_noise_params.constantnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.muscle_noise_params.constantnoise_level*_noise
            elif self.muscle_noise_params.constantnoise_type == "red":
                self._constantnoise_acc += self.muscle_noise_params.constantnoise_level*_noise
                _selected_muscle_control += self._constantnoise_acc
            else:
                raise NotImplementedError(f"{self.muscle_noise_params.constantnoise_type}")

        # # Update smoothed online estimate of motor actuation
        # self._motor_act = (1 - self._motor_alpha) * self._motor_act \
        #                         + self._motor_alpha * np.clip(_selected_motor_control, 0, 1)
        motor_act = _selected_motor_control
        new_ctrl = new_ctrl.at[self._motor_actuators].set(self._mj_model.actuator_ctrlrange[self._motor_actuators, 0] + motor_act*(self._mj_model.actuator_ctrlrange[self._motor_actuators, 1] - self._mj_model.actuator_ctrlrange[self._motor_actuators, 0]))
        new_ctrl = new_ctrl.at[self._muscle_actuators].set(jp.clip(_selected_muscle_control, 0, 1))

        # implement abnormalities
        if self.muscle_condition == "fatigue":
            # import ipdb; ipdb.set_trace()
            _ctrl_after_fatigue, _, _ = self.muscle_fatigue.compute_act(
                new_ctrl[self._muscle_actuators]
            )
            new_ctrl = new_ctrl.at[self._muscle_actuators].set(_ctrl_after_fatigue)
        elif self.muscle_condition == "reafferentation":
            # redirect EIP --> EPL
            new_ctrl = new_ctrl.at[self.EPLpos].set(new_ctrl[self.EIPpos].copy())
            # Set EIP to 0
            new_ctrl = new_ctrl.at[self.EIPpos].set(0)

        return new_ctrl

    def get_obs_vec(self, data, info):
        obs_dict = self.get_obs_dict(data, info)
        obs_dict = self.update_obs_with_pixels(obs_dict, info)
        obs = self.obsdict2obsvec(obs_dict)
        _updated_info = self.update_info(info, obs_dict)
        return obs, _updated_info
    
    def obsdict2obsvec(self, obs_dict) -> jp.ndarray:
        obs_list = [jp.zeros(0)]
        for key in self.obs_keys:
            obs_list.append(obs_dict[key].ravel()) # ravel helps with images
        obsvec = jp.concatenate(obs_list)
        if not self.vision:
            return {"proprioception": obsvec}
        if self.vision_mode == "rgbd_only":
            return {"pixels/view_0": obs_dict["pixels/view_0"]}
        elif self.vision_mode == "depth_only":
            return {"pixels/depth": obs_dict["pixels/depth"]}
        elif self.vision_mode == "depth":
            return {"pixels/depth": obs_dict["pixels/depth"], "proprioception": obsvec}
        elif self.vision_mode == "depth_w_aux_task":
            _obsvec_w_aux_task = self.get_obs_vec_aux_task(obs_dict, obsvec)
            if _obsvec_w_aux_task is not None:
                return _obsvec_w_aux_task
            else:
                raise NotImplementedError(f"Cannot get observation vector for vision_mode 'depth_w_aux_task': get_obs_vec_aux_task() is missing from {self.__class__}")
        vision_obs = {
            "proprioception": obsvec,
            "pixels/view_0": obs_dict["pixels/view_0"],
        }
        if self.vision_mode == 'rgb+depth':
            vision_obs['pixels/depth'] = obs_dict['pixels/depth']
        return vision_obs
    
    def get_obs_vec_aux_task(self, obs_dict, obsvec) -> jp.ndarray:
        pass

    def update_obs_with_pixels(self, obs_dict, info):
        if self.vision:
            if (
                self.vision_mode == "rgb"
                or self.vision_mode == "rgbd"
                or self.vision_mode == "rgbd_only"
            ):
                obs_dict["pixels/view_0"] = info["pixels/view_0"]
            if (
                self.vision_mode == "rgb+depth"
                or self.vision_mode == "depth_only"
                or self.vision_mode == "depth"
                or self.vision_mode == "depth_w_aux_task"
            ):
                obs_dict["pixels/depth"] = info["pixels/depth"]
        return obs_dict

    def generate_pixels(self, data, render_token=None):
        # Generates the view of the environment using the batch renderer
        update_info = {}
        if render_token is None:  # Initialize renderer during reset
            render_token, rgb, depth = self.batch_renderer.init(data, self._mjx_model)
            update_info.update({"render_token": render_token})
        else:  # Render during step
            _, rgb, depth = self.batch_renderer.render(render_token, data)
        pixels = rgb[0][..., :3].astype(jp.float32) / 255.0
        depth = depth[0].astype(jp.float32)

        if self.vision_mode == "rgb":
            update_info.update({"pixels/view_0": pixels})
        elif self.vision_mode == "rgbd" or self.vision_mode == "rgbd_only":
            # combine pixels and depth into a single image
            rgbd = jp.concatenate([pixels, depth], axis=-1)
            update_info.update({"pixels/view_0": rgbd})
        elif (
            self.vision_mode == "rgb+depth"
            or self.vision_mode == "depth_only"
            or self.vision_mode == "depth"
            or self.vision_mode == "depth_w_aux_task"
        ):
            update_info.update({"pixels/view_0": pixels, "pixels/depth": depth})
        else:
            raise ValueError(f"Invalid vision mode: {self.vision_mode}")
        return update_info
    

    ########################################
    # Functions used for offline rendering 
    # only (not compatible with Madrona):
    ########################################
    def update_task_visuals(self, mj_model, state):
        pass

    def render(
        self,
        trajectory: List[State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[
            Sequence[Callable[[mujoco.MjvScene], None]]
        ] = None,
    ) -> Sequence[np.ndarray]:
        return self.render_array(
            self.mj_model,
            trajectory,
            height,
            width,
            camera,
            scene_option=scene_option,
            modify_scene_fns=modify_scene_fns,
        )

    def render_array(self,
        mj_model: mujoco.MjModel,
        trajectory: Union[List[State], State],
        height: int = 480,
        width: int = 640,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[
            Sequence[Callable[[mujoco.MjvScene], None]]
        ] = None,
        hfield_data: Optional[jax.Array] = None,
    ):
        """Renders a trajectory as an array of images."""
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        camera = camera if camera is not None else -1

        if hfield_data is not None:
            mj_model.hfield_data = hfield_data.reshape(mj_model.hfield_data.shape)
            mujoco.mjr_uploadHField(mj_model, renderer._mjr_context, 0)

        def get_image(state, modify_scn_fn=None) -> np.ndarray:
            d = mujoco.MjData(mj_model)
            d.qpos, d.qvel = state.data.qpos, state.data.qvel
            d.mocap_pos, d.mocap_quat = state.data.mocap_pos, state.data.mocap_quat
            d.xfrc_applied = state.data.xfrc_applied
            self.update_task_visuals(mj_model=mj_model, state=state)
            # d.xpos, d.xmat = state.data.xpos, state.data.xmat.reshape(mj_model.nbody, -1)  #for bodies/geoms without joints (target spheres etc.)
            # d.geom_xpos, d.geom_xmat = state.data.geom_xpos, state.data.geom_xmat.reshape(mj_model.ngeom, -1)  #for geoms in bodies without joints (target spheres etc.)
            # d.site_xpos, d.site_xmat = state.data.site_xpos, state.data.site_xmat.reshape(mj_model.nsite, -1)
            mujoco.mj_forward(mj_model, d)
            renderer.update_scene(d, camera=camera, scene_option=scene_option)
            if modify_scn_fn is not None:
                modify_scn_fn(renderer.scene)
            return renderer.render()

        if isinstance(trajectory, list):
            out = []
            for i, state in enumerate(tqdm.tqdm(trajectory)):
                if modify_scene_fns is not None:
                    modify_scene_fn = modify_scene_fns[i]
                else:
                    modify_scene_fn = None
                out.append(get_image(state, modify_scene_fn))
        else:
            out = get_image(trajectory)

        renderer.close()
        return out
    

    ########################################
    # Other functions to be 
    # implemented by the child class:
    ########################################

    def eval_metrics(self, rollout, eval_metrics_keys={}):
        """Calculate task-specific evaluation metrics (only to be used for eval logging)."""
        return {}
