# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for MyoUser Arm Pointing model."""
from datetime import datetime
from typing import Any, Dict, Optional, Union
import collections

from etils import epath
import numpy as np
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State

from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src
from myosuite.envs.myo.fatigue import CumulativeFatigue


def default_config() -> config_dict.ConfigDict:
    #TODO: update/make use of env_config parameters!
    env_config = config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.002,
        # episode_length=400,
        max_duration=4., # timelimit per target, in seconds
        noise_config=config_dict.create(
            reset_noise_scale=1e-1,
        ),
        reward_config=config_dict.create(
            reach=1,
            bonus=8,
            neural_effort=0,  #1e-4,
        )
    )

    vision_config = config_dict.create(
      gpu_id=0,
      render_batch_size=1024,
      render_width=64,
      render_height=64,
      use_rasterizer=False,
      enabled_geom_groups=[0, 1, 2],
    )

    rl_config = config_dict.create(
        num_timesteps=15_000_000,  #50_000_000,
        log_training_metrics=True,
        num_evals=0,  #16,
        reward_scaling=0.1,
        # episode_length=env_config.episode_length,
        episode_length=int(env_config.max_duration / env_config.ctrl_dt),  #TODO: fix, as this dependency is not automatically updated...
        clipping_epsilon=0.3,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=8,  #128,  #32
        num_updates_per_batch=8,  #2,  #8
        num_resets_per_eval=1,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0.001,
        num_envs=1024,  #8192,
        batch_size=128,  #512,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 256),
            value_hidden_layer_sizes=(256, 256),
            # policy_obs_key="state",
            # value_obs_key="state",
            # distribution_type="tanh",
        )
    )
    env_config["vision_config"] = vision_config
    env_config["ppo_config"] = rl_config
    return env_config


class PlaygroundArmPointing(mjx_env.MjxEnv):
    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'qacc', 'ee_pos', 'act', 'target_pos', 'target_radius']
    # DEFAULT_RWD_KEYS_AND_WEIGHTS = {
    #     "reach": 1.0,
    #     "bonus": 8.0,
    #     #"penalty": 50,
    #     "neural_effort": 0,  #1e-4,
    # }

    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            is_msk=True,
            vision=False,  #TODO: define vision in config
    ) -> None:
        super().__init__(config, config_overrides)
        xml_path = "myosuite/simhive/uitb_sim/mobl_arms_index_eepos_pointing.xml"  #rf"../assets/arm/myoarm_relocate.xml"  #TODO: use 'wrapper' xml file in assets rather than raw simhive file
        # xml_path = '../assets/arm/myoarm_pose.xml'  #rf"../assets/arm/myoarm_relocate.xml"
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = self.sim_dt

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path

        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.disableflags = self._mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

        self.max_duration = config.max_duration
        self.weighted_reward_keys = config.reward_config

        # init_target_area_width_scale = config.adaptive_task_config.init_target_area_width_scale

        self._prepare_bm_model()
        
        self._setup()  #**kwargs)

        # Do a forward step so stuff like geom and body positions are calculated [using MjData rather than mjx.Data, to reduce computational overheat]
        # rng_init = jax.random.PRNGKey(self.seed)
        # init_state = self.reset(rng_init, target_pos=jp.zeros(3))
        # _data = init_state.pipeline_state
        _data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, _data)

        self._prepare_after_init(_data)

        self.vision = vision
        if self.vision:
            from madrona_mjx.renderer import BatchRenderer

            #TODO: replace kwargs arguments with config entries
            vision_mode = kwargs['vision']['vision_mode']
            allowed_vision_modes = ('rgbd', 'rgb', 'rgb+depth')
            assert vision_mode in allowed_vision_modes, f"Invalid vision mode: {vision_mode} (allowed modes: {allowed_vision_modes})"
            self.vision_mode = vision_mode
            self.batch_renderer = BatchRenderer(m = self.mjx_model,  #TODO: use mj_model instead?
                                                gpu_id = kwargs['vision']['gpu_id'],
                                                num_worlds = kwargs['num_envs'],
                                                batch_render_view_width = kwargs['vision']['render_width'],
                                                batch_render_view_height = kwargs['vision']['render_height'],
                                                enabled_geom_groups = np.asarray([0, 1, 2]),
                                                enabled_cameras = np.asarray(kwargs['vision']['enabled_cameras']),
                                                use_rasterizer = False,
                                                viz_gpu_hdls = None,
                                                add_cam_debug_geo = False,
                                                )
            
        self.eval_mode = False

    def preprocess_spec(self, spec:mujoco.MjSpec):
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.conaffinity = 0
                geom.contype = 0
                print(f"Disabled contacts for cylinder geom named \"{geom.name}\"")
        return spec

    def _prepare_bm_model(self):
        # Total number of actuators
        self._nu = self.mjx_model.nu

        # Number of muscle actuators
        self._na = self.mjx_model.na

        # Number of motor actuators
        self._nm = self._nu - self._na
        # self._motor_act = jp.zeros((self._nm,))
        self._motor_alpha = 0.9*jp.ones(1)

        # Get actuator names (muscle and motor)
        self._actuator_names = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.mjx_model.nu)]
        self._muscle_actuator_names = set(np.array(self._actuator_names)[self.mj_model.actuator_trntype==mujoco.mjtTrn.mjTRN_TENDON])  #model.actuator_dyntype==mujoco.mjtDyn.mjDYN_MUSCLE
        self._motor_actuator_names = set(self._actuator_names) - self._muscle_actuator_names

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._muscle_actuator_names = sorted(self._muscle_actuator_names, key=self._actuator_names.index)
        self._motor_actuator_names = sorted(self._motor_actuator_names, key=self._actuator_names.index)

        # Find actuator indices in the simulation
        self._muscle_actuators = jp.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                                for actuator_name in self._muscle_actuator_names], dtype=jp.int32)
        self._motor_actuators = jp.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                                for actuator_name in self._motor_actuator_names], dtype=jp.int32)
    
        # Get joint names (dependent and independent)
        self._joint_names = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.mjx_model.njnt)]
        self._dependent_joint_names = {self._joint_names[idx] for idx in
                                    np.unique(self.mj_model.eq_obj1id[self.mj_model.eq_active0.astype(bool)])} \
        if self.mjx_model.eq_obj1id is not None else set()
        self._independent_joint_names = set(self._joint_names) - self._dependent_joint_names

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._dependent_joint_names = sorted(self._dependent_joint_names, key=self._joint_names.index)
        self._independent_joint_names = sorted(self._independent_joint_names, key=self._joint_names.index)

        # Find dependent and independent joint indices in the simulation
        self._dependent_joints = jp.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                                for joint_name in self._dependent_joint_names], dtype=jp.int32)
        self._independent_joints = jp.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                                for joint_name in self._independent_joint_names], dtype=jp.int32)

        # If there are 'free' type of joints, we'll need to be more careful with which dof corresponds to
        # which joint, for both qpos and qvel/qacc. There should be exactly one dof per independent/dependent joint.
        def get_dofs(joint_indices):
            qpos = jp.array([], dtype=jp.int32)
            dofs = jp.array([], dtype=jp.int32)
            for joint_idx in joint_indices:
                if self.mjx_model.jnt_type[joint_idx] not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                    raise NotImplementedError(f"Only 'hinge' and 'slide' joints are supported, joint "
                                            f"{self._joint_names[joint_idx]} is of type {mujoco.mjtJoint(self.mjx_model.jnt_type[joint_idx]).name}")
                qpos = jp.append(qpos, self.mjx_model.jnt_qposadr[joint_idx])
                dofs = jp.append(dofs, self.mjx_model.jnt_dofadr[joint_idx])
            return qpos, dofs
        self._dependent_qpos, self._dependent_dofs = get_dofs(self._dependent_joints)
        self._independent_qpos, self._independent_dofs = get_dofs(self._independent_joints)

    def _setup(self,
            # target_pos_range:dict = {'IFtip': jp.array([[-0.1, 0.225, -0.3], [0.1, 0.35, 0.3]]),}
            target_pos_range:dict = {'fingertip': jp.array([[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]]),},
            # target_radius_range:dict = {'IFtip': jp.array([0.05, 0.05]),},
            target_radius_range:dict = {'fingertip': jp.array([0.05, 0.05]),},
            target_origin_rel:list = jp.zeros(3),  #[0.225, -0.1, 0.05],  #NOTE: target area offset should be directly added to target_pos_range
            # ref_site = 'R.Shoulder_marker',
            ref_site = 'humphant',
            muscle_condition = None,
            sex = None,
            max_trials = 1,
            sigdepnoise_type = None,   #"white"
            sigdepnoise_level = 0.103,
            constantnoise_type = None,   #"white"
            constantnoise_level = 0.185,
            reset_type = "range_uniform",  #TODO: move to config
            obs_keys:list = DEFAULT_OBS_KEYS,
            # weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            # episode_length = 800,
            # frame_skip = 25,  #NOTE: deprecated; use ctrl_dt and sim_dt instead; frame_skip=25 corresponds to 20Hz; with episode_length=800, this results in 40s maximum time per episode
            **kwargs,
        ):
        # self.target_origin = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos + jp.array(target_origin_rel)
        self.target_pos_range = target_pos_range
        self.target_radius_range = target_radius_range
        self.target_origin_rel = target_origin_rel
        self.ref_site = ref_site

        zero = jp.zeros(1)
        
        # Define training properties (sex might be used for fatigue model)
        self.muscle_condition = muscle_condition
        self.sex = sex

        # Define a maximum number of trials per episode (if needed for e.g. evaluation / visualisation)
        self.max_trials = max_trials

        # self.init_target_area_width_scale = 1.  #sample from full target area for non-adaptive tasks

        # Define signal-dependent noise
        self.sigdepnoise_type = sigdepnoise_type
        self.sigdepnoise_level = sigdepnoise_level * jp.ones(1)
        self.sigdepnoise_acc = zero  #only used for red/Brownian noise
        
        # Define constant (i.e., signal-independent) noise
        self.constantnoise_type = constantnoise_type
        self.constantnoise_level = constantnoise_level * jp.ones(1)
        self.constantnoise_acc = zero  #only used for red/Brownian noise

        self.initializeConditions()

        # Define reset type
        self.reset_type = reset_type
        ## valid reset types: 
        valid_reset_types = ("zero", "epsilon_uniform", "range_uniform", None)
        assert self.reset_type in valid_reset_types, f"Invalid reset type '{self.reset_type} (valid types are {valid_reset_types})."

        # # Initialise other variables required for setup (i.e., by get_obs_vec, get_obs_dict, get_reward_dict, or get_env_infos)
        # self.dwell_threshold = zero  #0.
        
        # Setup reward function and observation components, and other meta-parameters
        self.rwd_keys_wt = self.weighted_reward_keys
        self.obs_keys = obs_keys

        self.tip_sids = []
        self.target_sids = []
        sites = self.target_pos_range.keys()
        if sites:
            for site in sites:
                self.tip_sids.append(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site))
                self.target_sids.append(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target'))
        # self._episode_length = episode_length
        
        self.target_body_id = self.mj_model.body('target').id
        self.target_geom_id = self.mj_model.geom('target_sphere').id

        # self._normalize_act = normalize_act

        #### Task-specific settings (data-dependent settings need to be defined in _prepare_after_init) ####
        # # Define success/fail log buffer and its pointer
        # self._trial_success_log = -1*jp.ones(self.success_log_buffer_length, dtype=jp.int32)  #logging whether trials since last target limit adjustment were successful (2D array, with first entry batch index, and second entry trial_id within that single run)
        # self._trial_success_log_pointer_index = jp.zeros(1, dtype=jp.int32)  #1D array of batch size, where each entry corresponds to the current pointer index, defining for each run where to append the next success/failure boolean
        
        # Dwelling based selection -- fingertip needs to be inside target for some time
        self.dwell_threshold = 0.25/self.dt  #corresponds to 250ms; for visual-based pointing use 0.5/self.dt; note that self.dt=self._mjx_model.opt.timestep*self._n_frames

        if 'vision' in kwargs:
            print(f'Using vision, so doubling dwell threshold to {self.dwell_threshold*2}')
            self.dwell_threshold *= 2   

        # Use early termination if target is not hit in time
        # self._steps_since_last_hit = jp.zeros(1)
        self._max_steps_without_hit = self.max_duration/self.dt #corresponds to {max_duration} seconds; note that self.dt=self.mj_model.opt.timestep*self._n_frames
    
    
    def _prepare_after_init(self, data):
        # Define target origin, relative to which target positions will be generated
        self.target_coordinates_origin = data.site_xpos[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ref_site)].copy() + jp.array(self.target_origin_rel)  #jp.zeros(3,)
    
    def initializeConditions(self):
        # for muscle weakness we assume that a weaker muscle has a
        # reduced maximum force
        if self.muscle_condition == "sarcopenia":
            for mus_idx in range(self.mjx_model.actuator_gainprm.shape[0]):
                self.mjx_model.actuator_gainprm[mus_idx, 2] = (
                    0.5 * self.mjx_model.actuator_gainprm[mus_idx, 2].copy()
                )

        # for muscle fatigue we used the 3CC-r model
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.mjx_model, frame_skip=self.n_substeps, sex=self.sex, seed=self.get_input_seed()
            )

        # Tendon transfer to redirect EIP --> EPL
        # https://www.assh.org/handcare/condition/tendon-transfer-surgery
        elif self.muscle_condition == "reafferentation":
            self.EPLpos = self.mjx_model.actuator_name2id("EPL")
            self.EIPpos = self.mjx_model.actuator_name2id("EIP")

    def enable_eval_mode(self):
        # TODO: eval wrapper should call this function at initialization
        self.eval_mode = True

    def disable_eval_mode(self):
        self.eval_mode = False

    def get_ctrl(self, state: State, action: jp.ndarray, rng: jp.ndarray):
        new_ctrl = action.copy()

        _selected_motor_control = jp.clip(action[:self._nm], 0, 1)
        _selected_muscle_control = jp.clip(action[self._nm:], 0, 1)

        if self.sigdepnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.sigdepnoise_type == "white":
                _added_noise = self.sigdepnoise_level*_selected_muscle_control*_noise
                _selected_muscle_control += _added_noise
            elif self.sigdepnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.sigdepnoise_level*_selected_muscle_control*_noise
            elif self.sigdepnoise_type == "red":
                # self.sigdepnoise_acc *= 1 - 0.1
                self.sigdepnoise_acc += self.sigdepnoise_level*_selected_muscle_control*_noise
                _selected_muscle_control += self.sigdepnoise_acc
            else:
                raise NotImplementedError(f"{self.sigdepnoise_type}")
        if self.constantnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.constantnoise_type == "white":
                _selected_muscle_control += self.constantnoise_level*_noise
            elif self.constantnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.constantnoise_level*_noise
            elif self.constantnoise_type == "red":
                self.constantnoise_acc += self.constantnoise_level*_noise
                _selected_muscle_control += self.constantnoise_acc
            else:
                raise NotImplementedError(f"{self.constantnoise_type}")

        # # Update smoothed online estimate of motor actuation
        # self._motor_act = (1 - self._motor_alpha) * self._motor_act \
        #                         + self._motor_alpha * np.clip(_selected_motor_control, 0, 1)
        motor_act = _selected_motor_control

        new_ctrl = new_ctrl.at[self._motor_actuators].set(self.mj_model.actuator_ctrlrange[self._motor_actuators, 0] + motor_act*(self.mj_model.actuator_ctrlrange[self._motor_actuators, 1] - self.mj_model.actuator_ctrlrange[self._motor_actuators, 0]))
        new_ctrl = new_ctrl.at[self._muscle_actuators].set(jp.clip(_selected_muscle_control, 0, 1))

        isNormalized = False  #TODO: check whether we can integrate the default normalization from BaseV0.step
        
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
    
    def generate_pixels(self, data, render_token=None):
        # Generates the view of the environment using the batch renderer
        update_info = {}
        if render_token is None: # Initialize renderer during reset
            render_token, rgb, depth = self.batch_renderer.init(data, self.sys)
            update_info.update({"render_token": render_token})
        else: # Render during step
            _, rgb, depth = self.batch_renderer.render(render_token, data)
        pixels = rgb[0][..., :3].astype(jp.float32) / 255.0
        depth = depth[0].astype(jp.float32)

        if self.vision_mode == 'rgb':
            update_info.update({"pixels/view_0": pixels})
        elif self.vision_mode == 'rgbd':
            # combine pixels and depth into a single image
            rgbd = jp.concatenate([pixels, depth], axis=-1)
            update_info.update({"pixels/view_0": rgbd})
        elif self.vision_mode == 'rgb+depth':
            update_info.update({"pixels/view_0": pixels, "pixels/depth": depth})
        else:
            raise ValueError(f"Invalid vision mode: {self.vision_mode}")
        return update_info
    
    
    def add_target_pos_to_data(self, data, target_pos):
        xpos = data.xpos
        geom_xpos = data.geom_xpos

        xpos = xpos.at[self.target_body_id].set(target_pos)
        geom_xpos = geom_xpos.at[self.target_geom_id].set(target_pos)
        data = data.replace(xpos=xpos, geom_xpos=geom_xpos)
        return data
    
    def generate_target_pos(self, rng, target_pos=None):
        # jax.debug.print(f"Generate new target (target area scale={target_area_dynamic_width_scale})")

        # Set target location
        ##TODO: implement _new_target_distance_threshold constraint with rejection sampling!; improve code efficiency (remove for-loop)
        if target_pos is None:
            target_pos = jp.array([])
            # Sample target position
            rng, *rngs = jax.random.split(rng, len(self.target_pos_range)+1)
            for (site, span), _rng in zip(self.target_pos_range.items(), rngs):
                # if self.adaptive_task:
                #     span = self.get_current_target_pos_range(span, target_area_dynamic_width_scale)
                # sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target')
                new_position = self.target_coordinates_origin + jax.random.uniform(_rng, shape=self.target_coordinates_origin.shape, minval=span[0], maxval=span[1])
                target_pos = jp.append(target_pos, new_position.copy())
                # self.mj_model.site_pos.at[sid].set(new_position)

        # self._steps_inside_target = jp.zeros(1)

        return target_pos
    
    def generate_target_size(self, rng, target_radius=None):
        # jax.debug.print(f"Generate new target (target area scale={target_area_dynamic_width_scale})")

        # Set target size
        ##TODO: improve code efficiency (remove for-loop)
        if target_radius is None:
            target_radius = jp.array([])
            # Sample target radius
            rng, *rngs = jax.random.split(rng, len(self.target_radius_range)+1)
            for (site, span), _rng in zip(self.target_radius_range.items(), rngs):
                # sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target')
                new_radius = jax.random.uniform(_rng, minval=span[0], maxval=span[1])
                target_radius = jp.append(target_radius, new_radius.copy())
                # self.mj_model.site_size[sid][0] = new_radius

        # self._steps_inside_target = jp.zeros(1)

        return target_radius

    def get_current_target_pos_range(self, span, target_area_dynamic_width_scale):
        return target_area_dynamic_width_scale*(span - jp.mean(span, axis=0)) + jp.mean(span, axis=0)
    

    def get_obs_vec(self, data, info):
        obs_dict = self.get_obs_dict(data, info)
        obs = self.obsdict2obsvec(obs_dict)
        _updated_info = self.update_info(info, obs_dict)
        return obs, _updated_info

    def get_obs_dict(self, data, info):
        rng = info['rng']

        obs_dict = {}
        obs_dict['time'] = jp.array(data.time)
        
        # Normalise qpos
        jnt_range = self.mjx_model.jnt_range[self._independent_joints]
        qpos = data.qpos[self._independent_qpos].copy()
        qpos = (qpos - jnt_range[:, 0]) / (jnt_range[:, 1] - jnt_range[:, 0])
        qpos = (qpos - 0.5) * 2
        obs_dict['qpos'] = qpos

        # Get qvel, qacc
        obs_dict['qvel'] = data.qvel[self._independent_dofs].copy()  #*self.dt
        obs_dict['qacc'] = data.qacc[self._independent_dofs].copy()

        # Normalise act
        if self._na > 0:
            obs_dict['act']  = (data.act - 0.5) * 2
        
        # # Smoothed average of motor actuation (only for motor actuators)  #; normalise
        # obs_dict['motor_act'] = info['motor_act']  #(self._motor_act.copy() - 0.5) * 2

        # Store current control input
        obs_dict['last_ctrl'] = data.ctrl.copy()

        # End-effector and target position
        obs_dict['ee_pos'] = jp.vstack([data.site_xpos[self.tip_sids[isite]].copy() for isite in range(len(self.tip_sids))])
        obs_dict['target_pos'] = info['target_pos']  #jp.vstack([data.site_xpos[self.target_sids[isite]].copy() for isite in range(len(self.tip_sids))])

        # Distance to target (used for rewards later)
        obs_dict['reach_dist'] = jp.linalg.norm(jp.array(obs_dict['target_pos']) - jp.array(obs_dict['ee_pos']), axis=-1)

        # Target radius
        obs_dict['target_radius'] = info['target_radius']   #jp.array([self.mj_model.site_size[self.target_sids[isite]][0] for isite in range(len(self.tip_sids))])
        # jax.debug.print(f"STEP-Obs: {obs_dict['target_radius']}")
        obs_dict['inside_target'] = jp.squeeze(obs_dict['reach_dist'] < obs_dict['target_radius'])
        # print(obs_dict['inside_target'], jp.ones(1))

        ## we require all end-effector--target pairs to have distance below the respective target radius
        # obs_dict['steps_inside_target'] = (info['steps_inside_target'] + jp.select(obs_dict['inside_target'], jp.ones(1))) * jp.select(obs_dict['inside_target'], jp.ones(1))
        # print(info['steps_inside_target'], jp.select(obs_dict['inside_target'], jp.ones(1)), (info['steps_inside_target'] + jp.select(obs_dict['inside_target'], jp.ones(1))), obs_dict['steps_inside_target'])
        _steps_inside_target = jp.select([obs_dict['inside_target']], [info['steps_inside_target'] + 1], 0)
        _target_timeout = info['steps_since_last_hit'] >= self._max_steps_without_hit
        # print("steps_inside_target", obs_dict['steps_inside_target'])
        obs_dict['target_success'] = _steps_inside_target >= self.dwell_threshold
        obs_dict['target_fail'] = ~obs_dict['target_success'] & _target_timeout
        
        obs_dict['steps_inside_target'] = jp.select([obs_dict['target_success']], [0], _steps_inside_target)
        obs_dict['steps_since_last_hit'] = jp.select([obs_dict['target_success'] | obs_dict['target_fail']], [0], info['steps_since_last_hit'])
        obs_dict['trial_idx'] = info['trial_idx'] + jp.select([obs_dict['target_success'] | obs_dict['target_fail']], jp.ones(1))
        # print("trial_idx", obs_dict['trial_idx'])
        obs_dict['task_completed'] = obs_dict['trial_idx'] >= self.max_trials

        # obs_dict['target_area_dynamic_width_scale'] = info['target_area_dynamic_width_scale']

        if self.vision:
            obs_dict['pixels/view_0'] = info['pixels/view_0']
            if self.vision_mode == 'rgb+depth':
                obs_dict['pixels/depth'] = info['pixels/depth']
        return obs_dict
    
    def obsdict2obsvec(self, obs_dict) -> jp.ndarray:
        obs_list = [jp.zeros(0)]
        for key in self.obs_keys:
            obs_list.append(obs_dict[key].ravel()) # ravel helps with images
        obsvec = jp.concatenate(obs_list)
        if not self.vision:
            # return obsvec
            return {"proprioception": obsvec}
        vision_obs = {'proprioception': obsvec, 'pixels/view_0': obs_dict['pixels/view_0']}
        if self.vision_mode == 'rgb+depth':
            vision_obs['pixels/depth'] = obs_dict['pixels/depth']
        return vision_obs
    
    def update_info(self, info, obs_dict):
        ## Update state info of internal variables at the end of each env step
        info['last_ctrl'] = obs_dict['last_ctrl']
        info['steps_since_last_hit'] = obs_dict['steps_since_last_hit']
        info['steps_inside_target'] = obs_dict['steps_inside_target']
        info['trial_idx'] = obs_dict['trial_idx'].copy()
        info['reach_dist'] = obs_dict['reach_dist']

        # Also store variables useful for evaluation
        info['target_success'] = obs_dict['target_success']
        info['target_fail'] = obs_dict['target_fail']
        info['task_completed'] = obs_dict['task_completed']

        return info

    def get_reward_dict(self, obs_dict):
        reach_dist = obs_dict['reach_dist']
        target_radius = obs_dict['target_radius']
        # reach_dist_to_target_bound = jp.linalg.norm(jp.moveaxis(reach_dist-target_radius, -1, -2), axis=-1)
        reach_dist_to_target_bound = jp.linalg.norm(reach_dist-target_radius, axis=-1)
        # steps_inside_target = jp.linalg.norm(obs_dict['steps_inside_target'], axis=-1)
        ctrl_magnitude = jp.linalg.norm(obs_dict['last_ctrl'], axis=-1)
        # trial_idx = jp.linalg.norm(obs_dict['trial_idx'], axis=-1)

        act_mag = jp.linalg.norm(obs_dict['act'], axis=-1)/self._na if self._na != 0 else 0
        # far_th = self.far_th*len(self.tip_sids) if jp.squeeze(obs_dict['time'])>2*self.dt else jp.inf
        # near_th = len(self.tip_sids)*.0125
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   1.*(jp.exp(-reach_dist_to_target_bound*10.) - 1.)/10.),  #-1.*reach_dist)
            ('bonus',   1.*(obs_dict['target_success'])),  #1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('neural_effort', -1.*(ctrl_magnitude ** 2)),
            ('act_reg', -1.*act_mag),
            # ('penalty', -1.*(np.any(reach_dist > far_th))),
            # Must keys
            ('sparse',  -1.*(jp.linalg.norm(reach_dist, axis=-1) ** 2)),
            ('solved',  1.*(obs_dict['target_success'])),
            ('done',    1.*(obs_dict['task_completed'])), #np.any(reach_dist > far_th))),
        ))
        # print(rwd_dict.items())
        rwd_dict['dense'] = jp.sum(jp.array([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()]), axis=0)

        return rwd_dict
    
    # def get_env_infos(self, state: State, data: mjx.Data):
    #     """
    #     Get information about the environment.
    #     """
    #     ## TODO: update this function!!

    #     # # resolve if current visuals are available
    #     # if self.visual_dict and "time" in self.visual_dict.keys() and self.visual_dict['time']==self.obs_dict['time']:
    #     #     visual_dict = self.visual_dict
    #     # else:
    #     #     visual_dict = {}

    #     env_info = {
    #         'time': self.obs_dict['time'][()],          # MDP(t)
    #         'rwd_dense': self.rwd_dict['dense'][()],    # MDP(t)
    #         'rwd_sparse': self.rwd_dict['sparse'][()],  # MDP(t)
    #         'solved': self.rwd_dict['solved'][()],      # MDP(t)
    #         'done': self.rwd_dict['done'][()],          # MDP(t)
    #         'obs_dict': self.obs_dict,                  # MDP(t)
    #         # 'visual_dict': visual_dict,                 # MDP(t), will be {} if user hasn't explicitly updated self.visual_dict at the current time
    #         # 'proprio_dict': self.proprio_dict,          # MDP(t)
    #         'rwd_dict': self.rwd_dict,                  # MDP(t)
    #         'state': state.data,              # MDP(t)
    #     }

    #     return env_info

    # def reset(self, rng: jp.ndarray) -> State:
    #     """Resets the environment to an initial state."""
    #     rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    #     low, hi = -self._config.noise_config.reset_noise_scale, self._config.noise_config.reset_noise_scale
    #     qpos = self.mjx_model.qpos0 + jax.random.uniform(
    #         rng1, (self.mjx_model.nq,), minval=self.mjx_model.jnt_range[:,0], maxval=self.mjx_model.jnt_range[:,1]
    #     )
    #     qvel = jp.zeros(self.mjx_model.nv, dtype=jp.float32)
    #     target_angle = jax.random.uniform(
    #         rng3, (1,), minval=self._config.healthy_angle_range[0], maxval=self._config.healthy_angle_range[1]
    #     )

    #     # We store the target angle in the info, can't store it as an instance variable,
    #     # as it has to be determined in a parallelized manner
    #     info = {'rng': rng, 'target_angle': target_angle}

    #     data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros((self.mjx_model.nu,)))

    #     obs = self._get_obs(data, jp.zeros(self.mjx_model.nu), info)
    #     reward, done, zero = jp.zeros(3)
    #     metrics = {
    #         'angle_reward': zero,
    #         'reward_quadctrl': zero,
    #     }
    #     return State(data, {"state": obs}, reward, done, metrics, info)

    def reset(self, rng, **kwargs):
        # jax.debug.print(f"RESET INIT")

        _, rng = jax.random.split(rng, 2)

        # Reset counters
        steps_since_last_hit, steps_inside_target, trial_idx = jp.zeros(3)
        # self._target_success = jp.array(False)
        
        # Reset last control (used for observations only)
        last_ctrl = jp.zeros(self._nu)

        # self.robot.sync_sims(self.sim, self.sim_obsd)

        if self.reset_type == "zero":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng)
        elif self.reset_type == "epsilon_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_epsilon_uniform(rng)
        elif self.reset_type == "range_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng)
            data = mjx_env.init(self.mjx_model, qpos=reset_qpos, qvel=reset_qvel, act=reset_act)
            reset_qpos, reset_qvel, reset_act = self._reset_range_uniform(rng, data)
        else:
            reset_qpos, reset_qvel, reset_act = None, None, None

        data = mjx_env.init(self.mjx_model, qpos=reset_qpos, qvel=reset_qvel, act=reset_act)

        self._reset_bm_model(rng)

        info = {'rng': rng,
                'last_ctrl': last_ctrl,
                'steps_inside_target': steps_inside_target,
                'reach_dist': jp.array(0.),
                'target_success': jp.array(False),
                'steps_since_last_hit': steps_since_last_hit,
                'target_fail': jp.array(False),
                'trial_idx': trial_idx,
                'task_completed': jp.array(False),
                }
        info['target_pos'] = self.generate_target_pos(rng, target_pos=kwargs.get("target_pos", None))
        info['target_radius'] = self.generate_target_size(rng, target_radius=kwargs.get("target_radius", None))
        if self.vision or self.eval_mode:
            data = self.add_target_pos_to_data(data, info['target_pos'])
        
        if self.vision:
            info.update(self.generate_pixels(data))
        obs, info = self.get_obs_vec(data, info)  #update info from observation made
        # obs_dict = self.get_obs_dict(data, info)
        # obs = self.obsdict2obsvec(obs_dict)

        # self.generate_target(rng, obs_dict)

        reward, done = jp.zeros(2)
        metrics = {'success_rate': 0.,
                    'reach_dist': 0.,
                    'target_success_sum': 0.,  #TODO: remove this and following lines
                    'target_fail_sum': 0.,
                    'target_success_final': 0.,
                    'target_fail_final': 0.,
                }  #'bonus': zero}
        
        return State(data, obs, reward, done, metrics, info)
    
    def reset_with_curriculum(self, rng, info_before_reset, **kwargs):
        return self.reset(rng, **kwargs)
    
        # jax.debug.print(f"""RESET WITH CURRICULUM {obs_dict_before_reset["trial_idx"]}, {obs_dict_before_reset["target_radius"]}""")

        rng, rng_reset = jax.random.split(rng, 2)

        # Reset counters
        steps_since_last_hit, steps_inside_target, trial_idx = jp.zeros(3)
        # trial_success_log_pointer_index = jp.zeros(1, dtype=jp.int32)
        # trial_success_log = -1*jp.ones(self.success_log_buffer_length, dtype=jp.int32)
        # self._target_success = jp.array(False)
        
        # Reset last control (used for observations only)
        last_ctrl = jp.zeros(self._nu)  #inserting last_ctrl into pipeline_init is not required, assuming that reset_with_curriculum is never called during instatiation of an environment (reset should be used instead)

        # self.robot.sync_sims(self.sim, self.sim_obsd)

        if self.reset_type == "zero":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng_reset)
        elif self.reset_type == "epsilon_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_epsilon_uniform(rng_reset)
        elif self.reset_type == "range_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng_reset)
            data = mjx_env.init(self.mjx_model, qpos=reset_qpos, qvel=reset_qvel, act=reset_act)
            reset_qpos, reset_qvel, reset_act = self._reset_range_uniform(rng_reset, data)
        else:
            reset_qpos, reset_qvel, reset_act = None, None, None

        data = mjx_env.init(self.mjx_model, qpos=reset_qpos, qvel=reset_qvel, act=reset_act)

        self._reset_bm_model(rng_reset)

        info = {'rng': rng_reset,
                'last_ctrl': last_ctrl,
                'steps_inside_target': steps_inside_target,
                'reach_dist': jp.array(0.),
                'target_success': jp.array(False),
                'steps_since_last_hit': steps_since_last_hit,
                'target_fail': jp.array(False),
                'trial_idx': trial_idx,
                'task_completed': jp.array(False),
                }
        info['target_pos'] = self.generate_target_pos(rng_reset, target_pos=kwargs.get("target_pos", None))
        info['target_radius'] = self.generate_target_size(rng_reset, target_radius=kwargs.get("target_radius", None))
        if self.vision or self.eval_mode:
            data = self.add_target_pos_to_data(data, info['target_pos'])

        if self.vision:
            info['render_token'] = info_before_reset['render_token']
            pixels_dict = self.generate_pixels(data, render_token=info['render_token'])
            info.update(pixels_dict)
        obs, info = self.get_obs_vec(data, info)  #update info from observation made
        # obs_dict = self.get_obs_dict(data, info)
        # obs = self.obsdict2obsvec(obs_dict)

        # jax.debug.print(f"obs: {obs}; info-target_radius: {info['target_radius']}")

        # self.generate_target(rng1, obs_dict)

        reward, done = jp.zeros(2)
        metrics = {'success_rate': 0., #obs_dict['success_rate'],
                    'reach_dist': 0.,
                   }  #'bonus': zero}
        
        return State(data, obs, reward, done, metrics, info)
    
    def _reset_zero(self, rng):
        """ Resets the biomechanical model. """

        # Set joint angles and velocities to zero
        rng, rng1 = jax.random.split(rng, 2)
        nqi = len(self._independent_qpos)
        qpos = jp.zeros((nqi,))
        qvel = jp.zeros((nqi,))
        reset_qpos = jp.zeros((self.mjx_model.nq,))
        reset_qvel = jp.zeros((self.mjx_model.nv,))

        # Randomly sample act within unit interval
        reset_act = jax.random.uniform(rng1, shape=self._na, minval=jp.zeros((self._na,)), maxval=jp.ones((self._na,)))

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
    
    def _reset_epsilon_uniform(self, rng):
        """ Resets the biomechanical model. """

        # Randomly sample qpos and qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        qpos = jax.random.uniform(rng1, shape=nqi, minval=jp.ones((nqi,))*-0.05, maxval=jp.ones((nqi,))*0.05)
        qvel = jax.random.uniform(rng2, shape=nqi, minval=jp.ones((nqi,))*-0.05, maxval=jp.ones((nqi,))*0.05)
        reset_qpos = jp.zeros((self.mjx_model.nq,))
        reset_qvel = jp.zeros((self.mjx_model.nv,))
        reset_act = jax.random.uniform(rng3, shape=self._na, minval=jp.zeros((self._na,)), maxval=jp.ones((self._na,)))

        zero = jp.zeros(1)

        # Set qpos and qvel
        ## TODO: ensure that constraints are initially satisfied
        reset_qpos = reset_qpos.at[self._dependent_qpos].set(zero)
        reset_qpos = reset_qpos.at[self._independent_qpos].set(qpos)
        reset_qvel = reset_qvel.at[self._dependent_dofs].set(zero)
        reset_qvel = reset_qvel.at[self._independent_dofs].set(qvel)
        
        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel, reset_act

    def _reset_range_uniform(self, rng, data):
        """ Resets the biomechanical model. """

        # Randomly sample qpos within joint range, qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        jnt_range = self.mjx_model.jnt_range[self._independent_joints]
        qpos = jax.random.uniform(rng1, shape=(nqi,), minval=jnt_range[:, 0], maxval=jnt_range[:, 1])
        qvel = jax.random.uniform(rng2, shape=(nqi,), minval=jp.ones((nqi,))*-0.05, maxval=jp.ones((nqi,))*0.05)
        reset_qpos = jp.zeros((self.mjx_model.nq,))
        reset_qvel = jp.zeros((self.mjx_model.nv,))
        reset_act = jax.random.uniform(rng3, shape=self._na, minval=jp.zeros((self._na,)), maxval=jp.ones((self._na,)))

        # Set qpos and qvel
        reset_qpos = reset_qpos.at[self._independent_qpos].set(qpos)
        # reset_qpos[self._dependent_qpos] = 0
        reset_qvel = reset_qvel.at[self._independent_dofs].set(qvel)
        # reset_qvel[self._dependent_dofs] = 0
        reset_qpos = self.ensure_dependent_joint_angles(data, reset_qpos)
        
        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel, reset_act

    def ensure_dependent_joint_angles(self, data, reset_qpos):
        """ Adjusts virtual joints according to active joint constraints. """

        _joint_constraints = self.mjx_model.eq_type == 2
        _active_eq_constraints = data.eq_active == 1

        eq_dep, eq_indep, poly_coefs = jp.array(self.mjx_model.eq_obj1id), \
            jp.array(self.mjx_model.eq_obj2id), \
            jp.array(self.mjx_model.eq_data[:, 4::-1])

        # constraint_info = jp.concat([jp.reshape(self.mjx_model.eq_obj1id, (-1, 1)), jp.reshape(self.mjx_model.eq_obj2id, (-1, 1)), self.mjx_model.eq_data[:, 4::-1]], axis=1)
        # reset_qpos_new = jp.select(jp.array([jp.any((constraint_info[:, 0] == i) & _joint_constraints & _active_eq_constraints) for i in range(self.mjx_model.njnt)]), 
        #                            jp.array([jp.polyval(constraint_info[jp.argwhere(constraint_info[:, 0] == i, size=1), 2:].flatten(), reset_qpos[constraint_info[jp.argwhere(constraint_info[:, 0] == i, size=1), 1].astype(jp.int32).flatten()]) for i in range(self.mjx_model.njnt)]),
        #                            reset_qpos)
        
        reset_qpos_new = jp.select(jp.array([jp.any((eq_dep == i) & _joint_constraints & _active_eq_constraints) for i in range(self.mjx_model.njnt)]), 
                                   jp.array([jp.polyval(poly_coefs[jp.argwhere(eq_dep == i, size=1).flatten(), :].flatten(), reset_qpos[eq_indep[jp.argwhere(eq_dep == i, size=1).flatten()]]) for i in range(self.mjx_model.njnt)]),
                                   reset_qpos)
        
        # reset_qpos_new = jp.where(virtual_joint_ids, reset_qpos)
        # for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
        #         self.mjx_model.eq_obj1id[_joint_constraints & _active_eq_constraints],
        #         self.mjx_model.eq_obj2id[_joint_constraints & _active_eq_constraints],
        #         self.mjx_model.eq_data[_joint_constraints & _active_eq_constraints, 4::-1]):
        #     if physical_joint_id >= 0:
        #         new_qpos = data.qpos  #TODO: copy required?
        #         new_qpos = new_qpos.at[virtual_joint_id].set(jp.polyval(poly_coefs, data.qpos[physical_joint_id]))  #TODO: check mapping between joints (njnt) and dofs (nq/nv)
        #         data.replace(qpos=new_qpos)

        return reset_qpos_new

    def _reset_bm_model(self, rng):
        #TODO: do not store anything in self in this function, as its values should mostly be discarded after it is called (no permanent env changes!)

        # Sample random initial values for motor activation
        # rng, rng1 = jax.random.split(rng, 2)
        # self._motor_act = jax.random.uniform(rng1, shape=(self._nm,), minval=jp.zeros((self._nm,)), maxval=jp.ones((self._nm,)))

        # Reset smoothed average of motor actuator activation
        self._motor_smooth_avg = jp.zeros((self._nm,))

        # Reset accumulative noise
        zero = jp.zeros(1)
        self._sigdepnoise_acc = zero
        self._constantnoise_acc = zero

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        # rng = jax.random.PRNGKey(seed=self.seed)  #TODO: fix/move this line, as it does not lead to random perturbations! (generate all random variables in reset function?)
        rng = state.info['rng']

        # increase step counter
        state.info['steps_since_last_hit'] = state.info['steps_since_last_hit'] + 1

        # Generate new target
        ##TODO: update data0 when target is updated?
        rng, rng1, rng2 = jax.random.split(rng, 3)
        state.info['target_pos'] = jp.select([(state.info['target_success'] | state.info['target_fail'])], [self.generate_target_pos(rng1)], state.info['target_pos'])
        state.info['target_radius'] = jp.select([(state.info['target_success'] | state.info['target_fail'])], [self.generate_target_size(rng2)], state.info['target_radius'])
        # state.info['target_radius'] = jp.select([(obs_dict['target_success'] | obs_dict['target_fail'])], [jp.array([-151.121])], obs_dict['target_radius']) + jax.random.uniform(rng2)
        # jax.debug.print(f"STEP-Info: {state.info['target_radius']}")

        data0 = state.data
        rng, rng_ctrl = jax.random.split(rng, 2)
        new_ctrl = self.get_ctrl(state, action, rng_ctrl)
        
        # step forward
        # self.last_ctrl = self.robot.step(
        #     ctrl_desired=new_ctrl,
        #     ctrl_normalized=isNormalized,
        #     step_duration=self.dt,
        #     realTimeSim=self.mujoco_render_frames,
        #     render_cbk=self.mj_render if self.mujoco_render_frames else None,
        # )
        data = mjx_env.step(self.mjx_model, data0, new_ctrl, n_substeps=self.n_substeps)
        if self.vision or self.eval_mode:
            data = self.add_target_pos_to_data(data, state.info['target_pos'])

        # collect observations and reward
        # obs = self.get_obs_vec(data, state.info)
        if self.vision:
            pixels_dict = self.generate_pixels(data, state.info['render_token'])
            state.info.update(pixels_dict)
        obs_dict = self.get_obs_dict(data, state.info)
        obs = self.obsdict2obsvec(obs_dict)
        rwd_dict = self.get_reward_dict(obs_dict)

        _updated_info = self.update_info(state.info, obs_dict)
        state.replace(info=_updated_info)
        
        _, state.info['rng'] = jax.random.split(rng, 2)  #update rng after each step to ensure variability across steps

        done = rwd_dict['done']
        state.metrics.update(
            success_rate=done*obs_dict['target_success'],
            reach_dist=done*jp.linalg.norm(obs_dict['reach_dist']),
            # bonus=rwd_dict['bonus'],
            target_success_sum = jp.array(obs_dict['target_success'], dtype=jp.float32),  #TODO: remove this and following lines
            target_fail_sum = jp.array(obs_dict['target_fail'], dtype=jp.float32),
            target_success_final = done*obs_dict['target_success'],
            target_fail_final = done*obs_dict['target_fail'],
        )

        # return self.forward(**kwargs)
        return state.replace(
            data=data, obs=obs, reward=rwd_dict['dense'], done=rwd_dict['done']
        )

    # Accessors.
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
