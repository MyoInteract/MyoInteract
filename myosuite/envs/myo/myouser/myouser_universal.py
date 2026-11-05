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
import collections
from copy import deepcopy
import math
import tqdm

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from omegaconf import MISSING

from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src
from myosuite.envs.myo.fatigue import CumulativeFatigue
from myosuite.envs.myo.myouser.base import MyoUserBase, BaseEnvConfig
from myosuite.envs.myo.myouser.target_validation import normalize_pointing_ranges, validate_target_sampling
from dataclasses import asdict, dataclass, field, make_dataclass
from typing import List, Dict, Union, Any, Type, Tuple

@dataclass
class ReachSettings:
    ref_site: str = "humphant"
    target_origin_rel: List[float] = field(default_factory=lambda: [0., 0., 0.])


@dataclass
class IndividualTargetConfig:
    name: str = MISSING
    rgb: List[float] = MISSING

@dataclass
class PointingTarget(IndividualTargetConfig):
    # penetrable: bool = False
    name: str = "pointing_target"
    # Position can either be a 3d vector or a 2 x list of 3d vectors specifying the min and max values for each dimension
    position: List[List[float]] = field(
        default_factory=lambda: [[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]])
    shape: str = "sphere"
    # Size can either be a single value or a list of 2 values specifying the min and max values
    size: List[float] = field(default_factory=lambda: [0.05, 0.15])
    # Any rewards received when inside the target
    reward_incentive: float = 0.0
    completion_bonus: float = 0.0
    dwell_duration: float = 0.25
    rgb: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])

@dataclass
class ButtonTarget(IndividualTargetConfig):
    position: List[float] = MISSING
    name: str = "button_target"
    geom_size: List[float] = field(default_factory=lambda: [0.025, 0.025, 0.01])
    site_size: List[float] = field(default_factory=lambda: [0.02, 0.02, 0.01])
    site_pos: List[float] = field(default_factory=lambda: [0, 0, 0.01])
    geom_margin: float = 0.001
    completion_bonus: float = 0.0
    min_touch_force: float = 1.0
    rgb: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    euler: List[float] = field(default_factory=lambda: [0, -0.79, 0])


@dataclass
class TargetsConfig:
    
    @property
    def targets(self):
        return [getattr(self, f"target_{i}") for i in range(self.num_targets)]

def create_target_config_class(num_targets: int) -> Type:
    """Create a dataclass with the specified number of target fields."""
    fields = [
        (f"target_{i}", IndividualTargetConfig, MISSING) 
        for i in range(num_targets)
    ]
    fields.append(("num_targets", int, num_targets))


    num2str = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']
    
    class_name = f"{num2str[num_targets]}TargetConfig"
    
    return make_dataclass(
        class_name,
        fields,
        frozen=False,
        bases=(TargetsConfig,),
    )

# # Usage
OneTargetConfig = create_target_config_class(1)
TwoTargetConfig = create_target_config_class(2)
ThreeTargetConfig = create_target_config_class(3)
FourTargetConfig = create_target_config_class(4)
FiveTargetConfig = create_target_config_class(5)
SixTargetConfig = create_target_config_class(6)
SevenTargetConfig = create_target_config_class(7)
EightTargetConfig = create_target_config_class(8)
NineTargetConfig = create_target_config_class(9)
TenTargetConfig = create_target_config_class(10)

@dataclass
class SingleTargetConfig(TargetsConfig):
    target_0: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]]))
    num_targets: int = 1


@dataclass
class DefaultTargetsConfig(TargetsConfig):
    target_0: PointingTarget = field(default_factory=lambda: PointingTarget())
    target_1: ButtonTarget = field(default_factory=lambda: ButtonTarget(position=[0.41, -0.05, -0.16], rgb=[0.8, 0.1, 0.8]))
    target_2: PointingTarget = field(default_factory=lambda: PointingTarget(rgb=[0.4, 0.0, 1.0]))
    target_3: ButtonTarget = field(default_factory=lambda: ButtonTarget(position=[0.41, 0.07, -0.16], rgb=[0.1, 0.8, 0.8]))
    target_4: PointingTarget = field(default_factory=lambda: PointingTarget(rgb=[0.0, 1.0, 0.4]))
    target_5: ButtonTarget = field(default_factory=lambda: ButtonTarget(position=[0.50, -0.05, -0.06], rgb=[0.4, 1.0, 0.2]))
    target_6: PointingTarget = field(default_factory=lambda: PointingTarget(rgb=[1.0, 1.0, 0.0]))
    target_7: ButtonTarget = field(default_factory=lambda: ButtonTarget(position=[0.50, 0.07, -0.06], rgb=[0.1, 0.1, 0.1]))
    num_targets: int = 8

@dataclass
class TenPointingTargetsConfig(TargetsConfig):
    num_targets: int = 10
    target_0: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.8, 0.1, 0.1]))
    target_1: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.8, 0.1]))
    target_2: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.1, 0.8]))
    target_3: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.8, 0.8, 0.1]))
    target_4: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.4, 0.4, 0.4]))
    target_5: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.4, 0.1, 0.4]))
    target_6: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.4, 0.4]))
    target_7: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.4, 0.4, 0.1]))
    target_8: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.1, 0.1]))
    target_9: PointingTarget = field(default_factory=lambda: PointingTarget(position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.8, 0.8, 0.8]))

LIST_CONFIGS = [
    (1, "one", OneTargetConfig),
    (2, "two", TwoTargetConfig),
    (3, "three", ThreeTargetConfig),
    (4, "four", FourTargetConfig),
    (5, "five", FiveTargetConfig),
    (6, "six", SixTargetConfig),
    (7, "seven", SevenTargetConfig),
    (8, "eight", EightTargetConfig),
    (9, "nine", NineTargetConfig),
    (10, "ten", TenTargetConfig),
    (7, "default", DefaultTargetsConfig),
    (10, "ten_pointing", TenPointingTargetsConfig),
    (1, "single", SingleTargetConfig),
]

@dataclass
class UniversalTaskConfig:
    reach_settings: ReachSettings = field(default_factory=lambda: ReachSettings())
    obs_keys: List[str] = field(default_factory=lambda: [
        "qpos",
        "qvel",
        "qacc",
        "ee_pos",
        "act",
    ])
    omni_keys: List[str] = field(default_factory=lambda: ["target_pos", "target_size", "phase_progress", "dwell_fraction"])
    weighted_reward_keys: Dict[str, float] = field(default_factory=lambda: {
        "reach": 1,
        "phase_bonus": 0,
        "done": 10,
        "neural_effort": 0.000,
        "jac_effort": 1,
    })
    reach_metric: float = 10.0
    max_duration: float = 6.
    dwell_duration: float = 0.25
    max_trials: int = 1
    reset_type: str = "range_uniform"
    targets: TargetsConfig = MISSING
    show_all_targets: bool = True
    target_init_seed: int = 1
    exponential_reach_reward: bool = False
    enable_extra_dist: bool = True

@dataclass
class UniversalEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserUniversal"
    model_path: str = "myosuite/envs/myo/assets/arm/mobl_arms_index_universal_myouser.xml"
    task_config: UniversalTaskConfig = field(default_factory=lambda: UniversalTaskConfig())

class TargetClass:
    def update_task_visuals_all_shown(self, mj_model: mujoco.MjModel, state: State, phase: int):
        if phase == self.phase_number:
            rgba = jp.array([0, 1, 0, 1]) # Currently focusing on the green target
        elif phase > self.phase_number:
            rgba = jp.array([0, 0, 1, 0.5]) # Blue targets completed, no longer focused on
        elif phase < self.phase_number:
            rgba = jp.array([1, 0, 0, 0.5]) # Red targets yet to do, but not focused on
        mj_model.geom(self.target_geom_name).rgba = rgba
        mj_model.body_pos[self.target_body_id, :] = state.info[f'target_{self.phase_number}/target_pos']
        mj_model.geom_size[self.target_geom_id, :] = state.info[f'target_{self.phase_number}/target_size']

    def update_task_visuals_one_at_a_time(self, mj_model: mujoco.MjModel, state: State, phase: int):
        if phase != self.phase_number:
            mj_model.geom(self.target_geom_name).rgba[-1] = 0.0 # Hide this target
        else:
            mj_model.geom(self.target_geom_name).rgba[-1] = 1.0 # Show this target
            mj_model.body_pos[self.target_body_id, :] = state.info[f'target_{phase}/target_pos']
            mj_model.geom_size[self.target_geom_id, :] = state.info[f'target_{phase}/target_size']

    def update_task_visuals(self, mj_model: mujoco.MjModel, state: State, phase: int):
        if self.show_all_targets:
            self.update_task_visuals_all_shown(mj_model, state, phase)
        else:
            self.update_task_visuals_one_at_a_time(mj_model, state, phase)
    
    @staticmethod
    def compute_next_phase(phase_completed: jp.ndarray, phase: jp.ndarray, num_phases: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        is_last_phase = phase == num_phases - 1
        next_phase = jp.select([is_last_phase], [0], phase + 1)
        task_completed = jp.select([is_last_phase], [phase_completed], 0)
        new_phase = jp.select([phase_completed], [next_phase], phase)
        return new_phase, task_completed

    @property
    def target_geom_name(self):
        return f"geom_target_{self.phase_number}"

    @property
    def target_body_name(self):
        return f"body_target_{self.phase_number}"

    def target_step(self, state: State):
        return state

    def get_task_reward_dict(self, obs_dict: Dict[str, Any]):
        ctrl_magnitude = jp.linalg.norm(obs_dict['last_ctrl'], axis=-1)

        rwd_dict = collections.OrderedDict((
            ('reach', -1.*obs_dict['reach_dist']),
            ('neural_effort', -1.*(ctrl_magnitude ** 2)),
            ('phase_bonus', 1.*obs_dict['phase_completed']),
            ('done', 1.*obs_dict['task_completed']),
        ))

        rwd_dict['dense'] = jp.sum(jp.array([wt*rwd_dict[key] for key, wt in self.weighted_reward_keys.items()]), axis=0)
        return rwd_dict

    def compute_extra_dist(self, info: Dict[str, Any]):
        extra_dist = 0.0
        if not self.enable_extra_dist:
            print(f"Not calculating extra dist for phase {self.phase_number} because enable_extra_dist is False")
            return extra_dist
        if not self.is_last_phase:
            #print(f"Calculating extra dist for phase {self.phase_number}")
            for i in range(self.phase_number, self.total_phases - 1):
                target_i_pos = info[f'target_{i}/target_pos']
                target_i_plus_1_pos = info[f'target_{i+1}/target_pos']
                extra_dist = extra_dist + jp.linalg.norm(target_i_pos - target_i_plus_1_pos, axis=-1)
        #else:
            #print(f"Not calculating extra dist for phase {self.phase_number}")
        return extra_dist

class PointingTargetClass(TargetClass):
    def __init__(self, phase_number: int, total_phases: int, target_pos_range: List[List[float]], target_radius_range: List[float], 
                dwell_steps: int, 
                target_coordinates_origin: jp.ndarray,
                weighted_reward_keys: Dict[str, float],
                show_all_targets: bool = True,
                enable_extra_dist: bool = True,
                target_name: str = "ee_pos", shape: str = "sphere"):
        self.phase_number = phase_number
        self.total_phases = total_phases
        self.is_last_phase = phase_number == total_phases - 1
        self.target_coordinates_origin = target_coordinates_origin
        target_pos_range, target_radius_range = normalize_pointing_ranges(target_pos_range, target_radius_range)
        self.target_pos_range = jp.array(target_pos_range)
        self.target_radius_range = jp.array(target_radius_range)
        if dwell_steps < 1:
            raise ValueError("Sphere dwell time must cover at least one control timestep.")
        self.dwell_steps = dwell_steps
        self.weighted_reward_keys = weighted_reward_keys
        self.show_all_targets = show_all_targets
        self.target_name = target_name
        assert shape == "sphere", "Only sphere shapes are supported for now"
        self.shape = shape
        self.target_geom_id = None
        self.target_body_id = None
        self.min_distance = 2 * self.target_radius_range[1]
        self.enable_extra_dist = enable_extra_dist

    def sampler(self, sampler_info):
        rng, subkey = jax.random.split(sampler_info['rng'])
        sampled_pos = jax.random.uniform(subkey, (3,), minval=self.target_pos_range[0], maxval=self.target_pos_range[1])
        sampled_pos = self.target_coordinates_origin + sampled_pos
        return {**sampler_info, 'rng': rng, 'sampled_pos': sampled_pos, 'attempts': sampler_info['attempts'] + 1}

    def satistifies_cond(self, sampler_info):
        distance = jp.linalg.norm(sampler_info['sampled_pos'] - sampler_info['prev_pos'], axis=-1)
        return (distance < self.min_distance) & (sampler_info['attempts'] < 128)
    
    def generate_target_pos(self, rng: jax.random.PRNGKey, prev_pos: jp.ndarray = None):
        """Sample at most 128 times, then use a preflight-certified farthest corner."""
        if prev_pos is None:
            return self.sampler({'rng': rng, 'prev_pos': jp.zeros(3),
                                 'sampled_pos': jp.zeros(3), 'attempts': jp.array(0)})['sampled_pos']
        sampler_info = {'rng': rng, 'prev_pos': prev_pos, 'sampled_pos': prev_pos, 'attempts': jp.array(0)}
        sampled = jax.lax.while_loop(self.satistifies_cond, self.sampler, sampler_info)['sampled_pos']
        bounds = self.target_pos_range + self.target_coordinates_origin
        # Preflight validation guarantees that this farthest corner is valid for
        # every possible previous target position. Never relax target separation.
        corner = jp.where(jp.abs(bounds[0] - prev_pos) >= jp.abs(bounds[1] - prev_pos),
                          bounds[0], bounds[1])
        return jp.where(jp.linalg.norm(sampled - prev_pos) >= self.min_distance, sampled, corner)

    def generate_target_size(self, rng: jax.random.PRNGKey):
        target_size = jax.random.uniform(rng, minval=self.target_radius_range[0], maxval=self.target_radius_range[1])
        return target_size

    def generate_target(self, rng: jax.random.PRNGKey, prev_pos: jp.ndarray = None):
        rng, pos_rng, size_rng = jax.random.split(rng, 3)
        target_pos = self.generate_target_pos(pos_rng, prev_pos=prev_pos)
        target_size = self.generate_target_size(size_rng)
        return target_pos, target_size
    
    def add_to_spec(self, spec: mujoco.MjSpec, rng: jax.random.PRNGKey, rgb: List[float]):
        worldbody = spec.worldbody
        target_pos, target_size = self.generate_target(rng)
        target_body = worldbody.add_body(name=self.target_body_name, pos=target_pos)
        rgba = jp.ones(4)
        rgba = rgba.at[:3].set(rgb)
        target_size = jp.ones(3) * target_size
        target_geom = target_body.add_geom(name=self.target_geom_name, pos=jp.zeros(3), size=target_size, rgba=rgba)
        #print(f"Added target {self.target_geom_name} to spec")
        return spec

    def target_reset(self, info: Dict[str, Any]):
        if self.phase_number == 0:
            prev_pos = None
        else:
            prev_pos = info[f'target_{self.phase_number - 1}/target_pos']
        rng, rng_task = jax.random.split(info['rng'], 2)
        target_pos, target_size = self.generate_target(rng_task, prev_pos=prev_pos)
        info[f'target_{self.phase_number}/target_pos'] = target_pos
        info[f'target_{self.phase_number}/target_size'] = target_size
        info['rng'] = rng
        return info

    def get_task_obs_dict(self, info: Dict[str, Any], obs_dict: Dict[str, Any], advance_task=True):
        ee_pos = obs_dict['ee_pos']
        target_pos = info[f'target_{self.phase_number}/target_pos']
        target_size = info[f'target_{self.phase_number}/target_size']
        reach_dist = jp.linalg.norm(ee_pos - target_pos, axis=-1)
        inside_target = reach_dist < target_size
        extra_dist = self.compute_extra_dist(info)
        reach_dist = reach_dist + extra_dist
        _steps_inside_target = jp.select([inside_target], [info['steps_inside_target'] + jp.asarray(advance_task, dtype=jp.float32)], 0)
        phase_completed = 1.0 * (advance_task & (_steps_inside_target >= self.dwell_steps))
        dwell_fraction = -1. + 2. * jp.minimum(1., _steps_inside_target / self.dwell_steps)
        next_phase, task_completed = self.compute_next_phase(phase_completed, self.phase_number, self.total_phases)
        obs_dict['task_completed'] = task_completed
        obs_dict['phase'] = next_phase
        #TODO: Implement - max steps without hitting target
        obs_dict['target_pos'] = target_pos
        obs_dict['target_size'] = target_size
        obs_dict['reach_dist'] = reach_dist
        obs_dict['inside_target'] = 1.0 * inside_target
        obs_dict['steps_inside_target'] = _steps_inside_target
        obs_dict['phase_completed'] = phase_completed
        obs_dict['extra_dist'] = extra_dist
        obs_dict['touch_force'] = 0.0
        obs_dict['button_pressed'] = 0.0
        obs_dict['dwell_fraction'] = dwell_fraction
        return obs_dict

class ButtonTargetClass(TargetClass):
    def __init__(self, phase_number: int, total_phases: int,
                position: List[float],
                geom_size: List[float],
                site_size: List[float],
                site_pos: List[float],
                geom_margin: float,
                euler: List[float],
                min_touch_force: float,
                weighted_reward_keys: Dict[str, float],
                show_all_targets: bool = True,
                enable_extra_dist: bool = True):
        self.phase_number = phase_number
        self.total_phases = total_phases
        self.is_last_phase = phase_number == total_phases - 1
        self.position = position
        self.geom_size = geom_size
        self.site_size = site_size
        self.site_pos = site_pos
        self.geom_margin = geom_margin
        self.euler = euler
        self.min_touch_force = min_touch_force
        self.weighted_reward_keys = weighted_reward_keys
        self.show_all_targets = show_all_targets
        self.enable_extra_dist = enable_extra_dist

    @property
    def target_site_name(self):
        return f"target_site_{self.phase_number}"

    @property
    def sensor_name(self):
        return f"target_sensor_{self.phase_number}"
    
    def add_to_spec(self, spec: mujoco.MjSpec, rng: jax.random.PRNGKey, rgb: List[float]):
        worldbody = spec.worldbody
        target_body = worldbody.add_body(name=self.target_body_name, pos=self.position, euler=self.euler)
        rgba = jp.ones(4)
        rgba = rgba.at[:3].set(rgb)
        target_geom = target_body.add_geom(name=self.target_geom_name, type=mujoco._enums.mjtGeom(6), size=self.geom_size, margin=self.geom_margin, rgba=rgba, contype=1, conaffinity=1)
        target_site = target_body.add_site(name=self.target_site_name, type=mujoco._enums.mjtGeom(6), pos=self.site_pos, size=self.site_size)
        #print(f"Added target {self.target_geom_name} to spec")
        # Add touch sensor for the button
        sensor = spec.add_sensor(name=self.sensor_name, type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=self.target_site_name)
        #print(f"Added sensor {self.sensor_name} to spec")
        return spec

    def target_reset(self, info: Dict[str, Any]):
        info[f'target_{self.phase_number}/target_pos'] = self.position
        info[f'target_{self.phase_number}/target_size'] = self.geom_size[0] #TODO: Should be 3D size
        return info

    def get_task_obs_dict(self, info: Dict[str, Any], obs_dict: Dict[str, Any], advance_task=True):
        # Set target position for reach distance calculation
        ee_pos = obs_dict['ee_pos']
        target_pos = info[f'target_{self.phase_number}/target_pos']
        reach_dist = jp.linalg.norm(ee_pos - target_pos, axis=-1)
        extra_dist = self.compute_extra_dist(info)
        reach_dist = reach_dist + extra_dist
        # Get sensor data for touch detection
        touch_force = obs_dict['sensor_data'][self.sensor_id]
        
        # Check if button is pressed with sufficient force
        button_pressed = touch_force >= self.min_touch_force
        # For button targets, we consider "inside target" as button being pressed
        inside_target = button_pressed
        
        _steps_inside_target = jp.select([inside_target], [info['steps_inside_target'] + jp.asarray(advance_task, dtype=jp.float32)], 0)
        # For buttons, we can use a shorter dwell time or immediate completion
        dwell_steps = 1  # Immediate completion on button press
        phase_completed = 1.0 * (advance_task & (_steps_inside_target >= dwell_steps))
        dwell_fraction = -1. + 2. * jp.minimum(1., _steps_inside_target / dwell_steps)
        next_phase, task_completed = self.compute_next_phase(phase_completed, self.phase_number, self.total_phases)
        obs_dict['task_completed'] = task_completed
        obs_dict['phase'] = next_phase
        #TODO: Implement - max steps without hitting target
        obs_dict['target_pos'] = target_pos
        obs_dict['target_size'] = self.geom_size[0] #TODO: Should be 3D size
        obs_dict['reach_dist'] = reach_dist
        obs_dict['inside_target'] = 1.0 * inside_target
        obs_dict['steps_inside_target'] = _steps_inside_target
        obs_dict['phase_completed'] = phase_completed
        obs_dict['extra_dist'] = extra_dist
        obs_dict['touch_force'] = touch_force
        obs_dict['button_pressed'] = 1.0 * button_pressed
        obs_dict['dwell_fraction'] = dwell_fraction
        
        return obs_dict


class MyoUserUniversal(MyoUserBase): 

    def __init__(self, config, config_overrides=None):
        # Stored research checkpoints predate some task flags and reward terms.
        # Normalize a private copy so evaluation never rewrites their config or
        # adds features to the policy's explicitly recorded observation vector.
        config = deepcopy(config)
        defaults = asdict(UniversalTaskConfig())
        with config.unlocked():
            task = config.task_config
            provided_weights = 'weighted_reward_keys' in task
            for name, value in defaults.items():
                if name != 'targets' and name not in task:
                    task[name] = deepcopy(value)
            if provided_weights:
                for name in defaults['weighted_reward_keys']:
                    if name not in task.weighted_reward_keys:
                        task.weighted_reward_keys[name] = 0.0
        super().__init__(config, config_overrides)
    
    def add_task_relevant_geoms(self, spec: mujoco.MjSpec):
        targets = self._config.task_config.targets
        num_targets = targets.num_targets
        targets = [getattr(targets, f"target_{i}") for i in range(num_targets)]
        validate_target_sampling(targets, self.dt, self._config.task_config.reach_settings.target_origin_rel)
        total_phases = len(targets)
        self.target_objs = []
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        target_coordinates_origin = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._config.task_config.reach_settings.ref_site)]
        rng = jax.random.PRNGKey(self._config.task_config.target_init_seed)
        for i, target in enumerate(targets):
            rng, rng_init = jax.random.split(rng, 2)
            if target.name == "pointing_target":
                target_obj = PointingTargetClass(phase_number=i,
                    total_phases=total_phases,
                    target_pos_range=jp.array(target.position),
                    target_radius_range=jp.array(target.size),
                    target_coordinates_origin=target_coordinates_origin,
                    dwell_steps=max(1, math.ceil(target.dwell_duration/self.dt - 1e-9)),
                    weighted_reward_keys=self._config.task_config.weighted_reward_keys,
                    show_all_targets=self._config.task_config.show_all_targets,
                    enable_extra_dist=self._config.task_config.enable_extra_dist
                )
            elif target.name == "button_target":
                target_obj = ButtonTargetClass(phase_number=i,
                    total_phases=total_phases,
                    position=jp.array(target.position) + target_coordinates_origin,
                    geom_size=jp.array(target.geom_size),
                    site_size=jp.array(target.site_size),
                    site_pos=jp.array(target.site_pos),
                    geom_margin=target.geom_margin,
                    euler=jp.array(target.euler),
                    min_touch_force=target.min_touch_force,
                    weighted_reward_keys=self._config.task_config.weighted_reward_keys,
                    show_all_targets=self._config.task_config.show_all_targets,
                    enable_extra_dist=self._config.task_config.enable_extra_dist
                )
            else:
                raise ValueError(f"Unsupported target type: {target['name']}")
            spec = target_obj.add_to_spec(spec, rng_init, target['rgb'])
            self.target_objs.append(target_obj)
        #print(f"Added {len(self.target_objs)} targets to spec")
        return spec

    def preprocess_spec(self, spec: mujoco.MjSpec):
        spec = self.add_task_relevant_geoms(spec)
        return super().preprocess_spec(spec)

    def model_render(self, camera: str):
        model = self.mj_model
        data = mujoco.MjData(model)

        with mujoco.Renderer(model, height=480, width=640) as renderer:
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            img = renderer.render()
        return img
    
    def get_renderings(self, cameras: List[str] = ['head_on_camera', 'side_camera']):
        return [(camera, self.model_render(camera)) for camera in cameras]


    def _setup(self):
        super()._setup()
        self.reach_settings = self._config.task_config.reach_settings
        self.obs_keys = list(self._config.task_config.obs_keys)
        self.omni_keys = list(self._config.task_config.omni_keys)
        
        if not self._config.vision.enabled:
            print(f"No vision, so adding {self.omni_keys} to obs_keys")
            self.obs_keys.extend(self.omni_keys)
        else:
            print(f"Vision, so not adding {self.omni_keys} to obs_keys")
        print(f"Obs keys: {self.obs_keys}")
        self.ee_pos_id = self.mj_model.site('fingertip').id
        self.non_accumulation_metrics = ['distance_to_target', ] + [f'target_{i}_completed' for i in range(len(self.target_objs)-1)]
        self.accumulation_metrics = ['target_final_completed', 'jac_effort_reward', 'phase_bonus_reward', 'done_reward', 'neural_effort_reward', 'distance_reward']


    def _prepare_after_init(self, data):
        super()._prepare_after_init(data)
        # Define target origin, relative to which target positions will be generated
        self.target_coordinates_origin = data.site_xpos[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.reach_settings.ref_site)].copy() + jp.array(self.reach_settings.target_origin_rel)  #jp.zeros(3,)
        for target_obj in self.target_objs:
            target_obj.target_coordinates_origin = self.target_coordinates_origin
            target_obj.target_body_id = self.mj_model.body(target_obj.target_body_name).id
            target_obj.target_geom_id = self.mj_model.geom(target_obj.target_geom_name).id
            # For button targets, also get the sensor ID
            if isinstance(target_obj, ButtonTargetClass):
                target_obj.sensor_id = self.mj_model.sensor(target_obj.sensor_name).id
        self.num_targets = len(self.target_objs)

    def auto_reset(self, state_before_reset):
        rng = state_before_reset.info['rng']
        return self.reset(rng, render_token=state_before_reset.info.get('render_token'))

    def task_render_inputs(self, data, info):
        """Build visual transforms and sizes without changing collision physics."""
        positions = data.geom_xpos
        sizes = self.mjx_model.geom_size
        colors = self.mjx_model.geom_rgba
        materials = self.mjx_model.geom_matid
        phase = info['phase']
        for target in self.target_objs:
            geom_id = target.target_geom_id
            positions = positions.at[geom_id].set(info[f'target_{target.phase_number}/target_pos'])
            if isinstance(target, PointingTargetClass):
                size = jp.full(3, info[f'target_{target.phase_number}/target_size'])
            else:
                size = target.geom_size
            if target.show_all_targets:
                color = jp.where(phase == target.phase_number, jp.array([0., 1., 0., 1.]),
                                 jp.where(phase > target.phase_number, jp.array([0., 0., 1., 0.5]),
                                          jp.array([1., 0., 0., 0.5])))
                colors = colors.at[geom_id].set(color)
            else:
                size = jp.where(phase == target.phase_number, size, jp.zeros(3))
            sizes = sizes.at[geom_id].set(size)
            # Madrona's color override material uses the current geom RGB.
            materials = materials.at[geom_id].set(-2)
        visual_model = self.mjx_model.replace(geom_size=sizes, geom_rgba=colors, geom_matid=materials)
        return data.replace(geom_xpos=positions), visual_model

    def generate_task_pixels(self, data, info, render_token=None):
        render_data, render_model = self.task_render_inputs(data, info)
        # Madrona.render updates transforms only. Its init primitive also copies
        # model sizes/colors, which change on reset and when the active phase changes.
        return self.generate_pixels(render_data, render_token, model=render_model, refresh_model=True)

    def get_obs_vec_aux_task(self, obs_dict, obsvec):
        return {
            'proprioception': obsvec,
            'pixels/depth': obs_dict['pixels/depth'],
            'vision_aux_targets': jp.concatenate([
                obs_dict['target_pos'].ravel(), jp.asarray(obs_dict['target_size']).reshape(1)
            ]),
        }

    def reset(self, rng, render_token=None, add_to_info=None):
        _, rng = jax.random.split(rng, 2)

        data = self._reset_bm_model(rng)

        last_ctrl = jp.zeros(self._nu)

        info = {"rng": rng,
                "last_ctrl": last_ctrl,
                "steps_inside_target": jp.array(0.),
                'reach_dist': jp.array(0.),
                'target_success': jp.array(False),
                'target_fail': jp.array(False),
                'task_completed': jp.array(False),
                'phase': jp.array(0),
                'num_phases': jp.array(len(self.target_objs)),
                'time_in_current_phase': jp.array(0.),
                'extra_dist': jp.array(0.),
                'ee_pos': jp.array(data.site_xpos[self.ee_pos_id]),
                }
        
        if add_to_info is not None:
            info.update(add_to_info)
        
        for target_obj in self.target_objs:
            update_info = target_obj.target_reset(info)
            info.update(update_info)
        
        info['target_pos'] = info['target_0/target_pos']
        info['target_size'] = info['target_0/target_size']        
        if self.vision:
            info.update(self.generate_task_pixels(data, info, render_token))
        reward, done = jp.zeros(2)
        metrics = {metric: 0.0 for metric in self.non_accumulation_metrics}
        metrics.update({metric: 0.0 for metric in self.accumulation_metrics})
        obs, info = self.get_obs_vec(data, info)
        return State(data, obs, reward, done, metrics, info)

    def get_obs_dict(self, data, info, advance_task=False):

        obs_dict = {}
        obs_dict['time'] = jp.array(data.time)
        
        # Normalise qpos
        jnt_range = self._mj_model.jnt_range[self._independent_joints]
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

        # Store current control input
        obs_dict['last_ctrl'] = data.ctrl.copy()
        obs_dict['ee_pos'] = data.site_xpos[self.ee_pos_id]
        obs_dict['sensor_data'] = data.sensordata
        
        # targets hit will be -1 for the first target and (technically but will never be computed as ) 1 once the last target is hit
        obs_dict['phase_progress'] = - 1. + 2. * (info['phase'] / self.num_targets)

        obs_funcs = [t.get_task_obs_dict for t in self.target_objs]
        obs_dict = jax.lax.switch(info['phase'], obs_funcs, info, obs_dict, advance_task)
        
        return obs_dict

    def update_info(self, info, obs_dict):
        info['last_ctrl'] = obs_dict['last_ctrl']
        info['steps_inside_target'] = jp.where(obs_dict['phase_completed'], 0., obs_dict['steps_inside_target'])
        info['reach_dist'] = obs_dict['reach_dist']
        info['target_pos'] = obs_dict['target_pos']
        info['target_size'] = obs_dict['target_size']
        info['phase'] = obs_dict['phase']
        info['extra_dist'] = obs_dict['extra_dist']
        info['ee_pos'] = obs_dict['ee_pos']
        return info

    def task_completed_metrics(self, phase: int):
        ### Called if task has been completed and sets all the metrics to 1
        return {metric: 1.0 for metric in self.default_metrics(phase).keys()}

    def default_metrics(self, phase: int):
        ### Called if task hasn't been completed and sets the phase_complete_metrics according to the current phase
        return {
            f'target_{i}_completed': 1.0 * (phase > i) for i in range(len(self.target_objs)-1)
        }
    
    def eval_metrics(self, rollout):
        results = []
        for i in range(len(rollout)):
            episode_rollout = rollout[i]

            completed = {
                phase_idx
                for idx, entry in enumerate(episode_rollout)
                for phase_idx in range(len(self.target_objs))
                if entry.metrics.get(
                    'target_final_completed' if phase_idx == len(self.target_objs) - 1
                    else f"target_{phase_idx}_completed", False
                )
            }

            for i in range(len(self.target_objs)):
                results.append({
                    "success": i in completed,
                    "target_size": episode_rollout[0].info[f"target_{i}/target_size"],
                    "target_pos": episode_rollout[0].info[f"target_{i}/target_pos"],
                    # "time_in_current_phase": episode_rollout[0].info["time_in_current_phase"]
                })
        return results


    def get_jac_effort_costs(self, obs_dict):
        r_effort = 0.00198*jp.linalg.norm(obs_dict['last_ctrl'])**2 
        r_jacc = 6.67e-6*jp.linalg.norm(obs_dict['qacc'])**2

        effort_cost = r_effort + r_jacc
        
        return effort_cost

    def get_reward_dict(self, obs_dict: Dict[str, Any]):
        ctrl_magnitude = jp.linalg.norm(obs_dict['last_ctrl'], axis=-1)
        if self._config.task_config.exponential_reach_reward:
            inside_target = obs_dict['inside_target']
            dist = obs_dict['reach_dist']
            rm = self._config.task_config.reach_metric
            print(f"Using exponential reach reward with reach metric {rm}")
            reach_reward = (1. - inside_target) * (jp.exp(-dist*rm) - 1.)/rm
        else:
            print(f"Using linear reach reward")
            reach_reward = -1.*obs_dict['reach_dist']

        rwd_dict = collections.OrderedDict((
            ('reach', reach_reward),
            ('neural_effort', -1.*(ctrl_magnitude ** 2)),
            ('jac_effort', -1.* self.get_jac_effort_costs(obs_dict)),
            ('phase_bonus', 1.*obs_dict['phase_completed']),
            ('done', 1.*obs_dict['task_completed']),
        ))

        rwd_dict['dense'] = jp.sum(jp.array([wt*rwd_dict[key] for key, wt in self._config.task_config.weighted_reward_keys.items()]), axis=0)
        return rwd_dict

    def step(self, state: State, action: jp.ndarray) -> State:
        rng = state.info['rng']
        data0 = state.data
        rng, rng_ctrl = jax.random.split(rng, 2)
        new_ctrl = self.get_ctrl(state, action, rng_ctrl)
        data = mjx_env.step(self.mjx_model, data0, new_ctrl, n_substeps=self.n_substeps)
        target_step_fns = [t.target_step for t in self.target_objs]
        state = jax.lax.switch(state.info['phase'], target_step_fns, state)

        obs_dict = self.get_obs_dict(data, state.info, advance_task=True)

        # rwd_funcs = [t.get_task_reward_dict for t in self.target_objs]
        rwd_dict = self.get_reward_dict(obs_dict)

        _updated_info = self.update_info(state.info, obs_dict)
        # Rewards describe the transition just taken; the next policy observation
        # must describe its new active target without advancing dwell time again.
        observation_dict = self.get_obs_dict(data, _updated_info)
        _updated_info = self.update_info(_updated_info, observation_dict)
        if self.vision:
            _updated_info.update(self.generate_task_pixels(data, _updated_info, _updated_info['render_token']))
            observation_dict = self.update_obs_with_pixels(observation_dict, _updated_info)
        obs = self.obsdict2obsvec(observation_dict)
        state = state.replace(info=_updated_info)
        _, state.info['rng'] = jax.random.split(rng, 2)
        done = rwd_dict['done']

        metric_options = [self.default_metrics, self.task_completed_metrics]
        phase_complete_metrics = jax.lax.switch(done.astype(jp.int32), metric_options, obs_dict['phase'])

        reward_weights = {
            'reach': jp.array([t.weighted_reward_keys['reach'] for t in self.target_objs]),
            'neural_effort': jp.array([t.weighted_reward_keys['neural_effort'] for t in self.target_objs]),
            'jac_effort': jp.array([t.weighted_reward_keys['jac_effort'] for t in self.target_objs]),
            'phase_bonus': jp.array([t.weighted_reward_keys['phase_bonus'] for t in self.target_objs]),
            'done': jp.array([t.weighted_reward_keys['done'] for t in self.target_objs]),
        }
        current_weights = {
            'reach': reward_weights['reach'][obs_dict['phase']],
            'neural_effort': reward_weights['neural_effort'][obs_dict['phase']],
            'jac_effort': reward_weights['jac_effort'][obs_dict['phase']],
            'phase_bonus': reward_weights['phase_bonus'][obs_dict['phase']],
            'done': reward_weights['done'][obs_dict['phase']],
        }

        metrics = {
            'distance_to_target': obs_dict['reach_dist'],
            'target_final_completed': obs_dict['task_completed'],
            'distance_reward': rwd_dict['reach']*current_weights['reach'],
            'phase_bonus_reward': rwd_dict['phase_bonus']*current_weights['phase_bonus'],
            'done_reward': rwd_dict['done']*current_weights['done'],
            'neural_effort_reward': rwd_dict['neural_effort']*current_weights['neural_effort'],
            'jac_effort_reward': rwd_dict['jac_effort']*current_weights['jac_effort'],
            **phase_complete_metrics,
        }

        return state.replace(
            data=data, obs=obs, reward=rwd_dict['dense'], done=done, metrics=metrics
        )

    def update_task_visuals(self, mj_model, state):
        phase = state.info['phase']
        for target_obj in self.target_objs:
            target_obj.update_task_visuals(mj_model, state, phase)
