import numpy as np
import mujoco

from .reward_functions import NegativeExpDistanceWithHitBonus
from ..base import BaseTask


class EndEffectorPosLowLevelControllerAdaptive(BaseTask):

  def __init__(self, model, data, end_effector, shoulder, **kwargs):
    super().__init__(model, data, **kwargs)

    # This task requires an end-effector to be defined
    if not isinstance(end_effector, list) and len(end_effector) != 2:
      raise RuntimeError("'end_effector' must be a list with two elements: first defining what type of mujoco element "
                         "it is, and second defining the name")
    self._end_effector = end_effector

    # Also a shoulder that is used to define the location of target plane
    if not isinstance(shoulder, list) and len(shoulder) != 2:
      raise RuntimeError("'shoulder' must be a list with two elements: first defining what type of mujoco element "
                         "it is, and second defining the name")
    self._shoulder = shoulder

    # Use early termination if target is not hit in time
    self._steps_since_last_hit = 0
    self._max_steps_without_hit = self._action_sample_freq*4  #corresponds to 4 seconds

    # Used for logging states
    self._info = {"target_hit": False, "inside_target": False, "target_spawned": False, "terminated": False,
                  "truncated": False, "termination": False,
                  "success_rate": 0, "target_area_dynamic_width_scale": 0}

    # Define a maximum number of trials per episode (if needed for e.g. evaluation / visualisation)
    self._trial_idx = 0  #number of total trials since last reset
    self._max_trials = kwargs.get('max_trials', 10)
    self._targets_hit = 0

    # Dwelling based selection -- fingertip needs to be inside target for some time
    self._steps_inside_target = 0
    self._dwell_threshold = int(0.25*self._action_sample_freq)  #for visual-based pointing: int(0.5*self._action_sample_freq)

    # Radius limits for target
    self._target_radius_limit = kwargs.get('target_radius_limit', np.array([0.01, 0.15]))
    self._target_radius = self._target_radius_limit[1]

    # Minimum distance to new spawned targets is twice the max target radius limit
    self._new_target_distance_threshold = 0  #2*self._target_radius_limit[1]

    # Define a default reward function
    #if self.reward_function is None:
    self._reward_function = NegativeExpDistanceWithHitBonus(k=10)

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Define plane where targets will move: 0.55m in front of and 0.1m to the right of shoulder, or the "humphant" body.
    # Note that this body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    ## NOTE: For MyoArm, x-axis points to left, y-axis to back, and z-axis to top (i.e., -90° rotated around z-axis compared to UitB models) -> 'target_origin_rel', 'target_limits_x', 'target_limits_y' and 'target_limits_z' need to be adjusted below!
    self._target_origin = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos + np.array(kwargs.get('target_origin_rel', [0.225, -0.1, 0.05]))
    self._target_position = self._target_origin.copy()
    self._target_limits_x = np.array(kwargs.get('target_limits_x', [0, 0.125]))
    self._target_limits_y = np.array(kwargs.get('target_limits_y', [-0.2, 0.2]))
    self._target_limits_z = np.array(kwargs.get('target_limits_z', [-0.35, 0.35]))

    # Additional variables needed for adaptive adjustment of target limits based on success rates since last target limit adjustment
    self.target_area_dynamic_width_scale = kwargs.get('init_target_area_width_scale', 0)  #scale factor for target area width (between 0 and 1), i.e., the percentage of the target limit range defined above that is currently used when spawning targets
    self.update_target_limits()
    self.adaptive_increase_success_rate = kwargs.get('adaptive_increase_success_rate', 0.9)  #success rate above which target area width is increased
    self.adaptive_decrease_success_rate = kwargs.get('adaptive_decrease_success_rate', 0.5)  #success rate below which target area width is decreased
    self.adaptive_change_step_size = kwargs.get('adaptive_change_step_size', 0.05)  #increase of target area width per adjustment (in meter)
    self.adaptive_change_min_trials = kwargs.get('adaptive_change_min_trials', 50)  #minimum number of trials with the latest target area width required before the next adjustment; should be chosen considerably larger than self._max_trials
    self.adaptive_change_trial_buffer_length = kwargs.get('adaptive_change_trial_buffer_length', None)  #maximum number of trials (since last adjustment) to consider for success rate calculation (default: consider all values since last adjustment)
    assert self.adaptive_change_min_trials >= 1, f"At least one trial is required to assess the success rate for adaptively adjusting the target area width. Set 'adaptive_change_min_trials' >= 1 (current value: {self.adaptive_change_min_trials})."
    self._trial_success_log = []  #logging whether trials since last target limit adjustment were successful
    self.n_hits_adj = 0  #number of successful trials since last target limit adjustment
    self.n_targets_adj = 0  #number of total trials since last target limit adjustment
    self.n_adjs = 0  #number of target limit adjustments
    self._success_rate = 0  #previous success rate

  def check_adaptive_target_area_width(self):
    if self.adaptive_change_trial_buffer_length is not None:
      # Clip success/fail buffer
      self._trial_success_log = self._trial_success_log[-self.adaptive_change_trial_buffer_length:]

    self.n_targets_adj = len(self._trial_success_log)
    if self.n_targets_adj >= self.adaptive_change_min_trials:
      
      self.n_hits_adj = sum(self._trial_success_log)
      self._success_rate = self.n_hits_adj / self.n_targets_adj
      # print(f"SUCCESS RATE: {self._success_rate*100}% ({self.n_hits_adj}/{self.n_targets_adj}) -- Last Adj. #{self.n_adjs}")

      if (self._success_rate >= self.adaptive_increase_success_rate) and (self.target_area_dynamic_width_scale < 1):
        new_target_area_width = self.target_area_dynamic_width_scale + self.adaptive_change_step_size
        self.update_adaptive_target_area_width(new_target_area_width)
      elif (self._success_rate <= self.adaptive_decrease_success_rate) and (self.target_area_dynamic_width_scale > 0):
        new_target_area_width = self.target_area_dynamic_width_scale - self.adaptive_change_step_size
        self.update_adaptive_target_area_width(new_target_area_width)

  def update_adaptive_target_area_width(self, new_target_area_width):
    self.n_adjs += 1
    print(f"ADAPTIVE TARGETS -- Adj. #{self.n_adjs}: {self.target_area_dynamic_width_scale*100}% -> {new_target_area_width*100}% (success_rate={self._success_rate})")

    # Reset internally used counters
    self._trial_success_log = []
    self.n_hits_adj = 0  #TODO: remove (useless)
    self.n_targets_adj = 0  #TODO: remove (useless)

    self.target_area_dynamic_width_scale = new_target_area_width
    # Update target limits
    self.update_target_limits()

  def update_target_limits(self):
    self._target_limits_adaptive_x = self.target_area_dynamic_width_scale*(self._target_limits_x - np.mean(self._target_limits_x)) + np.mean(self._target_limits_x)
    self._target_limits_adaptive_y = self.target_area_dynamic_width_scale*(self._target_limits_y - np.mean(self._target_limits_y)) + np.mean(self._target_limits_y)
    self._target_limits_adaptive_z = self.target_area_dynamic_width_scale*(self._target_limits_z - np.mean(self._target_limits_z)) + np.mean(self._target_limits_z)

  def _update(self, model, data):

    # Set some defaults
    terminated = False
    truncated = False
    self._info["target_spawned"] = False

    # Get end-effector position
    ee_position = getattr(data, self._end_effector[0])(self._end_effector[1]).xpos

    # Distance to target
    dist = np.linalg.norm(self._target_position - (ee_position - self._target_origin))

    # Check if fingertip is inside target
    if dist < self._target_radius:
      self._steps_inside_target += 1
      self._info["inside_target"] = True
    else:
      self._steps_inside_target = 0
      self._info["inside_target"] = False

    if self._info["inside_target"] and self._steps_inside_target >= self._dwell_threshold:

      # Update counters
      self._info["target_hit"] = True
      self._trial_idx += 1
      self._targets_hit += 1
      # self.n_hits_adj += 1
      self._trial_success_log += [1]
      self._steps_since_last_hit = 0
      self._steps_inside_target = 0
      self._info["acc_dist"] += dist
      self._spawn_target(model, data)
      self._info["target_spawned"] = True

    else:

      self._info["target_hit"] = False

      # Check if time limit has been reached
      self._steps_since_last_hit += 1
      if self._steps_since_last_hit >= self._max_steps_without_hit:
        self._trial_success_log += [0]
        # Spawn a new target
        self._steps_since_last_hit = 0
        self._trial_idx += 1
        self._info["acc_dist"] += dist
        self._spawn_target(model, data)
        self._info["target_spawned"] = True

    # Check if max number trials reached
    if self._trial_idx >= self._max_trials:
      self._info["dist_from_target"] = self._info["acc_dist"]/self._trial_idx
      truncated = True
      self._info["termination"] = "max_trials_reached"

    # Update info with variables to log in wandb
    self._info["target_area_dynamic_width_scale"] = self.target_area_dynamic_width_scale
    self._info["success_rate"] = self._success_rate

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # fingertip is inside target
    reward = self._reward_function.get(self, dist-self._target_radius, self._info.copy())

    return reward, terminated, truncated, self._info.copy()

  def _get_state(self, model, data):
    state = dict()
    state["target_position"] = self._target_origin.copy()+self._target_position.copy()
    state["target_radius"] = self._target_radius
    state["trial_idx"] = self._trial_idx
    state["targets_hit"] = self._targets_hit
    state.update(self._info)
    return state

  def _reset(self, model, data):

    # Reset counters
    self._steps_since_last_hit = 0
    self._steps_inside_target = 0
    self._trial_idx = 0
    self._targets_hit = 0

    self._info = {"target_hit": False, "inside_target": False, "target_spawned": False,
                  "terminated": False, "truncated": False, "termination": False, "llc_dist_from_target": 0, "dist_from_target": 0, "acc_dist": 0,
                  "success_rate": self._success_rate, "target_area_dynamic_width_scale": self.target_area_dynamic_width_scale}

    # Spawn a new location
    self._spawn_target(model, data)

    return self._info
  
  def get_target_position(self):
    return self._target_origin + self._target_position
  
  def get_target_radius(self):
    return self._target_radius

  def _spawn_target(self, model, data, new_position=None, new_radius=None):
    # Check if target area width should be updated
    self.check_adaptive_target_area_width()
    # self.n_targets_adj += 1

    if new_position is None:
      # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
      for _ in range(10):
        target_x = self._rng.uniform(*self._target_limits_adaptive_x)
        target_y = self._rng.uniform(*self._target_limits_adaptive_y)
        target_z = self._rng.uniform(*self._target_limits_adaptive_z)
        new_position = np.array([target_x, target_y, target_z])
        distance = np.linalg.norm(self._target_position - new_position)
        if distance > self._new_target_distance_threshold:
          break
    self._target_position = new_position

    # Set location
    model.body("target").pos[:] = self.get_target_position()

    if new_radius is None:
      # Sample target radius
      new_radius = self._rng.uniform(*self._target_radius_limit)
    self._target_radius = new_radius
    
    # Set target radius
    model.geom("target").size[0] = self._target_radius

    mujoco.mj_forward(model, data)

  def get_stateful_information(self, model, data):
    # # Time features (time left to reach target, time spent inside target)
    # targets_hit = -1.0 + 2*(self._trial_idx/self._max_trials)
    # dwell_time = -1.0 + 2 * np.min([1.0, self._steps_inside_target / self._dwell_threshold])
    # return np.array([dwell_time, targets_hit])

    return np.hstack((self.get_target_position(), self.get_target_radius()))
