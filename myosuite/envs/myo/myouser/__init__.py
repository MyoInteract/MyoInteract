""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """


from mujoco_playground import registry
from mujoco_playground._src.manipulation import _randomizer
## Register MyoUser environments and their variants
from myosuite.envs.myo.myouser.myouser_universal import MyoUserUniversal


# registry.manipulation.register_environment("MyoElbow", PlaygroundElbow, default_config)
registry.manipulation.register_environment("MyoUserUniversal", MyoUserUniversal, {})

## TODO: register variants
## WARNING: deprecated code, relies on outdated myosuite register_env_variant rather than mujoco_playground

from myosuite.utils import gym; register=gym.register
from myosuite.envs.env_variants import register_env_variant

# from .base import MyoUserBase

import os, sys
import numpy as np

# utility to register envs with all muscle conditions
def register_env_with_variants(id, entry_point, max_episode_steps, kwargs, pre_code=""):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )
    #register variants env with sarcopenia
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue'},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )

    #register variants with tendon transfer
    if id[:7] == "myoHand":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'reafferentation'},
            variant_id=id[:3]+"Reaf"+id[3:],
            silent=True
        )

def _register(id, entry_point, max_episode_steps, kwargs, pre_code=""):
    exec(pre_code)

    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )

curr_dir = os.path.dirname(os.path.abspath(__file__))

print("MyoSuite:> Registering MyoUser Envs")

# UitB LLC (direct UitB compatibility test)  ==============================
_register(id='myoarm_llc_eepos_pointing_adaptive_v101-v0',
        entry_point='myosuite.envs.myo.myouser.myoarm_llc_eepos_pointing_adaptive_v101__simulator:Simulator',
        max_episode_steps=80,  #200,
        kwargs={
            'simulator_folder': curr_dir+'/myoarm_llc_eepos_pointing_adaptive_v101/',
        #     'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
        #     'target_reach_range': {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),},
        #     'normalize_act': True,
        #     'frame_skip': 5,
        },
        pre_code='sys.path.insert(0, r"C:\\Users\\Florian\\Research\\uitb-sim2vr\\user-in-the-box\\simulators\\myoarm_llc_eepos_pointing_adaptive_v101")'
    )

# UitB LLC (reimplementation) ==============================
register_env_with_variants(id='myoElbow_llc_eepos_adaptive-v0',
        entry_point='myosuite.envs.myo.myouser.llc_eepos_adaptive_v0:LLCEEPosAdaptiveEnvV0',
        max_episode_steps=80,  #100,
        kwargs={
            'model_path': curr_dir+'/../assets/elbow/myoelbow_1dof6muscles.xml',
            'frame_skip': 25,
            'target_pos_range': {'wrist': ((-0.3, -0.35, -0.3), (0.1, -0.225, 0.4)),},
            'target_radius_range': {'wrist': ((0.01, 0.15)),},
            # 'ref_site': 'base',
            'adaptive_task': True,
            # 'init_target_area_width_scale': 0,
            # 'adaptive_increase_success_rate': 0.6,
            # 'adaptive_decrease_success_rate': 0.3,
            # 'adaptive_change_step_size': 0.05,
            # 'adaptive_change_min_trials': 50,
            'adaptive_change_trial_buffer_length': 500,
            # 'normalize_act': True,
            'reset_type': 'range_uniform'
        }
    )

register_env_with_variants(id='myoArm_llc_eepos_adaptive_direct_ctrl-v0',
        entry_point='myosuite.envs.myo.myouser.llc_eepos_adaptive_v0:LLCEEPosAdaptiveDirectCtrlEnvV0',
        max_episode_steps=80,  #100,
        kwargs={
            'model_path': curr_dir+'/../assets/arm/myoarm_pose.xml',
            'frame_skip': 25,
            'target_pos_range': {'IFtip': ((-0.3, -0.35, -0.3), (0.1, -0.225, 0.4)),},
            'target_radius_range': {'IFtip': ((0.01, 0.15)),},
            'ref_site': 'R.Shoulder_marker',
            'adaptive_task': True,
            # 'init_target_area_width_scale': 0,
            # 'adaptive_increase_success_rate': 0.6,
            # 'adaptive_decrease_success_rate': 0.3,
            # 'adaptive_change_step_size': 0.05,
            # 'adaptive_change_min_trials': 50,
            'adaptive_change_trial_buffer_length': 500,
            # 'normalize_act': True,
            'reset_type': 'range_uniform'
        }
    )

register_env_with_variants(id='myoArm_llc_eepos_adaptive-v0',
        entry_point='myosuite.envs.myo.myouser.llc_eepos_adaptive_v0:LLCEEPosAdaptiveEnvV0',
        max_episode_steps=80,  #100,
        kwargs={
            'model_path': curr_dir+'/../assets/arm/myoarm_pose.xml',
            'frame_skip': 25,
            'target_pos_range': {'IFtip': ((-0.3, -0.35, -0.3), (0.1, -0.225, 0.4)),},
            'target_radius_range': {'IFtip': ((0.01, 0.15)),},
            'ref_site': 'R.Shoulder_marker',
            'adaptive_task': True,
            # 'init_target_area_width_scale': 0,
            # 'adaptive_increase_success_rate': 0.6,
            # 'adaptive_decrease_success_rate': 0.3,
            # 'adaptive_change_step_size': 0.05,
            # 'adaptive_change_min_trials': 50,
            'adaptive_change_trial_buffer_length': 500,
            # 'normalize_act': True,
            'reset_type': 'range_uniform'
        }
    )

register_env_with_variants(id='mobl_arms_index_llc_eepos_adaptive-v0',
        entry_point='myosuite.envs.myo.myouser.llc_eepos_adaptive_v0:LLCEEPosAdaptiveEnvV0',
        max_episode_steps=80,  #100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/uitb_sim/mobl_arms_index_llc_eepos_pointing.xml',
            'frame_skip': 25,
            'target_pos_range': {'fingertip': ((0.225, -0.3, -0.3), (0.35, 0.1, 0.4)),},
            'target_radius_range': {'fingertip': ((0.01, 0.15)),},
            'ref_site': 'humphant',
            'adaptive_task': True,
            # 'init_target_area_width_scale': 0,
            # 'adaptive_increase_success_rate': 0.6,
            # 'adaptive_decrease_success_rate': 0.3,
            # 'adaptive_change_step_size': 0.05,
            # 'adaptive_change_min_trials': 50,
            'adaptive_change_trial_buffer_length': 500,
            # 'normalize_act': True,
            'reset_type': 'range_uniform'
        }
    )


### MJX variants
register_env_with_variants(id='myoArm_llc_eepos_adaptive_mjx-v0',
        entry_point='myosuite.envs.myo.myouser.llc_eepos_adaptive_mjx_v0:LLCEEPosAdaptiveEnvMJXV0',
        max_episode_steps=80,  #100,
        kwargs={
            'model_path': curr_dir+'/../assets/arm/myoarm_pose.xml',
            'frame_skip': 25,
            'target_pos_range': {'IFtip': ((-0.3, -0.35, -0.3), (0.1, -0.225, 0.4)),},
            'target_radius_range': {'IFtip': ((0.01, 0.15)),},
            'ref_site': 'R.Shoulder_marker',
            'adaptive_task': True,
            # 'init_target_area_width_scale': 0,
            # 'adaptive_increase_success_rate': 0.6,
            # 'adaptive_decrease_success_rate': 0.3,
            # 'adaptive_change_step_size': 0.05,
            # 'adaptive_change_min_trials': 50,
            'adaptive_change_trial_buffer_length': 500,
            # 'normalize_act': True,
            'reset_type': 'range_uniform'
        }
    )

register_env_with_variants(id='mobl_arms_index_llc_eepos_adaptive_mjx-v0',
        entry_point='myosuite.envs.myo.myouser.llc_eepos_adaptive_mjx_v0:LLCEEPosAdaptiveEnvMJXV0',
        max_episode_steps=80,  #100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/uitb_sim/mobl_arms_index_llc_eepos_pointing.xml',
            'frame_skip': 25,
            'target_pos_range': {'fingertip': ((0.225, -0.3, -0.3), (0.35, 0.1, 0.4)),},
            'target_radius_range': {'fingertip': ((0.01, 0.15)),},
            'ref_site': 'humphant',
            'adaptive_task': True,
            # 'init_target_area_width_scale': 0,
            # 'adaptive_increase_success_rate': 0.6,
            # 'adaptive_decrease_success_rate': 0.3,
            # 'adaptive_change_step_size': 0.05,
            # 'adaptive_change_min_trials': 50,
            'adaptive_change_trial_buffer_length': 500,
            # 'normalize_act': True,
            'reset_type': 'range_uniform'
        }
    )