import gradio as gr
import os
import wandb
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

if gr.NO_RELOAD:

    from gradio_rangeslider import RangeSlider
    from myosuite.envs.myo.myouser.myouser_universal import MyoUserUniversal
    from myosuite.envs.myo.myouser.train_jax_ppo import train
    from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
    import numpy as np
    import myosuite
    from pathlib import Path
    import json
    from dataclasses import dataclass, field
    from typing import List
    myosuite_path = Path(myosuite.__path__[0])


sphere_ranges = {
    "x": (0.2, 0.55),
    "y": (-0.25, 0.45),
    "z": (-0.3, 0.3),
    "size": (0.005, 0.15),  #radius
}

INIT_ELEMENTS = 2


parent_path = myosuite_path.parent
CHECKPOINT_PATH = os.path.join(parent_path, "tracked_checkpoints/universal")
LOGS_PATH = os.path.join(parent_path, "logs")
os.makedirs(LOGS_PATH, exist_ok=True)

@dataclass
class RunState:
    wandb_run_name: str = ""
    cfg_overrides: List[str] = field(default_factory=lambda:[])

def checkpoints_diff_folders():
    tracked_checkpoints = os.listdir(CHECKPOINT_PATH)
    logs_checkpoints = os.listdir(LOGS_PATH)
    return tracked_checkpoints, logs_checkpoints

def get_available_checkpoints():
    tracked_checkpoints, logs_checkpoints = checkpoints_diff_folders()
    checkpoints = tracked_checkpoints + logs_checkpoints + ["None"]
    return checkpoints


def is_number(s: str):
    if ("." not in s) and s.isdigit():
        return True
    return False
    
def get_available_checkpoint_numbers(checkpoint_run):
    if checkpoint_run == "None":
        return ["None"]
    tracked_checkpoints, logs_checkpoints = checkpoints_diff_folders()
    if checkpoint_run in tracked_checkpoints:
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_run)
    elif checkpoint_run in logs_checkpoints:
        checkpoint_path = os.path.join(LOGS_PATH, checkpoint_run)
    else:
        return ["None"]
    checkpoint_path = os.path.join(checkpoint_path, "checkpoints")
    available_numbers = [num for num in os.listdir(checkpoint_path) if is_number(num)]
    return available_numbers


def checkpoint_path_from_run_number(checkpoint_run, checkpoint_number):
    if checkpoint_run == "None" or checkpoint_number == "None":
        return "None"
    tracked_checkpoints, logs_checkpoints = checkpoints_diff_folders()
    if checkpoint_run in tracked_checkpoints:
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_run)
    elif checkpoint_run in logs_checkpoints:
        checkpoint_path = os.path.join(LOGS_PATH, checkpoint_run)
    else:
        return "None"
    checkpoint_path = os.path.join(checkpoint_path, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_path, checkpoint_number)
    return checkpoint_path

def update_checkpoint_numbers(checkpoint_run):
    choices = get_available_checkpoint_numbers(checkpoint_run)
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
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
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
    rgbs =[el*255 for el in target_rgb[i]]
    return f"rgba({rgbs[0]},{rgbs[1]},{rgbs[2]},1)"

def sanity_check_float(value, name):
    try:
        float(value)
    except Exception as e:
        gr.Warning(f"{name} must be a float, got {value}")
        return False
    return True

def sanity_check_bool(value, name):
    try:
        bool(value)
    except Exception as e:
        gr.Warning(f"{name} must be a boolean, got {value}")
        return False
    return True

def sanity_check_choices(value, name, choices):
    try:
        assert value in choices
    except Exception as e:
        gr.Warning(f"{name} must be a string and one of the following: {choices} but got {value}")
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
    except Exception as e:
        gr.Warning(f"{name} must be an integer, got {value}")
        return False
    return True

def sanity_check_number_targets(num_targets, name):
    try:
        assert 1 <= int(num_targets) <= 10
    except Exception as e:
        gr.Warning(f"Number of targetsmust be an integer between 1 and 10 but got {num_targets}")
        return False
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
    num_elements: int = 4
    reset_type_choices = ("zero", "epsilon_uniform", "range_uniform", "None")

    @staticmethod
    def get_parameters():
        with gr.Row():
            ctrl_dt = gr.Number(
                label="Control Timestep (s)",   
                value=0.05,
                minimum=0.01,
                maximum=0.5,
                step=0.01,
                interactive=True
            )
            reset_type = gr.Dropdown(
                label="Define how to reset the body pose", 
                choices=BMParameters.reset_type_choices,
                interactive=True,
                value="range_uniform"
            )
            sigdepnoise_enabled = gr.Checkbox(
                label="Signal-dependent motor noise",   
                value=False,
                interactive=True
            )
            constantnoise_enabled = gr.Checkbox(
                label="Constant motor noise",   
                value=False,
                interactive=True
            )
        return ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_number = 10*BoxParameters.num_elements
        sphere_number = 10*SphereParameters.num_elements
        bm_start = rl_number + num_targets + radio_number + box_number + sphere_number
        bm_end = rl_number + num_targets + radio_number + box_number + sphere_number + BMParameters.num_elements
        return all_args[bm_start:bm_end]

    @staticmethod
    def sanity_check_args(ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled):
        sanity_check = sanity_check_float(ctrl_dt, "Ctrl Timestep (s)")
        sanity_check &= sanity_check_choices(reset_type, "Reset type", BMParameters.reset_type_choices)
        sanity_check &= sanity_check_bool(sigdepnoise_enabled, "Signal-dependent motor noise")
        sanity_check &= sanity_check_bool(constantnoise_enabled, "Constant motor noise")
        return sanity_check
        
    @classmethod
    def parse_values(cls, all_args):
        ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled = cls.get_my_args(all_args)
        sanity_check = cls.sanity_check_args(ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled)
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(f"env.ctrl_dt={float(ctrl_dt)}")
        overrides.append(f"env.task_config.reset_type={reset_type}")
        sigdepnoise_type = "white" if sigdepnoise_enabled else "None"
        constantnoise_type = "white" if constantnoise_enabled else "None"
        overrides.append(f"env.muscle_config.noise_params.sigdepnoise_type={sigdepnoise_type}")
        overrides.append(f"env.muscle_config.noise_params.constantnoise_type={constantnoise_type}")
        return True, overrides

class BoxParameters:
    num_elements: int = 8
    
    @staticmethod
    def init_button_positions(i):
        button_positions = [[0.42, 0.02, 0.01],
                            [0.42, 0.32, 0.01],
                            [0.42, 0.02, -0.29],
                            [0.42, 0.32, -0.29],
                            [0.5, 0.0, -0.05],
                            [0.5, 0.07, -0.05],
                            [0.41, -0.07, 0.05],
                            [0.41, 0.07, 0.05],
                            [0.41, -0.07, -0.25],
                            [0.41, 0.0, -0.25]]
        return button_positions[i]



    @staticmethod
    def fields():
        return ["box_position_x", "box_position_y", "box_position_z", "box_size_x", "box_size_y", "min_touch_force", "orientation_angle", "rgb"]
    
    @staticmethod
    def get_parameters(i):
        # Boxes option row
        button_position = BoxParameters.init_button_positions(i)
        button_rgb = init_target_rgb(i)
        with gr.Row(visible=False) as box_row:
            with gr.Accordion(label=f"Box {i+1} Settings", open=False):
                with gr.Row():
                    box_position_x = gr.Slider(
                        label="Depth Position",
                        minimum=0.2,
                        maximum=0.55,
                        value=button_position[0],
                        step=0.001,
                        interactive=True
                    )
                    box_position_y = gr.Slider(
                        label="Horizontal Position",
                        minimum=-0.25,
                        maximum=0.45,
                        value=button_position[1],
                        step=0.001,
                        interactive=True
                    )
                    box_position_z = gr.Slider(
                        label="Vertical Position",
                        minimum=-0.3,
                        maximum=0.3,
                        value=button_position[2],
                        step=0.001,
                        interactive=True
                    )
                with gr.Row():
                    box_size_x_slider = gr.Slider(
                        label="Width",
                        minimum=0.005,
                        maximum=0.06,
                        value=0.05,
                        step=0.001,
                        interactive=True
                    )
                    box_size_y_slider = gr.Slider(
                        label="Height",
                        minimum=0.005,
                        maximum=0.06,
                        value=0.05,
                        step=0.001,
                        interactive=True
                    )
                with gr.Row():
                    min_touch_force = gr.Slider(
                        label="Minimum Touch Force",
                        info="Minimum force required to register a touch",
                        minimum=0.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.1,
                        interactive=True
                    )
                    orientation_angle = gr.Slider(
                        label="Orientation Angle",
                        minimum=0.0,
                        maximum=180.0,
                        value=45,
                        step=1.0,
                        interactive=True
                    )
                    rgb_btn = gr.ColorPicker(
                        label="RGB",
                        value=button_rgb,
                        interactive=True
                    )
        return box_row, (box_position_x, box_position_y, box_position_z, box_size_x_slider, box_size_y_slider, min_touch_force, orientation_angle, rgb_btn)

    @staticmethod
    def get_my_args(i, all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_start = rl_number + radio_number + num_targets
        box_end = box_start + 10*BoxParameters.num_elements
        box_args = all_args[box_start:box_end]
        start_index = i*BoxParameters.num_elements
        end_index = start_index + BoxParameters.num_elements
        return box_args[start_index:end_index]

    @staticmethod
    def sanity_check_args(i, box_position_x, box_position_y, box_position_z, box_size_x_slider, box_size_y_slider, min_touch_force, orientation_angle, rgb):
        sanity_check = sanity_check_float(box_position_x, f"Box {i+1} Position X")
        sanity_check &= sanity_check_float(box_position_y, f"Box {i+1} Position Y")
        sanity_check &= sanity_check_float(box_position_z, f"Box {i+1} Position Z")
        sanity_check &= sanity_check_float(box_size_x_slider, f"Box {i+1} Size X")
        sanity_check &= sanity_check_float(box_size_y_slider, f"Box {i+1} Size Y")
        sanity_check &= sanity_check_float(min_touch_force, f"Box {i+1} Min Touch Force")
        sanity_check &= sanity_check_float(orientation_angle, f"Box {i+1} Orientation Angle")
        sanity_check &= sanity_check_rgb(rgb, f"Box {i+1} RGB")
        return sanity_check
    
    @classmethod
    def parse_values(cls, i, all_args):
        box_position_x, box_position_y, box_position_z, box_size_x_slider, box_size_y_slider, min_touch_force, orientation_angle, rgb_btn = cls.get_my_args(i, all_args)
        try:
            rgb = hex_to_rgb(rgb_btn)
        except Exception as e:
            gr.Warning(f"Box {i+1} RGB must be a valid hex color code, got {rgb_btn}")
            return False, []
        sanity_check = cls.sanity_check_args(i, box_position_x, box_position_y, box_position_z, box_size_x_slider, box_size_y_slider, min_touch_force, orientation_angle, rgb)
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(f"env.task_config.targets.target_{i}.position=[{box_position_x},{box_position_y},{box_position_z}]")
        overrides.append(f"env.task_config.targets.target_{i}.geom_size=[{0.5*box_size_x_slider},{0.5*box_size_y_slider},0.01]")
        overrides.append(f"env.task_config.targets.target_{i}.site_size=[{0.5*box_size_x_slider},{0.5*box_size_y_slider},0.01]")
        overrides.append(f"env.task_config.targets.target_{i}.min_touch_force={min_touch_force}")
        overrides.append(f"env.task_config.targets.target_{i}.euler=[0,{-orientation_angle*np.pi/180},0]")
        overrides.append(f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]")
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
                    gr.Markdown("#### The coordinates for the sphere targets are randomly sampled from a range, please choose them below")
                with gr.Row():
                    x_slider = RangeSlider(
                        label=f"Depth Range",
                        minimum=sphere_ranges["x"][0],
                        maximum=sphere_ranges["x"][1],
                        value=(sphere_ranges["x"][0], sphere_ranges["x"][1]),
                        step=0.001,
                        interactive=True
                    )
                    y_slider = RangeSlider(
                        label=f"Horizontal Range",
                        minimum=sphere_ranges["y"][0],
                        maximum=sphere_ranges["y"][1],
                        value=(sphere_ranges["y"][0], sphere_ranges["y"][1]),
                        step=0.001,
                        interactive=True
                    )
                    z_slider = RangeSlider(
                        label=f"Vertical Range",
                        minimum=sphere_ranges["z"][0],
                        maximum=sphere_ranges["z"][1],
                        value=(sphere_ranges["z"][0], sphere_ranges["z"][1]),
                        step=0.001,
                        interactive=True
                    )
                with gr.Row():
                    size_slider = RangeSlider(
                        label=f"Size Range (Sphere Radius)",
                        minimum=sphere_ranges["size"][0],
                        maximum=sphere_ranges["size"][1],
                        value=(sphere_ranges["size"][0], sphere_ranges["size"][1]),
                        step=0.001,
                        interactive=True
                    )
                    dwell_duration = gr.Number(
                        label=f"Dwell Duration",
                        value=0.25,
                        minimum=dwell_duration_min,
                        maximum=1.0,
                        step=0.01,
                        interactive=True
                    )
                    color_picker = gr.ColorPicker(
                        label=f"RGB",
                        value=sphere_rgb,
                        interactive=True
                    )
        return sphere_row, (x_slider, y_slider, z_slider, size_slider, dwell_duration, color_picker)

    def get_my_args(i, all_args):
        rl_number = RLParameters.num_elements
        radio_number = 10
        box_number = 10*BoxParameters.num_elements
        num_targets=1
        sphere_start = rl_number + radio_number + box_number + num_targets
        sphere_args = all_args[sphere_start:]
        start_index = i*SphereParameters.num_elements
        end_index = start_index + SphereParameters.num_elements
        return sphere_args[start_index:end_index]
    
    @staticmethod
    def sanity_check_args(i, x_range, y_range, z_range, size_range, dwell_duration, rgb):
        sanity_check = sanity_check_float(x_range[0], f"Sphere {i+1} X Range Start")
        sanity_check &= sanity_check_float(x_range[1], f"Sphere {i+1} X Range End")
        sanity_check &= sanity_check_float(y_range[0], f"Sphere {i+1} Y Range Start")
        sanity_check &= sanity_check_float(y_range[1], f"Sphere {i+1} Y Range End")
        sanity_check &= sanity_check_float(z_range[0], f"Sphere {i+1} Z Range Start")
        sanity_check &= sanity_check_float(z_range[1], f"Sphere {i+1} Z Range End")
        sanity_check &= sanity_check_float(size_range[0], f"Sphere {i+1} Size Range Start")
        sanity_check &= sanity_check_float(size_range[1], f"Sphere {i+1} Size Range End")
        sanity_check &= sanity_check_float(dwell_duration, f"Sphere {i+1} Dwell Duration")
        sanity_check &= sanity_check_rgb(rgb, f"Sphere {i+1} RGB")
        return sanity_check
    
    @classmethod
    def parse_values(cls, i, all_args):
        x_range, y_range, z_range, size_range, dwell_duration, rgb = cls.get_my_args(i, all_args)
        try:
            rgb = hex_to_rgb(rgb)
        except Exception as e:
            gr.Warning(f"Sphere {i+1} RGB must be a valid hex color code, got {rgb}")
            return False, []
        sanity_check = cls.sanity_check_args(i, x_range, y_range, z_range, size_range, dwell_duration, rgb)
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(f"env.task_config.targets.target_{i}.position=[[{x_range[0]},{y_range[0]},{z_range[0]}],[{x_range[1]},{y_range[1]},{z_range[1]}]]")
        overrides.append(f"env.task_config.targets.target_{i}.size=[{size_range[0]},{size_range[1]}]")
        overrides.append(f"env.task_config.targets.target_{i}.dwell_duration={dwell_duration}")
        overrides.append(f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]")
        return True, overrides

class TaskParameters:
    num_elements: int = 1

    @staticmethod
    def get_parameters(ctrl_dt=0.05):
        with gr.Row():
            max_duration = gr.Number(
                label="Maximum Episode Duration (s)",   
                value=4.,
                minimum=0.5,
                maximum=120.,
                step=ctrl_dt,
                interactive=True
            )
        return max_duration, 

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_number = 10*BoxParameters.num_elements
        sphere_number = 10*SphereParameters.num_elements
        bm_number = BMParameters.num_elements
        task_start = rl_number + num_targets + radio_number + box_number + sphere_number + bm_number
        task_end = rl_number + num_targets + radio_number + box_number + sphere_number + bm_number + TaskParameters.num_elements
        return all_args[task_start:task_end]

    @classmethod
    def parse_values(cls, all_args):
        max_duration, = cls.get_my_args(all_args)
        sanity_check = sanity_check_float(max_duration, "Maximum Episode Duration (s)")
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(f"env.task_config.max_duration={float(max_duration)}")
        return True, overrides

class ObservationSpace:
    num_elements: int = 2

    possible_obs_keys = ["qpos", "qvel", "qacc", "ee_pos", "act"]
    possible_omni_keys = ["target_pos", "target_size", "phase", "dwell_fraction"]

    @staticmethod
    def get_parameters():
        obs_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_obs_keys:
                obs_key = gr.Checkbox(
                    label=k,   
                    value=True,
                    interactive=True
                )
                obs_keys.append(obs_key)
        
        omni_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_omni_keys:
                omni_key = gr.Checkbox(
                    label=k,   
                    value=True,
                    interactive=True
                )
                omni_keys.append(omni_key)
        
        return obs_keys, omni_keys

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_number = 10*BoxParameters.num_elements
        sphere_number = 10*SphereParameters.num_elements
        bm_number = BMParameters.num_elements
        task_number = TaskParameters.num_elements
        obs_start = rl_number + num_targets + radio_number + box_number + sphere_number + bm_number + task_number
        obs_end = rl_number + num_targets + radio_number + box_number + sphere_number + bm_number + task_number + len(ObservationSpace.possible_obs_keys)
        omni_start = obs_end
        omni_end = rl_number + num_targets + radio_number + box_number + sphere_number + bm_number + task_number + len(ObservationSpace.possible_obs_keys) + len(ObservationSpace.possible_omni_keys)
        return all_args[obs_start:obs_end], all_args[omni_start:omni_end]

    @classmethod
    def parse_values(cls, all_args):
        obs_keys, omni_keys = cls.get_my_args(all_args)
        obs_keys_selected = [ObservationSpace.possible_obs_keys[id] for id, k in enumerate(obs_keys) if k]  #[k.label for k in obs_keys if k.value]
        omni_keys_selected = [ObservationSpace.possible_omni_keys[id] for id, k in enumerate(omni_keys) if k]  #[k.label for k in omni_keys if k.value]
        overrides = []
        overrides.append(f"env.task_config.obs_keys={obs_keys_selected}")
        overrides.append(f"env.task_config.omni_keys={omni_keys_selected}")
        return True, overrides

class RewardFunction:
    num_elements: int = 1

    weights_default_min_max_step = {"distance": (1, 0, 10, 0.1),
                        "subtask_bonus": (0, 0, 10, 0.5),
                        "completion_bonus": (10, 0, 50, 1),
                        "neural_effort": (0, 0, 1, 0.01),
                    }
    keys=['distance', 'subtask_bonus', 'completion_bonus', 'neural_effort']

    @staticmethod
    def get_parameters():
        reward_weights = []
        with gr.Row():
            for k, (value, minimum, maximum, step) in RewardFunction.weights_default_min_max_step.items():
                reward_weight = gr.Number(
                    label=k,
                    value=value,
                    minimum=minimum,
                    maximum=maximum,
                    step=step,
                    interactive=True
                )
                reward_weights.append(reward_weight)
        
        return reward_weights,

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_number = 10*BoxParameters.num_elements
        sphere_number = 10*SphereParameters.num_elements
        bm_number = BMParameters.num_elements
        task_number = TaskParameters.num_elements
        obs_number = len(ObservationSpace.possible_obs_keys) + len(ObservationSpace.possible_omni_keys)
        reward_start = rl_number + num_targets + radio_number + box_number + sphere_number + bm_number + task_number + obs_number
        reward_end = rl_number + num_targets + radio_number + box_number + sphere_number + bm_number + task_number + obs_number + len(RewardFunction.weights_default_min_max_step)
        return all_args[reward_start:reward_end]

    @staticmethod
    def sanity_check_args(reward_weights):
        sanity_check = True
        for i, k in enumerate(reward_weights):
            key=RewardFunction.keys[i]
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
            key=RewardFunction.keys[i]
            rename_keys = {'distance': 'reach', 'subtask_bonus': 'phase_bonus', 'completion_bonus': 'done'}
            key = rename_keys.get(key, key)
            overrides.append(f"env.task_config.weighted_reward_keys.{key}={k}")
        return True, overrides

class RLParameters:
    num_elements: int = 9
    num_mds: int = 1

    @staticmethod
    def get_parameters():
        with gr.Row():
            num_timesteps = gr.Number(
                label="Number of Training Steps",
                value=15000000,
                minimum=0,
                maximum=50000000,
                step=100_000,
                interactive=True
            )
            num_checkpoints = gr.Number(
                label="Number of Checkpoints During/After Training",
                value=1,
                minimum=1,
                maximum=10,
                interactive=True
            )
            num_evaluations = gr.Number(
                label="Number of Evaluations During/After Training",
                value=1,
                minimum=1,
                maximum=10,
                interactive=True
            )
        md_rl_note = gr.Markdown(
            "<span style='font-size: 1em;'>"
            "<b><span style='color:red'>Note:</span></b> Ensure that (<i>batch_size</i> * <i>num_minibatches</i>) is a multiple of <i>num_envs</i>."
            "</span>",
            elem_id="hint-text"
        )
        with gr.Row():
            batch_size = gr.Number(
                label="Batch Size",
                value=128,
                minimum=0,
                maximum=512,
                interactive=True
            )
            num_envs = gr.Number(
                label="Number of Parallel Environments",
                value=1024,
                minimum=0,
                maximum=4096,
                interactive=True
            )
            num_minibatches = gr.Number(
                label="Number of Minibatches",
                value=8,
                minimum=0,
                maximum=40,
                interactive=True
            )
        with gr.Row():
            choices = get_available_checkpoints()
            select_checkpoint_run = gr.Dropdown(
                label="Select Experiment from which to load checkpoints",
                choices=choices,
                interactive=True,
                value="None"
            )
            choices = get_available_checkpoint_numbers(select_checkpoint_run.value)
            select_checkpoint_number = gr.Dropdown(
                label="Select Checkpoint Number",
                choices=choices,
                interactive=True,
                value="None"
            )
            select_checkpoint_run.change(
                update_checkpoint_numbers,
                inputs=select_checkpoint_run,
                outputs=select_checkpoint_number
            )
        target_init_seed = gr.Number(
            label="Target Initial Seed",
            value = 0,
            minimum=0,
            maximum=1000000,
            interactive=True,
            visible=False
        )
        return num_timesteps, num_checkpoints, num_evaluations, batch_size, num_envs, num_minibatches, select_checkpoint_run, select_checkpoint_number, target_init_seed, md_rl_note

    @staticmethod
    def get_my_args(all_args):
        return all_args[:RLParameters.num_elements]

    @staticmethod
    def sanity_check_args(num_targets, num_timesteps, num_checkpoints, num_evaluations, batch_size, num_envs, num_minibatches, select_checkpoint_run, select_checkpoint_number, target_init_seed):
        sanity_check = sanity_check_number_targets(num_targets, "Number of Targets")
        sanity_check &= sanity_check_int(num_timesteps, "Number of Training Steps")
        sanity_check &= sanity_check_int(num_checkpoints, "Number of Checkpoints During/After Training")
        sanity_check &= sanity_check_int(num_evaluations, "Number of Evaluations During/After Training")
        sanity_check &= sanity_check_int(batch_size, "Batch Size")
        sanity_check &= sanity_check_int(num_envs, "Number of Parallel Environments")
        sanity_check &= sanity_check_int(num_minibatches, "Number of Minibatches")
        sanity_check &= sanity_check_choices(select_checkpoint_run, "Select Experiment from which to load checkpoints", get_available_checkpoints())
        sanity_check &= sanity_check_choices(select_checkpoint_number, "Select Checkpoint Number", get_available_checkpoint_numbers(select_checkpoint_run))
        sanity_check &= sanity_check_int(target_init_seed, "Target Initial Seed")
        return sanity_check

    @classmethod
    def parse_values(cls, all_args):
        num_timesteps, num_checkpoints, num_evaluations, batch_size, num_envs, num_minibatches, select_checkpoint_run, select_checkpoint_number, target_init_seed = cls.get_my_args(all_args)
        overrides = []
        num_targets = cls.get_number_targets(all_args)
        sanity_check = cls.sanity_check_args(num_targets, num_timesteps, num_checkpoints, num_evaluations, batch_size, num_envs, num_minibatches, select_checkpoint_run, select_checkpoint_number, target_init_seed)
        if not sanity_check:
            return False, []
        to_text = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        overrides.append(f"env/task_config/targets={to_text[int(num_targets)]}")
        overrides.append(f"rl.num_timesteps={int(num_timesteps)}")
        overrides.append(f"rl.num_checkpoints={int(num_checkpoints)}")
        overrides.append(f"rl.num_evals={int(num_evaluations)}")
        overrides.append(f"rl.batch_size={int(batch_size)}")
        overrides.append(f"rl.num_envs={int(num_envs)}")
        overrides.append(f"rl.num_minibatches={int(num_minibatches)}")
        exact_checkpoint_path = checkpoint_path_from_run_number(select_checkpoint_run, select_checkpoint_number)
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

class ConfigSaver:    
    def check_folder(self):
        if not os.path.exists("gradio_configs"):
            os.makedirs("gradio_configs")
        files = os.listdir("gradio_configs")
        json_files = [file for file in files if file.endswith(".json")]
        file_names = [file.split(".")[0] for file in json_files]
        return file_names
    
    def __init__(self):
        self.my_configs = self.check_folder()

    def add_config(self, config_name, data):
        with open(f"gradio_configs/{config_name}.json", "w") as f:
            json.dump(data, f)
        
    def config_save_clicked(self, config_name_input, *args):
        if config_name_input in self.my_configs:
            gr.Warning("There is already a configuration with this name. Please choose a different name.")
            return gr.skip()
        if config_name_input == "":
            gr.Warning("Please enter a name for the configuration.")
        else:
            self.add_config(config_name_input, args)
            self.my_configs.append(config_name_input)
        return gr.update(choices=self.my_configs, value=config_name_input)

    def available_configs(self):
        return self.my_configs

    def load_config(self, config_name):
        with open(f"gradio_configs/{config_name}.json", "r") as f:
            data = json.load(f)
        return tuple([gr.update(value=k) for k in data])

def get_ui(project_name, run_state = RunState()):
    # Fix box handlers properly
    with gr.Blocks() as demo:
        pageview = gr.Radio(
            choices=["Simple", "Advanced"],
            label=f"Show Advanced Options",
            value="Simple",
            interactive=True
        )

        gr.Markdown("**Weights & Biases URL:**")
        url_display = gr.Textbox(
            value=f"https://wandb.ai/{wandb.api.default_entity}/{project_name}",
            label="Results Dashboard",
            interactive=False,
            show_copy_button=True,
            info="Once your run starts, you can view the training progress for your projects here. Click to copy the URL to go to the wandb project and monitor training progress"
                    )
        
        gr.Markdown("### Pre-saved configurations")
        gr.Markdown('Select a pre-saved configuration or create your own.')
        config_saver = ConfigSaver()
        with gr.Row():
            pre_saved_configs = gr.Dropdown(
                label="Pre-saved configurations",
                choices=config_saver.available_configs(),
                value="",
                interactive=True,
                visible=True
            )

        md1 = gr.Markdown("### Biomechanical Model Parameters")
        bm_params = BMParameters.get_parameters()
        ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled = bm_params
        
        md2 = gr.Markdown("### Task Parameters")
        md21 = gr.Markdown("#### Target Setup")
        num_elements = gr.Number(
            label="Number of Targets",
            value=INIT_ELEMENTS,
            minimum=0,
            maximum=10,
            precision=0,
            interactive=True
        )
        
        # Create 10 sets of elements (maximum possible), first INIT_ELEMENTS visible by default
        dynamic_rows = []
        radios = []
        dwell_durations = []
        box_rows = []
        sphere_rows = []
        
        # Store all components for easy access
        all_components = {
            'boxes': [],
            'spheres': []
        }

        # Create variable that sums up all selected dwell times
        total_dwell_duration = gr.Number(
                    label=f"total Dwell Time",
                    value=0,
                    minimum=0,
                    step=0.01,
                    interactive=False,
                    visible=False
                )
        
        for i in range(10):
            # Main row with radio selection
            with gr.Row(visible=(i < INIT_ELEMENTS)) as main_row:
                with gr.Column():
                    radio = gr.Radio(
                        choices=["Box", "Sphere"],
                        label=f"Target {i+1} Type",
                        value="Sphere",
                        interactive=True
                    )
                    box_row, box_params = BoxParameters.get_parameters(i)
                    box_position_x, box_position_y, box_position_z, box_size_x_slider, box_size_y_slider, min_touch_force, orientation_angle, rgb_btn = box_params
                    # Store box components
                    all_components['boxes'].append({
                        key: value for key, value in zip(BoxParameters.fields(), box_params)
                    })

                    sphere_row, sphere_params = SphereParameters.get_parameters(i, dwell_duration_min=ctrl_dt.value)
                    x_slider, y_slider, z_slider, size_slider, dwell_duration, color_picker = sphere_params
                    ctrl_dt.change(fn=update_dwell_duration, inputs=(dwell_duration, ctrl_dt), outputs=dwell_duration, preprocess=False)
                    # Store sphere components
                    all_components['spheres'].append({
                        key: value for key, value in zip(SphereParameters.fields(), sphere_params)
                    })                    
            
            # Store references
            dynamic_rows.append(main_row)
            radios.append(radio)
            dwell_durations.append(dwell_duration)
            box_rows.append(box_row)
            sphere_rows.append(sphere_row)

        md22 = gr.Markdown("#### Other Task Parameters")
        task_params = TaskParameters.get_parameters(ctrl_dt=ctrl_dt.value)
        max_duration, = task_params

        md23 = gr.Markdown("#### Observation Space")
        obs_keys, omni_keys = ObservationSpace.get_parameters()

        md24 = gr.Markdown("#### Reward Weights")
        
        def get_max_dist(num_elements, *radios_and_box_and_sphere_positions):
            ee_pos0 = [0., -0.27, 0.37]  #TODO: infer from model!

            _num_targets_max = int(len(radios_and_box_and_sphere_positions) // (1 + 2*3))
            radios = radios_and_box_and_sphere_positions[:num_elements]
            box_positions_x = radios_and_box_and_sphere_positions[_num_targets_max:2*_num_targets_max]
            box_positions_y = radios_and_box_and_sphere_positions[2*_num_targets_max:3*_num_targets_max]
            box_positions_z = radios_and_box_and_sphere_positions[3*_num_targets_max:4*_num_targets_max]
            sphere_positions_x = radios_and_box_and_sphere_positions[4*_num_targets_max:5*_num_targets_max]
            sphere_positions_y = radios_and_box_and_sphere_positions[5*_num_targets_max:6*_num_targets_max]
            sphere_positions_z = radios_and_box_and_sphere_positions[6*_num_targets_max:7*_num_targets_max]
            target_positions = [ee_pos0]
            for target_id in range(num_elements):
                if radios[target_id] == "Box":
                    box_pos_array = [box_positions_x[target_id], box_positions_y[target_id], box_positions_z[target_id]]
                    sanity_check = sanity_check_float_array(box_pos_array, f"Box {target_id+1} Position")
                    if not sanity_check:
                        return False, 0
                    target_positions.append(box_pos_array)
                elif radios[target_id] == "Sphere":
                    sphere_x = sphere_positions_x[target_id]
                    sphere_y = sphere_positions_y[target_id]
                    sphere_z = sphere_positions_z[target_id]
                    sanity_check = sanity_check_float_array(sphere_x, f"Sphere {target_id+1} Range Position (Depth)")
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(sphere_y, f"Sphere {target_id+1} Range Position (Horizontal)")
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(sphere_z, f"Sphere {target_id+1} Range Position (Vertical)")
                    if not sanity_check:
                        return False, 0
                    target_positions.append([np.mean(sphere_x), np.mean(sphere_y), np.mean(sphere_z)])
                else:
                    raise NotImplementedError()
            target_positions = np.array(target_positions)
            max_dist = np.sum(np.linalg.norm(np.diff(target_positions, axis=0), axis=1), axis=0).item()
            return True, max_dist
        
        def update_max_dist(num_elements, *radios_and_box_and_sphere_positions):
            sanity_check, max_dist = get_max_dist(num_elements, *radios_and_box_and_sphere_positions)
            if not sanity_check:
                return gr.skip()
            return gr.update(value=max_dist)
        
        radios_and_box_and_sphere_positions = radios + [b["box_position_x"] for b in all_components['boxes']] + \
                                                [b["box_position_y"] for b in all_components['boxes']] + \
                                                [b["box_position_z"] for b in all_components['boxes']] + \
                                                [s["x_range"] for s in all_components['spheres']] + \
                                                [s["y_range"] for s in all_components['spheres']] + \
                                                [s["z_range"] for s in all_components['spheres']]
        max_dist = gr.Number(
                label="Maximum Path Length",   
                value=get_max_dist(num_elements.value, *[r.value for r in radios_and_box_and_sphere_positions])[1],
                interactive=False,
                visible=False,
            )
        for k in radios_and_box_and_sphere_positions + [num_elements]:
            k.change(fn=update_max_dist, 
                    inputs=(num_elements, *radios_and_box_and_sphere_positions),
                    outputs=max_dist,
                    preprocess=False
                    )

        num_ctrls = 26  #TODO: infer from chosen MuJoCo model
        def reward_fct_view(num_elements, reach, phase_bonus, done, neural_effort, max_dist):
            return  f"""<span style='font-size: 1em;'>The following **Reward** will be provided at each time step *n*, depending on the current target *i*:
            <div align="center">
            $r_n =$ <span title="min: {-1*reach*max_dist:.4g};
            max: {0}">${-1*reach} \cdot (\\text{{distance to current target }} i + \sum_{{j=i+1}}^{{{num_elements}}}\\text{{dist(target }}j\\text{{, target }} j-1\\text{{)}})$</span>

            <span title="min: {0};
            max: {1*phase_bonus}">$+ {1*phase_bonus} \cdot (\\text{{current target }} i \\text{{ successfully hit for the first time}})$</span>
            
            <span title="min: {0};
            max: {1*done}">$+ {1*done} \cdot (\\text{{task successfully completed}})$</span>
            
            <span title="min: {-1*neural_effort*num_ctrls};
            max: {0}">$- {1*neural_effort} \cdot (\\text{{squared control effort costs}})$</span>
            </div></span>"""
        _reach_d, _phase_bonus_d, _done_d, _neural_effort_d = RewardFunction.weights_default_min_max_step['distance'][0], RewardFunction.weights_default_min_max_step['subtask_bonus'][0], RewardFunction.weights_default_min_max_step['completion_bonus'][0], RewardFunction.weights_default_min_max_step['neural_effort'][0],
        reward_function_text = gr.Markdown(
            reward_fct_view(num_elements=num_elements.value, reach=_reach_d, phase_bonus=_phase_bonus_d, done=_done_d, neural_effort=_neural_effort_d, max_dist=max_dist.value),
            elem_id="reward-function",
            line_breaks=True,
            latex_delimiters=[{"left": "$", "right": "$", "display": False}],
        )
        def update_reward_fct_view(num_elements, reach, phase_bonus, done, neural_effort, max_dist):
            return gr.update(value=reward_fct_view(num_elements=num_elements, reach=reach, phase_bonus=phase_bonus, done=done, neural_effort=neural_effort, max_dist=max_dist))
        reward_weights, = RewardFunction.get_parameters()
        weighted_reward_keys_gr = {k.label: k for k in reward_weights}
        for k in reward_weights + [num_elements, max_dist]:
            k.change(update_reward_fct_view, [num_elements, weighted_reward_keys_gr['distance'], weighted_reward_keys_gr['subtask_bonus'], weighted_reward_keys_gr['completion_bonus'], weighted_reward_keys_gr['neural_effort'], max_dist], reward_function_text)

        md3 = gr.Markdown("### RL Parameters")
        rl_params_and_mds = RLParameters.get_parameters()
        rl_params = rl_params_and_mds[:RLParameters.num_elements]
        md_rl_notes = rl_params_and_mds[-RLParameters.num_mds:]
        num_timesteps = rl_params[0]
        target_init_seed = rl_params[-1]


        gr.Markdown("### View of the environment")
        render_button = gr.Button("Render Environment", variant="primary", size="lg")
        with gr.Row(visible=False) as env_view_row:
            env_view_1 = gr.Image(label="Environment View", interactive=False)
            env_view_2 = gr.Image(label="Environment View", interactive=False)
        
                
        gr.Markdown("### Save current configuration")
        with gr.Row():
            config_name_input = gr.Textbox(label="Configuration Name", value="", interactive=True)
            save_config_button = gr.Button("Save Configuration", variant="primary", size="lg")
        # Add Run button and output
        gr.Markdown("### Run Configuration")
        gr.Markdown("Click the save button above to save the configuration first!")
        with gr.Row():
            run_button = gr.Button("Run", variant="primary", size="lg")
        
        output_text = gr.Textbox(
            label="Configuration Output",
            lines=20,
            max_lines=30,
            interactive=False,
            show_copy_button=True
        )

        def args_to_cfg_overrides(run_name, *args):
            """Print all configuration details"""
            # Extract values from args
            num_targets = args[RLParameters.num_elements]
            radio_start = RLParameters.num_elements + 1
            radio_end = radio_start + 10
            radio_values = args[radio_start:radio_end]
            
            cfg_overrides = ["env=universal", "run.using_gradio=True", f"wandb.project={project_name}", f"wandb.entity={wandb.api.default_entity}", f"wandb.name={run_name}"]
            all_sanity_checks = True
            sanity_check, bm_overrides = BMParameters.parse_values(args)
            cfg_overrides.extend(bm_overrides)

            for i in range(int(num_targets)):
                target_type = radio_values[i]
                cfg_overrides.append(f"+env/task_config/targets/target_{i}={target_type.lower()}")
                
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
            
            sanity_check, task_overrides = TaskParameters.parse_values(args)
            all_sanity_checks &= sanity_check
            cfg_overrides.extend(task_overrides)
            sanity_check, obs_overrides = ObservationSpace.parse_values(args)
            all_sanity_checks &= sanity_check
            cfg_overrides.extend(obs_overrides)
            sanity_check, rl_overrides = RLParameters.parse_values(args)
            all_sanity_checks &= sanity_check
            sanity_check, reward_overrides = RewardFunction.parse_values(args)
            all_sanity_checks &= sanity_check
            cfg_overrides.extend(reward_overrides)
            cfg_overrides.extend(rl_overrides)

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
            gr.Info("Go back to the notebook and run the next cells to start training!")
            run_state.cfg_overrides.clear()
            run_state.cfg_overrides.extend(cfg_overrides)
            run_state.wandb_run_name = run_name
            # train(cfg)
            text = "Go to the next notebook in the cell and run the training!\n"
            text += "\n".join(run_state.cfg_overrides)
            return text

        def render_environment(run_name, *args):
            target_init_seed = RLParameters.get_target_init_seed(args)
            next_seed = target_init_seed + 1
            sanity_check, cfg_overrides = args_to_cfg_overrides(run_name, *args)
            if not sanity_check:
                gr.Warning("Invalid configuration.")
                return gr.skip()
            config = load_config_interactive(cfg_overrides)
            env = MyoUserUniversal(config.env)
            imgs = env.get_renderings()
            img1 = imgs[0][1]
            img2 = imgs[1][1]
            return gr.update(value=next_seed), gr.update(visible=True), gr.update(value=img1), gr.update(value=img2)
            

        def update_dynamic_elements(num):
            """Show/hide dynamic rows based on the number input"""
            num = max(0, min(int(num) if num is not None else 0, 10))
            updates = []
            for i in range(10):
                if i < num:
                    updates.append(gr.update(visible=True))
                else:
                    updates.append(gr.update(visible=False))
            return updates

        def update_num_timesteps(num_elements, ctrl_dt, *dwell_durations_and_radios):
            _num_targets_max = int(len(dwell_durations_and_radios) // 2)
            dwell_durations = dwell_durations_and_radios[:num_elements]
            radios = dwell_durations_and_radios[_num_targets_max:_num_targets_max+num_elements]
            total_dwell_duration = sum(map(lambda x, y: max(0, x - ctrl_dt) * (y == "Sphere"), dwell_durations, radios))

            target_value = 5000000 + max((num_elements - 3), 0) * 1000000 + int(total_dwell_duration // 0.3) * 1000000
            return gr.update(value=target_value)

        def toggle_interface_type(radio_value):
            """Show appropriate interface based on radio selection"""
            if radio_value == "Box":
                return gr.update(visible=True), gr.update(visible=False)
            else:  # Sphere
                return gr.update(visible=False), gr.update(visible=True)
        
        def update_pageview(pageview, *advanced_options):
            visible = (pageview == "Advanced")
            return [gr.update(visible=visible) for opt in advanced_options]

        advanced_options_and_markdowns = [*bm_params, *task_params, *obs_keys, *omni_keys, *rl_params, *md_rl_notes] + [md1, md22, md23, md3]
        pageview.change(fn=update_pageview, inputs=(pageview, *advanced_options_and_markdowns), outputs=advanced_options_and_markdowns)
        # also call this method once at the very beginning
        demo.load(fn=update_pageview, inputs=(pageview, *advanced_options_and_markdowns), outputs=advanced_options_and_markdowns)

        # Event handler for dynamic elements
        num_elements.change(
            update_dynamic_elements,
            inputs=num_elements,
            outputs=dynamic_rows
        )

        # Event handler for number of targets to suggested number of training steps
        num_elements.change(
            update_num_timesteps,
            inputs=(num_elements, ctrl_dt, *dwell_durations, *radios),
            outputs=num_timesteps,
            preprocess=False
        )

        # Event handlers for each radio button to control interface type and num of training steps
        for i in range(10):
            radios[i].change(
                toggle_interface_type,
                inputs=radios[i],
                outputs=[box_rows[i], sphere_rows[i]]
            )
            radios[i].change(
                update_num_timesteps,
                inputs=(num_elements, ctrl_dt, *dwell_durations, *radios),
                outputs=num_timesteps,
                preprocess=False
            )

        # Event handler for each dwell duration to num of training steps
        for i in range(10):
            dwell_durations[i].change(
                update_num_timesteps,
                inputs=(num_elements, ctrl_dt, *dwell_durations, *radios),
                outputs=num_timesteps,
                preprocess=False
            )

        # Prepare inputs for run button
        run_inputs = [*rl_params, num_elements]
        run_inputs.extend(radios)
        
        # Add all box components
        for i in range(10):
            for key in BoxParameters.fields():
                run_inputs.append(all_components['boxes'][i][key])
        
        # Add all sphere components
        for i in range(10):
            for key in SphereParameters.fields():
                run_inputs.append(all_components['spheres'][i][key])

        run_inputs.extend(bm_params)
        run_inputs.extend(task_params)
        run_inputs.extend(obs_keys)
        run_inputs.extend(omni_keys)
        run_inputs.extend(reward_weights)

        # Run button event
        run_or_save_inputs = [config_name_input] + run_inputs
        run_button.click(
            run_training,
            inputs=run_or_save_inputs,
            outputs=output_text
        )

        pre_saved_configs.change(
            config_saver.load_config,
            inputs=pre_saved_configs,
            outputs=run_inputs
        )

        save_config_button.click(
            config_saver.config_save_clicked,
            inputs=run_or_save_inputs,
            outputs=pre_saved_configs
        )

        render_button.click(
            render_environment,
            inputs=run_or_save_inputs,
            outputs=[target_init_seed, env_view_row, env_view_1, env_view_2]
        )

    return demo

if __name__ == "__main__":
    wandb_url = None
    demo = get_ui(wandb_url)
    demo.launch(share=True)