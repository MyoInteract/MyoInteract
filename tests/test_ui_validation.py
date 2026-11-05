from copy import deepcopy

import pytest

from myosuite.envs.myo.myouser.validation import (
    boolean, finite_number, integer, parse_color, validate_training_config,
)


def valid_config():
    return {"env": {"ctrl_dt": .05, "sim_dt": .002, "task_config": {"max_duration": 4., "obs_keys": ["qpos"], "omni_keys": [], "target_init_seed": 0}}, "rl": {"num_timesteps": 1000, "num_envs": 1024, "batch_size": 128, "num_minibatches": 8, "num_evals": 1, "num_checkpoints": 1}}


@pytest.mark.parametrize("value", [1.9, "2", True, None, float("nan"), float("inf")])
def test_integer_rejects_truncation_coercion_and_nonfinite(value):
    with pytest.raises(ValueError):
        integer(value, "count")


def test_bool_and_finite_number_are_strict():
    with pytest.raises(ValueError):
        boolean("false", "noise")
    with pytest.raises(ValueError):
        finite_number(True, "timestep")


@pytest.mark.parametrize("updates", [{"num_envs": 0}, {"num_envs": 1000}, {"batch_size": 0}, {"num_minibatches": 0}, {"num_timesteps": -1}, {"batch_size": 128.5}])
def test_training_dimensions_fail_before_the_trainer(updates):
    cfg = valid_config()
    cfg["rl"].update(updates)
    with pytest.raises(ValueError):
        validate_training_config(cfg)


def test_zero_steps_remains_valid_for_checkpoint_evaluation():
    cfg = valid_config()
    cfg["rl"]["num_timesteps"] = 0
    assert validate_training_config(cfg) is cfg


def test_zero_resets_per_eval_remains_supported():
    cfg = valid_config()
    cfg["rl"]["num_resets_per_eval"] = 0
    assert validate_training_config(cfg) is cfg
    cfg["rl"]["num_resets_per_eval"] = -1
    with pytest.raises(ValueError, match="num_resets_per_eval"):
        validate_training_config(cfg)


@pytest.mark.parametrize("value", ["rgba", "rgba(1,2,3)", "rgba(1,2,3,2)", "rgb(256,0,0)", "rgb(nan,0,0)", "#zz0000", "", None])
def test_color_parser_rejects_malformed_and_out_of_range_colors(value):
    with pytest.raises(ValueError, match="Color must"):
        parse_color(value)


@pytest.mark.parametrize("value", ["#ff0000", "#f00", "#ff000080", "rgb(255, 0, 0)", "rgba(255, 0, 0, .5)"])
def test_supported_color_formats_have_identical_rgb(value):
    assert parse_color(value) == (1., 0., 0.)


@pytest.mark.parametrize("updates", [{"ctrl_dt": None}, {"ctrl_dt": 0}, {"ctrl_dt": .051}])
def test_biomechanical_timestep_is_validated(updates):
    cfg = valid_config()
    cfg["env"].update(updates)
    with pytest.raises(ValueError):
        validate_training_config(cfg)


def test_validation_is_read_only_and_checks_observation_and_episode_shape():
    cfg = valid_config()
    before = deepcopy(cfg)
    validate_training_config(cfg)
    assert cfg == before
    cfg["env"]["task_config"]["max_duration"] = .01
    with pytest.raises(ValueError, match="control timestep"):
        validate_training_config(cfg)
    cfg = valid_config()
    cfg["env"]["task_config"]["obs_keys"] = []
    with pytest.raises(ValueError, match="observation"):
        validate_training_config(cfg)


@pytest.mark.parametrize("updates", [{"eval_episodes": 0}, {"eval_episodes": 1.5}, {"seed": -1}, {"seed": 2**32}, {"eval_seed": 2**32}, {"play_only": "false"}, {"using_gradio": 1}, {"rscope_envs": -1}])
def test_run_options_are_validated_before_training(updates):
    cfg = valid_config()
    cfg["run"] = updates
    with pytest.raises(ValueError):
        validate_training_config(cfg)


def test_uint32_seed_boundaries_and_disabled_optional_monitor_are_valid():
    cfg = valid_config()
    cfg["run"] = {"seed": 0, "eval_seed": 2**32 - 1, "eval_episodes": 1, "rscope_envs": 0, "play_only": False}
    assert validate_training_config(cfg) is cfg


@pytest.mark.parametrize("vision", [{"enabled": "false"}, {"enabled": True, "num_worlds": 0}, {"enabled": True, "render_batch_size": -1}, {"render_width": 0}, {"gpu_id": -1}, {"use_rasterizer": 1}])
def test_vision_dimensions_and_flags_are_preflight_checked(vision):
    cfg = valid_config()
    cfg["vision"] = vision
    with pytest.raises(ValueError):
        validate_training_config(cfg)


@pytest.mark.parametrize("section,key,value", [("env", "eval_mode", "false"), ("rl", "episode_length", 0), ("rl", "normalize_observations", "true"), ("wandb", "enabled", 1)])
def test_runtime_control_types_are_strict(section, key, value):
    cfg = valid_config()
    cfg.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        validate_training_config(cfg)


def test_cli_rejects_invalid_evaluation_settings_before_allocating_environment():
    from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
    with pytest.raises(ValueError, match="run.eval_episodes"):
        load_config_interactive(["run.eval_episodes=0"], cfg_only=True)
    with pytest.raises(ValueError, match="run.eval_seed"):
        load_config_interactive(["run.eval_seed=4294967296"], cfg_only=True)
