"""Vision recipes and reset semantics through the native configuration boundary."""

from __future__ import annotations

import json
import ast
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

import mujoco  # noqa: E402
from ml_collections import ConfigDict  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.scene import SceneCfg  # noqa: E402
from mjlab.sim import MujocoCfg, SimulationCfg  # noqa: E402

from myoInteract import register_mjlab_myointeract_tasks as task  # noqa: E402
from myoInteract.myointeract_hydra_cli import (  # noqa: E402
    checkpoint_task_overrides,
    extra_hydra_overrides,
    gradio_json_to_overrides,
    load_config_interactive,
    load_config_from_checkpoint_dir,
    training_config_to_labelled_dict,
)


@pytest.mark.parametrize(
    "mode,channels",
    [("rgb", ("rgb",)), ("depth", ("depth",)), ("rgbd", ("rgb", "depth"))],
)
def test_native_vision_config_honors_selected_inputs_and_depth_bounds(
    monkeypatch, mode, channels
):
    config = load_config_interactive(
        [
            "env=universal",
            "vision=enabled",
            f"vision.vision_mode={mode}",
            "vision.render_width=64",
            "vision.render_height=80",
            "vision.depth_min=0.05",
            "vision.depth_cutoff=2.0",
        ]
    )
    monkeypatch.setattr(task, "_get_muscle_names", lambda _cfg: ("muscle",))
    monkeypatch.setattr(task, "_compute_target_origin", lambda _cfg: np.zeros(3))
    cfg = task._make_universal_env_cfg(config)
    sensor = cfg.scene.sensors[0]
    assert tuple(sensor.data_types) == channels
    assert (sensor.width, sensor.height) == (64, 80)
    terms = cfg.observations["vision_mono"].terms
    assert list(terms) == [f"fixed-eye_{channel}" for channel in channels]
    if "depth" in channels:
        assert terms["fixed-eye_depth"].params == {
            "sensor_name": "fixed-eye",
            "cutoff_distance": 2.0,
            "min_depth": 0.05,
        }


@pytest.mark.parametrize(
    "key,value",
    [
        ("render_width", 0),
        ("render_height", -1),
        ("depth_min", -0.1),
        ("depth_cutoff", 0.01),
    ],
)
def test_invalid_new_vision_recipe_is_rejected(key, value):
    config = load_config_interactive(
        ["env=universal", "vision=enabled", f"vision.{key}={value}"]
    )
    with pytest.raises(ValueError, match="Vision"):
        task._vision_camera_recipe(config.env.vision)


def test_auxiliary_vision_mode_is_preserved_but_not_silently_emulated():
    config = load_config_interactive(
        [
            "env=universal",
            "vision=enabled",
            "vision.vision_mode=depth_w_aux_task",
        ]
    )
    original = config.to_dict()
    overrides = gradio_json_to_overrides(
        training_config_to_labelled_dict(original)
    ) + extra_hydra_overrides(original)
    restored = load_config_interactive(overrides)
    assert restored.env.vision.enabled
    assert restored.env.vision.vision_mode == "depth_w_aux_task"
    with pytest.raises(ValueError, match="Unsupported vision mode"):
        task._vision_camera_recipe(restored.env.vision)


def test_legacy_checkpoint_recipe_survives_override_and_gradio_reconstruction(
    tmp_path: Path,
):
    requested = [
        "env=universal",
        "vision=enabled",
        "vision.vision_mode=rgb",
        "vision.render_width=64",
    ]
    restored = load_config_interactive(checkpoint_task_overrides(requested))
    expected = task._vision_camera_recipe(restored.env.vision)
    assert expected["data_types"] == ("rgb", "depth")
    assert (expected["width"], expected["height"]) == (128, 128)
    assert expected["depth_cutoff"] == 0.5

    original = load_config_interactive(requested).to_dict()
    for vision in (original["vision"], original["env"]["vision"]):
        vision.pop("recipe_version", None)
        vision.pop("depth_min", None)
        vision.pop("depth_cutoff", None)
    labelled = training_config_to_labelled_dict(original)
    overrides = gradio_json_to_overrides(labelled) + extra_hydra_overrides(original)
    rebuilt = load_config_interactive(overrides)
    assert task._vision_camera_recipe(rebuilt.env.vision) == expected
    assert task._vision_camera_recipe(ConfigDict(original["vision"])) == expected
    (tmp_path / "myointeract_config.json").write_text(json.dumps(original))
    saved = load_config_from_checkpoint_dir(str(tmp_path))
    assert task._vision_camera_recipe(saved.env.vision) == expected

    from myoInteract.studio.worker import run_overrides

    (tmp_path / "original_hydra_config.json").write_text(json.dumps(requested))
    assert (
        task._vision_camera_recipe(
            load_config_interactive(run_overrides(tmp_path)).env.vision
        )
        == expected
    )


def test_new_vision_recipe_is_versioned_and_preserved_on_reload():
    original = load_config_interactive(
        [
            "env=universal",
            "vision=enabled",
            "vision.vision_mode=depth",
            "vision.render_width=64",
            "vision.render_height=80",
            "vision.depth_cutoff=2.0",
        ]
    ).to_dict()
    labelled = training_config_to_labelled_dict(original)
    overrides = gradio_json_to_overrides(labelled) + extra_hydra_overrides(original)
    assert "vision.recipe_version=2" in overrides
    assert checkpoint_task_overrides(overrides) == overrides
    restored = load_config_interactive(overrides)
    assert task._vision_camera_recipe(
        restored.env.vision
    ) == task._vision_camera_recipe(ConfigDict(original["vision"]))
    assert checkpoint_task_overrides(["env=universal", "vision=disabled"]) == [
        "env=universal",
        "vision=disabled",
    ]


def test_gradio_environment_cache_uses_effective_vision_recipe():
    source = Path(__file__).resolve().parents[1] / "myoInteract/scripts/ui.py"
    function = next(
        node
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "_render_struct_key"
    )
    namespace = {"_vision_camera_recipe": task._vision_camera_recipe}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    key = namespace["_render_struct_key"]
    base = [
        "env=universal",
        "vision=enabled",
        "vision.vision_mode=rgb",
        "vision.render_width=64",
    ]
    current = load_config_interactive(base)
    legacy = load_config_interactive(base + ["vision.recipe_version=1"])
    assert key(current) != key(legacy)
    assert key(legacy) == key(
        load_config_interactive(
            base + ["vision.recipe_version=1", "vision.render_width=80"]
        )
    )
    for override in (
        "vision.render_width=80",
        "vision.render_height=80",
        "vision.depth_min=0.05",
        "vision.depth_cutoff=2.0",
        "vision.enabled_geom_groups=[0,3]",
        "vision.vision_mode=rgbd",
    ):
        assert key(current) != key(load_config_interactive(base + [override]))


def _minimal_config(kind: str, curriculum: bool = False) -> ConfigDict:
    return ConfigDict(
        {
            "env": {
                "model_path": "README.md",
                "ctrl_dt": 0.01,
                "task_config": {"reset_type": kind, "targets": {"num_targets": 1}},
                "curriculum_config": {"enabled": curriculum},
            }
        }
    )


def _minimal_env(base: ConfigDict, name: str = "universal_robot") -> ManagerBasedRlEnv:
    xml = """<mujoco><worldbody>
      <body name="upper"><joint name="x" type="slide" axis="1 0 0"/><geom name="upper_geom" type="sphere" size=".01"/>
        <body name="lower"><joint name="y" type="slide" axis="0 1 0"/><geom name="lower_geom" type="sphere" size=".01"/></body>
      </body>
      <body name="body_target_0"><geom name="geom_target_0" type="sphere" size=".1"/></body>
    </worldbody></mujoco>"""
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=2,
            entities={
                name: task.TaskEntityCfg(
                    spec_fn=lambda: mujoco.MjSpec.from_string(xml), base_config=base
                )
            },
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.01, gravity=(0, 0, 0)), njmax=16, nconmax=8
        ),
        decimation=1,
        episode_length_s=1.0,
    )
    return ManagerBasedRlEnv(
        cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )


@pytest.mark.parametrize("order", [("None", "zero"), ("zero", "None")])
def test_reset_preservation_uses_each_environment_recipe(monkeypatch, order):
    native_init = ManagerBasedRlEnv.__init__
    while getattr(native_init, "_preserve_qpos_patched", False):
        native_init = native_init.__kwdefaults__["_orig"]
    monkeypatch.setattr(ManagerBasedRlEnv, "__init__", native_init)
    monkeypatch.setattr(task, "_MJLAB_REGISTRY", {})
    monkeypatch.setattr(task, "register_mjlab_task", lambda **_kwargs: None)
    monkeypatch.setattr(task, "_make_universal_env_cfg", lambda cfg, play: cfg)
    monkeypatch.setattr(task, "_universal_ppo_runner_cfg", lambda cfg: None)
    environments = []
    try:
        for kind in order:
            task.register_mjlab_myointeract_task(_minimal_config(kind))
        for kind, curriculum, entity_name in [
            ("None", False, "universal_robot"),
            ("zero", False, "universal_robot"),
            ("None", True, "universal_robot"),
            ("None", False, "other_robot"),
        ]:
            env = _minimal_env(_minimal_config(kind, curriculum), entity_name)
            environments.append(env)
            entity = env.scene[entity_name]
            entity.write_joint_state_to_sim(
                torch.full((2, 2), 0.75, device=env.device),
                torch.zeros(2, 2, device=env.device),
            )
            env.sim.reset(torch.tensor([0], device=env.device))
            expected = (
                0.75
                if kind == "None"
                and not curriculum
                and entity_name == "universal_robot"
                else 0.0
            )
            torch.testing.assert_close(
                entity.data.joint_pos[0], torch.full((2,), expected, device=env.device)
            )
            torch.testing.assert_close(
                entity.data.joint_pos[1], torch.full((2,), 0.75, device=env.device)
            )
    finally:
        for env in environments:
            env.close()
