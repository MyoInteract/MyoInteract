"""Spatial tendon paths must survive the GPU-to-CPU render synchronization."""

from __future__ import annotations

from types import SimpleNamespace
import importlib.util

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myoInteract.mjlab_offscreen_renderer import _sync_render_data  # noqa: E402


_TENDON_XML = """<mujoco><worldbody>
      <site name="anchor" pos="0 0 0"/>
      <body name="moving" pos="1 0 0"><joint name="slide" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".05"/><site name="tip"/>
      </body>
    </worldbody><tendon><spatial name="tendon" width=".01"><site site="anchor"/><site site="tip"/></spatial></tendon></mujoco>"""


def test_render_sync_preserves_tendons_at_each_endpoint_position():
    model = mujoco.MjModel.from_xml_string(_TENDON_XML)
    reference, rendered = mujoco.MjData(model), mujoco.MjData(model)
    fields = (
        "xpos",
        "xquat",
        "xmat",
        "geom_xpos",
        "geom_xmat",
        "site_xpos",
        "site_xmat",
        "ten_wrapadr",
        "ten_wrapnum",
        "wrap_obj",
        "wrap_xpos",
    )
    option, camera = mujoco.MjvOption(), mujoco.MjvCamera()
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    for position in (0.5, -0.25):
        reference.qpos[0] = position
        mujoco.mj_forward(model, reference)
        sim = SimpleNamespace(
            mj_model=model,
            data=SimpleNamespace(
                **{
                    name: torch.as_tensor(getattr(reference, name).copy()).unsqueeze(0)
                    for name in fields
                }
            ),
            model=SimpleNamespace(
                geom_rgba=torch.as_tensor(model.geom_rgba.copy()).unsqueeze(0)
            ),
        )
        _sync_render_data(sim, rendered)
        for name in ("ten_wrapadr", "ten_wrapnum", "wrap_obj", "wrap_xpos"):
            np.testing.assert_allclose(
                getattr(rendered, name), getattr(reference, name)
            )
        scenes = [mujoco.MjvScene(model, maxgeom=100) for _ in range(2)]
        for data, scene in zip((reference, rendered), scenes):
            mujoco.mjv_updateScene(
                model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
            )
        tendons = [
            [
                scene.geoms[i]
                for i in range(scene.ngeom)
                if scene.geoms[i].objtype == mujoco.mjtObj.mjOBJ_TENDON
            ]
            for scene in scenes
        ]
        assert len(tendons[0]) == len(tendons[1]) == 1
        np.testing.assert_allclose(tendons[0][0].pos, tendons[1][0].pos)
        np.testing.assert_allclose(tendons[0][0].size, tendons[1][0].size)


@pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="Optional native backend unavailable",
)
def test_render_sync_reads_real_mjlab_tendon_bridge_fields():
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=2,
            entities={
                "arm": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(_TENDON_XML))
            },
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.01, gravity=(0, 0, 0)), njmax=16, nconmax=8
        ),
        decimation=1,
        episode_length_s=1.0,
    )
    env = ManagerBasedRlEnv(
        cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    try:
        env.scene["arm"].write_joint_state_to_sim(
            torch.tensor([[0.5], [-0.25]], device=env.device),
            torch.zeros(2, 1, device=env.device),
        )
        env.sim.forward()
        rendered = mujoco.MjData(env.sim.mj_model)
        _sync_render_data(env.sim, rendered)
        assert rendered.ten_wrapnum.tolist() == [2]
        option, camera = mujoco.MjvOption(), mujoco.MjvCamera()
        option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        scene = mujoco.MjvScene(env.sim.mj_model, maxgeom=100)
        mujoco.mjv_updateScene(
            env.sim.mj_model,
            rendered,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        tendons = [
            scene.geoms[i]
            for i in range(scene.ngeom)
            if scene.geoms[i].objtype == mujoco.mjtObj.mjOBJ_TENDON
        ]
        assert len(tendons) == 1
        np.testing.assert_allclose(tendons[0].pos, [0.75, 0, 0])
    finally:
        env.close()
