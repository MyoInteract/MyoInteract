"""Buttons: where their position is measured, and that a fast fingertip does not sink into them."""

import mujoco
import numpy as np

from myoInteract.myointeract_utils import (
    FINGERTIP_BIT,
    ButtonTarget,
    add_button_to_spec,
)


def _button_model(speed: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """A button facing up at z = 0.5 and a fingertip-sized capsule flying down onto its front face."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.gravity = [0, 0, 0]
    cfg = ButtonTarget(position=[[0, 0, 0.5], [0, 0, 0.5]], euler=[0, 0, 0])
    add_button_to_spec(spec, cfg, 0, seed=0)
    tip = spec.worldbody.add_body(name="tip", pos=[0, 0, 0.53])
    tip.add_joint(name="tip_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    tip.add_geom(
        name="tip_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.007, 0.008, 0],
        mass=0.02,
        contype=FINGERTIP_BIT,
        conaffinity=0,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qvel[0] = -speed
    return model, data


def test_button_position_is_its_front_face() -> None:
    model, data = _button_model(0.0)
    mujoco.mj_forward(model, data)
    body, geom = model.body("body_target_0").id, model.geom("geom_target_0").id
    assert np.allclose(
        data.xipos[body],
        data.geom_xpos[geom] + [0, 0, model.geom_size[geom, 2]],
        atol=1e-9,
    )
    assert np.allclose(
        data.site_xpos[model.site("site_target_0").id], data.xipos[body], atol=1e-9
    )


def test_fast_fingertip_stays_out_of_the_button() -> None:
    model, data = _button_model(3.0)  # faster than the trained policies press
    pad, tip = model.geom("button_pad_geom_0").id, model.geom("tip_geom").id
    deepest, force, fromto = 0.0, 0.0, np.zeros(6)
    for _ in range(200):
        mujoco.mj_step(model, data)
        deepest = max(
            deepest, -mujoco.mj_geomDistance(model, data, tip, pad, 0.05, fromto)
        )
        force = max(force, float(data.sensor("sensor_target_0").data[0]))
    assert (
        deepest < 0.004
    ), f"the fingertip sank {1000 * deepest:.1f} mm into the button"
    assert force > 1.0  # and the press was sensed
