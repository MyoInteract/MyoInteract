"""Sphere targets are visual only: they never collide, whatever the spec's main default."""

import mujoco
import numpy as np

from myoInteract.myointeract_utils import PointingTarget, add_sphere_to_spec


def _target_bits(spec: mujoco.MjSpec) -> tuple[int, int]:
    cfg = PointingTarget(position=[[0.3, 0, 0], [0.3, 0, 0]], size=[0.02, 0.02])
    model = add_sphere_to_spec(spec, cfg, 0, seed=0).compile()
    geom = model.geom("geom_target_0")
    return int(geom.contype[0]), int(geom.conaffinity[0])


def test_sphere_target_does_not_collide() -> None:
    assert _target_bits(mujoco.MjSpec()) == (0, 0)


def test_sphere_target_does_not_collide_after_attach() -> None:
    """A body model attached with an empty prefix must not decide the target's collision bits."""
    parent = mujoco.MjSpec()
    child = mujoco.MjSpec.from_string(
        '<mujoco><default><geom contype="1" conaffinity="1"/></default>'
        '<worldbody><body name="arm"><geom size=".05"/></body></worldbody></mujoco>'
    )
    parent.attach(child, prefix="", frame=parent.worldbody.add_frame(pos=np.zeros(3)))
    assert _target_bits(parent) == (0, 0)
