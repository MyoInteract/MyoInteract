# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from __future__ import annotations

from dataclasses import dataclass
import mujoco


@dataclass(frozen=True)
class MyoModelNames:
    actuator_names: list[str]
    tendon_actuator_names: list[str]
    non_tendon_actuator_names: list[str]
    tendon_names: list[str]
    joint_names: list[str]
    dependent_joint_names: list[str]
    independent_joint_names: list[str]


def myo_get_model_names(model: mujoco.MjModel) -> MyoModelNames:
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    tendon_actuator_names = [
        actuator_names[i]
        for i in range(model.nu)
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON
    ]
    non_tendon_actuator_names = [
        actuator_names[i]
        for i in range(model.nu)
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON
    ]
    tendon_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        for i in range(model.ntendon)
    ]
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    dependent_joint_ids = {
        int(model.eq_obj1id[eq_id])
        for eq_id in range(model.neq)
        if model.eq_active0[eq_id]
        and model.eq_type[eq_id] == mujoco.mjtEq.mjEQ_JOINT
        and 0 <= int(model.eq_obj1id[eq_id]) < model.njnt
    }
    return MyoModelNames(
        actuator_names=actuator_names,
        tendon_actuator_names=tendon_actuator_names,
        non_tendon_actuator_names=non_tendon_actuator_names,
        tendon_names=tendon_names,
        joint_names=joint_names,
        dependent_joint_names=[
            joint_names[i] for i in range(model.njnt) if i in dependent_joint_ids
        ],
        independent_joint_names=[
            joint_names[i] for i in range(model.njnt) if i not in dependent_joint_ids
        ],
    )
