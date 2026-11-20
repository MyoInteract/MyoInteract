"""CPU MuJoCo body-model construction shared by Studio and training."""

import mujoco
import myo_sim
import numpy as np

from myoInteract.paths import REPO_ROOT


# myo_sim's `dev` branch restructured its flat XML layout (arm/assets/myoarm_assets.xml,
# consumed via hardcoded relative <include>s from this file) into an installable Python
# package with a compose API (myo_sim.load_spec(name)). myoarm_universal_myointeract.xml's
# and myoarm_nohand_universal_myointeract.xml's own <include>s no longer resolve, so these
# three model_path values are dispatched to _load_myoarm_r_based_spec below instead of the
# generic MjSpec.from_file path. Every other model_path is deliberately left on the old
# MjSpec.from_file route.
_MYOARM_UNIVERSAL_MODEL_PATH = (
    "myosuite/envs/myo/assets/arm/myoarm_universal_myointeract.xml"
)
_MYOARM_NOHAND_UNIVERSAL_MODEL_PATH = (
    "myosuite/envs/myo/assets/arm/myoarm_nohand_universal_myointeract.xml"
)
# "MyoArm_reach": the exact arm-reach model of the myoArmReach* envs (myoarm_r with
# rigid, joint-free digits and no finger/thumb muscles or thumb joints), built by
# myosuite.envs.myo.myoedits.edit_fn_arm_reaching. The file is only a sentinel (an
# empty MJCF): the model itself is composed from myo_sim, like the two above.
_MYOARM_REACH_MODEL_PATH = (
    "myosuite/envs/myo/assets/arm/myoarm_reach_universal_myointeract.xml"
)
# "MyoArm_pointing": the arm-reach model with the hand in a pointing pose (index finger
# straight, the other fingers curled, the thumb on the middle finger), rigid like the
# arm-reach hand. Also only a sentinel path, composed from myo_sim.
_MYOARM_POINTING_MODEL_PATH = (
    "myosuite/envs/myo/assets/arm/myoarm_pointing_universal_myointeract.xml"
)
_MYOARM_NJMAX = (
    400  # constraint rows per world for the MyoArm models (overflows seen at ~125)
)
_MYOARM_VARIANT_BY_MODEL_PATH = {
    _MYOARM_UNIVERSAL_MODEL_PATH: "full",
    _MYOARM_NOHAND_UNIVERSAL_MODEL_PATH: "nohand",
    _MYOARM_REACH_MODEL_PATH: "reach",
    _MYOARM_POINTING_MODEL_PATH: "pointing",
}

_MYOTORSO_ARMS_UNIVERSAL_MODEL_PATH = (
    "myosuite/envs/myo/assets/torso/myotorso_arms_universal_myointeract.xml"
)
_MYOTORSO_ARMS_NOHAND_MODEL_PATH = (
    "myosuite/envs/myo/assets/torso/myotorso_arms_nohand_universal_myointeract.xml"
)
_MYOTORSO_ARMS_REACH_MODEL_PATH = (
    "myosuite/envs/myo/assets/torso/myotorso_arms_reach_universal_myointeract.xml"
)
_MYOTORSO_ARMS_ACTIVE_HEAD_MODEL_PATH = (
    "myosuite/envs/myo/assets/torso/myotorso_arms_active_head_universal_myointeract.xml"
)
_MYOTORSO_ARMS_NJMAX = 800
_MYOTORSO_ARMS_VARIANT_BY_MODEL_PATH = {
    _MYOTORSO_ARMS_UNIVERSAL_MODEL_PATH: "full",
    _MYOTORSO_ARMS_NOHAND_MODEL_PATH: "nohand",
    _MYOTORSO_ARMS_REACH_MODEL_PATH: "reach",
    _MYOTORSO_ARMS_ACTIVE_HEAD_MODEL_PATH: "active_head",
}

# Joint angles (degrees) of the pointing hand, on myo_sim's names: middle, ring and little
# finger curled, the thumb tucked onto the curled middle finger (found by search: its distal
# phalanx ends about 6 mm from the middle finger's middle phalanx); the index finger straight.
_POINTING_HAND_DEG = {
    "mcp3_flexion_r": 80,
    "pm3_flexion_r": 90,
    "md3_flexion": 60,
    "mcp4_flexion_r": 80,
    "pm4_flexion": 90,
    "md4_flexion_r": 60,
    "mcp5_flexion_r": 80,
    "pm5_flexion_r": 90,
    "md5_flexion_r": 60,
    "cmc_flexion": -34,
    "cmc_abduction": -10,
    "mp_flexion_r": -2,
    "ip_flexion_r": -40,
}
_DIGIT_ROOTS = ("firstmc_r", "secondmc_r", "thirdmc_r", "fourthmc_r", "fifthmc_r")


def _pose_digits(spec: mujoco.MjSpec, degrees: dict[str, float]) -> None:
    """Bake joint angles of the digits into the bodies' rest poses (before the joints are removed)."""
    model = spec.copy().compile()
    data = mujoco.MjData(model)
    for name, deg in degrees.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = np.radians(deg)
    mujoco.mj_kinematics(model, data)

    def bodies(body):
        yield body
        for child in body.bodies:
            yield from bodies(child)

    for root in _DIGIT_ROOTS:
        for body in bodies(spec.body(root)):
            i, parent = (
                model.body(body.name).id,
                model.body_parentid[model.body(body.name).id],
            )
            rot_parent = data.xmat[parent].reshape(3, 3)
            body.pos = rot_parent.T @ (data.xpos[i] - data.xpos[parent])
            conj = np.zeros(4)
            mujoco.mju_negQuat(conj, data.xquat[parent])
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, conj, data.xquat[i])
            body.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
            body.quat = quat


# myo_sim's composed "myoarm_r" model (passive torso scaffold + right arm/hand) suffixes
# every joint/site name with "_r" (some sites are renamed outright, not just suffixed --
# confirmed by direct inspection, not a general pattern this can be derived from). This
# renames back to myosuite's own existing convention -- _independent_joint_names and
# ee_site/ref_site in task configs (e.g. pointing_anthro_myoarm.json) both hardcode the
# unsuffixed names -- rather than updating every hardcoded reference across this file.
_MYOARM_JOINT_RENAMES = {
    "elv_angle_r": "elv_angle",
    "shoulder_elv_r": "shoulder_elv",
    "shoulder_rot_r": "shoulder_rot",
    "elbow_flexion_r": "elbow_flexion",
    "pro_sup_r": "pro_sup",
}
_MYOARM_SITE_RENAMES = {
    "IFtip_r": "IFtip",
    "Shoulder_marker_r": "R.Shoulder_marker",
}

# myoarm_nohand_universal_myointeract.xml ("MyoArm model WITHOUT HAND MUSCLES for mid-air
# interaction tasks", per its own <mujoco model="..."> title) keeps every hand joint,
# body, and site (including IFtip, still used as ee_site by every config that selects
# this model -- e.g. pointing_anthro_myoarm_nohand.json) and removes only the finger/
# thumb MUSCLE actuators, reducing the action space for tasks that don't need dexterous
# hand control while keeping IFtip usable as a passive fingertip reference point. This
# is NOT "delete the hand bodies" (confirmed by testing: that also deletes IFtip, which
# every consumer of this model_path requires) -- confirmed by muscle-count match against
# myoarm_nohand_universal_myointeract.xml's own pre-migration dependency (26 arm muscles
# survive, matching exactly).
_MYOARM_NOHAND_KEPT_MUSCLES = frozenset(
    {
        "DELT1",
        "DELT2",
        "DELT3",
        "SUPSP",
        "INFSP",
        "SUBSC",
        "TMIN",
        "TMAJ",
        "PECM1",
        "PECM2",
        "PECM3",
        "LAT1",
        "LAT2",
        "LAT3",
        "CORB",
        "TRIlong",
        "TRIlat",
        "TRImed",
        "ANC",
        "SUP",
        "BIClong",
        "BICshort",
        "BRA",
        "BRD",
        "PT",
        "PQ",
    }
)

_myoarm_spec_cache: dict[str, mujoco.MjSpec] = {}


def _load_myoarm_r_based_spec(variant: str) -> mujoco.MjSpec:
    """Build a myoarm_r-based model: ``"full"`` replaces myoarm_universal_myointeract.xml,
    ``"nohand"`` myoarm_nohand_universal_myointeract.xml (hand muscles pruned) and
    ``"reach"`` is the rigid-hand arm-reach model of the myoArmReach* envs (no
    finger/thumb muscles, no finger/thumb joints; wrist muscles and joints kept).

    Cached (module-level, keyed only by variant -- the base spec is identical across
    every config sharing one of these model_paths): MjSpec objects support .copy()
    cheaply for per-call task-geom mutation without rebuilding from myo_sim each time.
    """
    if variant in _myoarm_spec_cache:
        return _myoarm_spec_cache[variant].copy()

    spec = myo_sim.load_spec("myoarm_r")
    if variant in ("reach", "pointing"):
        # Same edit (and thus the same model) as the myoArmReach* envs; runs on the
        # myo_sim `_r` names, before the renames below. It also adds a world-frame
        # "IFtip_r_target" marker site the GUI does not want (it has its own targets).
        # The pointing hand is posed first; the edit then makes the digits rigid as posed.
        from myosuite.envs.myo.myoedits import edit_fn_arm_reaching

        if variant == "pointing":
            _pose_digits(spec, _POINTING_HAND_DEG)
        edit_fn_arm_reaching(spec)
        # The marker's name differs between MyoSuite versions; newer edits also add their own "IFtip" copy,
        # which would clash with IFtip_r's rename below. (Lookup by name misses sites the edit just added.)
        names = {site.name for site in spec.sites}
        for site in list(spec.sites):
            if site.name in ("IFtip_r_target", "IFtip_target") or (
                site.name == "IFtip" and "IFtip_r" in names
            ):
                spec.delete(site)
    if variant == "nohand":
        for actuator in list(spec.actuators):
            if actuator.name not in _MYOARM_NOHAND_KEPT_MUSCLES:
                spec.delete(actuator)

    # add_torso_contact_sensor_to_spec (myointeract_utils.py) hardcodes body name "thorax"
    # (myosuite's old flat-layout convention); myo_sim's composed model names the
    # equivalent body "torso".
    spec.body("torso").name = "thorax"
    for joint in spec.joints:
        if joint.name in _MYOARM_JOINT_RENAMES:
            joint.name = _MYOARM_JOINT_RENAMES[joint.name]
    for site in spec.sites:
        if site.name in _MYOARM_SITE_RENAMES:
            site.name = _MYOARM_SITE_RENAMES[site.name]
    # Equality constraints (myoarm's scapulothoracic-rhythm phantom joints, coupled to
    # the driver joints above) reference joints by name in their own name1/name2
    # fields -- renaming the joint object alone leaves these dangling.
    for eq in spec.equalities:
        if eq.name1 in _MYOARM_JOINT_RENAMES:
            eq.name1 = _MYOARM_JOINT_RENAMES[eq.name1]
        if eq.name2 in _MYOARM_JOINT_RENAMES:
            eq.name2 = _MYOARM_JOINT_RENAMES[eq.name2]

    # Placement: myoarm_universal_myointeract.xml (and myoarm_nohand_universal_myointeract.xml
    # -- byte-identical apart from which assets file it includes, confirmed via diff)
    # wrapped the old bare arm+hand chain (no torso) in a "full_body" body at
    # pos=(-0.01913405, -0.03180473, 1.00001133), euler=(0, 0, 1.57). myo_sim's
    # composed model comes with its own default torso placement instead. This pos is
    # NOT that same pos reused verbatim -- it's a different value, numerically solved
    # (not derived analytically) so that R.Shoulder_marker's world position exactly
    # reproduces the pre-migration value documented in _compute_target_origin's own
    # docstring ([-0.0068, -0.1747, 1.0257]): apply the same euler rotation at the
    # origin, read the resulting R.Shoulder_marker offset, then solve pos = target -
    # that offset. Verified this same transform also reproduces IFtip's pre-migration
    # world position to within 1cm, confirming the underlying kinematic chain is
    # otherwise unchanged.
    root_body = spec.body("Full Body")
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, [0.0, 0.0, 1.57], "xyz")
    root_body.quat = quat.tolist()
    root_body.pos = [0.0821302244, -0.000105605942, 0.518801120]

    # Re-add content myo_sim's registry doesn't provide at all: myoarm_universal_myointeract.xml's
    # own eye mesh/camera + 2 fixed viewpoint cameras + light. Cosmetic/rendering-only
    # (vision is disabled by default in every config using this model_path so far), kept
    # for parity rather than dropped silently.
    spec.add_mesh(
        name="eye",
        file=str(REPO_ROOT / "myo_user/simhive/uitb_sim/assets/basic_eye_2.stl"),
        scale=[0.05, 0.05, 0.05],
    )
    spec.worldbody.add_light(
        name="scene_light",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=0,
        cutoff=4,
        exponent=20,
        diffuse=[1, 1, 1],
        specular=[0, 0, 0],
        pos=[0.5, 0.5, 3.0],
    )
    fixed_eye = spec.worldbody.add_body(
        name="fixed-eye",
        pos=[0, 0, 1.2],
        quat=[0.583833, 0.399104, -0.399421, -0.583368],
    )
    fixed_eye.add_geom(
        name="fixed-eye",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        quat=[0.7106, 0.25537, 0.616985, 0.221728],
        meshname="eye",
        rgba=[1, 1, 1, 0],
        contype=0,  # invisible, never collides (MuJoCo 4 would default it to 1/1)
        conaffinity=0,
    )
    fixed_eye.add_camera(name="fixed-eye", pos=[0, 0, 0], fovy=120)
    side_camera = spec.worldbody.add_body(
        name="side_camera", pos=[0.25, -0.27, 1.0], euler=[1.5708, 0, 0]
    )
    side_camera.add_camera(name="side_camera", pos=[0, 0, 1.2], fovy=70)
    head_on_camera = spec.worldbody.add_body(
        name="head_on_camera", pos=[1.5, 0.0, 1.0], euler=[1.5708, 1.5708, 0.0]
    )
    head_on_camera.add_camera(name="head_on_camera", pos=[0, 0, 12], fovy=10)

    _myoarm_spec_cache[variant] = spec
    return spec.copy()


def _myoarm_r_variant_for(model_path: str) -> str | None:
    """Return the _load_myoarm_r_based_spec variant model_path routes to, or None if
    model_path isn't one of the myo_sim migration model_path sentinels."""
    return _MYOARM_VARIANT_BY_MODEL_PATH.get(model_path)


_HALF_SQRT2 = float(np.sqrt(0.5))
_HEAD_CAMERA_QUAT = [_HALF_SQRT2, 0.0, -_HALF_SQRT2, 0.0]
_BILATERAL_DIGIT_ROOTS = (
    "firstmc_r",
    "secondmc_r",
    "thirdmc_r",
    "fourthmc_r",
    "fifthmc_r",
    "firstmc_l",
    "secondmc_l",
    "thirdmc_l",
    "fourthmc_l",
    "fifthmc_l",
)
_myotorso_arms_spec_cache: dict[str, mujoco.MjSpec] = {}
_myoarm_hand_muscles_cache: frozenset[str] | None = None


def _set_poly_or_scalar(obj: object, attr: str, value: float) -> None:
    current = getattr(obj, attr)
    if isinstance(current, (int, float)):
        setattr(obj, attr, float(value))
    else:
        setattr(obj, attr, [float(value), 0.0, 0.0])


def unlock_cervical_head_joints(
    spec: mujoco.MjSpec,
    *,
    include_roll: bool = False,
    stiffness: float = 6.0,
    damping: float = 0.8,
    armature: float = 0.01,
    add_actuators: bool = True,
    kp: float = 25.0,
    kv: float = 2.0,
) -> mujoco.MjSpec:
    """Unlock cervical degrees of freedom on ``neck`` and ``head`` in place."""
    try:
        from myo_sim.build.head import unlock_head_joints  # type: ignore[import-not-found]
    except ImportError:
        unlock_head_joints = None

    if unlock_head_joints is not None:
        return unlock_head_joints(
            spec,
            include_roll=include_roll,
            stiffness=stiffness,
            damping=damping,
            armature=armature,
            add_actuators=add_actuators,
            kp=kp,
            kv=kv,
        )

    if stiffness < 0.0 or damping < 0.0 or armature < 0.0:
        raise ValueError("stiffness, damping, and armature must be non-negative")
    if add_actuators and (kp <= 0.0 or kv < 0.0):
        raise ValueError(
            "kp must be positive and kv non-negative when add_actuators=True"
        )

    neck = spec.body("neck")
    head = spec.body("head")
    if neck is None or head is None:
        raise KeyError("Spec must contain both 'neck' and 'head' bodies.")

    joint_defs = [
        (neck, "neck_rotation", [-0.02, 0.4, 0.0], [0.2, 1.0, 0.0], [-1.4, 1.4]),
        (head, "neck_flexion", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-0.87, 1.05]),
    ]
    if include_roll:
        joint_defs.append(
            (head, "neck_bending", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.6, 0.6])
        )

    existing_joints = {j.name: j for j in spec.joints}
    existing_actuators = {a.name: a for a in spec.actuators}
    for body, jname, jpos, jaxis, jrange in joint_defs:
        jnt = existing_joints.get(jname)
        if jnt is None:
            jnt = body.add_joint(name=jname)
        jnt.type = mujoco.mjtJoint.mjJNT_HINGE
        jnt.pos = jpos
        jnt.axis = jaxis
        jnt.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        jnt.range = jrange
        _set_poly_or_scalar(jnt, "stiffness", stiffness)
        _set_poly_or_scalar(jnt, "damping", damping)
        jnt.armature = float(armature)

        if add_actuators:
            act_name = f"{jname}_act"
            act = existing_actuators.get(act_name)
            if act is None:
                act = spec.add_actuator(name=act_name)
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.target = jname
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = float(kp)
            act.biasprm[0] = 0.0
            act.biasprm[1] = -float(kp)
            act.biasprm[2] = -float(kv)
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = jrange
    return spec


def attach_head_ocular_rig(
    spec: mujoco.MjSpec,
    *,
    ipd: float = 0.063,
    fovy_fovea: float = 24.0,
    fovy_periphery: float = 90.0,
    fovy_binocular: float = 75.0,
    add_head_imu: bool = True,
) -> mujoco.MjSpec:
    """Attach cyclopean/binocular cameras, gaze sites, and 6-axis IMU sensors to ``head``."""
    try:
        from myo_sim.build.head import add_ocular_rig  # type: ignore[import-not-found]
    except ImportError:
        add_ocular_rig = None

    if add_ocular_rig is not None:
        return add_ocular_rig(
            spec,
            ipd=ipd,
            fovy_fovea=fovy_fovea,
            fovy_periphery=fovy_periphery,
            fovy_binocular=fovy_binocular,
            add_head_imu=add_head_imu,
        )

    if ipd <= 0.0 or not np.isfinite(ipd):
        raise ValueError(f"ipd must be positive and finite, got {ipd}")
    for label, val in (
        ("fovy_fovea", fovy_fovea),
        ("fovy_periphery", fovy_periphery),
        ("fovy_binocular", fovy_binocular),
    ):
        if not (0.0 < float(val) < 180.0):
            raise ValueError(f"{label} must lie in (0, 180), got {val}")

    head = spec.body("head")
    if head is None:
        raise KeyError("Spec must contain a 'head' body.")

    existing_sites = {s.name: s for s in spec.sites}
    existing_cams = {c.name: c for c in spec.cameras}
    existing_sensors = {s.name: s for s in spec.sensors}

    eye_x, eye_y = 0.085, 0.065
    for sname, spos in (
        ("gaze_origin", [eye_x, eye_y, 0.0]),
        ("gaze_forward", [eye_x + 0.10, eye_y, 0.0]),
    ):
        site = existing_sites.get(sname)
        if site is None:
            site = head.add_site(name=sname)
        site.pos = spos
        site.size = [0.004, 0.004, 0.004]
        site.group = 3

    half_ipd = 0.5 * float(ipd)
    for cname, cpos, cfovy in (
        ("eye_l", [eye_x, eye_y, -half_ipd], float(fovy_binocular)),
        ("eye_r", [eye_x, eye_y, half_ipd], float(fovy_binocular)),
        ("head_fovea", [eye_x, eye_y, 0.0], float(fovy_fovea)),
        ("head_periphery", [eye_x, eye_y, 0.0], float(fovy_periphery)),
    ):
        cam = existing_cams.get(cname)
        if cam is None:
            cam = head.add_camera(name=cname)
        cam.pos = cpos
        cam.quat = list(_HEAD_CAMERA_QUAT)
        cam.fovy = cfovy

    if add_head_imu:
        imu_site = existing_sites.get("head_imu")
        if imu_site is None:
            imu_site = head.add_site(name="head_imu")
        imu_site.pos = [0.0, eye_y, 0.0]
        imu_site.quat = list(_HEAD_CAMERA_QUAT)
        imu_site.size = [0.005, 0.005, 0.005]
        imu_site.group = 3

        for sname, stype in (
            ("head_accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
            ("head_gyro", mujoco.mjtSensor.mjSENS_GYRO),
            ("head_quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT),
        ):
            sen = existing_sensors.get(sname)
            if sen is None:
                sen = spec.add_sensor(name=sname)
            sen.type = stype
            sen.objtype = mujoco.mjtObj.mjOBJ_SITE
            sen.objname = "head_imu"
    return spec


def _bilateral_hand_muscle_names() -> frozenset[str]:
    global _myoarm_hand_muscles_cache
    if _myoarm_hand_muscles_cache is None:
        arm_model = myo_sim.load_spec("myoarm_r").compile()
        right_acts = {arm_model.actuator(i).name for i in range(arm_model.nu)}
        right_hand = right_acts - _MYOARM_NOHAND_KEPT_MUSCLES
        _myoarm_hand_muscles_cache = frozenset(
            right_hand
            | {f"{n}_l" for n in right_hand}
            | {n.replace("_r", "_l") for n in right_hand}
        )
    return _myoarm_hand_muscles_cache


def _immobilize_digit_subtree(spec: mujoco.MjSpec, root_body: mujoco.MjsBody) -> None:
    def _walk(body: mujoco.MjsBody) -> list[mujoco.MjsBody]:
        nodes = [body]
        for child in body.bodies:
            nodes.extend(_walk(child))
        return nodes

    removed_joints: set[str] = set()
    bodies = _walk(root_body)
    for b in bodies:
        for j in list(b.joints):
            removed_joints.add(j.name)
            spec.delete(j)

    if removed_joints:
        for eq in list(spec.equalities):
            if eq.type == mujoco.mjtEq.mjEQ_JOINT and (
                eq.name1 in removed_joints or eq.name2 in removed_joints
            ):
                spec.delete(eq)
        for act in list(spec.actuators):
            if (
                act.trntype == mujoco.mjtTrn.mjTRN_JOINT
                and act.target in removed_joints
            ):
                spec.delete(act)


def myotorso_arms_spec(variant: str = "nohand") -> mujoco.MjSpec:
    """Compose a bimanual upper-body ``myotorso_arms`` spec for the requested variant.

    Args:
        variant: One of ``"full"``, ``"nohand"`` (bilateral finger/thumb muscles pruned),
            ``"reach"`` (rigid bilateral digits + zero-moment tendons pruned), or
            ``"active_head"`` (``"nohand"`` plus unlocked cervical neck joints, servo
            actuators, dual-acuity ocular cameras, and 6-axis head IMU sensors).

    Returns:
        A fresh ``MjSpec`` copy ready for task geom attachment or compilation.
    """
    if variant not in ("full", "nohand", "reach", "active_head"):
        raise ValueError(
            f"Unknown myotorso_arms variant {variant!r}; expected "
            "'full', 'nohand', 'reach', or 'active_head'."
        )
    if variant in _myotorso_arms_spec_cache:
        return _myotorso_arms_spec_cache[variant].copy()

    spec = myo_sim.load_spec("myotorso_arms")
    if variant == "reach":
        for root_name in _BILATERAL_DIGIT_ROOTS:
            root_body = spec.body(root_name)
            if root_body is not None:
                _immobilize_digit_subtree(spec, root_body)
        tmp_model = spec.copy().compile()
        tmp_data = mujoco.MjData(tmp_model)
        mujoco.mj_step(tmp_model, tmp_data, nstep=50)
        j_tendon = np.empty((tmp_model.ntendon, tmp_model.nv))
        mujoco.mju_sparse2dense(
            j_tendon,
            tmp_data.ten_J,
            tmp_model.ten_J_rownnz,
            tmp_model.ten_J_rowadr,
            tmp_model.ten_J_colind,
        )
        for tendon in list(spec.tendons):
            tid = tmp_model.tendon(tendon.name).id
            if float(np.sum(np.abs(j_tendon[tid, :]))) < 1e-10:
                for act in list(spec.actuators):
                    if act.target == tendon.name:
                        spec.delete(act)
                spec.delete(tendon)
    elif variant in ("nohand", "active_head"):
        hand_muscles = _bilateral_hand_muscle_names()
        for actuator in list(spec.actuators):
            if actuator.name in hand_muscles:
                spec.delete(actuator)

    if variant == "active_head":
        unlock_cervical_head_joints(spec, add_actuators=True)
        attach_head_ocular_rig(spec, add_head_imu=True)

    torso_body = spec.body("torso")
    if torso_body is not None:
        torso_body.name = "thorax"

    site_names = {s.name for s in spec.sites}
    if "IFtip" not in site_names and "IFtip_r" in site_names:
        if_r = spec.site("IFtip_r")
        distph = spec.body("distph2_r")
        if if_r is not None and distph is not None:
            distph.add_site(
                name="IFtip",
                size=if_r.size,
                pos=if_r.pos,
                rgba=if_r.rgba,
                type=if_r.type,
            )

    existing_lights = {light.name for light in spec.lights}
    if "scene_light" not in existing_lights:
        spec.worldbody.add_light(
            name="scene_light",
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            castshadow=0,
            cutoff=4,
            exponent=20,
            diffuse=[1, 1, 1],
            specular=[0, 0, 0],
            pos=[0.5, 0.5, 3.0],
        )

    _myotorso_arms_spec_cache[variant] = spec
    return spec.copy()


def _myotorso_arms_variant_for(model_path: str) -> str | None:
    return _MYOTORSO_ARMS_VARIANT_BY_MODEL_PATH.get(model_path)


def base_spec(model_path: str) -> mujoco.MjSpec:
    """Spec of a body model (without task geoms) from a config ``model_path``.

    Args:
        model_path: Repo-root-relative model path, or one of the MyoArm / MyoTorsoArms
            sentinels that route to ``myo_sim``.

    Returns:
        A fresh spec the caller may edit.
    """
    variant = _myoarm_r_variant_for(model_path)
    if variant is not None:
        return _load_myoarm_r_based_spec(variant)
    torso_variant = _myotorso_arms_variant_for(model_path)
    if torso_variant is not None:
        return myotorso_arms_spec(torso_variant)
    return mujoco.MjSpec.from_file(str(REPO_ROOT / model_path))


def compile_base_model(model_path: str) -> mujoco.MjModel:
    """Compile a body model (without task geoms) from a config ``model_path`` (see ``base_spec``)."""
    return base_spec(model_path).compile()
