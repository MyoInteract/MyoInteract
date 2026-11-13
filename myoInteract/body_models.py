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


def base_spec(model_path: str) -> mujoco.MjSpec:
    """Spec of a body model (without task geoms) from a config ``model_path``.

    Args:
        model_path: Repo-root-relative model path, or one of the MyoArm sentinels that route
            to MyoSuite's ``myoarm_r``.

    Returns:
        A fresh spec the caller may edit.
    """
    variant = _myoarm_r_variant_for(model_path)
    if variant is not None:
        return _load_myoarm_r_based_spec(variant)
    return mujoco.MjSpec.from_file(str(REPO_ROOT / model_path))


def compile_base_model(model_path: str) -> mujoco.MjModel:
    """Compile a body model (without task geoms) from a config ``model_path`` (see ``base_spec``)."""
    return base_spec(model_path).compile()
