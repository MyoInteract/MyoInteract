import re

import mujoco
import numpy as np

from omegaconf import MISSING
from dataclasses import dataclass, field

ARM_LENGTH: float = (
    0.65  # conservative reachability radius from shoulder (metres); used by reject_oor
)


@dataclass
class IndividualTargetConfig:
    name: str = MISSING
    rgb: list[float] = MISSING


@dataclass
class PointingTarget(IndividualTargetConfig):
    # penetrable: bool = False
    name: str = "pointing_target"
    # Position can either be a 3d vector or a 2 x list of 3d vectors specifying the min and max values for each dimension
    position: list[list[float]] = field(
        default_factory=lambda: [[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]]
    )
    shape: str = "sphere"
    # Size can either be a single value or a list of 2 values specifying the min and max values
    size: list[float] = field(default_factory=lambda: [0.05, 0.15])
    site_pos: list[float] = field(
        default_factory=lambda: [0, 0, 0.01]
    )  # deprecated, should be removed
    # Any rewards received when inside the target
    reward_incentive: float = 0.0
    completion_bonus: float = 0.0
    dwell_duration: float = 0.25
    rgb: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    reject_oor: bool = (
        False  # reject samples whose distance from shoulder exceeds ARM_LENGTH
    )


@dataclass
class ButtonTarget(IndividualTargetConfig):
    position: list[list[float]] = MISSING
    name: str = "button_target"
    size: list[list[float]] = field(
        default_factory=lambda: [[0.025, 0.025, 0.01], [0.025, 0.025, 0.01]]
    )
    site_pos: list[float] = field(default_factory=lambda: [0, 0, 0.01])
    geom_margin: float = 0.0
    completion_bonus: float = 0.0
    min_touch_force: float = 1.0
    rgb: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    euler: list[float] = field(default_factory=lambda: [0, -0.79, 0])
    reject_oor: bool = (
        False  # reject samples whose distance from shoulder exceeds ARM_LENGTH
    )


def generate_target(
    target_pos_range: np.ndarray,
    target_radius_range: np.ndarray,
    target_coordinates_origin: np.ndarray,
    rng: np.random.Generator,
    prev_pos: np.ndarray = None,
    min_distance: float = 0.05,
    reject_oor: bool = False,
):
    target_pos = generate_target_pos(
        target_pos_range,
        target_coordinates_origin,
        rng,
        prev_pos=prev_pos,
        min_distance=min_distance,
        reject_oor=reject_oor,
    )
    target_size = generate_target_size(target_radius_range, rng)
    return target_pos, target_size


def generate_target_pos(
    target_pos_range,
    target_coordinates_origin,
    rng: np.random.Generator,
    prev_pos: np.ndarray = None,
    min_distance: float = 0.05,
    reject_oor: bool = False,
):
    # Rejection sampling with a bounded number of draws: a range that lies entirely out of reach
    # (or within min_distance of prev_pos) would otherwise loop forever.
    for _ in range(10_000):
        new_pos = _target_pos_sampler(target_pos_range, target_coordinates_origin, rng)
        if prev_pos is not None and _unsatisfied_dist_constr(
            new_pos, prev_pos, min_distance=min_distance
        ):
            continue
        if (
            reject_oor
            and np.linalg.norm(new_pos - target_coordinates_origin) > ARM_LENGTH
        ):
            continue
        return new_pos
    raise ValueError(
        f"no target position in the range {np.asarray(target_pos_range).tolist()} (relative to the shoulder) "
        f"satisfies the constraints: within reach ({ARM_LENGTH} m) = {reject_oor}, at least {min_distance} m "
        "from the previous target. Move the target closer or turn off out-of-reach rejection."
    )


def out_of_reach_targets(targets_cfg) -> list[int]:
    """Indices of targets whose whole position range lies farther than ARM_LENGTH from the shoulder."""
    out = []
    for i in range(targets_cfg.num_targets):
        t = getattr(targets_cfg, f"target_{i}", None) or targets_cfg.target_0
        if not getattr(t, "reject_oor", False):
            continue
        lo, hi = (
            np.asarray(t.position[0], dtype=float),
            np.asarray(t.position[1], dtype=float),
        )
        closest = np.clip(
            0.0, np.minimum(lo, hi), np.maximum(lo, hi)
        )  # point of the box nearest the shoulder
        if np.linalg.norm(closest) > ARM_LENGTH:
            out.append(i)
    return out


def _unsatisfied_dist_constr(new_pos, prev_pos, min_distance=0.1):
    return np.linalg.norm(new_pos - prev_pos) < min_distance


def _target_pos_sampler(
    target_pos_range, target_coordinates_origin, rng: np.random.Generator
):
    sampled_pos = (
        rng.random(3) * (target_pos_range[1] - target_pos_range[0])
        + target_pos_range[0]
    )
    sampled_pos = target_coordinates_origin + sampled_pos
    return sampled_pos


def generate_target_size(target_radius_range, rng: np.random.Generator):
    target_size = (
        rng.random(3) * (target_radius_range[1] - target_radius_range[0])
        + target_radius_range[0]
    )
    return target_size


def add_sphere_to_spec(
    spec: mujoco.MjSpec,
    target_cfg: PointingTarget,
    target_id: int,
    target_coordinates_origin: np.ndarray = np.zeros(3),
    seed: int = None,
):
    ## TODO: unify usage of torch vs numpy
    rng = np.random.default_rng(seed)
    target_body_name = f"body_target_{target_id}"
    target_geom_name = f"geom_target_{target_id}"
    target_site_name = f"site_target_{target_id}"
    target_sensor_name = f"sensor_target_{target_id}"

    worldbody = spec.body("root-variant")
    if worldbody is None:
        worldbody = spec.worldbody
    else:
        # worldbody.gravcomp = 1  #counteract gravity to keep the target in place  #TODO: does not work?
        root_free_joint = worldbody.add_joint(
            name="root-variant-joint", type=mujoco.mjtJoint.mjJNT_FREE
        )
        root_free_joint.damping[0] = (
            1e10  # set huge damping on the free joint to keep both the body and the target added below in place
        )
    target_pos, target_size = generate_target(
        np.array(target_cfg.position),
        np.array(target_cfg.size),
        target_coordinates_origin,
        rng,
        reject_oor=getattr(target_cfg, "reject_oor", False),
    )
    target_body = worldbody.add_body(name=target_body_name, pos=target_pos)
    rgba = np.ones(4)
    rgba[:3] = target_cfg.rgb
    target_size = np.ones(3) * target_size  ##TODO: deprecated -- remove
    target_body.add_geom(
        name=target_geom_name, pos=np.zeros(3), size=target_size, rgba=rgba
    )
    # print(f"Added target {target_geom_name} to spec")

    #### only required for consistency with add_button_to_spec
    target_body.add_site(
        name=target_site_name,
        type=mujoco._enums.mjtGeom(6),
        pos=target_cfg.site_pos,
        rgba=rgba,
        size=0.001 * np.ones(3),
    )
    spec.add_sensor(
        name=target_sensor_name,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=target_site_name,
    )

    return spec


# Collision bit shared by the fingertips and the buttons' press pads, and the pad's half thickness (m).
FINGERTIP_BIT = 4  # 2 is the torso-contact bit (TORSO_CONTACT_BIT)
BUTTON_PAD_HALF_THICKNESS = 0.001
# Contact of a button: MuJoCo's default (time constant 0.02 s) lets a fingertip arriving at 1.5 m/s sink
# about 1 cm into it. The stiffest stable setting is twice the simulation step; ``priority`` makes
# it apply instead of being averaged with the finger's.
BUTTON_SOLREF_STEPS = 2.0
BUTTON_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
_FINGERTIP_BODY = re.compile(r"(distph\d|\ddistph|distal_thumb)")


def restrict_button_presses(spec: mujoco.MjSpec, ee_site: str) -> list[str]:
    """Let only the fingertips press buttons; call once after the targets are in the spec.

    Fingertip geoms (the collision geoms on the distal finger bodies, and on the body carrying the
    end-effector site) move to their own collision bit: they keep colliding with everything they
    collided with before, except the buttons' bodies, and they alone collide with the buttons'
    press pads (``add_button_to_spec``). The rest of the hand and arm still collides with the
    buttons' bodies but cannot produce a press.

    Args:
        spec: Model spec with the body model and the targets.
        ee_site: Name of the end-effector site (its body counts as a fingertip).

    Returns:
        Names of the fingertip geoms.
    """
    tip_bodies = {
        b.name for b in spec.bodies if _FINGERTIP_BODY.search(b.name.split("/")[-1])
    }
    for site in spec.sites:
        if site.name.split("/")[-1] == ee_site.split("/")[-1]:
            tip_bodies.add(site.parent.name)
    tips = []
    for body in spec.bodies:
        is_button = any(
            child.name.split("/")[-1].startswith("button_pad_") for child in body.bodies
        )
        for geom in body.geoms:
            if body.name in tip_bodies and geom.contype & 1:
                geom.contype = (geom.contype & ~1) | FINGERTIP_BIT
                tips.append(geom.name)
            elif geom.conaffinity & 1 and not is_button:
                geom.conaffinity |= (
                    FINGERTIP_BIT  # everything else the fingertips could touch before
                )
    return tips


def add_button_to_spec(
    spec: mujoco.MjSpec,
    target_cfg: ButtonTarget,
    target_id: int,
    target_coordinates_origin: np.ndarray = np.zeros(3),
    seed: int = None,
):
    rng = np.random.default_rng(seed)
    target_body_name = f"body_target_{target_id}"
    target_geom_name = f"geom_target_{target_id}"
    target_site_name = f"site_target_{target_id}"
    target_sensor_name = f"sensor_target_{target_id}"

    worldbody = spec.worldbody
    target_pos, target_size = generate_target(
        np.array(target_cfg.position),
        np.array(target_cfg.size),
        target_coordinates_origin,
        rng,
        reject_oor=getattr(target_cfg, "reject_oor", False),
    )
    target_body = worldbody.add_body(
        name=target_body_name, pos=target_pos, euler=target_cfg.euler
    )
    rgba = np.ones(4)
    rgba[:3] = target_cfg.rgb
    # The button's body: what every part of the arm and hand except the fingertips collides with.
    target_body.add_geom(
        name=target_geom_name,
        type=mujoco._enums.mjtGeom(6),
        size=target_size,
        margin=target_cfg.geom_margin,
        rgba=rgba,
        contype=1,
        conaffinity=1,
        priority=1,
        solref=[BUTTON_SOLREF_STEPS * spec.option.timestep, 1.0],
        solimp=BUTTON_SOLIMP,
    )
    # A press counts only where a fingertip pushes on the front face. The press pad is a copy of the
    # button's box on a body of its own, so the touch sensor reads its contacts alone, and only
    # fingertip geoms collide with it (restrict_button_presses). It has the button's full depth: a
    # fast fingertip tunnels through a thin slab and nothing behind it would stop it. The sensor's
    # site is a thin slab around the front face: contacts on the pad's sides or back are outside it.
    front = (
        1.0 if target_cfg.site_pos[2] >= 0 else -1.0
    )  # the face the sensor's site was on
    half_depth = float(target_size[2])
    pad_half = min(BUTTON_PAD_HALF_THICKNESS, 0.5 * half_depth)
    pad_size = [float(target_size[0]), float(target_size[1]), pad_half]
    # The target's position (observation and distance reward read the body's centre of mass) is the
    # centre of the front face, the surface the fingertip has to reach, not the middle of the box.
    target_body.explicitinertial = True
    target_body.ipos = [0, 0, front * half_depth]
    target_body.mass = 1e-3
    target_body.inertia = [1e-6, 1e-6, 1e-6]
    pad_body = target_body.add_body(name=f"button_pad_{target_id}")
    pad_body.add_geom(
        name=f"button_pad_geom_{target_id}",
        type=mujoco._enums.mjtGeom(6),
        size=[float(target_size[0]), float(target_size[1]), half_depth],
        margin=target_cfg.geom_margin,
        rgba=rgba,
        contype=0,
        conaffinity=FINGERTIP_BIT,
        priority=1,
        solref=[BUTTON_SOLREF_STEPS * spec.option.timestep, 1.0],
        solimp=BUTTON_SOLIMP,
    )
    pad_body.add_site(
        name=target_site_name,
        type=mujoco._enums.mjtGeom(6),
        pos=[0, 0, front * half_depth],
        rgba=[*rgba[:3], 0.0],
        size=pad_size,
    )
    if not np.allclose(target_cfg.size[0], target_cfg.size[1]):
        print(
            f"[myoInteract] button target {target_id}: its size is drawn anew at every reset, but its press pad keeps "
            "the size drawn when the model was built; give the button one size for exact press detection."
        )
    # print(f"Added target {target_geom_name} to spec")
    # Add touch sensor for the button
    spec.add_sensor(
        name=target_sensor_name,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=target_site_name,
    )
    # print(f"Added sensor {target_sensor_name} to spec")
    return spec


# Torso collision proxies + a dedicated contype/conaffinity bit for
# add_torso_contact_sensor_to_spec, below. mobl_arms_index_body.xml ships these exact
# 4 geoms (name/type/size/pos/euler, nested under a "thorax_coll_body" with this exact
# pos/euler) but with the WHOLE nested <body> commented out -- likely for collision
# cost/stability reasons, since re-enabling collision for the entire model isn't done
# lightly. Recreated here verbatim (not touching the shared MJCF) so the ORIGINAL
# authors' own carefully-sized collision primitives are reused rather than reinvented
# or approximated with the (much more expensive, less robust) full "thorax" mesh geom.
_TORSO_COLL_BODY_POS = (0.1, -0.455, -0.02)
_TORSO_COLL_BODY_EULER = (-1.57, 0.0, 0.0)
_TORSO_COLL_GEOMS = (
    {
        "name": "torso_contact_coll0",
        "type": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "size": (0.080, 0.077, 0),
        "pos": (-0.065, 0, 0.070),
        "euler": (1.57, 0, 0),
    },
    {
        "name": "torso_contact_coll1",
        "type": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        "size": (0.095, 0.140, 0.140),
        "pos": (-0.075, 0, 0.200),
        "euler": (1.57, 0, 0),
    },
    {
        "name": "torso_contact_coll2",
        "type": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "size": (0.110, 0.065, 0),
        "pos": (-0.090, 0, 0.340),
        "euler": (1.57, 0, 0),
    },
    {
        "name": "torso_contact_coll3",
        "type": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "size": (0.075, 0.120, 0),
        "pos": (-0.110, 0, 0.425),
        "euler": (1.57, 0, 0),
    },
)
# A bit not otherwise used by any existing contype/conaffinity value in this model
# (verified: every geom in mobl_arms_index_universal_myointeract.xml is currently either 0
# or 1 -- bit 1 (value 1) is already load-bearing for button/fingertip touch detection,
# see add_button_to_spec above, so this claims the NEXT bit instead of reusing it).
TORSO_CONTACT_BIT = 2

# MyoArm-lineage models (MyoArm, MyoArm_nohand) ship these exact 4 geoms as REAL,
# uncommented collision geometry already (verified byte-identical name/type/size/pos/euler
# to _TORSO_COLL_GEOMS above). The MoBL body files ship them commented-out. Order matches _TORSO_COLL_GEOMS positionally.
_NATIVE_TORSO_COLL_GEOM_NAMES = (
    "thorax_coll0",
    "thorax_coll1",
    "thorax_coll2",
    "thorax_coll3",
)

# Public: the sensor names add_torso_contact_sensor_to_spec wires up, for callers
# (e.g. a reward term) that need to read them back via env.scene[f"{entity}/{name}"].
TORSO_CONTACT_SENSOR_NAMES = tuple(f"sensor_{g['name']}" for g in _TORSO_COLL_GEOMS)


def add_torso_contact_sensor_to_spec(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Enable (real, physics-based) contact detection between the torso and every
    other body-part collision geom, and wire up touch sensors to read it.

    Scoped via a dedicated contype/conaffinity bit (TORSO_CONTACT_BIT) rather than a
    blanket contype=1/conaffinity=1: this model's arm/hand collision proxies
    (default classes "arm_skin"/"hand_skin") are contype=0/conaffinity=0 by design
    (self-collision is disabled model-wide, presumably for performance/stability), so
    a blanket re-enable would also newly collide the whole body against the floor/
    world/target geoms -- unwanted side effects well beyond "penalize torso contact".
    Only ADDS the new bit (via |=) to existing arm/hand geoms' contype, never
    overwriting their existing bits (e.g. "fingertip_contact"'s contype=1, load-bearing
    for the button-touch sensor above).

    Call only when the torso-contact-penalty reward term is actually enabled (see
    _torso_contact_penalty's docstring in register_mjlab_myointeract_tasks.py) -- the
    added geoms/sensors are otherwise pure overhead for tasks that don't use them.

    `humerus_coll` is deliberately excluded from the penalized group: at this
    model's neutral/default pose (no policy input at all), `humerus_coll` already
    overlaps `torso_contact_coll2`/`torso_contact_coll3` by design (the shoulder
    sits structurally against the upper torso) -- confirmed via direct
    `data.contact` inspection. Including it would make the penalty fire
    constantly regardless of policy behavior, which is very likely why the
    original MJCF authors commented these geoms out entirely rather than ship a
    always-on false positive. Excluding only `humerus_coll` (verified empirically
    to read a hard zero at neutral pose once excluded) while keeping the forearm
    (`radius_coll` + the unnamed arm_skin forearm capsules) and every `hand_skin`
    geom in the penalized group still correctly detects genuine self-collision --
    e.g. reaching a hand/forearm across the chest -- confirmed via a random-pose
    sweep that found large nonzero contact forces (radius_coll, hand phalanges,
    fingertip_contact all vs. torso_contact_coll1-3) once the arm is driven into
    such a pose.

    Caveat specific to MyoArm-lineage models (see below): `myoarm_assets.xml`
    ships explicit `<pair geom1="humerus_coll" geom2="thorax_coll0/1">` and
    `<pair geom1="radius_coll" geom2="thorax_coll0..3">` overrides, which force
    those specific geom pairs to collide *regardless* of contype/conaffinity --
    so excluding `humerus_coll` from this function's own bit-tagging does NOT
    suppress its contact with the torso on MyoArm models the way it does on
    MoBL (where no such explicit pair exists). Empirically this only produces a
    small, near-zero-penetration reading at the model's natural equilibrium
    pose (confirmed via a 60s settle test: `sensordata` for the affected sensor
    stays in the ~10, vs. ~1000+ at the unsettled `qpos0`, and every other
    sensor reads exactly zero) -- a pre-existing physical contact from MyoArm's
    own authors, not something introduced by this function, and left as-is.

    MyoArm-lineage models (MyoArm, MyoArm_nohand) ship the 4 torso collision geoms as real, active geometry already (named
    `thorax_coll0`..`thorax_coll3`, `class="myoarm_coll"`), instead of MoBL's
    commented-out `torso_contact_coll_body`. In that case this function reuses
    the existing geoms (only ADDing TORSO_CONTACT_BIT to their `conaffinity`,
    same additive discipline as everywhere else here) rather than adding a
    second, redundant, exactly-overlapping set of capsules -- and the
    class-name check below also recognizes `myoarm_coll`/`myohand_coll` (MyoArm's
    self-collision proxy classes) alongside MoBL's `arm_skin`/`hand_skin`, since
    without it no geom would ever be tagged with TORSO_CONTACT_BIT and the
    resulting sensors would silently always read zero.
    """
    _HUMERUS_EXCLUDE = {"humerus_coll"}
    _PENALIZED_CLASSES = ("arm_skin", "hand_skin", "myoarm_coll", "myohand_coll")
    thorax = spec.body("thorax")
    if thorax is None:
        raise ValueError(
            "add_torso_contact_sensor_to_spec: no 'thorax' body found in this model."
        )

    # myo_sim's composed MyoArm ships thorax_coll1..3 by name and the fourth torso capsule (the
    # abdomen, thorax_coll0 in the older files) without a name. Unnamed, it was not recognised, the
    # model was taken for a MoBL one and got MoBL's set of torso shapes: 36 cm too low on this model,
    # drawn, and carrying the contact sensors. Give the capsule its name back.
    named = {g.name for g in thorax.geoms}
    if (
        "thorax_coll0" not in named
        and {"thorax_coll1", "thorax_coll2", "thorax_coll3"} <= named
    ):
        unnamed = [
            g
            for g in thorax.geoms
            if not g.name and g.type == mujoco.mjtGeom.mjGEOM_CAPSULE and g.contype
        ]
        if len(unnamed) == 1:
            unnamed[0].name = "thorax_coll0"

    native_torso_geoms = {}
    for body in spec.bodies:
        for geom in body.geoms:
            if geom.name in _NATIVE_TORSO_COLL_GEOM_NAMES:
                native_torso_geoms[geom.name] = (body, geom)

    if len(native_torso_geoms) == len(_NATIVE_TORSO_COLL_GEOM_NAMES):
        # MyoArm-lineage model: the torso shapes exist. They cannot carry the sensors themselves:
        # the model pairs the upper arm with them explicitly (the shoulder rests against the torso),
        # so a touch sensor on them reads a contact at every step. The sensors sit on copies of the
        # four shapes instead, at the same place on a body of their own, invisible, and touched only
        # by the geoms tagged below (forearm and hand, not the upper arm).
        compiled = spec.copy().compile()
        coll_body = thorax.add_body(name="torso_contact_coll_body")
        for geom_spec, native_name in zip(
            _TORSO_COLL_GEOMS, _NATIVE_TORSO_COLL_GEOM_NAMES
        ):
            body, geom = native_torso_geoms[native_name]
            if body.name != thorax.name:
                raise ValueError(
                    f"add_torso_contact_sensor_to_spec: {native_name} is not on the thorax body."
                )
            gid = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, native_name)
            pos, quat = compiled.geom_pos[gid].copy(), compiled.geom_quat[gid].copy()
            coll_body.add_geom(
                name=geom_spec["name"],
                type=geom.type,
                size=np.array(geom.size),
                pos=pos,
                quat=quat,
                contype=0,
                conaffinity=TORSO_CONTACT_BIT,
                group=4,
                rgba=[0.8, 0.8, 0.8, 0.0],
            )
            site = coll_body.add_site(
                name=f"site_{geom_spec['name']}",
                type=geom.type,
                pos=pos,
                quat=quat,
                size=np.array(geom.size),
                rgba=[0.8, 0.8, 0.8, 0.0],
            )
            spec.add_sensor(
                name=f"sensor_{geom_spec['name']}",
                type=mujoco.mjtSensor.mjSENS_TOUCH,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                objname=site.name,
            )
    else:
        # MoBL lineage: the source MJCF ships these geoms commented out --
        # recreate them verbatim under a new body.
        coll_body = thorax.add_body(
            name="torso_contact_coll_body",
            pos=np.array(_TORSO_COLL_BODY_POS),
            euler=np.array(_TORSO_COLL_BODY_EULER),
        )
        for geom_spec in _TORSO_COLL_GEOMS:
            coll_body.add_geom(
                name=geom_spec["name"],
                type=geom_spec["type"],
                size=np.array(geom_spec["size"]),
                pos=np.array(geom_spec["pos"]),
                euler=np.array(geom_spec["euler"]),
                contype=0,
                conaffinity=TORSO_CONTACT_BIT,
            )
            # The site defining a touch sensor's zone must match the geom's own
            # type/size/pos/euler (not a tiny decorative point, unlike add_sphere_to_spec's
            # target site above): mjSENS_TOUCH sums contact forces whose contact point
            # falls within the SITE's own volume, not merely "any contact on this body" --
            # verified empirically: a tiny (0.001) site next to a real, confirmed
            # (data.contact-visible) overlap read a hard zero, while matching the site to
            # the geom's actual shape correctly read the nonzero contact force.
            site = coll_body.add_site(
                name=f"site_{geom_spec['name']}",
                type=geom_spec["type"],
                pos=np.array(geom_spec["pos"]),
                euler=np.array(geom_spec["euler"]),
                size=np.array(geom_spec["size"]),
            )
            spec.add_sensor(
                name=f"sensor_{geom_spec['name']}",
                type=mujoco.mjtSensor.mjSENS_TOUCH,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                objname=site.name,
            )

    _torso_geom_names = {g["name"] for g in _TORSO_COLL_GEOMS} | set(
        _NATIVE_TORSO_COLL_GEOM_NAMES
    )
    for geom in spec.geoms:
        # (the composed MyoArm names the upper arm's shape humerus_coll_r)
        if geom.name in _torso_geom_names or any(
            geom.name.startswith(n) for n in _HUMERUS_EXCLUDE
        ):
            continue
        default_name = geom.classname.name if geom.classname is not None else ""
        if default_name in _PENALIZED_CLASSES:
            geom.contype |= TORSO_CONTACT_BIT

    return spec
