"""What the page needs to run an example policy itself (MuJoCo in WebAssembly, ``static/sim.js``).

    python -m myoInteract.studio.browser --name pointing

writes into ``examples/<name>/``: ``sim_model.xml`` (the task's physics model without visual meshes
and textures; the page draws its own scene), ``sim_policy.bin`` (the actor's weights, float32) and
``sim.json`` (what ``sim.js`` reads: names, timing, target ranges, network layout, and a short
reference rollout of native MuJoCo that the page's own simulation is tested against).
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path

import mujoco
import numpy as np

from myoInteract.studio.examples import EXAMPLES

try:  # myo-sim >= 0.2.5 (MyoSuite 3.1): the exact 3CC-r step; older installs keep the explicit one below
    from myo_sim.physics.fatigue import three_cc_r_step
except ImportError:
    three_cc_r_step = None

ENTITY = "universal_robot/"
_OPTIONS = (
    "timestep",
    "integrator",
    "cone",
    "solver",
    "iterations",
    "tolerance",
    "ls_iterations",
    "impratio",
    "jacobian",
    "disableflags",
    "enableflags",
    "noslip_iterations",
)


def _unwrap_defaults(parent: ET.Element) -> None:
    """Attached specs write a default class nested in itself, which MuJoCo's parser rejects."""
    for child in list(parent.findall("default")):
        _unwrap_defaults(child)
        if child.get("class") is not None and child.get("class") == parent.get("class"):
            at = list(parent).index(child)
            parent.remove(child)
            for k, sub in enumerate(list(child)):
                parent.insert(at + k, sub)


def physics_xml(spec: mujoco.MjSpec, compiled: mujoco.MjModel) -> str:
    """MJCF of ``spec`` with the dynamics of ``compiled`` and nothing that is only drawn.

    Visual mesh geoms, their meshes and file textures are removed; bodies keep the inertia those
    geoms gave them, and the options are the compiled model's (mjlab sets them after the spec).
    """
    for body in spec.bodies[1:]:
        if body.explicitinertial:
            continue
        i = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_BODY, body.name)
        body.explicitinertial = True
        body.mass, body.ipos, body.iquat, body.inertia = (
            compiled.body_mass[i],
            compiled.body_ipos[i],
            compiled.body_iquat[i],
            compiled.body_inertia[i],
        )
    mesh = mujoco.mjtGeom.mjGEOM_MESH
    for geom in [
        g
        for g in spec.geoms
        if g.type == mesh and g.contype == 0 and g.conaffinity == 0
    ]:
        spec.delete(geom)
    used = {g.meshname for g in spec.geoms if g.type == mesh}
    if used:
        raise ValueError(
            f"collision meshes are not shipped to the page yet: {sorted(used)[:3]}"
        )
    for m in list(spec.meshes):
        spec.delete(m)
    files = {t.name for t in spec.textures if t.file}
    for material in list(spec.materials):
        if any(t in files for t in material.textures):
            for geom in spec.geoms:
                if geom.material == material.name:
                    geom.material = ""
            spec.delete(material)
    for texture in list(spec.textures):
        if texture.name in files:
            spec.delete(texture)
    for name in _OPTIONS:
        setattr(spec.option, name, getattr(compiled.opt, name))
    spec.compile()
    root = ET.fromstring(spec.to_xml())
    _unwrap_defaults(root.find("default"))
    return ET.tostring(root, encoding="unicode")


def actor_weights(onnx_path: Path) -> tuple[dict, bytes]:
    """Layout and float32 weights of a checkpoint's actor (observation normaliser, then the MLP)."""
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    w = {
        t.name: numpy_helper.to_array(t)
        for t in onnx.load(str(onnx_path)).graph.initializer
    }
    std = next(v for k, v in w.items() if k.startswith("onnx::Div"))
    layers = sorted({int(k.split(".")[2]) for k in w if k.startswith("actor.mlp.")})
    arrays = [w["actor.obs_normalizer._mean"].ravel(), std.ravel()]
    sizes = []
    for k in layers:
        weight = w[f"actor.mlp.{k}.weight"]
        arrays += [weight.ravel(), w[f"actor.mlp.{k}.bias"].ravel()]
        sizes.append([int(weight.shape[1]), int(weight.shape[0])])
    return {"obs": int(std.size), "layers": sizes, "activation": "elu"}, np.concatenate(
        arrays
    ).astype("<f4").tobytes()


OBS_TERMS = (
    "qpos",
    "qvel",
    "qacc",
    "act",
    "ee_pos",
    "MA",
    "MR",
    "MF",
    "target_pos",
    "target_size",
    "phase_progress",
    "dwell_fraction",
)


def fatigue_params(mjlab_env) -> dict | None:
    """Per-muscle constants of the env's 3CC-r fatigue model and how it starts an episode (None without fatigue)."""
    term = mjlab_env.action_manager.get_term("muscles")
    state = term._fatigue
    if state is None:
        return None
    from myosuite.core.muscle_conditions import FATIGUE_REST_THRESHOLD  # noqa: PLC0415

    if term.cfg.fatigue_reset_vec is not None:
        raise ValueError(
            "the page's own simulation covers a fresh or a random fatigue state at reset so far"
        )
    lists = {
        k: getattr(state, "_" + k).cpu().numpy().astype(float).tolist()
        for k in ("F", "R", "r", "tauact", "taudeact")
    }
    return {
        **lists,
        "rest_threshold": FATIGUE_REST_THRESHOLD,
        "reset": "random" if term.cfg.fatigue_reset_random else "fresh",
    }


JOINTS = ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup")


def start_params(config: dict, model: mujoco.MjModel, mjlab_env=None) -> dict:
    """How an episode starts (the task's ``reset_type``), as data for ``sim.js``.

    ``joints`` are qpos addresses with their limits; an episode draws ``base + U(-eps, eps)``
    (``range``: anywhere within the limits), clamped to the limits, velocities ``U(-vel, vel)`` on the
    five arm joints, and for the fixed "ready" pose the muscle activations as well.
    """
    from myoInteract.register_mjlab_myointeract_tasks import (
        _INITPOSE001_ACT,
        _INITPOSE001_QPOS,
    )  # noqa: PLC0415

    kind = config["env"]["task_config"]["reset_type"]
    arm = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ENTITY + n) for n in JOINTS
    ]
    limits = lambda ids: model.jnt_range[ids].tolist()  # noqa: E731
    adr = lambda ids: [int(model.jnt_qposadr[j]) for j in ids]  # noqa: E731
    out = {"kind": kind, "approximate": False, "vel": 0.1, "act": None}
    if kind in ("epsilon_uniform", "zero", "visited_success") or (
        config["env"].get("curriculum_config") or {}
    ).get("enabled"):
        # visited_success and the curriculum start from states collected during training, which a
        # page does not have: it starts them like epsilon_uniform (flagged as approximate).
        eps = 0.0 if kind == "zero" else 0.1
        out.update(
            qadr=adr(arm),
            limits=limits(arm),
            base=model.qpos0[adr(arm)].tolist(),
            eps=eps,
            vel=eps,
            approximate=kind not in ("epsilon_uniform", "zero"),
        )
    elif kind == "range_uniform":
        lo, hi = model.jnt_range[arm].T
        out.update(
            qadr=adr(arm),
            limits=limits(arm),
            base=((lo + hi) / 2).tolist(),
            eps=((hi - lo) / 2).tolist(),
        )
    elif kind in ("initpose001", "initpose001_epsilon_uniform"):
        names = [
            n
            for n in (
                mjlab_env.scene["universal_robot"].joint_names
                if mjlab_env is not None
                else []
            )
        ]
        ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ENTITY + n)
            for n in names
        ]
        if len(ids) != len(_INITPOSE001_QPOS) or model.nu != len(_INITPOSE001_ACT):
            raise ValueError(
                "the fixed start pose needs the task's environment and the MoBL-ARMS (w/o hand muscles) model"
            )
        eps = 0.1 if kind.endswith("epsilon_uniform") else 0.0
        out.update(
            qadr=adr(ids),
            limits=limits(ids),
            base=list(_INITPOSE001_QPOS),
            eps=eps,
            vel=0.0,
            act={"base": list(_INITPOSE001_ACT), "eps": eps},
        )
    else:
        raise ValueError(
            f"the page's own simulation does not know the reset type {kind!r}"
        )
    out["vadr"] = [int(model.jnt_dofadr[j]) for j in arm]
    return out


def sim_task(
    config: dict,
    model: mujoco.MjModel,
    obs_terms: list[str] | None = None,
    fatigue: dict | None = None,
    start: dict | None = None,
) -> dict:
    """Names, timing and target ranges of a task (sphere or button targets, in sequence), as ``sim.js`` reads them.

    ``obs_terms`` is the order of the policy's observation terms (the env's), ``fatigue`` the output
    of :func:`fatigue_params`, ``start`` that of :func:`start_params` (computed here if not given).
    """
    env = config["env"]
    task, cfg = env["task_config"], env["task_config"]["targets"]
    noise = env["muscle_config"]["noise_params"]
    on = lambda k: noise[k + "_type"] not in (None, "None")  # noqa: E731
    # applied excitation = clip(u + signal_dependent * u * n1 + constant * n2, 0, 1), two standard normals per muscle
    motor_noise = {
        "signal_dependent": noise["sigdepnoise_level"] if on("sigdepnoise") else 0.0,
        "constant": noise["constantnoise_level"] if on("constantnoise") else 0.0,
    }
    obs_terms = list(
        obs_terms
        or (
            "qpos",
            "qvel",
            "qacc",
            "act",
            "ee_pos",
            "target_pos",
            "target_size",
            "phase_progress",
            "dwell_fraction",
        )
    )
    unknown = [t for t in obs_terms if t not in OBS_TERMS]
    if unknown:
        raise ValueError(
            f"the page's own simulation does not know the observation terms {unknown}"
        )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ref = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, ENTITY + task["reach_settings"]["ref_site"]
    )
    origin = data.site_xpos[ref] + np.asarray(
        task["reach_settings"]["target_origin_rel"]
    )
    targets = []
    for i in range(cfg["num_targets"]):
        t = cfg.get(f"target_{i}") or cfg["target_0"]
        button = t["name"] == "button_target"
        size = t["size"]
        targets.append(
            {
                "body": f"{ENTITY}body_target_{i}",
                "geom": f"{ENTITY}geom_target_{i}",
                # A button counts once its touch sensor reads the force; a sphere once the fingertip dwelt inside.
                "sensor": f"{ENTITY}sensor_target_{i}" if button else None,
                "min_force": t.get("min_touch_force", 0.0),
                "dwell_steps": t.get("dwell_duration", 0.0) / env["ctrl_dt"],
                "position": t["position"],
                "size": [[size[0]] * 3, [size[1]] * 3]
                if isinstance(size[0], (int, float))
                else size,
            }
        )
    return {
        "obs_terms": obs_terms,
        "fatigue": fatigue,
        "noise": motor_noise if max(motor_noise.values()) > 0 else None,
        "start": start or start_params(config, model),
        "joints": [ENTITY + n for n in JOINTS],
        "ee_site": ENTITY + task["reach_settings"]["ee_site"],
        "targets": targets,
        "substeps": round(env["ctrl_dt"] / env["sim_dt"]),
        "ctrl_dt": env["ctrl_dt"],
        "max_steps": round(task["max_duration"] / env["ctrl_dt"]),
        "origin": origin.tolist(),
    }


class Reference:
    """The browser's task/actor contract on the installed native MuJoCo engine.

    Physics follows that engine version; browser parity requires a version-matched
    reference rather than assuming different MuJoCo releases have identical dynamics.
    """

    def __init__(self, model: mujoco.MjModel, task: dict, layout: dict, weights: bytes):
        self.m, self.d, self.task = model, mujoco.MjData(model), task
        ids = lambda kind, names: [mujoco.mj_name2id(model, kind, n) for n in names]  # noqa: E731
        joints = ids(mujoco.mjtObj.mjOBJ_JOINT, task["joints"])
        self.qadr, self.vadr, self.range = (
            model.jnt_qposadr[joints],
            model.jnt_dofadr[joints],
            model.jnt_range[joints],
        )
        self.site = ids(mujoco.mjtObj.mjOBJ_SITE, [task["ee_site"]])[0]
        self.bodies = ids(
            mujoco.mjtObj.mjOBJ_BODY, [t["body"] for t in task["targets"]]
        )
        self.geoms = ids(mujoco.mjtObj.mjOBJ_GEOM, [t["geom"] for t in task["targets"]])
        self.sensors = [
            None
            if t["sensor"] is None
            else int(
                model.sensor_adr[ids(mujoco.mjtObj.mjOBJ_SENSOR, [t["sensor"]])[0]]
            )
            for t in task["targets"]
        ]
        if min(joints + self.bodies + self.geoms + [self.site]) < 0:
            raise ValueError(
                "the model lacks a joint, site, body or geom the task names"
            )
        flat, n = np.frombuffer(weights, "<f4"), layout["obs"]
        self.mean, self.std, at = flat[:n], flat[n : 2 * n], 2 * n
        self.layers = []
        for n_in, n_out in layout["layers"]:
            self.layers.append(
                (
                    flat[at : at + n_in * n_out].reshape(n_out, n_in),
                    flat[at + n_in * n_out : at + n_in * n_out + n_out],
                )
            )
            at += n_in * n_out + n_out
        self.current, self.inside = 0, 0
        self.rng, self.noisy = (
            np.random.default_rng(0),
            True,
        )  # motor noise; off for the reference rollout
        f = task.get("fatigue")
        self.fatigue = (
            None
            if f is None
            else {
                k: np.asarray(f[k], np.float32)
                for k in ("F", "R", "r", "tauact", "taudeact")
            }
        )
        n = model.nu
        self.MA, self.MF, self.MR = (
            np.zeros(n, np.float32),
            np.zeros(n, np.float32),
            np.ones(n, np.float32),
        )

    def reset(self, start, positions, sizes, fatigue_state=None) -> None:
        """Start an episode: ``start`` holds ``qpos`` (for the task's start joints), ``qvel`` (arm joints) and
        optionally ``act``; per target a world position and a size (3); with fatigue ``(MA, MF, MR)``."""
        n = self.m.nu
        ma, mf, mr = (
            fatigue_state
            if fatigue_state is not None
            else (np.zeros(n), np.zeros(n), np.ones(n))
        )
        self.MA, self.MF, self.MR = (
            np.asarray(x, np.float32).copy() for x in (ma, mf, mr)
        )
        m, d, st = self.m, self.d, self.task["start"]
        mujoco.mj_resetData(m, d)
        d.qpos[st["qadr"]] = np.clip(start["qpos"], *np.asarray(st["limits"]).T)
        d.qvel[st["vadr"]] = start["qvel"]
        if start.get("act") is not None:
            d.act[:] = np.clip(start["act"], 0, 1)
        for body, geom, position, size in zip(
            self.bodies, self.geoms, positions, sizes
        ):
            m.body_pos[body] = position
            m.geom_size[geom] = size
        mujoco.mj_forward(m, d)
        self.current, self.inside = 0, 0

    def sample(self, rng: np.random.Generator) -> tuple:
        """A start and targets drawn as training draws them (every axis on its own)."""
        task, st = self.task, self.task["start"]
        n = len(st["qadr"])
        state = {
            "qpos": np.asarray(st["base"])
            + rng.uniform(-1, 1, n) * np.asarray(st["eps"]),
            "qvel": rng.uniform(-1, 1, len(st["vadr"])) * st["vel"],
        }
        if st["act"] is not None:
            state["act"] = (
                np.asarray(st["act"]["base"])
                + rng.uniform(-1, 1, self.m.nu) * st["act"]["eps"]
            )
        positions = [
            np.asarray(task["origin"]) + rng.uniform(*t["position"])
            for t in task["targets"]
        ]
        start = (state, positions, [rng.uniform(*t["size"]) for t in task["targets"]])
        if task.get("fatigue") and task["fatigue"]["reset"] == "random":
            # as TorchFatigueState.reset: a non-fatigued share nf, of which a share ap is active
            nf, ap = rng.uniform(0, 1, self.m.nu), rng.uniform(0, 1, self.m.nu)
            return (*start, (nf * ap, 1 - nf, nf * (1 - ap)))
        return start

    def fatigue_step(self, excitation: np.ndarray) -> np.ndarray:
        """One control step of the 3CC-r model (myosuite ``TorchFatigueState.step``); returns the muscle ctrl ``MA``."""
        f, dt, e = (
            self.fatigue,
            np.float32(self.task["ctrl_dt"]),
            excitation.astype(np.float32),
        )
        ma, mf, mr = self.MA, self.MF, self.MR
        if three_cc_r_step is not None:  # myo-sim >= 0.2.5: the exact step
            ma, mr, mf = three_cc_r_step(
                np,
                ma,
                mf,
                e,
                F=f["F"],
                R=f["R"],
                r=f["r"],
                tauact=f["tauact"],
                taudeact=f["taudeact"],
                dt=float(dt),
                rest_threshold=self.task["fatigue"]["rest_threshold"],
            )
            self.MA, self.MF, self.MR = (
                np.asarray(x, np.float32) for x in (ma, mf, mr)
            )
            return self.MA
        ld = 1.0 / (f["tauact"] * (0.5 + 1.5 * ma))
        lr = (0.5 + 1.5 * ma) / f["taudeact"]
        ld, lr = -np.expm1(-ld * dt) / dt, -np.expm1(-lr * dt) / dt
        rr = np.where(
            e <= self.task["fatigue"]["rest_threshold"], f["r"] * f["R"], f["R"]
        )
        c = np.where(ma < e, ld * np.minimum(e - ma, mr), lr * (e - ma))
        c = np.clip(
            c,
            np.maximum(-ma / dt + f["F"] * ma, (mr - 1.0) / dt + rr * mf),
            np.minimum((1.0 - ma) / dt + f["F"] * ma, mr / dt + rr * mf),
        )
        self.MA = (ma + (c - f["F"] * ma) * dt).astype(np.float32)
        self.MF = (mf + (f["F"] * ma - rr * mf) * dt).astype(np.float32)
        self.MR = (mr + (-c + rr * mf) * dt).astype(np.float32)
        return self.MA

    def observe(self) -> np.ndarray:
        d, n = self.d, len(self.bodies)
        k = min(self.current, n - 1)
        dwell = self.task["targets"][k]["dwell_steps"]
        q = (d.qpos[self.qadr] - self.range[:, 0]) / (
            self.range[:, 1] - self.range[:, 0]
        )
        held = self.inside / max(dwell, 1) if dwell > 0 else 0.0
        terms = {
            "qpos": (q - 0.5) * 2,
            "qvel": d.qvel[self.vadr],
            "qacc": d.qacc[self.vadr],
            "act": (d.act - 0.5) * 2,
            "ee_pos": d.site_xpos[self.site],
            "MA": self.MA,
            "MR": self.MR,
            "MF": self.MF,
            "target_pos": d.xipos[self.bodies[k]],
            "target_size": self.m.geom_size[self.geoms[k]],
            "phase_progress": [-1 + 2 * self.current / n],
            "dwell_fraction": [-1 + 2 * held],
        }
        return np.concatenate(
            [np.asarray(terms[t], np.float64).ravel() for t in self.task["obs_terms"]]
        ).astype(np.float32)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize in JavaScript number precision, then store a Float32Array."""
        return ((np.asarray(obs, np.float64) - self.mean) / self.std).astype(np.float32)

    @staticmethod
    def excitation(action: np.ndarray) -> np.ndarray:
        """Apply TaskSim's sigmoid with one float32 rounding at its output."""
        action = np.asarray(action, np.float64)
        # Math.exp returns Infinity for large positive arguments. NumPy's
        # overflow result is the same; silence only that expected warning.
        with np.errstate(over="ignore"):
            return (1 / (1 + np.exp(-5 * (action - 0.5)))).astype(np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Match TaskSim's float64 arithmetic and float32 layer storage.

        JavaScript numbers hold each normalization, product sum and activation
        in float64; writing its Float32Array rounds once at each layer boundary.
        A float32 BLAS reduction or SIMD exponential changes that contract.
        """
        x = self.normalize(obs)
        for k, (w, b) in enumerate(self.layers):
            x = np.einsum("ij,j->i", w, x, dtype=np.float64, optimize=False) + b
            if k < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
            x = x.astype(np.float32)
        return x

    def step(self) -> bool:
        """One control step of the policy; True once every target is done."""
        excitation = self.excitation(self.act(self.observe()))
        noise = self.task.get("noise")
        if noise and self.noisy:
            n1, n2 = (
                self.rng.standard_normal(excitation.shape),
                self.rng.standard_normal(excitation.shape),
            )
            excitation = np.clip(
                excitation
                + noise["signal_dependent"] * excitation * n1
                + noise["constant"] * n2,
                0,
                1,
            ).astype(np.float32)
        self.d.ctrl[:] = (
            excitation if self.fatigue is None else self.fatigue_step(excitation)
        )
        for _ in range(self.task["substeps"]):
            mujoco.mj_step(self.m, self.d)
        k = min(self.current, len(self.bodies) - 1)
        target = self.task["targets"][k]
        distance = np.linalg.norm(
            self.d.site_xpos[self.site] - self.d.xipos[self.bodies[k]]
        )
        inside = distance < self.m.geom_size[self.geoms[k], 0]
        self.inside += int(inside)
        pressed = (
            self.sensors[k] is None
            or self.d.sensordata[self.sensors[k]] >= target["min_force"]
        )
        entered_or_button = self.sensors[k] is not None or inside
        if self.inside >= target["dwell_steps"] and pressed and entered_or_button:
            self.current, self.inside = self.current + 1, 0
        return self.current >= len(self.bodies)


def reference_rollout(ref: Reference, steps: int = 20) -> dict:
    """A fixed start and its first control steps, for the page-side test."""
    task, st = ref.task, ref.task["start"]
    pattern = np.resize([0.5, -0.8, 0.2, 0.9, -0.4], len(st["qadr"]))
    start = {
        "qpos": (np.asarray(st["base"]) + pattern * np.asarray(st["eps"])).tolist(),
        "qvel": (np.array([0.3, -0.6, 1.0, -0.2, 0.7]) * st["vel"]).tolist(),
    }
    if st["act"] is not None:
        start["act"] = np.clip(
            np.asarray(st["act"]["base"]) + 0.5 * st["act"]["eps"], 0, 1
        ).tolist()
    ref.noisy = False  # the page compares without noise
    positions = [
        (np.asarray(task["origin"]) + np.mean(t["position"], axis=0)).tolist()
        for t in task["targets"]
    ]
    sizes = [t["size"][1] for t in task["targets"]]
    n = ref.m.nu
    # with fatigue: muscles that start partly fatigued, differently per muscle
    state = (
        None
        if ref.fatigue is None
        else (
            (0.05 + 0.1 * np.arange(n) / n).tolist(),
            (0.2 + 0.4 * np.arange(n) / n).tolist(),
            (0.75 - 0.5 * np.arange(n) / n).tolist(),
        )
    )
    ref.reset(start, positions, sizes, state)
    qpos = []
    for _ in range(steps):
        ref.step()
        qpos.append(ref.d.qpos.tolist())
    ref.noisy = True
    return {
        "mujoco_version": mujoco.mj_versionString(),
        "start": start,
        "positions": positions,
        "sizes": sizes,
        "fatigue_state": state,
        "qpos": qpos,
        "current": ref.current,
        "MF": ref.MF.tolist(),
    }


def success_rate(ref: Reference, episodes: int, seed: int = 0) -> float:
    """Closed-loop success of the reference loop on freshly sampled starts and targets."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(episodes):
        ref.reset(*ref.sample(rng))
        wins += any(ref.step() for _ in range(ref.task["max_steps"]))
    return wins / episodes


def build(name: str, root: Path = EXAMPLES, episodes: int = 200) -> dict:
    """Write the page-side simulation files of the example ``name`` and return ``sim.json``."""
    from myoInteract.studio.worker import build_env, run_overrides  # noqa: PLC0415

    folder = root / name
    env, _ = build_env(run_overrides(folder), 1, "cuda:0")
    xml = physics_xml(env.scene.spec, env.sim.mj_model)
    obs_terms = list(env.observation_manager.active_terms["proprioception"])
    fatigue = fatigue_params(env)
    config = json.loads((folder / "myointeract_config.json").read_text())
    start = start_params(config, mujoco.MjModel.from_xml_string(xml), env)
    env.close()
    (folder / "sim_model.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    layout, weights = actor_weights(folder / "model_final.onnx")
    (folder / "sim_policy.bin").write_bytes(weights)
    task = sim_task(config, model, obs_terms, fatigue, start)
    ref = Reference(model, task, layout, weights)
    sim = {
        "model": "sim_model.xml",
        "policy": "sim_policy.bin",
        "network": layout,
        "task": task,
        "bodies": [model.body(b).name for b in range(model.nbody)],
        "native_success_rate": success_rate(ref, episodes),
        "reference": reference_rollout(ref),
    }
    (folder / "sim.json").write_text(json.dumps(sim) + "\n")
    return sim


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True, help="example to prepare, e.g. pointing")
    p.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="episodes of the native closed-loop check",
    )
    args = p.parse_args()
    sim = build(args.name, episodes=args.episodes)
    print(
        f"{args.name}: native MuJoCo success {sim['native_success_rate']:.3f} over {args.episodes} episodes"
    )


if __name__ == "__main__":
    main()
