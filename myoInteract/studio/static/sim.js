// An example policy running in the page: MuJoCo (WebAssembly) steps the task's physics model and a
// small MLP picks the muscle excitations. Mirrors studio/browser.py's Reference step for step.

const sigmoid = (a) => 1 / (1 + Math.exp(-5 * (a - 0.5)));
// a standard normal draw (Box-Muller)
const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

export class TaskSim {
  // base: folder of an example (sim.json, sim_model.xml, sim_policy.bin); mujocoUrl: the vendored module.
  static async load(base, mujocoUrl = new URL("./vendor/mujoco/mujoco.js", import.meta.url).href) {
    const [{ default: loadMujoco }, info] = await Promise.all([import(mujocoUrl), fetch(`${base}/sim.json`).then((r) => {
      if (!r.ok) throw new Error(`could not load simulation metadata: HTTP ${r.status}`);
      return r.json();
    })]);
    const [mujoco, xml, weights] = await Promise.all([
      TaskSim.runtime || (TaskSim.runtime = loadMujoco().catch((e) => { TaskSim.runtime = null; throw e; })),
      fetch(`${base}/${info.model}`).then((r) => { if (!r.ok) throw new Error(`could not load simulation model: HTTP ${r.status}`); return r.text(); }),
      fetch(`${base}/${info.policy}`).then((r) => { if (!r.ok) throw new Error(`could not load simulation policy: HTTP ${r.status}`); return r.arrayBuffer(); }),
    ]);
    return new TaskSim(mujoco, info, xml, new Float32Array(weights));
  }

  constructor(mujoco, info, xml, weights) {
    this.mj = mujoco;
    this.info = info;
    this.task = info.task;
    const m = (this.model = mujoco.MjModel.from_xml_string(xml));
    this.data = new mujoco.MjData(m);
    const id = (kind, name) => {
      const i = mujoco.mj_name2id(m, kind.value, name);
      if (i < 0) throw new Error(`the model has no ${name}`);
      return i;
    };
    const joints = this.task.joints.map((n) => id(mujoco.mjtObj.mjOBJ_JOINT, n));
    this.qadr = joints.map((j) => m.jnt_qposadr[j]);
    this.vadr = joints.map((j) => m.jnt_dofadr[j]);
    this.range = joints.map((j) => [m.jnt_range[2 * j], m.jnt_range[2 * j + 1]]);
    this.site = id(mujoco.mjtObj.mjOBJ_SITE, this.task.ee_site);
    // muscles: actuators with an activation state, named without the entity prefix ("universal_robot/DELT1" -> "DELT1")
    this.actAdr = [];
    this.muscleNames = [];
    for (let a = 0; a < m.nu; a++) {
      if (m.actuator_actadr[a] < 0) continue;
      this.actAdr.push(m.actuator_actadr[a]);
      this.muscleNames.push(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR.value, a).split("/").pop());
    }
    // joints the scene reports (as studio/worker.py shown_joints): limited hinges and slides, not driven
    // by another joint through an equality constraint and not on a mocap body
    const driven = new Set();
    for (let e = 0; e < m.neq; e++) if (m.eq_type[e] === mujoco.mjtEq.mjEQ_JOINT.value) driven.add(m.eq_obj1id[e]);
    this.jointAdr = [];
    this.jointInfo = [];
    for (let j = 0; j < m.njnt; j++) {
      const kind = m.jnt_type[j];
      if ((kind !== mujoco.mjtJoint.mjJNT_HINGE.value && kind !== mujoco.mjtJoint.mjJNT_SLIDE.value) || !m.jnt_limited[j] || driven.has(j) || m.body_mocapid[m.jnt_bodyid[j]] >= 0) continue;
      this.jointAdr.push(m.jnt_qposadr[j]);
      this.jointInfo.push({ name: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT.value, j).split("/").pop(), range: [m.jnt_range[2 * j], m.jnt_range[2 * j + 1]] });
    }
    // per target: its body and geom, and for a button the address of its touch sensor's reading
    this.targets = this.task.targets.map((t) => ({
      ...t,
      bodyId: id(mujoco.mjtObj.mjOBJ_BODY, t.body),
      geomId: id(mujoco.mjtObj.mjOBJ_GEOM, t.geom),
      sensorAdr: t.sensor ? m.sensor_adr[id(mujoco.mjtObj.mjOBJ_SENSOR, t.sensor)] : -1,
    }));
    // actor: observation mean and std, then (weight, bias) per layer
    const n = info.network.obs;
    this.mean = weights.subarray(0, n);
    this.std = weights.subarray(n, 2 * n);
    let at = 2 * n;
    this.layers = info.network.layers.map(([nIn, nOut]) => {
      const layer = { nIn, nOut, w: weights.subarray(at, at + nIn * nOut), b: weights.subarray(at + nIn * nOut, at + nIn * nOut + nOut) };
      at += nIn * nOut + nOut;
      return layer;
    });
    this.obs = new Float32Array(n);
    this._normalized = new Float32Array(n);
    this._layerOutputs = this.layers.map((l) => new Float32Array(l.nOut));
    this._excitation = new Float32Array(m.nu);
    // muscle fatigue (3CC-r): active, fatigued and resting share per muscle; fresh muscles without it
    this.fatigue = this.task.fatigue || null;
    this.MA = new Float32Array(m.nu);
    this.MF = new Float32Array(m.nu);
    this.MR = new Float32Array(m.nu).fill(1);
    this.noisy = true;  // motor noise, where the task has it (the parity test switches it off)
    this.current = 0;  // index of the target the policy is on
    this.inside = 0;
    this.steps = 0;
    this.reset();
  }

  // A start and targets as training draws them (every axis on its own), or the given ones:
  // start = { qpos (for the task's start joints), qvel (arm joints), act (optional) }, per target a
  // world position and a size (3 numbers), and with fatigue the compartments [MA, MF, MR].
  reset({ start, positions, sizes, fatigueState } = {}) {
    const { mj, model: m, data: d, task } = this, st = task.start;
    const uni = (lo, hi) => lo + Math.random() * (hi - lo), eps = (k) => (Array.isArray(st.eps) ? st.eps[k] : st.eps);
    start = start || {
      qpos: st.qadr.map((_, k) => st.base[k] + uni(-1, 1) * eps(k)),
      qvel: st.vadr.map(() => uni(-1, 1) * st.vel),
      act: st.act ? st.act.base.map((b) => b + uni(-1, 1) * st.act.eps) : null,
    };
    positions = positions || this.targets.map((t) => task.origin.map((o, k) => o + uni(t.position[0][k], t.position[1][k])));
    sizes = sizes || this.targets.map((t) => [0, 1, 2].map((k) => uni(t.size[0][k], t.size[1][k])));
    mj.mj_resetData(m, d);
    st.qadr.forEach((a, k) => { d.qpos[a] = Math.min(st.limits[k][1], Math.max(st.limits[k][0], start.qpos[k])); });
    st.vadr.forEach((a, k) => { d.qvel[a] = start.qvel[k]; });
    if (start.act) start.act.forEach((v, k) => { d.act[k] = Math.min(1, Math.max(0, v)); });
    this.targets.forEach((t, i) => {
      for (let k = 0; k < 3; k++) { m.body_pos[3 * t.bodyId + k] = positions[i][k]; m.geom_size[3 * t.geomId + k] = sizes[i][k]; }
    });
    mj.mj_forward(m, d);
    if (fatigueState) { this.MA.set(fatigueState[0]); this.MF.set(fatigueState[1]); this.MR.set(fatigueState[2]); }
    else if (this.fatigue && this.fatigue.reset === "random") {
      // a non-fatigued share nf, of which a share ap is active
      for (let k = 0; k < m.nu; k++) { const nf = Math.random(), ap = Math.random(); this.MA[k] = nf * ap; this.MF[k] = 1 - nf; this.MR[k] = nf * (1 - ap); }
    } else { this.MA.fill(0); this.MF.fill(0); this.MR.fill(1); }
    this.current = 0;
    this.inside = 0;
    this.steps = 0;
  }

  // One control step of the fatigue model (myosuite's TorchFatigueState.step): the excitation the
  // policy asks for goes in, the active share MA comes out and is what the muscles receive.
  fatigueStep(excitation) {
    const f = this.fatigue, dt = this.task.ctrl_dt, { MA, MF, MR } = this;
    for (let k = 0; k < MA.length; k++) {
      const ma = MA[k], mf = MF[k], mr = MR[k], e = Math.fround(excitation[k]), F = f.F[k];
      const ld = -Math.expm1(-dt / (f.tauact[k] * (0.5 + 1.5 * ma))) / dt;
      const lr = -Math.expm1((-(0.5 + 1.5 * ma) / f.taudeact[k]) * dt) / dt;
      const rr = e <= f.rest_threshold ? f.r[k] * f.R[k] : f.R[k];
      let c = ma < e ? ld * Math.min(e - ma, mr) : lr * (e - ma);
      c = Math.min(Math.max(c, Math.max(-ma / dt + F * ma, (mr - 1) / dt + rr * mf)), Math.min((1 - ma) / dt + F * ma, mr / dt + rr * mf));
      MA[k] = ma + (c - F * ma) * dt;
      MF[k] = mf + (F * ma - rr * mf) * dt;
      MR[k] = mr + (-c + rr * mf) * dt;
    }
    return MA;
  }

  get meanFatigue() { let s = 0; for (const v of this.MF) s += v; return s / this.MF.length; }

  // The policy's observation, term by term in the order the training environment uses.
  observe() {
    const { model: m, data: d, obs } = this;
    const n = this.targets.length, t = this.targets[Math.min(this.current, n - 1)];
    let at = 0;
    const put = (v) => { obs[at++] = v; };
    for (const term of this.task.obs_terms) {
      if (term === "qpos") this.qadr.forEach((a, k) => { const [lo, hi] = this.range[k]; put(((d.qpos[a] - lo) / (hi - lo) - 0.5) * 2); });
      else if (term === "qvel") this.vadr.forEach((a) => put(d.qvel[a]));
      else if (term === "qacc") this.vadr.forEach((a) => put(d.qacc[a]));
      else if (term === "act") for (let k = 0; k < m.na; k++) put((d.act[k] - 0.5) * 2);
      else if (term === "ee_pos") for (let k = 0; k < 3; k++) put(d.site_xpos[3 * this.site + k]);
      else if (term === "MA" || term === "MF" || term === "MR") this[term].forEach(put);
      else if (term === "target_pos") for (let k = 0; k < 3; k++) put(d.xipos[3 * t.bodyId + k]);
      else if (term === "target_size") for (let k = 0; k < 3; k++) put(m.geom_size[3 * t.geomId + k]);
      else if (term === "phase_progress") put(-1 + 2 * this.current / n);
      else if (term === "dwell_fraction") put(-1 + 2 * (t.dwell_steps > 0 ? this.inside / Math.max(t.dwell_steps, 1) : 0));
      else throw new Error(`unknown observation term ${term}`);
    }
    return obs;
  }

  act(obs) {
    let x = this._normalized;
    for (let k = 0; k < obs.length; k++) x[k] = (obs[k] - this.mean[k]) / this.std[k];
    this.layers.forEach(({ nIn, nOut, w, b }, l) => {
      const y = this._layerOutputs[l], last = l === this.layers.length - 1;
      for (let i = 0; i < nOut; i++) {
        let s = b[i];
        for (let j = 0, row = i * nIn; j < nIn; j++) s += w[row + j] * x[j];
        y[i] = last || s > 0 ? s : Math.expm1(s);
      }
      x = y;
    });
    return x;
  }

  // One control step of the policy; true once every target is done.
  step() {
    const { mj, model: m, data: d, task } = this;
    const action = this.act(this.observe());
    const excitation = this._excitation;
    for (let k = 0; k < action.length; k++) excitation[k] = sigmoid(action[k]);
    if (task.noise && this.noisy) {
      // motor noise as in training: clip(u + signal_dependent * u * n1 + constant * n2, 0, 1)
      for (let k = 0; k < excitation.length; k++) {
        const u = excitation[k];
        excitation[k] = Math.min(1, Math.max(0, u + task.noise.signal_dependent * u * gauss() + task.noise.constant * gauss()));
      }
    }
    d.ctrl.set(this.fatigue ? this.fatigueStep(excitation) : excitation);
    for (let k = 0; k < task.substeps; k++) mj.mj_step(m, d);
    const n = this.targets.length, t = this.targets[Math.min(this.current, n - 1)];
    let dist = 0;
    for (let k = 0; k < 3; k++) dist += (d.site_xpos[3 * this.site + k] - d.xipos[3 * t.bodyId + k]) ** 2;
    const inside = Math.sqrt(dist) < m.geom_size[3 * t.geomId];
    if (inside) this.inside++;
    // a sphere counts once the fingertip dwelt inside, a button once its touch sensor reads the force
    const pressed = t.sensorAdr < 0 ? inside : d.sensordata[t.sensorAdr] >= t.min_force;
    if (this.inside >= t.dwell_steps && pressed) { this.current++; this.inside = 0; }
    this.steps++;
    return this.current >= n;
  }

  get timedOut() { return this.steps >= this.task.max_steps; }

  // The scene's live frame: per body position (3), then orientation (w, x, y, z), then each target's
  // position, then each muscle's activation, then each joint's angle (as the backend's live feed, studio/worker.py).
  frame(out) {
    const { model: m, data: d } = this, nb = m.nbody, nt = this.targets.length, na = this.actAdr.length;
    const f = out || new Float32Array(this.frameLength);
    f.set(d.xpos.subarray(0, 3 * nb), 0);
    f.set(d.xquat.subarray(0, 4 * nb), 3 * nb);
    this.targets.forEach((t, i) => f.set(d.xipos.subarray(3 * t.bodyId, 3 * t.bodyId + 3), 7 * nb + 3 * i));
    this.actAdr.forEach((a, k) => { f[7 * nb + 3 * nt + k] = d.act[a]; });
    this.jointAdr.forEach((a, k) => { f[7 * nb + 3 * nt + na + k] = d.qpos[a]; });
    return f;
  }

  get frameLength() { return 7 * this.model.nbody + 3 * this.targets.length + this.actAdr.length + this.jointAdr.length; }

  // Targets reached in the running episode (they are visited in order).
  get reached() { return this.current; }

  // Fatigued share (MF) per actuator, or null for a task without fatigue.
  get muscleFatigue() { return this.fatigue ? this.MF : null; }

  dispose() { this.data.delete(); this.model.delete(); }
}
