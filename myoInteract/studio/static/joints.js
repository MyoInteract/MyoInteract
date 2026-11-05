// Joint angles of a policy playing in the 3D view. Each track spans the joint's limits (lower limit at
// the left edge, upper at the right); the dot is the angle now, the band the range the joint covers per
// episode, averaged over the last episodes. Fed with one angle vector per control step (the live frames'
// tail), in the order of the hello's `joints` ({ name, range: [lo, hi] }).

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const FINGERS = { 2: "index", 3: "middle", 4: "ring", 5: "little" };
const NAMES = {
  elv_angle: "Shoulder elevation plane", shoulder_elv: "Shoulder elevation", shoulder_rot: "Shoulder rotation",
  elbow_flexion: "Elbow flexion", pro_sup: "Pronation / supination", deviation: "Wrist deviation", flexion: "Wrist flexion",
  cmc_flexion: "Thumb CMC flexion", cmc_abduction: "Thumb CMC abduction", mp_flexion: "Thumb MCP flexion", ip_flexion: "Thumb IP flexion",
};
// Weight of the newest episode in the averaged range (about the last five episodes count).
const SMOOTHING = 0.3;

// Readable name of a joint ("mcp2_flexion_r" -> "Index MCP flexion").
export function jointLabel(name) {
  const n = name.replace(/_r$/, "");
  if (NAMES[n]) return NAMES[n];
  const m = n.match(/^(mcp|pm|md)(\d)_(flexion|abduction)$/);
  if (!m) return name;
  const finger = FINGERS[m[2]] || m[2];
  return `${finger[0].toUpperCase() + finger.slice(1)} ${{ mcp: "MCP", pm: "PIP", md: "DIP" }[m[1]]} ${m[3]}`;
}

// An angle for display: degrees for hinges (ranges up to a few radians), metres otherwise.
const show = (v, range) => (Math.abs(range[1] - range[0]) <= 7 ? `${Math.round((v * 180) / Math.PI)}°` : v.toFixed(3));

export class JointMeter {
  constructor(joints) {
    this.joints = joints;
    this.reset();
  }

  reset() {
    const n = this.joints.length;
    this.current = new Float32Array(n);
    this.epMin = new Float64Array(n).fill(Infinity);
    this.epMax = new Float64Array(n).fill(-Infinity);
    this.epSteps = 0;
    this.avgMin = null;  // per-episode range, averaged over the last episodes
    this.avgMax = null;
    this.episodes = 0;
    this.order = this.joints.map((_, k) => k);
  }

  // Position of angle v of joint k within its limits (0..1).
  u(k, v) {
    const [lo, hi] = this.joints[k].range;
    return hi > lo ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : 0.5;
  }

  push(q) {
    this.current.set(q);
    for (let k = 0; k < q.length; k++) {
      this.epMin[k] = Math.min(this.epMin[k], q[k]);
      this.epMax[k] = Math.max(this.epMax[k], q[k]);
    }
    this.epSteps++;
  }

  endEpisode() {
    if (!this.epSteps) return;
    if (!this.avgMin) { this.avgMin = Float64Array.from(this.epMin); this.avgMax = Float64Array.from(this.epMax); }
    else for (let k = 0; k < this.joints.length; k++) {
      this.avgMin[k] += SMOOTHING * (this.epMin[k] - this.avgMin[k]);
      this.avgMax[k] += SMOOTHING * (this.epMax[k] - this.avgMax[k]);
    }
    this.episodes++;
    this.epMin.fill(Infinity); this.epMax.fill(-Infinity);
    this.epSteps = 0;
    this.rank();
  }

  // The range shown for joint k: the averaged one, or the running episode's before the first one ends.
  band(k) {
    return this.avgMin ? [this.avgMin[k], this.avgMax[k]] : [Math.min(this.epMin[k], this.current[k]), Math.max(this.epMax[k], this.current[k])];
  }

  // The joints that move most first (share of their limits the band covers), re-ranked only between episodes.
  rank() {
    const used = this.joints.map((_, k) => { const [a, b] = this.band(k); return b > a ? this.u(k, b) - this.u(k, a) : 0; });
    this.order = this.joints.map((_, k) => k).sort((a, b) => used[b] - used[a]);
  }

  top(n) {
    if (!this.episodes && this.epSteps % 20 === 1) this.rank();
    return this.order.slice(0, n);
  }
}

// The `n` joints that move most: the track spans the limits, the band is the averaged range per
// episode, the dot (red at a limit) and the value are the angle now.
export function jointBarsHtml(meter, n) {
  return meter.top(n).map((k) => {
    const j = meter.joints[k], v = meter.current[k], label = esc(jointLabel(j.name)), [lo, hi] = meter.band(k);
    const a = meter.u(k, lo), b = meter.u(k, hi), now = meter.u(k, v), limit = now < 0.03 || now > 0.97;
    return `<div class="mbar jrow" title="${label} (${esc(j.name)}): limits ${show(j.range[0], j.range)} to ${show(j.range[1], j.range)}; range per episode ${show(lo, j.range)} to ${show(hi, j.range)}"><span class="mbar-name">${label}</span><span class="mbar-track jtrack"><span class="jband" style="left:${(100 * a).toFixed(1)}%;width:${(100 * Math.max(0, b - a)).toFixed(1)}%"></span><span class="jdot${limit ? " limit" : ""}" style="left:${(100 * now).toFixed(1)}%"></span></span><span class="mbar-val mono">${show(v, j.range)}</span></div>`;
  }).join("");
}

// Legend under the bars; `unit` is "episode" or the recording's window ("4 s").
export function jointNote(meter, unit = "episode") {
  const per = unit === "episode" ? "episode" : `${unit} window`;
  const band = meter.episodes ? `range per ${per}, averaged` : `range so far this ${per}`;
  return `<span class="mcard-note"><span class="jdot-key"></span> angle now · <span class="jband-key"></span> ${band} · edges: joint limits</span>`;
}
