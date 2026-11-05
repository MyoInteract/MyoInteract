// Muscle activity of a policy playing in the 3D view: live bars of the busiest muscles and per-episode
// means of the muscle groups. Fed with one activation vector per control step (the live frames' tail).

// Anatomical groups of the arm models' muscles (MyoArm and MoBL share the OpenSim names).
export const MUSCLE_GROUPS = [
  ["Deltoid", /^DELT\d$/],
  ["Rotator cuff", /^(SUPSP|INFSP|SUBSC|TMIN)$/],
  ["Chest and back", /^(PECM\d|LAT\d|TMAJ|CORB)$/],
  ["Elbow flexors", /^(BIClong|BICshort|BRA|BRD)$/],
  ["Elbow extensors", /^(TRIlong|TRIlat|TRImed|ANC)$/],
  ["Forearm rotation", /^(SUP|PT|PQ)$/],
  ["Wrist", /^(ECRL|ECRB|ECU|FCR|FCU|PL)$/],
  ["Finger flexors", /^(FDS\d|FDP\d|FPL)$/],
  ["Finger extensors", /^(EDC\d|EDM|EIP|EPL|EPB|APL)$/],
  ["Hand", /^(OP|RI\d|LU_RB\d|UI_UB\d)$/],
];

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const FINGERS = { 2: "index", 3: "middle", 4: "ring", 5: "little" };
const NAMES = {
  DELT1: "Deltoid, anterior", DELT2: "Deltoid, lateral", DELT3: "Deltoid, posterior",
  SUPSP: "Supraspinatus", INFSP: "Infraspinatus", SUBSC: "Subscapularis", TMIN: "Teres minor", TMAJ: "Teres major",
  PECM1: "Pectoralis major, clavicular", PECM2: "Pectoralis major, sternal", PECM3: "Pectoralis major, costal",
  LAT1: "Latissimus dorsi, upper", LAT2: "Latissimus dorsi, middle", LAT3: "Latissimus dorsi, lower", CORB: "Coracobrachialis",
  TRIlong: "Triceps, long head", TRIlat: "Triceps, lateral head", TRImed: "Triceps, medial head", ANC: "Anconeus",
  SUP: "Supinator", BIClong: "Biceps, long head", BICshort: "Biceps, short head", BRA: "Brachialis", BRD: "Brachioradialis",
  PT: "Pronator teres", PQ: "Pronator quadratus", ECRL: "Wrist extensor, radial long", ECRB: "Wrist extensor, radial short",
  ECU: "Wrist extensor, ulnar", FCR: "Wrist flexor, radial", FCU: "Wrist flexor, ulnar", PL: "Palmaris longus",
  EDM: "Little finger extensor", EIP: "Index finger extensor", EPL: "Thumb extensor, long", EPB: "Thumb extensor, short",
  FPL: "Thumb flexor, long", APL: "Thumb abductor, long", OP: "Thumb opponens",
};

// Readable name of a muscle ("FDS2" -> "Finger flexor, superficial (index)").
export function muscleLabel(name) {
  if (NAMES[name]) return NAMES[name];
  const m = name.match(/^(FDS|FDP|EDC|RI|LU_RB|UI_UB)(\d)$/);
  if (!m) return name;
  const kind = { FDS: "Finger flexor, superficial", FDP: "Finger flexor, deep", EDC: "Finger extensor", RI: "Radial interosseus", LU_RB: "Lumbrical", UI_UB: "Ulnar interosseus" }[m[1]];
  return `${kind} (${FINGERS[m[2]] || m[2]})`;
}

// Colour of an activation (0..1): from a pale rose to the scene's muscle red, as in the 3D view.
export const activityColor = (v) => `hsl(${Math.round(18 - 12 * v)} ${Math.round(35 + 60 * v)}% ${Math.round(80 - 32 * v)}%)`;

export class MuscleMeter {
  // names: the muscles in the order of the activation vectors.
  constructor(names) {
    this.names = names;
    this.groups = MUSCLE_GROUPS.map(([label, re]) => ({ label, idx: names.flatMap((n, k) => (re.test(n) ? [k] : [])) })).filter((g) => g.idx.length);
    const grouped = new Set(this.groups.flatMap((g) => g.idx));
    const other = names.flatMap((n, k) => (grouped.has(k) ? [] : [k]));
    if (other.length) this.groups.push({ label: "Other", idx: other });
    this.reset();
  }

  reset() {
    const n = this.names.length;
    this.current = new Float32Array(n);
    this.epSum = new Float64Array(n);
    this.epSteps = 0;
    this.sessionSum = new Float64Array(n);  // sum over episodes of each episode's mean
    this.episodes = 0;
    this.last = null;  // { mean, groups, fatigue } of the last finished episode
    this.order = this.names.map((_, k) => k);
  }

  push(act) {
    this.current.set(act);
    for (let k = 0; k < act.length; k++) this.epSum[k] += act[k];
    this.epSteps++;
  }

  // Mean activation per muscle: of the session, or of the running episode before the first one ends.
  get mean() {
    if (this.episodes) return this.sessionSum.map((s) => s / this.episodes);
    return this.epSum.map((s) => s / Math.max(1, this.epSteps));
  }

  // Close an episode; `fatigue` (fatigued share per muscle at its end) is optional.
  endEpisode(fatigue = null) {
    if (!this.epSteps) return;
    const mean = this.epSum.map((s) => s / this.epSteps);
    for (let k = 0; k < mean.length; k++) this.sessionSum[k] += mean[k];
    this.episodes++;
    const avg = (v, idx) => idx.reduce((s, k) => s + v[k], 0) / idx.length;
    const session = this.mean;
    this.last = {
      steps: this.epSteps,
      groups: this.groups.map((g) => ({
        label: g.label, n: g.idx.length, mean: avg(mean, g.idx), session: avg(session, g.idx),
        peak: Math.max(...g.idx.map((k) => mean[k])),
        fatigue: fatigue && fatigue.length === this.names.length ? avg(fatigue, g.idx) : null,
      })),
      total: mean.reduce((s, v) => s + v, 0) / mean.length,
    };
    this.epSum.fill(0);
    this.epSteps = 0;
    this.rank();
  }

  // The busiest muscles first (by mean activation), re-ranked only between episodes so bars stay put.
  rank() {
    const mean = this.mean;
    this.order = this.names.map((_, k) => k).sort((a, b) => mean[b] - mean[a]);
  }

  top(n) {
    if (!this.episodes && this.epSteps % 20 === 1) this.rank();  // before the first episode ends, every second or so
    return this.order.slice(0, n);
  }
}

// Live bars: the `n` busiest muscles, current activation as the bar, the session mean as a tick.
export function barsHtml(meter, n) {
  const mean = meter.mean;
  return meter.top(n).map((k) => {
    const v = meter.current[k], name = meter.names[k], label = esc(muscleLabel(name));
    return `<div class="mbar" title="${label} (${esc(name)})"><span class="mbar-name">${label}</span><span class="mbar-track"><span class="mbar-fill" style="width:${(100 * v).toFixed(1)}%;background:${activityColor(v)}"></span><span class="mbar-tick" style="left:${(100 * mean[k]).toFixed(1)}%"></span></span><span class="mbar-val mono">${v.toFixed(2)}</span></div>`;
  }).join("");
}

// Per-episode table of the muscle groups: last episode's mean (bar), the session's mean (tick), fatigue.
export function groupsHtml(meter, unit = "episode") {
  const last = meter.last;
  if (!last) return `<p class="muted mstats-empty">Group means appear after the first ${unit}.</p>`;
  const fatigue = last.groups.some((g) => g.fatigue != null);
  const rows = last.groups.map((g) => `<div class="mgroup"><span class="mgroup-name">${esc(g.label)} <span class="muted">${g.n}</span></span><span class="mbar-track"><span class="mbar-fill" style="width:${(100 * g.mean).toFixed(1)}%;background:${activityColor(g.mean)}"></span><span class="mbar-tick" style="left:${(100 * g.session).toFixed(1)}%"></span></span><span class="mbar-val mono">${g.mean.toFixed(2)}</span>${fatigue ? `<span class="mbar-val mono" title="fatigued share at the end">${g.fatigue.toFixed(2)}</span>` : ""}</div>`).join("");
  return `<div class="mgroups${fatigue ? " with-fatigue" : ""}"><div class="mgroup mgroup-head muted"><span>Group <span class="muted">muscles</span></span><span>last ${unit} · <span class="tick-key"></span> mean of ${meter.episodes}</span><span>mean</span>${fatigue ? "<span>MF</span>" : ""}</div>${rows}</div>`;
}
