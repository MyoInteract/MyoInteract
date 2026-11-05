// No DOM/WebGL emulation library: exercise actual Three.js resources and the shipped MuJoCo WASM.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import vm from "node:vm";
import { performance } from "node:perf_hooks";
import test from "node:test";
import * as THREE from "../myoInteract/studio/static/vendor/three/three.module.min.js";
import { StudioScene } from "../myoInteract/studio/static/scene.js";
import { TaskSim } from "../myoInteract/studio/static/sim.js";
import loadMujoco from "../myoInteract/studio/static/vendor/mujoco/mujoco.js";

const page = await fs.readFile(new URL("../myoInteract/studio/static/index.html", import.meta.url), "utf8");
// Execute the real inline callbacks in a controlled scope, with network and scheduling supplied by
// each test. These tests assert observable behavior, rather than the page's source text.
function callback(start, end, scope, expose) {
  const source = page.slice(page.indexOf(start), page.indexOf(end, page.indexOf(start)));
  return vm.runInNewContext(source + `\n;(${expose})`, scope);
}
function codecScope() {
  const scope = callback("const MODELS =", "try { const t = localStorage", { structuredClone }, "({S,DEFAULTS,MODELS,RESETS,OBS,EXTRA_REWARDS,COLORS})");
  scope.structuredClone = structuredClone;
  scope.esc = callback("const esc =", "const h =", {}, "esc");
  scope.rewardOverride = callback("function rewardOverride(", "function toPreset()", scope, "rewardOverride");
  Object.assign(scope, callback("function presetObject(", "// The Gradio task builder", scope,
    "({presetObject,presetNumber,presetText,normalizePreset,normalizeTarget,normalizeScreen,fromPreset,hexToRgba,rgbaToHex})"));
  scope.liveTargets = () => scope.S.targets.filter((t) => !t.disabled);
  scope.quatToEuler = null;
  scope.toPreset = callback("function toPreset()", "function presetObject(", scope, "toPreset");
  return scope;
}
const deferred = () => { let resolve; const promise = new Promise((r) => { resolve = r; }); return { promise, resolve }; };
const tick = () => new Promise((resolve) => setImmediate(resolve));

globalThis.document = {
  documentElement: {},
  createElement: () => ({ getContext: () => ({ beginPath() {}, arc() {}, fill() {}, fillText() {} }) }),
};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });

function scene() {
  const s = Object.create(StudioScene.prototype);
  Object.assign(s, {
    world: new THREE.Group(), targetGroup: new THREE.Group(), liveGroup: new THREE.Group(),
    bodies: [], bodyByName: new Map(), models: new Map(), origin: new THREE.Vector3(),
    grid: new THREE.Group(), targets: [], editable: false, _viewSet: true,
    gizmo: { detach() {} }, _emit() {},
  });
  s.world.add(s.targetGroup, s.liveGroup);
  return s;
}
const target = () => ({ type: "Sphere", x: [0.1, 0.2], y: [0, 0.1], z: [0, 0], size: [0.01, 0.03], color: "#aa3322" });

// Watch real disposal events, de-duplicating shared resources just as a renderer's ownership needs.
function watch(objects) {
  const resources = new Map();
  const add = (o) => { if (o && !resources.has(o)) { resources.set(o, 0); o.addEventListener("dispose", () => resources.set(o, resources.get(o) + 1)); } };
  for (const root of objects.filter(Boolean)) root.traverse((o) => {
    add(o.geometry);
    if (o.isInstancedMesh) add(o);
    for (const mat of (Array.isArray(o.material) ? o.material : o.material ? [o.material] : [])) {
      add(mat);
      for (const value of Object.values(mat)) if (value?.isTexture) add(value);
    }
  });
  return () => { assert.ok(resources.size); for (const count of resources.values()) assert.equal(count, 1); return resources.size; };
}

test("repeated target edits dispose every old geometry, material and label texture once", () => {
  const s = scene();
  let released = 0;
  for (let i = 0; i < 100; i++) {
    s.setTargets([target(), { ...target(), type: "Button" }], null);
    const check = watch(s.targets);
    s.setTargets([], null);
    released += check();
    assert.equal(s.targetGroup.children.length, 0);
  }
  console.log(`resource lifecycle: ${released} target resources released across 100 edits`);
});

test("model swaps dispose shared mesh geometry once and release reach and instancing resources", () => {
  const s = scene();
  const sc = {
    model: "a", index_type: "uint16", meshes: [{ vert: [0, 3], face: [36, 1] }],
    bodies: [{ name: "a", pos: [0, 0, 0], quat: [1, 0, 0, 0] }],
    geoms: [0, 1].map(() => ({ type: "mesh", mesh: 0, body: 0, pos: [0, 0, 0], quat: [1, 0, 0, 0], rgba: [1, 1, 1, 1] })),
    muscles: [{ radius: 0.001, points: [[0, 0, 0, 0], [0, 0.1, 0, 0]] }], origin: [0, 0, 0], reach: 0.5,
  };
  const buffer = new ArrayBuffer(42);
  new Float32Array(buffer, 0, 9).set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  new Uint16Array(buffer, 36, 3).set([0, 1, 2]);
  s._buildModel(sc, buffer);
  assert.equal(s.bodies[0].children[0].geometry, s.bodies[0].children[1].geometry);
  const check = watch([...s.bodies, s._reach, s._shoulder, s._muscles]);
  s._buildModel(sc, buffer);
  check();
});

test("screen and live-feed replacements release their resources", () => {
  const s = scene(), original = THREE.TextureLoader.prototype.load;
  THREE.TextureLoader.prototype.load = () => new THREE.Texture();
  try {
    s.setScreen({ image: "image", width: 100, height: 100, widthM: 0.1, pos: [0, 0, 0] });
    const check = watch([s.screen]);
    s.setScreen(null);
    check();
  } finally { THREE.TextureLoader.prototype.load = original; }
  s.setLive({ bodies: [], targets: [{ radius: 0.02, rgba: [1, 0, 0, 1] }] });
  const check = watch([...s.liveGroup.children]);
  s.setLive(null);
  check();
  assert.equal(s.liveGroup.children.length, 0);
});

test("static muscles avoid matrix recomputation/uploads; a new pose updates once", () => {
  const s = scene();
  s.bodies = [new THREE.Group()];
  s._buildMuscles([{ radius: 0.001, points: Array.from({ length: 129 }, (_, i) => [0, i * 0.001, 0, 0]) }]);
  const initial = s._muscles.instanceMatrix.version, scratch = s._muscleScratch;
  const start = performance.now();
  for (let i = 0; i < 10000; i++) s._updateMuscles();
  const elapsed = performance.now() - start;
  assert.equal(s._muscles.instanceMatrix.version, initial);
  s._musclesDirty = true;
  s._updateMuscles();
  s._updateMuscles();
  assert.equal(s._muscles.instanceMatrix.version, initial + 1);
  assert.equal(s._muscleScratch, scratch);
  console.log(`static 128-segment muscles: 10000 checks in ${elapsed.toFixed(2)} ms, 0 redundant matrix uploads`);
});

test("rebuilding muscles remaps an unchanged live feed's activation names", () => {
  const s = scene(), names = ["DELT1", "DELT2"];
  s.bodies = [new THREE.Group()];
  const muscle = (name) => ({ name, radius: 0.001, points: [[0, 0, 0, 0], [0, 0.1, 0, 0]] });
  s._buildMuscles([muscle("DELT1"), muscle("DELT2")]);
  s.setMuscleActivity([0, 1], names);
  const hot = new THREE.Color();
  s._muscles.getColorAt(1, hot);
  s._buildMuscles([muscle("DELT2"), muscle("DELT1")]);
  s.setMuscleActivity([0, 1], names);
  assert.deepEqual(s._activityMap, [1, 0]);
  const actual = new THREE.Color();
  s._muscles.getColorAt(0, actual);
  assert.deepEqual(actual, hot);
});

test("model requests are deduplicated and returning to the loaded model cancels a pending switch", async () => {
  const s = scene(), original = globalThis.fetch, pending = deferred();
  let calls = 0;
  s.modelName = "a"; s.modelKey = "\0a"; s.bodies = [{}];
  s._buildModel = (json) => { s.built = json.model; };
  globalThis.fetch = async (url) => {
    calls++;
    if (url.endsWith(".json")) { await pending.promise; return { ok: true, json: async () => ({ model: "b", buffer: "b.bin" }) }; }
    return { ok: true, arrayBuffer: async () => new ArrayBuffer(0) };
  };
  try {
    const first = s.loadModel("b"), second = s.loadModel("b");
    assert.equal(calls, 1);
    assert.equal(await s.loadModel("a"), true);
    pending.resolve();
    assert.equal(await first, false); assert.equal(await second, false);
    assert.equal(s.built, undefined);
    assert.equal(await s.loadModel("b"), true);
    assert.equal(s.built, "b");
  } finally { globalThis.fetch = original; }
});

test("same model name from different backends gets its own scene cache entry", async () => {
  const s = scene(), original = globalThis.fetch;
  const urls = [];
  s._buildModel = () => { s.bodies = [{}]; };
  globalThis.fetch = async (url) => {
    urls.push(url);
    return { ok: true, json: async () => ({ buffer: "mesh.bin" }), arrayBuffer: async () => new ArrayBuffer(0) };
  };
  try { await s.loadModel("arm", "http://a/"); await s.loadModel("arm", "http://b/"); }
  finally { globalThis.fetch = original; }
  assert.equal(urls.length, 4);
  assert.equal(s.models.size, 2);
});

function syncScope() {
  const scope = {
    S: { step: 1, bm: { bm_model: "arm" }, targets: [], sim: null, live: null, screen: null },
    SC: { loadModel: async () => true, setTargets() {}, setScreen() {}, setEditable() {} },
    sceneKeys: {}, sceneSyncVersion: 0, applyPanels() {},
    $: () => ({ classList: { toggle() {} } }), liveWanted: () => false, simWanted: () => false,
    replayWanted: () => false, sceneBase: () => "", renderSceneBar() {},
  };
  scope.screenKey = callback("const screenKey =", "// The scene centres", scope, "screenKey");
  scope.sync = callback("function syncScene()", "function renderSceneBar()", scope, "syncScene");
  return scope;
}

test("equal-length replacement uploads refresh the screen without rebuilding unchanged images", async () => {
  const scope = syncScope(), images = [];
  scope.SC.setScreen = (screen) => images.push(screen.image);
  scope.S.screen = { image: "AAAA", width: 100, height: 100, widthM: 0.1, pos: [0, 0, 0], quat: [0, 0, 0, 1] };
  scope.sync(); scope.sync(); scope.S.screen.image = "BBBB"; scope.sync();
  assert.deepEqual(images, ["AAAA", "BBBB"]);
  await tick();
});

test("an older scene callback cannot reattach a stale live feed", async () => {
  const scope = syncScope(), a = deferred(), b = deferred(), attached = [];
  scope.liveWanted = () => true; scope.S.liveSocket = {};
  scope.SC.setLive = (live) => attached.push(live);
  let calls = 0;
  scope.SC.loadModel = () => (++calls === 1 ? a.promise : b.promise);
  const old = { model: "a" }, latest = { model: "b" };
  scope.S.live = old; scope.sync(); scope.S.live = latest; scope.sync();
  b.resolve(true); await tick(); a.resolve(true); await tick();
  assert.deepEqual(attached, [latest]);
});

test("task simulation becomes ineligible after changing the target type or count", () => {
  const scope = { S: { sim: { name: "pointing", step: 1 }, step: 1, bm: { bm_model: "arm", fatigue_enabled: false }, targets: [target()], examples: [
    { name: "pointing", sim: true, sim_kind: "sphere", replay: { hello: { model: "arm" } }, summary: { targets: 1, fatigue: false } },
  ] } };
  scope.liveTargets = () => scope.S.targets.filter((t) => !t.disabled);
  const wanted = callback("const simWanted =", "function stopSim()", scope, "simWanted");
  assert.equal(wanted(), true);
  scope.S.targets[0].type = "Button"; assert.equal(wanted(), false);
  scope.S.targets[0].type = "Sphere"; scope.S.targets.push(target()); assert.equal(wanted(), false);
  scope.S.targets[1].disabled = true; assert.equal(wanted(), true);
  scope.S.bm.fatigue_enabled = true; assert.equal(wanted(), false);
});

test("a zero-dwell sphere still requires the fingertip to enter its radius", () => {
  const sim = Object.create(TaskSim.prototype);
  Object.assign(sim, {
    mj: { mj_step() {} }, model: { geom_size: new Float64Array([0.1, 0.1, 0.1]) },
    data: { site_xpos: new Float64Array([1, 0, 0]), xipos: new Float64Array(3), ctrl: new Float64Array(1) },
    task: { substeps: 1 }, targets: [{ bodyId: 0, geomId: 0, sensorAdr: -1, dwell_steps: 0 }],
    site: 0, current: 0, inside: 0, steps: 0, _excitation: new Float32Array(1),
    observe: () => [], act: () => new Float32Array([0.5]),
  });
  assert.equal(sim.step(), false);
  assert.equal(sim.current, 0);
  sim.data.site_xpos[0] = 0;
  assert.equal(sim.step(), true);
});

test("training polling retries a transient failure and clears the visible error after recovery", async () => {
  let calls = 0, renders = 0, sleeps = 0;
  const scope = {
    S: { backend: "http://a", step: 5, run: { id: "run", active: true } },
    api: async () => { if (++calls === 1) throw new Error("offline"); return { status: "completed", metrics: { points: [] } }; },
    sleep: async () => { sleeps++; }, render: () => { renders++; },
  };
  const follow = callback("async function followRun(", "function drawRun()", scope, "followRun");
  await follow("run");
  assert.equal(calls, 2); assert.equal(sleeps, 1); assert.equal(renders, 2);
  assert.equal(scope.S.run.active, false); assert.equal(scope.S.run.error, null);
});

test("an in-flight status response cannot overwrite a different selected run or backend", async () => {
  for (const change of ["run", "backend"]) {
    const pending = deferred();
    const scope = { S: { backend: "http://a", step: 5, run: { id: "run", active: true } }, api: () => pending.promise, render() { throw new Error("stale render"); } };
    const follow = callback("async function followRun(", "function drawRun()", scope, "followRun");
    const finished = follow("run");
    if (change === "run") scope.S.run = { id: "run", active: true }; else scope.S.backend = "http://b";
    pending.resolve({ status: "completed" });
    await finished;
    assert.equal(scope.S.run.status, undefined); assert.equal(scope.S.run.active, true);
  }
});

test("stopping keeps training active, disables duplicate controls and polls the final checkpoint", async () => {
  const pending = deferred();
  let deletes = 0;
  const scope = {
    S: { backend: "http://a", step: 5, run: { id: "run", active: true, status: "running", viewerStarted: true } },
    api: async () => { deletes++; await pending.promise; }, render() {},
  };
  const stop = callback("async function stopRun()", "function startRun()", scope, "stopRun");
  const finished = stop();
  await stop();
  assert.equal(deletes, 1); assert.equal(scope.S.run.active, true);
  scope.stopRun = stop; scope.esc = codecScope().esc; scope.gradioWarning = () => ""; scope.h = (html) => html; scope.drawRun = () => {};
  scope.head = (title) => title; scope.wandbOn = () => false; scope.RUN_METRICS = [];
  const chunks = [], ws = { append: (html) => chunks.push(html), querySelector: () => ({ addEventListener() {} }), querySelectorAll: () => [] };
  callback("function renderRun(", "async function stopRun()", scope, "renderRun")(ws);
  assert.match(chunks.join(""), /id="start-run" disabled/);
  assert.match(chunks.join(""), /id="stop-run" disabled>Stopping/);
  pending.resolve(); await finished;
  assert.equal(scope.S.run.status, "stopping"); assert.equal(scope.S.run.active, true);
  await stop(); assert.equal(deletes, 1);
  let polls = 0;
  scope.api = async () => (++polls === 1 ? { status: "running" } : { status: "stopped", checkpoints: ["model_final.pt"] });
  scope.sleep = async () => { assert.equal(scope.S.run.active, true); assert.equal(scope.S.run.status, "stopping"); };
  await callback("async function followRun(", "function drawRun()", scope, "followRun")("run");
  assert.equal(polls, 2); assert.equal(scope.S.run.active, false); assert.equal(scope.S.run.status, "stopped");
  assert.deepEqual(scope.S.run.checkpoints, ["model_final.pt"]);
});

test("a failed stop retains the active run and its error while polling continues", async () => {
  const scope = {
    S: { backend: "http://a", step: 5, run: { id: "run", active: true, status: "running", viewerStarted: true } },
    api: async () => { throw new Error("offline"); }, render() {}, sleep: async () => {},
  };
  await callback("async function stopRun()", "function startRun()", scope, "stopRun")();
  assert.equal(scope.S.run.active, true); assert.equal(scope.S.run.stopRequested, false);
  assert.match(scope.S.run.error, /Could not stop training: offline/);
  let polls = 0;
  scope.api = async () => (++polls === 1 ? { status: "running" } : { status: "completed" });
  await callback("async function followRun(", "function drawRun()", scope, "followRun")("run");
  assert.equal(polls, 2); assert.equal(scope.S.run.active, false);
  assert.match(scope.S.run.error, /Could not stop training: offline/);
});

test("a late stop response cannot replace a final polled status with stopping", async () => {
  const pending = deferred(), scope = { S: { backend: "http://a", run: { id: "run", active: true, status: "running" } }, api: () => pending.promise, render() {} };
  const finished = callback("async function stopRun()", "function startRun()", scope, "stopRun")();
  scope.S.run.active = false; scope.S.run.status = "stopped";
  pending.resolve(); await finished;
  assert.equal(scope.S.run.active, false); assert.equal(scope.S.run.status, "stopped");
});

test("switching backend clears old monitoring and ignores a stale training-start response", async () => {
  const pending = deferred();
  const scope = {
    S: { backend: "http://a", run: null }, api: () => pending.promise, render() {}, toPreset: () => ({}),
    followRun() { throw new Error("stale monitoring started"); },
    $: () => ({ classList: { add() {} } }), renderPresetList() {}, loadWandb() {},
  };
  const start = callback("async function startBackendRun()", "async function followRun(", scope, "startBackendRun");
  const finished = start();
  const connect = callback("function setConnected(", "const fmt =", scope, "setConnected");
  connect("http://b", "B");
  assert.equal(scope.S.run, null);
  scope.S.run = { id: "new", active: true };
  pending.resolve({ id: "old" });
  await finished;
  assert.equal(scope.S.run.id, "new");
});

test("simulation HTTP failures report the failing asset rather than parsing an error response", async () => {
  const originalFetch = globalThis.fetch, originalRuntime = TaskSim.runtime;
  const factory = "data:text/javascript,export default async function(){return {}}";
  try {
    for (const failed of ["sim.json", "model.xml", "policy.bin"]) {
      TaskSim.runtime = null;
      globalThis.fetch = async (url) => ({
        ok: !url.endsWith(failed), status: url.endsWith(failed) ? 404 : 200,
        json: async () => ({ model: "model.xml", policy: "policy.bin" }),
        text: async () => "", arrayBuffer: async () => new ArrayBuffer(0),
      });
      await assert.rejects(TaskSim.load("example", factory), /could not load simulation (metadata|model|policy): HTTP 404/);
    }
  } finally { globalThis.fetch = originalFetch; TaskSim.runtime = originalRuntime; }
});

test("a transient WASM initialization failure can be retried", async () => {
  const originalFetch = globalThis.fetch, originalRuntime = TaskSim.runtime;
  let attempts = 0;
  globalThis.__studioTestRuntime = async () => { attempts++; throw new Error("temporary WASM failure"); };
  const factory = "data:text/javascript,export default function(){return globalThis.__studioTestRuntime()}";
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ model: "m.xml", policy: "p.bin" }), text: async () => "", arrayBuffer: async () => new ArrayBuffer(0) });
  TaskSim.runtime = null;
  try {
    await assert.rejects(TaskSim.load("example", factory), /temporary WASM failure/);
    await assert.rejects(TaskSim.load("example", factory), /temporary WASM failure/);
    assert.equal(attempts, 2);
  } finally { globalThis.fetch = originalFetch; TaskSim.runtime = originalRuntime; delete globalThis.__studioTestRuntime; }
});

test("the complete inline application script parses", () => {
  const start = page.indexOf("<script>") + "<script>".length;
  const script = page.slice(start, page.indexOf("</script>", start));
  assert.doesNotThrow(() => new vm.Script(script));
});

test("all shipped presets pass typed import and more than ten targets survive a round trip", async () => {
  const scope = codecScope(), root = new URL("../myoInteract/studio/examples/", import.meta.url);
  let checked = 0;
  for (const name of await fs.readdir(root)) {
    const preset = JSON.parse(await fs.readFile(new URL(`${name}/preset.json`, root), "utf8"));
    scope.fromPreset(preset);
    assert.equal(scope.S.targets.length, preset.num_targets, name);
    assert.equal(scope.toPreset().num_targets, preset.num_targets, name);
    checked++;
  }
  assert.equal(checked, 5);
  const preset = JSON.parse(await fs.readFile(new URL("pointing/preset.json", root), "utf8"));
  preset.num_targets = 21;
  for (let i = 1; i <= 21; i++) {
    preset[`target_${i}`] = structuredClone(preset.target_1);
    preset[`target_${i}`].sphere.x_range = [i * 0.01, i * 0.01 + 0.1];
  }
  scope.fromPreset(preset);
  const roundTrip = scope.toPreset();
  assert.equal(roundTrip.num_targets, 21);
  assert.equal(roundTrip.target_21.sphere.x_range[0], 0.21);
  scope.fromPreset(roundTrip);
  assert.equal(scope.S.targets.length, 21);
  assert.equal(scope.S.targets[20].x[1], 0.31);
});

test("non-editor overrides survive edits, repeated imports and partial preset resets", () => {
  const scope = codecScope(), overrides = ["env.sim_dt=0.005", "env.muscle_config.noise_params.constantnoise_level=0.1", "++custom.values=[1,2]"];
  scope.fromPreset({ extra_overrides: [...overrides, "++env.task_config.weighted_reward_keys.action_rate=0.2"] });
  scope.S.reward.action_rate = 0.3;
  scope.S.task.max_duration = 7;
  const saved = scope.toPreset();
  assert.deepEqual(Array.from(saved.extra_overrides), [...overrides, "++env.task_config.weighted_reward_keys.action_rate=0.3"]);
  scope.fromPreset(saved);
  assert.deepEqual(Array.from(scope.toPreset().extra_overrides), [...overrides, "++env.task_config.weighted_reward_keys.action_rate=0.3"]);
  scope.S.reward.action_rate = 0;
  assert.deepEqual(Array.from(scope.toPreset().extra_overrides), [...overrides, "++env.task_config.weighted_reward_keys.action_rate=0"]);
  scope.fromPreset({});
  assert.deepEqual(Array.from(scope.toPreset().extra_overrides), []);
});

test("reward overrides retain their precedence relative to imported dictionary overrides", () => {
  const scope = codecScope(), dictionary = "++env.task_config.weighted_reward_keys={action_rate:0.7}";
  scope.fromPreset({ extra_overrides: ["++env.task_config.weighted_reward_keys.action_rate=0.2", dictionary, "env.sim_dt=0.005"] });
  scope.S.reward.action_rate = 0.3;
  assert.deepEqual(Array.from(scope.toPreset().extra_overrides), ["++env.task_config.weighted_reward_keys.action_rate=0.3", dictionary, "env.sim_dt=0.005"]);
});

test("zero and negative scalar rewards still override earlier dictionary settings", () => {
  const scope = codecScope(), dictionary = "++env.task_config.weighted_reward_keys={action_rate:0.7}";
  for (const weight of [0, -0.2, 0.3]) {
    const scalar = `++env.task_config.weighted_reward_keys.action_rate=${weight}`;
    scope.fromPreset({ extra_overrides: [dictionary, scalar] });
    assert.deepEqual(Array.from(scope.toPreset().extra_overrides), [dictionary, scalar]);
    scope.fromPreset(scope.toPreset());
    assert.equal(scope.S.reward.action_rate, weight);
    assert.deepEqual(Array.from(scope.toPreset().extra_overrides), [dictionary, scalar]);
  }
  scope.S.reward.action_rate = 0;
  assert.deepEqual(Array.from(scope.toPreset().extra_overrides), [dictionary, "++env.task_config.weighted_reward_keys.action_rate=0"]);
});

test("draft saves coalesce drag events and flush the latest state once before page exit", () => {
  const timers = new Map(), writes = [];
  let next = 0, serializations = 0;
  const scope = {
    S: { preset: "first", screen: { image: "data:image/png;base64," + "A".repeat(2_000_000) }, targets: [] },
    setupPayload: () => { serializations++; return JSON.stringify({ name: scope.S.preset }); },
    localStorage: { setItem: (key, value) => writes.push([key, value]) },
    setTimeout: (fn) => { timers.set(++next, fn); return next; }, clearTimeout: (id) => timers.delete(id),
  };
  const { saveDraft, flushDraft } = callback("let draftSaveTimer =", "function restoreDraft()", scope, "({saveDraft,flushDraft})");
  const events = {}, document = { hidden: false, addEventListener: (name, fn) => { events[name] = fn; } };
  callback('window.addEventListener("pagehide"', "if (!importSetupFromHash())", {
    flushDraft, document, window: { addEventListener: (name, fn) => { events[name] = fn; } },
  }, "null");
  for (let i = 0; i < 100; i++) { scope.S.preset = `edit ${i}`; saveDraft(); }
  assert.equal(serializations, 0); assert.equal(writes.length, 0); assert.equal(timers.size, 1);
  [...timers.values()][0]();
  assert.equal(serializations, 1); assert.equal(writes.length, 1); assert.equal(timers.size, 0);
  assert.equal(JSON.parse(writes[0][1]).name, "edit 99");
  scope.S.preset = "last edit"; saveDraft(); events.visibilitychange();
  assert.equal(writes.length, 1);  // returning to a visible tab does not flush an edit early
  document.hidden = true; events.visibilitychange(); events.pagehide();
  assert.equal(writes.length, 2); assert.equal(timers.size, 0);
  assert.equal(JSON.parse(writes[1][1]).name, "last edit");
  scope.S.preset = "exit edit"; saveDraft(); events.pagehide();
  assert.equal(writes.length, 3); assert.equal(JSON.parse(writes[2][1]).name, "exit edit");
  console.log("draft persistence: 100 drag events produce 1 image serialization/storage write");
});

test("malformed and executable import fields fail before any editor state changes", () => {
  const scope = codecScope(), malicious = '\"><img src=x onerror="globalThis.pwned=1">';
  scope.S.targets[0].x = [0.7, 0.8];
  const original = JSON.stringify(scope.S);
  const cases = [
    [null, /preset must be an object/], [[], /preset must be an object/],
    [{ bm: { ctrl_dt: malicious } }, /bm.ctrl_dt/], [{ bm: { ctrl_dt: Infinity } }, /bm.ctrl_dt/],
    [{ bm: { fatigue_enabled: "false" } }, /bm.fatigue_enabled/], [{ bm: { reset_type: malicious } }, /bm.reset_type/],
    [{ rl: { "Number of Total Training Steps": null } }, /rl.Number of Total Training Steps/],
    [{ rl: { "Target Initial Seed": -1 } }, /rl.Target Initial Seed/],
    [{ num_targets: 1.5 }, /num_targets/], [{ num_targets: null }, /num_targets/],
    [{ num_targets: 2, target_1: { radio: "Sphere", sphere: { x_range: [0, 1], y_range: [0, 0], z_range: [0, 0] } } }, /target_2/],
    [{ target_1: { radio: "Sphere", sphere: {} } }, /target_1.x/],
    [{ target_1: { radio: "Sphere", sphere: { x_range: [1, 0], y_range: [0, 0], z_range: [0, 0] } } }, /target_1.x/],
    [{ target_1: { radio: "Sphere", sphere: { x_range: [0, 1], y_range: [0, 0], z_range: [0, 0], rgb: malicious } } }, /target_1.color/],
    [{ obs: { qpos: ["true"] } }, /obs.qpos/], [{ reward: { distance: malicious } }, /reward.distance/],
    [{ extra_overrides: [null] }, /extra_overrides/],
  ];
  for (const [preset, error] of cases) {
    assert.throws(() => scope.fromPreset(preset), error);
    assert.equal(JSON.stringify(scope.S), original);
  }
  scope.fromPreset(JSON.parse('{"bm":{"__proto__":{"polluted":true}},"__proto__":{"polluted":true}}'));
  assert.equal(scope.S.bm.polluted, undefined);
  assert.equal({}.polluted, undefined);
  assert.equal(scope.S.targets[0].x[0], scope.DEFAULTS.targets[0].x[0]);
  assert.notEqual(scope.S.targets, scope.DEFAULTS.targets);
});

test("strict colors and draft vectors reject invalid values and normalize valid forms", () => {
  const scope = codecScope();
  assert.equal(scope.rgbaToHex("#Aa00Ff"), "#aa00ff");
  assert.equal(scope.rgbaToHex("rgba(204.0,25.5,25.5,1)"), "#cc1a1a");
  for (const value of ["red", "rgba(256,0,0,1)", "rgba(0,0,0,2)", "rgb(0,0,0,1)", "rgba(0,0,0)", 123]) assert.equal(scope.rgbaToHex(value), null);
  assert.throws(() => scope.normalizeTarget({ ...target(), quat: [0, 0, 0, 0] }, "draft", 0), /quaternion/);
  assert.throws(() => scope.normalizeTarget({ ...target(), disabled: "false" }, "draft", 0), /disabled/);
  assert.throws(() => scope.normalizeTarget({ ...target(), x: [0, NaN] }, "draft", 0), /draft.x/);
  assert.throws(() => scope.normalizeScreen({ image: "javascript:alert(1)" }), /image data URL/);
});

test("hash imports reject bad metadata and executable presets transactionally, with field errors", () => {
  for (const data of [
    { preset: { bm: { ctrl_dt: '\"><img src=x onerror="alert(1)">' } }, step: 0 },
    { preset: {}, step: 7 }, { preset: {}, step: -1 }, { preset: {}, name: {} },
  ]) {
    const scope = codecScope(), original = JSON.stringify(scope.DEFAULTS);
    Object.assign(scope, {
      location: { hash: "#setup=" + Buffer.from(JSON.stringify(data)).toString("base64url"), pathname: "/", search: "" },
      unb64url: (v) => Buffer.from(v, "base64url").toString("utf8"), STEPS: Array(7), history: { replaceState() {} },
    });
    const read = callback("function importSetupFromHash()", "// The screen (its image included)", scope, "importSetupFromHash");
    assert.equal(read(), true);
    assert.match(scope.S.importError, /could not be read: (bm.ctrl_dt|setup.step|setup.name)/);
    for (const key of Object.keys(scope.DEFAULTS)) assert.equal(JSON.stringify(scope.S[key]), JSON.stringify(JSON.parse(original)[key]));
  }
  const scope = codecScope();
  Object.assign(scope, { location: { hash: "#setup=" + Buffer.from(JSON.stringify({ preset: {}, step: 6, name: "my setup" })).toString("base64url"), pathname: "/", search: "" }, unb64url: (v) => Buffer.from(v, "base64url").toString(), STEPS: Array(7), history: { replaceState() {} } });
  callback("function importSetupFromHash()", "// The screen (its image included)", scope, "importSetupFromHash")();
  assert.equal(scope.S.step, 6); assert.equal(scope.S.preset, "my setup"); assert.equal(scope.S.importError, null);
});

test("restored drafts validate disabled targets and screen data before replacing state", () => {
  const scope = codecScope(), preset = scope.toPreset(), original = JSON.stringify(scope.S.targets);
  let draft = { preset, targets: [{ ...target(), color: '\"><img src=x onerror="alert(1)">' }] };
  scope.localStorage = { getItem: () => JSON.stringify(draft) };
  const restore = callback("function restoreDraft()", "function openConnect()", scope, "restoreDraft");
  restore();
  assert.match(scope.S.importError, /draft.targets\[0\].color/);
  assert.equal(JSON.stringify(scope.S.targets), original);
  draft = { preset, targets: [target(), { ...target(), disabled: true, screenEl: 0 }] };
  restore();
  assert.equal(scope.S.targets.length, 2); assert.equal(scope.S.targets[1].disabled, true);
  assert.equal(scope.S.targets[1].screenEl, undefined);
  draft.screen = { image: "javascript:alert(1)" };
  const valid = JSON.stringify(scope.S.targets);
  restore();
  assert.match(scope.S.importError, /screen.image/);
  assert.equal(JSON.stringify(scope.S.targets), valid);
});

test("Figma secrets, URLs and frame names are encoded as text/attribute values", () => {
  const scope = codecScope(), attack = '\"><img src=x onerror="globalThis.pwned=1">';
  scope.S.figma = { url: attack };
  scope.S.screen = { name: attack, widthM: 0.1, width: 100, height: 100, elements: [] };
  scope.figmaToken = () => attack; scope.figmaRemembered = () => false;
  const html = callback("function screenHtml()", "const figmaRemembered =", scope, "screenHtml")();
  assert.ok(html.includes('value="&quot;&gt;&lt;img'));
  assert.ok(!html.includes("<img"));
  assert.ok(html.includes("&lt;img src=x onerror=&quot;globalThis.pwned=1&quot;&gt;"));
  const nodes = new Map();
  scope.$ = (id) => { if (!nodes.has(id)) nodes.set(id, {}); return nodes.get(id); };
  scope.S.presets = [{ name: attack }];
  callback("function renderPresetList()", '$("preset-load").addEventListener', scope, "renderPresetList")();
  assert.ok(!nodes.get("preset-select").innerHTML.includes("<img"));
});

test("parallel-environment selects stay numeric and old numeric string presets normalize", () => {
  const scope = codecScope();
  let input;
  const el = { type: "select-one", value: "4096", dataset: { bind: "rl.envs" }, addEventListener: (_, handler) => { input = handler; } };
  scope.renderSummary = () => {};
  const bind = callback("function bindInputs(", "/* ---------------------------------------------------------------- 1 body", scope, "bindInputs");
  bind({ querySelectorAll: () => [el] }); input();
  assert.equal(scope.S.rl.envs, 4096);
  scope.fromPreset({ rl: { "Number of Parallel Environments": "1024" } });
  assert.equal(scope.S.rl.envs, 1024);
});

test("actual WASM policy rollouts match all five version-matched browser reference trajectories", async () => {
  const mujoco = await loadMujoco(), examples = new URL("../myoInteract/studio/examples/", import.meta.url);
  assert.equal(mujoco.mj_versionString(), "3.15.0", "vendored WASM upgrades require native/reference parity checks");
  const profile = JSON.parse(await fs.readFile(new URL("./fixtures/studio/browser-3.15.0.json", import.meta.url), "utf8"));
  assert.equal(profile.native_mujoco_version, "3.15.0"); assert.equal(profile.wasm_mujoco_version, "3.15.0");
  const references = new Map(profile.examples.map((e) => [e.name, e]));
  assert.equal(references.size, 5);
  let checked = 0, maxError = 0;
  for (const name of await fs.readdir(examples)) {
    const base = new URL(`${name}/`, examples);
    let info;
    try { info = JSON.parse(await fs.readFile(new URL("sim.json", base), "utf8")); }
    catch (e) { if (e.code === "ENOENT") continue; throw e; }
    const bytes = await fs.readFile(new URL(info.policy, base));
    const xml = await fs.readFile(new URL(info.model, base), "utf8"), expected = references.get(name);
    assert.ok(expected, `${name}: browser engine profile`);
    const sha256 = (value) => createHash("sha256").update(value).digest("hex");
    assert.equal(sha256(xml.replace(/\r\n/g, "\n")), expected.provenance.model_sha256, `${name}: model hash`);
    assert.equal(sha256(bytes), expected.provenance.policy_sha256, `${name}: policy hash`);
    const sim = new TaskSim(mujoco, info, xml, new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
    try {
      const r = expected.reference;
      assert.equal(r.mujoco_version, profile.native_mujoco_version);
      sim.noisy = false;
      sim.reset({ start: r.start, positions: r.positions, sizes: r.sizes, fatigueState: r.fatigue_state });
      const outputs = [...sim._layerOutputs], normalized = sim._normalized, excitation = sim._excitation;
      const frame = new Float32Array(7 * sim.model.nbody + 3 * sim.targets.length);
      for (const qpos of r.qpos) {
        sim.step();
        const err = Math.max(...qpos.map((v, i) => Math.abs(sim.data.qpos[i] - v)));
        // Same-state actors and fixed-control physics are independently checked at 1e-6.
        // Native/WASM physics rounding can cross float32 controller boundaries: a single
        // one-ULP excitation pulse at step 2 causes ~1e-5 native closed-loop qpos drift.
        assert.ok(err <= 2e-5, `${name}: closed-loop qpos error ${err}`);
        maxError = Math.max(maxError, err);
        assert.equal(sim.frame(frame), frame);
      }
      assert.equal(sim.current, r.current, name);
      assert.ok(Math.max(...r.MF.map((v, i) => Math.abs(sim.MF[i] - v))) < 1e-6, name);
      assert.equal(sim._normalized, normalized); assert.equal(sim._excitation, excitation);
      sim._layerOutputs.forEach((out, i) => assert.equal(out, outputs[i]));
      checked++;
    } finally { sim.dispose(); }
  }
  assert.equal(checked, 5);
  console.log(`WASM/native trajectory check: ${checked} examples, max qpos error ${maxError.toExponential(3)}`);
});
