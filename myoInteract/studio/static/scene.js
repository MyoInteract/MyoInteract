// Studio's 3D scene: the body model (geometry exported from MuJoCo, see studio/scene.py), the
// target regions with drag handles, an optional screen (e.g. a Figma frame) and the live feed of a
// policy. Everything inside `world` is in MuJoCo's frame (x forward, y left, z up, metres); the
// page talks to the scene through methods and the events "select", "change", "add",
// "screen-change" and "status".
import * as THREE from "./vendor/three/three.module.min.js";
import { OrbitControls } from "./vendor/three/addons/OrbitControls.js";
import { TransformControls } from "./vendor/three/addons/TransformControls.js";

// Orientation of a button: its own (three order x, y, z, w) or, from the tilt angle alone, MuJoCo's
// euler [0, -angle, 0] (90 degrees: upright, facing the user).
export const buttonQuat = (t) => (t.euler ? new THREE.Quaternion().setFromEuler(new THREE.Euler(...t.euler, "XYZ")) : t.quat ? new THREE.Quaternion().fromArray(t.quat) : new THREE.Quaternion().setFromEuler(new THREE.Euler(0, (-(t.angle ?? 90) * Math.PI) / 180, 0, "XYZ")));
// MuJoCo euler (x, y, z in radians, its default sequence) of a three quaternion.
export const quatToEuler = (q) => { const e = new THREE.Euler().setFromQuaternion(new THREE.Quaternion().fromArray(q), "XYZ"); return [e.x, e.y, e.z].map((v) => Math.round(v * 1e5) / 1e5); };
const MJ_QUAT = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]); // MuJoCo w, x, y, z -> three
const STEP = 0.005; // 5 mm grid of the target ranges
const MUSCLE_IDLE = "#c9b3ae", MUSCLE_HOT = "#ff2d14"; // muscle colour at activation 0 and 1 (live view)
const snap = (v) => Math.round(v / STEP) * STEP;
const BONE = "#e4dccb"; // the models' grey bones, as warm ivory
const MIN_FPS = 20; // below this the scene drops its shadows (see _watchFrameRate)
// "?quality=high" keeps the shadows on every device, "low" never draws them.
const QUALITY = new URLSearchParams(globalThis.location?.search).get("quality") || "auto";
const css = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

// Removing an Object3D does not release its GPU resources. A model can share a geometry between
// several bodies, so collect the whole replacement before disposing each resource once.
function disposeObjects(objects) {
  const geometries = new Set(), materials = new Set(), textures = new Set();
  for (const root of objects) {
    if (!root) continue;
    root.removeFromParent();
    root.traverse((o) => {
      if (o.geometry) geometries.add(o.geometry);
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      for (const mat of mats) {
        materials.add(mat);
        for (const value of Object.values(mat)) if (value?.isTexture) textures.add(value);
      }
      if (o.isInstancedMesh) o.dispose();
    });
  }
  textures.forEach((t) => t.dispose());
  materials.forEach((m) => m.dispose());
  geometries.forEach((g) => g.dispose());
}

export class StudioScene extends EventTarget {
  constructor(container) {
    super();
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;  // keeps the targets' and muscles' colours
    this.renderer.shadowMap.enabled = QUALITY !== "low";
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2; // MuJoCo z-up inside, three y-up outside
    this.scene.add(this.world);
    this._buildLights();
    this.grid = fadingGrid();
    this.scene.add(this.grid);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
    this.gizmo.setSpace("local");
    this.gizmo.setSize(0.8);
    this.gizmo.addEventListener("dragging-changed", (e) => { this.controls.enabled = !e.value; });
    this.gizmo.addEventListener("objectChange", () => this._onGizmo());
    // A resize is relative to the region's size when the drag started.
    this.gizmo.addEventListener("mouseDown", () => { const o = this.gizmo.object; if (o && o.userData.ext) o.userData.ext0 = [...o.userData.ext]; });
    this.gizmo.addEventListener("mouseUp", () => {
      const o = this.gizmo.object;
      if (!o || !o.userData.ext0) return;
      delete o.userData.ext0;
      o.scale.set(1, 1, 1);
      o.userData.marker.scale.set(1, 1, 1);
      o.userData.region.scale.set(...o.userData.ext.map((x) => Math.max(x, 0.002)));
    });
    this.scene.add(this.gizmo.getHelper());

    this.models = new Map(); // source and name -> { json, buffer }
    this.bodies = []; // body groups of the shown model, by body index
    this.bodyByName = new Map();
    this.origin = new THREE.Vector3(); // shoulder (target origin), MuJoCo frame
    this.targetGroup = new THREE.Group();
    this.liveGroup = new THREE.Group();
    this.world.add(this.targetGroup, this.liveGroup);
    this.targets = [];
    this.selected = null;
    this.editable = true;
    this.live = null;
    this.screen = null;
    this.hover = null; // pointer on the plane through the shoulder that faces the camera
    this._raycaster = new THREE.Raycaster();
    this._bindPointer();
    this._theme();
    new ResizeObserver(() => this._resize()).observe(container);
    this._resize();
    const loop = (now) => { requestAnimationFrame(loop); if (!document.hidden) { this.controls.update(); this._updateMuscles(); this.renderer.render(this.scene, this.camera); this._watchFrameRate(now); } };
    requestAnimationFrame(loop);
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", () => this._theme());
    new MutationObserver(() => this._theme()).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  _emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail })); }

  // Soft studio light: a blurred environment for ambient light and reflections, a key light with
  // shadows on an otherwise invisible floor, and a cool fill and a rim from the other sides.
  _buildLights() {
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(studioEnvironment(), 0.04).texture;
    pmrem.dispose();
    this.scene.environmentIntensity = 0.55;
    const key = new THREE.DirectionalLight(0xfff4e6, 1.85);
    key.position.set(0.7, 3.8, 0.9);
    key.target.position.set(0.1, 0.8, 0);
    key.castShadow = QUALITY !== "low";
    key.shadow.mapSize.setScalar(matchMedia("(pointer: coarse)").matches ? 1024 : 2048);
    Object.assign(key.shadow.camera, { left: -1.2, right: 1.2, top: 1.2, bottom: -1.2, near: 0.5, far: 7 });
    key.shadow.radius = 4;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.004;
    const fill = new THREE.DirectionalLight(0xdce6ff, 0.45);
    fill.position.set(-2, 1.4, 2.2);
    const rim = new THREE.DirectionalLight(0xffffff, 0.7);
    rim.position.set(-0.8, 2.4, -2.6);
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), new THREE.ShadowMaterial({ opacity: 0.2 }));
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.receiveShadow = true;
    this.floor.raycast = () => {};  // never in the way of picking
    this.floor.visible = key.castShadow;
    this.key = key;
    this.scene.add(key, key.target, fill, rim, this.floor);
  }

  // A device too slow for the shadows drops them and renders at its base resolution, once.
  _watchFrameRate(now) {
    if (QUALITY !== "auto" || !this.key.castShadow) return;
    const f = (this._frames ||= { n: 0, t0: now });
    if (now - f.last > 500) { f.n = 0; f.t0 = now; }  // the tab was hidden or the page busy loading
    f.last = now;
    if (++f.n < 90) return;
    if (f.n / ((now - f.t0) / 1000) < MIN_FPS) {
      this.key.castShadow = false;
      this.renderer.shadowMap.enabled = false;
      this.floor.visible = false;
      this.renderer.setPixelRatio(1);
      this._resize();
    }
    f.n = 0; f.t0 = now;
  }

  _theme() {
    const bg = css("--scene-bg", "#e6e5df");
    this.scene.background?.dispose?.();
    this.scene.background = backdrop(bg);
    this.grid.material.color = new THREE.Color(css("--scene-grid", "#c9ceda"));
    this.floor.material.opacity = new THREE.Color(bg).getHSL({}).l < 0.3 ? 0.45 : 0.2;
    if (this._reach) this._reach.material.color = new THREE.Color(css("--scene-reach", "#8a90a6"));
  }

  _resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Centre the view in the free space between the overlaid panels (picking follows the offset).
    if (this.focus) this.camera.setViewOffset(w, h, w / 2 - (this.focus[0] + this.focus[1]) / 2, 0, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
  }

  // Free horizontal span of the canvas (CSS pixels from its left edge), or null for the whole width.
  setFocus(span) {
    const same = span && this.focus && Math.abs(span[0] - this.focus[0]) < 1 && Math.abs(span[1] - this.focus[1]) < 1;
    if (same || (!span && !this.focus)) return;
    this.focus = span;
    this._resize();
  }

  // ------------------------------------------------------------------ body model
  async loadModel(name, base = "") {
    const key = `${base}\0${name}`;
    if (this.modelKey === key && this.bodies.length) {
      this._modelRequest = null; // returning to the loaded model cancels a pending switch
      return true;
    }
    let request = this._modelRequest;
    if (!request || request.key !== key) {
      request = { key };
      this._modelRequest = request;
      request.promise = (async () => {
        let entry = this.models.get(key);
        if (!entry) {
          const json = await fetch(`${base}scenes/${name}.json`).then((r) => { if (!r.ok) throw new Error(`no scene for ${name}`); return r.json(); });
          // The backend sends binary meshes; static copies ship them as base64 text.
          const res = await fetch(`${base}scenes/${json.buffer}`);
          if (!res.ok) throw new Error(`no meshes for ${name}`);
          const buffer = json.buffer.endsWith(".b64.txt") ? Uint8Array.from(atob(await res.text()), (c) => c.charCodeAt(0)).buffer : await res.arrayBuffer();
          entry = { json, buffer };
          this.models.set(key, entry);
        }
        if (this._modelRequest !== request) return false;
        this._buildModel(entry.json, entry.buffer);
        this.modelName = name;
        this.modelKey = key;
        return true;
      })().finally(() => { if (this._modelRequest === request) this._modelRequest = null; });
    }
    return request.promise;
  }

  _buildModel(sc, buffer) {
    disposeObjects(this.bodies);
    this.bodies = [];
    this.bodyByName.clear();
    this.sceneInfo = sc;
    const Index = sc.index_type === "uint32" ? Uint32Array : Uint16Array;
    const meshGeoms = sc.meshes.map((m) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buffer, m.vert[0], m.vert[1] * 3), 3));
      g.setIndex(new THREE.BufferAttribute(new Index(buffer, m.face[0], m.face[1] * 3), 1));
      g.computeVertexNormals();
      return g;
    });
    sc.bodies.forEach((b) => {
      const grp = new THREE.Group();
      grp.name = b.name;
      grp.userData.rest = { pos: b.pos, quat: b.quat };
      grp.position.fromArray(b.pos);
      grp.quaternion.copy(MJ_QUAT(b.quat));
      this.world.add(grp);
      this.bodies.push(grp);
      this.bodyByName.set(b.name, grp);
    });
    for (const g of sc.geoms) {
      const geo = g.type === "mesh" ? meshGeoms[g.mesh] : primitive(g);
      if (!geo) continue;
      if (g.body === 0 && isRoomSized(geo)) continue;  // the model's room meshes: Studio draws its own floor
      const grey = Math.abs(g.rgba[0] - g.rgba[1]) < 0.02 && Math.abs(g.rgba[1] - g.rgba[2]) < 0.02;
      const mat = new THREE.MeshStandardMaterial({
        color: grey && g.type === "mesh" ? new THREE.Color(BONE) : new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]), roughness: 0.58, metalness: 0.0,
        transparent: g.rgba[3] < 1, opacity: g.rgba[3],
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = mesh.receiveShadow = g.rgba[3] >= 1;
      mesh.position.fromArray(g.pos);
      mesh.quaternion.copy(MJ_QUAT(g.quat));
      if (g.type === "capsule" || g.type === "cylinder") mesh.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2));
      mesh.userData.body = true;
      mesh.userData.volume = !!g.volume;
      if (g.volume) mesh.visible = this.showSkin !== false;
      this.bodies[g.body].add(mesh);
    }
    this._buildMuscles(sc.muscles || []);
    this.origin.fromArray(sc.origin);
    this.grid.position.set(0, 0, 0);
    this._buildReach(sc.reach);
    this.targetGroup.position.copy(this.origin); // targets are relative to the shoulder
    this.liveGroup.position.set(0, 0, 0);
    if (!this._viewSet) this.setView("perspective");
    this._emit("model", { name: sc.model });
  }

  _buildReach(radius) {
    disposeObjects([this._reach, this._shoulder]);
    const source = new THREE.IcosahedronGeometry(radius, 3);
    const geo = new THREE.EdgesGeometry(source);
    source.dispose();
    this._reach = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: css("--scene-reach", "#8a90a6"), transparent: true, opacity: 0.14 }));
    this._reach.position.copy(this.origin);
    this._reach.visible = this.showReach !== false;
    this._shoulder = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12), new THREE.MeshBasicMaterial({ color: css("--accent", "#d9432a") }));
    this._shoulder.position.copy(this.origin);
    this.world.add(this._reach, this._shoulder);
  }

  setReachVisible(on) { this.showReach = on; if (this._reach) this._reach.visible = on; }

  // ------------------------------------------------------------------ muscles
  // Every muscle is drawn site to site as thin cylinders; the sites ride on their bodies, so the paths
  // follow the pose (also in the live view). One instanced mesh holds all segments.
  _buildMuscles(muscles) {
    if (this._muscles) { disposeObjects([this._muscles]); this._muscles = null; }
    this._segments = [];
    this._muscleNames = muscles.map((m) => (m.name || "").split("/").pop());  // unnamed muscles keep their rest colour
    this._activityMap = null;
    this._activityNames = null;
    muscles.forEach((m, j) => {
      for (let k = 1; k < m.points.length; k++) this._segments.push([m.points[k - 1], m.points[k], m.radius, j]);
    });
    if (!this._segments.length) return;
    const geo = new THREE.CylinderGeometry(1, 1, 1, 12, 1, true);  // unit cylinder along y, scaled per segment
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.42 });  // the colour is per segment
    this._muscles = new THREE.InstancedMesh(geo, mat, this._segments.length);
    this._muscles.castShadow = this._muscles.receiveShadow = true;
    this._muscles.frustumCulled = false;
    this._muscles.visible = this.showMuscles !== false;
    this.world.add(this._muscles);
    this._musclesDirty = true;
    this.setMuscleActivity(null);
    this._updateMuscles(true);
  }

  // Muscles coloured by activation (0..1, in the order of `names`), or all in their rest colour (null).
  setMuscleActivity(values, names = null) {
    if (!this._muscles) return;
    const rest = new THREE.Color(css("--scene-muscle", "#c4473d")), idle = new THREE.Color(MUSCLE_IDLE), hot = new THREE.Color(MUSCLE_HOT), c = new THREE.Color();
    if (values && names !== this._activityNames) {  // scene muscle -> index into `values`
      this._activityNames = names;
      const at = new Map(names.map((n, k) => [n, k]));
      this._activityMap = this._muscleNames.map((n) => (at.has(n) ? at.get(n) : -1));
    }
    this._segments.forEach((s, i) => {
      const k = values && this._activityMap ? this._activityMap[s[3]] : -1;
      this._muscles.setColorAt(i, k < 0 ? rest : c.copy(idle).lerp(hot, Math.min(1, Math.max(0, values[k]))));
    });
    this._muscles.instanceColor.needsUpdate = true;
  }

  setMusclesVisible(on) { this.showMuscles = on; if (this._muscles) this._muscles.visible = on; }

  // The skin: the body volumes around the bones (primitive geoms, e.g. MoBL-ARMS' capsules).
  setSkinVisible(on) {
    this.showSkin = on;
    for (const b of this.bodies) b.traverse((o) => { if (o.userData.volume) o.visible = on; });
  }
  get hasSkin() { return (this.sceneInfo?.geoms || []).some((g) => g.volume); }

  _updateMuscles(force = false) {
    if (!this._muscles || (!force && (!this._muscles.visible || !this._musclesDirty))) return;
    const { a, b, dir, up, q, m, scale } = this._muscleScratch || (this._muscleScratch = {
      a: new THREE.Vector3(), b: new THREE.Vector3(), dir: new THREE.Vector3(), up: new THREE.Vector3(0, 1, 0),
      q: new THREE.Quaternion(), m: new THREE.Matrix4(), scale: new THREE.Vector3(),
    });
    for (let i = 0; i < this._segments.length; i++) {
      const [p0, p1, r] = this._segments[i], body0 = this.bodies[p0[0]], body1 = this.bodies[p1[0]];
      a.set(p0[1], p0[2], p0[3]).applyQuaternion(body0.quaternion).add(body0.position);
      b.set(p1[1], p1[2], p1[3]).applyQuaternion(body1.quaternion).add(body1.position);
      dir.subVectors(b, a);
      const len = dir.length();
      q.setFromUnitVectors(up, len > 1e-9 ? dir.divideScalar(len) : up);
      m.compose(a.add(b).multiplyScalar(0.5), q, scale.set(r, len, r));
      this._muscles.setMatrixAt(i, m);
    }
    this._muscles.instanceMatrix.needsUpdate = true;
    this._musclesDirty = false;
  }

  // ------------------------------------------------------------------ cameras
  _toThree(v) { this.world.updateMatrixWorld(); return this.world.localToWorld(v.clone()); }

  setView(kind) {
    this._viewSet = true;
    // From the user's own eyes the skull is in the way: it is drawn in every other view only.
    const head = this.bodyByName.get("head");
    if (head) head.traverse((o) => { if (o.isMesh && o.userData.body) o.visible = kind !== "user"; });
    const o = this.origin.clone();
    let eye, look = o.clone(), up = new THREE.Vector3(0, 0, 1), fov = 40;
    if (kind === "top") { eye = o.clone().add(new THREE.Vector3(0.3, 0, 2.4)); look = o.clone().add(new THREE.Vector3(0.3, 0, 0)); up = new THREE.Vector3(1, 0, 0); }
    else if (kind === "side") { eye = o.clone().add(new THREE.Vector3(0.3, -2.4, 0)); look = o.clone().add(new THREE.Vector3(0.3, 0, 0)); }
    else if (kind === "user") {
      const cam = (this.sceneInfo?.cameras || []).find((c) => c.name === "fixed-eye") || this.sceneInfo?.cameras?.[0];
      if (cam) {
        const m = cam.mat; // row-major 3x3; columns are camera x, y, z (z points backwards)
        eye = new THREE.Vector3(...cam.pos);
        look = eye.clone().sub(new THREE.Vector3(m[2], m[5], m[8]).multiplyScalar(0.6));
        up = new THREE.Vector3(m[1], m[4], m[7]);
        fov = Math.min(100, cam.fovy);
      }
    }
    // Default: over the right shoulder, so screens and targets face the viewer as they face the user.
    if (!eye) { eye = o.clone().add(new THREE.Vector3(-1.05, -1.45, 0.75)); look = o.clone().add(new THREE.Vector3(0.3, 0.05, -0.2)); }
    this.camera.position.copy(this._toThree(eye));
    this.camera.up.copy(up.applyQuaternion(this.world.quaternion));
    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(this._toThree(look));
    this.camera.lookAt(this.controls.target);
    this.controls.update();
  }

  // ------------------------------------------------------------------ targets (relative to the shoulder)
  setTargets(targets, selected) {
    this.targetData = targets;
    // Only the targets are rebuilt: a screen shares their group and stays until it is removed.
    this.gizmo.detach();
    disposeObjects(this.targets || []);
    this.targets = targets.map((t, i) => this._buildTarget(t, i));
    this.select(selected ?? null, true);
  }

  _buildTarget(t, i) {
    const grp = new THREE.Group();
    // A target that is switched off is drawn grey and without its number.
    const color = new THREE.Color(t.disabled ? "#9aa0a6" : t.color);
    const number = t.disabled ? 0 : this.targetData.slice(0, i + 1).filter((o) => !o.disabled).length;
    const lo = (r) => Math.min(...r), hi = (r) => Math.max(...r), mid = (r) => (lo(r) + hi(r)) / 2;
    grp.position.set(mid(t.x), mid(t.y), mid(t.z));
    const ext = [t.x, t.y, t.z].map((r) => hi(r) - lo(r));
    // The region: where the target centre is drawn each episode (a proxy the handles move and scale).
    const region = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, depthWrite: false }));
    region.scale.set(...ext.map((e) => Math.max(e, 0.002)));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(region.geometry), new THREE.LineBasicMaterial({ color }));
    region.add(edges);
    region.userData.target = i;
    grp.add(region);
    let marker;
    if (t.type === "Button") {
      // width, height and thickness (m); the thin axis is the button's normal
      const extent = (value, fallback) => Array.isArray(value) ? (value[0] + value[1]) / 2 : value ?? fallback;
      marker = new THREE.Mesh(new THREE.BoxGeometry(extent(t.bw, 0.05), extent(t.bh, 0.05), extent(t.bt, 0.02)), new THREE.MeshStandardMaterial({ color, roughness: 0.5, ...(t.screenEl != null ? { transparent: true, opacity: 0.45 } : {}) }));  // on a screen: the design stays visible
      marker.quaternion.copy(buttonQuat(t));
    } else {
      const r = Math.max(...t.size);
      marker = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 16), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.25, transparent: true, opacity: 0.6, roughness: 0.3 }));
      const minR = Math.min(...t.size);
      if (minR < r) marker.add(new THREE.Mesh(new THREE.SphereGeometry(minR, 24, 12), new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.5 })));
    }
    marker.userData.target = i;
    grp.add(marker);
    if (number) grp.add(label(String(number), t.color));
    else marker.traverse((o) => { if (o.material) { o.material.transparent = true; o.material.opacity = Math.min(o.material.opacity, 0.3); } });
    grp.userData = { i, region, marker, ext };
    this.targetGroup.add(grp);
    return grp;
  }

  select(i, silent = false) {
    this.selected = i;
    this.gizmo.detach();
    this.targets.forEach((g, k) => { g.userData.region.material.opacity = k === i ? 0.22 : 0.12; });
    if (i != null && this.targets[i] && this.editable) this._attachTarget(i);
    else if (i === "screen" && this.screen && this.editable) { this.gizmo.attach(this.screen); this.gizmo.setMode(this.mode === "rotate" ? "rotate" : "translate"); }
    if (!silent) this._emit("select", { i });
  }

  _attachTarget(i) {
    const g = this.targets[i];
    // The handles act on the whole target group: translate moves the region, scale resizes it.
    g.scale.set(1, 1, 1);
    // Rotate turns a button itself (its marker), not the region its centre is drawn from.
    if (this.mode === "rotate" && this.targetData[i].type === "Button") { this.gizmo.attach(g.userData.marker); this.gizmo.setMode("rotate"); return; }
    this.gizmo.attach(g);
    this.gizmo.setMode(this.mode === "scale" ? "scale" : "translate");
  }

  setMode(mode) {
    this.mode = mode;
    if (this.selected === "screen") this.gizmo.setMode(mode === "rotate" ? "rotate" : "translate");
    else if (this.selected != null) this._attachTarget(this.selected);
  }

  _onGizmo() {
    const obj = this.gizmo.object;
    if (!obj) return;
    if (obj === this.screen) { this._emit("screen-change", this.screenPose()); return; }
    if (obj.userData.target != null && obj.parent && obj.parent.userData.marker === obj) {  // a button turned by hand
      const k = obj.userData.target;
      this.targetData[k].quat = obj.quaternion.toArray().map((v) => Math.round(v * 1e5) / 1e5);
      delete this.targetData[k].euler;
      this.targetData[k].angle = -quatToEuler(this.targetData[k].quat)[1] * 180 / Math.PI;
      this._emit("rotate", { i: k });
      return;
    }
    const { i } = obj.userData;
    const ext = obj.userData.ext0 || obj.userData.ext;
    const t = this.targetData[i];
    const c = obj.position;
    // Scaling stretches the region (never the marker), from at least one grid step.
    const e = ext.map((x, k) => (obj.scale.getComponent(k) === 1 ? x : Math.max(0, snap(Math.max(x, STEP) * obj.scale.getComponent(k)))));
    const range = (centre, extent) => [snap(centre - extent / 2), snap(centre + extent / 2)].map((v) => Math.round(v * 1000) / 1000);
    t.x = range(c.x, e[0]); t.y = range(c.y, e[1]); t.z = range(c.z, e[2]);
    if (this.gizmo.mode === "scale") {
      obj.userData.ext = e;
      obj.userData.region.scale.set(...e.map((x) => Math.max(x, 0.002)));
      // Undo the group's stretch on the marker and label; the handles keep their own scale.
      obj.userData.marker.scale.set(...[0, 1, 2].map((k) => 1 / obj.scale.getComponent(k)));
      obj.userData.region.scale.divide(obj.scale);
    }
    this._emit("change", { i, target: t });
  }

  // ------------------------------------------------------------------ pointer: select, add, hover
  _bindPointer() {
    const el = this.renderer.domElement;
    const ndc = (ev) => { const r = el.getBoundingClientRect(); return new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1); };
    let down = null;
    el.addEventListener("pointerdown", (ev) => { down = [ev.clientX, ev.clientY]; });
    el.addEventListener("pointermove", (ev) => { this.hover = this._planePoint(ndc(ev)); });
    el.addEventListener("pointerleave", () => { this.hover = null; });
    el.addEventListener("pointerup", (ev) => {
      if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 4 || this.gizmo.dragging || !this.editable) return;
      this._raycaster.setFromCamera(ndc(ev), this.camera);
      const pick = [...this.targetGroup.children, ...(this.screen ? [this.screen] : [])];
      // The ray hits the screen's image plane, a child of the screen frame.
      const onScreen = (o) => !!this.screen && (o === this.screen || o.parent === this.screen);
      const hit = this._raycaster.intersectObjects(pick, true).find((h) => h.object.userData.target != null || onScreen(h.object));
      if (hit) this.select(onScreen(hit.object) ? "screen" : hit.object.userData.target);
      else if (this.selected != null) this.select(null);
    });
    el.addEventListener("dblclick", (ev) => {
      if (!this.editable) return;
      const p = this._planePoint(ndc(ev));
      if (p) this._emit("add", { pos: p });
    });
  }

  // Point under the pointer on the plane through the shoulder facing the camera, relative to the shoulder.
  _planePoint(v) {
    if (!this.bodies.length) return null;
    this._raycaster.setFromCamera(v, this.camera);
    const o = this._toThree(this.origin);
    const n = new THREE.Vector3().subVectors(this.camera.position, o).normalize();
    const hit = this._raycaster.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(n, o), new THREE.Vector3());
    if (!hit) return null;
    const local = this.world.worldToLocal(hit).sub(this.origin);
    return [snap(local.x), snap(local.y), snap(local.z)];
  }

  setEditable(on) {
    const was = this.editable;
    this.editable = on;
    if (!on) this.gizmo.detach();
    else if (!was) this.select(this.selected, true); // handles back on the selected target
  }

  // Move the target groups to their current ranges (no rebuild: a drag on the screen keeps going).
  moveTargets() {
    const mid = (r) => (Math.min(...r) + Math.max(...r)) / 2;
    this.targets.forEach((g, i) => {
      const t = this.targetData[i];
      g.position.set(mid(t.x), mid(t.y), mid(t.z));
      if (t.type === "Button") g.userData.marker.quaternion.copy(buttonQuat(t));
    });
  }

  // ------------------------------------------------------------------ screen (e.g. a Figma frame)
  // Pose relative to the shoulder; the image faces the user (normal -x), its left edge on the user's left (+y).
  setScreen(sc) {
    if (this.screen) { this.gizmo.detach(); disposeObjects([this.screen]); this.screen = null; }
    if (!sc) return;
    const tex = new THREE.TextureLoader().load(sc.image);
    tex.colorSpace = THREE.SRGBColorSpace;
    const h = sc.widthM * (sc.height / sc.width);
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(sc.widthM, h), new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, toneMapped: false }));
    const frame = new THREE.Group();
    frame.add(plane);
    frame.position.fromArray(sc.pos);
    frame.quaternion.fromArray(sc.quat || [0, 0, 0, 1]);
    plane.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0)));
    frame.userData = { ...sc, plane };
    this.screen = frame;
    this.targetGroup.add(frame);
  }

  screenPose() {
    if (!this.screen) return null;
    return { pos: this.screen.position.toArray().map((v) => Math.round(v * 1000) / 1000), quat: this.screen.quaternion.toArray() };
  }

  // Shoulder-relative centre of a screen element (pixel box), its half size, width and height in
  // metres and the screen's orientation there (x along the screen's width, z its normal towards the
  // user). ``lift`` moves the centre off the screen along the normal (a button lying on it).
  screenElement(el, lift = 0) {
    const s = this.screen.userData;
    const m = s.widthM / s.width;
    const u = (el.x + el.w / 2 - s.width / 2) * m, v = (s.height / 2 - (el.y + el.h / 2)) * m;
    const quat = this.screen.quaternion.clone().multiply(s.plane.quaternion);
    const p = new THREE.Vector3(u, v, lift).applyQuaternion(quat).add(this.screen.position);
    return { pos: p.toArray().map((x) => Math.round(x * 1000) / 1000), half: (Math.min(el.w, el.h) / 2) * m,
      w: el.w * m, h: el.h * m, quat: quat.toArray().map((x) => Math.round(x * 1e5) / 1e5) };
  }

  // ------------------------------------------------------------------ live feed
  // hello: { bodies: [names], targets: [{ rgba, radius }] }; then frames from applyFrame.
  setLive(hello) {
    this.live = hello;
    disposeObjects([...this.liveGroup.children]);
    this.targetGroup.visible = !hello;
    if (!hello) {
      this.bodies.forEach((b) => { b.position.fromArray(b.userData.rest.pos); b.quaternion.copy(MJ_QUAT(b.userData.rest.quat)); });
      this._musclesDirty = true;
      this.setMuscleActivity(null);
      return;
    }
    this.setEditable(false);
    hello.map = hello.bodies.map((n) => this.bodyByName.get(n) || null);
    hello.targetMeshes = hello.targets.map((t) => {
      // a button is drawn as its box (half extents and orientation from the model), a sphere by its radius
      const box = t.shape === "box" && t.size;
      const shape = box ? new THREE.BoxGeometry(2 * t.size[0], 2 * t.size[1], 2 * t.size[2]) : new THREE.SphereGeometry(Math.max(t.radius, 0.005), 32, 16);
      const m = new THREE.Mesh(shape, new THREE.MeshStandardMaterial({ color: new THREE.Color(t.rgba[0], t.rgba[1], t.rgba[2]), emissive: new THREE.Color(t.rgba[0], t.rgba[1], t.rgba[2]), emissiveIntensity: 0.25, roughness: 0.3, transparent: true, opacity: box ? 0.95 : 0.85 }));
      if (box && t.quat) m.quaternion.copy(MJ_QUAT(t.quat));
      this.liveGroup.add(m);
      return m;
    });
  }

  // Radius of live target k (the page's own simulation draws a new size per episode).
  setLiveTargetRadius(k, radius) {
    const L = this.live;
    if (L && L.targetMeshes[k]) L.targetMeshes[k].scale.setScalar(radius / Math.max(L.targets[k].radius, 0.005));
  }

  applyFrame(f) {
    const L = this.live;
    if (!L) return;
    const nb = L.bodies.length;
    for (let b = 0; b < nb; b++) {
      const g = L.map[b];
      if (!g) continue;
      g.position.set(f[3 * b], f[3 * b + 1], f[3 * b + 2]);
      const q = 3 * nb + 4 * b;
      g.quaternion.set(f[q + 1], f[q + 2], f[q + 3], f[q]);
    }
    const t0 = 7 * nb;
    L.targetMeshes.forEach((m, k) => m.position.set(f[t0 + 3 * k], f[t0 + 3 * k + 1], f[t0 + 3 * k + 2]));
    this._musclesDirty = true;
  }

  // ------------------------------------------------------------------ export
  async exportGLB() {
    const { GLTFExporter } = await import("./vendor/three/addons/GLTFExporter.js");
    const helperVisible = this.gizmo.getHelper().visible;
    this.gizmo.getHelper().visible = false;
    try {
      const glb = await new GLTFExporter().parseAsync(this.world, { binary: true, onlyVisible: true });
      return new Blob([glb], { type: "model/gltf-binary" });
    } finally {
      this.gizmo.getHelper().visible = helperVisible;
    }
  }
}

// A mesh too large to belong to a body (the room meshes of some models).
function isRoomSized(geo) {
  geo.computeBoundingSphere();
  return geo.boundingSphere.radius > 1;
}

// The light room the environment map is made from: bright above, darker below, with soft panels.
function studioEnvironment() {
  const env = new THREE.Scene();
  const panel = (color, strength, w, h, pos) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(strength), side: THREE.DoubleSide }));
    m.position.set(...pos);
    m.lookAt(0, 0, 0);
    env.add(m);
  };
  env.add(new THREE.Mesh(new THREE.SphereGeometry(20, 16, 8), new THREE.MeshBasicMaterial({ color: new THREE.Color(0x8f8c86), side: THREE.BackSide })));
  panel(0xfff6e8, 5, 12, 12, [4, 12, 6]);
  panel(0xdfe8ff, 2.5, 10, 12, [-12, 4, 4]);
  panel(0xffffff, 2, 12, 6, [2, 5, -12]);
  panel(0x6f6a62, 1, 30, 30, [0, -12, 0]);
  return env;
}

// Scene background: the theme's colour, a little lighter towards the top.
function backdrop(color) {
  const c = document.createElement("canvas");
  c.width = 2; c.height = 256;
  const ctx = c.getContext("2d"), base = new THREE.Color(color), up = base.clone().lerp(new THREE.Color(0xffffff), base.getHSL({}).l < 0.3 ? 0.07 : 0.45);
  const g = ctx.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0, `#${up.getHexString()}`);
  g.addColorStop(1, `#${base.clone().multiplyScalar(0.94).getHexString()}`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 2, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// The floor grid: fades out with the distance from the body, every metre a little stronger.
function fadingGrid(radius = 3.2, cell = 0.25) {
  const pos = [], col = [], steps = Math.round((2 * radius) / cell);
  const alpha = (x, z, strong) => { const r = Math.hypot(x, z) / radius, f = Math.max(0, 1 - r); return (strong ? 0.85 : 0.5) * f * f * (3 - 2 * f); };
  for (let i = 0; i <= steps; i++) {
    const u = -radius + i * cell, strong = Math.abs(u / 1 - Math.round(u / 1)) < 1e-6;
    for (let j = 0; j < steps; j++) {
      const a = -radius + j * cell, b = a + cell;
      pos.push(u, 0, a, u, 0, b, a, 0, u, b, 0, u);
      col.push(1, 1, 1, alpha(u, a, strong), 1, 1, 1, alpha(u, b, strong), 1, 1, 1, alpha(a, u, strong), 1, 1, 1, alpha(b, u, strong));
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 4));
  return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, toneMapped: false }));
}

function primitive(g) {
  const s = g.size;
  if (g.type === "sphere") return new THREE.SphereGeometry(s[0], 24, 16);
  if (g.type === "capsule") return new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16);
  if (g.type === "cylinder") return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24);
  if (g.type === "box") return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  if (g.type === "ellipsoid") return new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
  return null;
}

function label(text, color) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const ctx = c.getContext("2d");
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(32, 32, 28, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "600 34px system-ui, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, 32, 34);
  // The target's place in the sequence: small and see-through, so it does not hide a screen behind it.
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true, opacity: 0.55 }));
  sprite.scale.set(0.022, 0.022, 1);
  sprite.position.set(0, 0, 0.03);
  sprite.renderOrder = 10;
  return sprite;
}
