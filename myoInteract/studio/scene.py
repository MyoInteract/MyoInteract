"""3D scene of a body model for Studio's browser view.

The browser draws the same MuJoCo model the task trains on: visible geoms with their meshes,
attached to bodies posed at the neutral pose (where the target origin is measured). The live
view later moves the same bodies by name.

    python -m myoInteract.studio.scene --out <dir>    # static scenes/<model>.json + .bin for hosted pages
"""

from __future__ import annotations

import argparse
import base64
import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from myoInteract.myointeract_hydra_cli import BM_MODEL_MAP
from myoInteract.myointeract_utils import ARM_LENGTH

GEOM_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}
VISIBLE_GROUPS = 2  # MuJoCo's viewer shows geom groups 0-2 by default
# The body models Studio offers: the MyoArm (myo_sim) without hand (the default), with a pointing hand or with
# hand, and MoBL-ARMS with index finger or full hand.
STUDIO_MODELS = [
    "MyoArm_nohand",
    "MyoArm_pointing",
    "MyoArm",
    "MoBL_Arms_Index",
    "MoBL_Arms_Hand",
]


def body_short_name(name: str) -> str:
    """Body name without an entity prefix (``universal_robot/humerus`` -> ``humerus``)."""
    return name.rsplit("/", 1)[-1]


def _floats(a: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(a).ravel()]


def export_scene(
    model: mujoco.MjModel, ref_site: str, ee_site: str, name: str
) -> tuple[dict, bytes]:
    """Scene description and mesh buffer of a compiled body model.

    Args:
        model: Compiled body model (no task geoms).
        ref_site: Site the target coordinates are relative to (the shoulder).
        ee_site: Site that has to reach the targets (the fingertip).
        name: Studio's name of the body model.

    Returns:
        ``(scene, buffer)``: JSON-ready scene and the binary buffer with the mesh vertices
        (float32) and faces (uint16 or uint32) the scene's ``meshes`` entries point into.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Binding enums and NumPy integer scalars need not implement symmetric equality.
    # Normalize both sides before dictionary membership or indexing at this boundary.
    geom_types = [int(kind) for kind in model.geom_type]
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    geoms = [
        g
        for g in range(model.ngeom)
        if geom_types[g] in GEOM_TYPES
        and model.geom_group[g] <= VISIBLE_GROUPS
        and model.geom_rgba[g, 3] > 0
    ]
    mesh_ids = sorted(
        {int(model.geom_dataid[g]) for g in geoms if geom_types[g] == mesh_type}
    )
    big = any(int(model.mesh_vertnum[m]) > 65535 for m in mesh_ids)
    index_dtype = np.uint32 if big else np.uint16

    verts, faces, mesh_index, meshes = [], [], {}, []
    offset = 0
    for m in mesh_ids:
        v0, nv = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
        verts.append(model.mesh_vert[v0 : v0 + nv].astype(np.float32).tobytes())
        mesh_index[m] = len(meshes)
        meshes.append({"vert": [offset, nv]})
        offset += nv * 12
    for m in mesh_ids:
        f0, nf = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
        faces.append(model.mesh_face[f0 : f0 + nf].astype(index_dtype).tobytes())
        meshes[mesh_index[m]]["face"] = [offset, nf]
        offset += nf * 3 * np.dtype(index_dtype).itemsize
        offset += (-offset) % 4  # keep every typed array 4-byte aligned
        faces[-1] += b"\0" * ((-len(faces[-1])) % 4)

    site = lambda s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)  # noqa: E731
    # Muscles: the path sites of each muscle's tendon, in the frame of the site's body, so the browser can
    # redraw the path when the bodies move (wrapping geoms are left out: the path is drawn site to site).
    muscles = []
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_TENDON:
            continue
        t = int(model.actuator_trnid[a, 0])
        wraps = range(model.tendon_adr[t], model.tendon_adr[t] + model.tendon_num[t])
        sites = [
            int(model.wrap_objid[w])
            for w in wraps
            if model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_SITE
        ]
        if len(sites) >= 2:
            muscles.append(
                {
                    "name": model.actuator(a).name,
                    "radius": round(float(model.tendon_width[t]), 5),
                    "points": [
                        [int(model.site_bodyid[i]), *_floats(model.site_pos[i])]
                        for i in sites
                    ],
                }
            )
    scene = {
        "version": 1,
        "model": name,
        "frame": "mujoco",  # x forward, y left, z up (metres)
        "bodies": [
            {
                "name": body_short_name(model.body(b).name),
                "parent": int(model.body_parentid[b]),
                "pos": _floats(data.xpos[b]),
                "quat": _floats(data.xquat[b]),
            }
            for b in range(model.nbody)
        ],
        "geoms": [
            {
                "body": int(model.geom_bodyid[g]),
                "type": GEOM_TYPES[geom_types[g]],
                "size": _floats(model.geom_size[g]),
                "pos": _floats(model.geom_pos[g]),
                "quat": _floats(model.geom_quat[g]),
                "rgba": _floats(model.geom_rgba[g], 3),
                # Primitive shapes are body volumes (e.g. MoBL-ARMS' collision capsules), bones are meshes.
                "volume": geom_types[g] != mesh_type,
                **(
                    {"mesh": mesh_index[int(model.geom_dataid[g])]}
                    if geom_types[g] == mesh_type
                    else {}
                ),
            }
            for g in geoms
        ],
        "meshes": meshes,
        "muscles": muscles,
        "index_type": "uint32" if big else "uint16",
        "origin": _floats(data.site_xpos[site(ref_site)]),
        "reach": ARM_LENGTH,
        "ee_site": {"name": ee_site, "pos": _floats(data.site_xpos[site(ee_site)])},
        "cameras": [
            {
                "name": model.camera(c).name,
                "pos": _floats(data.cam_xpos[c]),
                "mat": _floats(
                    data.cam_xmat[c]
                ),  # columns: camera x (right), y (up), -z (view direction)
                "fovy": round(float(model.cam_fovy[c]), 3),
            }
            for c in range(model.ncam)
        ],
    }
    return scene, b"".join(verts) + b"".join(faces)


@lru_cache(maxsize=16)
def scene_for(bm_model: str) -> tuple[str, bytes]:
    """Scene JSON and mesh buffer of one of Studio's body models (cached).

    Raises:
        KeyError: Unknown body model.
    """
    from myoInteract.body_models import compile_base_model  # noqa: PLC0415

    model_path, ee_site, ref_site = BM_MODEL_MAP[bm_model]
    scene, buffer = export_scene(
        compile_base_model(model_path), ref_site, ee_site, bm_model
    )
    scene["buffer"] = f"{bm_model}.bin"
    return json.dumps(scene, separators=(",", ":")), buffer


def write_static_scenes(out: Path, models: list[str]) -> None:
    """Write ``<model>.json`` and its mesh buffer for pages without a backend.

    The buffer is base64 text (``<model>.b64.txt``), which every static host serves;
    the page decodes it.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name in models:
        text, buffer = scene_for(name)
        scene = json.loads(text)
        scene["buffer"] = f"{name}.b64.txt"
        (out / f"{name}.json").write_text(json.dumps(scene, separators=(",", ":")))
        (out / scene["buffer"]).write_text(base64.b64encode(buffer).decode())
        print(f"{name}: {len(buffer) / 1e6:.1f} MB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", nargs="*", default=STUDIO_MODELS)
    args = p.parse_args()
    write_static_scenes(args.out, args.models)


if __name__ == "__main__":
    main()
