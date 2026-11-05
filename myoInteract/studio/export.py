"""Export a Studio task as a self-contained MuJoCo model (MJCF + meshes in one zip).

The zip opens in MuJoCo's viewer, Blender (MuJoCo importers) and Unity through MuJoCo's Unity
plug-in ("Import MuJoCo Scene"). Target regions are boxes, target spheres sites, both without
collisions, so the body model itself is unchanged.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET  # noqa: N817
import zipfile
from pathlib import Path

import mujoco
import numpy as np
from ml_collections import ConfigDict


def _mesh_source(spec: mujoco.MjSpec, file: str) -> Path:
    path = Path(file)
    if path.is_absolute():
        return path
    return Path(spec.modelfiledir) / spec.meshdir / path


def _normalize_export_xml(xml: str, target_region_names: set[str]) -> str:
    """Preserve parseable defaults and visual-only target regions in saved MJCF.

    Specs composed with ``attach`` (MyoArm) write an attached part's defaults as an unnamed nested
    class, which MuJoCo's parser rejects. Naming that class changes which defaults the worldbody's
    target geoms inherit, so explicitly retain their noncollision masks after serialization.
    """
    root = ET.fromstring(xml)
    for k, d in enumerate(root.iterfind(".//default/default")):
        if "class" not in d.attrib:
            d.set("class", f"attached_{k}")
    for geom in root.findall("./worldbody/geom"):
        if geom.get("name") in target_region_names:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    return ET.tostring(root, encoding="unicode")


def task_mjcf_zip(config: dict, name: str = "task") -> bytes:
    """Zip with ``<name>.xml`` and its meshes for a resolved task config.

    Args:
        config: Resolved task config as a dict (``server.resolve_preset``).
        name: File name of the model inside the zip.

    Returns:
        The zip archive.
    """
    from myoInteract.body_models import base_spec  # noqa: PLC0415

    config = ConfigDict(config)
    task = config.env.task_config
    spec = base_spec(config.env.model_path)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, task.reach_settings.ref_site
    )
    origin = data.site_xpos[site] + np.asarray(task.reach_settings.target_origin_rel)

    for i in range(task.targets.num_targets):
        t = getattr(task.targets, f"target_{i}", None) or task.targets.target_0
        lo, hi = np.asarray(t.position[0], float), np.asarray(t.position[1], float)
        centre, half = origin + (lo + hi) / 2, np.maximum(np.abs(hi - lo) / 2, 1e-4)
        rgb = [float(c) for c in list(t.rgb)[:3]]
        radius = float(np.max(np.asarray(t.size, float)))
        spec.worldbody.add_geom(
            name=f"target_{i + 1}_region",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=centre,
            size=half,
            rgba=[*rgb, 0.15],
            contype=0,
            conaffinity=0,
            group=2,
        )
        spec.worldbody.add_site(
            name=f"target_{i + 1}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=centre,
            size=[radius] * 3,
            rgba=[*rgb, 0.6],
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Meshes get flat, unique names under meshes/; to_xml recompiles, so the spec gets
        # their bytes as assets instead of reading the renamed files from disk.
        written: dict[Path, str] = {}
        assets: dict[str, bytes] = {}
        for mesh in spec.meshes:
            if not mesh.file:
                continue
            src = _mesh_source(spec, mesh.file)
            if src not in written:
                written[src] = f"{len(written):03d}_{src.name}"
                assets[f"meshes/{written[src]}"] = src.read_bytes()
                zf.writestr(f"meshes/{written[src]}", assets[f"meshes/{written[src]}"])
            mesh.file = written[src]
        spec.meshdir = "meshes"
        spec.assets = assets
        target_region_names = {
            f"target_{i + 1}_region" for i in range(task.targets.num_targets)
        }
        zf.writestr(
            f"{name}.xml", _normalize_export_xml(spec.to_xml(), target_region_names)
        )
    return buf.getvalue()
