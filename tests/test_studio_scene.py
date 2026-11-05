"""Studio's 3D scene backend: model geometry for the browser, MuJoCo export, Figma import."""

import io
import json
import socket
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

import mujoco  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from myoInteract.studio.figma import FigmaError, frame_elements, parse_figma_url  # noqa: E402
from myoInteract.studio.scene import export_scene, scene_for  # noqa: E402
from myoInteract.studio.server import BUNDLED_PRESETS, create_app, resolve_preset  # noqa: E402

POINTING = json.loads((BUNDLED_PRESETS / "pointing.json").read_text())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path, allow_origins=[], viewer_port=_free_port()))


def _geometry_diagnostic(model: mujoco.MjModel) -> dict:
    return {
        "mujoco": mujoco.__version__,
        "geom_type_dtype": str(model.geom_type.dtype),
        "types_groups_alpha": sorted(
            {
                (
                    int(model.geom_type[g]),
                    int(model.geom_group[g]),
                    float(model.geom_rgba[g, 3]),
                )
                for g in range(model.ngeom)
            }
        ),
        "enum_hashes": [
            (int(kind), hash(kind))
            for kind in (
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                mujoco.mjtGeom.mjGEOM_BOX,
                mujoco.mjtGeom.mjGEOM_MESH,
            )
        ],
    }


def test_scene_preserves_every_supported_geometry_type_and_visibility() -> None:
    """Use actual compiled integer arrays, including primitive groups 0, 1 and 2."""
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <default><geom contype="0" conaffinity="0"/></default>
      <asset><mesh name="tetra" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/></asset>
      <worldbody><site name="anchor"/>
        <geom type="sphere" size="0.1" group="0"/>
        <geom type="capsule" size="0.1 0.2" group="1"/>
        <geom type="ellipsoid" size="0.1 0.2 0.3" group="2"/>
        <geom type="cylinder" size="0.1 0.2"/>
        <geom type="box" size="0.1 0.2 0.3"/>
        <geom type="mesh" mesh="tetra"/>
        <geom type="sphere" size="0.1" group="3"/>
        <geom type="box" size="0.1 0.2 0.3" rgba="1 1 1 0"/>
      </worldbody></mujoco>""")
    scene, buffer = export_scene(model, "anchor", "anchor", "all-geometries")
    expected = {"sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"}
    assert len(scene["geoms"]) == len(expected), _geometry_diagnostic(model)
    assert {geom["type"] for geom in scene["geoms"]} == expected, _geometry_diagnostic(
        model
    )
    assert all(geom["volume"] == (geom["type"] != "mesh") for geom in scene["geoms"])
    assert len(scene["meshes"]) == 1 and buffer


def test_scene_geometry_matches_the_model() -> None:
    text, buffer = scene_for("MoBL_Arms_Index")
    sc = json.loads(text)
    index = np.uint32 if sc["index_type"] == "uint32" else np.uint16
    for m in sc["meshes"]:
        verts = np.frombuffer(buffer, np.float32, m["vert"][1] * 3, m["vert"][0])
        faces = np.frombuffer(buffer, index, m["face"][1] * 3, m["face"][0])
        assert np.isfinite(verts).all() and faces.max() < m["vert"][1]
    assert all(0 <= g["body"] < len(sc["bodies"]) for g in sc["geoms"])
    # The target origin is the shoulder site at the neutral pose, as in the task.
    from myoInteract.body_models import compile_base_model  # noqa: PLC0415
    from myoInteract.myointeract_hydra_cli import BM_MODEL_MAP  # noqa: PLC0415

    model_path, _, ref_site = BM_MODEL_MAP["MoBL_Arms_Index"]
    model = compile_base_model(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ref_site)
    assert np.allclose(sc["origin"], data.site_xpos[site], atol=1e-5)
    assert "fixed-eye" in [c["name"] for c in sc["cameras"]]


def test_scene_routes(client: TestClient) -> None:
    sc = client.get("/scenes/MoBL_Arms_Index.json").json()
    buf = client.get("/scenes/MoBL_Arms_Index.bin").content
    assert (
        sc["buffer"] == "MoBL_Arms_Index.bin"
        and len(buf) == scene_for("MoBL_Arms_Index")[1].__len__()
    )
    assert client.get("/scenes/Nope.json").status_code == 404
    assert client.get("/scenes/MoBL_Arms_Index.txt").status_code == 404
    assert client.get("/static/scene.js").status_code == 200


def test_mujoco_export_loads_back_with_its_targets(tmp_path: Path) -> None:
    from myoInteract.studio.export import task_mjcf_zip  # noqa: PLC0415

    _, config = resolve_preset(POINTING)
    zipfile.ZipFile(io.BytesIO(task_mjcf_zip(config, "pointing"))).extractall(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "pointing.xml"))
    for i in range(POINTING["num_targets"]):
        assert (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"target_{i + 1}") >= 0
        )


def test_mujoco_export_route(client: TestClient) -> None:
    r = client.post("/api/export/mjcf", json={"preset": POINTING, "name": "my task"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert 'filename="my_task.zip"' in r.headers["content-disposition"]
    assert "my_task.xml" in zipfile.ZipFile(io.BytesIO(r.content)).namelist()


def test_figma_links_and_elements() -> None:
    assert parse_figma_url(
        "https://www.figma.com/design/AbC123/My-UI?node-id=12-34&t=x"
    ) == ("AbC123", "12:34")
    with pytest.raises(FigmaError, match="no frame"):
        parse_figma_url("https://www.figma.com/file/AbC123/My-UI")
    with pytest.raises(FigmaError, match="not a link"):
        parse_figma_url("https://example.org/design/x")
    box = lambda x, y, w, h: {"x": x, "y": y, "width": w, "height": h}  # noqa: E731
    frame = {
        "name": "Login",
        "absoluteBoundingBox": box(100, 100, 400, 300),
        "children": [
            {
                "type": "TEXT",
                "name": "Title",
                "absoluteBoundingBox": box(120, 110, 200, 30),
            },
            {
                "type": "FRAME",
                "name": "Form",
                "absoluteBoundingBox": box(100, 150, 400, 200),
                "children": [
                    {
                        "type": "INSTANCE",
                        "name": "Input",
                        "absoluteBoundingBox": box(120, 160, 300, 40),
                        "children": [
                            {
                                "type": "RECTANGLE",
                                "name": "bg",
                                "absoluteBoundingBox": box(120, 160, 300, 40),
                            }
                        ],
                    },
                    {
                        "type": "RECTANGLE",
                        "name": "Sign in button",
                        "absoluteBoundingBox": box(120, 300, 120, 40),
                    },
                    {
                        "type": "RECTANGLE",
                        "name": "Hidden button",
                        "visible": False,
                        "absoluteBoundingBox": box(0, 0, 9, 9),
                    },
                ],
            },
        ],
    }
    out = frame_elements(frame)
    assert (out["width"], out["height"]) == (400, 300)
    assert [e["name"] for e in out["elements"]] == ["Input", "Sign in button"]
    assert out["elements"][1] == {
        "name": "Sign in button",
        "x": 20,
        "y": 200,
        "w": 120,
        "h": 40,
    }
    # A chip with its own background counts as a whole, not as the icon inside it; what is scrolled
    # out of the frame is dropped; a node with a prototype interaction is an element whatever its name.
    fill = [{"type": "SOLID", "visible": True}]
    frame["children"] += [
        {
            "type": "FRAME",
            "name": "Sort",
            "fills": fill,
            "cornerRadius": 8,
            "absoluteBoundingBox": box(300, 110, 80, 32),
            "children": [
                {
                    "type": "INSTANCE",
                    "name": "chevron",
                    "absoluteBoundingBox": box(350, 118, 16, 16),
                },
                {
                    "type": "TEXT",
                    "name": "Sort",
                    "absoluteBoundingBox": box(310, 118, 30, 16),
                },
            ],
        },
        {
            "type": "FRAME",
            "name": "Tab bar",
            "fills": fill,
            "absoluteBoundingBox": box(100, 420, 400, 60),
            "children": [
                {
                    "type": "INSTANCE",
                    "name": "Home",
                    "absoluteBoundingBox": box(120, 430, 40, 40),
                }
            ],
        },
        {
            "type": "GROUP",
            "name": "Card",
            "interactions": [{"trigger": {"type": "ON_CLICK"}}],
            "absoluteBoundingBox": box(120, 345, 360, 50),
            "children": [
                {
                    "type": "TEXT",
                    "name": "Location",
                    "absoluteBoundingBox": box(130, 350, 100, 20),
                }
            ],
        },
    ]
    out = frame_elements(frame)
    assert [e["name"] for e in out["elements"]] == [
        "Sort",
        "Input",
        "Sign in button",
        "Card",
    ]
    assert out["elements"][0] == {"name": "Sort", "x": 200, "y": 10, "w": 80, "h": 32}


def test_figma_route_reports_bad_links(client: TestClient) -> None:
    r = client.post(
        "/api/figma",
        json={"url": "https://www.figma.com/design/AbC123/x", "token": "t"},
    )
    assert r.status_code == 422 and "no frame" in r.json()["detail"]


def test_live_view_defaults_to_the_stream_job(client: TestClient) -> None:
    assert (
        client.post(
            "/api/viewer", json={"run": "../../etc", "kind": "stream"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/viewer", json={"preset": POINTING, "kind": "nope"}
        ).status_code
        == 422
    )


def test_static_scenes_ship_the_buffer_as_base64(tmp_path: Path) -> None:
    import base64  # noqa: PLC0415

    from myoInteract.studio.scene import write_static_scenes  # noqa: PLC0415

    write_static_scenes(tmp_path, ["MoBL_Arms_Index"])
    sc = json.loads((tmp_path / "MoBL_Arms_Index.json").read_text())
    assert sc["buffer"] == "MoBL_Arms_Index.b64.txt"
    assert (
        base64.b64decode((tmp_path / sc["buffer"]).read_text())
        == scene_for("MoBL_Arms_Index")[1]
    )


def test_scene_exports_muscle_paths_and_marks_body_volumes():
    """Muscles are site paths on existing bodies; primitive geoms (the MoBL capsules) are the skin, meshes are bones."""
    scene = json.loads(scene_for("MoBL_Arms_Index")[0])
    assert len(scene["muscles"]) == 26
    nbody = len(scene["bodies"])
    for muscle in scene["muscles"]:
        assert len(muscle["points"]) >= 2 and muscle["radius"] > 0
        assert all(0 <= p[0] < nbody and len(p) == 4 for p in muscle["points"])
    volumes = [g for g in scene["geoms"] if g["volume"]]
    if not volumes:
        from myoInteract.body_models import compile_base_model  # noqa: PLC0415
        from myoInteract.myointeract_hydra_cli import BM_MODEL_MAP  # noqa: PLC0415

        pytest.fail(
            f"MoBL scene has no body volumes: {_geometry_diagnostic(compile_base_model(BM_MODEL_MAP['MoBL_Arms_Index'][0]))}"
        )
    assert all(g["type"] != "mesh" for g in volumes)
    assert any(g["type"] == "mesh" and not g["volume"] for g in scene["geoms"])
