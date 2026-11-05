"""CPU Studio must work without importing its optional training backends."""

import subprocess
import sys
from pathlib import Path


def test_studio_preview_and_export_without_training_packages(tmp_path: Path) -> None:
    """Compile and step every Studio export with optional training imports blocked."""
    script = """
import importlib.abc
import io
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

import mujoco
import numpy as np

class NoTrainingPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mjlab', 'warp', 'rsl_rl'}:
            raise ModuleNotFoundError(f'Blocked optional training package: {fullname}', name=fullname)

sys.meta_path.insert(0, NoTrainingPackages())
import myoInteract
from fastapi.testclient import TestClient
from myoInteract.studio.scene import STUDIO_MODELS
from myoInteract.studio.server import BUNDLED_PRESETS, create_app, resolve_preset

with TestClient(create_app(Path(sys.argv[1]), allow_origins=[])) as client:
    assert client.get('/api/health').json()['device'] == 'CPU'
    preset = json.loads((BUNDLED_PRESETS / 'pointing.json').read_text())
    preset['num_targets'] = 2
    for body_model in STUDIO_MODELS:
        preset['bm']['bm_model'] = body_model
        validated = client.post('/api/config/validate', json={'preset': preset})
        assert validated.status_code == 200, (body_model, validated.text)
        response = client.get(f'/scenes/{body_model}.json')
        assert response.status_code == 200, (body_model, response.text)
        scene = response.json()
        assert scene['model'] == body_model and scene['frame'] == 'mujoco', body_model
        scene_counts = {
            'model': body_model,
            'mujoco': mujoco.__version__,
            'meshes': len(scene['meshes']),
            'muscles': len(scene['muscles']),
            'geoms': len(scene['geoms']),
            'volumes': sum(geom['volume'] for geom in scene['geoms']),
        }
        assert scene['meshes'] and scene['muscles'], scene_counts
        assert np.isfinite(scene['origin']).all()
        assert np.isfinite(scene['ee_site']['pos']).all()
        response = client.get(f'/scenes/{body_model}.bin')
        assert response.status_code == 200, body_model
        buffer = response.content
        index_dtype = np.uint32 if scene['index_type'] == 'uint32' else np.uint16
        for mesh in scene['meshes']:
            vertices = np.frombuffer(buffer, np.float32, mesh['vert'][1] * 3, mesh['vert'][0])
            faces = np.frombuffer(buffer, index_dtype, mesh['face'][1] * 3, mesh['face'][0])
            assert np.isfinite(vertices).all() and faces.size, body_model
            assert faces.max() < mesh['vert'][1], body_model

        exported = client.post('/api/export/mjcf', json={'preset': preset, 'name': body_model})
        assert exported.status_code == 200, (body_model, exported.text)
        destination = Path(sys.argv[1]) / 'exports' / body_model
        destination.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            names = archive.namelist()
            assert archive.testzip() is None and len(set(names)) == len(names), body_model
            assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
            xml = ET.fromstring(archive.read(f'{body_model}.xml'))
            meshdir = xml.find('compiler').get('meshdir', '')
            for mesh in xml.findall('./asset/mesh'):
                if 'file' in mesh.attrib:
                    asset = PurePosixPath(meshdir, mesh.attrib['file'])
                    assert not asset.is_absolute() and '..' not in asset.parts, body_model
                    assert asset.as_posix() in names, (body_model, asset)
            archive.extractall(destination)

        model = mujoco.MjModel.from_xml_path(str(destination / f'{body_model}.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert model.nu > 0 and model.nq > 0 and model.nmesh > 0, body_model
        assert model.nbody == len(scene['bodies']), body_model
        ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, scene['ee_site']['name'])
        assert ee >= 0, body_model
        np.testing.assert_allclose(data.site_xpos[ee], scene['ee_site']['pos'], atol=1e-6)
        _, config = resolve_preset(preset)
        task = config['env']['task_config']
        origin = np.asarray(scene['origin']) + np.asarray(task['reach_settings']['target_origin_rel'])
        for i in range(preset['num_targets']):
            target = task['targets'][f'target_{i}']
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'target_{i + 1}')
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'target_{i + 1}_region')
            assert site >= 0 and geom >= 0, body_model
            expected = origin + np.asarray(target['position']).mean(axis=0)
            np.testing.assert_allclose(data.site_xpos[site], expected, atol=1e-6)
            np.testing.assert_allclose(model.geom_pos[geom], expected, atol=1e-6)
            assert model.site_size[site, 0] > 0, body_model
            assert model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0, body_model
        data.ctrl[:] = 0
        for _ in range(10):
            mujoco.mj_step(model, data)
            assert all(np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc, data.act)), body_model
        np.testing.assert_allclose(data.time, 10 * model.opt.timestep, atol=1e-12)
        assert not np.any(data.warning.number), (body_model, data.warning.number)
assert not {'torch', 'mjlab', 'warp', 'rsl_rl'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
