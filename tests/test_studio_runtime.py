"""Studio's advertised CPU features need no optional training packages."""

from pathlib import Path
import subprocess
import sys


def test_cpu_studio_reads_tensorboard_and_generates_localhost_tls(
    tmp_path: Path,
) -> None:
    script = r"""
import importlib.abc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, sys.argv[1])
class NoTrainingPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mjlab', 'warp', 'rsl_rl'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, NoTrainingPackages())

from cryptography import x509
from cryptography.hazmat.primitives.serialization import load_pem_private_key, Encoding, PublicFormat
from fastapi.testclient import TestClient
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from myoInteract.studio.server import create_app, localhost_certificate

root = Path(sys.argv[2])
run = root / 'logs/rsl_rl/experiment/run'
run.mkdir(parents=True)
(run / 'model_final.onnx').touch()
(run / 'myointeract_config.json').write_text(json.dumps({'env': {'ctrl_dt': 0.05}}))
writer = EventFileWriter(str(run))
writer.add_event(Event(wall_time=time.time(), step=3, summary=Summary(value=[
    Summary.Value(tag='Train/mean_reward', simple_value=1.25),
    Summary.Value(tag='Train/mean_episode_length', simple_value=10),
])))
writer.close()
with TestClient(create_app(root, [])) as client:
    response = client.get('/api/runs/experiment/run')
    assert response.status_code == 200, response.text
    point = response.json()['metrics']['points'][0]
    assert point['reward'] == 1.25 and point['episode_s'] == 0.5, point
cert, key = localhost_certificate(root / 'tls')
certificate = x509.load_pem_x509_certificate(cert.read_bytes())
private = load_pem_private_key(key.read_bytes(), password=None)
assert certificate.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo) == private.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
assert 'localhost' in certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(x509.DNSName)
old = (cert.read_bytes(), key.read_bytes())
assert localhost_certificate(root / 'tls') == (cert, key)
assert old == (cert.read_bytes(), key.read_bytes())
assert not {'torch', 'mjlab', 'warp', 'rsl_rl'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(__file__).resolve().parents[1]),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
