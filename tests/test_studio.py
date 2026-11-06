"""Studio backend: API without a GPU, the preset converter it relies on, and the metrics reader."""

import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from myoInteract.myointeract_hydra_cli import BM_MODEL_MAP, gradio_json_to_overrides  # noqa: E402
from myoInteract.studio.examples import (  # noqa: E402
    PAGE_FILES,
    SIM_FILES,
    example_dir,
    list_examples,
)  # noqa: E402
from myoInteract.studio.runs import RunManager, metrics  # noqa: E402
from myoInteract.studio.server import BUNDLED_PRESETS, STATIC, create_app  # noqa: E402

POINTING = json.loads((BUNDLED_PRESETS / "pointing.json").read_text())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    # A free viewer port, so a viewer running on this machine is not picked up by the proxy.
    return TestClient(create_app(tmp_path, allow_origins=[], viewer_port=_free_port()))


def test_health_options_and_page(client: TestClient) -> None:
    assert client.get("/api/health").json()["ok"] is True
    assert set(client.get("/api/options").json()["models"]) == set(BM_MODEL_MAP)
    page = client.get("/")
    assert page.status_code == 200 and "MyoInteract Studio" in page.text


def test_presets_list_load_and_save(client: TestClient) -> None:
    names = {p["name"] for p in client.get("/api/presets").json()}
    assert "pointing" in names
    assert (
        client.get("/api/presets/pointing").json()["num_targets"]
        == POINTING["num_targets"]
    )
    assert (
        client.post(
            "/api/presets", json={"name": "mine", "preset": POINTING}
        ).status_code
        == 200
    )
    assert {"name": "mine", "source": "user"} in client.get("/api/presets").json()
    assert (
        client.post(
            "/api/presets", json={"name": "../evil", "preset": POINTING}
        ).status_code
        == 422
    )
    assert client.get("/api/presets/..%2Fpyproject").status_code == 404


def test_validate_counts_iterations_and_reports_errors(client: TestClient) -> None:
    preset = json.loads(json.dumps(POINTING))
    preset["rl"].update(
        {
            "Number of Total Training Steps": 1024 * 10 * 30,
            "Number of Parallel Environments": 1024,
            "Unroll Length": 10,
        }
    )
    body = client.post("/api/config/validate", json={"preset": preset}).json()
    assert body["iterations"] == 30
    bad = client.post(
        "/api/config/validate", json={"preset": {**preset, "bm": {"bm_model": "Nope"}}}
    )
    assert bad.status_code == 422 and "Nope" in bad.json()["detail"]


def test_unknown_runs_and_paths_are_rejected(client: TestClient) -> None:
    assert [
        r for r in client.get("/api/runs").json() if r["source"] != "examples"
    ] == []
    assert client.get("/api/runs/examples/nope").status_code == 404
    assert client.get("/api/runs/../../etc").status_code == 404
    assert (
        client.post(
            "/api/evaluations", json={"run": "../../etc", "checkpoint": "x"}
        ).status_code
        == 404
    )
    assert client.get("/api/jobs/nope").status_code == 404


def test_viewer_needs_a_preset_or_a_known_run(client: TestClient) -> None:
    assert client.get("/api/viewer").json()["status"] == "none"
    assert client.post("/api/viewer", json={}).status_code == 422
    assert client.post("/api/viewer", json={"run": "../../etc"}).status_code == 404
    assert (
        client.post(
            "/api/viewer", json={"run": "../../etc", "follow": True}
        ).status_code
        == 404
    )
    assert client.delete("/api/viewer").status_code == 200
    assert (
        client.get("/viewer/").status_code == 503
    )  # proxied viewer page: nothing running yet


def test_converter_reads_the_total_steps_and_the_body_model() -> None:
    preset = json.loads(json.dumps(POINTING))
    preset["rl"]["Number of Total Training Steps"] = 1_234_000
    preset["bm"]["bm_model"] = "MoBL_Arms_Hand"
    overrides = gradio_json_to_overrides(preset)
    assert "rl.num_timesteps=1234000" in overrides
    assert any("mobl_arms_hand_universal" in o for o in overrides)


def test_metrics_turn_termination_counts_into_a_success_rate(tmp_path: Path) -> None:
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "agent.yaml").write_text("num_steps_per_env: 10\n")
    (tmp_path / "params" / "env.yaml").write_text("scene:\n  num_envs: 64\n")
    scalars = {
        "Episode_Termination/episode_success": [(0, 0.0), (1, 3.0)],
        "Episode_Termination/time_out": [(0, 4.0), (1, 1.0)],
        "Train/mean_reward": [(0, -1.0), (1, 2.0)],
        "Train/mean_episode_length": [(0, 80.0), (1, 40.0)],
    }
    pts = metrics(tmp_path, scalars, ctrl_dt=0.05)["points"]
    assert [p["steps"] for p in pts] == [640, 1280]
    assert pts[1]["success"] == pytest.approx(0.75)
    assert pts[1]["episode_s"] == pytest.approx(2.0)


def test_runs_from_other_tools_are_listed(tmp_path: Path) -> None:
    run = tmp_path / "logs" / "rsl_rl" / "myo_universal" / "2026-01-01_00-00-00_x"
    run.mkdir(parents=True)
    (run / "model_0.onnx").write_bytes(b"")
    (run / "model_final.onnx").write_bytes(b"")
    listed = RunManager(tmp_path).list()
    assert listed[0]["id"] == "myo_universal/2026-01-01_00-00-00_x"
    assert listed[0]["checkpoints"] == ["model_0.onnx", "model_final.onnx"]


def test_full_training_log_of_a_studio_run(tmp_path: Path) -> None:
    client = TestClient(
        create_app(tmp_path, allow_origins=[], viewer_port=_free_port())
    )
    run = tmp_path / "logs" / "studio" / "runs" / "20260101-000000_x"
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps({"id": run.name, "name": "x", "pid": None})
    )
    (run / "train.log").write_text("iteration 1\niteration 2\n")
    r = client.get(f"/api/runs/{run.name}/log")
    assert r.status_code == 200 and r.text == "iteration 1\niteration 2\n"
    assert client.get("/api/runs/nope/log").status_code == 404


def test_unreachable_targets_are_refused_before_training(client: TestClient) -> None:
    preset = json.loads(json.dumps(POINTING))
    preset["task"]["reject_oor"] = True
    preset["target_1"]["sphere"].update(
        {"x_range": [0.61, 0.61], "y_range": [0.0, 0.0], "z_range": [-0.3, -0.3]}
    )
    r = client.post("/api/config/validate", json={"preset": preset})
    assert r.status_code == 422 and "out of reach" in r.json()["detail"]
    assert (
        client.post("/api/runs", json={"preset": preset, "name": "x"}).status_code
        == 422
    )


def test_target_sampling_terminates() -> None:
    import numpy as np

    from myoInteract.myointeract_utils import generate_target_pos

    rng = np.random.default_rng(0)
    fixed = np.array(
        [[0.3, 0.0, -0.2], [0.3, 0.0, -0.2]]
    )  # a reachable fixed point used to loop forever
    assert np.allclose(
        generate_target_pos(fixed, np.zeros(3), rng, reject_oor=True), [0.3, 0.0, -0.2]
    )
    with pytest.raises(ValueError, match="within reach"):
        generate_target_pos(fixed * 3, np.zeros(3), rng, reject_oor=True)


def test_viewer_refuses_a_busy_port(tmp_path: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        client = TestClient(
            create_app(tmp_path, allow_origins=[], viewer_port=sock.getsockname()[1])
        )
        r = client.post("/api/viewer", json={"preset": POINTING})
        assert r.status_code == 409 and "in use" in r.json()["detail"]


def test_concurrent_viewer_requests_start_one_viewer_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    import myoInteract.studio.runs as runs_module

    state = {"running": 0, "most": 0, "started": []}
    lock = threading.Lock()

    def start(self, kind: str, args: list[str], inputs: dict | None = None) -> dict:
        with lock:
            state["running"] += 1
            state["most"] = max(state["most"], state["running"])
        time.sleep(0.3)  # a worker that has not bound the port yet
        with lock:
            state["running"] -= 1
            state["started"].append(args)
        return {"id": f"job{len(state['started'])}", "alive": True}

    monkeypatch.setattr(runs_module.JobManager, "start", start)
    monkeypatch.setattr(runs_module.JobManager, "stop", lambda self, jid: None)
    client = TestClient(
        create_app(tmp_path, allow_origins=[], viewer_port=_free_port())
    )
    other = {**POINTING, "max_duration": 3}
    codes: list[int] = []
    threads = [
        threading.Thread(
            target=lambda b=b: codes.append(
                client.post("/api/viewer", json={"preset": b}).status_code
            )
        )
        for b in (POINTING, other)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert codes == [200, 200] and len(state["started"]) == 2 and state["most"] == 1


def test_hosted_pages_need_the_pairing_code(tmp_path: Path) -> None:
    site = "https://studio.example.org"
    client = TestClient(
        create_app(
            tmp_path,
            allow_origins=[site],
            viewer_port=_free_port(),
            pairing_code="abc123",
        )
    )
    hosted = {"Origin": site}
    # A hosted page learns only that a server is there, and cannot use it without the code.
    assert client.get("/api/health", headers=hosted).json() == {
        "ok": True,
        "pairing": True,
    }
    assert client.get("/api/runs", headers=hosted).status_code == 401
    assert (
        client.get(
            "/api/runs", headers={**hosted, "X-Studio-Token": "wrong"}
        ).status_code
        == 401
    )
    r = client.get("/api/runs", headers={**hosted, "X-Studio-Token": "abc123"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == site
    assert (
        "device"
        in client.get(
            "/api/health", headers={**hosted, "X-Studio-Token": "abc123"}
        ).json()
    )
    # The page served by Studio itself needs no code; scenes are public geometry.
    assert client.get("/api/runs").status_code == 200
    assert (
        client.post(
            "/api/config/validate",
            json={"preset": POINTING},
            headers={"Origin": "http://testserver"},
        ).status_code
        == 200
    )
    assert client.get("/scenes/MoBL_Arms_Index.json", headers=hosted).status_code == 200
    # Chrome's local network preflight is answered for allowed origins only.
    pre = {
        "Origin": site,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Private-Network": "true",
    }
    r = client.options("/api/runs", headers=pre)
    # The preflight has to succeed, not only carry the header (Starlette >= 1.x refuses it with a 400 otherwise).
    assert (
        r.status_code == 200
        and r.headers.get("access-control-allow-private-network") == "true"
    )
    other = {**pre, "Origin": "https://evil.example"}
    refused = client.options(
        "/api/runs", headers=other
    )  # other origins: the preflight fails, the origin is never granted
    assert (
        refused.status_code != 200
        and refused.headers.get("access-control-allow-origin") != "https://evil.example"
    )
    # The live feed's websocket refuses hosted pages without the code.
    with pytest.raises(Exception):
        with client.websocket_connect("/viewer/stream", headers=hosted) as ws:
            ws.receive_text()


def test_examples_hold_a_run_a_stored_result_and_a_replay() -> None:
    for example in list_examples():
        folder, replay = example_dir(example["name"]), example["replay"]
        for name in (
            *PAGE_FILES,
            "model_final.onnx",
            "original_hydra_config.json",
            "myointeract_config.json",
        ):
            assert (folder / name).is_file(), name
        # One per bundled task example, and in step with it.
        assert json.loads((folder / "preset.json").read_text()) == json.loads(
            (BUNDLED_PRESETS / f"{example['name']}.json").read_text()
        )
        hello = replay["hello"]
        assert hello["muscles"], "replays carry the muscle activations"
        assert hello["joints"] and all(
            j["range"][0] < j["range"][1] for j in hello["joints"]
        ), "and the joint angles"
        assert replay["floats"] == 7 * len(hello["bodies"]) + 3 * len(
            hello["targets"]
        ) + len(hello["muscles"]) + len(hello["joints"])
        assert len(example["evaluation"]["targets_reached"]) == len(hello["targets"])
        assert (folder / replay["file"]).stat().st_size == 4 * replay[
            "floats"
        ] * replay["frames"]
        assert 0 <= example["evaluation"]["success_rate"] <= 1


def test_examples_are_listed_as_runs(client: TestClient) -> None:
    listed = [r for r in client.get("/api/runs").json() if r["source"] == "examples"]
    assert [r["id"] for r in listed] == [
        f"examples/{e['name']}" for e in list_examples()
    ]
    for run in listed:
        assert run["evaluable"] and run["checkpoints"] == ["model_final.onnx"]
        assert (
            client.get(f"/api/runs/{run['id']}").json()["example"]["evaluation"][
                "episodes"
            ]
            > 0
        )
        assert client.get(f"/api/runs/{run['id']}/log").status_code == 404


def test_site_ships_the_examples_without_their_checkpoints(tmp_path: Path) -> None:
    from myoInteract.studio.build_site import build_site  # noqa: PLC0415

    build_site(tmp_path / "site", ["MyoArm_nohand"])
    index = json.loads((tmp_path / "site" / "examples" / "index.json").read_text())
    assert [e["name"] for e in index] == [
        e["name"]
        for e in list_examples()
        if e["replay"]["hello"]["model"] == "MyoArm_nohand"
    ]
    assert not list((tmp_path / "site").rglob("*.onnx"))
    # the visit counter is on the hosted page only, never on the page myo_studio serve sends
    assert "goatcounter" in (tmp_path / "site" / "index.html").read_text()
    assert "goatcounter" not in (STATIC / "index.html").read_text()
    for example in index:
        shipped = {
            p.name for p in (tmp_path / "site" / "examples" / example["name"]).iterdir()
        }
        assert shipped == {*PAGE_FILES, *(SIM_FILES if example["sim"] else ())}


def test_proxied_pages_count_as_studios_own_page(tmp_path: Path) -> None:
    from myoInteract.studio.colab import serve_command  # noqa: PLC0415
    from myoInteract.studio.server import _cross_origin  # noqa: PLC0415

    # Forwarding headers do not establish trust on the local listener.
    assert _cross_origin("https://me-studio.hf.space", "10.0.0.5:7860")
    assert not _cross_origin("https://me-studio.hf.space", "me-studio.hf.space")
    paired = TestClient(
        create_app(
            tmp_path / "a",
            allow_origins=[],
            viewer_port=_free_port(),
            pairing_code="abc123",
        )
    )
    assert (
        paired.get(
            "/api/runs", headers={"Origin": "https://x.colab.googleusercontent.com"}
        ).status_code
        == 401
    )
    # --behind-proxy (what the Colab launcher starts): no pairing code, whatever the proxy does to Host.
    assert "--behind-proxy" in serve_command(8000)
    proxied = TestClient(
        create_app(
            tmp_path / "b",
            allow_origins=[],
            viewer_port=_free_port(),
            pairing_code=None,
        )
    )
    assert (
        proxied.get(
            "/api/runs", headers={"Origin": "https://x.colab.googleusercontent.com"}
        ).status_code
        == 200
    )


def test_page_side_simulation_files_reproduce_on_native_mujoco() -> None:
    """The installed native engine reproduces its own reviewed physics profile."""
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from myoInteract.studio.browser import Reference  # noqa: PLC0415
    from studio_reference_profiles import load_profile  # noqa: PLC0415

    profile = load_profile("native", mujoco.mj_versionString())
    references = {
        example["name"]: example["reference"] for example in profile["examples"]
    }
    reports = []
    for example in list_examples():
        folder = example_dir(example["name"])
        if not all((folder / f).is_file() for f in SIM_FILES):
            continue
        sim = json.loads((folder / "sim.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(folder / "sim_model.xml"))
        assert model.nmesh == 0  # physics only: the page draws its own scene
        assert [model.body(b).name for b in range(model.nbody)] == sim["bodies"]
        ref = Reference(
            model, sim["task"], sim["network"], (folder / "sim_policy.bin").read_bytes()
        )
        r = references[example["name"]]
        ref.noisy = False
        ref.reset(r["start"], r["positions"], r["sizes"], r.get("fatigue_state"))
        qpos_errors = []
        for qpos in r["qpos"]:
            ref.step()
            qpos_errors.append(float(np.abs(ref.d.qpos - qpos).max()))
        reports.append(
            {
                "name": example["name"],
                "max_qpos_error": max(qpos_errors),
                "qpos_errors": qpos_errors,
                "current": ref.current,
                "expected_current": r["current"],
                "max_fatigue_error": float(np.abs(ref.MF - r["MF"]).max()),
            }
        )
    # Complete every example before asserting, so platform failures report the full envelope.
    # Each archive is version-matched: upstream engine improvements can change dynamics.
    # Even the same version includes build rounding amplified by controller feedback:
    # a one-ULP excitation pulse can cause 1e-5 drift in the same engine.
    # Separate identical-input actor and identical-state integration tests retain 1e-6 gates.
    diagnostic = json.dumps(reports, indent=2)
    assert all(r["max_qpos_error"] < 2e-5 for r in reports), diagnostic
    assert all(r["current"] == r["expected_current"] for r in reports), diagnostic
    assert all(r["max_fatigue_error"] < 1e-6 for r in reports), diagnostic


@pytest.mark.slow
def test_page_side_native_success_rate() -> None:
    """Statistical evaluation is separate from the fast deterministic replay."""
    import mujoco

    from myoInteract.studio.browser import Reference, success_rate

    for example in list_examples():
        folder = example_dir(example["name"])
        sim = json.loads((folder / "sim.json").read_text())
        benchmark_engine = sim["reference"].get("mujoco_version", "3.11.0")
        if mujoco.mj_versionString() != benchmark_engine:
            pytest.skip(
                f"Historical success-rate benchmark was measured with MuJoCo {benchmark_engine}; "
                f"installed engine is {mujoco.mj_versionString()}. "
                "Versioned native, browser physics and actor compatibility checks run separately."
            )
        model = mujoco.MjModel.from_xml_path(str(folder / "sim_model.xml"))
        ref = Reference(
            model, sim["task"], sim["network"], (folder / "sim_policy.bin").read_bytes()
        )
        assert abs(success_rate(ref, 60) - sim["native_success_rate"]) < 0.2


def test_page_side_fatigue_step_is_myosuites() -> None:
    """The reference's fatigue step (mirrored by ``static/sim.js``) against MyoSuite's own, on random excitations."""
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    torch = pytest.importorskip("torch")
    from myosuite.core.muscle_conditions import (
        FATIGUE_REST_THRESHOLD,
        TorchFatigueState,
    )  # noqa: PLC0415

    from myoInteract.studio.browser import Reference  # noqa: PLC0415

    folder = example_dir("pointing")
    if folder is None or not (folder / "sim.json").is_file():
        pytest.skip("no example with simulation files")
    sim = json.loads((folder / "sim.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(folder / "sim_model.xml"))
    theirs = TorchFatigueState.from_mj_model(model, num_envs=1)
    params = {
        k: getattr(theirs, "_" + k).numpy().tolist()
        for k in ("F", "R", "r", "tauact", "taudeact")
    }
    task = {
        **sim["task"],
        "fatigue": {
            **params,
            "rest_threshold": FATIGUE_REST_THRESHOLD,
            "reset": "fresh",
        },
    }
    ref = Reference(
        model, task, sim["network"], (folder / "sim_policy.bin").read_bytes()
    )
    rng = np.random.default_rng(0)
    for step in range(300):
        e = rng.uniform(0, 1, model.nu).astype(np.float32) * (
            step % 50 < 35
        )  # work, then rest
        mine = ref.fatigue_step(e)
        ctrl = theirs.step(torch.tensor(e)[None], task["ctrl_dt"])[0].numpy()
        assert np.abs(mine - ctrl).max() < 1e-5
    assert np.abs(ref.MF - theirs.MF[0].numpy()).max() < 1e-5 and ref.MF.max() > 0.01


@pytest.fixture
def wandb_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """W&B accepts only "good-key"; the machine itself has no login."""
    from myoInteract.studio.wandb_account import WandbAccount, WandbError

    def lookup(api_key: str | None) -> dict:
        if api_key != "good-key":
            raise WandbError("W&B did not accept the key")
        return {"user": "ada", "entity": "ada-lab", "teams": ["ada", "ada-lab"]}

    monkeypatch.setattr(WandbAccount, "_lookup", staticmethod(lookup))


def test_wandb_key_is_checked_kept_and_never_returned(
    client: TestClient, wandb_stub: None
) -> None:
    assert client.get("/api/wandb").json() == {"installed": True, "connected": False}
    assert client.post("/api/wandb", json={"api_key": "bad"}).status_code == 422
    r = client.post("/api/wandb", json={"api_key": "good-key"})
    assert (
        r.status_code == 200 and r.json()["user"] == "ada" and "good-key" not in r.text
    )
    status = client.get("/api/wandb").json()
    assert (
        status["connected"]
        and status["source"] == "studio"
        and status["teams"] == ["ada", "ada-lab"]
    )
    assert client.delete("/api/wandb").json()["connected"] is False


def test_wandb_runs_get_the_login_in_their_environment_only(
    tmp_path: Path, wandb_stub: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import myoInteract.studio.runs as runs_module

    spawned: list[dict] = []

    class Done:
        pid = None
        returncode = 0

        def poll(self) -> int:
            return 0

    def spawn(
        cmd: list[str],
        cwd: Path,
        log: Path,
        env: dict | None = None,
        stdin: int | None = None,
    ) -> Done:
        spawned.append(env or {})
        log.write_text(
            "wandb: 🚀 View run x at: https://wandb.ai/ada-lab/myointeract/runs/ab12cd\n",
            encoding="utf-8",
        )
        return Done()

    monkeypatch.setattr(runs_module, "_spawn", spawn)
    client = TestClient(
        create_app(tmp_path, allow_origins=[], viewer_port=_free_port())
    )
    body = {
        "preset": POINTING,
        "name": "w",
        "wandb": {"project": "myointeract", "entity": "ada-lab"},
    }
    assert client.post("/api/runs", json=body).status_code == 409  # no W&B login yet
    assert (
        client.post("/api/runs", json={**body, "wandb": {"project": "a b"}}).status_code
        == 422
    )
    client.post("/api/wandb", json={"api_key": "good-key"})
    run = client.post("/api/runs", json=body).json()
    env = spawned[-1]
    assert env["MYOINTERACT_LOGGER"] == "wandb" and env["WANDB_API_KEY"] == "good-key"
    assert env["WANDB_PROJECT"] == "myointeract" and env["WANDB_USERNAME"] == "ada-lab"
    assert run["wandb"] == {"project": "myointeract", "entity": "ada-lab"}
    assert run["wandb_url"] == "https://wandb.ai/ada-lab/myointeract/runs/ab12cd"
    assert (
        "good-key"
        not in (
            tmp_path / "logs" / "studio" / "runs" / run["id"] / "run.json"
        ).read_text()
    )
    client.post("/api/runs", json={"preset": POINTING, "name": "plain"})
    assert (
        spawned[-1]["MYOINTERACT_LOGGER"] == "tensorboard"
        and "WANDB_API_KEY" not in spawned[-1]
    )


def test_live_frames_end_with_the_muscle_activations_and_joint_angles() -> None:
    from types import SimpleNamespace

    import numpy as np

    torch = pytest.importorskip(
        "torch"
    )  # the Studio CI jobs run without the training stack

    from myoInteract.register_mjlab_myointeract_tasks import compile_base_model
    from myoInteract.studio.worker import live_frame, muscle_acts, shown_joints

    model = compile_base_model(BM_MODEL_MAP["MyoArm_nohand"][0])
    assert [model.actuator(a).name.rsplit("/", 1)[-1] for a in muscle_acts(model)][
        :3
    ] == ["DELT1", "DELT2", "DELT3"]
    # the arm's own joints, not the shoulder joints coupled to them
    joints = [model.joint(j).name.rsplit("/", 1)[-1] for j in shown_joints(model)]
    assert joints[:5] == [
        "elv_angle",
        "shoulder_elv",
        "shoulder_rot",
        "elbow_flexion",
        "pro_sup",
    ]
    data = SimpleNamespace(
        xpos=torch.zeros(1, 2, 3),
        xquat=torch.zeros(1, 2, 4),
        geom_xpos=torch.ones(1, 3, 3),
        act=torch.tensor([[0.1, 0.2, 0.3]]),
        qpos=torch.tensor([[0.5, 0.6]]),
    )
    frame = np.frombuffer(live_frame(data, [1], [2, 0], [1]), dtype=np.float32)
    assert frame.size == 7 * 2 + 3 + 2 + 1 and frame[-3:] == pytest.approx(
        [0.3, 0.1, 0.6]
    )
