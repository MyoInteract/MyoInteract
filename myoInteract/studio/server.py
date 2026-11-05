"""``myo_studio serve``: the Studio page and its HTTP API on one port.

    myo_studio serve                  # http://localhost:8000
    myo_studio serve --port 8001 --allow-origin https://example.org

Training runs ``myo_train`` from a preset in a subprocess; previews and evaluations run
:mod:`myoInteract.studio.worker` subprocesses. Data goes to :func:`myoInteract.paths.data_root`.
The server listens on localhost only unless ``--host`` says otherwise: anyone who can reach it can
start jobs on this machine's GPU.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import secrets
import socket
import sys
import threading
import time
import webbrowser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from pydantic import BaseModel, Field

from myoInteract.paths import data_root
from myoInteract.studio.ood import task_changes
from myoInteract.studio.runs import JobManager, RunManager, _contained_path
from myoInteract.studio.wandb_account import WandbAccount

STATIC = Path(__file__).parent / "static"
# Hosted copies of the Studio page that may use a local server (with its pairing code).
HOSTED_ORIGINS = ["https://myointeract.github.io"]
BUNDLED_PRESETS = Path(__file__).resolve().parents[2] / "myo_config" / "gradio_tasks"
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_hydra_lock = threading.Lock()  # Hydra's global state is not thread-safe


class PresetBody(BaseModel):
    preset: dict[str, Any]
    name: str = "untitled"


class ViewerBody(BaseModel):
    preset: dict[str, Any] | None = (
        None  # with a run: the run's policy on this setup (out of distribution)
    )
    run: str | None = None
    checkpoint: str = "model_final.onnx"
    follow: bool = False  # show each new checkpoint of the run as training saves it
    kind: Literal["stream", "viser"] = (
        "stream"  # Studio's own scene, or the viser viewer
    )


class FigmaBody(BaseModel):
    url: str
    token: str


class WandbLoginBody(BaseModel):
    api_key: str
    remember: bool = False


class WandbRunBody(BaseModel):
    project: str = Field("myointeract", pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    entity: str | None = Field(None, pattern=r"^[A-Za-z0-9_.-]{1,128}$")


class RunBody(PresetBody):
    wandb: WandbRunBody | None = (
        None  # log this run to W&B as well (local tensorboard files stay)
    )


class EvaluationBody(BaseModel):
    run: str
    checkpoint: str = "model_final.onnx"
    episodes: int = Field(64, ge=1, le=4096)
    video_seconds: float = Field(8.0, ge=0.0, le=60.0)
    preset: dict[str, Any] | None = (
        None  # the setup in the editor instead of the run's own (out of distribution)
    )


def resolve_preset(preset: dict) -> tuple[list[str], dict]:
    """Hydra overrides and the resolved config of a Gradio-format preset (raises ``HTTPException``)."""
    from myoInteract.myointeract_hydra_cli import (
        gradio_json_to_overrides,
        load_config_interactive,
    )  # noqa: PLC0415

    try:
        overrides = gradio_json_to_overrides(preset)
        with _hydra_lock:
            config = load_config_interactive(overrides)
    except KeyError as exc:
        raise HTTPException(422, f"unknown value in the preset: {exc}") from exc
    except Exception as exc:  # any other config error goes back to the page
        raise HTTPException(422, f"{type(exc).__name__}: {exc}") from exc
    from myoInteract.myointeract_utils import ARM_LENGTH, out_of_reach_targets  # noqa: PLC0415

    unreachable = out_of_reach_targets(config.env.task_config.targets)
    if unreachable:
        names = ", ".join(str(i + 1) for i in unreachable)
        raise HTTPException(
            422,
            f"Target {names} lies entirely out of reach (more than {ARM_LENGTH} m from the shoulder) while "
            "out-of-reach targets are rejected. Move it inside the dashed reach circle or turn off the rejection.",
        )
    return overrides, config.to_dict()


def shifted_task(log_dir: Path, preset: dict) -> dict:
    """Job input for a run's policy on the editor's setup: its overrides and what it changes.

    Raises ``HTTPException`` if the setup is invalid or changes what the policy depends on.
    """
    from myoInteract.myointeract_hydra_cli import load_config_interactive  # noqa: PLC0415

    overrides, setup = resolve_preset(preset)
    with _hydra_lock:
        trained = load_config_interactive(
            json.loads((log_dir / "original_hydra_config.json").read_text())
        ).to_dict()
    try:
        changes = task_changes(trained, setup)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"overrides": overrides, "changes": changes}


def _port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _device() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return {"device": "CPU", "cuda": None}

    if torch.cuda.is_available():
        return {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
    return {"device": "CPU", "cuda": None}


def _cross_origin(origin: str | None, host: str | None) -> bool:
    """Whether a page's origin differs from the listener's host.

    Forwarding headers are untrusted on the local listener. An authenticated
    proxy deployment can explicitly disable pairing with ``--behind-proxy``.
    """
    return origin is not None and urlparse(origin).netloc != (host or "")


def create_app(
    root: Path,
    allow_origins: list[str],
    max_runs: int = 1,
    viewer_port: int = 8080,
    pairing_code: str | None = None,
    warm_worker: bool = False,
) -> FastAPI:
    """The Studio app: ``/`` serves the page, ``/api/*`` the backend.

    Pages served elsewhere (``allow_origins``, e.g. the hosted Studio) reach ``/api/*`` and
    ``/viewer`` only with ``pairing_code`` (header ``X-Studio-Token``, or ``?token=`` for the live
    feed's websocket), which ``myo_studio serve`` prints; the page served here needs none.

    The live 3D feed (or the viser viewer) runs as a job on localhost:``viewer_port``, one at a time,
    and is proxied under ``/viewer/`` so the browser only needs Studio's own port. ``/scenes/`` serves
    the body models' geometry for the page's own 3D scene.
    ``warm_worker`` keeps one job worker waiting with its imports done (``myo_studio serve`` uses it),
    so a live view starts in a few seconds instead of about 15.
    """
    runs, jobs = RunManager(root), JobManager(root, warm=warm_worker)
    admission_lock = threading.Lock()
    viewer_lock = (
        threading.Lock()
    )  # one viewer at a time: concurrent requests must not both start one
    wandb_account = WandbAccount()
    viewer_job: dict[str, str | None] = {"id": None, "key": None}

    def _stop_viewer() -> None:
        # Jobs run detached from the server; a viewer left behind would keep the port busy.
        if viewer_job["id"]:
            jobs.stop(viewer_job["id"])

    app = FastAPI(title="MyoInteract Studio", on_shutdown=[_stop_viewer, jobs.close])

    def _paired(token: str | None) -> bool:
        return pairing_code is None or secrets.compare_digest(token or "", pairing_code)

    # Middlewares added later wrap earlier ones: pairing check, then CORS (so a refusal is
    # readable by the page), gzip, and outermost the local-network preflight answer.
    @app.middleware("http")
    async def _pairing(request: Request, call_next):
        path = request.url.path
        guarded = path.startswith(("/api/", "/viewer")) and path != "/api/health"
        # Forwarding headers are client-controlled on the local listener. Proxy deployments
        # explicitly disable pairing with --behind-proxy instead of weakening this check.
        if (
            guarded
            and request.method != "OPTIONS"
            and _cross_origin(
                request.headers.get("origin"), request.headers.get("host")
            )
        ):
            if not _paired(request.headers.get("x-studio-token")):
                return JSONResponse(
                    {
                        "detail": "Enter the pairing code that myo_studio serve printed in its terminal."
                    },
                    status_code=401,
                )
        return await call_next(request)

    if allow_origins:
        # Starlette >= 1.x answers Chrome's local-network preflight itself and refuses it (400) unless allowed.
        pna = (
            {"allow_private_network": True}
            if "allow_private_network"
            in inspect.signature(CORSMiddleware.__init__).parameters
            else {}
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            **pna,
        )
    app.add_middleware(GZipMiddleware, minimum_size=2048)  # scene meshes and JSON

    @app.middleware("http")
    async def _local_network(request: Request, call_next):
        # Chrome asks before a public page may reach a server on this machine (Private / Local
        # Network Access); the answer is only given to origins CORS allows.
        response = await call_next(request)
        if (
            request.headers.get("access-control-request-private-network") == "true"
            and request.headers.get("origin") in allow_origins
        ):
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    user_presets = root / "logs" / "studio" / "presets"
    user_presets.mkdir(parents=True, exist_ok=True)
    device = _device()

    def _get(fn, *args):
        try:
            return fn(*args)
        except KeyError as exc:
            raise HTTPException(404, f"not found: {exc}") from exc

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/scenes/{name}", include_in_schema=False)
    def scene(name: str) -> Response:
        """3D scene of a body model: ``<model>.json`` and its mesh buffer ``<model>.bin``."""
        from myoInteract.studio.scene import scene_for  # noqa: PLC0415

        model, _, ext = name.rpartition(".")
        if ext not in ("json", "bin"):
            raise HTTPException(404, "not a scene file")
        text, buffer = _get(scene_for, model)
        if ext == "json":
            return Response(text, media_type="application/json")
        return Response(buffer, media_type="application/octet-stream")

    @app.post("/api/export/mjcf")
    def export_mjcf(body: PresetBody) -> Response:
        """The task as a MuJoCo model zip (MJCF + meshes), e.g. for MuJoCo's Unity plug-in."""
        from myoInteract.studio.export import task_mjcf_zip  # noqa: PLC0415

        _, config = resolve_preset(body.preset)
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", body.name).strip("_") or "task"
        return Response(
            task_mjcf_zip(config, name),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
        )

    @app.post("/api/figma")
    def figma_frame(body: FigmaBody) -> dict:
        """A Figma frame (PNG) and its interactive elements; the token is used for this request only."""
        from myoInteract.studio.figma import FigmaError, fetch_frame  # noqa: PLC0415

        try:
            return fetch_frame(body.url, body.token)
        except FigmaError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/health")
    def health(request: Request) -> dict:
        if _cross_origin(
            request.headers.get("origin"), request.headers.get("host")
        ) and not _paired(request.headers.get("x-studio-token")):
            return {
                "ok": True,
                "pairing": True,
            }  # a hosted page learns no more than that a server is here
        try:
            ver = version("myoInteract")
        except PackageNotFoundError:
            ver = "dev"
        return {
            "ok": True,
            "version": ver,
            "backend": "local",
            "data_root": str(root),
            "active_runs": runs.active(),
            **device,
        }

    @app.get("/api/options")
    def options() -> dict:
        from myoInteract.myointeract_hydra_cli import BM_MODEL_MAP  # noqa: PLC0415

        return {"models": list(BM_MODEL_MAP), "max_runs": max_runs}

    @app.get("/api/presets")
    def presets() -> list[dict]:
        bundled = [
            {"name": p.stem, "source": "bundled"}
            for p in sorted(BUNDLED_PRESETS.glob("*.json"))
        ]
        return [
            {"name": p.stem, "source": "user"}
            for p in sorted(user_presets.glob("*.json"))
        ] + bundled

    @app.get("/api/presets/{name}")
    def preset(name: str) -> dict:
        for folder in (user_presets, BUNDLED_PRESETS):
            if _NAME.match(name) and (folder / f"{name}.json").is_file():
                return json.loads((folder / f"{name}.json").read_text())
        raise HTTPException(404, f"no preset {name!r}")

    @app.post("/api/presets")
    def save_preset(body: PresetBody) -> dict:
        if not _NAME.match(body.name):
            raise HTTPException(
                422, "preset names use letters, digits, '.', '_' and '-'"
            )
        resolve_preset(body.preset)
        (user_presets / f"{body.name}.json").write_text(
            json.dumps(body.preset, indent=1)
        )
        return {"name": body.name, "source": "user"}

    @app.post("/api/config/validate")
    def validate(body: PresetBody) -> dict:
        overrides, resolved = resolve_preset(body.preset)
        rl = resolved["rl"]
        return {
            "ok": True,
            "overrides": overrides,
            "iterations": rl["num_timesteps"]
            // max(1, rl["num_envs"] * rl["unroll_length"]),
            "ctrl_dt": resolved["env"]["ctrl_dt"],
        }

    @app.post("/api/preview")
    def preview(body: PresetBody) -> dict:
        resolve_preset(body.preset)
        return jobs.start(
            "preview", ["--preset", "{dir}/preset.json"], {"preset.json": body.preset}
        )

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return runs.list()

    @app.post("/api/runs")
    def start_run(body: RunBody) -> dict:
        overrides, resolved = resolve_preset(body.preset)
        wandb = None
        if body.wandb:
            if not wandb_account.status().get("connected"):
                raise HTTPException(
                    409,
                    "W&B is not connected on this backend: add an API key in the Training step",
                )
            wandb = {"project": body.wandb.project, "entity": body.wandb.entity}
        env = (
            wandb_account.run_env(wandb["project"], wandb["entity"]) if wandb else None
        )
        # Resolve outside the lock, then reserve capacity and spawn atomically.
        # Concurrent requests must not both observe the same available GPU slot.
        with admission_lock:
            if len(runs.active()) >= max_runs:
                raise HTTPException(
                    409,
                    "a training run is already active; stop it first or start the server with --max-runs",
                )
            return runs.start(
                body.preset, body.name, overrides, resolved, env=env, wandb=wandb
            )

    @app.get("/api/wandb")
    def wandb_status() -> dict:
        """The W&B login runs would use: user, default entity and teams (never the key)."""
        return wandb_account.status()

    @app.post("/api/wandb")
    def wandb_login(body: WandbLoginBody) -> dict:
        from myoInteract.studio.wandb_account import WandbError  # noqa: PLC0415

        try:
            return wandb_account.login(body.api_key, body.remember)
        except WandbError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/wandb")
    def wandb_logout() -> dict:
        return wandb_account.logout()

    # Before the catch-all run route below, which would otherwise swallow ".../log".
    @app.get("/api/runs/{rid:path}/log", response_class=PlainTextResponse)
    def run_log(rid: str) -> str:
        """The full training log (stdout and stderr of myo_train) of a Studio run."""
        path = _get(runs.log_path, rid)
        if path is None:
            raise HTTPException(
                404, "this run was not started from Studio, so Studio has no log of it"
            )
        return path.read_text(errors="replace")

    @app.get("/api/runs/{rid:path}")
    def get_run(rid: str) -> dict:
        return _get(runs.get, rid)

    @app.delete("/api/runs/{rid:path}")
    def stop_run(rid: str) -> dict:
        return _get(runs.stop, rid)

    def _checked_run(run: str, checkpoint: str) -> Path:
        log_dir = _get(runs.log_dir, run)
        if log_dir is None or not (log_dir / "original_hydra_config.json").exists():
            raise HTTPException(
                422,
                "this run has no original_hydra_config.json to rebuild its task from",
            )
        if (
            not _NAME.match(checkpoint)
            or not _get(_contained_path, log_dir, checkpoint).is_file()
        ):
            raise HTTPException(404, f"no checkpoint {checkpoint!r} in this run")
        return log_dir

    @app.post("/api/evaluations")
    def evaluate(body: EvaluationBody) -> dict:
        log_dir = _checked_run(body.run, body.checkpoint)
        args = [
            "--log-dir",
            str(log_dir),
            "--checkpoint",
            body.checkpoint,
            "--episodes",
            str(body.episodes),
            "--video-seconds",
            str(body.video_seconds),
        ]
        if body.preset is None:
            return jobs.start("evaluate", args)
        task = shifted_task(log_dir, body.preset)
        return jobs.start(
            "evaluate", [*args, "--task", "{dir}/task.json"], {"task.json": task}
        )

    @app.post("/api/viewer")
    def open_viewer(body: ViewerBody) -> dict:
        """Live 3D feed (Studio's scene) or viser viewer of a preset (zero excitation) or a run; replaces the open one."""
        if body.run and body.follow:
            log_dir = _get(runs.log_dir, body.run)
            if log_dir is None or not (log_dir / "original_hydra_config.json").exists():
                raise HTTPException(
                    422,
                    "this run has no original_hydra_config.json to rebuild its task from",
                )
            args, inputs = ["--log-dir", str(log_dir), "--follow"], None
        elif body.run:
            log_dir = _checked_run(body.run, body.checkpoint)
            args, inputs = (
                ["--log-dir", str(log_dir), "--checkpoint", body.checkpoint],
                None,
            )
        elif body.preset is not None:
            resolve_preset(body.preset)
            args, inputs = (
                ["--preset", "{dir}/preset.json"],
                {"preset.json": body.preset},
            )
        else:
            raise HTTPException(422, "send a preset or a run")
        if (
            body.run and body.preset is not None
        ):  # the run's policy on the editor's setup
            args = [*args, "--task", "{dir}/task.json"]
            inputs = {"task.json": shifted_task(log_dir, body.preset)}
        kind = "viewer" if body.kind == "viser" else "stream"
        key = json.dumps([kind, args, inputs], sort_keys=True)
        with viewer_lock:
            # The same view again (e.g. the page re-opening the Evaluate step): keep the running viewer.
            if viewer_job["id"] and viewer_job["key"] == key:
                current = jobs.get(viewer_job["id"])
                if current and current.get("alive"):
                    return current
            if viewer_job["id"]:
                jobs.stop(viewer_job["id"])
                time.sleep(1.0)  # let the old viser server release the port
            if _port_in_use(viewer_port):
                raise HTTPException(
                    409,
                    f"Port {viewer_port} for the 3D viewer is in use, probably by a viewer left over from an earlier "
                    f"Studio server (`lsof -i :{viewer_port}` shows it). Stop that process or start Studio with "
                    "--viewer-port.",
                )
            job = jobs.start(
                kind, [*args, "--host", "127.0.0.1", "--port", str(viewer_port)], inputs
            )
            viewer_job["id"], viewer_job["key"] = job["id"], key
            return job

    @app.get("/api/viewer")
    def viewer_status() -> dict:
        return (
            _get(jobs.get, viewer_job["id"])
            if viewer_job["id"]
            else {"id": None, "status": "none", "alive": False}
        )

    @app.delete("/api/viewer")
    def close_viewer() -> dict:
        with viewer_lock:
            if viewer_job["id"]:
                jobs.stop(viewer_job["id"])
        return {"id": viewer_job["id"], "status": "stopped"}

    # The viewer is proxied under /viewer/ so the browser needs only Studio's port (SSH tunnels,
    # VS Code port forwarding); viser itself listens on viewer_port on localhost.
    @app.get("/viewer", include_in_schema=False)
    def viewer_slash() -> RedirectResponse:
        return RedirectResponse("/viewer/")

    @app.get("/viewer/{path:path}", include_in_schema=False)
    async def viewer_http(path: str, request: Request) -> Response:
        import httpx  # noqa: PLC0415

        url = f"http://127.0.0.1:{viewer_port}/{path}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(url, params=request.query_params)
        except httpx.HTTPError:
            return Response(
                "The 3D viewer is not running (start it from Studio).", status_code=503
            )
        keep = {
            k: v
            for k, v in r.headers.items()
            if k.lower() in ("content-type", "cache-control", "etag")
        }
        return Response(r.content, status_code=r.status_code, headers=keep)

    @app.websocket("/viewer")
    @app.websocket("/viewer/{path:path}")
    async def viewer_ws(websocket: WebSocket, path: str = "") -> None:
        import asyncio  # noqa: PLC0415

        import websockets  # noqa: PLC0415

        # Websockets skip the HTTP middlewares; a hosted page sends the pairing code as ?token=.
        if _cross_origin(
            websocket.headers.get("origin"), websocket.headers.get("host")
        ) and not (
            websocket.headers.get("origin") in allow_origins
            and _paired(websocket.query_params.get("token"))
        ):
            await websocket.close(code=1008)
            return

        protocols = websocket.scope.get("subprotocols") or []
        try:
            upstream = await websockets.connect(
                f"ws://127.0.0.1:{viewer_port}/{path}",
                subprotocols=protocols or None,
                max_size=None,
                open_timeout=10,
            )
        except (OSError, websockets.WebSocketException):
            await websocket.close(code=1011)
            return
        await websocket.accept(subprotocol=upstream.subprotocol)

        async def to_viser() -> None:
            try:
                while True:
                    msg = await websocket.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    await upstream.send(
                        msg["bytes"] if msg.get("bytes") is not None else msg["text"]
                    )
            except WebSocketDisconnect:
                pass

        async def to_browser() -> None:
            async for data in upstream:
                if isinstance(data, bytes):
                    await websocket.send_bytes(data)
                else:
                    await websocket.send_text(data)

        tasks = [asyncio.create_task(to_viser()), asyncio.create_task(to_browser())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            task.cancel()
        await upstream.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed by the browser

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str) -> dict:
        return _get(jobs.get, jid)

    @app.get("/api/jobs/{jid}/files/{name}")
    def job_file(jid: str, name: str) -> FileResponse:
        return FileResponse(_get(jobs.path, jid, name))

    return app


def localhost_certificate(folder: Path) -> tuple[Path, Path]:
    """A self-signed certificate for ``localhost`` (made once in ``folder``, then reused).

    Safari lets an https page reach ``https://localhost`` but never ``http://localhost``; the browser
    has to trust this certificate once (open the server's address and accept it, or add the ``.pem``
    to the keychain as trusted). Valid for 397 days, the longest Apple accepts.
    """
    import datetime  # noqa: PLC0415
    import ipaddress  # noqa: PLC0415

    from cryptography import x509  # noqa: PLC0415
    from cryptography.hazmat.primitives import hashes, serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric import ec  # noqa: PLC0415
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID  # noqa: PLC0415

    cert, key = folder / "localhost.pem", folder / "localhost-key.pem"
    if cert.exists() and key.exists():
        loaded = x509.load_pem_x509_certificate(cert.read_bytes())
        if loaded.not_valid_after_utc > datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(days=7):
            return cert, key
    folder.mkdir(parents=True, exist_ok=True)
    private = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "MyoInteract Studio (localhost)")]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=397))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(private, hashes.SHA256())
    )
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key.chmod(0o600)
    return cert, key


def main() -> None:
    """Console entry point ``myo_studio``."""
    p = argparse.ArgumentParser(prog="myo_studio", description="MyoInteract Studio")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve Studio and its backend")
    s.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to listen on (default: localhost only)",
    )
    s.add_argument("--port", type=int, default=8000)
    s.add_argument(
        "--root",
        type=Path,
        default=None,
        help="data directory (default: $MYOINTERACT_HOME or the checkout)",
    )
    s.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help=f"extra origin of a hosted Studio page allowed to use this server (repeatable; {', '.join(HOSTED_ORIGINS)} always is)",
    )
    s.add_argument(
        "--pairing-code",
        default=None,
        help="code hosted pages must send (default: a new random one per start)",
    )
    s.add_argument(
        "--behind-proxy",
        action="store_true",
        help="Studio is reached through a proxy that restricts who gets in and may rewrite the Host header "
        "(Colab, a private Hugging Face Space): no pairing code is asked for",
    )
    s.add_argument(
        "--max-runs", type=int, default=1, help="training runs at the same time"
    )
    s.add_argument(
        "--viewer-port",
        type=int,
        default=8080,
        help="local port of the 3D viewer; Studio proxies it under /viewer/",
    )
    s.add_argument("--no-browser", action="store_true")
    s.add_argument(
        "--https",
        action="store_true",
        help="serve over HTTPS (needed for hosted pages in Safari); without --cert/--key a self-signed localhost certificate is made",
    )
    s.add_argument(
        "--cert",
        type=Path,
        default=None,
        help="certificate for --https (e.g. from mkcert)",
    )
    s.add_argument("--key", type=Path, default=None, help="private key for --https")
    args = p.parse_args(sys.argv[1:])

    import uvicorn  # noqa: PLC0415

    root = (args.root or data_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    code = None if args.behind_proxy else args.pairing_code or secrets.token_hex(3)
    origins = list(dict.fromkeys([*HOSTED_ORIGINS, *args.allow_origin]))
    app = create_app(
        root,
        origins,
        args.max_runs,
        viewer_port=args.viewer_port,
        pairing_code=code,
        warm_worker=True,
    )
    tls = {}
    if args.https or args.cert:
        if bool(args.cert) != bool(args.key):
            p.error("--cert and --key go together")
        cert, key = (
            (args.cert, args.key)
            if args.cert
            else localhost_certificate(root / "logs" / "studio" / "tls")
        )
        tls = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    scheme = "https" if tls else "http"
    url = f"{scheme}://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"MyoInteract Studio: {url}  (data: {root})", flush=True)
    if code is None:
        print(
            "No pairing code is asked for (--behind-proxy): whoever reaches this server can use it.",
            flush=True,
        )
    else:
        print(
            f"Hosted Studio pages ({', '.join(origins)}): connect to {url} with pairing code {code}",
            flush=True,
        )
    if tls and not args.cert:
        print(
            f"Certificate: {tls['ssl_certfile']}. Trust it once in your browser: open {url}, accept the warning "
            "(Safari: Show Details -> visit this website), or add the file to your keychain as trusted.",
            flush=True,
        )
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **tls)


if __name__ == "__main__":
    main()
