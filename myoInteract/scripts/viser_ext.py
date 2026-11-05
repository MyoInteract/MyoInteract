"""Customised ViserPlayViewer with camera-state preservation across re-renders.

Background
----------
viser's ``ViserServer.on_client_connect`` (viser/_viser.py) not only queues the
callback for *future* connections — it also immediately submits it to a thread
pool for every *already-connected* client.  This means that when
``ViserPlayViewer.setup()`` calls ``create_scene_gui()``, which registers an
``on_client_connect`` handler that resets the camera to a computed default
offset, that reset fires on the live browser session.

``CameraRestoringViserViewer`` overrides ``setup()`` to capture the saved
camera state and, once the parent's handler has been submitted, spawn a short
daemon thread (50 ms delay) that restores the saved position.  The 50 ms gap
is far longer than the sub-millisecond WebSocket round-trip needed for the
camera-reset message, so by the time we restore the camera is settled.

``server.initial_camera`` is also updated so that any *new* connections (e.g.
a browser tab refresh) open at the same saved angle.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch  # noqa: F401  (loaded before mjlab, as in the training entry points)
import viser
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import ViserPlayViewer

from myoInteract.register_mjlab_myointeract_tasks import (
    _muscles_fatigue,
)

_TRAINING_METRIC_PREFIXES = (
    "Episode_Metrics/",
    "Train/mean_reward",
    "Train/mean_episode_length",
)


def _viser_to_mjv_camera(position: Any, lookat: Any) -> Any:
    """Convert a viser camera (position + look_at in MuJoCo world frame) to MjvCamera.

    client.camera.position/look_at are already in MuJoCo world coordinates.
    The free-camera conventions observed empirically:
      azimuth=0   → camera in +X direction from lookat
      azimuth=90  → camera in −Y direction from lookat
      positive elevation → camera BELOW lookat (negative Z offset)
    """
    import mujoco
    import numpy as np

    cam = mujoco.MjvCamera()
    diff = np.asarray(position, dtype=float) - np.asarray(lookat, dtype=float)
    distance = float(np.linalg.norm(diff))
    cam.lookat[:] = np.asarray(lookat, dtype=float)
    cam.distance = max(distance, 1e-3)
    if distance > 1e-6:
        d = diff / distance
        cam.elevation = float(np.degrees(np.arcsin(np.clip(-d[2], -1.0, 1.0))))
        # MuJoCo convention: azimuth=0 → camera in the −Y direction.
        # Forward: d = [cos(el)*sin(az), -cos(el)*cos(az), sin(el)]
        # Inverse: az = atan2(d[0], -d[1])
        cam.azimuth = float(np.degrees(np.arctan2(d[0], -d[1])) + 90)
    return cam


_TRAINING_METRICS_UPDATE_INTERVAL_S = 5


def _read_tfevents_history(log_dir: str) -> dict[str, tuple]:
    """Return {tag: (steps, values)} for display-relevant scalars in log_dir.

    Uses size_guidance=0 (unlimited) so the full training history is returned.
    Each ScalarEvent has .step (iteration) and .value (float).
    Tags with fewer than 2 events are omitted (uplot needs at least 2 points).
    """
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    result = {}
    for tag in ea.Tags().get("scalars", []):
        if any(tag.startswith(p) for p in _TRAINING_METRIC_PREFIXES):
            events = ea.Scalars(tag)
            if len(events) >= 2:
                result[tag] = (
                    np.array([e.step for e in events], dtype=np.float64),
                    np.array([e.value for e in events], dtype=np.float64),
                )
    return result


class CameraRestoringViserViewer(ViserPlayViewer):
    """ViserPlayViewer that preserves the user's camera angle across re-renders.

    Pass ``post_setup_camera`` as a ``{"position": np.ndarray, "look_at":
    np.ndarray}`` dict captured from the connected client just before the
    previous viewer was torn down.  If ``None`` (first render), the default
    camera configured via ``env_cfg.viewer`` is used unchanged.

    Also provides a "Recording" panel in the viser GUI for toggling MP4 capture.
    Videos are written to ``viser_videos/{record_name}-{timestamp}.mp4`` relative
    to the current working directory using ``mediapy.write_video``.
    """

    def __init__(
        self,
        *args: Any,
        post_setup_camera: dict | None = None,
        record_name: str = "",
        **kwargs: Any,
    ) -> None:
        self._post_setup_camera = post_setup_camera
        self._record_name = record_name
        self._recording = False
        self._recorded_frames: list = []
        self._record_lock = threading.Lock()
        self._record_target_fps = 30.0
        self._record_thread: Any = None
        self._mj_renderer: Any = None
        self._render_mj_data: Any = None
        self._record_button: Any = None
        self._record_status_text: Any = None
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        # Intercept the tab group so we can inject Recording controls at the very
        # top of the "Visualization" tab (before create_overlay_gui() adds its content).
        _gui = self._server.gui
        _orig_atg = _gui.add_tab_group

        def _patched_atg(*args: Any, **kwargs: Any) -> Any:
            tab_group = _orig_atg(*args, **kwargs)
            _orig_at = tab_group.add_tab

            def _patched_at(label: str, *targs: Any, **tkwargs: Any) -> Any:
                ctx = _orig_at(label, *targs, **tkwargs)
                if label == "Visualization":
                    _real_ctx = ctx
                    _self = self

                    class _VizTabCtx:
                        def __enter__(inner: Any) -> Any:
                            result = _real_ctx.__enter__()
                            _self._setup_recording_ui()
                            return result

                        def __exit__(inner: Any, *exc: Any) -> Any:
                            return _real_ctx.__exit__(*exc)

                    return _VizTabCtx()
                return ctx

            tab_group.add_tab = _patched_at
            return tab_group

        # Intercept add_button so a "Reset Fatigue" button lands immediately after
        # mjlab's own "Reset Environment" button, inside the same "Simulation"
        # folder -- add_button is called synchronously while that folder's
        # container context is still active (see GuiFolderHandle.__enter__/__exit__
        # in viser/_gui_handles.py: the "current container" is a push/pop stack,
        # so anything added here lands in the exact same container as the call we
        # intercepted). Only added when the loaded env actually has fatigue
        # enabled (mirrors "Reset Environment" being unconditional, but this one
        # is meaningless otherwise).
        _orig_add_button = _gui.add_button

        def _patched_add_button(label: str, *bargs: Any, **bkwargs: Any) -> Any:
            handle = _orig_add_button(label, *bargs, **bkwargs)
            if (
                label == "Reset Environment"
                and _muscles_fatigue(self.env.unwrapped) is not None
            ):
                fatigue_button = _orig_add_button("Reset Fatigue")

                @fatigue_button.on_click
                def _(_: Any) -> None:
                    fatigue = _muscles_fatigue(self.env.unwrapped)
                    if fatigue is not None:
                        fatigue.reset(None)

            return handle

        _gui.add_tab_group = _patched_atg
        _gui.add_button = _patched_add_button
        try:
            super().setup()
        finally:
            _gui.add_tab_group = _orig_atg
            _gui.add_button = _orig_add_button

        if self._post_setup_camera is not None:
            camera_state = self._post_setup_camera
            server = self._server

            # Update initial_camera now (synchronous, no race) so future new
            # connections open at the saved angle.
            try:
                server.initial_camera.position = camera_state["position"]
                server.initial_camera.look_at = camera_state["look_at"]
            except Exception:
                pass

            # super().setup() registered an on_client_connect handler that resets
            # the camera for existing clients via the server's thread pool.
            # We wait 50 ms (>> WebSocket round-trip) so that message has been
            # sent, then push the restore on top.
            def _delayed_restore() -> None:
                time.sleep(0.05)
                for client in server.get_clients().values():
                    try:
                        client.camera.position = camera_state["position"]
                        client.camera.look_at = camera_state["look_at"]
                    except Exception:
                        pass

            threading.Thread(target=_delayed_restore, daemon=True).start()

    def _setup_recording_ui(self) -> None:
        """Create the recording button and status text inside the current GUI container."""
        self._record_button = self._server.gui.add_button(
            "⏺ Start Recording", color="red"
        )
        self._record_status_text = self._server.gui.add_text("Status", initial_value="")

        @self._record_button.on_click
        def _toggle_recording(_: Any) -> None:
            with self._record_lock:
                if not self._recording:
                    self._recording = True
                    self._recorded_frames = []
                    self._record_button.label = "⏹ Stop Recording"
                    self._record_status_text.value = "Recording…"
                    self._record_thread = threading.Thread(
                        target=self._capture_loop, daemon=True
                    )
                    self._record_thread.start()
                else:
                    self._recording = False
                    self._record_button.label = "⏺ Start Recording"
                    self._record_status_text.value = "Stopping…"

    # ── Frame capture ─────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Render frames in-process using mujoco.Renderer at target FPS, then save.

        Camera angle is synced from the first connected viser client every frame,
        converted from (position, look_at) to MuJoCo's (azimuth, elevation, distance,
        lookat) parameterisation.  This approach eliminates the WebSocket round-trip
        of client.get_render() and can easily achieve 30+ fps.
        """
        import mujoco

        interval = 1.0 / self._record_target_fps
        t_start = time.perf_counter()

        def _get_cam() -> Any:
            import numpy as np

            try:
                clients = self._server.get_clients()
                if not clients:
                    return None
                c = next(iter(clients.values()))
                pos_mj = np.array(c.camera.position, dtype=float)
                look_mj = np.array(c.camera.look_at, dtype=float)
                # client.camera coords are already in MuJoCo world frame.
                # _scene_offset shifts all viser meshes by -tracked_body_pos so the
                # tracked body appears at the scene origin; subtract it to recover
                # actual MuJoCo world positions before passing to the renderer.
                scene_offset = getattr(
                    getattr(self, "_scene", None), "_scene_offset", None
                )
                if scene_offset is not None:
                    pos_mj = pos_mj - scene_offset
                    look_mj = look_mj - scene_offset
                # Sync field of view from the browser (stored in radians) to the
                # MuJoCo renderer (degrees).  vis.global_.fovy is the vertical FoV.
                fov_rad = float(c.camera.fov)
                if fov_rad > 0:
                    self.env.unwrapped.sim.mj_model.vis.global_.fovy = np.degrees(
                        fov_rad
                    )
                return _viser_to_mjv_camera(pos_mj, look_mj)
            except Exception:
                return None

        def _ensure_renderer() -> Any:
            if self._mj_renderer is None:
                sim = self.env.unwrapped.sim
                # Match the browser viewport resolution if a client is connected.
                h, w = 720, 1280
                clients = self._server.get_clients()
                if clients:
                    c = next(iter(clients.values()))
                    ch, cw = c.camera.image_height, c.camera.image_width
                    if ch > 0 and cw > 0:
                        h, w = ch, cw
                self._mj_renderer = mujoco.Renderer(sim.mj_model, height=h, width=w)
                # Dedicated MjData for rendering (sim.mj_data is stale — physics
                # runs entirely on GPU via Warp and never updates the CPU MjData).
                self._render_mj_data = mujoco.MjData(sim.mj_model)
            return self._mj_renderer

        def _sync_render_data() -> Any:
            """Copy env 0's GPU spatial state directly into a CPU MjData.

            We copy the pre-computed spatial quantities (xpos, xmat, geom_xpos, …)
            directly from the GPU simulation rather than running mj_forward from
            qpos.  This is necessary because task-managed objects (e.g. pointing
            targets) have their positions set directly on the GPU and are not
            recoverable from qpos/mocap alone.
            """
            sim = self.env.unwrapped.sim
            db = sim.data  # WarpBridge — shapes (nworld, ...)
            mj_data = self._render_mj_data

            # Body kinematics: covers all rigid bodies including task targets.
            mj_data.xpos[:] = db.xpos.cpu().numpy()[0]  # (nbody, 3)
            mj_data.xquat[:] = db.xquat.cpu().numpy()[0]  # (nbody, 4)
            mj_data.xmat[:] = db.xmat.cpu().numpy()[0].reshape(-1, 9)  # (nbody, 9)

            # Geom world positions (used directly by the renderer for geom drawing).
            mj_data.geom_xpos[:] = db.geom_xpos.cpu().numpy()[0]  # (ngeom, 3)
            mj_data.geom_xmat[:] = db.geom_xmat.cpu().numpy()[0].reshape(-1, 9)

            # Site world positions (tendons, sensors, site markers).
            if sim.mj_model.nsite > 0:
                mj_data.site_xpos[:] = db.site_xpos.cpu().numpy()[0]  # (nsite, 3)
                mj_data.site_xmat[:] = db.site_xmat.cpu().numpy()[0].reshape(-1, 9)

            return mj_data

        while True:
            with self._record_lock:
                if not self._recording:
                    break
            t0 = time.perf_counter()
            try:
                renderer = _ensure_renderer()
                cam = _get_cam()
                mj_data = _sync_render_data()
                renderer.update_scene(mj_data, camera=cam)
                frame = renderer.render()  # (H, W, 3) uint8
                with self._record_lock:
                    self._recorded_frames.append(frame)
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        t_end = time.perf_counter()
        with self._record_lock:
            frames = list(self._recorded_frames)
        duration = t_end - t_start
        actual_fps = (
            len(frames) / duration
            if (duration > 0 and frames)
            else self._record_target_fps
        )
        if self._record_status_text is not None:
            self._record_status_text.value = f"Saving {len(frames)} frames…"
        self._write_video(frames, actual_fps)

    def _write_video(self, frames: list, fps: float | None = None) -> None:
        if not frames:
            if self._record_status_text is not None:
                self._record_status_text.value = "No frames captured."
            return
        import mediapy
        from datetime import datetime
        from pathlib import Path

        output_dir = Path("viser_videos")
        output_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = self._record_name or "recording"
        path = output_dir / f"{name}-{ts}.mp4"
        effective_fps = fps if fps is not None else self._record_target_fps
        mediapy.write_video(str(path), frames, fps=effective_fps)
        if self._record_status_text is not None:
            self._record_status_text.value = f"Saved: {path}"
        print(f"[INFO] Video saved: {path} ({effective_fps:.1f} fps)")
        # Release the offscreen renderer until the next recording session.
        if self._mj_renderer is not None:
            try:
                self._mj_renderer.close()
            except Exception:
                pass
            self._mj_renderer = None
        self._render_mj_data = None


_POLL_INTERVAL_S = 15


class PollingViserViewer(CameraRestoringViserViewer):
    """CameraRestoringViserViewer that polls for new checkpoints in the background.

    Every _POLL_INTERVAL_S seconds the checkpoint manager's fetch_available() is
    called.  When previously-unseen checkpoints appear a green "Load: <name>" button
    appears below the checkpoint dropdown in the Checkpoints tab.  Notifications are
    removed individually: clicking a button loads that specific checkpoint; selecting
    a checkpoint from the dropdown removes its notification if one exists.  No bulk
    clearing happens.
    """

    def __init__(self, *args: Any, log_dir: str | None = None, **kwargs: Any) -> None:
        self._log_dir = log_dir
        self._training_metrics_md: Any = None
        self._training_metrics_tab_id: str | None = None
        self._training_metric_plots: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        # Intercept the tab group created by ViserPlayViewer.setup() so we can
        # add "Training Metrics" as a sibling of "Checkpoints", not a new group.
        _orig = self._server.gui.add_tab_group
        _captured: list = []

        def _intercept(*args: Any, **kwargs: Any) -> Any:
            handle = _orig(*args, **kwargs)
            _captured.append(handle)
            return handle

        self._server.gui.add_tab_group = _intercept
        try:
            super().setup()
        finally:
            self._server.gui.add_tab_group = _orig

        if self._ckpt_mgr is not None:
            # Store the Checkpoints tab container ID so _add_notification() can
            # inject new elements into the same tab dynamically from the poll thread.
            self._ckpt_tab_container_id = self._ckpt_dropdown._impl.parent_container_id
            # Keyed by checkpoint filename so we can remove by name.
            self._ckpt_notification_handles: dict[str, Any] = {}

            # When the user selects from the dropdown, remove only the matching button.
            @self._ckpt_dropdown.on_update
            def _(_: Any) -> None:
                if self._ckpt_user_event.is_set():
                    name = self._ckpt_dropdown.value.split("  (")[0]
                    self._remove_notification_for(name)

            threading.Thread(target=self._poll_loop, daemon=True).start()

        if self._log_dir is not None and _captured:
            with _captured[0].add_tab("Training Metrics", icon=viser.Icon.CHART_BAR):
                self._training_metrics_tab_id = self._server.gui._get_container_uuid()
                self._training_metrics_md = self._server.gui.add_markdown(
                    "<small><em>Awaiting first metrics update (≤5s)…</em></small>"
                )
            threading.Thread(target=self._training_metrics_loop, daemon=True).start()

    # ── training metrics panel ────────────────────────────────────────────────

    def _training_metrics_loop(self) -> None:
        """Read the TFEvents log every 30 s and update per-metric uplot history panels."""
        while True:
            time.sleep(_TRAINING_METRICS_UPDATE_INTERVAL_S)
            if self._log_dir is None or self._training_metrics_tab_id is None:
                continue
            try:
                history = _read_tfevents_history(self._log_dir)
            except Exception:
                continue
            if not history:
                continue

            # Remove placeholder markdown on first data arrival.
            if self._training_metrics_md is not None:
                try:
                    self._training_metrics_md.remove()
                except Exception:
                    pass
                self._training_metrics_md = None

            for tag, (steps, values) in sorted(history.items()):
                label = tag.split("/", 1)[-1]
                if tag not in self._training_metric_plots:
                    try:
                        prev_id = self._server.gui._get_container_uuid()
                        self._server.gui._set_container_uuid(
                            self._training_metrics_tab_id
                        )
                        handle = self._server.gui.add_uplot(
                            data=(steps, values),
                            series=(
                                viser.uplot.Series(label="Iteration"),
                                viser.uplot.Series(
                                    label=label, stroke="#1f77b4", width=2
                                ),
                            ),
                            scales={
                                "x": viser.uplot.Scale(time=False, auto=True),
                                "y": viser.uplot.Scale(auto=True),
                            },
                            legend=viser.uplot.Legend(show=False),
                            title=label,
                            aspect=2.0,
                        )
                        self._server.gui._set_container_uuid(prev_id)
                        self._training_metric_plots[tag] = handle
                    except Exception:
                        pass
                else:
                    try:
                        self._training_metric_plots[tag].data = (steps, values)
                    except Exception:
                        pass

    # ── background polling ────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        known: set[str] = set()
        first = True
        while True:
            time.sleep(_POLL_INTERVAL_S)
            try:
                entries = self._ckpt_mgr.fetch_available()
            except Exception:
                continue
            names = {n for n, _ in entries}
            new = names - known
            if not first and new:
                for checkpoint_name in sorted(new):
                    self._add_notification(checkpoint_name)
                self._actions.append((ViewerAction.FETCH_CHECKPOINT, "latest_silent"))
            known = names
            first = False

    # ── per-checkpoint notification buttons ──────────────────────────────────

    def _add_notification(self, checkpoint_name: str) -> None:
        """Add a load button into the Checkpoints tab for a newly available checkpoint."""
        if checkpoint_name in self._ckpt_notification_handles:
            return  # already shown
        try:
            prev_id = self._server.gui._get_container_uuid()
            self._server.gui._set_container_uuid(self._ckpt_tab_container_id)
            btn = self._server.gui.add_button(
                f"Load: {checkpoint_name}",
                color="green",
                hint="Click to load this checkpoint",
            )
            self._ckpt_notification_handles[checkpoint_name] = btn
            self._server.gui._set_container_uuid(prev_id)

            @btn.on_click
            def _(_: Any) -> None:
                self._load_checkpoint_from_notification(checkpoint_name)
        except Exception:
            pass

    def _remove_notification_for(self, checkpoint_name: str) -> None:
        """Remove the notification button for one specific checkpoint, if present."""
        btn = self._ckpt_notification_handles.pop(checkpoint_name, None)
        if btn is not None:
            try:
                btn.remove()
            except Exception:
                pass

    def _load_checkpoint_from_notification(self, checkpoint_name: str) -> None:
        """Remove this notification and load the checkpoint directly.

        Loads inline rather than via the action queue so that the parent's
        "selected" handler never calls ``dropdown.value = cur`` while
        ``_ckpt_user_event`` is set — which would fire ``on_update`` and
        accidentally remove other notification buttons.
        """
        self._remove_notification_for(checkpoint_name)
        if self._ckpt_mgr is None:
            return
        labels = list(self._ckpt_dropdown.options)
        matching = next(
            (lbl for lbl in labels if lbl.startswith(checkpoint_name)),
            checkpoint_name,
        )
        # Keep user_event cleared during the value assignment so the synchronous
        # on_update callback does not remove any other notification buttons.
        self._ckpt_user_event.clear()
        self._ckpt_dropdown.value = matching
        self._ckpt_user_event.set()
        if checkpoint_name != self._ckpt_mgr.current_name:
            print(f"[INFO]: Loading {checkpoint_name}...")
            self.policy = self._ckpt_mgr.load_checkpoint(checkpoint_name)
            self._ckpt_mgr.current_name = checkpoint_name
            self.reset_environment()
            print(f"[INFO]: Loaded {checkpoint_name}")

    # ── action handler ────────────────────────────────────────────────────────

    def _handle_custom_action(self, action: Any, payload: Any) -> bool:
        # "Use Latest" button: after super() loads, remove that checkpoint's notification.
        if action == ViewerAction.FETCH_CHECKPOINT and payload == "latest":
            result = super()._handle_custom_action(action, payload)
            self._remove_notification_for(self._ckpt_mgr.current_name)
            return result

        if action != ViewerAction.FETCH_CHECKPOINT or payload != "latest_silent":
            return super()._handle_custom_action(action, payload)

        if self._ckpt_mgr is None:
            return True
        entries = self._ckpt_mgr.fetch_available()
        if not entries:
            return True
        labels = [f"{n}  ({t})" if t else n for n, t in entries]
        # The displayed label may have a stale timestamp string, so find the
        # updated label for the same checkpoint name and restore it explicitly.
        # This prevents viser from resetting the dropdown to options[0] when the
        # current value is no longer in the new options list.
        displayed_name = self._ckpt_dropdown.value.split("  (")[0]
        restored_label = next(
            (lbl for lbl in labels if lbl.startswith(displayed_name)),
            self._ckpt_dropdown.value,
        )
        self._ckpt_user_event.clear()
        self._ckpt_dropdown.options = labels
        self._ckpt_dropdown.value = restored_label
        self._ckpt_user_event.set()
        # No auto-loading: user must click a notification button or select from dropdown.
        return True
