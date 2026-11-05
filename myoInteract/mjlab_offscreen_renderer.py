import mujoco
import numpy as np

from myosuite.utils.video_io import add_text_to_frame


def _prepare_mjlab_offscreen_renderer(_sim, env_cfg, width=None, height=None):
    _renderer = mujoco.Renderer(
        _sim.mj_model,
        height=height or env_cfg.viewer.height,
        width=width or env_cfg.viewer.width,
    )
    # Dedicated MjData for rendering (sim.mj_data is stale -- physics runs entirely on
    # GPU via Warp and never updates the CPU MjData), mirroring _capture_loop.
    _render_mj_data = mujoco.MjData(_sim.mj_model)
    _cam = mujoco.MjvCamera()
    _cam.lookat[:] = env_cfg.viewer.lookat
    _cam.distance = env_cfg.viewer.distance
    _cam.elevation = env_cfg.viewer.elevation
    _cam.azimuth = env_cfg.viewer.azimuth
    return _renderer, _render_mj_data, _cam


def _sync_render_data(_sim, _render_mj_data):
    # Copy env 0's GPU spatial state directly into the CPU MjData -- same
    # approach as viser_ext.py's _capture_loop._sync_render_data, needed because
    # task-managed objects (e.g. pointing targets) have positions set directly on
    # the GPU and aren't recoverable from qpos/mocap alone.
    db = _sim.data  # WarpBridge -- shapes (nworld, ...)
    _render_mj_data.xpos[:] = db.xpos[0].cpu().numpy()
    _render_mj_data.xquat[:] = db.xquat[0].cpu().numpy()
    _render_mj_data.xmat[:] = db.xmat[0].cpu().numpy().reshape(-1, 9)
    _render_mj_data.geom_xpos[:] = db.geom_xpos[0].cpu().numpy()
    _render_mj_data.geom_xmat[:] = db.geom_xmat[0].cpu().numpy().reshape(-1, 9)
    if _sim.mj_model.nsite > 0:
        _render_mj_data.site_xpos[:] = db.site_xpos[0].cpu().numpy()
        _render_mj_data.site_xmat[:] = db.site_xmat[0].cpu().numpy().reshape(-1, 9)

    # geom_rgba is a *model* field, not data -- but for any task that writes per-env
    # colors at runtime (e.g. SequentialTaskLogic highlighting the current target via
    # asset.data.model.geom_rgba), the offscreen
    # mujoco.Renderer above was built from _sim.mj_model, the single static CPU model
    # shared across envs -- it never sees those per-env writes on its own. Sync env 0's
    # row from the batched model (_sim.model is the same WarpBridge as _sim.data.model)
    # into it every frame so the rendered color always matches env 0's actual state.
    _sim.mj_model.geom_rgba[:] = _sim.model.geom_rgba[0].cpu().numpy()

    return _render_mj_data


def _draw_fatigue_status(
    frame: np.ndarray, fatigue_status: float, fatigue_type: str = "Maximum"
) -> np.ndarray:
    """Draw a bottom-center HUD fatigue bar onto *frame*."""

    out = np.asarray(frame, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    fatigue = float(np.clip(fatigue_status, 0.0, 1.0))

    bar_w = max(120, int(round(w * 0.28)))
    bar_h = max(12, int(round(h * 0.035)))
    x0 = (w - bar_w) // 2
    y0 = h - bar_h - 18
    x1 = x0 + bar_w
    y1 = y0 + bar_h

    out[max(0, y0 - 4) : min(h, y1 + 4), max(0, x0 - 4) : min(w, x1 + 4)] = (20, 20, 20)
    out[y0:y1, x0:x1] = (60, 60, 60)

    fill_w = int(round(bar_w * fatigue))
    if fill_w > 0:
        if fatigue < 0.4:
            fill = np.array((40, 220, 60), dtype=np.uint8)
        elif fatigue < 0.8:
            fill = np.array((240, 190, 40), dtype=np.uint8)
        else:
            fill = np.array((230, 60, 60), dtype=np.uint8)
        out[y0:y1, x0 : x0 + fill_w] = fill

    out[y0 : y0 + 2, x0:x1] = 230
    out[y1 - 2 : y1, x0:x1] = 230
    out[y0:y1, x0 : x0 + 2] = 230
    out[y0:y1, x1 - 2 : x1] = 230
    return np.array(
        add_text_to_frame(
            out,
            f"{fatigue_type} Fatigue {int(round(fatigue * 100.0)):d}%",
            pos=(x0 + 20, max(0, y0 - 25)),
            color=(0, 0, 0),
            fontsize=18,
        )
    )


def _draw_qpos_tracking_bars(
    frame: np.ndarray,
    joint_names: list[str],
    current_values: np.ndarray,
    target_values: np.ndarray,
    joint_ranges: np.ndarray,
    origin: tuple[int, int] = (10, 60),
) -> np.ndarray:
    """Draw a stacked left-side HUD: one range bar per tracked joint, mirroring
    :func:`_draw_fatigue_status`'s bar style. Each bar spans that joint's
    ``joint_ranges`` limits, with a blue vertical line at the current qpos value and
    an orange vertical line at the target/reference value -- a 3D marker at the
    joint's anchor position is not a useful overlay (joint anchors sit inside the
    limb mesh and are occluded almost every frame), so the tracking gap is shown as a
    number line instead.

    Args:
        frame: RGB frame to draw onto (not modified in place).
        joint_names: Tracked joint names, one bar per entry, in the same order as
                     ``current_values``/``target_values``/``joint_ranges``.
        current_values: Live qpos value per tracked joint, shape ``(n,)``.
        target_values: Reference/target qpos value per tracked joint, shape ``(n,)``.
        joint_ranges: ``(n, 2)`` ``[lo, hi]`` limits per tracked joint, used as each
                      bar's scale. A degenerate (unlimited, ``hi <= lo``) range falls
                      back to a span centered on the current/target values.
        origin: Top-left ``(x, y)`` pixel of the first bar; bars stack downward.

    Returns:
        A copy of ``frame`` with the bars drawn.
    """
    out = np.asarray(frame, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    x0, y_start = origin

    bar_w = max(100, int(round(w * 0.32)))
    bar_h = max(10, int(round(h * 0.028)))
    label_h = int(round(bar_h * 1.4))
    gap = bar_h + label_h + 6

    for i, name in enumerate(joint_names):
        y0 = y_start + i * gap + label_h
        y1 = y0 + bar_h
        x1 = x0 + bar_w
        if y1 + 4 > h:
            break  # ran out of vertical space

        lo, hi = float(joint_ranges[i][0]), float(joint_ranges[i][1])
        if hi <= lo:
            center = 0.5 * (float(current_values[i]) + float(target_values[i]))
            spread = max(abs(float(current_values[i]) - float(target_values[i])), 0.1)
            lo, hi = center - spread, center + spread
        span = max(hi - lo, 1e-6)

        def _to_px(val: float) -> int:
            frac = float(np.clip((val - lo) / span, 0.0, 1.0))
            return x0 + int(round(frac * bar_w))

        out[max(0, y0 - 2) : min(h, y1 + 2), max(0, x0 - 2) : min(w, x1 + 2)] = (
            20,
            20,
            20,
        )
        out[y0:y1, x0:x1] = (60, 60, 60)

        cur_x = _to_px(float(current_values[i]))
        out[y0:y1, max(x0, cur_x - 2) : min(x1, cur_x + 2)] = (30, 100, 255)
        tgt_x = _to_px(float(target_values[i]))
        out[y0:y1, max(x0, tgt_x - 2) : min(x1, tgt_x + 2)] = (255, 140, 0)

        out[y0 : y0 + 1, x0:x1] = 230
        out[y1 - 1 : y1, x0:x1] = 230
        out[y0:y1, x0 : x0 + 1] = 230
        out[y0:y1, x1 - 1 : x1] = 230

        out = np.array(
            add_text_to_frame(
                out,
                name,
                pos=(x0, y0 - label_h + 2),
                color=(0, 0, 0),
                fontsize=13,
            )
        )
    return out
