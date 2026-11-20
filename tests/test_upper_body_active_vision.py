"""Tests for upper-body myotorso_arms composition, active gaze streams, and curriculum."""

from __future__ import annotations

import subprocess
import sys

import mujoco
import numpy as np
import pytest

from myoInteract.active_vision import GazeDynamicsConfig, GazeState
from myoInteract.body_models import (
    _MYOTORSO_ARMS_ACTIVE_HEAD_MODEL_PATH,
    _MYOTORSO_ARMS_NOHAND_MODEL_PATH,
    _MYOTORSO_ARMS_REACH_MODEL_PATH,
    _MYOTORSO_ARMS_UNIVERSAL_MODEL_PATH,
    base_spec,
    compile_base_model,
    myotorso_arms_spec,
)
from myoInteract.curriculum import (
    ActionChunker,
    CurriculumStage,
    ProgressiveJointSchedule,
)


def test_studio_modules_avoid_eager_torch_or_mjlab_imports() -> None:
    code = (
        "import sys; "
        "import myoInteract.body_models, myoInteract.active_vision, myoInteract.curriculum; "
        "assert 'torch' not in sys.modules, list(sys.modules); "
        "assert 'mjlab' not in sys.modules, list(sys.modules)"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


def test_myotorso_arms_variants_compile_and_step() -> None:
    full_model = compile_base_model(_MYOTORSO_ARMS_UNIVERSAL_MODEL_PATH)
    nohand_model = compile_base_model(_MYOTORSO_ARMS_NOHAND_MODEL_PATH)
    reach_model = compile_base_model(_MYOTORSO_ARMS_REACH_MODEL_PATH)
    active_spec = base_spec(_MYOTORSO_ARMS_ACTIVE_HEAD_MODEL_PATH)
    assert active_spec.body("thorax") is not None
    try:
        from myoInteract.myointeract_utils import add_torso_contact_sensor_to_spec
    except ImportError:
        add_torso_contact_sensor_to_spec = None
    if add_torso_contact_sensor_to_spec is not None:
        add_torso_contact_sensor_to_spec(active_spec)
    active_model = active_spec.compile()

    assert full_model.nu > nohand_model.nu
    # Bilateral 37-muscle hands pruned: 74 fewer actuators in nohand.
    assert full_model.nu - nohand_model.nu == 74
    assert reach_model.nq < nohand_model.nq

    # Active-head variant adds 2 cervical hinge joints + 2 position servos,
    # ocular cameras, gaze sites, and 6-axis head IMU sensors.
    assert active_model.nq == nohand_model.nq + 2
    assert active_model.nu == nohand_model.nu + 2
    for jname in ("neck_rotation", "neck_flexion"):
        assert active_model.joint(jname).id >= 0
    for cname in ("head_fovea", "head_periphery", "eye_l", "eye_r"):
        assert active_model.camera(cname).id >= 0
    for sname in ("gaze_origin", "gaze_forward", "head_imu", "IFtip", "IFtip_r", "IFtip_l"):
        assert active_model.site(sname).id >= 0
    for sen_name in ("head_accel", "head_gyro", "head_quat"):
        assert active_model.sensor(sen_name).id >= 0

    data = mujoco.MjData(active_model)
    yaw_act = active_model.actuator("neck_rotation_act").id
    pitch_act = active_model.actuator("neck_flexion_act").id
    data.ctrl[yaw_act] = 0.35
    data.ctrl[pitch_act] = -0.2
    mujoco.mj_step(active_model, data, nstep=25)
    assert np.all(np.isfinite(data.qpos))
    assert abs(float(data.qpos[active_model.joint("neck_rotation").qposadr[0]])) > 0.01

    with pytest.raises(ValueError, match="Unknown myotorso_arms variant"):
        myotorso_arms_spec("invalid")


def test_gaze_state_numpy_velocity_target_and_reset() -> None:
    cfg = GazeDynamicsConfig(max_saccade_speed_uv=2.0, uv_margin=0.1, init_uv=(0.5, 0.5))
    state = GazeState(cfg, batch_size=3)
    assert state.uv.shape == (3, 2)

    uv1 = state.step(np.array([[1.0, 0.0], [-1.0, 1.0], [0.0, 0.0]]), dt=0.1, mode="velocity")
    np.testing.assert_allclose(uv1[0], [0.7, 0.5], atol=1e-6)
    np.testing.assert_allclose(uv1[1], [0.3, 0.7], atol=1e-6)
    np.testing.assert_allclose(uv1[2], [0.5, 0.5], atol=1e-6)

    # Margin clamping
    for _ in range(10):
        state.step(np.ones((3, 2)), dt=0.2, mode="velocity")
    np.testing.assert_allclose(state.uv, np.full((3, 2), 0.9), atol=1e-6)

    # Selective reset of env 1
    state.reset(env_ids=[1])
    np.testing.assert_allclose(state.uv[1], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(state.uv[0], [0.9, 0.9], atol=1e-6)


def test_crop_and_dual_acuity_streams_torch() -> None:
    torch = pytest.importorskip("torch")
    from myoInteract.active_vision import (
        concat_dual_acuity_streams,
        crop_foveal_patch_tensor,
        extract_dual_acuity_streams,
    )

    # Build a synthetic 2-env image with distinct quadrant intensities:
    # top-left=0.1, top-right=0.4, bottom-left=0.7, bottom-right=1.0
    img = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
    img[:, :, :32, :32] = 0.1
    img[:, :, :32, 32:] = 0.4
    img[:, :, 32:, :32] = 0.7
    img[:, :, 32:, 32:] = 1.0

    uv = torch.tensor([[0.2, 0.2], [0.8, 0.8]], dtype=torch.float32, requires_grad=True)
    crops = crop_foveal_patch_tensor(img, uv, patch_height=16, patch_width=16, fov_ratio=0.2)
    assert crops.shape == (2, 3, 16, 16)
    assert float(crops[0].mean().item()) == pytest.approx(0.1, abs=1e-5)
    assert float(crops[1].mean().item()) == pytest.approx(1.0, abs=1e-5)

    loss = crops.sum()
    loss.backward()
    assert uv.grad is not None and torch.isfinite(uv.grad).all()

    fovea, periphery = extract_dual_acuity_streams(
        img, uv.detach(), stream_height=24, stream_width=24, fov_ratio=0.25
    )
    assert fovea.shape == (2, 3, 24, 24)
    assert periphery.shape == (2, 3, 24, 24)
    # Peripheral stream averages all four quadrants (~0.55), while env 0 fovea stays at 0.1.
    assert float(periphery[0].mean().item()) == pytest.approx(0.55, abs=0.03)
    assert float(fovea[0].mean().item()) == pytest.approx(0.1, abs=1e-5)

    combined = concat_dual_acuity_streams(
        img, uv.detach(), stream_height=24, stream_width=24, fov_ratio=0.25
    )
    assert combined.shape == (2, 6, 24, 24)


def test_action_chunker_and_progressive_joint_schedule() -> None:
    chunker = ActionChunker(chunk_steps=3, ema_alpha=0.5)
    a0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    a1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    out0 = chunker.step(a0)
    out1 = chunker.step(a1)
    out2 = chunker.step(a1)
    # Steps 0, 1, 2 within chunk 0 hold a0.
    np.testing.assert_allclose(out0, a0)
    np.testing.assert_allclose(out1, a0)
    np.testing.assert_allclose(out2, a0)

    # Step 3 starts chunk 1 and blends 0.5 * a0 + 0.5 * a1.
    out3 = chunker.step(a1)
    np.testing.assert_allclose(out3, 0.5 * a0 + 0.5 * a1)

    # Selective reset of env 0 forces immediate fresh latch without EMA carry-over.
    chunker.reset(env_ids=[0])
    a2 = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    out4 = chunker.step(a2)
    np.testing.assert_allclose(out4[0], [0.8, 0.2])
    np.testing.assert_allclose(out4[1], out3[1])

    schedule = ProgressiveJointSchedule(
        num_actions=4,
        stages=[
            CurriculumStage(unlock_step=0, indices=(0, 1)),
            CurriculumStage(unlock_step=100, indices=(2, 3)),
        ],
        baseline=[0.1, 0.1, 0.2, 0.2],
        ramp_steps=50,
    )
    raw = np.ones(4, dtype=np.float32)
    # At step 0: ramp starts at 0 -> pure baseline
    np.testing.assert_allclose(schedule.apply(raw, step=0), [0.1, 0.1, 0.2, 0.2])
    # At step 50: stage 0 at 1.0, stage 1 still locked at baseline
    np.testing.assert_allclose(schedule.apply(raw, step=50), [1.0, 1.0, 0.2, 0.2])
    # At step 125: stage 1 halfway through 50-step ramp -> 0.2 + 0.5 * 0.8 = 0.6
    np.testing.assert_allclose(schedule.apply(raw, step=125), [1.0, 1.0, 0.6, 0.6])


def test_action_chunker_and_schedule_torch_parity() -> None:
    torch = pytest.importorskip("torch")

    chunker = ActionChunker(chunk_steps=2, ema_alpha=1.0)
    t0 = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
    t1 = torch.tensor([[0.9, 0.1], [0.3, 0.5]])
    assert torch.allclose(chunker.step(t0), t0)
    assert torch.allclose(chunker.step(t1), t0)
    assert torch.allclose(chunker.step(t1), t1)

    schedule = ProgressiveJointSchedule(
        num_actions=2,
        stages=[CurriculumStage(unlock_step=0, indices=(0,))],
        baseline=[0.25, 0.75],
    )
    applied = schedule.apply(torch.ones((2, 2)), step=10)
    expected = torch.tensor([[1.0, 0.75], [1.0, 0.75]])
    assert torch.allclose(applied, expected)


def test_dual_acuity_vision_config_validation() -> None:
    pytest.importorskip("mjlab")
    from myoInteract.vision import DualAcuityVisionConfig, myo_create_vision

    cfg = DualAcuityVisionConfig(
        periphery_camera_name="head_periphery",
        fovea_camera_name="head_fovea",
        width=96,
        height=96,
        stream_width=48,
        stream_height=48,
        fov_ratio=0.35,
    )
    comp = myo_create_vision(cfg)
    assert comp is not None

    with pytest.raises(ValueError, match="fov_ratio"):
        DualAcuityVisionConfig(fov_ratio=1.5)
    with pytest.raises(ValueError, match="at least one of use_rgb or use_depth"):
        DualAcuityVisionConfig(use_rgb=False, use_depth=False)
