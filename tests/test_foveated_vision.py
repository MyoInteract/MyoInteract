"""Tests for eccentricity-dependent foveated visual acuity across Studio and mjlab stacks."""

from __future__ import annotations

import math

import pytest

from myoInteract.foveation import _PLAN_CACHE, FoveationParams


def test_foveation_params_validation_and_normalization() -> None:
    params = FoveationParams(
        fovea_sigma0_deg=0.05,
        half_resolution_deg=2.0,
        levels=6,
        camera_fovy_deg=75.0,
        fixation_uv=[0.25, 0.75],  # type: ignore[arg-type]
    )
    assert params.fixation_uv == (0.25, 0.75)
    assert hash(params) == hash(FoveationParams(0.05, 2.0, 6, 75.0, (0.25, 0.75)))

    with pytest.raises(ValueError, match="fovea_sigma0_deg"):
        FoveationParams(fovea_sigma0_deg=-0.01)
    with pytest.raises(ValueError, match="half_resolution_deg"):
        FoveationParams(half_resolution_deg=0.0)
    with pytest.raises(ValueError, match="levels"):
        FoveationParams(levels=1)
    with pytest.raises(ValueError, match="levels"):
        FoveationParams(levels=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="camera_fovy_deg"):
        FoveationParams(camera_fovy_deg=180.0)
    with pytest.raises(ValueError, match="fixation_uv"):
        FoveationParams(fixation_uv=(1.2, 0.5))
    with pytest.raises(ValueError, match="fixation_uv"):
        FoveationParams(fixation_uv=(0.5,))  # type: ignore[arg-type]


def test_eccentricity_deg_map_pinhole_geometry() -> None:
    pytest.importorskip("torch")
    from myoInteract.foveation import eccentricity_deg_map

    params = FoveationParams(camera_fovy_deg=90.0, fixation_uv=(0.5, 0.5))
    ecc = eccentricity_deg_map(129, 129, params)
    assert ecc.shape == (129, 129)
    assert float(ecc[64, 64].item()) == pytest.approx(0.0, abs=1e-5)

    # Top-center pixel on a 129x129 image sits at y=0, cy=64, fy=64.5 -> atan(64/64.5) deg.
    expected_top = math.degrees(math.atan(64.0 / 64.5))
    assert float(ecc[0, 64].item()) == pytest.approx(expected_top, rel=1e-4)
    assert (
        float(ecc[0, 0].item()) > float(ecc[0, 64].item()) > float(ecc[48, 64].item())
    )

    shifted = eccentricity_deg_map(
        101, 101, FoveationParams(camera_fovy_deg=90.0, fixation_uv=(0.2, 0.8))
    )
    assert float(shifted[80, 20].item()) == pytest.approx(0.0, abs=1e-5)


def test_foveate_tensor_contrast_attenuation_and_cache_reuse() -> None:
    torch = pytest.importorskip("torch")
    from myoInteract.foveation import foveate_tensor

    _PLAN_CACHE.clear()
    params = FoveationParams(
        fovea_sigma0_deg=0.18,
        half_resolution_deg=2.0,
        levels=5,
        camera_fovy_deg=90.0,
        fixation_uv=(0.5, 0.5),
    )

    uniform = torch.full((2, 3, 64, 64), 0.37, dtype=torch.float32)
    out_uniform = foveate_tensor(uniform, params)
    assert torch.allclose(out_uniform, uniform, atol=1e-5)

    yy, xx = torch.meshgrid(
        torch.arange(128, dtype=torch.float32),
        torch.arange(128, dtype=torch.float32),
        indexing="ij",
    )
    checker = (((xx.long() // 3) + (yy.long() // 3)) % 2).to(torch.float32)
    batch_rgb = checker.view(1, 1, 128, 128).expand(4, 3, 128, 128).clone()

    foveated = foveate_tensor(batch_rgb, params)
    assert foveated.shape == (4, 3, 128, 128)
    center_std = float(foveated[0, 0, 58:70, 58:70].std().item())
    mid_std = float(foveated[0, 0, 28:40, 58:70].std().item())
    periph_std = float(foveated[0, 0, 2:14, 2:14].std().item())
    assert center_std > mid_std > periph_std
    assert center_std / max(periph_std, 1e-6) > 5.0

    # Reuse cached pyramid plan across different batch and channel counts at the same resolution.
    cache_len_before = len(_PLAN_CACHE)
    depth_like = checker.view(1, 1, 128, 128).expand(2, 1, 128, 128).clone()
    foveated_depth = foveate_tensor(depth_like, params)
    assert foveated_depth.shape == (2, 1, 128, 128)
    assert len(_PLAN_CACHE) == cache_len_before

    # Shifting fixation toward top-left keeps top-left sharper than bottom-right.
    tl_params = FoveationParams(
        fovea_sigma0_deg=0.18,
        half_resolution_deg=2.0,
        levels=5,
        camera_fovy_deg=90.0,
        fixation_uv=(0.15, 0.15),
    )
    foveated_tl = foveate_tensor(batch_rgb[:1], tl_params)
    tl_std = float(foveated_tl[0, 0, 14:26, 14:26].std().item())
    br_std = float(foveated_tl[0, 0, 102:114, 102:114].std().item())
    assert tl_std > 4.0 * max(br_std, 1e-6)

    # Zero sigma0 returns an unblurred copy.
    identity_out = foveate_tensor(batch_rgb[:1], FoveationParams(fovea_sigma0_deg=0.0))
    assert torch.equal(identity_out, batch_rgb[:1])


def test_hydra_foveation_config_roundtrip() -> None:
    pytest.importorskip("hydra")
    from myoInteract.myointeract_hydra_cli import (
        extra_hydra_overrides,
        gradio_json_to_overrides,
        load_config_interactive,
        training_config_to_labelled_dict,
    )

    default_cfg = load_config_interactive(["env=universal", "vision=enabled"]).to_dict()
    default_labelled = training_config_to_labelled_dict(default_cfg)
    assert not any("foveat" in entry for entry in default_labelled["extra_overrides"])

    fov_cfg = load_config_interactive(
        [
            "env=universal",
            "vision=enabled",
            "vision.foveated_acuity=True",
            "vision.fovea_sigma0_deg=0.08",
            "vision.half_resolution_deg=1.9",
            "vision.foveation_levels=6",
            "vision.camera_fovy_deg=80.0",
            "vision.fixation_uv=[0.3,0.7]",
        ]
    ).to_dict()
    labelled = training_config_to_labelled_dict(fov_cfg)
    overrides = gradio_json_to_overrides(labelled) + extra_hydra_overrides(fov_cfg)
    restored = load_config_interactive(overrides)
    assert restored.env.vision.foveated_acuity is True
    assert restored.env.vision.fovea_sigma0_deg == pytest.approx(0.08)
    assert restored.env.vision.half_resolution_deg == pytest.approx(1.9)
    assert restored.env.vision.foveation_levels == 6
    assert restored.env.vision.camera_fovy_deg == pytest.approx(80.0)
    assert list(restored.env.vision.fixation_uv) == pytest.approx([0.3, 0.7])


def test_native_foveated_vision_recipe_and_cnn_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    import numpy as np

    from myoInteract import register_mjlab_myointeract_tasks as task
    from myoInteract.myointeract_hydra_cli import load_config_interactive
    from myoInteract.vision.cnn.cnn_vision_config import CnnVisionConfig

    cnn_cfg = CnnVisionConfig(
        camera_name="fixed-eye",
        foveated_acuity=True,
        fovea_sigma0_deg=0.06,
        fixation_uv=[0.4, 0.6],  # type: ignore[arg-type]
    )
    assert cnn_cfg.fixation_uv == (0.4, 0.6)
    assert cnn_cfg.foveation_params() == FoveationParams(
        fovea_sigma0_deg=0.06, fixation_uv=(0.4, 0.6)
    )
    assert CnnVisionConfig(camera_name="fixed-eye").foveation_params() is None

    config = load_config_interactive(
        [
            "env=universal",
            "vision=enabled",
            "vision.vision_mode=rgbd",
            "vision.foveated_acuity=True",
            "vision.fovea_sigma0_deg=0.07",
        ]
    )
    monkeypatch.setattr(task, "_get_muscle_names", lambda _cfg: ("muscle",))
    monkeypatch.setattr(task, "_compute_target_origin", lambda _cfg: np.zeros(3))
    env_cfg = task._make_universal_env_cfg(config)
    terms = env_cfg.observations["vision_mono"].terms
    assert terms["fixed-eye_rgb"].func is task._foveated_camera_rgb
    assert terms["fixed-eye_depth"].func is task._foveated_camera_depth
    assert terms["fixed-eye_rgb"].params["foveation"] == FoveationParams(
        fovea_sigma0_deg=0.07
    )

    bad_cfg = load_config_interactive(
        [
            "env=universal",
            "vision=enabled",
            "vision.foveated_acuity=True",
            "vision.fovea_sigma0_deg=-0.2",
        ]
    )
    with pytest.raises(ValueError, match="Vision"):
        task._vision_camera_recipe(bad_cfg.env.vision)
