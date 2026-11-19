"""Eccentricity-dependent foveated visual acuity filter for batched camera tensors."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

_MAX_CACHE_ENTRIES = 32


@dataclass(frozen=True)
class FoveationParams:
    """Parameters for Geisler-Perry eccentricity-dependent spatial acuity falloff.

    Models Gaussian blur standard deviation ``sigma_deg(e) = sigma_0 * (1 + e / e_2)``
    as a function of angular retinal eccentricity ``e`` [deg] from ``fixation_uv``.
    """

    fovea_sigma0_deg: float = 0.04
    half_resolution_deg: float = 2.3
    levels: int = 5
    camera_fovy_deg: float = 90.0
    fixation_uv: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        if isinstance(self.fovea_sigma0_deg, bool) or not (
            math.isfinite(self.fovea_sigma0_deg) and self.fovea_sigma0_deg >= 0.0
        ):
            raise ValueError("Vision fovea_sigma0_deg must be finite and non-negative.")
        if isinstance(self.half_resolution_deg, bool) or not (
            math.isfinite(self.half_resolution_deg) and self.half_resolution_deg > 0.0
        ):
            raise ValueError("Vision half_resolution_deg must be finite and positive.")
        if (
            isinstance(self.levels, bool)
            or not isinstance(self.levels, int)
            or self.levels < 2
        ):
            raise ValueError("Vision foveation levels must be an integer >= 2.")
        if isinstance(self.camera_fovy_deg, bool) or not (
            math.isfinite(self.camera_fovy_deg) and 0.0 < self.camera_fovy_deg < 180.0
        ):
            raise ValueError("Vision camera_fovy_deg must lie in (0, 180).")
        if (
            isinstance(self.fixation_uv, (str, bytes))
            or not isinstance(self.fixation_uv, Sequence)
            or len(self.fixation_uv) != 2
        ):
            raise ValueError(
                "Vision fixation_uv must be a 2-element (u, v) coordinate."
            )
        u_raw, v_raw = self.fixation_uv
        if (
            isinstance(u_raw, bool)
            or isinstance(v_raw, bool)
            or not (math.isfinite(float(u_raw)) and math.isfinite(float(v_raw)))
        ):
            raise ValueError("Vision fixation_uv entries must be finite floats.")
        u, v = float(u_raw), float(v_raw)
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError("Vision fixation_uv entries must lie in [0, 1].")
        object.__setattr__(self, "fixation_uv", (u, v))


@dataclass(frozen=True)
class _CachedPyramidPlan:
    identity: bool
    kernels: tuple[tuple[Any, Any, int], ...]
    idx0: Any
    idx1: Any
    alpha: Any


_PLAN_CACHE: OrderedDict[tuple[Any, ...], _CachedPyramidPlan] = OrderedDict()


def eccentricity_deg_map(
    height: int,
    width: int,
    params: FoveationParams = FoveationParams(),
    *,
    device: Any = None,
    dtype: Any = None,
) -> torch.Tensor:
    """Compute pinhole ray-angle eccentricity [deg] of every pixel from ``fixation_uv``."""
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be positive, got ({height}, {width}).")
    import torch

    target_dtype = torch.float32 if dtype is None else dtype
    fy = 0.5 * float(height) / math.tan(0.5 * math.radians(params.camera_fovy_deg))
    fx = fy
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)

    xs = torch.arange(width, device=device, dtype=target_dtype)
    ys = torch.arange(height, device=device, dtype=target_dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    rays = torch.stack(
        ((grid_x - cx) / fx, (grid_y - cy) / fy, torch.ones_like(grid_x)),
        dim=-1,
    )
    rays = rays / torch.linalg.norm(rays, dim=-1, keepdim=True)

    u0, v0 = params.fixation_uv
    fx0 = u0 * float(width - 1)
    fy0 = v0 * float(height - 1)
    fix_ray = torch.tensor(
        [(fx0 - cx) / fx, (fy0 - cy) / fy, 1.0],
        device=device,
        dtype=target_dtype,
    )
    fix_ray = fix_ray / torch.linalg.norm(fix_ray)

    cos_theta = torch.clamp(torch.sum(rays * fix_ray, dim=-1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _gaussian_1d(sigma: float, radius: int, device: Any, dtype: Any) -> torch.Tensor:
    import torch

    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    weights = torch.exp(-0.5 * (coords / sigma) ** 2)
    return weights / weights.sum()


def _get_pyramid_plan(
    height: int,
    width: int,
    params: FoveationParams,
    device: Any,
    dtype: Any,
) -> _CachedPyramidPlan:
    import torch

    key = (str(device), dtype, height, width, params)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        _PLAN_CACHE.move_to_end(key)
        return cached

    ecc_deg = eccentricity_deg_map(
        height, width, params, device=device, dtype=torch.float32
    )
    fy = 0.5 * float(height) / math.tan(0.5 * math.radians(params.camera_fovy_deg))
    px_per_deg = fy * (math.pi / 180.0)
    sigma_deg = params.fovea_sigma0_deg * (1.0 + ecc_deg / params.half_resolution_deg)
    sigma_px = sigma_deg * px_per_deg
    sigma_max = float(sigma_px.max().item())

    if sigma_max <= 1e-6:
        plan = _CachedPyramidPlan(
            identity=True, kernels=(), idx0=None, idx1=None, alpha=None
        )
    else:
        n_levels = params.levels
        if n_levels == 2:
            nonzero_sigmas = [sigma_max]
        else:
            sigma_min = max(
                min(0.35, 0.25 * sigma_max),
                sigma_max * (0.5 ** (n_levels - 2)),
            )
            ratio = (sigma_max / sigma_min) ** (1.0 / float(n_levels - 2))
            nonzero_sigmas = [sigma_min * (ratio**i) for i in range(n_levels - 1)]

        sigmas = torch.tensor(
            [0.0, *nonzero_sigmas], device=device, dtype=torch.float32
        )
        clamped = sigma_px.clamp(0.0, sigma_max)
        idx1 = torch.searchsorted(sigmas, clamped).clamp(1, n_levels - 1)
        idx0 = idx1 - 1
        s0 = sigmas[idx0]
        s1 = sigmas[idx1]
        alpha = ((clamped - s0) / (s1 - s0).clamp_min(1e-12)).clamp(0.0, 1.0)

        kernels: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for s_val in nonzero_sigmas:
            radius = max(1, int(math.ceil(3.0 * float(s_val))))
            k1d = _gaussian_1d(float(s_val), radius, device=device, dtype=dtype)
            kx = k1d.view(1, 1, 1, 2 * radius + 1)
            ky = k1d.view(1, 1, 2 * radius + 1, 1)
            kernels.append((kx, ky, radius))

        plan = _CachedPyramidPlan(
            identity=False,
            kernels=tuple(kernels),
            idx0=idx0.to(dtype=torch.long).view(1, 1, 1, height, width),
            idx1=idx1.to(dtype=torch.long).view(1, 1, 1, height, width),
            alpha=alpha.to(dtype=dtype).view(1, 1, height, width),
        )

    _PLAN_CACHE[key] = plan
    while len(_PLAN_CACHE) > _MAX_CACHE_ENTRIES:
        _PLAN_CACHE.popitem(last=False)
    return plan


def foveate_tensor(
    images: torch.Tensor,
    params: FoveationParams = FoveationParams(),
) -> torch.Tensor:
    """Apply eccentricity-dependent Gaussian pyramid acuity falloff to ``(B, C, H, W)`` images."""
    if images.ndim != 4:
        raise ValueError(
            f"foveate_tensor expects a 4D (B, C, H, W) tensor, got shape {tuple(images.shape)}."
        )
    import torch
    import torch.nn.functional as F

    batch, channels, height, width = images.shape
    if batch == 0 or channels == 0:
        return images.clone()

    plan = _get_pyramid_plan(
        height, width, params, device=images.device, dtype=images.dtype
    )
    if plan.identity:
        return images.clone()

    flat = images.reshape(batch * channels, 1, height, width)
    levels = [images]
    for kx, ky, radius in plan.kernels:
        pad_x = "reflect" if radius < width else "replicate"
        pad_y = "reflect" if radius < height else "replicate"
        horiz = F.conv2d(F.pad(flat, (radius, radius, 0, 0), mode=pad_x), kx)
        vert = F.conv2d(F.pad(horiz, (0, 0, radius, radius), mode=pad_y), ky)
        levels.append(vert.reshape(batch, channels, height, width))

    pyramid = torch.stack(levels, dim=2)
    expand_shape = (batch, channels, 1, height, width)
    low = torch.gather(pyramid, 2, plan.idx0.expand(expand_shape)).squeeze(2)
    high = torch.gather(pyramid, 2, plan.idx1.expand(expand_shape)).squeeze(2)
    return low + plan.alpha * (high - low)


__all__ = ["FoveationParams", "eccentricity_deg_map", "foveate_tensor"]
