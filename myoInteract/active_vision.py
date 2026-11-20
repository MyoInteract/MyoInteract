"""Dynamic 2D gaze fixation state and dual-acuity fovea/periphery tensor streams."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


def _is_torch_tensor(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


@dataclass(frozen=True)
class GazeDynamicsConfig:
    """Configuration for bounded 2D saccadic and smooth-pursuit UV fixation control."""

    max_saccade_speed_uv: float = 4.0
    smoothing_alpha: float = 1.0
    uv_margin: float = 0.05
    init_uv: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        if isinstance(self.max_saccade_speed_uv, bool) or not (
            math.isfinite(self.max_saccade_speed_uv) and self.max_saccade_speed_uv > 0.0
        ):
            raise ValueError("max_saccade_speed_uv must be finite and positive.")
        if isinstance(self.smoothing_alpha, bool) or not (
            math.isfinite(self.smoothing_alpha) and 0.0 < self.smoothing_alpha <= 1.0
        ):
            raise ValueError("smoothing_alpha must lie in (0, 1].")
        if isinstance(self.uv_margin, bool) or not (
            math.isfinite(self.uv_margin) and 0.0 <= self.uv_margin < 0.5
        ):
            raise ValueError("uv_margin must lie in [0, 0.5).")
        if (
            isinstance(self.init_uv, (str, bytes))
            or not isinstance(self.init_uv, Sequence)
            or len(self.init_uv) != 2
        ):
            raise ValueError("init_uv must be a 2-element (u, v) coordinate.")
        u, v = float(self.init_uv[0]), float(self.init_uv[1])
        lo, hi = self.uv_margin, 1.0 - self.uv_margin
        if not (
            math.isfinite(u) and math.isfinite(v) and lo <= u <= hi and lo <= v <= hi
        ):
            raise ValueError(f"init_uv must lie within [{lo}, {hi}]^2.")
        object.__setattr__(self, "init_uv", (u, v))


class GazeState:
    """Stateful 2D fixation point in normalized image coordinates ``[0, 1]^2``."""

    def __init__(
        self,
        cfg: GazeDynamicsConfig = GazeDynamicsConfig(),
        *,
        batch_size: int | None = None,
        device: Any = None,
    ) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.device = device
        self._use_torch = device is not None
        if self._use_torch:
            import torch

            b = 1 if batch_size is None else int(batch_size)
            if b < 1:
                raise ValueError("batch_size must be >= 1")
            self._uv: Any = torch.tensor(
                cfg.init_uv, dtype=torch.float32, device=device
            ).repeat(b, 1)
        else:
            if batch_size is None:
                self._uv = np.asarray(cfg.init_uv, dtype=np.float32).copy()
            else:
                if int(batch_size) < 1:
                    raise ValueError("batch_size must be >= 1")
                self._uv = np.tile(
                    np.asarray(cfg.init_uv, dtype=np.float32), (int(batch_size), 1)
                )

    @property
    def uv(self) -> Any:
        """Current fixation coordinates of shape ``(2,)`` or ``(B, 2)``."""
        return self._uv

    def reset(self, env_ids: Any = None) -> Any:
        """Reset all or selected environments back to ``cfg.init_uv``."""
        u0, v0 = self.cfg.init_uv
        if self._use_torch:
            import torch

            init_t = torch.tensor(
                [u0, v0], dtype=self._uv.dtype, device=self._uv.device
            )
            if env_ids is None:
                self._uv[:] = init_t
            else:
                self._uv[env_ids] = init_t
            return self._uv

        init_np = np.array([u0, v0], dtype=np.float32)
        if env_ids is None:
            self._uv[...] = init_np
        else:
            self._uv[env_ids] = init_np
        return self._uv

    def step(
        self,
        command: Any,
        dt: float,
        *,
        mode: str = "velocity",
    ) -> Any:
        """Advance fixation state for one control interval ``dt`` [s].

        Args:
            command: ``(2,)`` or ``(B, 2)`` normalized velocity in ``[-1, 1]`` when
                ``mode='velocity'``, or desired target UV in ``[0, 1]`` (or ``[-1, 1]``
                mapped via ``0.5 * (cmd + 1)`` when ``mode='target_signed'``).
            dt: Control interval [s] (positive).
            mode: One of ``'velocity'``, ``'target'``, or ``'target_signed'``.

        Returns:
            Updated fixation UV array/tensor.
        """
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"dt must be finite and positive, got {dt}")
        if mode not in ("velocity", "target", "target_signed"):
            raise ValueError(
                f"Unknown gaze control mode {mode!r}; expected 'velocity', 'target', or 'target_signed'."
            )

        max_step = self.cfg.max_saccade_speed_uv * float(dt)
        lo = self.cfg.uv_margin
        hi = 1.0 - self.cfg.uv_margin
        alpha = self.cfg.smoothing_alpha

        if self._use_torch or _is_torch_tensor(command):
            import torch

            cmd = torch.as_tensor(command, dtype=self._uv.dtype, device=self._uv.device)
            if cmd.shape != self._uv.shape:
                raise ValueError(
                    f"command shape {tuple(cmd.shape)} does not match GazeState {tuple(self._uv.shape)}"
                )
            if mode == "velocity":
                delta = cmd.clamp(-1.0, 1.0) * max_step
                desired = (self._uv + delta).clamp(lo, hi)
            else:
                target = (
                    0.5 * (cmd.clamp(-1.0, 1.0) + 1.0)
                    if mode == "target_signed"
                    else cmd
                )
                target = target.clamp(lo, hi)
                diff = target - self._uv
                dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-8)
                step_scale = torch.clamp(max_step / dist, max=1.0)
                desired = self._uv + diff * step_scale
            self._uv = ((1.0 - alpha) * self._uv + alpha * desired).clamp(lo, hi)
            return self._uv

        cmd_np = np.asarray(command, dtype=np.float32)
        if cmd_np.shape != self._uv.shape:
            raise ValueError(
                f"command shape {cmd_np.shape} does not match GazeState {self._uv.shape}"
            )
        if mode == "velocity":
            delta_np = np.clip(cmd_np, -1.0, 1.0) * max_step
            desired_np = np.clip(self._uv + delta_np, lo, hi)
        else:
            target_np = (
                0.5 * (np.clip(cmd_np, -1.0, 1.0) + 1.0)
                if mode == "target_signed"
                else cmd_np
            )
            target_np = np.clip(target_np, lo, hi)
            diff_np = target_np - self._uv
            dist_np = np.maximum(np.linalg.norm(diff_np, axis=-1, keepdims=True), 1e-8)
            step_scale_np = np.minimum(1.0, max_step / dist_np)
            desired_np = self._uv + diff_np * step_scale_np
        self._uv = np.clip(
            (1.0 - alpha) * self._uv + alpha * desired_np, lo, hi
        ).astype(np.float32)
        return self._uv


def crop_foveal_patch_tensor(
    images: torch.Tensor,
    fixation_uv: Any,
    patch_height: int = 48,
    patch_width: int = 48,
    *,
    fov_ratio: float = 0.35,
) -> torch.Tensor:
    """Extract per-environment differentiable foveal crops around ``fixation_uv``.

    Args:
        images: Camera batch of shape ``(B, C, H, W)``.
        fixation_uv: Fixation coordinates in ``[0, 1]^2``, shape ``(2,)`` or ``(B, 2)``.
        patch_height: Output crop height in pixels.
        patch_width: Output crop width in pixels.
        fov_ratio: Linear crop span fraction in ``(0, 1]`` relative to the input frame.

    Returns:
        Cropped foveal tensor of shape ``(B, C, patch_height, patch_width)``.
    """
    if images.ndim != 4:
        raise ValueError(
            f"Expected 4D (B, C, H, W) tensor, got shape {tuple(images.shape)}."
        )
    if patch_height < 1 or patch_width < 1:
        raise ValueError("patch_height and patch_width must be positive integers.")
    if not (math.isfinite(fov_ratio) and 0.0 < fov_ratio <= 1.0):
        raise ValueError(f"fov_ratio must lie in (0, 1], got {fov_ratio}.")

    import torch
    import torch.nn.functional as F

    bsz, channels, _, _ = images.shape
    work_dtype = (
        images.dtype
        if images.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    work = images.to(dtype=work_dtype)

    uv = torch.as_tensor(fixation_uv, dtype=work_dtype, device=images.device)
    if uv.ndim == 1:
        if uv.numel() != 2:
            raise ValueError(
                f"fixation_uv must have 2 elements, got {tuple(uv.shape)}."
            )
        uv = uv.unsqueeze(0).expand(bsz, 2)
    elif uv.shape != (bsz, 2):
        raise ValueError(
            f"fixation_uv must have shape (2,) or ({bsz}, 2), got {tuple(uv.shape)}."
        )

    centers = 2.0 * uv.clamp(0.0, 1.0) - 1.0
    theta = torch.zeros((bsz, 2, 3), dtype=work_dtype, device=images.device)
    theta[:, 0, 0] = float(fov_ratio)
    theta[:, 1, 1] = float(fov_ratio)
    theta[:, 0, 2] = centers[:, 0]
    theta[:, 1, 2] = centers[:, 1]

    grid = F.affine_grid(
        theta,
        size=[bsz, channels, int(patch_height), int(patch_width)],
        align_corners=True,
    )
    cropped = F.grid_sample(
        work,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return cropped.to(dtype=images.dtype) if work_dtype != images.dtype else cropped


def extract_dual_acuity_streams(
    wide_images: torch.Tensor,
    fixation_uv: Any = (0.5, 0.5),
    *,
    stream_height: int = 48,
    stream_width: int = 48,
    fov_ratio: float = 0.35,
    narrow_images: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce matched ``(fovea, periphery)`` streams of shape ``(B, C, Hs, Ws)``."""
    if wide_images.ndim != 4:
        raise ValueError(
            f"wide_images must be 4D (B, C, H, W), got {tuple(wide_images.shape)}."
        )
    if stream_height < 1 or stream_width < 1:
        raise ValueError("stream_height and stream_width must be positive.")

    import torch
    import torch.nn.functional as F

    work_dtype = (
        wide_images.dtype
        if wide_images.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    wide_work = wide_images.to(dtype=work_dtype)
    periphery = F.interpolate(
        wide_work,
        size=(int(stream_height), int(stream_width)),
        mode="bilinear",
        align_corners=False,
    )

    if narrow_images is not None:
        if narrow_images.ndim != 4 or narrow_images.shape[:2] != wide_images.shape[:2]:
            raise ValueError(
                "narrow_images must be 4D with matching (B, C) dimensions as wide_images."
            )
        narrow_work = narrow_images.to(dtype=work_dtype)
        if narrow_work.shape[2:] == (int(stream_height), int(stream_width)):
            fovea = narrow_work
        else:
            fovea = F.interpolate(
                narrow_work,
                size=(int(stream_height), int(stream_width)),
                mode="bilinear",
                align_corners=False,
            )
    else:
        fovea = crop_foveal_patch_tensor(
            wide_work,
            fixation_uv,
            patch_height=stream_height,
            patch_width=stream_width,
            fov_ratio=fov_ratio,
        )

    if work_dtype != wide_images.dtype:
        return fovea.to(dtype=wide_images.dtype), periphery.to(dtype=wide_images.dtype)
    return fovea, periphery


def concat_dual_acuity_streams(
    wide_images: torch.Tensor,
    fixation_uv: Any = (0.5, 0.5),
    *,
    stream_height: int = 48,
    stream_width: int = 48,
    fov_ratio: float = 0.35,
    narrow_images: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate ``(fovea, periphery)`` along the channel axis ``(B, 2*C, Hs, Ws)``."""
    import torch

    fovea, periphery = extract_dual_acuity_streams(
        wide_images,
        fixation_uv,
        stream_height=stream_height,
        stream_width=stream_width,
        fov_ratio=fov_ratio,
        narrow_images=narrow_images,
    )
    return torch.cat((fovea, periphery), dim=1)


__all__ = [
    "GazeDynamicsConfig",
    "GazeState",
    "concat_dual_acuity_streams",
    "crop_foveal_patch_tensor",
    "extract_dual_acuity_streams",
]
