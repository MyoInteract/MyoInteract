"""Temporal motor chunking and progressive joint/actuator unlocking schedules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _is_torch_tensor(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


class ActionChunker:
    """Hold and optionally low-pass filter motor commands across ``chunk_steps`` steps.

    Operates without Python loops over environments on both CPU NumPy arrays
    (``(A,)`` or ``(B, A)``) and batched PyTorch tensors (``(B, A)``).
    """

    def __init__(self, chunk_steps: int = 4, *, ema_alpha: float = 1.0) -> None:
        if (
            isinstance(chunk_steps, bool)
            or not isinstance(chunk_steps, int)
            or chunk_steps < 1
        ):
            raise ValueError(
                f"chunk_steps must be an integer >= 1, got {chunk_steps!r}"
            )
        if isinstance(ema_alpha, bool) or not (
            math.isfinite(ema_alpha) and 0.0 < ema_alpha <= 1.0
        ):
            raise ValueError(f"ema_alpha must lie in (0, 1], got {ema_alpha!r}")
        self.chunk_steps = chunk_steps
        self.ema_alpha = float(ema_alpha)
        self._latched: Any = None
        self._smoothed: Any = None
        self._step_in_chunk: Any = None

    def reset(self, env_ids: Any = None) -> None:
        """Reset all state or selectively reset ``env_ids`` on next ``step``."""
        if env_ids is None or self._step_in_chunk is None:
            self._latched = None
            self._smoothed = None
            self._step_in_chunk = None
            return
        # Setting step counter to 0 forces immediate re-latching for those envs.
        self._step_in_chunk[env_ids] = 0
        if self._smoothed is not None:
            if _is_torch_tensor(self._smoothed):
                import torch

                self._smoothed[env_ids] = torch.nan
            else:
                self._smoothed[env_ids] = np.nan

    def step(self, raw_action: Any) -> Any:
        """Apply chunk latching and optional EMA filtering for one control step."""
        if _is_torch_tensor(raw_action):
            import torch

            act = raw_action.clone()
            if act.ndim == 1:
                if self._latched is None or self._step_in_chunk is None:
                    self._latched = act.clone()
                    self._smoothed = act.clone()
                    self._step_in_chunk = torch.zeros(
                        (), dtype=torch.long, device=act.device
                    )
                refresh = int(self._step_in_chunk.item()) % self.chunk_steps == 0
                if refresh:
                    self._latched = act.clone()
                    if bool(torch.isnan(self._smoothed).any().item()):
                        self._smoothed = act.clone()
                    else:
                        self._smoothed = (
                            1.0 - self.ema_alpha
                        ) * self._smoothed + self.ema_alpha * self._latched
                self._step_in_chunk = self._step_in_chunk + 1
                return self._smoothed.clone()

            bsz = act.shape[0]
            if (
                self._latched is None
                or self._step_in_chunk is None
                or self._latched.shape != act.shape
            ):
                self._latched = act.clone()
                self._smoothed = act.clone()
                self._step_in_chunk = torch.zeros(
                    bsz, dtype=torch.long, device=act.device
                )

            refresh_mask = (self._step_in_chunk % self.chunk_steps) == 0
            if bool(refresh_mask.any().item()):
                self._latched = torch.where(
                    refresh_mask.unsqueeze(-1), act, self._latched
                )
                was_reset = torch.isnan(self._smoothed).any(dim=-1)
                blended = (
                    1.0 - self.ema_alpha
                ) * self._smoothed + self.ema_alpha * self._latched
                updated = torch.where(was_reset.unsqueeze(-1), self._latched, blended)
                self._smoothed = torch.where(
                    refresh_mask.unsqueeze(-1), updated, self._smoothed
                )
            self._step_in_chunk = self._step_in_chunk + 1
            return self._smoothed.clone()

        act_np = np.asarray(raw_action, dtype=np.float32)
        if act_np.ndim == 1:
            if self._latched is None or self._step_in_chunk is None:
                self._latched = act_np.copy()
                self._smoothed = act_np.copy()
                self._step_in_chunk = np.array(0, dtype=np.int64)
            if int(self._step_in_chunk) % self.chunk_steps == 0:
                self._latched = act_np.copy()
                if np.isnan(self._smoothed).any():
                    self._smoothed = act_np.copy()
                else:
                    self._smoothed = (
                        1.0 - self.ema_alpha
                    ) * self._smoothed + self.ema_alpha * self._latched
            self._step_in_chunk = self._step_in_chunk + 1
            return self._smoothed.copy()

        bsz_np = act_np.shape[0]
        if (
            self._latched is None
            or self._step_in_chunk is None
            or self._latched.shape != act_np.shape
        ):
            self._latched = act_np.copy()
            self._smoothed = act_np.copy()
            self._step_in_chunk = np.zeros(bsz_np, dtype=np.int64)

        refresh_np = (self._step_in_chunk % self.chunk_steps) == 0
        if np.any(refresh_np):
            self._latched = np.where(refresh_np[:, None], act_np, self._latched)
            was_reset_np = np.isnan(self._smoothed).any(axis=-1)
            blended_np = (
                1.0 - self.ema_alpha
            ) * self._smoothed + self.ema_alpha * self._latched
            updated_np = np.where(was_reset_np[:, None], self._latched, blended_np)
            self._smoothed = np.where(refresh_np[:, None], updated_np, self._smoothed)
        self._step_in_chunk = self._step_in_chunk + 1
        return self._smoothed.astype(np.float32, copy=True)


@dataclass(frozen=True)
class CurriculumStage:
    """Unlock a subset of action/actuator indices starting at ``unlock_step``."""

    unlock_step: int
    indices: tuple[int, ...]
    max_gain: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.unlock_step, bool) or self.unlock_step < 0:
            raise ValueError("unlock_step must be a non-negative integer.")
        if not self.indices:
            raise ValueError("indices must be non-empty.")
        norm_idx = tuple(int(i) for i in self.indices)
        if any(i < 0 for i in norm_idx):
            raise ValueError("indices must be non-negative.")
        if isinstance(self.max_gain, bool) or not (
            math.isfinite(self.max_gain) and 0.0 <= self.max_gain <= 1.0
        ):
            raise ValueError("max_gain must lie in [0, 1].")
        object.__setattr__(self, "indices", norm_idx)
        object.__setattr__(self, "max_gain", float(self.max_gain))


class ProgressiveJointSchedule:
    """Stage-wise progressive unlocking of high-DOF muscle/joint action channels."""

    def __init__(
        self,
        num_actions: int,
        stages: Sequence[CurriculumStage],
        *,
        baseline: Sequence[float] | np.ndarray | None = None,
        ramp_steps: int = 0,
    ) -> None:
        if isinstance(num_actions, bool) or num_actions < 1:
            raise ValueError("num_actions must be >= 1.")
        if isinstance(ramp_steps, bool) or ramp_steps < 0:
            raise ValueError("ramp_steps must be >= 0.")
        self.num_actions = int(num_actions)
        self.stages = tuple(stages)
        self.ramp_steps = int(ramp_steps)

        for st in self.stages:
            if any(idx >= self.num_actions for idx in st.indices):
                raise ValueError(
                    f"Stage index out of range for num_actions={self.num_actions}: {st.indices}"
                )

        if baseline is None:
            self._baseline = np.zeros(self.num_actions, dtype=np.float32)
        else:
            base_arr = np.asarray(baseline, dtype=np.float32)
            if base_arr.shape != (self.num_actions,):
                raise ValueError(
                    f"baseline must have shape ({self.num_actions},), got {base_arr.shape}"
                )
            self._baseline = base_arr.copy()

    def weights_at(self, step: int) -> np.ndarray:
        """Compute per-channel interpolation weights in ``[0, 1]`` at training ``step``."""
        if step < 0:
            raise ValueError("step must be non-negative.")
        weights = np.zeros(self.num_actions, dtype=np.float32)
        for st in self.stages:
            if step < st.unlock_step:
                continue
            if self.ramp_steps == 0:
                gain = st.max_gain
            else:
                progress = min(
                    1.0, float(step - st.unlock_step) / float(self.ramp_steps)
                )
                gain = st.max_gain * progress
            idx_arr = np.asarray(st.indices, dtype=np.intp)
            weights[idx_arr] = np.maximum(weights[idx_arr], float(gain))
        return weights

    def apply(self, actions: Any, step: int) -> Any:
        """Blend ``actions`` with ``baseline`` according to ``weights_at(step)``."""
        w_np = self.weights_at(step)
        if _is_torch_tensor(actions):
            import torch

            w_t = torch.as_tensor(w_np, dtype=actions.dtype, device=actions.device)
            b_t = torch.as_tensor(
                self._baseline, dtype=actions.dtype, device=actions.device
            )
            return b_t + w_t * (actions - b_t)

        act_np = np.asarray(actions, dtype=np.float32)
        return (self._baseline + w_np * (act_np - self._baseline)).astype(np.float32)


__all__ = [
    "ActionChunker",
    "CurriculumStage",
    "ProgressiveJointSchedule",
]
