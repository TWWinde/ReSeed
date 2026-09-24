"""Timestep-varying classifier-free guidance schedules.

The helpers are shared by frozen-teacher inference diagnostics and native SD3
distillation.  They do not retain gradients: CFG is a teacher-side control.
"""

from __future__ import annotations

import math

import torch

from .flow_matching import shift_sigma


def symmetric_tvcfg_schedule(
    timesteps: torch.Tensor,
    mean_guidance_scale: float,
    span: float,
) -> tuple[torch.Tensor, float]:
    """Return the official symmetric TV-CFG values on a descending grid.

    The raw scale rises from ``omega - s`` to ``omega + s`` and falls back.
    A scalar normalization preserves the timestep-weighted mean ``omega``.
    """

    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError("symmetric teacher CFG requires at least two timesteps")
    if not math.isfinite(mean_guidance_scale) or mean_guidance_scale <= 0:
        raise ValueError("mean teacher CFG must be finite and positive")
    if not math.isfinite(span) or not 0.0 < span < mean_guidance_scale:
        raise ValueError("symmetric teacher CFG span s must satisfy 0 < s < omega")

    steps = int(timesteps.numel())
    half = steps // 2
    low = mean_guidance_scale - span
    high = mean_guidance_scale + span
    if steps % 2 == 0:
        up = torch.linspace(low, high, half + 1, dtype=torch.float64)[:-1]
        down = torch.linspace(high, low, half, dtype=torch.float64)
    else:
        up = torch.linspace(low, high, half + 1, dtype=torch.float64)
        down = torch.linspace(high, low, half + 1, dtype=torch.float64)[1:]
    raw = torch.cat((up, down))

    normalized = timesteps.detach().to(device="cpu", dtype=torch.float64)
    normalized = normalized / normalized.max()
    deltas = normalized - torch.cat(
        (normalized[1:], torch.zeros(1, dtype=torch.float64))
    )
    if not torch.isfinite(deltas).all() or not bool((deltas >= 0).all()):
        raise ValueError("teacher scheduler timesteps must be finite and descending")
    weighted_raw_mean = torch.dot(raw, deltas)
    if not torch.isfinite(weighted_raw_mean) or float(weighted_raw_mean) <= 0.0:
        raise ValueError("invalid symmetric teacher CFG normalization denominator")
    normalization = mean_guidance_scale / float(weighted_raw_mean)
    schedule = raw * normalization
    if not torch.isfinite(schedule).all():
        raise ValueError("symmetric teacher CFG schedule contains NaN/Inf")
    if abs(float(torch.dot(schedule, deltas)) - mean_guidance_scale) > 1e-8:
        raise AssertionError("symmetric teacher CFG weighted mean is not preserved")
    return schedule, float(normalization)


def flowmatch_symmetric_tvcfg_reference(
    *,
    inference_steps: int,
    shift: float,
    mean_guidance_scale: float,
    span: float,
    num_train_timesteps: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build Diffusers' shifted SD3 reference grid and its TV-CFG values.

    ``FlowMatchEulerDiscreteScheduler`` stores a shifted training ``sigma_min``.
    Its inference grid is linear between ``sigma_max`` and that stored minimum,
    then applies the configured shift to the grid. Reproducing both operations
    here makes the training-side reference identical to ``set_timesteps``.
    """

    if inference_steps < 2:
        raise ValueError("TV-CFG reference inference steps must be at least two")
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    stored_sigma_min = shift_sigma(
        torch.tensor(1.0 / num_train_timesteps, dtype=torch.float64), shift
    )
    base_sigmas = torch.linspace(
        1.0, float(stored_sigma_min), inference_steps, dtype=torch.float64
    )
    shifted_sigmas = shift_sigma(base_sigmas, shift)
    values, normalization = symmetric_tvcfg_schedule(
        shifted_sigmas, mean_guidance_scale, span
    )
    return shifted_sigmas, values, normalization


def interpolate_descending_schedule(
    query: torch.Tensor,
    reference: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate schedule values on a descending reference grid."""

    query = torch.as_tensor(query)
    reference = torch.as_tensor(reference, device=query.device, dtype=torch.float32).reshape(-1)
    values = torch.as_tensor(values, device=query.device, dtype=torch.float32).reshape(-1)
    if reference.numel() != values.numel() or reference.numel() < 2:
        raise ValueError("reference and values must have the same length >= 2")
    if not bool(torch.all(reference[:-1] > reference[1:])):
        raise ValueError("schedule reference must be strictly descending")
    ascending_x = reference.flip(0)
    ascending_y = values.flip(0)
    flat = query.float().reshape(-1).clamp(float(ascending_x[0]), float(ascending_x[-1]))
    upper = torch.searchsorted(ascending_x, flat, right=False).clamp(1, ascending_x.numel() - 1)
    lower = upper - 1
    x0, x1 = ascending_x[lower], ascending_x[upper]
    y0, y1 = ascending_y[lower], ascending_y[upper]
    weight = (flat - x0) / (x1 - x0)
    result = y0 + weight * (y1 - y0)
    if not torch.isfinite(result).all() or bool((result <= 0).any()):
        raise ValueError("interpolated TV-CFG scale must be finite and positive")
    return result.reshape(query.shape)


__all__ = [
    "flowmatch_symmetric_tvcfg_reference",
    "interpolate_descending_schedule",
    "symmetric_tvcfg_schedule",
]
