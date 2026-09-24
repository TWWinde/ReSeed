# SPDX-License-Identifier: Apache-2.0
"""Flow-matching helpers for ReSeed on Wan, in fastgen's own conventions.

The first four functions are ported verbatim in spirit from CRRA's
flow_matching.py (shift_sigma / unshift_sigma / boundary_aligned_teacher_sigmas
/ secant). The last two deliberately do NOT use fastgen's "flow" prediction
type: every step is expressed through x0 and fastgen's forward_process, so the
sign convention of the velocity can never disagree between teacher and student.

Identity used throughout (rectified flow, x_t = (1-t) x0 + t eps):

    Euler from t to t' with v = (x_t - x0)/t
      = x_t + (t' - t) v
      = x_t * (t'/t) + x0 * (1 - t'/t)
      = forward_process(x0, x0_to_eps(x_t, x0, t), t')

so the student's own ODE step in _student_sample_loop *is* an Euler step with
velocity (x_t - x0_hat)/t, and the teacher micro-steps below use the same
mechanics.
"""

from __future__ import annotations

from typing import Callable

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]   # (x, t_batch) -> x0_hat


def shift_sigma(base: torch.Tensor, shift: float) -> torch.Tensor:
    base = torch.as_tensor(base, dtype=torch.float64)
    return shift * base / (1.0 + (shift - 1.0) * base)


def unshift_sigma(t: torch.Tensor, shift: float) -> torch.Tensor:
    t = torch.as_tensor(t, dtype=torch.float64)
    return t / (shift - (shift - 1.0) * t)


def boundary_aligned_teacher_grid(t_list: torch.Tensor, interval: int, micro_steps: int, shift: float) -> torch.Tensor:
    """Teacher micro-grid whose endpoints are exactly one student interval.

    Interpolation is uniform in *unshifted* base-sigma space, then shifted
    back, then the two endpoints are pinned to the student's boundaries. For
    Wan (shift 5, 4 steps, 50-step teacher) the first interval holds 13 native
    teacher steps, and with micro_steps=13 this grid tracks them almost
    point-for-point while still landing on t_list[1] exactly.
    """
    t_list = torch.as_tensor(t_list, dtype=torch.float64)
    if t_list.ndim != 1 or t_list.numel() < 2:
        raise ValueError("t_list must be 1-D with >= 2 entries")
    if not bool(torch.all(t_list[:-1] > t_list[1:])):
        raise ValueError("t_list must be strictly decreasing")
    if not 0 <= interval < t_list.numel() - 1:
        raise ValueError("interval out of range")
    if micro_steps <= 0:
        raise ValueError("micro_steps must be positive")
    a, b = t_list[interval], t_list[interval + 1]
    base = torch.linspace(float(unshift_sigma(a, shift)), float(unshift_sigma(b, shift)), micro_steps + 1, dtype=torch.float64)
    grid = shift_sigma(base, shift)
    grid[0], grid[-1] = a, b
    return grid


def secant_velocity(start: torch.Tensor, end: torch.Tensor, t_start: float, t_end: float) -> torch.Tensor:
    """The single Euler velocity that carries `start` at t_start to `end` at t_end."""
    dt = float(t_end) - float(t_start)
    if dt == 0.0:
        raise ValueError("t_start and t_end must differ")
    v = (end.float() - start.float()) / dt
    if not torch.isfinite(v).all():
        raise RuntimeError("secant velocity is not finite")
    return v


def euler_step_via_x0(scheduler, x: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
    """Advance x from t to t_next given an x0 estimate, using fastgen's own ops."""
    eps = scheduler.x0_to_eps(xt=x, x0=x0, t=t)
    return scheduler.forward_process(x0, eps, t_next)


def student_velocity(x: torch.Tensor, x0_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Velocity implied by an x0 estimate: (x_t - x0_hat)/t, in float32."""
    tt = t.to(torch.float32).view(-1, *([1] * (x.ndim - 1)))
    return (x.float() - x0_hat.float()) / tt


@torch.no_grad()
def solve_teacher_interval(
    predict_x0: VelocityFn,
    scheduler,
    x: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Integrate one student interval with teacher micro-steps on `grid`."""
    cur = x.detach()
    B = x.shape[0]
    for t, t_next in zip(grid[:-1], grid[1:]):
        tb = torch.full((B,), float(t), device=x.device, dtype=scheduler.t_precision)
        tnb = torch.full((B,), float(t_next), device=x.device, dtype=scheduler.t_precision)
        x0 = predict_x0(cur, tb)
        cur = euler_step_via_x0(scheduler, cur, x0, tb, tnb)
    return cur.detach()
