"""Flow-matching trajectory primitives for SD3-family experiments.

SD3 predicts a velocity field.  Its deterministic Euler transition is

    x(next_sigma) = x(sigma) + (next_sigma - sigma) * velocity(x, sigma)

This module deliberately does not import or reuse epsilon/DDIM helpers.  The two parameterizations have different model outputs,
time coordinates, and transition equations.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch


VelocityPredictor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _elementwise_loss_and_gradient(
    residual: torch.Tensor,
    *,
    loss_kind: str,
    huber_c: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a mean loss and its FP32 gradient with respect to ``residual``."""

    value = residual.float()
    if loss_kind == "mse":
        return value.square().mean(), 2.0 * value / value.numel()
    if loss_kind != "huber":
        raise ValueError("loss_kind must be 'mse' or 'huber'")
    if huber_c <= 0:
        raise ValueError("huber_c must be positive")
    root = torch.sqrt(value.square() + huber_c**2)
    return (root - huber_c).mean(), value / root / value.numel()


def flow_antipodal_teacher_loss_and_endpoint_gradients(
    student_plus: torch.Tensor,
    student_minus: torch.Tensor,
    teacher_plus: torch.Tensor,
    teacher_minus: torch.Tensor,
    *,
    pair_lambda: float,
    loss_kind: str = "huber",
    huber_c: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Compute the SD3 FlowMatch ``(z,-z)`` teacher-response objective.

    The returned endpoint gradients allow the trainer to recompute and
    backpropagate the ``+z`` and ``-z`` student transitions separately.  This
    is algebraically identical to differentiating the joint loss, but avoids
    retaining two 1024px SD3 transformer graphs at once.
    """

    shapes = {
        tuple(student_plus.shape),
        tuple(student_minus.shape),
        tuple(teacher_plus.shape),
        tuple(teacher_minus.shape),
    }
    if len(shapes) != 1:
        raise ValueError("all antipodal endpoints must have the same shape")
    if pair_lambda < 0:
        raise ValueError("pair_lambda must be non-negative")

    plus_residual = student_plus.float() - teacher_plus.detach().float()
    minus_residual = student_minus.float() - teacher_minus.detach().float()
    plus_loss, plus_grad = _elementwise_loss_and_gradient(
        plus_residual, loss_kind=loss_kind, huber_c=huber_c
    )
    minus_loss, minus_grad = _elementwise_loss_and_gradient(
        minus_residual, loss_kind=loss_kind, huber_c=huber_c
    )
    anchor = 0.5 * (plus_loss + minus_loss)
    anchor_plus_grad = 0.5 * plus_grad
    anchor_minus_grad = 0.5 * minus_grad

    student_response = student_plus.float() - student_minus.float()
    teacher_response = teacher_plus.detach().float() - teacher_minus.detach().float()
    response_residual = student_response - teacher_response
    pair, response_grad = _elementwise_loss_and_gradient(
        response_residual, loss_kind=loss_kind, huber_c=huber_c
    )
    total = anchor + pair_lambda * pair
    return {
        "loss_anchor": anchor.detach(),
        "loss_pair": pair.detach(),
        "loss_total": total.detach(),
        "student_state_rms": torch.cat(
            [student_plus.float(), student_minus.float()], dim=0
        ).square().mean().sqrt().detach(),
        "teacher_state_rms": torch.cat(
            [teacher_plus.float(), teacher_minus.float()], dim=0
        ).square().mean().sqrt().detach(),
        "response_error_rms": response_residual.square().mean().sqrt().detach(),
        "grad_plus": (anchor_plus_grad + pair_lambda * response_grad).detach(),
        "grad_minus": (anchor_minus_grad - pair_lambda * response_grad).detach(),
    }


def shift_sigma(base_sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the SD3 FlowMatch rational sigma shift."""

    if shift <= 0:
        raise ValueError("shift must be positive")
    base_sigma = torch.as_tensor(base_sigma)
    if bool(torch.any((base_sigma < 0) | (base_sigma > 1))):
        raise ValueError("base sigma values must be in [0, 1]")
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


def unshift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Invert :func:`shift_sigma` on the closed interval [0, 1]."""

    if shift <= 0:
        raise ValueError("shift must be positive")
    sigma = torch.as_tensor(sigma)
    if bool(torch.any((sigma < 0) | (sigma > 1))):
        raise ValueError("shifted sigma values must be in [0, 1]")
    denominator = shift - (shift - 1.0) * sigma
    return sigma / denominator


def flow_sigma_boundaries(
    student_steps: int,
    shift: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return shifted SD3 sigma boundaries from pure noise to clean data."""

    if student_steps <= 0:
        raise ValueError("student_steps must be positive")
    base = torch.linspace(1.0, 0.0, student_steps + 1, device=device, dtype=torch.float32)
    boundaries = shift_sigma(base, shift).to(dtype=dtype)
    boundaries[0] = torch.as_tensor(1.0, device=device, dtype=dtype)
    boundaries[-1] = torch.as_tensor(0.0, device=device, dtype=dtype)
    return boundaries


def boundary_aligned_teacher_sigmas(
    boundaries: torch.Tensor,
    interval_index: int,
    micro_steps: int,
    shift: float,
) -> torch.Tensor:
    """Build a teacher grid sharing an exact student macro interval.

    Interpolation happens in unshifted base-sigma space.  Interpolating the
    already shifted values would change the SD3 FlowMatch time geometry.
    """

    if boundaries.ndim != 1 or boundaries.numel() < 2:
        raise ValueError("boundaries must be a one-dimensional tensor with at least two entries")
    if not 0 <= interval_index < boundaries.numel() - 1:
        raise ValueError("interval_index is outside the student schedule")
    if micro_steps <= 0:
        raise ValueError("micro_steps must be positive")
    if not bool(torch.all(boundaries[:-1] > boundaries[1:])):
        raise ValueError("FlowMatch boundaries must be strictly decreasing")

    start = boundaries[interval_index].float()
    end = boundaries[interval_index + 1].float()
    start_base = unshift_sigma(start, shift)
    end_base = unshift_sigma(end, shift)
    base_grid = torch.linspace(
        float(start_base.item()),
        float(end_base.item()),
        micro_steps + 1,
        device=boundaries.device,
        dtype=torch.float32,
    )
    grid = shift_sigma(base_grid, shift).to(dtype=boundaries.dtype)
    grid[0] = boundaries[interval_index]
    grid[-1] = boundaries[interval_index + 1]
    return grid


def _batch_sigma(value: torch.Tensor | float, sample: torch.Tensor) -> torch.Tensor:
    sigma = torch.as_tensor(value, device=sample.device, dtype=sample.dtype)
    if sigma.numel() == 1:
        sigma = sigma.reshape(1).expand(sample.shape[0])
    else:
        sigma = sigma.reshape(-1)
    if sigma.numel() != sample.shape[0]:
        raise ValueError("sigma must be scalar or have one value per sample")
    return sigma


def flow_velocity_to_clean(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Convert an SD3 rectified-flow velocity prediction to clean ``x0``.

    SD3 uses ``x_sigma = (1-sigma) * x0 + sigma * epsilon`` and
    ``v = epsilon - x0``.  Eliminating epsilon gives the exact conversion
    ``x0 = x_sigma - sigma * v``.  This representation is required when DFD
    queries teacher and fake-score at *different* noisy locations: subtracting
    their raw velocities would omit the difference between those locations.
    """

    if velocity.shape != sample.shape:
        raise ValueError(
            f"Flow velocity shape {tuple(velocity.shape)} does not match sample "
            f"{tuple(sample.shape)}"
        )
    sigma_batch = _batch_sigma(sigma, sample)
    if bool(torch.any((sigma_batch < 0) | (sigma_batch > 1))):
        raise ValueError("sigma must be in [0, 1]")
    sigma_view = sigma_batch.reshape(
        sample.shape[0], *((1,) * (sample.ndim - 1))
    )
    return sample - sigma_view * velocity


def flow_euler_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
    next_sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Advance one deterministic FlowMatch Euler step."""

    if velocity.shape != sample.shape:
        raise ValueError(
            f"Flow velocity shape {tuple(velocity.shape)} does not match sample {tuple(sample.shape)}"
        )
    current = _batch_sigma(sigma, sample)
    following = _batch_sigma(next_sigma, sample)
    if bool(torch.any(following > current)):
        raise ValueError("generation must move from larger sigma to smaller sigma")
    delta = (following - current).reshape(sample.shape[0], *((1,) * (sample.ndim - 1)))
    return sample + delta * velocity


def student_flow_interval(
    predict_velocity: VelocityPredictor,
    latents: torch.Tensor,
    *,
    sigma: torch.Tensor | float,
    next_sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate one differentiable student macro transition."""

    sigma_batch = _batch_sigma(sigma, latents)
    velocity = predict_velocity(latents, sigma_batch)
    return flow_euler_step(latents, velocity, sigma_batch, next_sigma)


def flow_interval_secant_velocity(
    start: torch.Tensor,
    endpoint: torch.Tensor,
    *,
    sigma: torch.Tensor | float,
    next_sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Return the single Euler velocity that lands exactly on ``endpoint``.

    For the FlowMatch convention ``dx / d sigma = v``, this is the secant
    velocity over one student macro interval, not the teacher's instantaneous
    velocity at either endpoint.
    """

    if start.shape != endpoint.shape:
        raise ValueError("start and endpoint must have identical shapes")
    sigma_batch = _batch_sigma(sigma, start)
    next_sigma_batch = _batch_sigma(next_sigma, start)
    delta = next_sigma_batch - sigma_batch
    if bool((delta == 0).any()):
        raise ValueError("sigma and next_sigma must differ")
    velocity = (endpoint - start) / delta.reshape(
        start.shape[0], *((1,) * (start.ndim - 1))
    )
    if not torch.isfinite(velocity).all():
        raise RuntimeError("secant velocity contains NaN or Inf")
    return velocity


@torch.no_grad()
def solve_teacher_flow_interval(
    predict_velocity: VelocityPredictor,
    latents: torch.Tensor,
    *,
    boundaries: torch.Tensor,
    interval_index: int,
    micro_steps: int,
    shift: float,
) -> torch.Tensor:
    """Integrate one student interval with endpoint-aligned teacher microsteps."""

    grid = boundary_aligned_teacher_sigmas(boundaries, interval_index, micro_steps, shift)
    current = latents.detach()
    for sigma, next_sigma in zip(grid[:-1], grid[1:]):
        sigma_batch = _batch_sigma(sigma, current)
        velocity = predict_velocity(current, sigma_batch)
        current = flow_euler_step(current, velocity, sigma_batch, next_sigma)
    return current.detach()


@torch.no_grad()
def solve_teacher_flow_rollout(
    predict_velocity: VelocityPredictor,
    noise: torch.Tensor,
    *,
    boundaries: torch.Tensor,
    micro_steps: int,
    shift: float,
) -> list[torch.Tensor]:
    """Return frozen-teacher states at every student boundary."""

    states = [noise.detach()]
    current = noise.detach()
    for interval_index in range(boundaries.numel() - 1):
        current = solve_teacher_flow_interval(
            predict_velocity,
            current,
            boundaries=boundaries,
            interval_index=interval_index,
            micro_steps=micro_steps,
            shift=shift,
        )
        states.append(current.detach())
    return states


@torch.no_grad()
def solve_student_flow_rollout(
    predict_velocity: VelocityPredictor,
    noise: torch.Tensor,
    *,
    boundaries: Sequence[float] | torch.Tensor,
) -> list[torch.Tensor]:
    """Return detached student states at every FlowMatch boundary."""

    boundaries = torch.as_tensor(boundaries, device=noise.device, dtype=noise.dtype)
    states = [noise.detach()]
    current = noise.detach()
    for sigma, next_sigma in zip(boundaries[:-1], boundaries[1:]):
        current = student_flow_interval(
            predict_velocity,
            current,
            sigma=sigma,
            next_sigma=next_sigma,
        )
        states.append(current.detach())
    return states


@torch.no_grad()
def solve_teacher_prefix_student_suffix(
    teacher_velocity: VelocityPredictor,
    student_velocity: VelocityPredictor,
    noise: torch.Tensor,
    *,
    boundaries: torch.Tensor,
    teacher_prefix_intervals: int,
    teacher_micro_steps: int,
    shift: float,
) -> list[torch.Tensor]:
    """Use aligned teacher microsteps for a prefix, then student macrosteps.

    The returned list contains the initial noise and one state at every
    student boundary.  The suffix stays on the original student grid instead
    of restarting a shortened scheduler at the switch point.
    """

    interval_count = boundaries.numel() - 1
    if not 0 < teacher_prefix_intervals < interval_count:
        raise ValueError("teacher_prefix_intervals must leave at least one student interval")
    states = [noise.detach()]
    current = noise.detach()
    for interval_index in range(interval_count):
        if interval_index < teacher_prefix_intervals:
            current = solve_teacher_flow_interval(
                teacher_velocity,
                current,
                boundaries=boundaries,
                interval_index=interval_index,
                micro_steps=teacher_micro_steps,
                shift=shift,
            )
        else:
            current = student_flow_interval(
                student_velocity,
                current,
                sigma=boundaries[interval_index],
                next_sigma=boundaries[interval_index + 1],
            ).detach()
        states.append(current.detach())
    return states


__all__ = [
    "VelocityPredictor",
    "boundary_aligned_teacher_sigmas",
    "flow_euler_step",
    "flow_interval_secant_velocity",
    "flow_velocity_to_clean",
    "flow_sigma_boundaries",
    "shift_sigma",
    "solve_student_flow_rollout",
    "solve_teacher_prefix_student_suffix",
    "solve_teacher_flow_interval",
    "solve_teacher_flow_rollout",
    "student_flow_interval",
    "unshift_sigma",
]
