"""Teacher-guided feature diversity objectives and gradient balancing.

The objective in this module is intentionally model agnostic.  A caller owns
the student/teacher rollouts and supplies four feature tensors.  This keeps the
PixArt training entry point focused on orchestration and makes the mathematical
contract directly unit-testable.
"""

from __future__ import annotations

from typing import Callable, Iterable, MutableMapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def teacher_guided_diversity_terms(
    student_first: Tensor,
    student_second: Tensor,
    teacher_first: Tensor,
    teacher_second: Tensor,
    *,
    gamma: float = 0.85,
    direction_weight: float = 0.25,
    teacher_delta_tau: float = 0.1,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Compute the independent-seed diversity floor and direction objective.

    All inputs have shape ``[B, F]``.  Teacher tensors are detached here even
    if the caller already produced them under ``torch.no_grad()``.
    """

    tensors = (student_first, student_second, teacher_first, teacher_second)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("Diversity features must all have shape [batch, feature].")
    if len({tuple(tensor.shape) for tensor in tensors}) != 1:
        raise ValueError("Student and teacher diversity feature shapes must match.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if direction_weight < 0.0 or teacher_delta_tau <= 0.0 or eps <= 0.0:
        raise ValueError("direction_weight must be non-negative; tau and eps must be positive.")

    student_first = F.normalize(student_first.float(), dim=-1, eps=eps)
    student_second = F.normalize(student_second.float(), dim=-1, eps=eps)
    teacher_first = F.normalize(teacher_first.detach().float(), dim=-1, eps=eps)
    teacher_second = F.normalize(teacher_second.detach().float(), dim=-1, eps=eps)

    student_distance = 1.0 - (student_first * student_second).sum(dim=-1)
    teacher_distance = 1.0 - (teacher_first * teacher_second).sum(dim=-1)
    floor_deficit = F.relu(gamma * teacher_distance - student_distance)
    floor_per_sample = (floor_deficit / (teacher_distance + eps)).square()
    loss_floor = floor_per_sample.mean()

    student_delta = student_first - student_second
    teacher_delta = teacher_first - teacher_second
    teacher_delta_norm = teacher_delta.norm(dim=-1)
    teacher_confidence = (teacher_delta_norm / teacher_delta_tau).clamp(max=1.0)
    direction_cosine = F.cosine_similarity(
        student_delta,
        teacher_delta,
        dim=-1,
        eps=eps,
    )
    loss_direction = (teacher_confidence * (1.0 - direction_cosine)).mean()
    loss_total = loss_floor + direction_weight * loss_direction

    outputs = {
        "loss_total": loss_total,
        "loss_floor": loss_floor,
        "loss_direction": loss_direction,
        "student_distance": student_distance.mean(),
        "teacher_distance": teacher_distance.mean(),
        "floor_active_fraction": (floor_deficit > 0).float().mean(),
        "direction_cosine": direction_cosine.mean(),
        "teacher_confidence": teacher_confidence.mean(),
        "student_delta_norm": student_delta.norm(dim=-1).mean(),
        "teacher_delta_norm": teacher_delta_norm.mean(),
    }
    if not all(torch.isfinite(value) for value in outputs.values()):
        raise FloatingPointError("Non-finite teacher-guided diversity objective.")
    return outputs


class FrozenDINOStateEncoder(nn.Module):
    """Differentiably decode latent states and extract normalized DINO features."""

    def __init__(
        self,
        *,
        state_to_rgb: Callable[[Tensor], Tensor],
        feature_model: nn.Module,
        image_size: int = 224,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        checkpoint_features: bool = True,
    ) -> None:
        super().__init__()
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("DINO normalization requires three-channel mean/std.")
        self.state_to_rgb = state_to_rgb
        self.feature_model = feature_model.eval().requires_grad_(False)
        self.image_size = int(image_size)
        self.checkpoint_features = bool(checkpoint_features)
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _forward_impl(self, states: Tensor) -> Tensor:
        rgb = self.state_to_rgb(states).float()
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Decoded states must have shape [B,3,H,W], got {tuple(rgb.shape)}")
        pixels = (rgb.clamp(-1.0, 1.0) + 1.0) * 0.5
        pixels = F.interpolate(
            pixels,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.image_mean) / self.image_std
        output = self.feature_model(pixel_values=pixels)
        features = getattr(output, "pooler_output", None)
        if features is None:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None or hidden.ndim != 3:
                raise RuntimeError("DINO output has neither pooler_output nor valid last_hidden_state.")
            features = hidden[:, 0]
        if features.ndim != 2 or features.shape[0] != states.shape[0]:
            raise RuntimeError(f"Invalid DINO feature shape: {tuple(features.shape)}")
        return F.normalize(features.float(), dim=-1)

    def forward(self, states: Tensor) -> Tensor:
        if self.checkpoint_features and states.requires_grad:
            return checkpoint(self._forward_impl, states, use_reentrant=False)
        return self._forward_impl(states)


def capture_parameter_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    clear: bool = False,
) -> list[Optional[Tensor]]:
    """Clone the current gradients, optionally freeing them from parameters."""

    captured: list[Optional[Tensor]] = []
    for parameter in parameters:
        captured.append(None if parameter.grad is None else parameter.grad.detach().clone())
        if clear:
            parameter.grad = None
    return captured


def _gradient_statistics(
    quality: Sequence[Optional[Tensor]],
    diversity: Sequence[Optional[Tensor]],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    quality_squared = torch.zeros((), device=device, dtype=torch.float32)
    diversity_squared = torch.zeros_like(quality_squared)
    dot = torch.zeros_like(quality_squared)
    for quality_grad, diversity_grad in zip(quality, diversity):
        if quality_grad is not None:
            quality_squared = quality_squared + quality_grad.float().square().sum()
        if diversity_grad is not None:
            diversity_squared = diversity_squared + diversity_grad.float().square().sum()
        if quality_grad is not None and diversity_grad is not None:
            dot = dot + (quality_grad.float() * diversity_grad.float()).sum()
    return quality_squared, diversity_squared, dot


def merge_quality_and_diversity_gradients(
    parameters: Iterable[nn.Parameter],
    quality_gradients: Sequence[Optional[Tensor]],
    controller_state: MutableMapping[str, float | int],
    *,
    target_ratio: float,
    ema_decay: float = 0.9,
    max_scale: float = 10.0,
    pcgrad: bool = False,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Balance separately-backpropagated diversity gradients and merge once.

    At entry, ``parameter.grad`` contains the diversity gradient while
    ``quality_gradients`` contains the saved TDM/frequency gradient.  Both have
    passed through the same Accelerate/GradScaler path, so their norm ratio and
    cosine are invariant to AMP scaling.  The merged gradient is installed in
    ``parameter.grad`` for the caller's single optimizer step.
    """

    parameters = list(parameters)
    if len(parameters) != len(quality_gradients):
        raise ValueError("Gradient list does not match the parameter list.")
    if not 0.0 <= target_ratio < 1.0:
        raise ValueError("target_ratio must be in [0, 1).")
    if not 0.0 <= ema_decay < 1.0 or max_scale <= 0.0:
        raise ValueError("ema_decay must be in [0,1) and max_scale must be positive.")

    diversity_gradients = capture_parameter_gradients(parameters, clear=True)
    device = parameters[0].device if parameters else torch.device("cpu")
    quality_sq, diversity_sq, dot = _gradient_statistics(
        quality_gradients,
        diversity_gradients,
        device=device,
    )
    quality_norm = quality_sq.sqrt()
    diversity_norm = diversity_sq.sqrt()
    if not torch.isfinite(quality_norm) or not torch.isfinite(diversity_norm) or not torch.isfinite(dot):
        raise FloatingPointError("Non-finite quality/diversity gradient statistics.")
    cosine = dot / (quality_norm * diversity_norm + eps)
    conflict = bool(dot.item() < 0.0)

    projected_gradients = diversity_gradients
    pcgrad_applied = bool(pcgrad and conflict and quality_sq.item() > eps)
    if pcgrad_applied:
        projection = dot / (quality_sq + eps)
        projected_gradients = []
        for quality_grad, diversity_grad in zip(quality_gradients, diversity_gradients):
            if diversity_grad is None:
                projected_gradients.append(None)
            elif quality_grad is None:
                projected_gradients.append(diversity_grad)
            else:
                projected_gradients.append(diversity_grad - projection.to(diversity_grad.dtype) * quality_grad)

    _, projected_sq, projected_dot = _gradient_statistics(
        quality_gradients,
        projected_gradients,
        device=device,
    )
    projected_norm = projected_sq.sqrt()
    updates = int(controller_state.get("updates", 0))
    previous_quality = float(controller_state.get("ema_quality_grad_norm", 0.0))
    previous_diversity = float(controller_state.get("ema_diversity_grad_norm", 0.0))
    if updates == 0:
        ema_quality = float(quality_norm.detach())
        ema_diversity = float(projected_norm.detach())
    else:
        ema_quality = ema_decay * previous_quality + (1.0 - ema_decay) * float(quality_norm.detach())
        ema_diversity = ema_decay * previous_diversity + (1.0 - ema_decay) * float(projected_norm.detach())
    controller_state["ema_quality_grad_norm"] = ema_quality
    controller_state["ema_diversity_grad_norm"] = ema_diversity
    controller_state["updates"] = updates + 1

    effective_scale = min(max_scale, target_ratio * ema_quality / (ema_diversity + eps))
    for parameter, quality_grad, diversity_grad in zip(
        parameters, quality_gradients, projected_gradients
    ):
        if quality_grad is None and diversity_grad is None:
            parameter.grad = None
        elif quality_grad is None:
            parameter.grad = diversity_grad.mul(effective_scale)
        elif diversity_grad is None:
            parameter.grad = quality_grad
        else:
            parameter.grad = quality_grad + diversity_grad.mul(effective_scale)

    actual_ratio = effective_scale * projected_norm / (quality_norm + eps)
    return {
        "quality_grad_norm": quality_norm.detach(),
        "diversity_grad_norm": diversity_norm.detach(),
        "projected_diversity_grad_norm": projected_norm.detach(),
        "gradient_cosine": cosine.detach(),
        "projected_gradient_dot": projected_dot.detach(),
        "gradient_conflict": torch.tensor(float(conflict), device=device),
        "pcgrad_applied": torch.tensor(float(pcgrad_applied), device=device),
        "effective_scale": torch.tensor(effective_scale, device=device),
        "actual_ratio": actual_ratio.detach(),
        "ema_quality_grad_norm": torch.tensor(ema_quality, device=device),
        "ema_diversity_grad_norm": torch.tensor(ema_diversity, device=device),
    }
