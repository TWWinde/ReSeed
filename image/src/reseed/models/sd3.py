"""Stable Diffusion 3 model adapter for native FlowMatch experiments.

The adapter exposes velocity prediction and prompt conditioning without
leaking SD3-specific pooled embeddings or CFG handling into the trajectory
solver.  Heavy Diffusers imports are intentionally lazy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from ..checkpoint import resolve_student_dir
from .sd3_contract import (
    SD3ContractError,
    SD3FamilyContract,
    SD3_FAMILY_CONTRACTS,
    fingerprint_sd3_snapshot,
    validate_sd3_pipeline_contract,
    validate_sd3_snapshot_contract,
)


@dataclass(frozen=True)
class SD3Conditioning:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None = None
    negative_pooled_prompt_embeds: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.prompt_embeds.shape[0])

    @property
    def has_unconditional(self) -> bool:
        return self.negative_prompt_embeds is not None and self.negative_pooled_prompt_embeds is not None


def encode_sd3_prompts(
    pipe: Any,
    prompts: list[str],
    *,
    device: torch.device,
    max_sequence_length: int = 256,
    include_unconditional: bool = True,
) -> SD3Conditioning:
    """Encode all three SD3 text-conditioning branches through the pipeline."""

    if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a non-empty list of strings")
    negative = [""] * len(prompts) if include_unconditional else None
    with torch.no_grad():
        prompt_embeds, negative_embeds, pooled, negative_pooled = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=include_unconditional,
            negative_prompt=negative,
            negative_prompt_2=None,
            negative_prompt_3=None,
            clip_skip=None,
            max_sequence_length=max_sequence_length,
        )
    return SD3Conditioning(prompt_embeds, pooled, negative_embeds, negative_pooled)


def _model_velocity(
    model: Any,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    output = model(
        hidden_states=latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]
    if output.shape != latents.shape:
        raise ValueError(
            f"SD3 velocity shape {tuple(output.shape)} does not match latent shape {tuple(latents.shape)}"
        )
    return output


def predict_sd3_velocity(
    model: Any,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    conditioning: SD3Conditioning,
    *,
    guidance_scale: float | torch.Tensor,
    num_train_timesteps: int = 1000,
    cfg_control: float | None = None,
) -> torch.Tensor:
    """Predict the guided SD3 FlowMatch velocity at shifted sigma coordinates.

    CFG is applied directly to velocity predictions.  No epsilon conversion,
    alpha schedule, or predicted-clean-sample formula is involved.
    """

    sigmas = torch.as_tensor(sigmas, device=latents.device, dtype=latents.dtype).reshape(-1)
    if sigmas.numel() == 1:
        sigmas = sigmas.expand(latents.shape[0])
    if sigmas.numel() != latents.shape[0]:
        raise ValueError("sigmas must be scalar or have one value per latent")
    if conditioning.batch_size != latents.shape[0]:
        raise ValueError("conditioning batch size does not match latents")
    if bool(torch.any((sigmas < 0) | (sigmas > 1))):
        raise ValueError("SD3 shifted sigmas must be in [0, 1]")
    timesteps = sigmas * float(num_train_timesteps)
    conditional = _model_velocity(
        model,
        latents,
        timesteps,
        conditioning.prompt_embeds,
        conditioning.pooled_prompt_embeds,
    )
    guidance = torch.as_tensor(
        guidance_scale, device=latents.device, dtype=torch.float32
    ).reshape(-1)
    if guidance.numel() == 1:
        guidance = guidance.expand(latents.shape[0])
    if guidance.numel() != latents.shape[0]:
        raise ValueError("guidance_scale must be scalar or have one value per latent")
    if not torch.isfinite(guidance).all() or bool((guidance <= 0).any()):
        raise ValueError("guidance_scale must be finite and positive")
    if bool((guidance == 1.0).all()):
        return conditional
    guidance_mask = None
    if cfg_control is not None:
        if not 0.0 <= cfg_control <= 1.0:
            raise ValueError("cfg_control must be in [0, 1]")
        guidance_mask = sigmas <= cfg_control
        # Match sd_distill's ``if mask.any()`` guard: the very noisy region
        # skips the unconditional model call entirely.
        if not bool(guidance_mask.any()):
            return conditional
    if not conditioning.has_unconditional:
        raise ValueError("guidance_scale != 1 requires unconditional SD3 embeddings")
    unconditional = _model_velocity(
        model,
        latents,
        timesteps,
        conditioning.negative_prompt_embeds,
        conditioning.negative_pooled_prompt_embeds,
    )
    guidance = guidance.to(dtype=conditional.dtype).reshape(
        latents.shape[0], *((1,) * (latents.ndim - 1))
    )
    guided = unconditional + guidance * (conditional - unconditional)
    if guidance_mask is None:
        return guided
    # sd_distill Stage-1 only enables teacher CFG at and below this shifted
    # sigma threshold. Student and fake models still take one conditional
    # forward because their guidance scale is fixed to one.
    mask = guidance_mask.reshape(
        latents.shape[0], *((1,) * (latents.ndim - 1))
    )
    return torch.where(mask, guided, conditional)


def load_sd3_pipeline(model_path: str | Path, *, torch_dtype: torch.dtype) -> Any:
    from diffusers import StableDiffusion3Pipeline

    return StableDiffusion3Pipeline.from_pretrained(str(model_path), torch_dtype=torch_dtype)


def _looks_like_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def resolve_sd3_transformer_path(path: str | Path, *, subfolder: str = "transformer") -> tuple[str, str | None]:
    """Resolve base-model, canonical-checkpoint, or direct transformer paths."""

    source = Path(path).expanduser()
    if _looks_like_model_dir(source):
        return str(source), None
    if source.is_dir():
        for name in ("student", "transformer"):
            candidate = source / name
            if _looks_like_model_dir(candidate):
                return str(candidate), None
        try:
            return str(resolve_student_dir(source)), None
        except FileNotFoundError:
            pass
    return str(path), subfolder


def load_sd3_transformer(
    path: str | Path,
    *,
    torch_dtype: torch.dtype,
    subfolder: str = "transformer",
) -> Any:
    from diffusers import SD3Transformer2DModel

    resolved, resolved_subfolder = resolve_sd3_transformer_path(path, subfolder=subfolder)
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if resolved_subfolder is not None:
        kwargs["subfolder"] = resolved_subfolder
    return SD3Transformer2DModel.from_pretrained(resolved, **kwargs)


def sd3_latent_shape(
    transformer: Any,
    vae_scale_factor: int,
    *,
    batch_size: int,
    resolution: int,
) -> tuple[int, int, int, int]:
    if resolution <= 0 or resolution % vae_scale_factor != 0:
        raise ValueError("resolution must be positive and divisible by the VAE scale factor")
    channels = int(transformer.config.in_channels)
    spatial = resolution // int(vae_scale_factor)
    return batch_size, channels, spatial, spatial


def decode_sd3_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Decode SD3 latents to float RGB images in [0, 1]."""

    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    shift_factor = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    decoded = vae.decode(latents / scaling_factor + shift_factor, return_dict=False)[0]
    return (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)


def encode_sd3_images(vae: Any, images: torch.Tensor) -> torch.Tensor:
    """Encode RGB tensors in ``[-1, 1]`` into the native SD3 latent space.

    The posterior mode is used so an offline real-image reference dataset is
    deterministic.  This is the exact inverse scaling convention used by
    :func:`decode_sd3_latents`.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("SD3 image encoder expects [B,3,H,W]")
    posterior = vae.encode(images, return_dict=True).latent_dist
    raw_latents = posterior.mode()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    shift_factor = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    return (raw_latents - shift_factor) * scaling_factor


def select_sd3_trainable_blocks(student: Any, configured: str | Iterable[int]) -> list[torch.nn.Parameter]:
    """Enable all SD3 parameters or an explicit transformer-block subset."""

    blocks = student.transformer_blocks
    if configured == "all":
        student.requires_grad_(True)
        parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("SD3 model contains no trainable parameters")
        return parameters
    indices = []
    for raw in configured:
        index = int(raw)
        if index < 0:
            index += len(blocks)
        if not 0 <= index < len(blocks):
            raise ValueError(f"trainable block {raw} is outside [0, {len(blocks) - 1}]")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("at least one SD3 transformer block must be trainable")
    student.requires_grad_(False)
    for index in indices:
        blocks[index].requires_grad_(True)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("selected SD3 blocks contain no trainable parameters")
    return parameters


__all__ = [
    "SD3Conditioning",
    "SD3ContractError",
    "SD3FamilyContract",
    "SD3_FAMILY_CONTRACTS",
    "decode_sd3_latents",
    "encode_sd3_prompts",
    "fingerprint_sd3_snapshot",
    "load_sd3_pipeline",
    "load_sd3_transformer",
    "predict_sd3_velocity",
    "resolve_sd3_transformer_path",
    "sd3_latent_shape",
    "select_sd3_trainable_blocks",
    "validate_sd3_pipeline_contract",
    "validate_sd3_snapshot_contract",
]
