"""FLUX.1 model adapter for native FlowMatch experiments.

Physical-noise-level convention
-------------------------------
FLUX is a rectified-flow model in the *same* convention as SD3, with sigma
running from 1 (pure noise) down to 0 (clean data):

    x_sigma = (1 - sigma) * x0 + sigma * epsilon          (forward / scale_noise)
    v       = epsilon - x0                                (model prediction)
    x_sigma' = x_sigma + (sigma' - sigma) * v             (deterministic Euler)

This matches ``FlowMatchEulerDiscreteScheduler`` exactly:
``scale_noise`` computes ``sigma * noise + (1 - sigma) * sample`` and ``step``
computes ``prev_sample = sample + (sigma_next - sigma) * model_output``.

Teacher microsteps for the student's first interval are placed in sigma
*between the student's first two boundaries*, equally spaced in the student's
unshifted (base) time -- the FLUX analogue of
``flow_matching.boundary_aligned_teacher_sigmas``.  The student's shift is
inverted to base time, the microstep grid is linspaced there, and the grid is
re-shifted.  FLUX.1-schnell's scheduler has ``use_dynamic_shifting = false`` and
``shift = 1.0``, so its shift is the identity and the aligned grid degenerates
to equal spacing directly in sigma; the general code path is still used so the
same helper works for a shifted student.

What was reproduced from diffusers 0.34 ``FluxPipeline`` / scheduler
--------------------------------------------------------------------
* Prompt encoding: CLIP-L **pooled** output (``tokenizer.model_max_length`` = 77,
  padded to max_length) plus the T5-XXL last hidden state at
  ``max_sequence_length`` (512 for dev, 256 for schnell).  ``text_ids`` is a
  **2-D** ``zeros(prompt_embeds.shape[1], 3)`` tensor -- no batch axis.
* Latents: ``num_channels_latents = transformer.config.in_channels // 4`` = 16,
  ``height = 2 * (H // (vae_scale_factor * 2))``.  At 1024x1024 with
  ``vae_scale_factor = 8`` that is a 16x128x128 latent, packed by
  ``_pack_latents`` into 4096 tokens of 64 channels.
* Sigma schedule: ``sigmas = linspace(1, 1/steps, steps)``,
  ``mu = calculate_shift(image_seq_len, ...)``,
  ``scheduler.set_timesteps(sigmas=sigmas, mu=mu)``.  With
  ``use_dynamic_shifting`` the scheduler applies the exponential time shift
  ``exp(mu) / (exp(mu) + (1/t - 1))``; without it, the static rational shift
  ``shift * t / (1 + (shift - 1) * t)``.  A terminal ``0.0`` is appended.
* Timestep fed to the transformer: the pipeline computes
  ``timestep = t.expand(B).to(latents.dtype)`` (t = sigma * 1000, fp32) and then
  passes ``timestep / 1000``.  The **cast to the latent dtype happens before the
  division**, so at bf16 the transformer sees ``bf16(sigma * 1000) / 1000``, not
  ``bf16(sigma)``.  ``predict_flux_velocity`` reproduces that rounding exactly.
* Guidance: dev has ``guidance_embeds = true`` and gets
  ``torch.full([1], guidance_scale, dtype=fp32).expand(B)`` (the transformer
  multiplies it by 1000 internally); schnell has ``guidance_embeds = false`` and
  gets ``None``.
* Euler step: the scheduler upcasts the sample to fp32, adds
  ``(sigma_next - sigma) * model_output``, and casts back to the model dtype.
* VAE decode: ``unpack``, then ``latents / scaling_factor + shift_factor``,
  then ``vae.decode``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch


FLUX_NUM_TRAIN_TIMESTEPS = 1000
FLUX_DEFAULT_MAX_SEQUENCE_LENGTH = {"dev": 512, "schnell": 256}
FLUX_DEFAULT_STEPS = {"dev": 28, "schnell": 4}
FLUX_DEFAULT_GUIDANCE = {"dev": 3.5, "schnell": None}


# --------------------------------------------------------------------------- #
# conditioning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FluxConditioning:
    """The three text-conditioning tensors a FLUX transformer call needs."""

    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.prompt_embeds.shape[0])

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.prompt_embeds
        yield self.pooled_prompt_embeds
        yield self.text_ids


def encode_flux_prompts(
    pipe: Any,
    prompts: list[str],
    *,
    max_sequence_length: int,
    device: torch.device | None = None,
) -> FluxConditioning:
    """Encode prompts through the pipeline's CLIP-L and T5-XXL branches.

    FLUX has no unconditional branch in the distilled-guidance setting: dev
    carries guidance in an embedding and schnell carries none, so neither model
    runs true classifier-free guidance here.
    """

    if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a non-empty list of strings")
    if not 0 < int(max_sequence_length) <= 512:
        raise ValueError("max_sequence_length must be in (0, 512]")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=int(max_sequence_length),
        )
    if text_ids.ndim != 2:
        raise ValueError(
            f"FLUX text_ids must be 2-D (seq, 3); got {tuple(text_ids.shape)}"
        )
    return FluxConditioning(prompt_embeds, pooled_prompt_embeds, text_ids)


# --------------------------------------------------------------------------- #
# packing / geometry
# --------------------------------------------------------------------------- #
def flux_latent_spatial(size: int, vae_scale_factor: int) -> int:
    """Reproduce the pipeline's ``2 * (size // (vae_scale_factor * 2))``."""

    return 2 * (int(size) // (int(vae_scale_factor) * 2))


def flux_latent_shape(
    height: int, width: int, *, latent_channels: int = 16, vae_scale_factor: int = 8
) -> tuple[int, int, int]:
    """Unpacked latent shape ``(C, H, W)`` -- the shape the fp32 noise is drawn in."""

    return (
        int(latent_channels),
        flux_latent_spatial(height, vae_scale_factor),
        flux_latent_spatial(width, vae_scale_factor),
    )


def flux_image_seq_len(
    height: int, width: int, *, vae_scale_factor: int = 8
) -> int:
    """Number of packed image tokens: ``(h // 2) * (w // 2)`` (4096 at 1024x1024)."""

    spatial_h = flux_latent_spatial(height, vae_scale_factor)
    spatial_w = flux_latent_spatial(width, vae_scale_factor)
    return (spatial_h // 2) * (spatial_w // 2)


def pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
    """``FluxPipeline._pack_latents``: (B,C,H,W) -> (B, H/2*W/2, C*4)."""

    if latents.ndim != 4:
        raise ValueError(f"expected a 4-D latent, got {tuple(latents.shape)}")
    batch_size, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError("FLUX packing requires even latent height and width")
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)


def unpack_flux_latents(
    latents: torch.Tensor, height: int, width: int, vae_scale_factor: int = 8
) -> torch.Tensor:
    """``FluxPipeline._unpack_latents``: (B, N, C*4) -> (B, C, H, W)."""

    if latents.ndim != 3:
        raise ValueError(f"expected a packed 3-D latent, got {tuple(latents.shape)}")
    batch_size, _num_patches, channels = latents.shape
    spatial_h = flux_latent_spatial(height, vae_scale_factor)
    spatial_w = flux_latent_spatial(width, vae_scale_factor)
    latents = latents.view(batch_size, spatial_h // 2, spatial_w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, spatial_h, spatial_w)


def prepare_flux_latent_image_ids(
    height: int,
    width: int,
    *,
    vae_scale_factor: int = 8,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``FluxPipeline._prepare_latent_image_ids`` for a full image size.

    The pipeline calls it with ``height // 2, width // 2`` of the *latent*
    grid, i.e. a 64x64 id grid at 1024x1024, and returns a 2-D ``(N, 3)``
    tensor with no batch axis.
    """

    grid_h = flux_latent_spatial(height, vae_scale_factor) // 2
    grid_w = flux_latent_spatial(width, vae_scale_factor) // 2
    latent_image_ids = torch.zeros(grid_h, grid_w, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(grid_h)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(grid_w)[None, :]
    return latent_image_ids.reshape(grid_h * grid_w, 3).to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def _config_get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def calculate_flux_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """``diffusers.pipelines.flux.pipeline_flux.calculate_shift`` verbatim.

    A linear interpolation of the shift between ``base_seq_len`` and
    ``max_seq_len``.  At 1024x1024 the packed sequence is exactly 4096 tokens,
    which is ``max_seq_len``, so ``mu`` lands exactly on ``max_shift`` = 1.15
    for both FLUX.1-dev and FLUX.1-schnell.
    """

    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def flux_time_shift(
    mu: float, sigma: float, t: np.ndarray, *, time_shift_type: str = "exponential"
) -> np.ndarray:
    """``FlowMatchEulerDiscreteScheduler.time_shift``."""

    if time_shift_type == "exponential":
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    if time_shift_type == "linear":
        return mu / (mu + (1 / t - 1) ** sigma)
    raise ValueError("time_shift_type must be 'exponential' or 'linear'")


def flux_time_unshift(
    mu: float, t: np.ndarray, *, time_shift_type: str = "exponential"
) -> np.ndarray:
    """Invert :func:`flux_time_shift` at ``sigma = 1`` (the only case used).

    ``s = k / (k + (1/t - 1))``  =>  ``t = 1 / (k * (1/s - 1) + 1)``
    with ``k = exp(mu)`` for the exponential form and ``k = mu`` for the linear
    one.
    """

    scale = math.exp(mu) if time_shift_type == "exponential" else float(mu)
    if time_shift_type not in {"exponential", "linear"}:
        raise ValueError("time_shift_type must be 'exponential' or 'linear'")
    return 1.0 / (scale * (1.0 / t - 1.0) + 1.0)


def flux_static_shift(shift: float, t: np.ndarray) -> np.ndarray:
    """The non-dynamic branch: ``shift * t / (1 + (shift - 1) * t)``."""

    return shift * t / (1 + (shift - 1) * t)


def flux_static_unshift(shift: float, t: np.ndarray) -> np.ndarray:
    """Invert :func:`flux_static_shift`: ``t = s / (shift - (shift - 1) * s)``."""

    return t / (shift - (shift - 1) * t)


def _apply_shift(
    sigmas: np.ndarray, scheduler_config: Any, image_seq_len: int
) -> tuple[np.ndarray, float | None, float]:
    """Return shifted sigmas plus ``(mu, static_shift)`` for provenance."""

    use_dynamic = bool(_config_get(scheduler_config, "use_dynamic_shifting", False))
    static_shift = float(_config_get(scheduler_config, "shift", 1.0))
    time_shift_type = str(_config_get(scheduler_config, "time_shift_type", "exponential"))
    mu = calculate_flux_shift(
        image_seq_len,
        int(_config_get(scheduler_config, "base_image_seq_len", 256)),
        int(_config_get(scheduler_config, "max_image_seq_len", 4096)),
        float(_config_get(scheduler_config, "base_shift", 0.5)),
        float(_config_get(scheduler_config, "max_shift", 1.15)),
    )
    if use_dynamic:
        return flux_time_shift(mu, 1.0, sigmas, time_shift_type=time_shift_type), mu, static_shift
    # ``mu`` is still computed by the pipeline in this branch but ignored by the
    # scheduler, so it is reported as None to avoid implying it was applied.
    return flux_static_shift(static_shift, sigmas), None, static_shift


def flux_sigma_boundaries(
    num_steps: int,
    scheduler_config: Any,
    image_seq_len: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reproduce ``scheduler.sigmas`` after the pipeline's ``set_timesteps``.

    Returns ``num_steps + 1`` boundaries from 1.0 down to the appended terminal
    0.0.  The arithmetic follows diffusers byte for byte: ``np.linspace`` in
    float64, cast to float32, shifted in float32, then wrapped as a float32
    tensor with ``0.0`` appended.

    At 1024x1024 (``image_seq_len`` 4096) this gives
    ``[1.0, 0.75, 0.5, 0.25, 0.0]`` for FLUX.1-schnell at 4 steps -- schnell's
    scheduler has ``use_dynamic_shifting = false`` and ``shift = 1.0``, so no
    shift is applied at all -- and the 28-step exponentially shifted grid with
    ``mu = 1.15`` for FLUX.1-dev.
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float32)
    shifted, _mu, _static = _apply_shift(sigmas, scheduler_config, image_seq_len)
    boundaries = torch.from_numpy(np.asarray(shifted, dtype=np.float32))
    boundaries = torch.cat([boundaries, torch.zeros(1, dtype=torch.float32)])
    return boundaries.to(device=device, dtype=dtype)


def flux_schedule_metadata(
    num_steps: int, scheduler_config: Any, image_seq_len: int
) -> dict[str, Any]:
    """Every schedule value that belongs in ``generation_protocol.json``."""

    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float32)
    shifted, mu, static_shift = _apply_shift(sigmas, scheduler_config, image_seq_len)
    boundaries = [float(value) for value in shifted] + [0.0]
    return {
        "num_inference_steps": int(num_steps),
        "image_seq_len": int(image_seq_len),
        "unshifted_sigmas": [float(value) for value in sigmas],
        "sigmas": boundaries,
        "timesteps": [float(value) * FLUX_NUM_TRAIN_TIMESTEPS for value in shifted],
        "use_dynamic_shifting": bool(
            _config_get(scheduler_config, "use_dynamic_shifting", False)
        ),
        "time_shift_type": str(
            _config_get(scheduler_config, "time_shift_type", "exponential")
        ),
        "mu": None if mu is None else float(mu),
        "static_shift": float(static_shift),
        "base_shift": float(_config_get(scheduler_config, "base_shift", 0.5)),
        "max_shift": float(_config_get(scheduler_config, "max_shift", 1.15)),
        "base_image_seq_len": int(_config_get(scheduler_config, "base_image_seq_len", 256)),
        "max_image_seq_len": int(_config_get(scheduler_config, "max_image_seq_len", 4096)),
        "num_train_timesteps": int(
            _config_get(scheduler_config, "num_train_timesteps", FLUX_NUM_TRAIN_TIMESTEPS)
        ),
        "sigma_convention": "x_sigma = (1 - sigma) * x0 + sigma * epsilon; v = epsilon - x0",
    }


def flux_boundary_aligned_teacher_sigmas(
    boundaries: torch.Tensor,
    interval_index: int,
    micro_steps: int,
    scheduler_config: Any,
    image_seq_len: int,
) -> torch.Tensor:
    """FLUX analogue of ``flow_matching.boundary_aligned_teacher_sigmas``.

    The two endpoints are the *student's* macro boundaries and are restored
    exactly after the round trip.  Interpolation happens in the student's
    unshifted base time so the schedule geometry is preserved; with schnell's
    identity shift this reduces to equal spacing directly in sigma.
    """

    if boundaries.ndim != 1 or boundaries.numel() < 2:
        raise ValueError("boundaries must be a 1-D tensor with at least two entries")
    if not 0 <= interval_index < boundaries.numel() - 1:
        raise ValueError("interval_index is outside the student schedule")
    if micro_steps <= 0:
        raise ValueError("micro_steps must be positive")
    if not bool(torch.all(boundaries[:-1] > boundaries[1:])):
        raise ValueError("FlowMatch boundaries must be strictly decreasing")

    use_dynamic = bool(_config_get(scheduler_config, "use_dynamic_shifting", False))
    time_shift_type = str(_config_get(scheduler_config, "time_shift_type", "exponential"))
    static_shift = float(_config_get(scheduler_config, "shift", 1.0))
    mu = calculate_flux_shift(
        image_seq_len,
        int(_config_get(scheduler_config, "base_image_seq_len", 256)),
        int(_config_get(scheduler_config, "max_image_seq_len", 4096)),
        float(_config_get(scheduler_config, "base_shift", 0.5)),
        float(_config_get(scheduler_config, "max_shift", 1.15)),
    )

    start = float(boundaries[interval_index])
    end = float(boundaries[interval_index + 1])

    def to_base(values: np.ndarray) -> np.ndarray:
        if use_dynamic:
            return flux_time_unshift(mu, values, time_shift_type=time_shift_type)
        return flux_static_unshift(static_shift, values)

    def to_shifted(values: np.ndarray) -> np.ndarray:
        if use_dynamic:
            return flux_time_shift(mu, 1.0, values, time_shift_type=time_shift_type)
        return flux_static_shift(static_shift, values)

    # sigma = 0 is a fixed point of both shifts but 1/t is singular there, so
    # the terminal boundary is handled without a round trip.
    endpoints = np.array([start, end], dtype=np.float64)
    finite = endpoints > 0.0
    base = np.zeros_like(endpoints)
    base[finite] = to_base(endpoints[finite])
    grid_base = np.linspace(base[0], base[1], micro_steps + 1)
    grid = np.zeros_like(grid_base)
    positive = grid_base > 0.0
    grid[positive] = to_shifted(grid_base[positive])

    result = torch.from_numpy(grid.astype(np.float32)).to(
        device=boundaries.device, dtype=boundaries.dtype
    )
    result[0] = boundaries[interval_index]
    result[-1] = boundaries[interval_index + 1]
    return result


# --------------------------------------------------------------------------- #
# velocity / Euler
# --------------------------------------------------------------------------- #
def flux_timestep_input(sigma: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reproduce the pipeline's ``t.to(latents.dtype) / 1000`` rounding.

    The pipeline holds ``t = sigma * 1000`` in fp32, casts it to the *latent*
    dtype, and only then divides by 1000.  At bf16 that is a materially
    different value from ``bf16(sigma)``: for sigma = 0.9884086, ``t`` rounds to
    988.0 and the transformer sees 0.988, whereas ``bf16(sigma)`` would give
    0.98828125.  Getting this wrong silently perturbs every dev step.
    """

    timestep = (sigma.to(torch.float32) * FLUX_NUM_TRAIN_TIMESTEPS).to(dtype)
    return timestep / FLUX_NUM_TRAIN_TIMESTEPS


def _batch_sigma(value: torch.Tensor | float, batch_size: int, device, dtype) -> torch.Tensor:
    sigma = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.expand(batch_size)
    if sigma.numel() != batch_size:
        raise ValueError("sigma must be scalar or have one value per sample")
    if bool(torch.any((sigma < 0) | (sigma > 1))):
        raise ValueError("FLUX sigmas must be in [0, 1]")
    return sigma


def predict_flux_velocity(
    transformer: Any,
    packed_latents: torch.Tensor,
    sigma: torch.Tensor | float,
    conditioning: FluxConditioning,
    guidance_scale: float | None = None,
    *,
    latent_image_ids: torch.Tensor,
) -> torch.Tensor:
    """Predict the FLUX FlowMatch velocity for packed latents at ``sigma``.

    ``guidance_scale`` is the distilled-guidance embedding value for FLUX.1-dev
    (``guidance_embeds = true``); pass ``None`` for FLUX.1-schnell, whose
    transformer has no guidance embedding.  No classifier-free guidance branch
    exists for either model.
    """

    if packed_latents.ndim != 3:
        raise ValueError(f"expected packed latents (B, N, C), got {tuple(packed_latents.shape)}")
    batch_size = packed_latents.shape[0]
    if conditioning.batch_size != batch_size:
        raise ValueError("conditioning batch size does not match latents")

    sigma_batch = _batch_sigma(sigma, batch_size, packed_latents.device, packed_latents.dtype)
    timestep = flux_timestep_input(sigma_batch, packed_latents.dtype)

    guidance_embeds = bool(_config_get(transformer.config, "guidance_embeds", False))
    if guidance_scale is None:
        if guidance_embeds:
            raise ValueError(
                "this transformer has guidance_embeds=True and requires a guidance_scale"
            )
        guidance = None
    else:
        if not guidance_embeds:
            raise ValueError(
                "this transformer has guidance_embeds=False and must be called with "
                "guidance_scale=None"
            )
        guidance = torch.full(
            [1], float(guidance_scale), device=packed_latents.device, dtype=torch.float32
        ).expand(batch_size)

    output = transformer(
        hidden_states=packed_latents,
        timestep=timestep,
        guidance=guidance,
        pooled_projections=conditioning.pooled_prompt_embeds,
        encoder_hidden_states=conditioning.prompt_embeds,
        txt_ids=conditioning.text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]
    if output.shape != packed_latents.shape:
        raise ValueError(
            f"FLUX velocity shape {tuple(output.shape)} does not match latents "
            f"{tuple(packed_latents.shape)}"
        )
    return output


def flux_euler_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
    next_sigma: torch.Tensor | float,
) -> torch.Tensor:
    """One deterministic Euler step, matching ``scheduler.step`` numerics.

    The scheduler upcasts the sample to fp32, applies
    ``sample + (sigma_next - sigma) * model_output`` and casts the result back
    to the model-output dtype.
    """

    if velocity.shape != sample.shape:
        raise ValueError(
            f"velocity shape {tuple(velocity.shape)} does not match sample "
            f"{tuple(sample.shape)}"
        )
    current = float(torch.as_tensor(sigma, dtype=torch.float32).reshape(-1)[0])
    following = float(torch.as_tensor(next_sigma, dtype=torch.float32).reshape(-1)[0])
    if following > current:
        raise ValueError("generation must move from larger sigma to smaller sigma")
    result = sample.to(torch.float32) + (following - current) * velocity.to(torch.float32)
    return result.to(velocity.dtype)


def flux_interval_secant_velocity(
    start: torch.Tensor,
    endpoint: torch.Tensor,
    *,
    sigma: float,
    next_sigma: float,
) -> torch.Tensor:
    """The single velocity whose Euler step lands exactly on ``endpoint``."""

    if start.shape != endpoint.shape:
        raise ValueError("start and endpoint must have identical shapes")
    delta = float(next_sigma) - float(sigma)
    if delta == 0.0:
        raise ValueError("sigma and next_sigma must differ")
    velocity = (endpoint.float() - start.float()) / delta
    if not torch.isfinite(velocity).all():
        raise RuntimeError("secant velocity contains NaN or Inf")
    return velocity


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_flux_pipeline(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype,
    without_transformer: bool = False,
) -> Any:
    """Load a ``FluxPipeline`` from a snapshot directory.

    ``without_transformer`` skips the 12B transformer entirely, which is how the
    trainer and the generator obtain the shared components (CLIP-L, T5-XXL,
    tokenizers, VAE, scheduler) without paying for a denoiser they replace.
    """

    from diffusers import FluxPipeline

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if without_transformer:
        kwargs["transformer"] = None
    return FluxPipeline.from_pretrained(str(model_path), **kwargs)


def load_flux_transformer(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype,
    subfolder: str = "transformer",
) -> Any:
    """Load a ``FluxTransformer2DModel`` from a snapshot or checkpoint dir."""

    from diffusers import FluxTransformer2DModel

    path = Path(model_path)
    if (path / "config.json").is_file():
        return FluxTransformer2DModel.from_pretrained(str(path), torch_dtype=torch_dtype)
    return FluxTransformer2DModel.from_pretrained(
        str(path), subfolder=subfolder, torch_dtype=torch_dtype
    )


def load_flux_scheduler_config(model_path: str | Path) -> dict[str, Any]:
    """Read ``scheduler/scheduler_config.json`` without importing diffusers."""

    import json

    path = Path(model_path)
    candidate = path / "scheduler" / "scheduler_config.json"
    if not candidate.is_file():
        candidate = path / "scheduler_config.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"no scheduler_config.json under {model_path}")
    return json.loads(candidate.read_text(encoding="utf-8"))


def decode_flux_latents(
    vae: Any,
    packed_latents: torch.Tensor,
    *,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Unpack, undo the VAE scaling/shift, and decode to image space."""

    latents = unpack_flux_latents(packed_latents, height, width, vae_scale_factor)
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    return vae.decode(latents.to(vae.dtype), return_dict=False)[0]


__all__ = [
    "FLUX_DEFAULT_GUIDANCE",
    "FLUX_DEFAULT_MAX_SEQUENCE_LENGTH",
    "FLUX_DEFAULT_STEPS",
    "FLUX_NUM_TRAIN_TIMESTEPS",
    "FluxConditioning",
    "calculate_flux_shift",
    "decode_flux_latents",
    "encode_flux_prompts",
    "flux_boundary_aligned_teacher_sigmas",
    "flux_euler_step",
    "flux_image_seq_len",
    "flux_interval_secant_velocity",
    "flux_latent_shape",
    "flux_latent_spatial",
    "flux_schedule_metadata",
    "flux_sigma_boundaries",
    "flux_static_shift",
    "flux_static_unshift",
    "flux_time_shift",
    "flux_time_unshift",
    "flux_timestep_input",
    "load_flux_pipeline",
    "load_flux_scheduler_config",
    "load_flux_transformer",
    "pack_flux_latents",
    "predict_flux_velocity",
    "prepare_flux_latent_image_ids",
    "unpack_flux_latents",
]
