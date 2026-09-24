"""Frequency-shaped latent noise utilities.

These helpers do not sample randomness.  They transform an already sampled
Gaussian tensor, which keeps the caller's RNG stream unchanged.
"""

from __future__ import annotations

import math
from fractions import Fraction

import torch


def normalized_radial_frequency_mask(
    height: int,
    width: int,
    *,
    cutoff: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the 2-D mask rho <= cutoff with Nyquist-corner rho == 1."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not math.isfinite(cutoff) or not 0.0 < cutoff <= 1.0:
        raise ValueError("cutoff must be finite and in (0, 1]")
    fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=device, dtype=torch.float32)
    rho = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    rho = rho / math.sqrt(0.5**2 + 0.5**2)
    return rho <= cutoff


def normalized_radial_frequency_band_mask(
    height: int,
    width: int,
    *,
    band: str,
    low_cutoff: float,
    mid_cutoff: float,
    device: torch.device,
) -> torch.Tensor:
    """Return one disjoint low/mid/high mask on the normalized FFT grid."""
    if not math.isfinite(low_cutoff) or not math.isfinite(mid_cutoff) or not (
        0.0 < low_cutoff < mid_cutoff < 1.0
    ):
        raise ValueError("frequency cutoffs must satisfy 0 < low < mid < 1")
    if band not in ("low", "mid", "high"):
        raise ValueError(f"unsupported frequency band: {band!r}")
    low = normalized_radial_frequency_mask(
        height, width, cutoff=low_cutoff, device=device
    )
    through_mid = normalized_radial_frequency_mask(
        height, width, cutoff=mid_cutoff, device=device
    )
    return {
        "low": low,
        "mid": through_mid & ~low,
        "high": ~through_mid,
    }[band]


@torch.no_grad()
def symmetric_frequency_tangent_pair(
    source: torch.Tensor,
    probe: torch.Tensor,
    *,
    band: str,
    angle_degrees: float,
    low_cutoff: float = 0.20,
    mid_cutoff: float = 0.50,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Construct an equal-norm symmetric perturbation in one FFT band.

    ``probe`` is band-pass filtered and projected onto the tangent space of
    the constant-norm sphere at ``source``.  The returned pair is

    ``cos(theta) * source +/- sin(theta) * ||source|| * unit_tangent``.

    Randomness is owned by the caller so auxiliary sampling can use a private
    generator without changing the main training RNG stream.
    """
    if source.ndim != 4 or source.shape != probe.shape or source.shape[0] <= 0:
        raise ValueError("source and probe must share non-empty shape [B,C,H,W]")
    if not torch.isfinite(source).all() or not torch.isfinite(probe).all():
        raise ValueError("source and probe must contain only finite values")
    if not math.isfinite(angle_degrees) or not 0.0 < angle_degrees < 90.0:
        raise ValueError("angle_degrees must be finite and in (0, 90)")

    original_dtype = source.dtype
    source_fp32 = source.float()
    probe_fp32 = probe.float()
    mask = normalized_radial_frequency_band_mask(
        source.shape[-2],
        source.shape[-1],
        band=band,
        low_cutoff=low_cutoff,
        mid_cutoff=mid_cutoff,
        device=source.device,
    )
    mask = mask[None, None]
    source_band = torch.fft.ifft2(
        torch.fft.fft2(source_fp32, dim=(-2, -1)) * mask,
        dim=(-2, -1),
    ).real
    direction = torch.fft.ifft2(
        torch.fft.fft2(probe_fp32, dim=(-2, -1)) * mask,
        dim=(-2, -1),
    ).real
    reduce_dims = (1, 2, 3)
    projection = (
        (direction * source_band).sum(dim=reduce_dims, keepdim=True)
        / source_band.square().sum(dim=reduce_dims, keepdim=True).clamp_min(eps)
    )
    direction = direction - projection * source_band
    direction_norm = torch.linalg.vector_norm(direction.flatten(1), dim=1)
    source_norm = torch.linalg.vector_norm(source_fp32.flatten(1), dim=1)
    if torch.any(direction_norm <= eps) or torch.any(source_norm <= eps):
        raise RuntimeError("degenerate source or frequency tangent direction")
    unit = direction / direction_norm[:, None, None, None]
    tangent_dot = (unit * source_fp32).sum(dim=reduce_dims).abs()
    tangent_relative_dot = tangent_dot / source_norm.clamp_min(eps)
    if torch.any(tangent_relative_dot > 2e-5):
        raise RuntimeError("frequency direction is not tangent to the source sphere")

    angle = math.radians(angle_degrees)
    radial = source_norm[:, None, None, None] * unit
    plus = math.cos(angle) * source_fp32 + math.sin(angle) * radial
    minus = math.cos(angle) * source_fp32 - math.sin(angle) * radial
    plus_norm = torch.linalg.vector_norm(plus.flatten(1), dim=1)
    minus_norm = torch.linalg.vector_norm(minus.flatten(1), dim=1)
    plus_relative_error = (plus_norm / source_norm - 1.0).abs()
    minus_relative_error = (minus_norm / source_norm - 1.0).abs()
    if torch.any(plus_relative_error > 2e-5) or torch.any(minus_relative_error > 2e-5):
        raise RuntimeError("frequency tangent pair does not preserve source norm")

    pair_difference_rms = (plus - minus).square().mean(dim=reduce_dims).sqrt()
    outputs = (plus.to(original_dtype), minus.to(original_dtype))
    stats = {
        "angle_degrees": source.new_tensor(angle_degrees, dtype=torch.float32),
        "source_norm": source_norm.mean().detach(),
        "direction_norm": direction_norm.mean().detach(),
        "tangent_relative_dot": tangent_relative_dot.mean().detach(),
        "plus_norm_relative_error": plus_relative_error.mean().detach(),
        "minus_norm_relative_error": minus_relative_error.mean().detach(),
        "pair_difference_rms": pair_difference_rms.mean().detach(),
    }
    return outputs[0], outputs[1], stats


@torch.no_grad()
def bandpass_match_sample_rms(
    noise: torch.Tensor,
    *,
    band: str,
    low_cutoff: float = 0.20,
    mid_cutoff: float = 0.50,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project every sample onto one frequency band and restore its RMS."""
    if noise.ndim != 4 or noise.shape[0] <= 0:
        raise ValueError(
            f"noise must have non-empty shape [B,C,H,W], got {tuple(noise.shape)}"
        )
    if not torch.isfinite(noise).all():
        raise ValueError("noise contains NaN or Inf")
    original_dtype = noise.dtype
    work = noise.float()
    mask = normalized_radial_frequency_band_mask(
        work.shape[-2],
        work.shape[-1],
        band=band,
        low_cutoff=low_cutoff,
        mid_cutoff=mid_cutoff,
        device=work.device,
    )
    if not bool(mask.any()):
        raise RuntimeError(f"{band} frequency mask is empty")
    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    filtered = torch.fft.ifft2(
        spectrum * mask[None, None], dim=(-2, -1)
    ).real
    reduce_dims = (1, 2, 3)
    raw_rms = work.square().mean(dim=reduce_dims, keepdim=True).sqrt()
    filtered_rms = filtered.square().mean(dim=reduce_dims, keepdim=True).sqrt()
    if torch.any(filtered_rms <= eps):
        raise RuntimeError(f"{band} frequency filter produced a degenerate sample")
    output = (filtered * (raw_rms / filtered_rms.clamp_min(eps))).to(original_dtype)
    if not torch.isfinite(output).all():
        raise RuntimeError(f"{band} frequency output contains NaN or Inf")
    output_rms = output.float().square().mean(dim=reduce_dims).sqrt()
    raw_rms_flat = raw_rms.flatten()
    return output, {
        "raw_rms": raw_rms_flat.mean().detach(),
        "output_rms": output_rms.mean().detach(),
        "rms_ratio": (output_rms / raw_rms_flat.clamp_min(eps)).mean().detach(),
        "retained_frequency_fraction": mask.float().mean().detach(),
        "lowpass_fraction": output.new_tensor(
            float(band == "low"), dtype=torch.float32
        ),
        "midpass_fraction": output.new_tensor(
            float(band == "mid"), dtype=torch.float32
        ),
        "highpass_fraction": output.new_tensor(
            float(band == "high"), dtype=torch.float32
        ),
    }


def _mixture_assignment_mask(
    batch_size: int,
    *,
    fraction: float,
    device: torch.device,
    assignment_offset: int | None,
) -> torch.Tensor:
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("frequency fraction must be finite and in (0, 1)")
    if assignment_offset is None:
        requested = batch_size * fraction
        selected_count = int(round(requested))
        if not math.isclose(requested, selected_count, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "batch size times frequency fraction must be integral for an exact mixture"
            )
        return torch.arange(batch_size, device=device) < selected_count
    if assignment_offset < 0:
        raise ValueError("assignment_offset must be non-negative")
    rational = Fraction(fraction).limit_denominator(1_000_000)
    if not math.isclose(float(rational), fraction, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("frequency fraction cannot be represented by stream assignment")
    indices = torch.arange(
        assignment_offset,
        assignment_offset + batch_size,
        device=device,
        dtype=torch.int64,
    )
    previous = torch.div(
        indices * rational.numerator, rational.denominator, rounding_mode="floor"
    )
    following = torch.div(
        (indices + 1) * rational.numerator,
        rational.denominator,
        rounding_mode="floor",
    )
    return following > previous


@torch.no_grad()
def mix_frequency_band_and_gaussian(
    noise: torch.Tensor,
    *,
    band: str,
    low_cutoff: float = 0.20,
    mid_cutoff: float = 0.50,
    band_fraction: float = 0.50,
    assignment_offset: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Replace an exact sample fraction by RMS-matched band-limited noise."""
    if noise.ndim != 4 or noise.shape[0] <= 0:
        raise ValueError(
            f"noise must have non-empty shape [B,C,H,W], got {tuple(noise.shape)}"
        )
    selected = _mixture_assignment_mask(
        noise.shape[0],
        fraction=band_fraction,
        device=noise.device,
        assignment_offset=assignment_offset,
    )
    selected_count = int(selected.sum().item())
    if selected_count <= 0 or selected_count >= noise.shape[0]:
        raise ValueError("the mixed batch must contain band-limited and Gaussian samples")
    output = noise.clone()
    filtered, band_stats = bandpass_match_sample_rms(
        noise[selected],
        band=band,
        low_cutoff=low_cutoff,
        mid_cutoff=mid_cutoff,
    )
    output[selected] = filtered
    reduce_dims = (1, 2, 3)
    raw_rms = noise.float().square().mean(dim=reduce_dims).sqrt()
    output_rms = output.float().square().mean(dim=reduce_dims).sqrt()
    realized = selected_count / noise.shape[0]
    return output, {
        "raw_rms": raw_rms.mean().detach(),
        "output_rms": output_rms.mean().detach(),
        "rms_ratio": (output_rms / raw_rms.clamp_min(1e-12)).mean().detach(),
        "retained_frequency_fraction": output.new_tensor(
            realized * float(band_stats["retained_frequency_fraction"])
            + (1.0 - realized),
            dtype=torch.float32,
        ),
        "lowpass_fraction": output.new_tensor(
            realized if band == "low" else 0.0, dtype=torch.float32
        ),
        "midpass_fraction": output.new_tensor(
            realized if band == "mid" else 0.0, dtype=torch.float32
        ),
        "highpass_fraction": output.new_tensor(
            realized if band == "high" else 0.0, dtype=torch.float32
        ),
    }


@torch.no_grad()
def lowpass_match_sample_rms(
    noise: torch.Tensor,
    *,
    cutoff: float = 0.20,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Low-pass each latent channel and restore each sample's original RMS.

    Args:
        noise: Gaussian latent tensor ``[B, C, H, W]``.
        cutoff: Normalized radial-frequency cutoff.  The frequency-grid corner
            has radius one, matching the frequency diagnostic protocol.

    Returns:
        A tensor with the input dtype/device and detached scalar diagnostics.
    """
    if noise.ndim != 4:
        raise ValueError(f"noise must have shape [B,C,H,W], got {tuple(noise.shape)}")
    if noise.shape[0] == 0 or noise.shape[-2] <= 0 or noise.shape[-1] <= 0:
        raise ValueError("noise dimensions must be non-empty")
    if not torch.isfinite(noise).all():
        raise ValueError("noise contains NaN or Inf")

    original_dtype = noise.dtype
    work = noise.float()
    mask = normalized_radial_frequency_mask(
        work.shape[-2], work.shape[-1], cutoff=cutoff, device=work.device
    )
    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    filtered = torch.fft.ifft2(spectrum * mask[None, None], dim=(-2, -1)).real

    reduce_dims = (1, 2, 3)
    raw_rms = work.square().mean(dim=reduce_dims, keepdim=True).sqrt()
    filtered_rms = filtered.square().mean(dim=reduce_dims, keepdim=True).sqrt()
    if torch.any(filtered_rms <= eps):
        raise RuntimeError("low-pass filter produced a degenerate sample")
    filtered = filtered * (raw_rms / filtered_rms.clamp_min(eps))
    output = filtered.to(original_dtype)
    if not torch.isfinite(output).all():
        raise RuntimeError("low-pass output contains NaN or Inf")

    output_rms = output.float().square().mean(dim=reduce_dims).sqrt()
    raw_rms_flat = raw_rms.flatten()
    rms_ratio = output_rms / raw_rms_flat.clamp_min(eps)
    stats = {
        "raw_rms": raw_rms_flat.mean().detach(),
        "output_rms": output_rms.mean().detach(),
        "rms_ratio": rms_ratio.mean().detach(),
        "retained_frequency_fraction": mask.float().mean().detach(),
    }
    return output, stats


@torch.no_grad()
def pink_noise_match_sample_rms(
    noise: torch.Tensor,
    *,
    beta: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Color Gaussian latent noise with ``1 / rho**beta`` power.

    The transform is applied independently to every ``[H,W]`` latent channel.
    DC is not amplified: each channel's spatial mean is preserved exactly in
    FP32.  The non-DC component is then rescaled per sample so the final tensor
    has the same RMS as the source.  Consequently proposal comparisons change
    spectral allocation without introducing a mean or total-energy confound.

    ``beta=0`` is the identity and is useful as a numerical contract check.
    Randomness remains owned by the caller.
    """
    if noise.ndim != 4 or noise.shape[0] <= 0:
        raise ValueError(
            f"noise must have non-empty shape [B,C,H,W], got {tuple(noise.shape)}"
        )
    if noise.shape[-2] <= 1 or noise.shape[-1] <= 1:
        raise ValueError("pink-noise spatial dimensions must both exceed one")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if not torch.isfinite(noise).all():
        raise ValueError("noise contains NaN or Inf")

    original_dtype = noise.dtype
    work = noise.float()
    height, width = work.shape[-2:]
    fy = torch.fft.fftfreq(height, device=work.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=work.device, dtype=torch.float32)
    rho = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    nonzero = rho > 0
    min_nonzero = rho[nonzero].min()
    weights = rho.clamp_min(min_nonzero).pow(-0.5 * beta)
    weights[0, 0] = 1.0

    channel_mean = work.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    colored = torch.fft.ifft2(
        spectrum * weights[None, None], dim=(-2, -1)
    ).real
    colored_centered = colored - colored.mean(dim=(-2, -1), keepdim=True)

    reduce_dims = (1, 2, 3)
    raw_rms_sq = work.square().mean(dim=reduce_dims, keepdim=True)
    mean_rms_sq = channel_mean.square().mean(dim=1, keepdim=True)
    target_centered_rms_sq = (raw_rms_sq - mean_rms_sq).clamp_min(0.0)
    colored_centered_rms_sq = colored_centered.square().mean(
        dim=reduce_dims, keepdim=True
    )
    if torch.any(colored_centered_rms_sq <= eps):
        raise RuntimeError("pink-noise transform produced a degenerate sample")
    colored_centered = colored_centered * torch.sqrt(
        target_centered_rms_sq / colored_centered_rms_sq.clamp_min(eps)
    )
    output_fp32 = colored_centered + channel_mean
    output = output_fp32.to(original_dtype)
    if not torch.isfinite(output).all():
        raise RuntimeError("pink-noise output contains NaN or Inf")

    output_work = output.float()
    raw_rms = raw_rms_sq.flatten().sqrt()
    output_rms = output_work.square().mean(dim=reduce_dims).sqrt()
    channel_mean_error = (
        output_work.mean(dim=(-2, -1), keepdim=True) - channel_mean
    ).abs()
    output_power = torch.fft.fft2(output_work, dim=(-2, -1)).abs().square()
    normalized_rho = rho / math.sqrt(0.5**2 + 0.5**2)
    low = normalized_rho <= 0.20
    mid = (normalized_rho > 0.20) & (normalized_rho <= 0.50)
    high = normalized_rho > 0.50
    total_power = output_power.sum(dim=(-2, -1)).clamp_min(eps)

    def power_fraction(mask: torch.Tensor) -> torch.Tensor:
        return (output_power[..., mask].sum(dim=-1) / total_power).mean().detach()

    return output, {
        "beta": output.new_tensor(beta, dtype=torch.float32),
        "raw_rms": raw_rms.mean().detach(),
        "output_rms": output_rms.mean().detach(),
        "rms_ratio": (output_rms / raw_rms.clamp_min(eps)).mean().detach(),
        "channel_mean_max_abs_error": channel_mean_error.max().detach(),
        "low_power_fraction": power_fraction(low),
        "mid_power_fraction": power_fraction(mid),
        "high_power_fraction": power_fraction(high),
    }


@torch.no_grad()
def mix_lowpass_and_gaussian(
    noise: torch.Tensor,
    *,
    cutoff: float = 0.20,
    lowpass_fraction: float = 0.50,
    assignment_offset: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Low-pass an exact fraction of a batch and leave the rest Gaussian.

    Training batches are shuffled upstream, so using a fixed within-batch
    partition does not bias prompts and consumes no extra RNG. In the formal
    PixArt run, each local batch of 16 contains exactly eight low-pass and
    eight full-spectrum Gaussian initial states.
    """
    if noise.ndim != 4 or noise.shape[0] <= 0:
        raise ValueError(f"noise must have non-empty shape [B,C,H,W], got {tuple(noise.shape)}")
    if not math.isfinite(lowpass_fraction) or not 0.0 < lowpass_fraction < 1.0:
        raise ValueError("lowpass_fraction must be finite and in (0, 1)")
    if assignment_offset is None:
        requested = noise.shape[0] * lowpass_fraction
        lowpass_count = int(round(requested))
        if not math.isclose(requested, lowpass_count, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "batch size times lowpass_fraction must be integral for an exact mixture"
            )
        lowpass_mask = torch.arange(noise.shape[0], device=noise.device) < lowpass_count
    else:
        if assignment_offset < 0:
            raise ValueError("assignment_offset must be non-negative")
        # Select samples from one contiguous global sample stream.  The
        # cumulative-floor construction makes every prefix differ from the
        # requested fraction by less than one sample, without rounding each
        # microbatch independently.  Decimal sweep values such as 0.1 and 0.3
        # are first recovered as small exact rational numbers.
        fraction = Fraction(lowpass_fraction).limit_denominator(1_000_000)
        if not math.isclose(float(fraction), lowpass_fraction, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("lowpass_fraction cannot be represented by the stream assignment")
        indices = torch.arange(
            assignment_offset,
            assignment_offset + noise.shape[0],
            device=noise.device,
            dtype=torch.int64,
        )
        previous = torch.div(indices * fraction.numerator, fraction.denominator, rounding_mode="floor")
        following = torch.div(
            (indices + 1) * fraction.numerator,
            fraction.denominator,
            rounding_mode="floor",
        )
        lowpass_mask = following > previous
        lowpass_count = int(lowpass_mask.sum().item())
    if lowpass_count <= 0 or lowpass_count >= noise.shape[0]:
        raise ValueError("the mixed batch must contain both low-pass and Gaussian samples")

    output = noise.clone()
    filtered, lowpass_stats = lowpass_match_sample_rms(
        noise[lowpass_mask], cutoff=cutoff
    )
    output[lowpass_mask] = filtered
    reduce_dims = (1, 2, 3)
    raw_rms = noise.float().square().mean(dim=reduce_dims).sqrt()
    output_rms = output.float().square().mean(dim=reduce_dims).sqrt()
    stats = {
        "raw_rms": raw_rms.mean().detach(),
        "output_rms": output_rms.mean().detach(),
        "rms_ratio": (output_rms / raw_rms.clamp_min(1e-12)).mean().detach(),
        "retained_frequency_fraction": (
            lowpass_fraction * lowpass_stats["retained_frequency_fraction"]
            + (1.0 - lowpass_fraction)
        ).detach(),
        "lowpass_fraction": output.new_tensor(
            lowpass_count / noise.shape[0], dtype=torch.float32
        ),
    }
    return output, stats


@torch.no_grad()
def split_low_mid_high_match_sample_rms(
    noise: torch.Tensor,
    *,
    low_cutoff: float = 0.20,
    mid_cutoff: float = 0.50,
    assignment_offset: int = 0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Assign samples cyclically to low, mid and high frequency bands.

    Sample ``i`` uses band ``(assignment_offset + i) % 3``.  The caller can
    include the global update and distributed rank in ``assignment_offset`` so
    a global batch that is not divisible by three remains exactly balanced over
    three consecutive updates.  Filtering consumes no RNG.  Each filtered
    sample is independently rescaled to its original Gaussian RMS.
    """
    if noise.ndim != 4 or noise.shape[0] < 3:
        raise ValueError(
            f"three-band noise must have shape [B,C,H,W] with B >= 3, got {tuple(noise.shape)}"
        )
    if not torch.isfinite(noise).all():
        raise ValueError("noise contains NaN or Inf")
    if not math.isfinite(low_cutoff) or not math.isfinite(mid_cutoff) or not (
        0.0 < low_cutoff < mid_cutoff < 1.0
    ):
        raise ValueError("three-band cutoffs must satisfy 0 < low < mid < 1")
    if not isinstance(assignment_offset, int):
        raise TypeError("assignment_offset must be an integer")

    original_dtype = noise.dtype
    work = noise.float()
    height, width = work.shape[-2:]
    low_mask = normalized_radial_frequency_mask(
        height, width, cutoff=low_cutoff, device=work.device
    )
    through_mid_mask = normalized_radial_frequency_mask(
        height, width, cutoff=mid_cutoff, device=work.device
    )
    band_masks = torch.stack(
        (low_mask, through_mid_mask & ~low_mask, ~through_mid_mask), dim=0
    )
    if not torch.all(band_masks.flatten(1).any(dim=1)):
        raise RuntimeError("one of the requested frequency bands is empty")

    band_ids = (
        torch.arange(work.shape[0], device=work.device, dtype=torch.long)
        + assignment_offset
    ) % 3
    sample_masks = band_masks[band_ids]
    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    filtered = torch.fft.ifft2(
        spectrum * sample_masks[:, None], dim=(-2, -1)
    ).real

    reduce_dims = (1, 2, 3)
    raw_rms = work.square().mean(dim=reduce_dims, keepdim=True).sqrt()
    filtered_rms = filtered.square().mean(dim=reduce_dims, keepdim=True).sqrt()
    if torch.any(filtered_rms <= eps):
        raise RuntimeError("three-band filter produced a degenerate sample")
    filtered = filtered * (raw_rms / filtered_rms.clamp_min(eps))
    output = filtered.to(original_dtype)
    if not torch.isfinite(output).all():
        raise RuntimeError("three-band output contains NaN or Inf")

    output_rms = output.float().square().mean(dim=reduce_dims).sqrt()
    raw_rms_flat = raw_rms.flatten()
    counts = torch.stack([(band_ids == band).sum() for band in range(3)]).float()
    fractions = counts / work.shape[0]
    return output, {
        "raw_rms": raw_rms_flat.mean().detach(),
        "output_rms": output_rms.mean().detach(),
        "rms_ratio": (output_rms / raw_rms_flat.clamp_min(eps)).mean().detach(),
        "retained_frequency_fraction": sample_masks.float().mean().detach(),
        "lowpass_fraction": fractions[0].detach(),
        "midpass_fraction": fractions[1].detach(),
        "highpass_fraction": fractions[2].detach(),
        "assignment_offset_mod3": output.new_tensor(
            assignment_offset % 3, dtype=torch.float32
        ),
    }


@torch.no_grad()
def pair_gaussian_and_lowpass(
    noise: torch.Tensor,
    *,
    cutoff: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep even samples Gaussian and low-pass their independent odd partners.

    The caller repeat-interleaves every prompt, so indices ``(2i, 2i+1)``
    share text conditioning but start from independently sampled Gaussian
    tensors. Only the odd member is frequency filtered and RMS restored.
    """
    if noise.ndim != 4 or noise.shape[0] <= 0 or noise.shape[0] % 2:
        raise ValueError("paired noise must have shape [2N,C,H,W] with N > 0")
    output = noise.clone()
    filtered, lowpass_stats = lowpass_match_sample_rms(noise[1::2], cutoff=cutoff)
    output[1::2] = filtered
    reduce_dims = (1, 2, 3)
    raw_rms = noise.float().square().mean(dim=reduce_dims).sqrt()
    output_rms = output.float().square().mean(dim=reduce_dims).sqrt()
    return output, {
        "raw_rms": raw_rms.mean().detach(),
        "output_rms": output_rms.mean().detach(),
        "rms_ratio": (output_rms / raw_rms.clamp_min(1e-12)).mean().detach(),
        "retained_frequency_fraction": (
            0.5 * lowpass_stats["retained_frequency_fraction"] + 0.5
        ).detach(),
        "lowpass_fraction": output.new_tensor(0.5, dtype=torch.float32),
    }


@torch.no_grad()
def gaussian_lowpass_pair_from_source(
    source_noise: torch.Tensor,
    *,
    cutoff: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return one Gaussian tensor and its RMS-matched low-pass projection.

    Unlike :func:`pair_gaussian_and_lowpass`, both outputs originate from the
    exact same Gaussian draw.  This isolates the effect of removing frequencies
    above ``cutoff`` and is the pairing contract used by the frequency-response
    regularizer.  No randomness is sampled inside this function.
    """
    if source_noise.ndim != 4 or source_noise.shape[0] <= 0:
        raise ValueError(
            "source noise must have non-empty shape [B,C,H,W], "
            f"got {tuple(source_noise.shape)}"
        )
    if not torch.isfinite(source_noise).all():
        raise ValueError("source noise contains NaN or Inf")

    gaussian = source_noise.detach().clone()
    lowpass, lowpass_stats = lowpass_match_sample_rms(
        gaussian,
        cutoff=cutoff,
    )
    gaussian_rms = gaussian.float().square().mean(dim=(1, 2, 3)).sqrt()
    lowpass_rms = lowpass.float().square().mean(dim=(1, 2, 3)).sqrt()
    difference_rms = (
        lowpass.float() - gaussian.float()
    ).square().mean(dim=(1, 2, 3)).sqrt()
    stats = dict(lowpass_stats)
    stats.update(
        {
            "gaussian_rms": gaussian_rms.mean().detach(),
            "lowpass_rms": lowpass_rms.mean().detach(),
            "pair_rms_ratio": (
                lowpass_rms / gaussian_rms.clamp_min(1e-12)
            ).mean().detach(),
            "pair_difference_rms": difference_rms.mean().detach(),
        }
    )
    return gaussian, lowpass, stats


def pair_gaussian_antipodal(
    noise: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Create adjacent ``(z, -z)`` Gaussian views from one draw per prompt.

    ``noise`` has shape ``[N,C,H,W]`` and contains one independent Gaussian
    draw for every distinct prompt.  The output has shape ``[2N,C,H,W]``;
    rows ``(2i, 2i+1)`` are exactly antipodal and are intended to receive the
    same repeat-interleaved text condition.  No RMS rescaling is needed:
    negation preserves every sample norm and the standard Gaussian marginal.
    """
    if noise.ndim != 4 or noise.shape[0] <= 0:
        raise ValueError("antipodal source noise must have shape [N,C,H,W] with N > 0")
    output = torch.stack((noise, -noise), dim=1).reshape(-1, *noise.shape[1:])
    rms = output.float().square().mean(dim=(1, 2, 3)).sqrt()
    return output, {
        "raw_rms": rms.mean().detach(),
        "output_rms": rms.mean().detach(),
        "rms_ratio": output.new_tensor(1.0, dtype=torch.float32),
        "retained_frequency_fraction": output.new_tensor(1.0, dtype=torch.float32),
        "lowpass_fraction": output.new_tensor(0.0, dtype=torch.float32),
    }


@torch.no_grad()
def pair_mixed_lowpass_gaussian_antipodal(
    noise: torch.Tensor,
    *,
    cutoff: float = 0.20,
    lowpass_fraction: float = 0.70,
    assignment_offset: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Frequency-shape source particles, then create exact antipodal pairs.

    The mixture is assigned over the unique-prompt source stream before every
    source is expanded to ``(z, -z)``. Both members of a prompt pair therefore
    have the same spectrum and remain exact antipodes, as required by the
    teacher response-difference regularizer.
    """
    mixed, stats = mix_lowpass_and_gaussian(
        noise,
        cutoff=cutoff,
        lowpass_fraction=lowpass_fraction,
        assignment_offset=assignment_offset,
    )
    paired, _ = pair_gaussian_antipodal(mixed)
    paired_rms = paired.float().square().mean(dim=(1, 2, 3)).sqrt()
    stats = dict(stats)
    stats["output_rms"] = paired_rms.mean().detach()
    stats["antipodal_max_abs_error"] = (
        paired[0::2].float() + paired[1::2].float()
    ).abs().max().detach()
    return paired, stats
