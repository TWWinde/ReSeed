#!/usr/bin/env python3
"""Train first-interval residual biases on a frozen distilled FLUX student.

Objective
---------
The student transformer can be FLUX.1-schnell or a transformer-only distilled
checkpoint such as SenseFlow-FLUX. Its scheduler and shared pipeline components
come from ``--student-model`` while an optional transformer override comes from
``--student-transformer-model``. The teacher is FLUX.1-dev run with its
distilled guidance embedding at 3.5.

For the student's first macro interval ``[sigma0, sigma1]`` the
teacher is integrated with 7 boundary-aligned microsteps (see
``models.flux.flux_boundary_aligned_teacher_sigmas``) starting from the same
noise.  The resulting endpoint defines a *secant* velocity

    v_target = (x_teacher_end - x_noise) / (sigma1 - sigma0)

and the loss is ``MSE(student_velocity_at_sigma0_with_bias, sg(v_target))``.
Only the zero-initialised residual biases are trainable; every FLUX weight stays
frozen, and the trainer asserts each step that no frozen parameter received a
gradient.

Memory
------
FLUX.1-dev (12B) + FLUX.1-schnell (12B) + T5-XXL (4.7B) in bf16 is roughly
57 GB of weights before any activation.  Adding a 1024x1024 backward pass
through the student on top of that does **not** fit comfortably on an 80 GB
H100, even with gradient checkpointing on.  The intended path is therefore the
two-phase one:

1. ``--build-cache-only --teacher-target-cache DIR`` loads the teacher *only*
   (no student, no student optimizer) and writes one packed bf16 endpoint latent
   per ``(rank, step)`` together with the noise seed and prompt index.
2. The training run is given the same ``--teacher-target-cache DIR`` and never
   constructs the teacher at all, so only the student, its 233k bias parameters
   and the text encoders are resident.

Both phases derive their noise from a CPU fp32 generator seeded by
``(seed, rank, step)`` and draw prompts from the same seeded ``PromptStream``,
so phase 2 reproduces phase 1's batches exactly; the cached prompt index and
prompt digest are re-checked on every read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from reseed.data.prompts import read_prompts
from reseed.models.flux import (
    encode_flux_prompts,
    flux_boundary_aligned_teacher_sigmas,
    flux_euler_step,
    flux_image_seq_len,
    flux_interval_secant_velocity,
    flux_latent_shape,
    flux_schedule_metadata,
    flux_sigma_boundaries,
    load_flux_pipeline,
    load_flux_scheduler_config,
    load_flux_transformer,
    pack_flux_latents,
    predict_flux_velocity,
    prepare_flux_latent_image_ids,
)
from reseed.models.flux_residual_bias import (
    FluxBlockResidualBias,
    default_flux_blocks,
    load_flux_residual_bias_payload,
)
from reseed.trainers._common import (
    PromptStream,
    _all_reduce_gradients,
    _distributed,
    _mean,
)


HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
DEFAULT_TEACHER = (
    f"{HUB}/models--black-forest-labs--FLUX.1-dev/snapshots/"
    "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
)
DEFAULT_STUDENT = (
    f"{HUB}/models--black-forest-labs--FLUX.1-schnell/snapshots/"
    "741f7c3ce8b383c54771c7003378a50191e9efe9"
)


# --------------------------------------------------------------------------- #
# deterministic batch construction shared by both phases
# --------------------------------------------------------------------------- #
def noise_seed_for(seed: int, rank: int, step: int) -> int:
    """A stable per-(seed, rank, step) CPU noise seed.

    CPU generators are reproducible across processes and machines, which is what
    lets the cache-build phase and the training phase draw byte-identical noise
    without shipping the tensors twice.
    """

    return (int(seed) * 1_000_003 + int(rank) * 1_000_033 + int(step)) % (2**31 - 1)


def draw_noise(
    seed: int, rank: int, step: int, latent_shape: tuple[int, int, int], batch_size: int
) -> torch.Tensor:
    """fp32 CPU noise of shape ``(batch, C, H, W)`` for one (rank, step)."""

    generator = torch.Generator(device="cpu").manual_seed(noise_seed_for(seed, rank, step))
    return torch.randn((batch_size, *latent_shape), generator=generator, dtype=torch.float32)


def prompt_digest(prompts: list[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class BatchStream:
    """Reproducible ``(prompts, prompt_indices, noise)`` sequence for one rank.

    The prompt order is byte-identical to
    ``PromptStream(prompts, seed + 1009 * rank)`` -- same ``random.Random`` seed,
    same shuffle points -- but this variant also returns the index it drew so the
    cache can record it and the trainer can verify it.
    """

    def __init__(
        self,
        prompts: list[str],
        *,
        seed: int,
        rank: int,
        batch_size: int,
        latent_shape: tuple[int, int, int],
    ) -> None:
        self.prompts = prompts
        self.seed = seed
        self.rank = rank
        self.batch_size = batch_size
        self.latent_shape = latent_shape
        self._stream = PromptStream(prompts, seed + 1009 * rank)

    def _next_index(self) -> int:
        stream = self._stream
        if stream.cursor == len(stream.order):
            stream.cursor = 0
            stream.rng.shuffle(stream.order)
        index = stream.order[stream.cursor]
        stream.cursor += 1
        return index

    def next(self, step: int) -> tuple[list[str], list[int], torch.Tensor]:
        indices = [self._next_index() for _ in range(self.batch_size)]
        texts = [self.prompts[index] for index in indices]
        noise = draw_noise(self.seed, self.rank, step, self.latent_shape, self.batch_size)
        return texts, indices, noise

    def advance(self, steps: int) -> None:
        """Advance only the prompt stream when resuming at a global step."""

        for _ in range(int(steps) * self.batch_size):
            self._next_index()


# --------------------------------------------------------------------------- #
# teacher-target cache
# --------------------------------------------------------------------------- #
def cache_step_path(cache: Path, rank: int, step: int) -> Path:
    return cache / f"rank_{rank:02d}" / f"step_{step:06d}.pt"


def write_cache_entry(
    cache: Path,
    rank: int,
    step: int,
    *,
    endpoint: torch.Tensor,
    noise_seed: int,
    prompt_indices: list[int],
    prompts: list[str],
) -> Path:
    path = cache_step_path(cache, rank, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "cache_type": "flux-first-interval-teacher-endpoint-v1",
            "rank": int(rank),
            "step": int(step),
            "noise_seed": int(noise_seed),
            "prompt_indices": [int(value) for value in prompt_indices],
            "prompt_sha256": prompt_digest(prompts),
            "endpoint": endpoint.detach().to(torch.bfloat16).cpu(),
        },
        temporary,
    )
    temporary.rename(path)
    return path


def read_cache_entry(
    cache: Path,
    rank: int,
    step: int,
    *,
    noise_seed: int,
    prompt_indices: list[int],
    prompts: list[str],
) -> torch.Tensor:
    path = cache_step_path(cache, rank, step)
    if not path.is_file():
        raise FileNotFoundError(f"missing teacher-target cache entry: {path}")
    entry = torch.load(path, map_location="cpu", weights_only=True)
    if entry.get("cache_type") != "flux-first-interval-teacher-endpoint-v1":
        raise ValueError(f"unexpected cache payload in {path}")
    if int(entry["noise_seed"]) != int(noise_seed):
        raise RuntimeError(
            f"cache/noise mismatch at rank {rank} step {step}: "
            f"cached {entry['noise_seed']} vs recomputed {noise_seed}"
        )
    if [int(v) for v in entry["prompt_indices"]] != [int(v) for v in prompt_indices]:
        raise RuntimeError(f"cache/prompt-index mismatch at rank {rank} step {step}")
    if entry["prompt_sha256"] != prompt_digest(prompts):
        raise RuntimeError(f"cache/prompt-text mismatch at rank {rank} step {step}")
    return entry["endpoint"]


# --------------------------------------------------------------------------- #
def _save(
    adapter: FluxBlockResidualBias,
    optimizer: torch.optim.Optimizer,
    output: Path,
    *,
    step: int,
    metadata: dict[str, Any],
    keep_last: int,
) -> Path:
    checkpoint = output / f"checkpoint-{step}"
    temporary = output / f".checkpoint-{step}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    torch.save(adapter.payload(step=step), temporary / "residual_biases.pt")
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, temporary / "trainer_state.pt")
    (temporary / "manifest.json").write_text(
        json.dumps({**metadata, "step": step, **adapter.statistics()}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.rename(checkpoint)
    checkpoints = sorted(
        (path for path in output.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    for old in checkpoints[:-keep_last]:
        shutil.rmtree(old)
    return checkpoint


def _load_shared_components(model_path: Path, dtype: torch.dtype, device: torch.device) -> Any:
    """Load CLIP-L, T5-XXL, tokenizers and the VAE without the 12B transformer."""

    pipe = load_flux_pipeline(model_path, torch_dtype=dtype, without_transformer=True)
    pipe.to(device)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.to("cpu")
    for name in ("text_encoder", "text_encoder_2"):
        encoder = getattr(pipe, name, None)
        if encoder is not None:
            encoder.eval().requires_grad_(False)
    return pipe


def _restore_training_checkpoint(
    adapter: FluxBlockResidualBias,
    optimizer: torch.optim.Optimizer,
    checkpoint: Path,
) -> int:
    """Restore adapter and AdamW state and return the completed global step."""

    checkpoint = checkpoint.resolve()
    payload = load_flux_residual_bias_payload(checkpoint)
    state_path = checkpoint / "trainer_state.pt"
    manifest_path = checkpoint / "manifest.json"
    if not state_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete resume checkpoint: {checkpoint}")
    trainer_state = torch.load(state_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    steps = {
        "payload": int(payload["step"]),
        "trainer_state": int(trainer_state["step"]),
        "manifest": int(manifest["step"]),
    }
    if len(set(steps.values())) != 1:
        raise RuntimeError(f"resume checkpoint step mismatch: {steps}")
    if tuple(int(v) for v in payload["double_blocks"]) != adapter.double_block_indices:
        raise RuntimeError("resume checkpoint double-block selection differs from adapter")
    if tuple(int(v) for v in payload["single_blocks"]) != adapter.single_block_indices:
        raise RuntimeError("resume checkpoint single-block selection differs from adapter")
    adapter.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(trainer_state["optimizer"])
    return next(iter(steps.values()))


def _geometry(args: argparse.Namespace) -> tuple[int, tuple[int, int, int]]:
    image_seq_len = flux_image_seq_len(
        args.resolution, args.resolution, vae_scale_factor=args.vae_scale_factor
    )
    latent_shape = flux_latent_shape(
        args.resolution,
        args.resolution,
        latent_channels=args.latent_channels,
        vae_scale_factor=args.vae_scale_factor,
    )
    return image_seq_len, latent_shape


# --------------------------------------------------------------------------- #
# phase 1: teacher-only cache build
# --------------------------------------------------------------------------- #
def build_cache(args: argparse.Namespace) -> int:
    rank, _local_rank, world_size, device = _distributed()
    is_main = rank == 0
    dtype = torch.bfloat16
    cache = args.teacher_target_cache.resolve()
    if is_main:
        cache.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    (cache / f"rank_{rank:02d}").mkdir(parents=True, exist_ok=True)

    image_seq_len, latent_shape = _geometry(args)
    student_scheduler = load_flux_scheduler_config(args.student_model)
    boundaries = flux_sigma_boundaries(
        args.student_steps, student_scheduler, image_seq_len, dtype=torch.float32
    )
    grid = flux_boundary_aligned_teacher_sigmas(
        boundaries, 0, args.teacher_micro_steps, student_scheduler, image_seq_len
    )
    grid_values = [float(value) for value in grid]

    pipe = _load_shared_components(args.teacher_model, dtype, device)
    teacher = load_flux_transformer(args.teacher_model, torch_dtype=dtype).to(device)
    teacher.eval().requires_grad_(False)
    latent_image_ids = prepare_flux_latent_image_ids(
        args.resolution,
        args.resolution,
        vae_scale_factor=args.vae_scale_factor,
        device=device,
        dtype=dtype,
    )

    prompts = read_prompts(args.prompt_file, limit=args.prompt_limit)
    batches = BatchStream(
        prompts,
        seed=args.seed,
        rank=rank,
        batch_size=args.batch_size,
        latent_shape=latent_shape,
    )

    if is_main:
        (cache / "cache_manifest.json").write_text(
            json.dumps(
                {
                    "cache_type": "flux-first-interval-teacher-endpoint-v1",
                    "teacher": str(Path(args.teacher_model).resolve()),
                    "student_schedule_model": str(Path(args.student_model).resolve()),
                    "teacher_guidance_scale": args.teacher_guidance_scale,
                    "teacher_micro_steps": args.teacher_micro_steps,
                    "teacher_max_sequence_length": args.teacher_max_sequence_length,
                    "student_boundaries": [float(value) for value in boundaries],
                    "teacher_micro_sigmas": grid_values,
                    "resolution": args.resolution,
                    "latent_shape": list(latent_shape),
                    "packed_tokens": image_seq_len,
                    "world_size": world_size,
                    "batch_size_per_rank": args.batch_size,
                    "max_steps": args.max_steps,
                    "seed": args.seed,
                    "prompt_file": str(Path(args.prompt_file).resolve()),
                    "prompt_limit": args.prompt_limit,
                    "noise_recipe": "CPU torch.Generator seeded by (seed, rank, step)",
                    "schedule": flux_schedule_metadata(
                        args.student_steps, student_scheduler, image_seq_len
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    started = time.monotonic()
    # ``no_grad`` rather than ``inference_mode``: inference tensors carry
    # restrictions that make them awkward to serialize into the cache.
    with torch.no_grad():
        for step in range(1, args.max_steps + 1):
            texts, indices, noise_cpu = batches.next(step)
            path = cache_step_path(cache, rank, step)
            if args.resume and path.is_file():
                continue
            conditioning = encode_flux_prompts(
                pipe,
                texts,
                max_sequence_length=args.teacher_max_sequence_length,
                device=device,
            )
            latents = pack_flux_latents(noise_cpu.to(device=device, dtype=dtype))
            for sigma, next_sigma in zip(grid_values[:-1], grid_values[1:]):
                velocity = predict_flux_velocity(
                    teacher,
                    latents,
                    sigma,
                    conditioning,
                    args.teacher_guidance_scale,
                    latent_image_ids=latent_image_ids,
                )
                latents = flux_euler_step(latents, velocity, sigma, next_sigma)
            if not torch.isfinite(latents).all():
                raise RuntimeError(f"non-finite teacher endpoint at rank {rank} step {step}")
            write_cache_entry(
                cache,
                rank,
                step,
                endpoint=latents,
                noise_seed=noise_seed_for(args.seed, rank, step),
                prompt_indices=indices,
                prompts=texts,
            )
            if step == 1 or step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "phase": "cache",
                            "rank": rank,
                            "step": step,
                            "wall_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


# --------------------------------------------------------------------------- #
# phase 2: bias training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> int:
    rank, _local_rank, world_size, device = _distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    dtype = torch.bfloat16
    output = args.output.resolve()
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    use_cache = args.teacher_target_cache is not None
    cache = args.teacher_target_cache.resolve() if use_cache else None

    image_seq_len, latent_shape = _geometry(args)
    student_scheduler = load_flux_scheduler_config(args.student_model)
    boundaries = flux_sigma_boundaries(
        args.student_steps, student_scheduler, image_seq_len, dtype=torch.float32
    )
    sigma0 = float(boundaries[0])
    sigma1 = float(boundaries[1])
    grid = flux_boundary_aligned_teacher_sigmas(
        boundaries, 0, args.teacher_micro_steps, student_scheduler, image_seq_len
    )
    grid_values = [float(value) for value in grid]

    pipe = _load_shared_components(args.student_model, dtype, device)
    student_transformer_model = args.student_transformer_model or args.student_model
    student = load_flux_transformer(student_transformer_model, torch_dtype=dtype).to(device)
    student.eval().requires_grad_(False)
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()

    teacher = None
    teacher_pipe = None
    if not use_cache:
        teacher_pipe = _load_shared_components(args.teacher_model, dtype, device)
        teacher = load_flux_transformer(args.teacher_model, torch_dtype=dtype).to(device)
        teacher.eval().requires_grad_(False)

    default_double, default_single = default_flux_blocks(student)
    double_blocks = default_double if args.double_blocks is None else list(args.double_blocks)
    single_blocks = default_single if args.single_blocks is None else list(args.single_blocks)
    adapter = FluxBlockResidualBias(student, double_blocks, single_blocks).to(device)
    trainable = list(adapter.parameters())
    if not trainable:
        raise RuntimeError("residual-bias adapter selected no parameters")
    for parameter in trainable:
        if parameter.dtype != torch.float32:
            raise RuntimeError("residual-bias parameters must stay in fp32")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume_from_checkpoint is not None:
        start_step = _restore_training_checkpoint(
            adapter, optimizer, args.resume_from_checkpoint
        )
        if start_step >= args.max_steps:
            raise ValueError(
                f"resume step {start_step} must be smaller than max_steps {args.max_steps}"
            )

    latent_image_ids = prepare_flux_latent_image_ids(
        args.resolution,
        args.resolution,
        vae_scale_factor=args.vae_scale_factor,
        device=device,
        dtype=dtype,
    )
    prompts = read_prompts(args.prompt_file, limit=args.prompt_limit)
    batches = BatchStream(
        prompts,
        seed=args.seed,
        rank=rank,
        batch_size=args.batch_size,
        latent_shape=latent_shape,
    )
    batches.advance(start_step)

    metadata = {
        "method": "flux-first-interval-block-residual-bias",
        "student": str(Path(student_transformer_model).resolve()),
        "student_components_and_schedule": str(Path(args.student_model).resolve()),
        "student_transformer": str(Path(student_transformer_model).resolve()),
        "teacher": str(Path(args.teacher_model).resolve()),
        "teacher_target_cache": None if cache is None else str(cache),
        "teacher_loaded": teacher is not None,
        "residual_definition": (
            "double-block (text, image) outputs and single-block output plus learned "
            "per-channel biases"
        ),
        "zero_initialized": True,
        "double_blocks": list(adapter.double_block_indices),
        "single_blocks": list(adapter.single_block_indices),
        "bias_width": adapter.width,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "student_boundaries": [float(value) for value in boundaries],
        "student_sigma0": sigma0,
        "student_sigma1": sigma1,
        "teacher_micro_steps": args.teacher_micro_steps,
        "teacher_micro_sigmas": grid_values,
        "teacher_guidance_scale": args.teacher_guidance_scale,
        "teacher_max_sequence_length": args.teacher_max_sequence_length,
        "student_max_sequence_length": args.student_max_sequence_length,
        "student_guidance": None,
        "adapter_scope": "first student interval only; disable for suffix intervals",
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "loss": "MSE(student velocity at sigma0 with bias, teacher interval secant velocity)",
        "resume_from_checkpoint": (
            None
            if args.resume_from_checkpoint is None
            else str(args.resume_from_checkpoint.resolve())
        ),
        "start_step": start_step,
        "schedule": flux_schedule_metadata(args.student_steps, student_scheduler, image_seq_len),
    }
    if is_main:
        (output / "training_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        metrics_handle = (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
        print(json.dumps(metadata, indent=2), flush=True)
    else:
        metrics_handle = None

    started = time.monotonic()
    try:
        for step in range(start_step + 1, args.max_steps + 1):
            texts, indices, noise_cpu = batches.next(step)
            noise = pack_flux_latents(noise_cpu.to(device=device, dtype=dtype))

            if use_cache:
                teacher_endpoint = read_cache_entry(
                    cache,
                    rank,
                    step,
                    noise_seed=noise_seed_for(args.seed, rank, step),
                    prompt_indices=indices,
                    prompts=texts,
                ).to(device=device, dtype=dtype)
            else:
                with torch.no_grad():
                    teacher_conditioning = encode_flux_prompts(
                        teacher_pipe,
                        texts,
                        max_sequence_length=args.teacher_max_sequence_length,
                        device=device,
                    )
                    latents = noise
                    for sigma, next_sigma in zip(grid_values[:-1], grid_values[1:]):
                        velocity = predict_flux_velocity(
                            teacher,
                            latents,
                            sigma,
                            teacher_conditioning,
                            args.teacher_guidance_scale,
                            latent_image_ids=latent_image_ids,
                        )
                        latents = flux_euler_step(latents, velocity, sigma, next_sigma)
                    teacher_endpoint = latents.detach()

            with torch.no_grad():
                target_velocity = flux_interval_secant_velocity(
                    noise, teacher_endpoint, sigma=sigma0, next_sigma=sigma1
                )

            with torch.no_grad():
                student_conditioning = encode_flux_prompts(
                    pipe,
                    texts,
                    max_sequence_length=args.student_max_sequence_length,
                    device=device,
                )

            optimizer.zero_grad(set_to_none=True)
            adapter.enable()
            predicted_velocity = predict_flux_velocity(
                student,
                noise,
                sigma0,
                student_conditioning,
                None,
                latent_image_ids=latent_image_ids,
            )
            loss = F.mse_loss(predicted_velocity.float(), target_velocity)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            if any(parameter.grad is not None for parameter in student.parameters()):
                raise RuntimeError("a frozen FLUX student parameter received a gradient")
            if teacher is not None and any(
                parameter.grad is not None for parameter in teacher.parameters()
            ):
                raise RuntimeError("a frozen FLUX teacher parameter received a gradient")
            _all_reduce_gradients(trainable, world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient at step {step}")
            optimizer.step()

            with torch.no_grad():
                pred = predicted_velocity.detach().float().flatten(1)
                target = target_velocity.flatten(1)
                cosine = F.cosine_similarity(pred, target, dim=1).mean()
                relative_error = (pred - target).norm(dim=1) / target.norm(dim=1).clamp_min(1e-8)
                endpoint = (
                    noise.float() + (sigma1 - sigma0) * predicted_velocity.detach().float()
                )
                endpoint_error = (
                    endpoint - teacher_endpoint.float()
                ).flatten(1).norm(dim=1) / teacher_endpoint.float().flatten(1).norm(
                    dim=1
                ).clamp_min(1e-8)
                bias_values = torch.cat(
                    [parameter.detach().float().flatten() for parameter in trainable]
                )
            row = {
                "step": step,
                "loss": _mean(loss, world_size),
                "velocity_cosine": _mean(cosine, world_size),
                "velocity_relative_error": _mean(relative_error, world_size),
                "endpoint_relative_error": _mean(endpoint_error, world_size),
                "gradient_norm": _mean(torch.as_tensor(grad_norm, device=device), world_size),
                "bias_rms": _mean(bias_values.square().mean().sqrt(), world_size),
                "bias_abs_max": _mean(bias_values.abs().max(), world_size),
                "wall_seconds": time.monotonic() - started,
            }
            if is_main and (step == 1 or step % args.log_every == 0):
                metrics_handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.max_steps:
                if world_size > 1:
                    dist.barrier()
                if is_main:
                    checkpoint = _save(
                        adapter,
                        optimizer,
                        output,
                        step=step,
                        metadata=metadata,
                        keep_last=args.keep_last,
                    )
                    print(f"saved {checkpoint}", flush=True)
                    if args.checkpoint_hook is not None:
                        completed = subprocess.run(
                            [
                                sys.executable,
                                str(args.checkpoint_hook),
                                "--checkpoint",
                                str(checkpoint),
                                "--step",
                                str(step),
                            ],
                            check=False,
                        )
                        if completed.returncode != 0:
                            print(
                                f"checkpoint hook failed for step {step} with exit code "
                                f"{completed.returncode}; training continues",
                                flush=True,
                            )
                if world_size > 1:
                    dist.barrier()
    finally:
        adapter.close()
        if metrics_handle is not None:
            metrics_handle.close()
        if world_size > 1:
            dist.destroy_process_group()
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", type=Path, default=Path(DEFAULT_TEACHER))
    parser.add_argument("--student-model", type=Path, default=Path(DEFAULT_STUDENT))
    parser.add_argument(
        "--student-transformer-model",
        type=Path,
        default=None,
        help=(
            "optional transformer-only checkpoint; scheduler, tokenizers, text encoders, "
            "and VAE still come from --student-model"
        ),
    )
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--vae-scale-factor", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--student-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--double-blocks", type=int, nargs="+", default=None)
    parser.add_argument("--single-blocks", type=int, nargs="+", default=None)
    parser.add_argument("--teacher-micro-steps", type=int, default=7)
    parser.add_argument("--teacher-guidance-scale", type=float, default=3.5)
    parser.add_argument("--teacher-max-sequence-length", type=int, default=512)
    parser.add_argument("--student-max-sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="restore residual biases, optimizer state, and global step",
    )
    parser.add_argument(
        "--checkpoint-hook",
        type=Path,
        default=None,
        help="rank-zero program invoked after each atomic checkpoint save",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--teacher-target-cache",
        type=Path,
        default=None,
        help=(
            "directory of precomputed teacher endpoints; when present the trainer "
            "never loads the teacher"
        ),
    )
    parser.add_argument(
        "--build-cache-only",
        action="store_true",
        help="run the teacher only and populate --teacher-target-cache, then exit",
    )
    parser.add_argument("--resume", action="store_true", help="cache build: skip existing entries")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.build_cache_only:
        if args.teacher_target_cache is None:
            parser.error("--build-cache-only requires --teacher-target-cache")
        return build_cache(args)
    if args.output is None:
        parser.error("--output is required for training")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
