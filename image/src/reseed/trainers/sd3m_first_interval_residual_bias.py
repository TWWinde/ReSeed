#!/usr/bin/env python3
"""Train first-interval additive residual biases on frozen SD3M student blocks."""

from __future__ import annotations

import argparse
import json
import random
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
from reseed.diffusion.flow_matching import (
    flow_interval_secant_velocity,
    flow_sigma_boundaries,
    solve_teacher_flow_interval,
    student_flow_interval,
)
from reseed.models.sd3 import (
    encode_sd3_prompts,
    load_sd3_pipeline,
    load_sd3_transformer,
    predict_sd3_velocity,
)
from reseed.models.sd3_residual_bias import SD3BlockResidualBias
from reseed.trainers._common import (
    PromptStream,
    _all_reduce_gradients,
    _distributed,
    _mean,
)


INTERVAL_COUNT = 4


def parse_interval_index(value: str, student_steps: int = INTERVAL_COUNT) -> str | int:
    """Return ``"all"`` or a validated student-interval index."""

    if student_steps <= 0:
        raise ValueError("student_steps must be positive")
    if isinstance(value, str) and value.strip().lower() == "all":
        return "all"
    try:
        index = int(value)
    except (TypeError, ValueError):
        raise ValueError("--interval-index must be an integer or 'all'") from None
    if not 0 <= index < student_steps:
        raise ValueError(f"--interval-index must be in [0, {student_steps - 1}] or 'all'")
    return index


def _save(
    adapter: SD3BlockResidualBias,
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
    torch.save(
        {
            "adapter_type": "sd3-block-residual-bias-v1",
            "blocks": list(adapter.block_indices),
            "image_widths": adapter.image_widths,
            "context_widths": adapter.context_widths,
            "state_dict": adapter.state_dict(),
            "step": step,
            "interval_index": metadata["interval_index"],
            "adapter_scope": metadata["adapter_scope"],
        },
        temporary / "residual_biases.pt",
    )
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


def train(args: argparse.Namespace) -> int:
    rank, _local_rank, world_size, device = _distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    output = args.output.resolve()
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    pipe = load_sd3_pipeline(args.model, torch_dtype=dtype)
    shift = float(pipe.scheduler.config.shift)
    vae_scale_factor = int(pipe.vae_scale_factor)
    latent_channels = int(pipe.transformer.config.in_channels)
    pipe.transformer = None
    pipe.to(device)
    pipe.vae.to("cpu")
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipe, name, None)
        if encoder is not None:
            encoder.eval().requires_grad_(False)

    teacher = load_sd3_transformer(args.model, torch_dtype=dtype).to(device)
    teacher.eval().requires_grad_(False)
    student = load_sd3_transformer(args.student_checkpoint, torch_dtype=dtype).to(device)
    student.eval().requires_grad_(False)
    blocks = list(range(args.block_start, args.block_end + 1))
    adapter = SD3BlockResidualBias(student, blocks).to(device)
    trainable = list(adapter.parameters())
    if not trainable:
        raise RuntimeError("residual-bias adapter selected no parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    prompts = read_prompts(args.prompt_file, limit=args.prompt_limit)
    prompt_stream = PromptStream(prompts, args.seed + 1009 * rank)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 104729 * rank)
    boundaries = flow_sigma_boundaries(
        args.student_steps, shift, device=device, dtype=dtype
    )
    interval_index = parse_interval_index(args.interval_index, args.student_steps)
    interval_choices = (
        list(range(args.student_steps)) if interval_index == "all" else [interval_index]
    )
    interval_rng = random.Random(args.seed + 7919 * rank)
    spatial = args.resolution // vae_scale_factor
    latent_shape = (args.batch_size, latent_channels, spatial, spatial)
    metadata = {
        "method": "sd3m-first-interval-block-residual-bias",
        "base_student": str(args.student_checkpoint.resolve()),
        "teacher": str(args.model.resolve()),
        "residual_definition": "image/context block output plus learned channel bias",
        "zero_initialized": True,
        "blocks": blocks,
        "image_widths": adapter.image_widths,
        "context_widths": adapter.context_widths,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "teacher_micro_steps": args.teacher_micro_steps,
        "teacher_guidance_scale": args.teacher_guidance_scale,
        "teacher_cfg_threshold": args.teacher_cfg_threshold,
        "student_guidance_scale": 1.0,
        "student_steps": args.student_steps,
        "student_boundaries": [float(value) for value in boundaries],
        "interval_index": interval_index,
        "trained_intervals": interval_choices,
        "adapter_scope": (
            "first student interval only; disable for suffix intervals"
            if interval_index == 0
            else (
                "one shared bias trained on a uniformly sampled interval k in "
                f"{interval_choices}; enable on exactly the intervals it was trained on"
                if interval_index == "all"
                else f"student interval {interval_index} only; disable on every other interval"
            )
        ),
        "training_input_state": (
            "initial noise"
            if interval_index == 0
            else "frozen student (no adapter) rollout from noise to boundary k"
        ),
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "loss": "MSE(student velocity at sigma_k, teacher interval-k secant velocity)",
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
        for step in range(1, args.max_steps + 1):
            prompts_batch = [prompt_stream.next() for _ in range(args.batch_size)]
            conditioning = encode_sd3_prompts(
                pipe,
                prompts_batch,
                device=device,
                max_sequence_length=args.max_sequence_length,
                include_unconditional=True,
            )
            noise = torch.randn(latent_shape, generator=noise_generator, device=device, dtype=dtype)

            def teacher_velocity(latents: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
                with torch.autocast("cuda", dtype=dtype):
                    return predict_sd3_velocity(
                        teacher,
                        latents,
                        sigmas,
                        conditioning,
                        guidance_scale=args.teacher_guidance_scale,
                        cfg_control=args.teacher_cfg_threshold,
                    )

            def frozen_student_velocity(
                latents: torch.Tensor, sigmas: torch.Tensor
            ) -> torch.Tensor:
                with torch.autocast("cuda", dtype=dtype):
                    return predict_sd3_velocity(
                        student, latents, sigmas, conditioning, guidance_scale=1.0
                    )

            step_interval = (
                interval_choices[0]
                if len(interval_choices) == 1
                else interval_rng.choice(interval_choices)
            )
            sigma_k = boundaries[step_interval]
            sigma_next = boundaries[step_interval + 1]

            # The training input at boundary k is the frozen student's own state:
            # roll the base S7 forward with the adapter switched off so the prefix
            # is exactly the deployed suffix behaviour.
            state = noise
            if step_interval > 0:
                adapter.disable()
                with torch.no_grad():
                    for prefix_index in range(step_interval):
                        state = student_flow_interval(
                            frozen_student_velocity,
                            state,
                            sigma=boundaries[prefix_index],
                            next_sigma=boundaries[prefix_index + 1],
                        ).detach()
                adapter.enable()

            with torch.no_grad():
                teacher_endpoint = solve_teacher_flow_interval(
                    teacher_velocity,
                    state,
                    boundaries=boundaries,
                    interval_index=step_interval,
                    micro_steps=args.teacher_micro_steps,
                    shift=shift,
                )
                target_velocity = flow_interval_secant_velocity(
                    state.float(),
                    teacher_endpoint.float(),
                    sigma=sigma_k,
                    next_sigma=sigma_next,
                )

            optimizer.zero_grad(set_to_none=True)
            sigma_batch = sigma_k.to(dtype=dtype).expand(args.batch_size)
            with torch.autocast("cuda", dtype=dtype):
                predicted_velocity = predict_sd3_velocity(
                    student,
                    state,
                    sigma_batch,
                    conditioning,
                    guidance_scale=1.0,
                )
            loss = F.mse_loss(predicted_velocity.float(), target_velocity)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            if any(parameter.grad is not None for parameter in student.parameters()):
                raise RuntimeError("a frozen base-student parameter received a gradient")
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
                endpoint_error = (
                    state.float()
                    + (sigma_next - sigma_k) * predicted_velocity.detach().float()
                    - teacher_endpoint.float()
                ).flatten(1).norm(dim=1) / teacher_endpoint.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                bias_values = torch.cat([parameter.detach().float().flatten() for parameter in trainable])
            row = {
                "step": step,
                "interval_index": step_interval,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument(
        "--student-steps",
        type=int,
        default=4,
        help="Student step count whose first interval is corrected.",
    )
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--block-end", type=int, default=23)
    parser.add_argument(
        "--interval-index",
        default="0",
        help=(
            "Student interval the bias is trained on: an index in [0, 3], or 'all' to "
            "sample k uniformly from {0, 1, 2, 3} at every step and train one shared "
            "bias. For k > 0 the input state comes from a frozen adapter-free student "
            "rollout over intervals 0..k-1."
        ),
    )
    parser.add_argument("--teacher-micro-steps", type=int, default=7)
    parser.add_argument("--teacher-guidance-scale", type=float, default=7.0)
    parser.add_argument("--teacher-cfg-threshold", type=float, default=0.96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument(
        "--checkpoint-hook",
        type=Path,
        help="optional program called immediately after each atomic checkpoint save",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.student_steps < 1:
        parser.error("--student-steps must be positive")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
