#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""ReSeed on Wan2.1: fit a first-step residual bias on a frozen distilled student.

Port of the image-side sd3m_first_interval_residual_bias.py. Per update, per rank:

    z ~ N(0, I)                                            [1,16,21,60,104]
    x_end = teacher, m boundary-aligned Euler micro-steps   t0 -> t1, CFG
    v_T   = (x_end - z) / (t1 - t0)                        secant, stop-grad
    v_S   = (z - x0_hat(z; beta)) / t0                     student, bias ON
    L     = || v_S - v_T ||^2                              grads reach beta only

The model is built through the SAME calls the inference script uses
(init_model -> load_checkpoint -> cleanup_unused_modules), so the student here
is byte-for-byte the student that will be evaluated, and the teacher is the
one it was distilled against (shift 5, CFG 5).

Everything velocity-shaped is expressed through fastgen's own x0 ->
forward_process path, never its "flow" prediction type; see flow.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fastgen.utils.scripts import parse_args, setup  # noqa: E402
from fastgen.utils.distributed import get_rank, world_size as ws  # noqa: E402
from scripts.inference.inference_utils import (  # noqa: E402
    cleanup_unused_modules, init_checkpointer, init_model, load_checkpoint,
)
from evaluation.reseed.adapter import WanBlockResidualBias  # noqa: E402
from evaluation.reseed.flow import (  # noqa: E402
    boundary_aligned_teacher_grid, secant_velocity, solve_teacher_interval, student_velocity,
)

MAX_TOKENS = 512


class PromptStream:
    """Shuffled, endlessly cycling index stream; each rank gets its own order."""

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.rng, self.order, self.pos = n, random.Random(seed), [], 0

    def next(self) -> int:
        if self.pos >= len(self.order):
            self.order = list(range(self.n)); self.rng.shuffle(self.order); self.pos = 0
        i = self.order[self.pos]; self.pos += 1
        return i


def pad_embed(row: torch.Tensor, device, dtype) -> torch.Tensor:
    """[L, 4096] -> [1, 512, 4096], zero-padded exactly like WanTextEncoder.encode."""
    L = row.shape[0]
    out = torch.zeros(MAX_TOKENS, row.shape[1], dtype=dtype, device=device)
    out[:L] = row.to(device=device, dtype=dtype)
    return out.unsqueeze(0)


def mean_all(x: torch.Tensor, n: int) -> float:
    x = x.detach().float().clone()
    if n > 1:
        dist.all_reduce(x); x /= n
    return float(x)


def save_checkpoint(adapter, optimizer, out: Path, step: int, meta: Dict[str, Any], keep_last: int) -> Path:
    final = out / f"checkpoint-{step}"
    tmp = out / f".checkpoint-{step}.tmp"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    adapter.save(tmp, step=step, **{k: meta[k] for k in ("t_list", "shift", "micro_steps", "teacher_guidance_scale",
                                                            "active_interval", "student_checkpoint", "student_ckpt_iter")})
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, tmp / "trainer_state.pt")
    (tmp / "manifest.json").write_text(json.dumps({**meta, "step": step, **adapter.statistics()}, indent=2) + "\n")
    tmp.rename(final)
    olds = sorted((p for p in out.glob("checkpoint-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[-1]))
    for p in olds[:-keep_last]:
        shutil.rmtree(p)
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # fastgen's parse_args adds --config / --log_level / opts / --dryrun itself.
    ap.add_argument("--ckpt_path", required=True, help="student DCP prefix, e.g. .../dmd2_t2v_vipe/0001000")
    ap.add_argument("--embeddings", type=Path, required=True, help="output of encode_prompts.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--micro-steps", type=int, default=13)
    ap.add_argument("--teacher-guidance-scale", type=float, default=None, help="default: config.model.guidance_scale")
    ap.add_argument("--blocks", default="0-29")
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--adam-betas", default="0.9,0.999")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--keep-last", type=int, default=1000)
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="trade ~40 GB of activation memory for speed on the student pass")
    ap.add_argument("--skip-identity-check", action="store_true")
    ap.add_argument("--resume", default="auto", choices=("auto", "never"),
                    help="auto: continue from the latest checkpoint-N in --output if one exists")
    args = parse_args(ap)
    config = setup(args, evaluation=True)          # process group, cudnn flags, config overrides

    rank, world = get_rank(), ws()
    is_main = rank == 0
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank); random.seed(args.seed + rank)

    # ---- schedule contract from the official config ----------------------
    t_list = [float(t) for t in config.model.sample_t_cfg.t_list]
    shift = float(config.model.sample_t_cfg.shift)
    steps = int(config.model.student_sample_steps)
    if len(t_list) != steps + 1 or config.model.student_sample_type != "ode":
        raise SystemExit(f"unexpected student schedule: steps={steps} t_list={t_list} type={config.model.student_sample_type}")
    t0, t1 = t_list[0], t_list[1]
    grid = boundary_aligned_teacher_grid(t_list, 0, args.micro_steps, shift)
    guidance = float(args.teacher_guidance_scale if args.teacher_guidance_scale is not None else config.model.guidance_scale)

    # ---- model: identical construction path to the inference script -------
    model = init_model(config)
    checkpointer = init_checkpointer(config)
    ckpt_iter, _ = load_checkpoint(checkpointer, model, args.ckpt_path, config)
    if ckpt_iter is None:
        raise SystemExit("student checkpoint did not load -- refusing to fit a bias on the undistilled base model")
    cleanup_unused_modules(model, do_teacher_sampling=True)     # drops fake_score/discriminator, KEEPS teacher
    student = model.net.eval().to(dtype=model.precision, device=device).requires_grad_(False)
    teacher = model.teacher.eval().to(dtype=model.precision, device=device).requires_grad_(False)
    sched = student.noise_scheduler
    if args.no_gradient_checkpointing:
        student.transformer.disable_gradient_checkpointing()
    torch.cuda.empty_cache()

    lo, hi = (int(v) for v in args.blocks.split("-"))
    blocks = list(range(lo, hi + 1))
    adapter = WanBlockResidualBias(student.transformer, blocks).to(device)
    params = list(adapter.parameters())
    b1, b2 = (float(v) for v in args.adam_betas.split(","))
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(b1, b2), eps=1e-8, weight_decay=args.weight_decay)

    # ---- data --------------------------------------------------------------
    cache = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    if cache.get("format") != "wan-umt5-embeddings-v1":
        raise SystemExit(f"{args.embeddings} is not an encode_prompts.py cache")
    embeds: List[torch.Tensor] = cache["embeds"]
    neg = cache["negative_embed"].unsqueeze(0).to(device=device, dtype=model.precision)
    stream = PromptStream(len(embeds), args.seed + 1009 * rank)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 104729 * rank)
    latent_shape = (1, *config.model.input_shape)

    meta = {
        "method": "reseed-wan-first-interval-block-residual-bias",
        "student_checkpoint": str(args.ckpt_path), "student_ckpt_iter": int(ckpt_iter),
        "teacher": str(config.model.net.model_id_or_local_path),
        "t_list": t_list, "shift": shift, "active_interval": 0,
        "micro_steps": args.micro_steps, "teacher_grid": [float(g) for g in grid],
        "teacher_guidance_scale": guidance, "student_guidance_scale": None,
        "blocks": blocks, "trainable_parameters": sum(p.numel() for p in params),
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "embeddings": str(args.embeddings), "captions_sha256": cache.get("captions_sha256"), "n_prompts": len(embeds),
        "world_size": world, "batch_per_rank": 1, "global_batch": world,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "loss": "MSE(student first-step velocity, teacher boundary-aligned secant velocity)",
        "lr": args.learning_rate, "max_steps": args.max_steps, "seed": args.seed,
    }
    if is_main:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "training_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2), flush=True)
        metrics = (args.output / "metrics.jsonl").open("a", buffering=1)
    else:
        metrics = None
    if world > 1:
        dist.barrier()

    def cond_for(i: int) -> torch.Tensor:
        return pad_embed(embeds[i], device, model.precision)

    def teacher_x0(x: torch.Tensor, tb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # CFG on x0 equals CFG on velocity: x0 = x - t v is linear in v.
        xc = teacher(x, tb, condition=cond, fwd_pred_type="x0").float()
        xu = teacher(x, tb, condition=neg, fwd_pred_type="x0").float()
        return (xu + guidance * (xc - xu)).to(x.dtype)

    # ---- identity check: zero bias must be a no-op on the real model --------
    if not args.skip_identity_check:
        z = torch.randn(latent_shape, generator=noise_gen, device=device, dtype=model.precision)
        tb = torch.full((1,), t0, device=device, dtype=sched.t_precision)
        cond = cond_for(0)
        with torch.no_grad():
            adapter.disable(); ref = student(z, tb, condition=cond, fwd_pred_type="x0")
            adapter.enable(); got = student(z, tb, condition=cond, fwd_pred_type="x0")
        if not torch.equal(ref, got):
            raise SystemExit("zero-initialised adapter changed the student output -- hook placement is wrong")
        if is_main:
            print("identity check: zero-bias adapter is bit-identical to the bare student  OK", flush=True)

    # ---- resume: a 17 h run on a cluster with a third of its nodes down ----
    # must survive NODE_FAIL + --requeue. Adapter and optimizer come from the
    # checkpoint; each rank's noise generator and prompt stream come from its
    # own rank_state file so the presentation sequence continues rather than
    # restarting from the seed.
    start_step = 0
    if args.resume == "auto":
        ckpts = sorted((p for p in args.output.glob("checkpoint-*") if p.is_dir()),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            latest = ckpts[-1]
            payload = torch.load(latest / "residual_biases.pt", map_location="cpu", weights_only=True)
            adapter.load_state_dict(payload["state_dict"], strict=True)
            st = torch.load(latest / "trainer_state.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(st["optimizer"])
            start_step = int(st["step"])
            rs_path = latest / f"rank_state_{rank}.pt"
            if rs_path.is_file():
                rs = torch.load(rs_path, map_location="cpu", weights_only=False)
                noise_gen.set_state(rs["noise_gen"].to("cpu") if hasattr(rs["noise_gen"], "to") else rs["noise_gen"])
                stream.order, stream.pos = list(rs["stream_order"]), int(rs["stream_pos"])
                stream.rng.setstate(rs["stream_rng"])
                how = "with rank RNG/stream state"
            else:
                noise_gen.manual_seed(args.seed + 104729 * rank + 7 * start_step)
                how = "WITHOUT rank state (file missing) -- reseeded with a step offset"
            if is_main:
                print(f"resumed from {latest} at step {start_step} {how}", flush=True)
                metrics.write(json.dumps({"event": "resume", "from_step": start_step, "checkpoint": str(latest)}) + "\n")

    def rank_state() -> Dict[str, Any]:
        return {"rank": rank, "noise_gen": noise_gen.get_state(), "stream_order": list(stream.order),
                "stream_pos": stream.pos, "stream_rng": stream.rng.getstate()}

    t0b = torch.full((1,), t0, device=device, dtype=sched.t_precision)
    started = time.monotonic()
    try:
        for step in range(start_step + 1, args.max_steps + 1):
            idx = stream.next()
            cond = cond_for(idx)
            z = torch.randn(latent_shape, generator=noise_gen, device=device, dtype=model.precision)

            tt = time.monotonic()
            with torch.no_grad():                       # not inference_mode: v_T feeds an autograd op
                x_end = solve_teacher_interval(lambda x, tb: teacher_x0(x, tb, cond), sched, z, grid)
                v_t = secant_velocity(z, x_end, t0, t1)
            teacher_s = time.monotonic() - tt

            ts = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            adapter.enable()
            with torch.enable_grad():
                x0_hat = student(z, t0b, condition=cond, fwd_pred_type="x0")
            v_s = student_velocity(z, x0_hat, t0b)
            loss = F.mse_loss(v_s, v_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            if any(p.grad is not None for p in student.parameters()):
                raise RuntimeError("a frozen student parameter received a gradient")
            if step == 1 and not any(p.grad is not None and p.grad.abs().sum() > 0 for p in params):
                raise RuntimeError("bias received no gradient at step 1 -- gradient checkpointing is reentrant? "
                                   "rerun with --no-gradient-checkpointing")
            if world > 1:
                for p in params:
                    dist.all_reduce(p.grad); p.grad /= world
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            if not torch.isfinite(gnorm):
                raise RuntimeError(f"non-finite gradient at step {step}")
            optimizer.step()
            student_s = time.monotonic() - ts

            with torch.no_grad():
                p_, t_ = v_s.detach().flatten(1), v_t.flatten(1)
                cos = F.cosine_similarity(p_, t_, dim=1).mean()
                rel = ((p_ - t_).norm(dim=1) / t_.norm(dim=1).clamp_min(1e-8)).mean()
                x1_s = z.float() + (t1 - t0) * v_s.detach()
                ep = ((x1_s - x_end.float()).flatten(1).norm(dim=1) / x_end.float().flatten(1).norm(dim=1).clamp_min(1e-8)).mean()
                bv = torch.cat([p.detach().float().flatten() for p in params])
            row = {
                "step": step, "prompt_idx": idx,
                "loss": mean_all(loss, world), "velocity_cosine": mean_all(cos, world),
                "velocity_relative_error": mean_all(rel, world), "endpoint_relative_error": mean_all(ep, world),
                "gradient_norm": mean_all(torch.as_tensor(gnorm, device=device), world),
                "bias_rms": float(bv.square().mean().sqrt()), "bias_abs_max": float(bv.abs().max()),
                "teacher_seconds": round(teacher_s, 2), "student_seconds": round(student_s, 2),
                "wall_seconds": round(time.monotonic() - started, 1),
            }
            if is_main and (step == 1 or step % args.log_every == 0):
                metrics.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.max_steps:
                if world > 1:
                    dist.barrier()
                if is_main:
                    print(f"saved {save_checkpoint(adapter, optimizer, args.output, step, meta, args.keep_last)}", flush=True)
                if world > 1:
                    dist.barrier()
                torch.save(rank_state(), args.output / f"checkpoint-{step}" / f"rank_state_{rank}.pt")
                if world > 1:
                    dist.barrier()
    finally:
        adapter.close()
        if metrics is not None:
            metrics.close()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
