#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Inference overhead of the ReSeed adapter on the Wan2.1-1.3B DMD2 student.

Measures, on one GPU, the official 4-step student sampler (`model.generator_fn`,
the same call video_model_inference.py makes) with and without the adapter:

  * adapter parameter count and its share of the denoiser (transformer) params
  * per-step latency of the denoiser forward (CUDA events around each call of
    the network; step 1 is the only step where the adapter is active)
  * per-sample latency of generator_fn (4 denoiser steps; VAE decode excluded)
  * peak activation memory during sampling (max_memory_allocated minus the
    resident weights), with and without the adapter

Same prompts, same noise, same order for both configurations; the adapter
configuration is measured second and then first again (A B A) to cancel drift.

    python evaluation/reseed/bench_overhead.py --ckpt_path <dmd2 prefix> \
        --adapter <checkpoint-300 dir> --embeddings <umt5 cache> --output bench.json \
        --config fastgen/configs/experiments/WanT2V/config_dmd2.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from fastgen.utils.scripts import parse_args, setup  # noqa: E402
from scripts.inference.inference_utils import (  # noqa: E402
    cleanup_unused_modules, init_checkpointer, init_model, load_checkpoint,
)
from evaluation.reseed.adapter import FirstStepGate, load_wan_residual_bias_adapter  # noqa: E402
from evaluation.reseed.train_reseed import pad_embed  # noqa: E402


class StepTimer:
    """CUDA-event timing of every forward of the denoiser network."""

    def __init__(self, net: torch.nn.Module) -> None:
        self.times: List[float] = []
        self._start = None
        self._h1 = net.register_forward_pre_hook(self._pre)
        self._h2 = net.register_forward_hook(self._post)

    def _pre(self, *_: Any) -> None:
        self._start = torch.cuda.Event(enable_timing=True); self._start.record()

    def _post(self, *_: Any) -> None:
        end = torch.cuda.Event(enable_timing=True); end.record(); end.synchronize()
        self.times.append(self._start.elapsed_time(end))

    def remove(self) -> None:
        self._h1.remove(); self._h2.remove()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--adapter", required=True, help="checkpoint-N dir with residual_biases.pt")
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = parse_args(ap)
    config = setup(args, evaluation=True)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)

    model = init_model(config)
    checkpointer = init_checkpointer(config)
    ckpt_iter, _ = load_checkpoint(checkpointer, model, args.ckpt_path, config)
    if ckpt_iter is None:
        raise SystemExit("student checkpoint did not load")
    cleanup_unused_modules(model, do_teacher_sampling=False)
    student = model.net.eval().to(dtype=model.precision, device=device).requires_grad_(False)
    torch.cuda.empty_cache()

    t_list = [float(t) for t in config.model.sample_t_cfg.t_list]
    steps = int(config.model.student_sample_steps)
    adapter, payload = load_wan_residual_bias_adapter(student.transformer, args.adapter)
    adapter.to(device)
    adapter_params = sum(p.numel() for p in adapter.parameters())
    denoiser_params = sum(p.numel() for p in student.transformer.parameters())
    net_params = sum(p.numel() for p in student.parameters())
    adapter_bytes = sum(p.numel() * p.element_size() for p in adapter.parameters())

    cache = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    embeds = cache["embeds"][: args.warmup + args.n_prompts]
    neg = cache["negative_embed"].unsqueeze(0).to(device=device, dtype=model.precision)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    noises = [torch.randn([1, *config.model.input_shape], generator=gen, device=device, dtype=model.precision)
              for _ in embeds]
    kwargs = dict(neg_condition=neg, student_sample_steps=steps, student_sample_type=config.model.student_sample_type,
                  t_list=config.model.sample_t_cfg.t_list, precision_amp=model.precision_amp_infer)

    def run(with_adapter: bool) -> Dict[str, Any]:
        gate = FirstStepGate(adapter, t_list[0], t_list[1]).install(student) if with_adapter else None
        if not with_adapter:
            adapter.active = False
        timer = StepTimer(student)
        per_sample: List[float] = []
        per_step: List[List[float]] = []
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        resident = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for i, row in enumerate(embeds):
                cond = pad_embed(row, device, model.precision)
                timer.times.clear()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model.generator_fn(student, noises[i], condition=cond, **kwargs)
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000.0
                if i >= args.warmup:
                    per_sample.append(dt); per_step.append(list(timer.times))
        peak = torch.cuda.max_memory_allocated(device)
        timer.remove()
        if gate is not None:
            gate.remove()
        n_steps = min(len(s) for s in per_step)
        return {
            "per_sample_ms": per_sample,
            "per_sample_ms_median": st.median(per_sample), "per_sample_ms_mean": st.mean(per_sample),
            "per_step_ms_median": [st.median(s[k] for s in per_step) for k in range(n_steps)],
            "step1_ms_median": st.median(s[0] for s in per_step),
            "denoiser_calls_per_sample": n_steps,
            "resident_bytes": resident, "peak_bytes": peak, "peak_activation_bytes": peak - resident,
        }

    print("warmup+measure: baseline -> adapter -> baseline", flush=True)
    base_a = run(False); adap = run(True); base_b = run(False)
    base = base_a                      # the second baseline pass is only a drift check
    drift = base_b["per_sample_ms_median"] - base_a["per_sample_ms_median"]
    mib = 1024 ** 2
    result = {
        "student_checkpoint": str(args.ckpt_path), "student_ckpt_iter": int(ckpt_iter),
        "adapter": str(args.adapter), "adapter_step": payload.get("step"), "blocks": list(payload["blocks"]),
        "adapter_params": adapter_params, "adapter_bytes": adapter_bytes,
        "denoiser_params": denoiser_params, "net_params_incl_text_encoder_vae": net_params,
        "adapter_share_of_denoiser_pct": 100.0 * adapter_params / denoiser_params,
        "gpu": torch.cuda.get_device_name(device), "precision": str(model.precision),
        "input_shape": list(config.model.input_shape), "steps": steps, "t_list": t_list,
        "n_measured": args.n_prompts, "warmup": args.warmup,
        "baseline": base, "adapter_on": adap, "baseline_repeat": base_b,
        "baseline_drift_ms": drift,
        "step1_ms": [base["step1_ms_median"], adap["step1_ms_median"]],
        "per_sample_ms": [base["per_sample_ms_median"], adap["per_sample_ms_median"]],
        "per_sample_overhead_pct": 100.0 * (adap["per_sample_ms_median"] / base["per_sample_ms_median"] - 1.0),
        "step1_overhead_pct": 100.0 * (adap["step1_ms_median"] / base["step1_ms_median"] - 1.0),
        "peak_activation_change_mib": (adap["peak_activation_bytes"] - base["peak_activation_bytes"]) / mib,
        "peak_activation_mib": [base["peak_activation_bytes"] / mib, adap["peak_activation_bytes"] / mib],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("baseline", "adapter_on", "baseline_repeat")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
