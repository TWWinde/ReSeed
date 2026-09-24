#!/usr/bin/env python3
"""Distributed and data-loading helpers shared by the ReSeed bias trainers.

These utilities are backbone-agnostic: process-group setup, gradient
all-reduce, scalar reduction for logging, an atomic JSON writer, and a
deterministic prompt sampler. Both ``sd3m_first_interval_residual_bias`` and
``flux_first_interval_residual_bias`` import from here so the two trainers stay
in lockstep on distributed semantics.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _all_reduce_gradients(parameters: list[torch.nn.Parameter], world_size: int) -> None:
    if world_size == 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _mean(value: torch.Tensor, world_size: int) -> float:
    reduced = value.detach().float().mean()
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced.div_(world_size)
    return float(reduced.cpu())


class PromptStream:
    """Infinite deterministic prompt sampler with per-epoch reshuffling."""

    def __init__(self, prompts: list[str], seed: int) -> None:
        self.prompts = prompts
        self.rng = random.Random(seed)
        self.order = list(range(len(prompts)))
        self.rng.shuffle(self.order)
        self.cursor = 0

    def next(self) -> str:
        if self.cursor == len(self.order):
            self.cursor = 0
            self.rng.shuffle(self.order)
        prompt = self.prompts[self.order[self.cursor]]
        self.cursor += 1
        return prompt
