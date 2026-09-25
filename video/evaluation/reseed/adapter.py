# SPDX-License-Identifier: Apache-2.0
"""First-step residual-bias adapter for Wan transformer blocks (ReSeed, ported).

Port of the image-side SD3BlockResidualBias to Wan2.1. The SD3 adapter carried an
image bias and a context bias per joint block because MMDiT updates both
streams; Wan's blocks are single-stream -- cross-attention reads the text
embeddings but never writes them back -- so there is exactly one bias per
block:

    hidden_states_b <- hidden_states_b + beta_b            beta_b in R^1536

broadcast over batch and over all 21*30*52 = 32,760 spatio-temporal tokens.
That broadcast is the "direct port": the bias has no time axis. It is
zero-initialised, so an untrained adapter is an exact identity, and `active`
gates it so it can be switched on for the first solver step only.

The hook rides on nn.Module.__call__. fastgen calls each block as `block(...)`
(and, under gradient checkpointing, hands the module itself to the checkpoint
function), so forward hooks fire on both the normal and the recomputed pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn

ADAPTER_TYPE = "wan-block-residual-bias-v1"


def _block_width(block: nn.Module) -> int:
    """Hidden width from the self-attention output projection."""
    to_out = getattr(getattr(block, "attn1", None), "to_out", None)
    proj = to_out[0] if to_out is not None else None
    width = getattr(proj, "out_features", None)
    if not isinstance(width, int) or width <= 0:
        raise ValueError("could not infer Wan block width from attn1.to_out.0")
    return width


class WanBlockResidualBias(nn.Module):
    def __init__(self, transformer: Any, blocks: Iterable[int]) -> None:
        super().__init__()
        blocks = list(blocks)
        n = len(transformer.blocks)
        if not blocks or len(set(blocks)) != len(blocks):
            raise ValueError("blocks must be a non-empty list without duplicates")
        if min(blocks) < 0 or max(blocks) >= n:
            raise ValueError(f"block index outside [0, {n - 1}]")

        self.block_indices = tuple(blocks)
        self.biases = nn.ParameterDict()
        self.widths: Dict[int, int] = {}
        self.active = True
        self._handles: List[Any] = []

        for b in blocks:
            block = transformer.blocks[b]
            w = _block_width(block)
            self.widths[b] = w
            self.biases[str(b)] = nn.Parameter(torch.zeros(w, dtype=torch.float32))
            self._handles.append(block.register_forward_hook(self._post_hook(b)))

    def _post_hook(self, b: int):
        def hook(_module: Any, _args: tuple, output: Any) -> Any:
            if not self.active:
                return output
            if not torch.is_tensor(output):
                raise RuntimeError(f"Wan block {b} returned {type(output).__name__}, expected a tensor")
            bias = self.biases[str(b)]
            if output.shape[-1] != bias.numel():
                raise RuntimeError(f"width mismatch at block {b}: output {output.shape[-1]} vs bias {bias.numel()}")
            shape = (1,) * (output.ndim - 1) + (bias.numel(),)
            return output + bias.to(device=output.device, dtype=output.dtype).view(shape)
        return hook

    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def statistics(self) -> Dict[str, float]:
        v = torch.cat([p.detach().float().flatten() for p in self.parameters()])
        return {
            "bias_rms": float(v.square().mean().sqrt()),
            "bias_abs_mean": float(v.abs().mean()),
            "bias_abs_max": float(v.abs().max()),
            "num_parameters": int(v.numel()),
        }

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---- persistence ------------------------------------------------------
    def payload(self, **extra: Any) -> Dict[str, Any]:
        return {
            "adapter_type": ADAPTER_TYPE,
            "blocks": list(self.block_indices),
            "widths": dict(self.widths),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            **extra,
        }

    def save(self, directory: Path, **extra: Any) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "residual_biases.pt"
        torch.save(self.payload(**extra), path)
        return path


def load_wan_residual_bias_adapter(
    transformer: Any, checkpoint: str | Path
) -> tuple[WanBlockResidualBias, Dict[str, Any]]:
    checkpoint = Path(checkpoint)
    path = checkpoint / "residual_biases.pt" if checkpoint.is_dir() else checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("adapter_type") != ADAPTER_TYPE:
        raise ValueError(f"{path} is not a {ADAPTER_TYPE} checkpoint")
    adapter = WanBlockResidualBias(transformer, [int(b) for b in payload["blocks"]])
    adapter.load_state_dict(payload["state_dict"], strict=True)
    return adapter, payload


class FirstStepGate:
    """Switch an adapter on for the first student step only.

    Installed as a forward *pre*-hook on the Wan network. The network's second
    positional argument is the timestep batch `t`; the adapter is active iff
    t is at (or above) the first-interval start. With t_list =
    [0.999, 0.937, ...] that means step 1 and nothing else. Reading `t`
    consumes no RNG, so installing this leaves the sampled noise untouched.
    """

    def __init__(self, adapter: WanBlockResidualBias, t_first: float, t_second: float) -> None:
        if not t_first > t_second:
            raise ValueError("t_first must exceed t_second")
        self.adapter = adapter
        # Anything strictly above the second boundary is the first interval.
        self.threshold = 0.5 * (t_first + t_second)
        self._handle: Optional[Any] = None
        self.last_active: Optional[bool] = None

    def _pre_hook(self, _module: Any, args: tuple, kwargs: Dict[str, Any]) -> None:
        t = args[1] if len(args) > 1 else kwargs.get("t")
        if t is None:
            raise RuntimeError("FirstStepGate could not find the timestep argument")
        t0 = float(torch.as_tensor(t).flatten()[0])
        self.last_active = t0 > self.threshold
        self.adapter.active = self.last_active

    def install(self, net: nn.Module) -> "FirstStepGate":
        self._handle = net.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
