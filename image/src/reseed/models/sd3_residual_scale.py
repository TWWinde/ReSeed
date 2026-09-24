"""First-interval residual-gain adapters for SD3 joint-transformer blocks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn


class SD3BlockResidualScale(nn.Module):
    """Scale complete block residual updates while preserving identity skips.

    For each selected block this adapter replaces ``y = F(x)`` by

        y' = x + a_b (F(x) - x),

    independently for the image and context streams, using one shared scalar
    ``a_b`` per block.  All original block parameters remain untouched.
    """

    def __init__(self, transformer: Any, blocks: list[int]) -> None:
        super().__init__()
        if not blocks or len(set(blocks)) != len(blocks):
            raise ValueError("blocks must be a non-empty list without duplicates")
        if min(blocks) < 0 or max(blocks) >= len(transformer.transformer_blocks):
            raise ValueError("residual-scale block index is outside the transformer")
        self.block_indices = tuple(blocks)
        self.scales = nn.ParameterDict(
            {str(block): nn.Parameter(torch.ones((), dtype=torch.float32)) for block in blocks}
        )
        self.active = True
        self._input_stacks: dict[int, list[tuple[torch.Tensor, torch.Tensor | None]]] = (
            defaultdict(list)
        )
        self._handles: list[Any] = []
        for block_index in blocks:
            block = transformer.transformer_blocks[block_index]
            self._handles.append(
                block.register_forward_pre_hook(
                    self._pre_hook(block_index), with_kwargs=True
                )
            )
            self._handles.append(block.register_forward_hook(self._post_hook(block_index)))

    def _pre_hook(self, block_index: int):
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden = kwargs.get("hidden_states")
            context = kwargs.get("encoder_hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if context is None and len(args) > 1:
                context = args[1]
            if not torch.is_tensor(hidden):
                raise RuntimeError("could not locate SD3 block image hidden states")
            self._input_stacks[block_index].append((hidden, context))

        return hook

    @staticmethod
    def _scaled(
        block_input: torch.Tensor | None,
        block_output: torch.Tensor | None,
        scale: torch.Tensor,
    ) -> torch.Tensor | None:
        if block_output is None:
            return None
        if block_input is None or block_input.shape != block_output.shape:
            raise RuntimeError("SD3 residual input/output shapes do not match")
        value = scale.to(device=block_output.device, dtype=block_output.dtype)
        return block_input + value * (block_output - block_input)

    def _post_hook(self, block_index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            if not self._input_stacks[block_index]:
                raise RuntimeError("SD3 residual-scale post-hook has no matching input")
            hidden_input, context_input = self._input_stacks[block_index].pop()
            if not self.active:
                return output
            scale = self.scales[str(block_index)]
            if torch.is_tensor(output):
                return self._scaled(hidden_input, output, scale)
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("unexpected SD3 joint-transformer block output")
            context_output, hidden_output = output
            changed = (
                self._scaled(context_input, context_output, scale),
                self._scaled(hidden_input, hidden_output, scale),
            )
            return changed if isinstance(output, tuple) else list(changed)

        return hook

    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def scale_values(self) -> dict[int, float]:
        return {
            int(name): float(parameter.detach().cpu())
            for name, parameter in self.scales.items()
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._input_stacks.clear()


def load_sd3_residual_scale_adapter(
    transformer: Any, checkpoint: str | Path
) -> tuple[SD3BlockResidualScale, dict[str, Any]]:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint / "residual_scales.pt", map_location="cpu", weights_only=True)
    blocks = [int(value) for value in payload["blocks"]]
    adapter = SD3BlockResidualScale(transformer, blocks)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    return adapter, payload


__all__ = ["SD3BlockResidualScale", "load_sd3_residual_scale_adapter"]
