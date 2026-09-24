"""First-interval additive residual-bias adapters for SD3 joint blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _block_widths(block: nn.Module) -> tuple[int, int | None]:
    """Infer image/context output widths from the block's attention projections."""

    attention = getattr(block, "attn", None)
    to_out = getattr(attention, "to_out", None)
    image_projection = to_out[0] if to_out is not None else None
    image_width = getattr(image_projection, "out_features", None)
    if not isinstance(image_width, int) or image_width <= 0:
        raise ValueError("could not infer SD3 block image width from attn.to_out.0")

    if bool(getattr(block, "context_pre_only", False)):
        return image_width, None
    context_projection = getattr(attention, "to_add_out", None)
    context_width = getattr(context_projection, "out_features", None)
    if not isinstance(context_width, int) or context_width <= 0:
        raise ValueError("could not infer SD3 block context width from attn.to_add_out")
    return image_width, context_width


class SD3BlockResidualBias(nn.Module):
    """Add learned channel biases to complete joint-block residual outputs.

    For block ``b`` this adapter applies

        image_out' = image_out + bias_image[b]
        context_out' = context_out + bias_context[b]

    with each one-dimensional bias broadcast across batch and token axes.  All
    biases start at zero, so installing the adapter is initially an exact
    identity operation.  The adapter can be disabled after the first solver
    interval without changing any frozen student parameter.
    """

    def __init__(self, transformer: Any, blocks: list[int]) -> None:
        super().__init__()
        if not blocks or len(set(blocks)) != len(blocks):
            raise ValueError("blocks must be a non-empty list without duplicates")
        if min(blocks) < 0 or max(blocks) >= len(transformer.transformer_blocks):
            raise ValueError("residual-bias block index is outside the transformer")

        self.block_indices = tuple(blocks)
        self.image_biases = nn.ParameterDict()
        self.context_biases = nn.ParameterDict()
        self.image_widths: dict[int, int] = {}
        self.context_widths: dict[int, int | None] = {}
        self.active = True
        self._handles: list[Any] = []

        for block_index in blocks:
            block = transformer.transformer_blocks[block_index]
            image_width, context_width = _block_widths(block)
            self.image_widths[block_index] = image_width
            self.context_widths[block_index] = context_width
            self.image_biases[str(block_index)] = nn.Parameter(
                torch.zeros(image_width, dtype=torch.float32)
            )
            if context_width is not None:
                self.context_biases[str(block_index)] = nn.Parameter(
                    torch.zeros(context_width, dtype=torch.float32)
                )
            self._handles.append(block.register_forward_hook(self._post_hook(block_index)))

    @staticmethod
    def _add_bias(output: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor | None:
        if output is None:
            return None
        if bias is None:
            raise RuntimeError("missing residual bias for a tensor block output")
        if output.shape[-1] != bias.numel():
            raise RuntimeError(
                f"residual-bias width mismatch: output={output.shape[-1]} bias={bias.numel()}"
            )
        shape = (1,) * (output.ndim - 1) + (bias.numel(),)
        return output + bias.to(device=output.device, dtype=output.dtype).view(shape)

    def _post_hook(self, block_index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            if not self.active:
                return output
            image_bias = self.image_biases[str(block_index)]
            if torch.is_tensor(output):
                return self._add_bias(output, image_bias)
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("unexpected SD3 joint-transformer block output")
            context_output, image_output = output
            context_bias = self.context_biases.get(str(block_index))
            changed = (
                self._add_bias(context_output, context_bias),
                self._add_bias(image_output, image_bias),
            )
            return changed if isinstance(output, tuple) else list(changed)

        return hook

    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def statistics(self) -> dict[str, float]:
        values = torch.cat([parameter.detach().float().flatten() for parameter in self.parameters()])
        return {
            "bias_rms": float(values.square().mean().sqrt().cpu()),
            "bias_abs_mean": float(values.abs().mean().cpu()),
            "bias_abs_max": float(values.abs().max().cpu()),
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def load_sd3_residual_bias_adapter(
    transformer: Any, checkpoint: str | Path
) -> tuple[SD3BlockResidualBias, dict[str, Any]]:
    checkpoint = Path(checkpoint)
    payload = torch.load(
        checkpoint / "residual_biases.pt", map_location="cpu", weights_only=True
    )
    if payload.get("adapter_type") != "sd3-block-residual-bias-v1":
        raise ValueError("checkpoint is not an SD3 residual-bias adapter")
    blocks = [int(value) for value in payload["blocks"]]
    adapter = SD3BlockResidualBias(transformer, blocks)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    return adapter, payload


__all__ = ["SD3BlockResidualBias", "load_sd3_residual_bias_adapter"]
