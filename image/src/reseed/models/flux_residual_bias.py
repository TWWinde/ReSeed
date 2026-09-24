"""First-interval additive residual-bias adapters for FLUX transformer blocks.

FLUX (``FluxTransformer2DModel``, diffusers 0.34) has two block families and
they return different things.  Both were read off the installed source before
this adapter was written:

``transformer.transformer_blocks[i]`` -- ``FluxTransformerBlock``
    Double-stream.  ``forward`` returns the tuple
    ``(encoder_hidden_states, hidden_states)``: text tokens first, image tokens
    second (same order as the SD3 joint block).  Both streams are ``inner_dim``
    wide, so this adapter carries two *independent* bias vectors per double
    block -- one for the text stream, one for the image stream.

``transformer.single_transformer_blocks[j]`` -- ``FluxSingleTransformerBlock``
    Single-stream.  ``forward`` returns **one tensor**, not a tuple.  Before the
    single stack runs, the model does
    ``hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)``
    so that tensor holds ``[text tokens ; image tokens]`` concatenated along the
    sequence axis, and the image tokens are sliced back out after the stack via
    ``hidden_states[:, encoder_hidden_states.shape[1]:, ...]``.  A per-channel
    bias is applied uniformly to every token of that tensor, so exactly one
    vector per single block.

Widths come from the transformer config: ``inner_dim = num_attention_heads *
attention_head_dim`` (24 * 128 = 3072 for both FLUX.1-dev and FLUX.1-schnell).
The default selection is all blocks, giving

    19 * 2 * 3072  (double: text + image)  = 116,736
    38 * 1 * 3072  (single)                = 116,736
    ------------------------------------------------
                                             233,472 parameters

Every bias starts at exactly zero, so installing the adapter is an identity
operation until training moves it.  ``disable()`` turns the offsets off without
touching a single frozen student weight, which is what lets the caller apply the
correction on solver interval zero only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


ADAPTER_TYPE = "flux-block-residual-bias-v1"
PAYLOAD_NAME = "residual_biases.pt"


def flux_inner_dim(transformer: Any) -> int:
    """Infer the FLUX hidden width from the transformer config."""

    config = getattr(transformer, "config", None)
    if config is None:
        raise ValueError("transformer has no config; cannot infer the FLUX hidden width")
    heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "attention_head_dim", None)
    if heads is None or head_dim is None:
        # ``FrozenDict`` style configs support item access but not attributes.
        try:
            heads = config["num_attention_heads"] if heads is None else heads
            head_dim = config["attention_head_dim"] if head_dim is None else head_dim
        except Exception as error:  # pragma: no cover - defensive
            raise ValueError("could not read FLUX attention config") from error
    heads = int(heads)
    head_dim = int(head_dim)
    if heads <= 0 or head_dim <= 0:
        raise ValueError("FLUX attention config must be positive")
    return heads * head_dim


def default_flux_blocks(transformer: Any) -> tuple[list[int], list[int]]:
    """Return every double-stream and single-stream block index."""

    return (
        list(range(len(transformer.transformer_blocks))),
        list(range(len(transformer.single_transformer_blocks))),
    )


def _check_indices(name: str, indices: list[int], limit: int) -> tuple[int, ...]:
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicates")
    if indices and (min(indices) < 0 or max(indices) >= limit):
        raise ValueError(f"{name} index is outside the transformer (0..{limit - 1})")
    return tuple(sorted(indices))


class FluxBlockResidualBias(nn.Module):
    """Add learned per-channel biases to FLUX block residual outputs.

    For a selected double block ``i`` the adapter applies

        encoder_out' = encoder_out + double_text_biases[i]
        hidden_out'  = hidden_out  + double_image_biases[i]

    and for a selected single block ``j``

        hidden_out'  = hidden_out  + single_biases[j]

    with each one-dimensional bias broadcast across the batch and token axes.
    All biases are stored in fp32 and cast to the block's dtype at apply time.
    """

    def __init__(
        self,
        transformer: Any,
        double_blocks: list[int] | None = None,
        single_blocks: list[int] | None = None,
    ) -> None:
        super().__init__()
        default_double, default_single = default_flux_blocks(transformer)
        double_blocks = default_double if double_blocks is None else list(double_blocks)
        single_blocks = default_single if single_blocks is None else list(single_blocks)
        if not double_blocks and not single_blocks:
            raise ValueError("at least one double or single block must be selected")

        self.double_block_indices = _check_indices(
            "double_blocks", double_blocks, len(transformer.transformer_blocks)
        )
        self.single_block_indices = _check_indices(
            "single_blocks", single_blocks, len(transformer.single_transformer_blocks)
        )
        self.width = flux_inner_dim(transformer)

        self.double_text_biases = nn.ParameterDict()
        self.double_image_biases = nn.ParameterDict()
        self.single_biases = nn.ParameterDict()
        self.active = True
        self._handles: list[Any] = []

        for block_index in self.double_block_indices:
            self.double_text_biases[str(block_index)] = nn.Parameter(
                torch.zeros(self.width, dtype=torch.float32)
            )
            self.double_image_biases[str(block_index)] = nn.Parameter(
                torch.zeros(self.width, dtype=torch.float32)
            )
            block = transformer.transformer_blocks[block_index]
            self._handles.append(block.register_forward_hook(self._double_hook(block_index)))

        for block_index in self.single_block_indices:
            self.single_biases[str(block_index)] = nn.Parameter(
                torch.zeros(self.width, dtype=torch.float32)
            )
            block = transformer.single_transformer_blocks[block_index]
            self._handles.append(block.register_forward_hook(self._single_hook(block_index)))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_bias(output: torch.Tensor | None, bias: torch.Tensor) -> torch.Tensor | None:
        if output is None:
            return None
        if output.shape[-1] != bias.numel():
            raise RuntimeError(
                f"residual-bias width mismatch: output={output.shape[-1]} bias={bias.numel()}"
            )
        shape = (1,) * (output.ndim - 1) + (bias.numel(),)
        return output + bias.to(device=output.device, dtype=output.dtype).view(shape)

    def _double_hook(self, block_index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            if not self.active:
                return output
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError(
                    "unexpected FluxTransformerBlock output: expected "
                    "(encoder_hidden_states, hidden_states)"
                )
            encoder_output, hidden_output = output
            changed = (
                self._add_bias(encoder_output, self.double_text_biases[str(block_index)]),
                self._add_bias(hidden_output, self.double_image_biases[str(block_index)]),
            )
            return changed if isinstance(output, tuple) else list(changed)

        return hook

    def _single_hook(self, block_index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            if not self.active:
                return output
            bias = self.single_biases[str(block_index)]
            if torch.is_tensor(output):
                return self._add_bias(output, bias)
            # diffusers 0.34 returns a bare tensor; stay tolerant of a future tuple.
            if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
                changed = (self._add_bias(output[0], bias), *output[1:])
                return changed if isinstance(output, tuple) else list(changed)
            raise RuntimeError("unexpected FluxSingleTransformerBlock output")

        return hook

    # ------------------------------------------------------------------ #
    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def statistics(self) -> dict[str, float]:
        values = torch.cat(
            [parameter.detach().float().flatten() for parameter in self.parameters()]
        )
        return {
            "bias_rms": float(values.square().mean().sqrt().cpu()),
            "bias_abs_mean": float(values.abs().mean().cpu()),
            "bias_abs_max": float(values.abs().max().cpu()),
            "bias_parameter_count": int(values.numel()),
        }

    def payload(self, **extra: Any) -> dict[str, Any]:
        """Build the ``residual_biases.pt`` payload for this adapter."""

        return {
            "adapter_type": ADAPTER_TYPE,
            "double_blocks": list(self.double_block_indices),
            "single_blocks": list(self.single_block_indices),
            "width": self.width,
            "state_dict": self.state_dict(),
            **extra,
        }

    def save(self, directory: str | Path, **extra: Any) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / PAYLOAD_NAME
        torch.save(self.payload(**extra), path)
        return path

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def load_flux_residual_bias_payload(checkpoint: str | Path) -> dict[str, Any]:
    """Read and validate a ``residual_biases.pt`` payload from a checkpoint dir."""

    checkpoint = Path(checkpoint)
    path = checkpoint if checkpoint.is_file() else checkpoint / PAYLOAD_NAME
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("adapter_type") != ADAPTER_TYPE:
        raise ValueError(
            f"checkpoint is not a {ADAPTER_TYPE} adapter: {payload.get('adapter_type')!r}"
        )
    return payload


def load_flux_residual_bias_adapter(
    transformer: Any, checkpoint: str | Path
) -> tuple[FluxBlockResidualBias, dict[str, Any]]:
    """Install a saved adapter on ``transformer`` and return it with its payload."""

    payload = load_flux_residual_bias_payload(checkpoint)
    adapter = FluxBlockResidualBias(
        transformer,
        [int(value) for value in payload["double_blocks"]],
        [int(value) for value in payload["single_blocks"]],
    )
    adapter.load_state_dict(payload["state_dict"], strict=True)
    return adapter, payload


__all__ = [
    "ADAPTER_TYPE",
    "PAYLOAD_NAME",
    "FluxBlockResidualBias",
    "default_flux_blocks",
    "flux_inner_dim",
    "load_flux_residual_bias_adapter",
    "load_flux_residual_bias_payload",
]
