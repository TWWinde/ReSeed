#!/usr/bin/env python3
"""Per-case seeded FLUX.1 generation with an optional first-interval bias.

The sampling loop is written out by hand rather than delegated to
``FluxPipeline.__call__`` so the residual-bias hooks can be toggled *per solver
step*: the learned correction is applied on step 0 only and every later step
runs the frozen model unchanged.  The loop reproduces the diffusers 0.34
pipeline exactly -- see ``reseed.models.flux`` for the
line-by-line notes on prompt encoding, packing, the sigma schedule, the
``timestep / 1000`` bf16 rounding, the guidance vector and the VAE decode.

The schedule is not taken on trust.  Unless ``--skip-schedule-check`` is
passed, ``generate()`` builds the real ``FlowMatchEulerDiscreteScheduler`` from
the model snapshot, calls ``set_timesteps(sigmas=linspace(1, 1/steps, steps),
mu=calculate_shift(image_seq_len, ...))`` exactly as the pipeline does, and
asserts our sigma list equals ``scheduler.sigmas`` element for element.
``tests/test_flux_schedule.py`` pins the same numbers offline against hard-coded
expected values derived from the diffusers formulas.

Manifest, provenance, sharding and finalization follow the same contract as
``tools/sd3m_first_interval_bias_variant_suffix.py`` so
``reseed eval paired-score``, ``tools/feature_diversity_eval.py score``
and ``tools/prompt_lpips_diversity.py`` accept the finalized manifest unchanged:
every row carries ``prompt_id``, ``sample_id``, ``prompt``, ``evaluation_seed``,
``latent_sha256`` and ``image_path``.

Model defaults point at the local Hugging Face cache snapshots (``$HF_HUB_CACHE``)
and can be overridden through ``--dev-model`` / ``--schnell-model`` or the
``RESEED_FLUX_DEV_MODEL`` / ``RESEED_FLUX_SCHNELL_MODEL`` environment variables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


# Shared with tools/sd3m_first_interval_bias_variant_suffix.py so that a given
# --bias-random-seed / --bias-shuffle-seed names the same construction rule.
RANDOM_SEED_BASE = 20260910
SHUFFLE_SEED_BASE = 31415926

MODE = "flux-first-interval-residual-bias"
BASELINE_MODE = "flux-baseline"

HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
DEFAULT_MODEL_PATHS = {
    "dev": os.environ.get(
        "RESEED_FLUX_DEV_MODEL",
        f"{HUB}/models--black-forest-labs--FLUX.1-dev/snapshots/"
        "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
    ),
    "schnell": os.environ.get(
        "RESEED_FLUX_SCHNELL_MODEL",
        f"{HUB}/models--black-forest-labs--FLUX.1-schnell/snapshots/"
        "741f7c3ce8b383c54771c7003378a50191e9efe9",
    ),
    "senseflow": os.environ.get(
        "RESEED_SENSEFLOW_FLUX_TRANSFORMER",
        "models/SenseFlow/SenseFlow-FLUX",
    ),
}
DEFAULT_STEPS = {"dev": 28, "schnell": 4, "senseflow": 4}
DEFAULT_GUIDANCE = {"dev": 3.5, "schnell": None, "senseflow": None}
DEFAULT_MAX_SEQUENCE_LENGTH = {"dev": 512, "schnell": 256, "senseflow": 512}


# --------------------------------------------------------------------------- #
# prompt/seed bank
# --------------------------------------------------------------------------- #
def read_records(path: Path, expected_count: int, max_cases: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    # Only the three columns every prompt bank carries are mandatory, matching
    # ``reseed.evaluation.sd3m_generate._records``.  The COCO5K quality
    # bank has no ``prompt_id``/``sample_id`` -- it is one image per prompt --
    # while the 100x10 diversity bank does; whatever columns a bank supplies are
    # copied through to the manifest verbatim, so the DINO/CLIP/LPIPS scorers
    # still receive the grouping keys they need on the diversity bank.
    required = {"case_number", "evaluation_seed", "prompt"}
    missing = sorted(required - set(records[0] if records else required))
    if not records or missing:
        raise ValueError(f"prompt CSV is missing columns: {missing or sorted(required)}")
    records.sort(key=lambda row: int(row["case_number"]))
    if max_cases is not None:
        records = records[:max_cases]
    if len(records) != expected_count:
        raise ValueError(f"expected {expected_count} prompt/seed cases, found {len(records)}")
    if len({int(row["case_number"]) for row in records}) != expected_count:
        raise ValueError("case_number values must be unique")
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# bias variant construction (same rules as the SD3M variant tool)
# --------------------------------------------------------------------------- #
def _bias_keys(payload: dict[str, Any]) -> list[tuple[str, int, str]]:
    """Deterministic ``(kind, block, state-dict key)`` order used by every RNG draw.

    Double blocks ascending -- text bias then image bias for each -- followed by
    single blocks ascending.  Any RNG-driven variant consumes the generator in
    exactly this order, so a given seed reproduces the same variant.
    """

    state_dict = payload["state_dict"]
    keys: list[tuple[str, int, str]] = []
    for block in sorted(int(value) for value in payload["double_blocks"]):
        for kind, prefix in (("double_text", "double_text_biases"), ("double_image", "double_image_biases")):
            key = f"{prefix}.{block}"
            if key in state_dict:
                keys.append((kind, block, key))
    for block in sorted(int(value) for value in payload["single_blocks"]):
        key = f"single_biases.{block}"
        if key in state_dict:
            keys.append(("single", block, key))
    if not keys:
        raise ValueError("residual-bias payload contains no bias vectors")
    return keys


def build_variant_offsets(
    payload: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, dict[int, Any]], dict[str, Any]]:
    """Return ``{kind: {block: tensor}}`` offsets plus provenance metadata."""

    import torch

    keys = _bias_keys(payload)
    state_dict = payload["state_dict"]
    learned = {key: state_dict[key].detach().float().clone() for _, _, key in keys}

    variant_kind = "scale"
    if args.bias_random_seed is not None:
        variant_kind = "random"
        generator = torch.Generator(device="cpu").manual_seed(
            RANDOM_SEED_BASE + int(args.bias_random_seed)
        )
        for _, _, key in keys:
            reference = learned[key]
            norm = float(reference.norm())
            direction = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
            direction_norm = float(direction.norm())
            if direction_norm <= 0.0:
                raise RuntimeError(f"degenerate random direction for {key}")
            learned[key] = direction * (norm / direction_norm)
    elif args.bias_shuffle_seed is not None:
        variant_kind = "shuffle"
        generator = torch.Generator(device="cpu").manual_seed(
            SHUFFLE_SEED_BASE + int(args.bias_shuffle_seed)
        )
        for _, _, key in keys:
            reference = learned[key]
            permutation = torch.randperm(reference.numel(), generator=generator)
            learned[key] = reference[permutation].contiguous()

    alpha = float(args.bias_scale)
    offsets: dict[str, dict[int, Any]] = {"double_text": {}, "double_image": {}, "single": {}}
    norms: dict[str, float] = {}
    for kind, block, key in keys:
        tensor = (learned[key] * alpha).contiguous()
        offsets[kind][block] = tensor
        norms[key] = float(tensor.norm())

    if variant_kind == "scale":
        description = f"learned FLUX first-interval residual bias scaled by alpha={alpha}"
    elif variant_kind == "random":
        seed = RANDOM_SEED_BASE + int(args.bias_random_seed)
        description = (
            "gaussian random direction per bias vector with the learned per-vector L2 norm "
            f"(torch CPU generator seed {RANDOM_SEED_BASE}+{int(args.bias_random_seed)}={seed}), "
            f"alpha={alpha}"
        )
    else:
        seed = SHUFFLE_SEED_BASE + int(args.bias_shuffle_seed)
        description = (
            "channel permutation of each learned bias vector "
            f"(torch CPU generator seed {SHUFFLE_SEED_BASE}+{int(args.bias_shuffle_seed)}={seed}), "
            f"alpha={alpha}"
        )

    total = torch.cat([tensor.flatten() for tensor in learned.values()]) * alpha
    metadata = {
        "variant_kind": variant_kind,
        "variant_description": description,
        "bias_scale": alpha,
        "bias_random_seed": args.bias_random_seed,
        "bias_shuffle_seed": args.bias_shuffle_seed,
        "checkpoint_double_blocks": sorted(int(v) for v in payload["double_blocks"]),
        "checkpoint_single_blocks": sorted(int(v) for v in payload["single_blocks"]),
        "bias_vector_count": len(keys),
        "bias_l2_per_key": norms,
        "bias_rms": float(total.square().mean().sqrt()),
        "bias_abs_max": float(total.abs().max()),
        "rng_draw_order": (
            "double blocks ascending (text bias then image bias), then single blocks ascending"
        ),
    }
    return offsets, metadata


class FirstIntervalBiasOffset:
    """Add per-channel offsets to FLUX block outputs, gated by ``active``.

    Double-stream blocks return ``(encoder_hidden_states, hidden_states)`` and
    take two independent offsets; single-stream blocks return one tensor of
    concatenated ``[text ; image]`` tokens and take a single offset applied to
    every token.
    """

    def __init__(self, transformer: Any, offsets: dict[str, dict[int, Any]], *, device, dtype):
        import torch

        self.active = False
        self._handles: list[Any] = []
        self._torch = torch
        self._double_text: dict[int, Any] = {}
        self._double_image: dict[int, Any] = {}
        self._single: dict[int, Any] = {}

        double_blocks = sorted(set(offsets["double_text"]) | set(offsets["double_image"]))
        single_blocks = sorted(offsets["single"])
        if double_blocks and max(double_blocks) >= len(transformer.transformer_blocks):
            raise ValueError("double-block offset index is outside the transformer")
        if single_blocks and max(single_blocks) >= len(transformer.single_transformer_blocks):
            raise ValueError("single-block offset index is outside the transformer")

        for block_index in double_blocks:
            if block_index not in offsets["double_text"] or block_index not in offsets["double_image"]:
                raise ValueError(f"double block {block_index} needs both text and image offsets")
            self._double_text[block_index] = offsets["double_text"][block_index].to(
                device=device, dtype=dtype
            )
            self._double_image[block_index] = offsets["double_image"][block_index].to(
                device=device, dtype=dtype
            )
            block = transformer.transformer_blocks[block_index]
            self._handles.append(block.register_forward_hook(self._double_hook(block_index)))

        for block_index in single_blocks:
            self._single[block_index] = offsets["single"][block_index].to(device=device, dtype=dtype)
            block = transformer.single_transformer_blocks[block_index]
            self._handles.append(block.register_forward_hook(self._single_hook(block_index)))

    @staticmethod
    def _add(output, offset):
        if output is None:
            return None
        if output.shape[-1] != offset.numel():
            raise RuntimeError(
                f"offset width mismatch: output={output.shape[-1]} offset={offset.numel()}"
            )
        shape = (1,) * (output.ndim - 1) + (offset.numel(),)
        return output + offset.to(device=output.device, dtype=output.dtype).view(shape)

    def _double_hook(self, block_index: int):
        def hook(_module, _args, output):
            if not self.active:
                return output
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("unexpected FluxTransformerBlock output")
            encoder_output, hidden_output = output
            changed = (
                self._add(encoder_output, self._double_text[block_index]),
                self._add(hidden_output, self._double_image[block_index]),
            )
            return changed if isinstance(output, tuple) else list(changed)

        return hook

    def _single_hook(self, block_index: int):
        def hook(_module, _args, output):
            if not self.active:
                return output
            offset = self._single[block_index]
            if self._torch.is_tensor(output):
                return self._add(output, offset)
            if isinstance(output, (tuple, list)) and output and self._torch.is_tensor(output[0]):
                changed = (self._add(output[0], offset), *output[1:])
                return changed if isinstance(output, tuple) else list(changed)
            raise RuntimeError("unexpected FluxSingleTransformerBlock output")

        return hook

    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# finalize
# --------------------------------------------------------------------------- #
def finalize(args: argparse.Namespace) -> None:
    import torch

    from reseed.models.flux import flux_latent_shape

    rows: list[dict] = []
    protocols: list[dict] = []
    for shard in range(args.num_shards):
        path = args.output / f"generation_manifest.shard_{shard:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
        protocol_path = args.output / f"protocol.shard_{shard:02d}.json"
        protocols.append(json.loads(protocol_path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: int(row["case_number"]))
    expected = args.expected_count
    if len(rows) != expected or len({int(row["case_number"]) for row in rows}) != expected:
        raise RuntimeError(f"invalid finalized manifest: {len(rows)} rows, expected {expected}")
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise RuntimeError("shard generation protocols do not match")

    latent_shape = flux_latent_shape(args.resolution, args.resolution)
    for row in rows:
        if "latent_sha256" not in row:
            noise = torch.randn(
                latent_shape,
                generator=torch.Generator(device="cpu").manual_seed(int(row["evaluation_seed"])),
                dtype=torch.float32,
            )
            row["latent_sha256"] = hashlib.sha256(
                noise.contiguous().numpy().tobytes()
            ).hexdigest()

    protocol = protocols[0]
    protocol.update(
        {
            "schema_version": 1,
            "evaluation_protocol": args.evaluation_protocol,
            "mode": rows[0]["mode"],
            "num_prompts": expected,
            "resolution": args.resolution,
            "latent_shape": list(latent_shape),
            "latent_source": (
                "per-case CPU torch.Generator seeded by evaluation_seed; "
                "float32 hash before bf16 cast"
            ),
        }
    )
    write_jsonl(args.output / "generation_manifest.jsonl", rows)
    (args.output / "generation_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "COMPLETED").write_text(f"{expected} images\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# schedule cross-check against the real scheduler
# --------------------------------------------------------------------------- #
def verify_schedule_against_pipeline(
    model_path: Path, steps: int, image_seq_len: int, ours: list[float]
) -> list[float]:
    """Assert our sigma list equals ``scheduler.sigmas`` after ``set_timesteps``."""

    import numpy as np
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_path), subfolder="scheduler"
    )
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(sigmas=sigmas, mu=mu)
    reference = [float(value) for value in scheduler.sigmas]
    if len(reference) != len(ours) or any(
        abs(a - b) > 0.0 for a, b in zip(reference, ours)
    ):
        raise RuntimeError(
            "our FLUX sigma schedule does not match the diffusers scheduler:\n"
            f"  ours      = {ours}\n"
            f"  scheduler = {reference}"
        )
    return reference


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def generate(args: argparse.Namespace) -> None:
    import torch

    from reseed.models.flux import (
        decode_flux_latents,
        encode_flux_prompts,
        flux_euler_step,
        flux_image_seq_len,
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
    from reseed.models.flux_residual_bias import load_flux_residual_bias_payload

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_path = Path(args.model_path)
    transformer_path = Path(args.transformer_path)

    if args.model == "senseflow":
        # SenseFlow publishes only the distilled DiT.  Reuse the frozen
        # FLUX.1-dev text encoders, VAE and scheduler, exactly as instructed by
        # the release, but replace the transformer before moving the pipeline
        # to the GPU.  The released config pins guidance_embeds=false.
        pipe = load_flux_pipeline(model_path, torch_dtype=dtype, without_transformer=True)
        pipe.transformer = load_flux_transformer(transformer_path, torch_dtype=dtype)
    else:
        pipe = load_flux_pipeline(model_path, torch_dtype=dtype)
    pipe.to(device)
    transformer = pipe.transformer
    for component in list(pipe.components.values()):
        if isinstance(component, torch.nn.Module):
            component.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)

    vae_scale_factor = int(pipe.vae_scale_factor)
    image_seq_len = flux_image_seq_len(
        args.resolution, args.resolution, vae_scale_factor=vae_scale_factor
    )
    scheduler_config = load_flux_scheduler_config(model_path)
    schedule = flux_schedule_metadata(args.steps, scheduler_config, image_seq_len)
    boundaries = flux_sigma_boundaries(
        args.steps, scheduler_config, image_seq_len, device=device, dtype=torch.float32
    )
    sigma_list = [float(value) for value in boundaries]
    if not args.skip_schedule_check:
        verify_schedule_against_pipeline(model_path, args.steps, image_seq_len, sigma_list)
        schedule["verified_against_diffusers_scheduler"] = True
    else:
        schedule["verified_against_diffusers_scheduler"] = False

    guidance_embeds = bool(transformer.config.guidance_embeds)
    guidance = args.guidance
    if guidance_embeds and guidance is None:
        raise ValueError(f"--guidance is required for {args.model} (guidance_embeds=True)")
    if not guidance_embeds and guidance is not None:
        raise ValueError(f"--guidance must be omitted for {args.model} (guidance_embeds=False)")

    hooks = None
    variant: dict[str, Any] | None = None
    if args.first_interval_residual_bias_checkpoint is not None:
        payload = load_flux_residual_bias_payload(args.first_interval_residual_bias_checkpoint)
        offsets, variant = build_variant_offsets(payload, args)
        hooks = FirstIntervalBiasOffset(transformer, offsets, device=device, dtype=dtype)

    mode = MODE if hooks is not None else BASELINE_MODE
    records = read_records(args.prompts, args.expected_count, args.max_cases)
    shard_records = records[args.shard_id :: args.num_shards]

    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest_path = args.output / f"generation_manifest.shard_{args.shard_id:02d}.jsonl"

    manifest: list[dict] = []
    completed: dict[int, dict] = {}
    if args.resume:
        for row in read_jsonl(shard_manifest_path):
            if Path(row.get("image_path", "")).is_file():
                completed[int(row["case_number"])] = row
        print(
            f"shard {args.shard_id}: resuming with {len(completed)} completed cases",
            flush=True,
        )

    latent_shape = flux_latent_shape(
        args.resolution, args.resolution, vae_scale_factor=vae_scale_factor
    )
    latent_image_ids = prepare_flux_latent_image_ids(
        args.resolution,
        args.resolution,
        vae_scale_factor=vae_scale_factor,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for index, record in enumerate(shard_records, start=1):
            case = int(record["case_number"])
            seed = int(record["evaluation_seed"])
            if case in completed:
                manifest.append(completed[case])
                continue

            noise_fp32 = torch.randn(
                latent_shape,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                dtype=torch.float32,
            )
            latent_sha256 = hashlib.sha256(
                noise_fp32.contiguous().numpy().tobytes()
            ).hexdigest()
            latents = pack_flux_latents(noise_fp32[None].to(device=device, dtype=dtype))

            conditioning = encode_flux_prompts(
                pipe,
                [record["prompt"]],
                max_sequence_length=args.max_sequence_length,
                device=device,
            )

            for step, (sigma, next_sigma) in enumerate(zip(sigma_list[:-1], sigma_list[1:])):
                if hooks is not None:
                    if step == 0:
                        hooks.enable()
                    else:
                        hooks.disable()
                velocity = predict_flux_velocity(
                    transformer,
                    latents,
                    sigma,
                    conditioning,
                    guidance,
                    latent_image_ids=latent_image_ids,
                )
                latents = flux_euler_step(latents, velocity, sigma, next_sigma)
            if hooks is not None:
                hooks.disable()

            if not torch.isfinite(latents).all():
                raise RuntimeError(f"non-finite final latent for case {case}")
            decoded = decode_flux_latents(
                pipe.vae,
                latents,
                height=args.resolution,
                width=args.resolution,
                vae_scale_factor=vae_scale_factor,
            )
            image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            image_path = images_dir / f"case_{case:05d}_seed_{seed}.png"
            image.save(image_path)

            row = {
                **record,
                "case_number": case,
                "evaluation_seed": seed,
                "latent_sha256": latent_sha256,
                "image_path": str(image_path.resolve()),
                "mode": mode,
                "model_family": f"flux_1_{args.model}",
                "model_path": str(model_path.resolve()),
                "transformer_path": str(transformer_path.resolve()),
                "inference_steps": args.steps,
                "guidance_scale": guidance,
                "max_sequence_length": args.max_sequence_length,
            }
            if variant is not None:
                row.update(
                    {
                        "residual_bias_checkpoint": str(
                            Path(args.first_interval_residual_bias_checkpoint).resolve()
                        ),
                        "bias_enabled_steps": [0],
                        "bias_disabled_steps": list(range(1, args.steps)),
                        "variant_kind": variant["variant_kind"],
                        "variant_description": variant["variant_description"],
                        "bias_scale": variant["bias_scale"],
                    }
                )
            manifest.append(row)
            write_jsonl(shard_manifest_path, manifest)
            print(
                f"shard {args.shard_id}: {index}/{len(shard_records)} case {case}", flush=True
            )

    if hooks is not None:
        hooks.close()
    write_jsonl(shard_manifest_path, manifest)

    protocol = {
        "inference_only": True,
        "schema_version": 1,
        "evaluation_protocol": args.evaluation_protocol,
        "mode": mode,
        "model": args.model,
        "model_family": f"flux_1_{args.model}",
        "model_path": str(model_path.resolve()),
        "transformer_path": str(transformer_path.resolve()),
        "guidance_scale": guidance,
        "guidance_embeds": guidance_embeds,
        "max_sequence_length": args.max_sequence_length,
        "schedule": schedule,
        "sigmas": sigma_list,
        "num_prompts": args.expected_count,
        "expected_count": args.expected_count,
        "max_cases": args.max_cases,
        "samples_per_prompt": args.samples_per_prompt,
        "resolution": args.resolution,
        "vae_scale_factor": vae_scale_factor,
        "latent_shape": list(latent_shape),
        "packed_tokens": image_seq_len,
        "latent_source": (
            "per-case CPU torch.Generator seeded by evaluation_seed; "
            "float32 hash before bf16 cast"
        ),
        "prompt_csv": str(Path(args.prompts).resolve()),
        "dtype": "bf16",
        "batch_size": 1,
        "sampler": "hand-written Euler loop reproducing FluxPipeline.__call__",
    }
    if variant is not None:
        protocol.update(
            {
                "residual_bias_checkpoint": str(
                    Path(args.first_interval_residual_bias_checkpoint).resolve()
                ),
                "bias_enabled_steps": [0],
                "bias_disabled_steps": list(range(1, args.steps)),
                "variant": variant,
            }
        )
    (args.output / f"protocol.shard_{args.shard_id:02d}.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=("dev", "schnell", "senseflow"), required=True)
    result.add_argument("--dev-model", type=Path, default=Path(DEFAULT_MODEL_PATHS["dev"]))
    result.add_argument(
        "--schnell-model", type=Path, default=Path(DEFAULT_MODEL_PATHS["schnell"])
    )
    result.add_argument(
        "--senseflow-transformer",
        type=Path,
        default=Path(DEFAULT_MODEL_PATHS["senseflow"]),
        help="SenseFlow-FLUX transformer directory; shared components come from --dev-model",
    )
    result.add_argument("--steps", type=int, default=None, help="default: 28 dev, 4 schnell")
    result.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="distilled guidance embedding value; default 3.5 for dev, unset for schnell",
    )
    result.add_argument(
        "--no-guidance",
        action="store_true",
        help="force guidance to None even for a model whose default supplies one",
    )
    result.add_argument("--max-sequence-length", type=int, default=None)
    result.add_argument("--prompts", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--expected-count", type=int, required=True)
    result.add_argument("--evaluation-protocol", required=True)
    result.add_argument("--samples-per-prompt", type=int, default=10)
    result.add_argument("--resolution", type=int, default=1024)
    result.add_argument("--max-cases", type=int, default=None)
    result.add_argument(
        "--first-interval-residual-bias-checkpoint",
        type=Path,
        default=None,
        help="directory holding residual_biases.pt; the bias is applied on step 0 only",
    )
    result.add_argument("--bias-scale", type=float, default=1.0)
    result.add_argument("--bias-random-seed", type=int, default=None)
    result.add_argument("--bias-shuffle-seed", type=int, default=None)
    result.add_argument("--num-shards", type=int, default=1)
    result.add_argument("--shard-id", type=int, default=0)
    result.add_argument("--finalize-shards", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument(
        "--skip-schedule-check",
        action="store_true",
        help="skip building the reference scheduler to cross-check our sigma list",
    )
    return result


def resolve_defaults(args: argparse.Namespace) -> None:
    if args.model == "dev":
        args.model_path = args.dev_model
        args.transformer_path = args.dev_model / "transformer"
    elif args.model == "schnell":
        args.model_path = args.schnell_model
        args.transformer_path = args.schnell_model / "transformer"
    else:
        args.model_path = args.dev_model
        args.transformer_path = args.senseflow_transformer
    if args.steps is None:
        args.steps = DEFAULT_STEPS[args.model]
    if args.max_sequence_length is None:
        args.max_sequence_length = DEFAULT_MAX_SEQUENCE_LENGTH[args.model]
    if args.no_guidance:
        args.guidance = None
    elif args.guidance is None:
        args.guidance = DEFAULT_GUIDANCE[args.model]


def main() -> int:
    argument_parser = parser()
    args = argument_parser.parse_args()
    resolve_defaults(args)
    if args.bias_random_seed is not None and args.bias_shuffle_seed is not None:
        argument_parser.error("--bias-random-seed and --bias-shuffle-seed are mutually exclusive")
    if (
        args.first_interval_residual_bias_checkpoint is None
        and (args.bias_random_seed is not None or args.bias_shuffle_seed is not None)
    ):
        argument_parser.error(
            "bias variant options require --first-interval-residual-bias-checkpoint"
        )
    if args.samples_per_prompt < 1:
        argument_parser.error("--samples-per-prompt must be at least 1")
    if args.steps < 1:
        argument_parser.error("--steps must be at least 1")
    if args.max_cases is not None:
        if args.max_cases <= 0 or args.max_cases % args.samples_per_prompt != 0:
            argument_parser.error("--max-cases must be a positive multiple of --samples-per-prompt")
        if args.max_cases != args.expected_count:
            argument_parser.error("--expected-count must equal --max-cases when it is set")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        argument_parser.error("invalid --num-shards/--shard-id combination")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.finalize_shards:
        finalize(args)
    else:
        generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
