#!/usr/bin/env python3
"""Generate multi-seed PixArt samples and measure DINO/CLIP feature diversity.

The reported per-prompt metric is

    1 - 2 / (R * (R - 1)) * sum_{i < j} cosine(feature_i, feature_j).

`run-pixart` prepares an auditable latent bank, delegates image generation to
the repository's existing paired PixArt generator, and then scores the output.
`score` can score any compatible generation manifest without regenerating it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DINO_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"
LATENT_PROTOCOL = "pixart_feature_diversity_latents_v1"
EVALUATION_PROTOCOL = "pixart_feature_diversity_multiseed_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_prompts(path: Path, limit: int) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows: Iterable[Any] = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        rows = _jsonl(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("prompts", [])
    else:
        rows = [{"prompt": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = list(rows)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    result = []
    for prompt_id, row in enumerate(rows):
        prompt = str(row.get("prompt", row.get("caption", ""))).strip()
        if not prompt:
            raise ValueError(f"Empty prompt at input row {prompt_id}")
        result.append({"prompt_id": prompt_id, "prompt": prompt})
    return result


def build_latent_bank(
    prompt_rows: list[dict[str, Any]],
    *,
    samples_per_prompt: int,
    base_seed: int,
    latent_size: int,
    prompt_path: Path,
    output: Path,
    initial_noise_mode: str = "gaussian",
    lowpass_cutoff: float = 0.20,
    latent_protocol: str = LATENT_PROTOCOL,
) -> dict[str, Any]:
    import torch

    from reseed.diffusion.frequency_noise import lowpass_match_sample_rms

    if samples_per_prompt < 2:
        raise ValueError("Diversity requires at least two samples per prompt")
    if initial_noise_mode not in ("gaussian", "lowpass"):
        raise ValueError(f"unsupported initial noise mode: {initial_noise_mode}")
    if not 0.0 < lowpass_cutoff <= 1.0:
        raise ValueError("lowpass cutoff must be in (0, 1]")
    records, latents = [], []
    for prompt_row in prompt_rows:
        prompt_id = int(prompt_row["prompt_id"])
        prompt_hashes: set[str] = set()
        for sample_id in range(samples_per_prompt):
            # Every particle gets an independent CPU RNG stream. The recipe is
            # deterministic and is shared across models through this bank.
            seed = base_seed + prompt_id * samples_per_prompt + sample_id
            generator = torch.Generator(device="cpu").manual_seed(seed)
            latent = torch.randn((4, latent_size, latent_size), generator=generator).to(torch.float16)
            if initial_noise_mode == "lowpass":
                latent, _ = lowpass_match_sample_rms(
                    latent.unsqueeze(0), cutoff=lowpass_cutoff
                )
                latent = latent.squeeze(0).contiguous()
            latent_hash = hashlib.sha256(latent.contiguous().numpy().tobytes()).hexdigest()
            if latent_hash in prompt_hashes:
                raise AssertionError(f"Duplicate initial noise for prompt {prompt_id}")
            prompt_hashes.add(latent_hash)
            case_number = prompt_id * samples_per_prompt + sample_id
            records.append(
                {
                    "case_number": case_number,
                    "source": "feature-diversity",
                    "prompt": prompt_row["prompt"],
                    "evaluation_seed": seed,
                    "coco_id": prompt_id,
                    "prompt_id": prompt_id,
                    "sample_id": sample_id,
                    "latent_sha256": latent_hash,
                }
            )
            latents.append(latent)
    tensor = torch.stack(latents)
    expected = (len(prompt_rows) * samples_per_prompt, 4, latent_size, latent_size)
    if tuple(tensor.shape) != expected:
        raise AssertionError(f"Unexpected latent bank shape {tuple(tensor.shape)} != {expected}")
    payload = {
        "protocol": latent_protocol,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": sha256_file(prompt_path),
        "prompt_count": len(prompt_rows),
        "seeds_per_prompt": samples_per_prompt,
        "base_seed": base_seed,
        "seed_recipe": "base_seed + prompt_id * samples_per_prompt + sample_id",
        "initial_noise_mode": initial_noise_mode,
        "initial_noise_lowpass_cutoff": (
            lowpass_cutoff if initial_noise_mode == "lowpass" else None
        ),
        "records": records,
        "latents": tensor,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite latent bank: {output}")
    torch.save(payload, output)
    return payload


def diversity_from_features(features: Any) -> tuple[float, float, float]:
    """Return diversity, mean pairwise cosine, and pairwise cosine std."""
    if hasattr(features, "detach"):
        import torch
        import torch.nn.functional as F

        tensor = features.detach().float()
        if tensor.ndim != 2 or tensor.shape[0] < 2:
            raise ValueError(f"Expected [R, D] with R >= 2, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("Feature tensor contains NaN/Inf")
        if torch.any(torch.linalg.vector_norm(tensor, dim=1) <= 0):
            raise ValueError("Feature tensor contains a zero-norm row")
        tensor = F.normalize(tensor, dim=1)
        indexes = torch.triu_indices(tensor.shape[0], tensor.shape[0], offset=1, device=tensor.device)
        pairs = (tensor @ tensor.T)[indexes[0], indexes[1]]
        mean = float(pairs.mean())
        std = float(pairs.std(unbiased=True)) if pairs.numel() > 1 else 0.0
        return 1.0 - mean, mean, std

    rows = features
    rows = [[float(value) for value in row] for row in rows]
    if len(rows) < 2 or not rows or not rows[0]:
        raise ValueError("Expected [R, D] with R >= 2")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Feature rows have inconsistent dimensions")
    normalized = []
    for row in rows:
        if not all(math.isfinite(value) for value in row):
            raise ValueError("Feature tensor contains NaN/Inf")
        norm = math.sqrt(sum(value * value for value in row))
        if norm <= 0:
            raise ValueError("Feature tensor contains a zero-norm row")
        normalized.append([value / norm for value in row])
    pairs = [
        sum(left * right for left, right in zip(normalized[i], normalized[j]))
        for i in range(len(normalized))
        for j in range(i + 1, len(normalized))
    ]
    mean = sum(pairs) / len(pairs)
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in pairs) / (len(pairs) - 1))
        if len(pairs) > 1
        else 0.0
    )
    return 1.0 - mean, mean, std


def _resolve_dtype(name: str):
    import torch

    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _encoder_dtype(encoder: str, *, shared_dtype: str, dino_dtype: str | None, clip_dtype: str | None) -> str:
    """Resolve an encoder-specific inference dtype.

    DINOv3 ViT-L can produce non-finite pooled features in native fp16 on some
    accelerators.  Its default is therefore fp32, while CLIP retains the shared
    dtype for backward compatibility.  Explicit per-encoder values always win.
    """
    if encoder == "dino":
        return dino_dtype or "fp32"
    if encoder == "clip":
        return clip_dtype or shared_dtype
    raise ValueError(f"Unknown feature encoder: {encoder}")


def _validate_feature_shape(shape: tuple[int, ...], *, expected_rows: int, encoder: str) -> None:
    """Validate the feature matrix shape independently of numerical validity."""
    if len(shape) != 2:
        raise RuntimeError(f"Invalid {encoder} feature rank: expected [N, D], got {shape}")
    if shape[0] != expected_rows:
        raise RuntimeError(
            f"Invalid {encoder} feature row count: expected {expected_rows}, got {shape[0]} (shape={shape})"
        )
    if shape[1] < 1:
        raise RuntimeError(f"Invalid {encoder} feature dimension: {shape}")


def _encode_images(
    records: list[dict[str, Any]],
    *,
    encoder: str,
    model_id: str,
    device: str,
    dtype: str,
    batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    model_dtype = _resolve_dtype(dtype)
    if resolved_device.type == "cpu" and model_dtype == torch.float16:
        raise ValueError("fp16 feature inference on CPU is unsupported; use fp32 or bf16")
    processor = AutoImageProcessor.from_pretrained(model_id)
    if encoder == "clip":
        model = CLIPVisionModelWithProjection.from_pretrained(model_id, torch_dtype=model_dtype)
    else:
        model = AutoModel.from_pretrained(model_id, torch_dtype=model_dtype)
    model.eval().requires_grad_(False).to(resolved_device)
    config_json = json.dumps(model.config.to_dict(), sort_keys=True, separators=(",", ":"))
    metadata = {
        "model_id": model_id,
        "model_class": model.__class__.__name__,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "commit_hash": getattr(model.config, "_commit_hash", None),
    }
    chunks = []
    with torch.inference_mode():
        for first in range(0, len(records), batch_size):
            batch = records[first : first + batch_size]
            images = []
            for record in batch:
                path = Path(record["image_path"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=resolved_device, dtype=model_dtype)
            output = model(pixel_values=pixel_values)
            if encoder == "clip":
                feature = output.image_embeds
            elif getattr(output, "pooler_output", None) is not None:
                feature = output.pooler_output
            else:
                feature = output.last_hidden_state[:, 0]
            chunks.append(F.normalize(feature.float(), dim=1).cpu())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = torch.cat(chunks, dim=0)
    _validate_feature_shape(tuple(result.shape), expected_rows=len(records), encoder=encoder)
    finite = torch.isfinite(result)
    if not finite.all():
        nonfinite = int((~finite).sum().item())
        raise RuntimeError(
            f"Non-finite {encoder} feature values: {nonfinite}/{result.numel()} "
            f"(shape={tuple(result.shape)}, inference_dtype={dtype}). "
            "Use fp32 (recommended) or bf16 for this encoder."
        )
    metadata["feature_dimension"] = int(result.shape[1])
    return result, metadata


def _load_and_validate_manifest(path: Path, expected_samples: int) -> list[dict[str, Any]]:
    records = _jsonl(path)
    if not records:
        raise ValueError("Generation manifest is empty")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for key in ("prompt_id", "sample_id", "prompt", "evaluation_seed", "latent_sha256", "image_path"):
            if key not in record:
                raise ValueError(f"Manifest record missing {key!r}")
        grouped[int(record["prompt_id"])].append(record)
    expected_ids = list(range(len(grouped)))
    if sorted(grouped) != expected_ids:
        raise ValueError(f"Prompt ids must be contiguous: {sorted(grouped)}")
    ordered = []
    for prompt_id in expected_ids:
        group = sorted(grouped[prompt_id], key=lambda row: int(row["sample_id"]))
        if len(group) != expected_samples:
            raise ValueError(f"Prompt {prompt_id}: expected {expected_samples} samples, found {len(group)}")
        if len({int(row["evaluation_seed"]) for row in group}) != expected_samples:
            raise ValueError(f"Prompt {prompt_id}: evaluation seeds are not independent")
        if len({row["latent_sha256"] for row in group}) != expected_samples:
            raise ValueError(f"Prompt {prompt_id}: initial-noise latents are not independent")
        if len({row["prompt"] for row in group}) != 1:
            raise ValueError(f"Prompt {prompt_id}: prompt text differs within the group")
        ordered.extend(group)
    return ordered


def _image_set_sha256(records: list[dict[str, Any]]) -> str:
    """Bind a metric run to the exact ordered image bytes it consumed."""
    digest = hashlib.sha256()
    for record in records:
        path = Path(record["image_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(record["prompt_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sample_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _mean_std_stderr(values: list[float]) -> dict[str, float | int]:
    mean = sum(values) / len(values)
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "std": std,
        "stderr": std / math.sqrt(len(values)),
        "n_prompts": len(values),
    }


def score_manifest(args: argparse.Namespace) -> Path:
    initial_manifest_sha256 = sha256_file(args.manifest)
    records = _load_and_validate_manifest(args.manifest, args.samples_per_prompt)
    initial_image_set_sha256 = _image_set_sha256(records)
    feature_sets = {}
    model_metadata = {}
    for encoder, model_id in (("dino", args.dino_model), ("clip", args.clip_model)):
        encoder_dtype = _encoder_dtype(
            encoder,
            shared_dtype=args.dtype,
            dino_dtype=getattr(args, "dino_dtype", None),
            clip_dtype=getattr(args, "clip_dtype", None),
        )
        features, metadata = _encode_images(
            records,
            encoder=encoder,
            model_id=model_id,
            device=args.device,
            dtype=encoder_dtype,
            batch_size=args.feature_batch_size,
        )
        feature_sets[encoder] = features
        metadata["inference_dtype"] = encoder_dtype
        model_metadata[encoder] = metadata
    final_image_set_sha256 = _image_set_sha256(records)
    if final_image_set_sha256 != initial_image_set_sha256:
        raise RuntimeError("Generated images changed while feature scoring was running")
    if sha256_file(args.manifest) != initial_manifest_sha256:
        raise RuntimeError("Generation manifest changed while feature scoring was running")
    prompt_rows = []
    for first in range(0, len(records), args.samples_per_prompt):
        group = records[first : first + args.samples_per_prompt]
        row: dict[str, Any] = {
            "prompt_id": int(group[0]["prompt_id"]),
            "prompt": group[0]["prompt"],
            "samples": args.samples_per_prompt,
            "pair_count": args.samples_per_prompt * (args.samples_per_prompt - 1) // 2,
        }
        for encoder, features in feature_sets.items():
            diversity, cosine_mean, cosine_std = diversity_from_features(
                features[first : first + args.samples_per_prompt]
            )
            row[f"{encoder}_diversity"] = diversity
            row[f"{encoder}_pairwise_cosine_mean"] = cosine_mean
            row[f"{encoder}_pairwise_cosine_std"] = cosine_std
        prompt_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "diversity_per_prompt.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prompt_rows[0]))
        writer.writeheader()
        writer.writerows(prompt_rows)
    summary = {
        "schema_version": 1,
        "definition": "1 - 2/(R*(R-1)) * sum_{i<j} cosine(feature_i, feature_j)",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": initial_manifest_sha256,
        "image_count": len(records),
        "ordered_image_set_sha256": initial_image_set_sha256,
        "samples_per_prompt": args.samples_per_prompt,
        "pair_count_per_prompt": args.samples_per_prompt * (args.samples_per_prompt - 1) // 2,
        "models": model_metadata,
        "inference": {
            "device": args.device,
            "shared_dtype": args.dtype,
            "dino_dtype": _encoder_dtype(
                "dino",
                shared_dtype=args.dtype,
                dino_dtype=getattr(args, "dino_dtype", None),
                clip_dtype=getattr(args, "clip_dtype", None),
            ),
            "clip_dtype": _encoder_dtype(
                "clip",
                shared_dtype=args.dtype,
                dino_dtype=getattr(args, "dino_dtype", None),
                clip_dtype=getattr(args, "clip_dtype", None),
            ),
            "batch_size": args.feature_batch_size,
        },
        "metrics": {
            encoder: _mean_std_stderr([float(row[f"{encoder}_diversity"]) for row in prompt_rows])
            for encoder in ("dino", "clip")
        },
        "per_prompt_csv": str(csv_path.resolve()),
        "per_prompt_csv_sha256": sha256_file(csv_path),
    }
    json_path = args.output_dir / "diversity_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return json_path


def _run_pixart(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output / "latent_bank.pt"
    generation_dir = output / "generation"
    prompts = read_prompts(args.prompts, args.prompt_count)
    if not bank_path.exists():
        build_latent_bank(
            prompts,
            samples_per_prompt=args.samples_per_prompt,
            base_seed=args.base_seed,
            latent_size=args.resolution // 8,
            prompt_path=args.prompts,
            output=bank_path,
        )
    else:
        import torch

        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        if bank.get("prompt_file_sha256") != sha256_file(args.prompts):
            raise ValueError("Existing latent bank belongs to a different prompt file")
        if int(bank.get("prompt_count", -1)) != len(prompts):
            raise ValueError("Existing latent bank uses a different prompt count")
        if int(bank.get("seeds_per_prompt", -1)) != args.samples_per_prompt:
            raise ValueError("Existing latent bank uses a different samples-per-prompt value")
        if int(bank.get("base_seed", -1)) != args.base_seed:
            raise ValueError("Existing latent bank uses a different base seed")
        expected_shape = (len(prompts) * args.samples_per_prompt, 4, args.resolution // 8, args.resolution // 8)
        if tuple(bank["latents"].shape) != expected_shape:
            raise ValueError(f"Existing latent bank has wrong shape: {tuple(bank['latents'].shape)}")
    import torch

    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    expected_count = len(bank["records"])
    generator = project_root / "legacy" / "gad" / "generate_coco1k_benchmark.py"
    command = [
        sys.executable,
        str(generator),
        "--mode",
        args.mode,
        "--model",
        args.model,
        "--latent-bank",
        str(bank_path),
        "--output",
        str(generation_dir),
        "--batch-size",
        str(args.generation_batch_size),
        "--num-gpus",
        str(args.num_gpus),
        "--teacher-steps",
        str(args.teacher_steps),
        "--guidance",
        str(args.guidance),
        "--gad-steps",
        str(args.student_steps),
        "--total-steps",
        str(args.total_steps),
        "--expected-count",
        str(expected_count),
        "--resolution",
        str(args.resolution),
        "--latent-protocol",
        LATENT_PROTOCOL,
        "--evaluation-protocol",
        EVALUATION_PROTOCOL,
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.vae:
        command.extend(["--vae", args.vae])
    if args.resume:
        command.append("--resume")
    subprocess.run(command, check=True, cwd=project_root)
    score_args = argparse.Namespace(
        manifest=generation_dir / "generation_manifest.jsonl",
        output_dir=output / "metrics",
        samples_per_prompt=args.samples_per_prompt,
        dino_model=args.dino_model,
        clip_model=args.clip_model,
        device=args.device,
        dtype=args.dtype,
        feature_batch_size=args.feature_batch_size,
    )
    score_manifest(score_args)


def _add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples-per-prompt", type=int, default=9)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:1")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--dino-dtype",
        choices=("fp32", "fp16", "bf16"),
        default=None,
        help="DINOv3 inference dtype (default: fp32; avoids non-finite ViT-L fp16 features)",
    )
    parser.add_argument(
        "--clip-dtype",
        choices=("fp32", "fp16", "bf16"),
        default=None,
        help="CLIP inference dtype (default: inherit --dtype)",
    )
    parser.add_argument("--feature-batch-size", type=int, default=16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="Score an existing compatible generation manifest")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    _add_score_arguments(score)

    run = subparsers.add_parser("run-pixart", help="Generate PixArt samples and score their diversity")
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--mode", choices=("pixart", "tdm", "gad", "crra"), required=True)
    run.add_argument("--checkpoint")
    run.add_argument("--vae")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--prompt-count", type=int, default=10, help="0 uses every prompt")
    run.add_argument("--base-seed", type=int, default=2026081700)
    run.add_argument("--resolution", type=int, default=512)
    run.add_argument("--teacher-steps", type=int, default=20)
    run.add_argument("--guidance", type=float, default=4.5)
    run.add_argument("--student-steps", type=int, default=4)
    run.add_argument("--total-steps", type=int, default=900)
    run.add_argument("--generation-batch-size", type=int, default=4)
    run.add_argument("--num-gpus", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    _add_score_arguments(run)
    args = parser.parse_args()
    if args.samples_per_prompt < 2:
        parser.error("--samples-per-prompt must be at least 2")
    if args.feature_batch_size < 1:
        parser.error("--feature-batch-size must be positive")
    if args.command == "run-pixart":
        if args.resolution < 8 or args.resolution % 8:
            parser.error("--resolution must be a positive multiple of 8")
        if args.mode != "pixart" and (not args.checkpoint or not args.vae):
            parser.error("Distilled PixArt modes require --checkpoint and --vae")
        _run_pixart(args)
    else:
        score_manifest(args)


if __name__ == "__main__":
    main()
