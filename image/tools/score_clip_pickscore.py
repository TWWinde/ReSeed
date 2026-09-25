#!/usr/bin/env python
"""Score a generated COCO image set with OpenCLIP and PickScore.

The script writes one row per prompt before computing aggregate statistics, so
the final PixArt-vs-GAD comparison can be paired by the benchmark case id.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


_DEFAULT_PICKSCORE_FALLBACK = (
    "${HF_HOME:-~/.cache/huggingface}/hub/models--yuvalkirstain--PickScore_v1/"
    "snapshots/a4e4367c6dfa7288a00c550414478f865b875800"
)
DEFAULT_PICKSCORE = os.environ.get("RESEED_PICKSCORE_MODEL", _DEFAULT_PICKSCORE_FALLBACK)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, expected_count: int):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} generated records, found {len(records)}")
    case_numbers = [int(record["case_number"]) for record in records]
    if len(set(case_numbers)) != expected_count:
        raise ValueError("Generation manifest case_number values are not unique")
    missing = [record["image_path"] for record in records if not Path(record["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} generated image(s), first: {missing[:3]}")
    for record in records:
        image_path = Path(record["image_path"])
        expected_sha = record.get("image_sha256")
        if expected_sha is not None and sha256_file(image_path) != expected_sha:
            raise ValueError(f"Generated image SHA256 mismatch: {image_path}")
        expected_width = record.get("image_width")
        expected_height = record.get("image_height")
        if (expected_width is None) != (expected_height is None):
            raise ValueError(f"Generated image manifest has incomplete dimensions: {image_path}")
        if expected_width is not None:
            with Image.open(image_path) as image:
                if image.size != (int(expected_width), int(expected_height)):
                    raise ValueError(f"Generated image dimensions mismatch: {image_path}")
    return records


def load_generation_provenance(
    manifest: Path, records, expected_count: int, evaluation_protocol: str
):
    protocol_path = manifest.parent / "generation_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing generation protocol beside manifest: {protocol_path}")
    protocol_raw = protocol_path.read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    protocol = json.loads(protocol_raw)
    if protocol.get("evaluation_protocol") != evaluation_protocol:
        raise ValueError(
            "Generation/scoring protocol mismatch: "
            f"{protocol.get('evaluation_protocol')!r} != {evaluation_protocol!r}"
        )
    if int(protocol.get("num_prompts", -1)) != expected_count:
        raise ValueError("Generation protocol prompt count does not match --expected-count")
    modes = {record.get("mode") for record in records}
    if modes != {protocol.get("mode")}:
        raise ValueError(f"Manifest modes {modes} do not match generation protocol")

    request_path = manifest.parent / "generation_request.json"
    request_sha256 = None
    declared_request_sha256 = protocol.get("generation_request_sha256")
    if declared_request_sha256 is not None and not request_path.is_file():
        raise FileNotFoundError(
            f"Generation protocol declares a request SHA but the request is missing: {request_path}"
        )
    if request_path.is_file():
        request_raw = request_path.read_bytes()
        request = json.loads(request_raw)
        for key in (
            "evaluation_protocol",
            "mode",
            "num_prompts",
            "benchmark_num_prompts",
            "model",
            "vae",
            "checkpoint",
            "student",
            "latent_bank",
            "latent_bank_sha256",
            "latent_bank_identity",
            "student_steps",
            "student_total_steps",
        ):
            if request.get(key) != protocol.get(key):
                raise ValueError(f"Generation request/protocol mismatch for {key}")
        request_sha256 = hashlib.sha256(request_raw).hexdigest()
        if declared_request_sha256 != request_sha256:
            raise ValueError("Generation request SHA does not match generation protocol")
    return {
        "manifest_sha256": sha256_file(manifest),
        "path": str(protocol_path.resolve()),
        "sha256": protocol_sha256,
        "request_path": str(request_path.resolve()) if request_path.is_file() else None,
        "request_sha256": request_sha256,
        "payload": protocol,
    }


def batch_ranges(count, batch_size):
    for start in range(0, count, batch_size):
        yield start, min(count, start + batch_size)


def load_images(records):
    images = []
    for record in records:
        # convert() loads the image data, so it is safe to close the file handle.
        with Image.open(record["image_path"]) as image:
            images.append(image.convert("RGB"))
    return images


def assert_generation_unchanged(manifest: Path, records, generation, expected_count: int):
    """Fail closed if generation inputs change while metrics are running."""

    if sha256_file(manifest) != generation["manifest_sha256"]:
        raise ValueError("Generation manifest changed while metrics were running")
    if sha256_file(Path(generation["path"])) != generation["sha256"]:
        raise ValueError("Generation protocol changed while metrics were running")
    if generation["request_path"] is not None:
        if sha256_file(Path(generation["request_path"])) != generation["request_sha256"]:
            raise ValueError("Generation request changed while metrics were running")
    if load_manifest(manifest, expected_count) != records:
        raise ValueError("Generation records changed while metrics were running")


@torch.inference_mode()
def score_clip(records, images, device, model_name, pretrained, batch_size):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    scores = np.empty(len(records), dtype=np.float64)
    for start, end in batch_ranges(len(records), batch_size):
        image_batch = torch.stack([preprocess(image) for image in images[start:end]]).to(device)
        tokens = tokenizer([record["prompt"] for record in records[start:end]]).to(device)
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores[start:end] = (image_features * text_features).sum(dim=-1).float().cpu().numpy()
        print(f"[clip] {end}/{len(records)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return scores


def unwrap_features(outputs):
    # transformers <5 returned a tensor; transformers 5 returns a model output.
    return outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs


@torch.inference_mode()
def score_pickscore(records, images, device, model_path, batch_size):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).eval().to(device)
    scores = np.empty(len(records), dtype=np.float64)
    for start, end in batch_ranges(len(records), batch_size):
        image_inputs = processor(images=images[start:end], return_tensors="pt")
        image_values = image_inputs["pixel_values"].to(device)
        text_inputs = processor(
            text=[record["prompt"] for record in records[start:end]],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        image_embeddings = unwrap_features(model.get_image_features(pixel_values=image_values))
        text_embeddings = unwrap_features(model.get_text_features(**text_inputs))
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        scores[start:end] = (
            model.logit_scale.exp() * (image_embeddings * text_embeddings).sum(dim=-1)
        ).float().cpu().numpy()
        print(f"[pickscore] {end}/{len(records)}", flush=True)
    del model, processor
    torch.cuda.empty_cache()
    return scores


def metric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "stderr": float(values.std(ddof=1) / np.sqrt(values.size)),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(values.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--pickscore-model", default=DEFAULT_PICKSCORE)
    parser.add_argument("--pickscore-batch-size", type=int, default=8)
    parser.add_argument("--evaluation-protocol", default="coco1k_paired_v1")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=1000,
        help="Use a smaller explicit value only for a smoke test; formal evaluation uses 1,000.",
    )
    args = parser.parse_args()
    if args.clip_batch_size < 1 or args.pickscore_batch_size < 1:
        raise ValueError("Metric batch sizes must be positive")

    manifest_sha256_before_load = sha256_file(args.manifest)
    records = load_manifest(args.manifest, args.expected_count)
    generation = load_generation_provenance(
        args.manifest, records, args.expected_count, args.evaluation_protocol
    )
    if generation["manifest_sha256"] != manifest_sha256_before_load:
        raise ValueError("Generation manifest changed while it was being loaded")
    images = load_images(records)
    device = torch.device("cuda")
    clip_cosines = score_clip(
        records,
        images,
        device,
        args.clip_model,
        args.clip_pretrained,
        args.clip_batch_size,
    )
    pickscore_logits = score_pickscore(
        records, images, device, args.pickscore_model, args.pickscore_batch_size
    )
    assert_generation_unchanged(args.manifest, records, generation, args.expected_count)

    args.output.mkdir(parents=True, exist_ok=True)
    score_path = args.output / "scores.csv"
    temporary_score_path = score_path.with_suffix(".csv.tmp")
    columns = [
        "case_number",
        "coco_id",
        "evaluation_seed",
        "source",
        "prompt",
        "latent_sha256",
        "image_path",
        "mode",
        "clip_cosine",
        "clip_score_x100",
        "pickscore_logit",
    ]
    with temporary_score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record, clip_cosine, pickscore_logit in zip(records, clip_cosines, pickscore_logits):
            writer.writerow(
                {
                    **{column: record.get(column, "") for column in columns},
                    "clip_cosine": f"{clip_cosine:.10f}",
                    "clip_score_x100": f"{clip_cosine * 100:.10f}",
                    "pickscore_logit": f"{pickscore_logit:.10f}",
                }
            )
    temporary_score_path.replace(score_path)
    report = {
        "evaluation_protocol": args.evaluation_protocol,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": generation["manifest_sha256"],
        "generation_protocol": generation["path"],
        "generation_protocol_sha256": generation["sha256"],
        "generation_request": generation["request_path"],
        "generation_request_sha256": generation["request_sha256"],
        "scores_csv": str(score_path),
        "scores_csv_sha256": sha256_file(score_path),
        "n": len(records),
        "clip": {
            "model": args.clip_model,
            "pretrained": args.clip_pretrained,
            "open_clip_version": importlib.metadata.version("open-clip-torch"),
            "clip_cosine": metric_summary(clip_cosines),
            "clip_score_x100": metric_summary(clip_cosines * 100),
        },
        "pickscore": {
            "model": args.pickscore_model,
            "definition": "exp(logit_scale) * cosine(text_embedding, image_embedding)",
            "pickscore_logit": metric_summary(pickscore_logits),
        },
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
    }
    report_path = args.output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
