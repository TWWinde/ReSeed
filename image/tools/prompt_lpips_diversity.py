#!/usr/bin/env python3
"""Compute prompt-wise LPIPS diversity from a multi-seed generation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_groups(manifest: Path, samples_per_prompt: int):
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        for field in ("prompt_id", "sample_id", "image_path"):
            if field not in row:
                raise ValueError(f"manifest row is missing {field}")
        groups.setdefault(str(row["prompt_id"]), []).append(row)
    if not groups:
        raise ValueError("manifest is empty")
    for prompt_id, prompt_rows in groups.items():
        if len(prompt_rows) != samples_per_prompt:
            raise ValueError(
                f"prompt {prompt_id} has {len(prompt_rows)} samples, expected {samples_per_prompt}"
            )
        sample_ids = {int(row["sample_id"]) for row in prompt_rows}
        if len(sample_ids) != samples_per_prompt:
            raise ValueError(f"prompt {prompt_id} has duplicate sample ids")
        prompt_rows.sort(key=lambda row: int(row["sample_id"]))
    return groups


def summarize(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "stderr": std / math.sqrt(len(values)),
        "n_prompts": len(values),
    }


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from PIL import Image
    import lpips

    initial_manifest_sha = sha256_file(args.manifest)
    groups = load_groups(args.manifest, args.samples_per_prompt)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = lpips.LPIPS(net=args.net).to(device).eval().requires_grad_(False)
    output_rows = []
    with torch.inference_mode():
        for prompt_id in sorted(
            groups,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        ):
            records = groups[prompt_id]
            images = []
            for record in records:
                path = Path(record["image_path"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as source:
                    array = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
                images.append(torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0))
            pairs = [(i, j) for i in range(len(images)) for j in range(i + 1, len(images))]
            distances: list[float] = []
            for start in range(0, len(pairs), args.batch_size):
                chunk = pairs[start : start + args.batch_size]
                left = torch.stack([images[i] for i, _ in chunk]).to(device)
                right = torch.stack([images[j] for _, j in chunk]).to(device)
                values = model(left, right).flatten().float().cpu()
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"non-finite LPIPS values for prompt {prompt_id}")
                distances.extend(float(value) for value in values)
            output_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": records[0].get("prompt", ""),
                    "sample_count": len(images),
                    "pair_count": len(distances),
                    "lpips_diversity": statistics.fmean(distances),
                    "lpips_pair_std": statistics.stdev(distances) if len(distances) > 1 else 0.0,
                }
            )
    if sha256_file(args.manifest) != initial_manifest_sha:
        raise RuntimeError("generation manifest changed during LPIPS evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = args.output_dir / "lpips_diversity_per_prompt.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    payload = {
        "protocol": "prompt_lpips_diversity_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": initial_manifest_sha,
        "samples_per_prompt": args.samples_per_prompt,
        "pair_count_per_prompt": args.samples_per_prompt * (args.samples_per_prompt - 1) // 2,
        "lpips_net": args.net,
        "metric": summarize([float(row["lpips_diversity"]) for row in output_rows]),
        "per_prompt_csv": str(csv_path.resolve()),
        "per_prompt_csv_sha256": sha256_file(csv_path),
    }
    output = args.output_dir / "lpips_diversity_metrics.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=10)
    parser.add_argument("--net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.samples_per_prompt < 2 or args.batch_size < 1:
        parser.error("samples-per-prompt must be >=2 and batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
