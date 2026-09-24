#!/usr/bin/env python3
"""Strictly summarize multi-method DINO/CLIP feature-diversity results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--samples-per-prompt", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for variant in args.variant:
        directory = args.evaluation_root / variant
        metrics_path = directory / "diversity_metrics" / "diversity_metrics.json"
        csv_path = directory / "diversity_metrics" / "diversity_per_prompt.csv"
        manifest_path = directory / "generation_manifest.jsonl"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with csv_path.open(newline="", encoding="utf-8") as handle:
            prompt_rows = list(csv.DictReader(handle))
        manifest_count = sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if len(prompt_rows) != args.expected_prompts:
            raise ValueError(f"{variant}: expected {args.expected_prompts} prompt rows, got {len(prompt_rows)}")
        expected_images = args.expected_prompts * args.samples_per_prompt
        if manifest_count != expected_images or int(metrics["image_count"]) != expected_images:
            raise ValueError(f"{variant}: expected {expected_images} images")
        if int(metrics["samples_per_prompt"]) != args.samples_per_prompt:
            raise ValueError(f"{variant}: samples-per-prompt mismatch")
        rows.append(
            {
                "variant": variant,
                "dino_diversity_mean": finite(metrics["metrics"]["dino"]["mean"], f"{variant} DINO mean"),
                "dino_diversity_std": finite(metrics["metrics"]["dino"]["std"], f"{variant} DINO std"),
                "clip_diversity_mean": finite(metrics["metrics"]["clip"]["mean"], f"{variant} CLIP mean"),
                "clip_diversity_std": finite(metrics["metrics"]["clip"]["std"], f"{variant} CLIP std"),
                "manifest_sha256": sha256_file(manifest_path),
                "metrics_sha256": sha256_file(metrics_path),
                "per_prompt_csv_sha256": sha256_file(csv_path),
            }
        )

    payload = {
        "protocol": "pixart_diversity_100x9_five_method_v1",
        "prompt_count": args.expected_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "image_count_per_method": args.expected_prompts * args.samples_per_prompt,
        "variants": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PixArt five-method feature diversity (100 prompts × 9 seeds)",
        "",
        "| Method | DINOv3 diversity mean ± std | CLIP diversity mean ± std |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['dino_diversity_mean']:.6f} ± {row['dino_diversity_std']:.6f} "
            f"| {row['clip_diversity_mean']:.6f} ± {row['clip_diversity_std']:.6f} |"
        )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

