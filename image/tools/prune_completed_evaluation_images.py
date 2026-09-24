#!/usr/bin/env python3
"""Keep a compact quality preview after all COCO quality metrics are complete.

Only the COCO1K/COCO5K image directory is pruned. Diversity banks are never
touched because their images may be reused for later per-prompt analyses.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--quality-dir",
        default=None,
        help="quality directory below --root; auto-detects coco5k or nested_coco1k",
    )
    parser.add_argument("--keep", type=int, default=100)
    # A bank without reference images has no FID, so the completeness check has
    # to name the metrics rather than assume the COCO set: PartiPrompts ships
    # prompts only, and waiting for a metrics_fid.json that can never be written
    # would leave its images unpruned for ever.
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["hps", "aes", "fid"],
        help="extended metrics that must exist before pruning (default: hps aes fid)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.keep < 1:
        raise SystemExit("--keep must be positive")
    if args.quality_dir is not None:
        quality = root / args.quality_dir
    else:
        candidates = [root / "coco5k", root / "nested_coco1k", root / "coco1k"]
        found = [path for path in candidates if path.is_dir()]
        if len(found) != 1:
            raise SystemExit(
                f"could not uniquely auto-detect quality directory below {root}: {found}"
            )
        quality = found[0]
    required = [
        quality / "metrics.json",
        quality / "generation_manifest.jsonl",
        *(quality / f"extended_metrics/metrics_{name}.json" for name in args.metrics),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({"status": "not_pruned", "missing": missing}, indent=2))
        return 2

    manifest = quality / "generation_manifest.jsonl"
    images = sorted(
        Path(json.loads(line)["image_path"])
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    # Each image carries a <name>.identity.json sidecar. Deleting the image and
    # leaving the sidecar turned a 5,000-image directory into 100 images plus
    # 5,000 sidecars, and the file count, not the byte count, is what exhausted
    # the project quota. Everything a sidecar holds (latent_sha256, image_sha256,
    # width, height, image_path) is already in generation_manifest.jsonl for
    # every case, so the sidecar of a deleted image records nothing recoverable.
    sidecars = 0
    for image in images[args.keep :]:
        if image.is_file():
            image.unlink()
        sidecar = image.with_name(image.name + ".identity.json")
        if sidecar.is_file():
            sidecar.unlink()
            sidecars += 1
    marker = quality / "IMAGES_PRUNED.json"
    marker.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "quality_directory": str(quality),
                "kept": args.keep,
                "before": len(images),
                "after": min(len(images), args.keep),
                "sidecars_removed": sidecars,
                "diversity_images_pruned": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
