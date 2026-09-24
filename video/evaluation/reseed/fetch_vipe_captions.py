#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Pull ReSeed training captions from the ViPE `long` WebDataset.

ReSeed needs prompts and noise, nothing else -- no videos. But the captions
only exist inside the WDS tar shards, interleaved with 1-2 MB .mp4 members,
and long/manifest.json carries a 100-character caption_preview, not the text.
So: download N shards, keep the *.txt members, delete the tars.

Why this set: it is the distribution DMD2 and DFD were distilled on, so a
bias fitted here is fitted where the student already lives. The eval prompts
(web_dfd_eval_prompts.txt) are a different, shorter style and must never
appear in training; this script asserts zero overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
from pathlib import Path

REPO_ID = "csusupergear/vipe_fliter_complete"


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("﻿", "")).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="long")
    ap.add_argument("--shards", type=int, default=40, help="how many shards, from 000000 upward")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-prompts", type=Path, default=None, help="assert zero overlap with this file")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--keep-tars", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # Xet is ~0.5 MB/s on this cluster
    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out / ".tars"
    scratch.mkdir(exist_ok=True)

    rows, dropped = [], 0
    for i in range(args.shards):
        name = f"{args.split}/{i:06d}.tar"
        path = Path(hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=str(scratch)))
        n_before = len(rows)
        with tarfile.open(path, "r") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".txt"):
                    continue
                text = normalise(tf.extractfile(m).read().decode("utf-8", errors="replace"))
                if len(text) < args.min_chars:
                    dropped += 1
                    continue
                rows.append({"sample_key": Path(m.name).stem, "shard": path.name, "caption": text})
        print(f"  [{i + 1}/{args.shards}] {path.name}: +{len(rows) - n_before} captions", flush=True)
        if not args.keep_tars:
            path.unlink()

    # De-duplicate on text; keep first occurrence.
    seen, unique = set(), []
    for r in rows:
        if r["caption"] in seen:
            continue
        seen.add(r["caption"])
        unique.append(r)

    if args.eval_prompts and args.eval_prompts.is_file():
        eval_set = {normalise(l) for l in args.eval_prompts.read_text().splitlines() if l.strip()}
        overlap = [r for r in unique if r["caption"] in eval_set]
        if overlap:
            print(f"ERROR: {len(overlap)} training captions appear in the eval prompt file", file=sys.stderr)
            return 2
        print(f"  overlap with {args.eval_prompts.name}: 0  (asserted)")

    captions_txt = args.out / "captions.txt"
    captions_txt.write_text("\n".join(r["caption"] for r in unique) + "\n")
    (args.out / "captions_manifest.json").write_text(json.dumps({
        "repo": REPO_ID, "split": args.split, "shards": args.shards,
        "n_raw": len(rows), "n_dropped_short": dropped, "n_unique": len(unique),
        "captions_sha256": hashlib.sha256(captions_txt.read_bytes()).hexdigest(),
        "rows": unique,
    }, indent=1, ensure_ascii=False))
    lens = [len(r["caption"]) for r in unique]
    print(f"\nwrote {captions_txt}: {len(unique)} unique captions "
          f"(raw {len(rows)}, dropped {dropped} short); length mean {sum(lens)/len(lens):.0f} "
          f"min {min(lens)} max {max(lens)}")
    if not args.keep_tars:
        try:
            scratch.rmdir()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
