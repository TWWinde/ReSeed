#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Pre-encode ReSeed training captions with Wan's UMT5-XXL, once.

Training then never touches the 22.7 GB text encoder. Embeddings go through
the same WanTextEncoder.encode() the inference script uses, so the training
conditioning has the same shape and padding convention ([512, 4096], zero
padded past seq_len) as what the deployed student sees.

One acknowledged difference: inference runs UMT5 on CPU in fp32 (the script
never moves it), this runs it on GPU in fp32. Different accumulation order,
last-bit differences. It is the training signal, not the evaluation input,
so that is acceptable -- and it is recorded in the cache header.

Storage: the encoder zero-pads to 512 tokens; a full [512, 4096] bf16 row is
4 MB, so 4,000 captions would be 16 GB. Rows are stored trimmed to seq_len
(mean ~200 for ViPE long captions) and re-padded at load time.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MAX_TOKENS = 512


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--negative-prompt", type=Path, default=REPO / "scripts/inference/prompts/negative_prompt.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("LOCAL_FILES_ONLY", "true")
    from fastgen.networks.Wan.network import WanTextEncoder

    captions = [l.strip() for l in args.captions.read_text().splitlines() if l.strip()]
    if args.limit:
        captions = captions[: args.limit]
    neg = [l.strip() for l in args.negative_prompt.read_text().splitlines() if l.strip()][:1]
    if not captions or not neg:
        print("ERROR: empty captions or negative prompt", file=sys.stderr)
        return 2

    enc = WanTextEncoder(model_id_or_local_path=args.model_id).to(args.device)
    print(f"encoding {len(captions)} captions + 1 negative on {args.device}")

    def seq_lens(texts):
        tok = enc.tokenizer(texts, padding="max_length", max_length=MAX_TOKENS, truncation=True,
                            add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        return tok.attention_mask.gt(0).sum(dim=1).tolist()

    rows, lens = [], []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(captions), args.batch_size):
            chunk = captions[i:i + args.batch_size]
            emb = enc.encode(chunk)                      # [B, 512, 4096] fp32, zero-padded
            for e, L in zip(emb, seq_lens(chunk)):
                rows.append(e[:L].to(torch.bfloat16).cpu().contiguous())
                lens.append(int(L))
            if (i // args.batch_size) % 25 == 0:
                print(f"  {i + len(chunk)}/{len(captions)}  {(time.time() - t0):.0f}s", flush=True)
        neg_emb = enc.encode(neg)[0].to(torch.bfloat16).cpu().contiguous()   # keep full [512, 4096]

    payload = {
        "format": "wan-umt5-embeddings-v1",
        "model_id": args.model_id,
        "max_tokens": MAX_TOKENS,
        "dtype": "bfloat16",
        "encoder_device": args.device,
        "captions_path": str(args.captions),
        "captions_sha256": hashlib.sha256(args.captions.read_bytes()).hexdigest(),
        "n": len(rows),
        "seq_lens": lens,
        "captions": captions,
        "embeds": rows,                 # list of [L_i, 4096] bf16
        "negative_prompt": neg[0],
        "negative_embed": neg_emb,      # [512, 4096] bf16
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    size = args.out.stat().st_size / 1e9
    print(f"wrote {args.out}  ({size:.2f} GB, {len(rows)} rows, mean seq_len {sum(lens)/len(lens):.0f}, "
          f"{time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
