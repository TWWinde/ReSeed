#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Seed-grid figure: one frame per (arm, seed) for a prompt, arms as rows,
seeds as columns. Same seed = same initial noise across arms, so a column
shows what each arm does with identical noise.

    python evaluation/tools/seed_grid.py --prompts 26,13 \
        --arms teacher,dmd2,dmd2_reseed_ckpt300,dfd --frame 40 --out figs/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from evaluation.tools.eval_config import load_config, resolve  # noqa: E402

LABELS = {"teacher": "Teacher (50-step)", "dmd2": "DMD2", "dfd": "DFD",
          "dmd2_reseed_ckpt100": "DMD2+ReSeed@100", "dmd2_reseed_ckpt200": "DMD2+ReSeed@200",
          "dmd2_reseed_ckpt300": "DMD2+ReSeed@300", "dmd2_reseed_ckpt500": "DMD2+ReSeed@500"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "evaluation/config.yaml"))
    ap.add_argument("--prompts", required=True, help="comma-separated prompt ids")
    ap.add_argument("--arms", default="teacher,dmd2,dmd2_reseed_ckpt300,dfd")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--frame", type=int, default=40, help="frame index (0-80)")
    ap.add_argument("--thumb", default="208x120")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out_root = resolve(cfg, cfg["output_root"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tw, th = (int(x) for x in a.thumb.split("x"))
    arms = a.arms.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    meta = {}
    for line in (out_root / "metadata.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line); meta.setdefault(r["prompt_id"], r["prompt"])
    per_prompt = {}
    csv_path = resolve(cfg, cfg["results_root"]) / "diversity_per_prompt.csv"
    if csv_path.is_file():
        import csv
        for r in csv.DictReader(csv_path.open()):
            per_prompt[(r["model"], int(r["prompt_id"]))] = float(r["dino_diversity"])
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    label_w, header_h, pad = 170, 44, 4
    for pid in (int(p) for p in a.prompts.split(",")):
        W = label_w + len(seeds) * (tw + pad) + pad
        H = header_h + len(arms) * (th + pad) + pad
        canvas = Image.new("RGB", (W, H), "white"); draw = ImageDraw.Draw(canvas)
        draw.text((pad, 4), f"prompt {pid:03d}: {meta.get(pid, '')[:150]}", fill="black", font=font)
        draw.text((pad, 24), f"frame {a.frame}/80, columns = seeds {seeds[0]}..{seeds[-1]} (identical initial noise per column)", fill="gray", font=font)
        for ri, arm in enumerate(arms):
            y = header_h + ri * (th + pad)
            lab = LABELS.get(arm, arm)
            if (arm, pid) in per_prompt:
                lab += f"\nDINO div {per_prompt[(arm, pid)]:.3f}"
            draw.multiline_text((pad, y + th // 2 - 14), lab, fill="black", font=font)
            for ci, seed in enumerate(seeds):
                vp = out_root / arm / f"prompt_{pid:03d}" / f"seed_{seed:03d}.mp4"
                x = label_w + ci * (tw + pad)
                if not vp.is_file():
                    draw.rectangle([x, y, x + tw, y + th], outline="red"); continue
                fr = iio.imread(vp, index=a.frame, plugin="pyav")
                canvas.paste(Image.fromarray(fr).resize((tw, th), Image.LANCZOS), (x, y))
        p = out / f"seed_grid_p{pid:03d}_f{a.frame}.png"; canvas.save(p); print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
