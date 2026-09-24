#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Per-category diversity breakdown for a prompt set with a category sidecar.

    python evaluation/tools/category_breakdown.py \
        --results-root evaluation/results_vbench320 \
        --categories evaluation/prompts/vbench_category_320.categories.csv \
        --baseline dmd2

Reads <results_root>/diversity_per_prompt.csv, groups prompts by category and
writes category_breakdown.{md,json}: per-arm means and, for every other arm,
the paired per-prompt difference against --baseline with a bootstrap 95% CI
and the number of prompts where the arm is higher.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from pathlib import Path

METRICS = ["dino_diversity", "clip_diversity", "clip_cls_diversity", "clip_text_adherence"]


def paired(a: dict, b: dict, ids: list, key: str, B: int, rng: random.Random):
    d = [float(a[i][key]) - float(b[i][key]) for i in ids]
    boots = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return st.mean(d), boots[int(0.025 * B)], boots[int(0.975 * B)], sum(x > 0 for x in d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--categories", required=True)
    ap.add_argument("--baseline", default="dmd2")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    root = Path(a.results_root)
    cat_of = {int(r["prompt_id"]): r["category"] for r in csv.DictReader(open(a.categories, encoding="utf-8"))}
    by: dict = {}
    for r in csv.DictReader(open(root / "diversity_per_prompt.csv", encoding="utf-8")):
        by.setdefault(r["model"], {})[int(r["prompt_id"])] = r
    arms = sorted(by)
    if a.baseline not in by:
        raise SystemExit(f"baseline {a.baseline!r} not in results ({arms})")
    cats = sorted({cat_of[i] for m in by.values() for i in m if i in cat_of})
    rng = random.Random(0)
    out = {"baseline": a.baseline, "arms": arms, "categories": {}}
    lines = [f"# Per-category diversity ({root.name}; baseline = {a.baseline})", ""]
    for key in METRICS:
        lines += [f"## {key}", "", "| category | n | " + " | ".join(arms) + " | " +
                  " | ".join(f"{m} − {a.baseline}" for m in arms if m != a.baseline) + " |",
                  "|---|---|" + "---|" * len(arms) + "---|" * (len(arms) - 1)]
        for c in cats + ["ALL"]:
            ids = sorted(i for i in by[a.baseline] if c == "ALL" or cat_of.get(i) == c)
            ids = [i for i in ids if all(i in by[m] for m in arms)]
            if not ids:
                continue
            means = {m: st.mean(float(by[m][i][key]) for i in ids) for m in arms}
            diffs = {m: paired(by[m], by[a.baseline], ids, key, a.bootstrap, rng) for m in arms if m != a.baseline}
            out["categories"].setdefault(c, {})[key] = {"n": len(ids), "means": means,
                "paired_vs_baseline": {m: {"mean": d[0], "ci95": [d[1], d[2]], "wins": d[3]} for m, d in diffs.items()}}
            lines.append(f"| {c} | {len(ids)} | " + " | ".join(f"{means[m]:.4f}" for m in arms) + " | " +
                         " | ".join(f"{d[0]:+.4f} [{d[1]:+.3f}, {d[2]:+.3f}] {d[3]}/{len(ids)}" for d in diffs.values()) + " |")
        lines.append("")
    (root / "category_breakdown.json").write_text(json.dumps(out, indent=2))
    (root / "category_breakdown.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
