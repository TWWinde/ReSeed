#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Plot and summarise a ReSeed training run from its metrics.jsonl.

    python evaluation/reseed/plot_training.py <run_dir> [--window 25]

Prints a table at every saved checkpoint (rolling means, so the noisy per-step
loss is readable) and writes <run_dir>/training_curves.png. The question this
answers: has the fit flattened yet? The image experiments saw diversity saturate by update
300-400; if loss and velocity cosine are flat by then, the 2000-step budget is
not buying anything and the run can be stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KEYS = ["loss", "velocity_cosine", "velocity_relative_error", "endpoint_relative_error",
        "gradient_norm", "bias_rms", "bias_abs_max"]


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < 2 or w <= 1:
        return x
    w = min(w, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    head = np.array([x[: i + 1].mean() for i in range(w - 1)])
    return np.concatenate([head, out])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    path = args.run_dir / "metrics.jsonl"
    if not path.is_file():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        print("no rows yet"); return 0
    step = np.array([r["step"] for r in rows])
    series = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in KEYS}
    tsec = np.array([r.get("teacher_seconds", np.nan) for r in rows], dtype=float)
    ssec = np.array([r.get("student_seconds", np.nan) for r in rows], dtype=float)

    ckpts = sorted(int(p.name.split("-")[-1]) for p in args.run_dir.glob("checkpoint-*") if p.is_dir())
    marks = ckpts or [int(step[-1])]
    print(f"{len(rows)} logged steps, last step {step[-1]}, checkpoints: {ckpts or 'none yet'}")
    print(f"per-step time: teacher {np.nanmean(tsec):.1f}s  student {np.nanmean(ssec):.1f}s  "
          f"-> {np.nanmean(tsec + ssec):.1f}s/step")
    hdr = f"{'step':>6s} " + " ".join(f"{k[:14]:>14s}" for k in KEYS)
    print(hdr); print("-" * len(hdr))
    sm = {k: rolling(v, args.window) for k, v in series.items()}
    for m in marks:
        i = int(np.searchsorted(step, m, side="right") - 1)
        if i < 0:
            continue
        print(f"{int(step[i]):>6d} " + " ".join(f"{sm[k][i]:>14.5f}" for k in KEYS))

    # Flatness: relative change of rolling loss/cosine over the last 20% of steps.
    n = len(step)
    if n >= 40:
        a, b = int(n * 0.6), int(n * 0.8)
        for k in ("loss", "velocity_cosine"):
            v = sm[k]
            rel = (v[-1] - v[b]) / max(abs(v[b] - v[a]), 1e-12)
            print(f"flatness {k:16s}: change over last 20% of run is {rel:+.2f}x the change over the preceding 20%")

    if args.no_plot:
        return 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, k in zip(axes.flat, KEYS):
        ax.plot(step, series[k], lw=0.6, alpha=0.35)
        ax.plot(step, sm[k], lw=1.6)
        for m in ckpts:
            ax.axvline(m, color="k", lw=0.4, alpha=0.3)
        ax.set_title(k); ax.set_xlabel("update")
        if k in ("loss", "gradient_norm"):
            ax.set_yscale("log")
    ax = axes.flat[-1]
    ax.plot(step, tsec, lw=0.8, label="teacher s"); ax.plot(step, ssec, lw=0.8, label="student s")
    ax.set_title("seconds per update"); ax.legend()
    fig.suptitle(f"ReSeed training -- {args.run_dir.name}")
    fig.tight_layout()
    out = args.run_dir / "training_curves.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
