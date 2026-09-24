# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the diversity metric, on synthetic embeddings only.

Run: python evaluation/metrics/test_diversity_math.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.metrics.diversity_math import (  # noqa: E402
    bootstrap_ci, diversity, diversity_per_frame, pairwise_cosines, pool_frames, summarise,
)

L, D = 8, 64
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'OK ' if ok else 'BAD'}] {name:<46s} {detail}")


def main() -> int:
    rng = np.random.default_rng(0)

    # 1. Identical embeddings -> cosine 1 -> diversity 0.
    same = np.tile(rng.normal(size=D), (L, 1))
    d = diversity(same)
    check("identical embeddings -> ~0", abs(d) < 1e-9, f"D={d:.3e}")

    # 2. Mutually orthogonal -> cosine 0 -> diversity 1.
    orth = np.eye(L, D)
    d = diversity(orth)
    check("orthogonal embeddings -> ~1", abs(d - 1.0) < 1e-9, f"D={d:.6f}")

    # 3. Antipodal pairs push above 1; the metric is not clipped.
    anti = np.concatenate([np.tile(np.eye(1, D)[0], (L // 2, 1)),
                           np.tile(-np.eye(1, D)[0], (L // 2, 1))])
    d = diversity(anti)
    check("antipodal -> >1", d > 1.0, f"D={d:.6f}")

    # 4. Exactly C(8,2) = 28 pairs, each counted once.
    n = len(pairwise_cosines(rng.normal(size=(L, D))))
    check("pair count == C(8,2) == 28", n == 28 == len(list(combinations(range(L), 2))), f"n={n}")

    # 5. Permuting the seeds must not change the result.
    emb = rng.normal(size=(L, D))
    base = diversity(emb)
    perms = [diversity(emb[rng.permutation(L)]) for _ in range(20)]
    check("invariant to seed ordering", all(abs(p - base) < 1e-12 for p in perms),
          f"max dev={max(abs(p - base) for p in perms):.2e}")

    # 6. Scale invariance: cosine ignores magnitude.
    scaled = emb * rng.uniform(0.1, 10.0, size=(L, 1))
    check("invariant to per-video scaling", abs(diversity(scaled) - base) < 1e-9,
          f"delta={abs(diversity(scaled) - base):.2e}")

    # 7. More spread -> more diversity (monotone sanity check).
    anchor = rng.normal(size=D)
    tight = anchor + 0.01 * rng.normal(size=(L, D))
    loose = anchor + 1.00 * rng.normal(size=(L, D))
    check("spread-out > tight cluster", diversity(loose) > diversity(tight),
          f"{diversity(tight):.4f} < {diversity(loose):.4f}")

    # 8. Pooling happens BEFORE comparison. Frame-to-frame diversity of one
    #    video is a different quantity; this guards against conflating them.
    frames = rng.normal(size=(16, D))
    pooled = pool_frames(frames)
    check("pool_frames -> [D]", pooled.shape == (D,), f"{frames.shape} -> {pooled.shape}")
    unit = frames / np.linalg.norm(frames, axis=1, keepdims=True)
    check("pool_frames default == mean of unit frames", np.allclose(pooled, unit.mean(axis=0)), "(DP-DMD extractor convention)")
    check("pool_frames(l2=False) == plain mean", np.allclose(pool_frames(frames, l2_normalise_frames=False), frames.mean(axis=0)), "")
    within_video = diversity(frames)      # 16 frames of ONE video
    videos = np.stack([pool_frames(rng.normal(size=(16, D))) for _ in range(L)])
    across_videos = diversity(videos)
    check("frame-level != video-level", abs(within_video - across_videos) > 1e-3,
          f"within={within_video:.4f} across={across_videos:.4f}")

    # 8b. Per-frame variant: equals the pooled variant when every video is
    #     static (all 16 frames identical), and differs when frames move.
    static = np.repeat(rng.normal(size=(L, 1, D)), 16, axis=1)          # [L, 16, D], no motion
    pooled_static = diversity(np.stack([pool_frames(v) for v in static]))
    check("per-frame == pooled for static videos", abs(diversity_per_frame(static) - pooled_static) < 1e-9,
          f"{diversity_per_frame(static):.6f} vs {pooled_static:.6f}")
    moving = static + 0.5 * rng.normal(size=(L, 16, D))
    check("per-frame != pooled for moving videos",
          abs(diversity_per_frame(moving) - diversity(np.stack([pool_frames(v) for v in moving]))) > 1e-4, "")
    # Per-frame must NOT be secretly pooling: with iid per-frame jitter on top
    # of a shared-plus-per-video signal, pooling averages the jitter away (so
    # pooled vectors stay close), while per-frame keeps it (so per-frame
    # diversity is strictly higher). Random high-dim Gaussians alone are all
    # near-orthogonal (diversity ~1 either way), hence the shared component.
    common = rng.normal(size=(1, 1, D)); per_video = 0.3 * rng.normal(size=(L, 1, D))
    jitter = common + per_video + 0.6 * rng.normal(size=(L, 16, D))
    pooled_j = diversity(np.stack([pool_frames(v) for v in jitter])); pf_j = diversity_per_frame(jitter)
    check("per-frame keeps jitter that pooling averages out", pf_j > pooled_j + 1e-3,
          f"per-frame {pf_j:.4f} > pooled {pooled_j:.4f}")

    # 9. Rejects a single video (no pair exists).
    try:
        diversity(np.zeros((1, D)))
        check("rejects L<2", False, "no exception raised")
    except ValueError:
        check("rejects L<2", True, "ValueError")

    # 10. Bootstrap CI brackets the mean and narrows with more data.
    vals = rng.normal(0.2, 0.05, size=71)
    lo, hi = bootstrap_ci(vals, n_samples=2000, seed=0)
    check("bootstrap CI brackets mean", lo < vals.mean() < hi, f"[{lo:.4f}, {hi:.4f}]")
    lo2, hi2 = bootstrap_ci(rng.normal(0.2, 0.05, size=500), n_samples=2000, seed=0)
    check("CI narrows with more prompts", (hi2 - lo2) < (hi - lo), f"{hi-lo:.4f} -> {hi2-lo2:.4f}")

    s = summarise(vals)
    check("summarise reports n/mean/std/ci", all(s[k] is not None for k in ("mean", "std", "ci95")),
          f"n={s['n_prompts']} mean={s['mean']:.4f} std={s['std']:.4f}")

    bad = [n for n, ok, _ in results if not ok]
    print()
    if bad:
        print(f"FAILED: {bad}")
        return 1
    print(f"ALL {len(results)} DIVERSITY MATH TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
