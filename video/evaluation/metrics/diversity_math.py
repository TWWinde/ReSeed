# SPDX-License-Identifier: Apache-2.0
"""The DFD / DP-DMD visual diversity metric, isolated from any model loading.

Paper (Appendix B.3.1):

    Diversity = 1 - (2 / (L(L-1))) * sum_{i<j} cos(x_i, x_j)

with L = 8 videos per prompt, each from a different initial noise. x_i is the
VIDEO-level embedding: 16 evenly spaced frames are encoded and then mean-pooled
into one vector per video. The comparison is between videos, never between
frames of the same video -- confusing the two measures something else entirely
(temporal consistency), so the pooling happens first and is not optional here.

Since 2/(L(L-1)) * sum_{i<j} is exactly the mean over the C(L,2) unordered
pairs, the metric is 1 - (mean pairwise cosine).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np


def pool_frames(frame_embeddings: np.ndarray, l2_normalise_frames: bool = True) -> np.ndarray:
    """[T, D] frame embeddings -> [D] video embedding.

    DFD writes x_i = (1/16) sum_t f(I_{i,t}) and says the per-frame f is the
    DP-DMD extractor. That extractor (dpdmd/test_diversity.py) returns
    F.normalize()d vectors, so the frames being averaged are unit vectors:
    default True. False reproduces the earlier, literal-unnormalised numbers.
    """
    if frame_embeddings.ndim != 2:
        raise ValueError(f"expected [T, D], got {frame_embeddings.shape}")
    x = frame_embeddings.astype(np.float64)
    if l2_normalise_frames:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-12)
    return x.mean(axis=0)


def pairwise_cosines(embeddings: np.ndarray) -> np.ndarray:
    """[L, D] -> the C(L,2) unordered pairwise cosine similarities."""
    if embeddings.ndim != 2:
        raise ValueError(f"expected [L, D], got {embeddings.shape}")
    L = embeddings.shape[0]
    if L < 2:
        raise ValueError(f"need at least 2 videos to form a pair, got {L}")
    x = embeddings.astype(np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return np.array([float(np.dot(x[i], x[j])) for i, j in combinations(range(L), 2)])


def diversity(embeddings: np.ndarray) -> float:
    """1 - mean pairwise cosine over the C(L,2) pairs."""
    return float(1.0 - pairwise_cosines(embeddings).mean())


def diversity_per_frame(frame_embeddings: np.ndarray) -> float:
    """The paper's "Per-frame" column: [L, T, D] -> mean over t of the
    between-video diversity at frame t.

    No pooling over time. This is the control for the mean-pooled variant: a
    video with more motion averages into a "blurrier" pooled vector, which can
    make pooled vectors from different seeds look MORE alike -- so the pooled
    metric can read higher dynamics as lower diversity. Per-frame does not have
    that coupling.
    """
    x = np.asarray(frame_embeddings, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected [L, T, D], got {x.shape}")
    return float(np.mean([diversity(x[:, t, :]) for t in range(x.shape[1])]))


def bootstrap_ci(values: np.ndarray, n_samples: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap CI for the mean over prompts."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_samples, values.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarise(per_prompt: np.ndarray, n_samples: int = 2000, seed: int = 0) -> Dict[str, object]:
    v = np.asarray(per_prompt, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n_prompts": 0, "mean": None, "std": None, "ci95": None}
    ci = bootstrap_ci(v, n_samples=n_samples, seed=seed)
    return {
        "n_prompts": int(v.size),
        "mean": float(v.mean()),
        # Sample std (ddof=1): these prompts are a sample, not the population.
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "ci95": list(ci) if ci else None,
    }
