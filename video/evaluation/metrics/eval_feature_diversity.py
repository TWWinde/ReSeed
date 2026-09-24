#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""DFD / DP-DMD visual diversity for DMD2 vs DFD.

Protocol (paper Appendix B.3.1), per prompt and per model:

  1. L = 8 videos, one per initial-noise seed.
  2. Sample 16 evenly spaced frames from each video.
  3. Encode every frame with DINOv2-ViT-L and CLIP-ViT-L.
  4. Mean-pool the 16 frame embeddings -> ONE embedding per video.
  5. Diversity = 1 - (2 / (L(L-1))) * sum_{i<j} cos(x_i, x_j)

The pooling in step 4 is what makes this a between-video measure. Comparing
frames within a video would measure temporal consistency instead; the maths
lives in diversity_math.py and is unit-tested against exactly that confusion.

WHAT THE PAPER PINS DOWN, AND WHAT IT DOES NOT
The DFD paper (B.3.1) fixes the formula, L=8, 16 evenly spaced frames, mean
pooling, and the two architectures, and says "we adopt the diversity metric
from DP-DMD [32]". DFD's own evaluation code is not released, and DFD is a
different method from DP-DMD, so DP-DMD's public test_diversity.py is the
closest public reference for the *formula* -- not evidence of which CLIP
feature DFD's authors used. Two readings are both defensible:
  - "CLIP image features" in the usual sense: get_image_features() /
    image_embeds, 768-d, post-layernorm + visual_projection.
  - What DP-DMD's code does: CLIPVisionModel.last_hidden_state[:, 0], the
    vision CLS BEFORE post_layernorm and BEFORE projection, 1024-d.
The two spaces have different cosine geometry, so BOTH are computed from the
same forward pass and BOTH are reported (clip_proj_* and clip_cls_*). Neither
is labelled "the" reproduction. DINO is unaffected: pooler_output and
last_hidden_state[:, 0] are the same tensor for Dinov2Model (transformers
4.49, modeling_dinov2.py:703-705), and every reading agrees on it.
Per-frame vectors are L2-normalised before pooling by default (unit vectors
are what both readings feed to a cosine); --no-frame-norm reproduces the
literal unnormalised average.
Text-image adherence uses the projected joint space (image_embeds vs
text_embeds), the only space where that cosine is meaningful.

REPRODUCTION CHOICES (the paper names architectures, not checkpoints):
  - "DINOv2-ViT-Large" -> facebook/dinov2-large  (DP-DMD's own README uses
    dinov2-base and its code default points at dinov3; DFD's text says v2-L)
  - "CLIP-ViT-Large"   -> openai/clip-vit-large-patch14 (matches DP-DMD)
Each is recorded in the output JSON so a later reader can see what was assumed.

ADHERENCE (added for the ReSeed comparison): per video, the mean cosine between
the CLIP text embedding of the prompt and the 16 CLIP frame embeddings. CRRA
found every diversity-raising intervention pays in compositional accuracy, so
a diversity number without an adherence number next to it is half a result.
Reported as `clip_text_adherence`; higher = frames match the prompt better.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.metrics.diversity_math import diversity, diversity_per_frame, pool_frames, summarise  # noqa: E402
from evaluation.tools.eval_config import load_config, resolve  # noqa: E402


def sample_frame_indices(total: int, n: int) -> np.ndarray:
    """n evenly spaced indices spanning [0, total-1] inclusive."""
    if total < n:
        raise ValueError(f"video has {total} frames, need at least {n}")
    return np.linspace(0, total - 1, n).round().astype(int)


def color_normalize(frames: np.ndarray) -> np.ndarray:
    """Diagnostic: strip colour and contrast before feature extraction.

    DMD2's outputs are strongly over-saturated (HSV saturation ~0.85 vs ~0.63
    for DFD on the same prompt) and have ~35% higher pixel std. If a feature
    encoder reads seed-to-seed variation in those saturation artefacts as
    "diversity", the saturated arm gets a free boost that has nothing to do
    with layout or content. Grayscale + per-frame z-score (rescaled to 0-255,
    replicated to 3 channels) removes that channel; whatever diversity
    survives is structural.
    """
    g = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)   # [T,H,W]
    mu = g.mean(axis=(1, 2), keepdims=True); sd = g.std(axis=(1, 2), keepdims=True) + 1e-6
    z = (g - mu) / sd
    out = np.clip(127.5 + 40.0 * z, 0, 255).astype(np.uint8)          # ~3 sigma inside range
    return np.repeat(out[..., None], 3, axis=-1)


def read_frames(path: Path, n: int) -> np.ndarray:
    import imageio.v3 as iio

    for kwargs in ({"plugin": "pyav"}, {"plugin": "FFMPEG"}, {}):
        try:
            frames = iio.imread(path, **kwargs)
            break
        except Exception:  # noqa: BLE001
            frames = None
    if frames is None:
        raise RuntimeError(f"could not read {path}")
    return frames[sample_frame_indices(len(frames), n)]


class Encoders:
    """DINOv2 and CLIP image towers, loaded once."""

    def __init__(self, dino_id: str, clip_id: str, device: str, cache_dir: str | None):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection

        self.torch = torch
        self.device = device
        kw = {"cache_dir": cache_dir} if cache_dir else {}

        self.dino_proc = AutoImageProcessor.from_pretrained(dino_id, **kw)
        self.dino = AutoModel.from_pretrained(dino_id, **kw).to(device).eval()
        self.clip_proc = CLIPImageProcessor.from_pretrained(clip_id, **kw)
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_id, **kw).to(device).eval()
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        self.clip_tok = CLIPTokenizer.from_pretrained(clip_id, **kw)
        self.clip_text = CLIPTextModelWithProjection.from_pretrained(clip_id, **kw).to(device).eval()

        self.info = {
            "dino_model": dino_id,
            "clip_model": clip_id,
            "dino_feature": "pooler_output == layernorm(last_hidden_state)[:, 0] (CLS), as DP-DMD",
            "clip_cls_feature": "vision last_hidden_state[:, 0] -- pre-post_layernorm, pre-projection CLS (1024-d), as DP-DMD's code",
            "clip_proj_feature": "image_embeds -- post-layernorm + visual_projection (768-d), get_image_features()",
            "clip_adherence_feature": "image_embeds vs text_embeds (projected joint space)",
            "frame_l2_normalised_before_pooling": True,
            "clip_text_feature": "text_embeds (projected); prompts truncated to CLIP's 77 tokens",
            "dino_hidden": int(self.dino.config.hidden_size),
            "clip_dim": int(self.clip.config.projection_dim),
            "device": device,
        }

    def encode_text(self, prompt: str) -> np.ndarray:
        """[Dc] L2-normalised CLIP text embedding."""
        torch = self.torch
        with torch.inference_mode():
            tok = self.clip_tok([prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
            e = self.clip_text(**tok).text_embeds[0].float().cpu().numpy()
        return e / max(np.linalg.norm(e), 1e-12)

    def encode(self, frames: np.ndarray, batch_size: int) -> Dict[str, np.ndarray]:
        """[T,H,W,C] uint8 -> {'dino': [T,Dd], 'clip': [T,Dc]} float32."""
        torch = self.torch
        images = [f for f in frames]
        out: Dict[str, List[np.ndarray]] = {"dino": [], "clip": [], "clip_proj": []}
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                chunk = images[start:start + batch_size]
                d_in = self.dino_proc(images=chunk, return_tensors="pt").to(self.device)
                out["dino"].append(self.dino(**d_in).pooler_output.float().cpu().numpy())
                c_in = self.clip_proc(images=chunk, return_tensors="pt").to(self.device)
                c_out = self.clip(**c_in)
                out["clip"].append(c_out.last_hidden_state[:, 0].float().cpu().numpy())   # DP-DMD's CLS
                out["clip_proj"].append(c_out.image_embeds.float().cpu().numpy())         # for adherence only
        return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "evaluation/config.yaml"))
    ap.add_argument("--metadata", default=None, help="default: <output_root>/metadata.jsonl")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-prompts", type=int, default=None, help="evaluate only the first N prompt ids")
    ap.add_argument("--expected-seeds", type=int, default=8)
    ap.add_argument("--no-frame-norm", action="store_true",
                    help="average raw per-frame vectors instead of unit vectors (literal x_i = mean_t f)")
    ap.add_argument("--color-normalize", action="store_true",
                    help="diagnostic: grayscale + per-frame contrast normalisation before encoding")
    args = ap.parse_args()

    cfg = load_config(args.config)
    div_cfg = cfg["diversity"]
    out_root = resolve(cfg, cfg["output_root"])
    meta_path = Path(args.metadata) if args.metadata else out_root / "metadata.jsonl"
    results_root = Path(args.results_root) if args.results_root else resolve(cfg, cfg["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)

    if not meta_path.is_file():
        print(f"ERROR: {meta_path} not found -- run generate_multiseed.py first", file=sys.stderr)
        return 2

    records = [json.loads(ln) for ln in meta_path.read_text().splitlines() if ln.strip()]
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    skipped_invalid = 0
    for r in records:
        if not r.get("valid", False):
            skipped_invalid += 1
            continue
        if args.num_prompts is not None and r["prompt_id"] >= args.num_prompts:
            continue
        groups[(r["model"], r["prompt_id"])].append(r)

    if skipped_invalid:
        print(f"WARNING: {skipped_invalid} metadata records are marked invalid and are excluded")
    if not groups:
        print("ERROR: no valid videos to evaluate", file=sys.stderr)
        return 2

    n_frames = int(div_cfg["num_frames_sampled"])
    enc = Encoders(div_cfg["dino_model"], div_cfg["clip_model"], args.device,
                   str(resolve(cfg, cfg["paths"]["hf_home"])))
    print("encoders:", json.dumps(enc.info, indent=2))

    rows, failures = [], []
    started = time.time()
    for n, (key, group) in enumerate(sorted(groups.items()), 1):
        model, prompt_id = key
        group = sorted(group, key=lambda r: r["seed"])
        if len(group) != args.expected_seeds:
            failures.append(f"{model} prompt {prompt_id:03d}: {len(group)} seeds, expected {args.expected_seeds}")

        embeds = {"dino": [], "clip_cls": [], "clip_proj": []}
        frames_by = {"dino": [], "clip_cls": [], "clip_proj": []}   # unpooled [T, D] per video
        adherence = []
        text_emb = enc.encode_text(group[0]["prompt"])
        used = 0
        for rec in group:
            path = Path(rec["video_path"])
            if not path.is_absolute():
                path = REPO / path
            try:
                frames = read_frames(path, n_frames)
            except Exception as exc:  # noqa: BLE001 -- never skip silently
                failures.append(f"{model} prompt {prompt_id:03d} seed {rec['seed']}: {exc}")
                continue
            if args.color_normalize:
                frames = color_normalize(frames)
            per_frame = enc.encode(frames, int(div_cfg.get("batch_size", 32)))
            src = {"dino": per_frame["dino"], "clip_cls": per_frame["clip"], "clip_proj": per_frame["clip_proj"]}
            for k in embeds:
                embeds[k].append(pool_frames(src[k], l2_normalise_frames=not args.no_frame_norm))
                frames_by[k].append(src[k])
            f = per_frame["clip_proj"] / np.maximum(np.linalg.norm(per_frame["clip_proj"], axis=1, keepdims=True), 1e-12)
            adherence.append(float((f @ text_emb).mean()))          # mean over the 16 frames, projected space
            used += 1

        if used < 2:
            failures.append(f"{model} prompt {prompt_id:03d}: only {used} usable videos")
            continue

        row = {
            "model": model,
            "prompt_id": prompt_id,
            "dino_diversity": diversity(np.stack(embeds["dino"])),
            "clip_proj_diversity": diversity(np.stack(embeds["clip_proj"])),
            "clip_cls_diversity": diversity(np.stack(embeds["clip_cls"])),
            "num_valid_seeds": used,
            "clip_text_adherence": float(np.mean(adherence)),           # mean over seeds
            "dino_diversity_per_frame": diversity_per_frame(np.stack(frames_by["dino"])),
            "clip_proj_diversity_per_frame": diversity_per_frame(np.stack(frames_by["clip_proj"])),
            "clip_cls_diversity_per_frame": diversity_per_frame(np.stack(frames_by["clip_cls"])),
        }
        rows.append(row)
        if n % 10 == 0 or n == len(groups):
            rate = (time.time() - started) / n
            print(f"  [{n}/{len(groups)}] {model} p{prompt_id:03d} "
                  f"DINO={row['dino_diversity']:.4f} CLIPproj={row['clip_proj_diversity']:.4f} "
                  f"CLIPcls={row['clip_cls_diversity']:.4f} ({rate:.1f}s/group)")

    # ---- per-prompt CSV -----------------------------------------------------
    csv_path = results_root / "diversity_per_prompt.csv"
    with csv_path.open("w") as fh:
        # First five columns keep the originally requested schema; clip_diversity
        # there is the projected-feature reading. The CLS reading and the
        # per-frame variants follow as extra columns.
        fh.write("model,prompt_id,dino_diversity,clip_diversity,num_valid_seeds,"
                 "clip_cls_diversity,clip_text_adherence,"
                 "dino_diversity_per_frame,clip_proj_diversity_per_frame,clip_cls_diversity_per_frame\n")
        for r in sorted(rows, key=lambda r: (r["model"], r["prompt_id"])):
            fh.write(f"{r['model']},{r['prompt_id']},{r['dino_diversity']:.6f},"
                     f"{r['clip_proj_diversity']:.6f},{r['num_valid_seeds']},"
                     f"{r['clip_cls_diversity']:.6f},{r['clip_text_adherence']:.6f},"
                     f"{r['dino_diversity_per_frame']:.6f},{r['clip_proj_diversity_per_frame']:.6f},"
                     f"{r['clip_cls_diversity_per_frame']:.6f}\n")

    # ---- summary ------------------------------------------------------------
    summary: Dict[str, Any] = {
        "protocol": {
            "color_normalize": bool(args.color_normalize),
            "L_videos_per_prompt": args.expected_seeds,
            "frames_sampled_per_video": n_frames,
            "frame_sampling": "np.linspace(0, T-1, 16) rounded -- inclusive of first and last frame",
            "pooling": "mean over the 16 L2-normalised frame embeddings (DP-DMD extractor returns unit vectors)",
            "formula": "D = 1 - (2/(L(L-1))) * sum_{i<j} cos(x_i, x_j)",
            "pairs_per_prompt": args.expected_seeds * (args.expected_seeds - 1) // 2,
        },
        "reproduction_choices": {
            "note": ("The DFD paper names architectures ('DINOv2-ViT-Large', 'CLIP-ViT-Large') but pins "
                     "neither checkpoints nor the CLIP feature layer; its evaluation code is not released. "
                     "Both CLIP readings (projected image_embeds; pre-projection CLS as in DP-DMD's public "
                     "code, which the paper cites for the metric) are reported. Neither is 'the' reproduction. "
                     "DINO is the same tensor under every reading."),
            **enc.info,
        },
        "bootstrap": {"n_samples": int(div_cfg.get("bootstrap_samples", 2000)),
                      "seed": int(div_cfg.get("bootstrap_seed", 0)),
                      "method": "percentile bootstrap over per-prompt values"},
        "models": {},
        "failures": failures,
    }
    for model in sorted({r["model"] for r in rows}):
        vals = [r for r in rows if r["model"] == model]
        summary["models"][model] = {
            "n_prompts": len(vals),
            "dino": summarise(np.array([r["dino_diversity"] for r in vals]),
                              int(div_cfg.get("bootstrap_samples", 2000)),
                              int(div_cfg.get("bootstrap_seed", 0))),
            "clip_proj": summarise(np.array([r["clip_proj_diversity"] for r in vals]),
                                   int(div_cfg.get("bootstrap_samples", 2000)), int(div_cfg.get("bootstrap_seed", 0))),
            "clip_cls": summarise(np.array([r["clip_cls_diversity"] for r in vals]),
                                  int(div_cfg.get("bootstrap_samples", 2000)), int(div_cfg.get("bootstrap_seed", 0))),
            "dino_per_frame": summarise(np.array([r["dino_diversity_per_frame"] for r in vals]),
                                        int(div_cfg.get("bootstrap_samples", 2000)), int(div_cfg.get("bootstrap_seed", 0))),
            "clip_proj_per_frame": summarise(np.array([r["clip_proj_diversity_per_frame"] for r in vals]),
                                             int(div_cfg.get("bootstrap_samples", 2000)), int(div_cfg.get("bootstrap_seed", 0))),
            "clip_cls_per_frame": summarise(np.array([r["clip_cls_diversity_per_frame"] for r in vals]),
                                            int(div_cfg.get("bootstrap_samples", 2000)), int(div_cfg.get("bootstrap_seed", 0))),
            "clip_text_adherence": summarise(np.array([r["clip_text_adherence"] for r in vals]),
                                             int(div_cfg.get("bootstrap_samples", 2000)),
                                             int(div_cfg.get("bootstrap_seed", 0))),
        }

    (results_root / "diversity_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- markdown -----------------------------------------------------------
    lines = ["# Visual diversity (DINO / CLIP)", "",
             "Higher is more diverse. `D = 1 - mean pairwise cosine` over "
             f"C({args.expected_seeds},2) = {summary['protocol']['pairs_per_prompt']} seed pairs per prompt.", "",
             "| model | n | DINO (mean) | CLIP proj (mean) | CLIP cls (mean) | DINO (per-frame) | CLIP proj (pf) | CLIP cls (pf) | text adherence |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model, s in summary["models"].items():
        def fmt(m):
            ci = f" [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}]" if m.get("ci95") else ""
            return f"{m['mean']:.4f} ± {m['std']:.4f}{ci}"
        lines.append(f"| {model} | {s['n_prompts']} | {fmt(s['dino'])} | {fmt(s['clip_proj'])} | {fmt(s['clip_cls'])} | "
                     f"{fmt(s['dino_per_frame'])} | {fmt(s['clip_proj_per_frame'])} | {fmt(s['clip_cls_per_frame'])} | "
                     f"{fmt(s['clip_text_adherence'])} |")
    lines += ["", "Values are mean ± sample std over prompts, with a percentile bootstrap 95% CI.", "",
              "## Reproduction choices", "",
              summary["reproduction_choices"]["note"], "",
              f"- DINO: `{enc.info['dino_model']}`, feature = {enc.info['dino_feature']}",
              f"- CLIP proj: `{enc.info['clip_model']}`, feature = {enc.info['clip_proj_feature']}",
              f"- CLIP cls: `{enc.info['clip_model']}`, feature = {enc.info['clip_cls_feature']}",
              f"- Frames: {n_frames} evenly spaced, {summary['protocol']['pooling']}", ""]
    if failures:
        lines += ["## Failures", ""] + [f"- {f}" for f in failures] + [""]
    (results_root / "diversity_summary.md").write_text("\n".join(lines))

    print(f"\nwrote {csv_path}")
    print(f"wrote {results_root / 'diversity_summary.json'}")
    print(f"wrote {results_root / 'diversity_summary.md'}")
    for model, s in summary["models"].items():
        print(f"  {model:12s} n={s['n_prompts']:3d}  DINO={s['dino']['mean']:.4f}  "
              f"CLIPproj={s['clip_proj']['mean']:.4f}  CLIPcls={s['clip_cls']['mean']:.4f}  "
              f"DINO_pf={s['dino_per_frame']['mean']:.4f}  adherence={s['clip_text_adherence']['mean']:.4f}")
    if failures:
        print(f"\n{len(failures)} failure(s) recorded in the summary")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
