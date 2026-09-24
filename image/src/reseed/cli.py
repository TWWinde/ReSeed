#!/usr/bin/env python3
"""Thin dispatcher over the ReSeed training, generation, and scoring scripts.

Every stage is also runnable on its own (``python -m reseed.evaluation.sd3m_generate``
or ``python image/tools/flux_generate.py``); this CLI only gives them a single,
uniform front door so launch scripts and the README read the same way. Dispatch
is lazy: no heavy dependency (torch, diffusers) is imported until the chosen
stage runs, so ``python -m reseed --help`` is instant.

Examples
--------
  # train the SD3-Medium first-step bias adapter (single process)
  python -m reseed train sd3m --model $TEACHER --student-checkpoint $STUDENT \
      --prompt-file prompts.txt --output runs/sd3m_seed42

  # generate SD3.5-Medium images with a trained adapter
  python -m reseed generate sd35m --mode sd35m-dmd --model $MODEL \
      --checkpoint $STUDENT --first-interval-residual-bias-checkpoint $BIAS ...

  # score quality (HPSv2.1 / LAION-AeS / FID) and diversity (DINOv3 / CLIP / LPIPS)
  python -m reseed score quality suite --manifest .../generation_manifest.jsonl ...
  python -m reseed diversity feature score --manifest ... --output ...
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# module entrypoints inside the installed package (run via runpy as __main__)
_MODULES = {
    ("train", "sd3m"): "reseed.trainers.sd3m_first_interval_residual_bias",
    ("train", "flux"): "reseed.trainers.flux_first_interval_residual_bias",
    ("generate", "sd3m"): "reseed.evaluation.sd3m_generate",
    ("generate", "sd35m"): "reseed.evaluation.sd35m_generate",
    ("score", "quality"): "reseed.evaluation.robust_metrics",
    ("score", "antithetic"): "reseed.evaluation.antithetic_noise",
}
# script entrypoints under <project_root>/tools (run via runpy on their path)
_SCRIPTS = {
    ("generate", "flux"): "flux_generate.py",
    ("score", "clip-pickscore"): "score_clip_pickscore.py",
    ("diversity", "feature"): "feature_diversity_eval.py",
    ("diversity", "lpips"): "prompt_lpips_diversity.py",
    ("diversity", "summarize"): "summarize_feature_diversity.py",
}

_GROUPS = {
    "train": ("sd3m", "flux"),
    "generate": ("sd3m", "sd35m", "flux"),
    "score": ("quality", "clip-pickscore", "antithetic"),
    "diversity": ("feature", "lpips", "summarize"),
}


def _tools_dir() -> Path:
    from reseed.paths import project_root

    return project_root() / "tools"


def _usage() -> str:
    lines = ["usage: python -m reseed <group> <target> [args...]", "", "groups:"]
    for group, targets in _GROUPS.items():
        lines.append(f"  {group:<10} {' | '.join(targets)}")
    lines.append("")
    lines.append("run `python -m reseed <group> <target> --help` for a stage's own flags")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(_usage())
        return 0 if argv else 2
    if len(argv) < 2:
        print(_usage(), file=sys.stderr)
        return 2
    group, target, rest = argv[0], argv[1], argv[2:]
    key = (group, target)
    if key in _MODULES:
        module = _MODULES[key]
        sys.argv = [module, *rest]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    if key in _SCRIPTS:
        path = _tools_dir() / _SCRIPTS[key]
        if not path.is_file():
            print(f"missing script: {path}", file=sys.stderr)
            return 2
        sys.argv = [str(path), *rest]
        runpy.run_path(str(path), run_name="__main__")
        return 0
    print(f"unknown command: {group} {target}\n\n{_usage()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
