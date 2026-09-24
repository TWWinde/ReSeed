#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Generate the multi-seed evaluation set for DMD2 and DFD.

WHY THIS SHELLS OUT INSTEAD OF LOOPING IN-PROCESS
-------------------------------------------------
The obvious implementation -- load each model once, then loop over prompts and
seeds calling generator_fn -- would be faster and would allow per-video resume.
It would also silently change the noise.

video_model_inference.py seeds the RNG, then *builds the model*, and only then
draws the noise. Model construction consumes RNG: from_pretrained instantiates
the network from its config (randomly initialising every layer) before
overwriting it with the checkpoint, and DMD2Model builds three Wan transformers
plus a discriminator. Any reordering shifts the RNG stream and the noise with
it. The smoke test's bit-identical-noise guarantee between the two arms holds
precisely because both go through the same construction path.

So: one subprocess per (model, seed), through the official entry point, with
only --ckpt_path differing between arms.

CONSEQUENCE FOR RESUME
----------------------
The official script has no notion of "skip prompts already done" -- one run
emits every prompt for one seed. Resume therefore works at (model, seed)
granularity: a pair whose videos all exist and validate is skipped entirely;
otherwise the pair is regenerated. Generation is deterministic, so regenerating
reproduces the same bytes for the videos that were already there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.tools.eval_config import load_config, model_spec, parse_resolution, resolve  # noqa: E402
from evaluation.tools.video_probe import validate_video  # noqa: E402

SAMPLING_TIME = re.compile(r"Student sampling time:\s*([0-9.]+)s")
GENERATING = re.compile(r"\[(\d+)/(\d+)\]\s+Generating:")


def resolve_python(cfg: Dict[str, Any]) -> str:
    """Interpreter for the inference subprocess.

    Falls back to the interpreter running this script: a bare "python" is not
    on PATH on the compute nodes, and a missing one used to surface as a bare
    FileNotFoundError from subprocess with no hint about which setting was wrong.
    """
    configured = (cfg.get("paths", {}) or {}).get("python") or ""
    if configured:
        if shutil.which(configured) or Path(configured).is_file():
            return configured
        print(f"  WARNING: configured python {configured!r} not executable; "
              f"falling back to {sys.executable}")
    return sys.executable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts(path: Path, dedup: bool, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Return [{prompt_id, prompt, source_line}] after optional dedup/limit."""
    raw = [(i + 1, ln.strip()) for i, ln in enumerate(path.read_text().splitlines())]
    entries = [(n, t) for n, t in raw if t]

    if dedup:
        seen: Dict[str, int] = {}
        kept = []
        for line_no, text in entries:
            if text in seen:
                continue
            seen[text] = line_no
            kept.append((line_no, text))
        dropped = len(entries) - len(kept)
        if dropped:
            print(f"  dedup: {len(entries)} lines -> {len(kept)} unique ({dropped} duplicates dropped)")
        entries = kept

    if limit is not None:
        entries = entries[:limit]

    return [{"prompt_id": i, "prompt": t, "source_line": n} for i, (n, t) in enumerate(entries)]


def video_path(out_root: Path, model: str, prompt_id: int, seed: int) -> Path:
    return out_root / model / f"prompt_{prompt_id:03d}" / f"seed_{seed:03d}.mp4"


def pair_is_complete(out_root: Path, model: str, seed: int, prompts: List[Dict[str, Any]],
                     width: int, height: int, frames: int, fps: float) -> bool:
    for p in prompts:
        ok, _ = validate_video(video_path(out_root, model, p["prompt_id"], seed),
                               width, height, frames, fps)
        if not ok:
            return False
    return True


def run_one_pair(cfg: Dict[str, Any], model: str, seed: int, prompt_file: Path,
                 num_prompts: int, work_dir: Path, log_path: Path) -> tuple[int, str]:
    """Invoke the official inference script once. Returns (returncode, log text)."""
    gen = cfg["generation"]
    spec = model_spec(cfg, model)
    mode = spec.get("mode", "student")
    ckpt = resolve(cfg, cfg["paths"]["checkpoint_root"]) / spec["checkpoint"] if spec.get("checkpoint") else None

    # fastgen's import_config_from_python_file turns the --config path into a
    # module name, so an absolute path becomes a leading-dot relative import
    # and dies. It must be relative to the working directory (which is REPO).
    fastgen_cfg = resolve(cfg, gen["fastgen_config"])
    try:
        fastgen_cfg_arg = str(fastgen_cfg.relative_to(REPO))
    except ValueError:
        fastgen_cfg_arg = str(fastgen_cfg)

    if mode == "teacher":
        sampler = ["--do_student_sampling", "False", "--do_teacher_sampling", "True",
                   "--num_steps", str(int(spec["num_steps"]))]
    else:
        sampler = ["--do_student_sampling", "True", "--do_teacher_sampling", "False"]
    cmd = [
        resolve_python(cfg),
        str(resolve(cfg, gen["inference_script"])),
        *(["--ckpt_path", str(ckpt)] if ckpt else []),
        *sampler,
        "--prompt_file", str(prompt_file),
        "--config", fastgen_cfg_arg,
        "--save_as_gif", "False",
        "--video_save_dir", str(work_dir),
        "-",
        f"trainer.seed={seed}",
        "trainer.fsdp=True",
        "trainer.ddp=False",
        f"log_config.name=gen_{model}_seed{seed}",
    ]

    adapter = model_spec(cfg, model).get("reseed_adapter") or ""
    env = {
        **dict(os.environ),
        **({"FASTGEN_RESEED_ADAPTER": str(resolve(cfg, adapter))} if adapter else {}),
        "PYTHONPATH": str(REPO),
        "HF_HOME": str(resolve(cfg, cfg["paths"]["hf_home"])),
        "FASTGEN_OUTPUT_ROOT": str(resolve(cfg, cfg["paths"]["fastgen_output_root"])),
        "HF_HUB_DISABLE_XET": "1",
        "LOCAL_FILES_ONLY": "true",
        # Single process, rank 0 -> effective seed == trainer.seed.
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        # Port per (model, seed, prompt shard): shards of one seed can share a node.
        "MASTER_PORT": str(29500 + int(hashlib.sha1(f"{model}:{seed}:{prompt_file.name}".encode()).hexdigest(), 16) % 2000),
    }

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
    log = proc.stdout + proc.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log)
    print(f"    subprocess rc={proc.returncode} in {time.time() - started:.1f}s -> {log_path.name}")
    return proc.returncode, log


def parse_runtimes(log: str, n: int) -> List[Optional[float]]:
    """Per-video student sampling times, in emission order."""
    times = [float(m) for m in SAMPLING_TIME.findall(log)]
    return [times[i] if i < len(times) else None for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "evaluation/config.yaml"))
    ap.add_argument("--models", default=None, help="comma-separated subset, default all")
    ap.add_argument("--seeds", default=None, help="comma-separated subset, default from config")
    ap.add_argument("--num-prompts", type=int, default=None, help="use only the first N prompts")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--force", action="store_true", help="regenerate even if valid videos exist")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--prompt-range", default=None,
                    help="A:B -- generate only prompts [A, B) of the (deduplicated) prompt list, keeping their "
                         "prompt_ids. For sharding one (model, seed) over several GPUs/nodes; the noise is drawn "
                         "per run from the seed alone, so shards of one seed share the same noise. Requires "
                         "--no-metadata; a full finalize pass rebuilds metadata.jsonl afterwards.")
    ap.add_argument("--no-metadata", action="store_true",
                    help="generate only; do not rewrite metadata.jsonl (for parallel shard workers -- "
                         "run once more without this flag afterwards to rebuild metadata from disk)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gen = cfg["generation"]

    explicit = bool(args.models)
    models = args.models.split(",") if explicit else list(cfg["models"])
    # ReSeed arms need an adapter. Implicit runs skip an unconfigured arm with
    # a note; an explicit request for one must fail rather than quietly
    # generating plain DMD2 under the ReSeed name.
    kept = []
    for m in models:
        try:
            spec = model_spec(cfg, m)
        except KeyError:
            spec = {}
        if spec.get("opt_in") and not explicit:
            print(f"  note: skipping {m!r} -- opt-in arm, request it with --models")
            continue
        if "reseed_adapter" in spec:
            path = spec.get("reseed_adapter") or ""
            ok = bool(path) and (resolve(cfg, path) / "residual_biases.pt").is_file()
            if not ok:
                if explicit:
                    print(f"ERROR: model {m!r} needs reseed_adapter but got {path!r} "
                          f"(set DFD_RESEED_ADAPTER to a checkpoint-N dir)", file=sys.stderr)
                    return 2
                print(f"  note: skipping {m!r} -- no ReSeed adapter configured")
                continue
        kept.append(m)
    models = kept
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(gen["seeds"])
    num_prompts = args.num_prompts if args.num_prompts is not None else gen.get("num_prompts")
    prompt_file = resolve(cfg, args.prompt_file or gen["prompt_file"])
    out_root = resolve(cfg, args.output_root or cfg["output_root"])
    width, height = parse_resolution(gen["resolution"])
    frames, fps = int(gen["num_frames"]), float(gen["fps"])

    for m in models:
        try:
            model_spec(cfg, m)
        except KeyError:
            print(f"ERROR: unknown model {m!r} (not configured, and no configured name is a prefix)", file=sys.stderr)
            return 2

    prompts = load_prompts(prompt_file, bool(gen.get("dedup_prompts", True)), num_prompts)
    if not prompts:
        print("ERROR: no prompts", file=sys.stderr)
        return 2
    prange = None
    if args.prompt_range:
        if not args.no_metadata:
            print("ERROR: --prompt-range requires --no-metadata (metadata must be rebuilt by a full pass)", file=sys.stderr)
            return 2
        a, b = (int(v) for v in args.prompt_range.split(":"))
        if not (0 <= a < b <= len(prompts)):
            print(f"ERROR: --prompt-range {args.prompt_range} outside [0, {len(prompts)}]", file=sys.stderr)
            return 2
        prange = (a, b)

    out_root.mkdir(parents=True, exist_ok=True)
    used_prompts = out_root / "prompts_used.txt"
    # Atomic: another generate process (e.g. a teacher shard) may be launching
    # a subprocess that reads this file at the same instant. The temp name must
    # be per-process too: four shard workers sharing one ".tmp" raced each
    # other's rename and three of them died with FileNotFoundError.
    _tmp = used_prompts.with_name(f".prompts_used.{os.getpid()}.tmp")
    _tmp.write_text("\n".join(p["prompt"] for p in prompts) + "\n")
    _tmp.replace(used_prompts)
    prompts_sha = sha256_file(used_prompts)
    # Shard: the official script consumes a prompt file and names outputs by
    # position in it, so a shard gets its own file holding prompts [A, B) and
    # the position -> prompt_id mapping is offset by A below.
    run_prompts, run_prompt_file, offset = prompts, used_prompts, 0
    if prange is not None:
        run_prompts = prompts[prange[0]:prange[1]]
        run_prompt_file = out_root / f".prompts_shard_{prange[0]:04d}_{prange[1]:04d}.txt"
        run_prompt_file.write_text("\n".join(p["prompt"] for p in run_prompts) + "\n")
        offset = prange[0]

    total = len(models) * len(seeds) * len(run_prompts)
    print(f"\nplan: {len(models)} models x {len(seeds)} seeds x {len(run_prompts)} prompts = {total} videos"
          + (f"  (shard: prompts [{prange[0]}, {prange[1]}) of {len(prompts)})" if prange else ""))
    print(f"  models      : {models}")
    print(f"  seeds       : {seeds}")
    print(f"  prompts     : {prompt_file.name} -> {len(prompts)} used, sha256 {prompts_sha[:16]}")
    print(f"  geometry    : {width}x{height}, {frames} frames @ {fps} fps, {gen['num_steps']} steps")
    print(f"  output      : {out_root}")
    if args.dry_run:
        for p in prompts[:5]:
            print(f"    [{p['prompt_id']:03d}] (line {p['source_line']}) {p['prompt'][:70]}")
        return 0

    meta_path = out_root / "metadata.jsonl"
    # Records for the models in THIS run are rebuilt from what is on disk;
    # records for other models are carried over untouched. Without the
    # carry-over, adding an arm later (e.g. dmd2_reseed) would silently drop
    # the earlier arms from the metadata and every downstream metric.
    carried: List[Dict[str, Any]] = []
    if meta_path.is_file():
        for ln in meta_path.read_text().splitlines():
            if ln.strip():
                rec = json.loads(ln)
                if rec.get("model") not in models:
                    carried.append(rec)
        if carried:
            print(f"  carrying over {len(carried)} metadata records for "
                  f"{sorted({r['model'] for r in carried})}")
    records: List[Dict[str, Any]] = []
    failures: List[str] = []

    for model in models:
        spec = model_spec(cfg, model)
        ckpt = resolve(cfg, cfg["paths"]["checkpoint_root"]) / spec["checkpoint"] if spec.get("checkpoint") else None
        mode = spec.get("mode", "student")
        for seed in seeds:
            tag = f"{model}/seed{seed}"
            print(f"\n=== {tag} ===")

            if not args.force and pair_is_complete(out_root, model, seed, run_prompts, width, height, frames, fps):
                print("  already complete and valid -- skipping")
                runtimes: List[Optional[float]] = [None] * len(prompts)
                reused = True
            else:
                shard = f"_p{prange[0]:04d}-{prange[1]:04d}" if prange else ""
                work = out_root / ".work" / f"{model}_seed{seed}{shard}"
                shutil.rmtree(work, ignore_errors=True)
                work.mkdir(parents=True, exist_ok=True)
                log_path = out_root / "logs" / f"{model}_seed{seed}{shard}.log"

                rc, log = run_one_pair(cfg, model, seed, run_prompt_file, len(run_prompts), work, log_path)
                if rc != 0:
                    tail = "\n".join(log.strip().splitlines()[-15:])
                    failures.append(f"{tag}: inference exited {rc}\n{tail}")
                    print(f"  FAILED (rc={rc}); see {log_path}")
                    continue

                runtimes = parse_runtimes(log, len(run_prompts))
                produced = sorted((work / run_prompt_file.stem).glob("*.mp4"))
                if len(produced) != len(run_prompts):
                    failures.append(f"{tag}: produced {len(produced)} videos, expected {len(run_prompts)}")
                    print(f"  FAILED: produced {len(produced)} / {len(run_prompts)}")
                    continue

                for src in produced:
                    idx = int(re.search(r"_(\d{4})_seed", src.name).group(1)) + offset
                    dst = video_path(out_root, model, idx, seed)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                shutil.rmtree(work, ignore_errors=True)
                reused = False
                print(f"  generated {len(produced)} videos")

            for p in run_prompts:
                path = video_path(out_root, model, p["prompt_id"], seed)
                ok, reason = validate_video(path, width, height, frames, fps)
                if not ok:
                    failures.append(f"{tag} prompt {p['prompt_id']:03d}: invalid video ({reason})")
                records.append({
                    "model": model,
                    "base_model": spec["base_model"],
                    "prompt_id": p["prompt_id"],
                    "prompt": p["prompt"],
                    "prompt_source_line": p["source_line"],
                    "prompt_file_sha256": prompts_sha,
                    "seed": seed,
                    "mode": mode,
                    "checkpoint": str(ckpt) if ckpt else None,
                    "checkpoint_step": spec.get("step"),
                    "reseed_adapter": (str(resolve(cfg, spec["reseed_adapter"]))
                                       if spec.get("reseed_adapter") else None),
                    "num_steps": int(spec["num_steps"]) if mode == "teacher" else int(gen["num_steps"]),
                    "guidance_scale": float(spec["guidance_scale"]) if mode == "teacher" else gen["student_cfg"],
                    "sampler": "unipc-cfg" if mode == "teacher" else f"{gen['student_sample_type']}-{gen['num_steps']}step",
                    "resolution": gen["resolution"],
                    "num_frames": frames,
                    "fps": fps,
                    "t_list": gen["t_list"],
                    "student_sample_type": gen["student_sample_type"],
                    "student_cfg": gen["student_cfg"],
                    "precision": gen["precision"],
                    "video_path": str(path.relative_to(REPO)) if path.is_absolute() else str(path),
                    "video_sha256": sha256_file(path) if path.is_file() else None,
                    "runtime_sec": runtimes[p["prompt_id"] - offset] if p["prompt_id"] - offset < len(runtimes) else None,
                    "reused": reused,
                    "valid": ok,
                    "validation": reason,
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })

    if args.no_metadata:
        print(f"\n--no-metadata: leaving {meta_path} untouched ({len(records)} records not written)")
    else:
        with meta_path.open("w") as fh:
            for rec in carried + records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_valid = sum(1 for r in records if r["valid"])
    print(f"\n{'=' * 60}")
    print(f"metadata : {meta_path} ({len(records)} records)")
    print(f"valid    : {n_valid} / {total}")
    times = [r["runtime_sec"] for r in records if r["runtime_sec"]]
    if times:
        print(f"runtime  : {sum(times) / len(times):.2f}s per video (mean of {len(times)} timed)")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        print("\nNothing was skipped silently; rerun to resume.")
        return 1

    print("\nALL VIDEOS GENERATED AND VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
