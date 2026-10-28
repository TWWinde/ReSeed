# ReSeed — T2I models

Training, generation, and evaluation code for **SD3-Medium**,
**SD3.5-Medium**, and **SenseFlow-FLUX**. ReSeed trains per-block residual
biases while keeping the distilled student frozen. The biases are active
only during the first denoising step.

[![arXiv](https://img.shields.io/badge/arXiv-2610.37110-B31B1B?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2610.37110)
[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90-Project_Page-1976D2?style=flat-square)](https://twwinde.github.io/reseed-project-page/)

## Install

```bash
cd image
pip install -e .          # core: torch, diffusers, transformers, safetensors
pip install -e ".[eval]"  # + open-clip, pyiqa, lpips, pytorch-fid for scoring
```

Install `torch` first from https://pytorch.org matching your CUDA build.
HPSv2.1 is installed separately: `pip install git+https://github.com/tgxs002/HPSv2`.

Use the `reseed` CLI (`python -m reseed --help`) or run the corresponding
module or script directly.

```
reseed train    sd3m | flux
reseed generate sd3m | sd35m | flux
reseed score    quality | clip-pickscore | antithetic
reseed diversity feature | lpips | summarize
```

## Package layout (`src/reseed/`)

| path | role |
|---|---|
| `models/sd3.py`, `models/flux.py` | load frozen teacher/student pipelines and predict velocities |
| `models/sd3_residual_bias.py`, `models/flux_residual_bias.py` | the ReSeed adapter: zero-init per-block bias, first-step gate, load/save |
| `models/sd3_contract.py`, `config.py` | pinned SD3.5-Medium model contract (revision + component SHAs) |
| `diffusion/flow_matching.py` | rectified-flow boundaries, teacher interval secant, Euler solver |
| `diffusion/frequency_noise.py`, `diffusion/tvcfg.py` | initial-noise construction and threshold-CFG used by the teacher |
| `trainers/sd3m_first_interval_residual_bias.py` | SD3-M / SD3.5-M bias trainer (first-interval velocity matching) |
| `trainers/flux_first_interval_residual_bias.py` | FLUX bias trainer, with a teacher-target cache phase |
| `trainers/_common.py` | shared distributed / prompt-stream helpers |
| `evaluation/sd3m_generate.py`, `evaluation/sd35m_generate.py` | paired base/student generation under a sha-pinned protocol |
| `evaluation/robust_metrics.py` | HPSv2.1, LAION-AeS, FID |
| `evaluation/antithetic_noise.py` | z / −z first-step response diagnostic |
| `objectives/feature_diversity.py` | feature-diversity objective (research/diagnostic) |
| `cli.py` | the thin dispatcher behind `python -m reseed ...` |

`tools/`: `flux_generate.py` (FLUX generation), `feature_diversity_eval.py`
(DINOv3/CLIP diversity), `prompt_lpips_diversity.py` (LPIPS-Alex diversity),
`summarize_feature_diversity.py`, `score_clip_pickscore.py` (CLIP score +
PickScore, MIT, vendored from GAD), `prune_completed_evaluation_images.py`.

## 1. Train the bias adapter

The adapter matches the student's first-step velocity to the teacher's
average velocity over the first interval, computed with seven Euler microsteps. The
paper uses global batch 4, seeds 42/43/44, and reports checkpoints at 1,900
(SD3-M), 2,100 (SD3.5-M), and 2,000 (FLUX) updates.

**SD3-Medium / SD3.5-Medium** (same trainer; point `--model` and
`--student-checkpoint` at the SD3.5-M teacher/student for SD3.5-M):

```bash
torchrun --standalone --nproc_per_node=4 -m reseed.trainers.sd3m_first_interval_residual_bias \
    --model $TEACHER \
    --student-checkpoint $STUDENT \
    --prompt-file prompts.txt \
    --output runs/sd3m_seed42 \
    --resolution 1024 --max-steps 2000 --batch-size 1 \
    --block-start 0 --block-end 23 --interval-index 0 \
    --teacher-micro-steps 7 --teacher-guidance-scale 7 --teacher-cfg-threshold 0.96 \
    --seed 42 --save-every 100
```

**SenseFlow-FLUX** — build the teacher-target cache once, then train:

```bash
# cache phase (writes teacher first-interval targets for all four ranks)
torchrun --standalone --nproc_per_node=4 -m reseed.trainers.flux_first_interval_residual_bias \
    --teacher-model $FLUX_DEV --student-model $FLUX_DEV \
    --student-transformer-model $SENSEFLOW \
    --prompt-file prompts.txt --teacher-target-cache cache/seed42 \
    --resolution 1024 --max-steps 2000 --batch-size 1 --build-cache-only --seed 42

# train against the cache
torchrun --standalone --nproc_per_node=4 -m reseed.trainers.flux_first_interval_residual_bias \
    --teacher-model $FLUX_DEV --student-model $FLUX_DEV \
    --student-transformer-model $SENSEFLOW \
    --prompt-file prompts.txt --teacher-target-cache cache/seed42 \
    --output runs/flux_seed42 \
    --max-steps 2000 --batch-size 1 --seed 42 --save-every 100
```

Omitting `--double-blocks` and `--single-blocks` selects all blocks. To resume
training, pass `--resume-from-checkpoint <checkpoint-directory>`; `--resume`
only skips existing records during cache construction.
Build and train with the same process count, prompt file, batch size, seed,
and number of steps; cache files are indexed by rank and step.

## 2. Generate with the adapter (first step only)

The generators support teacher and student samples with shared prompts and
initial latents. Set the student's `--mode` (e.g. `sd35m-dmd`) and enable
ReSeed with `--first-interval-residual-bias-checkpoint`; leaving that flag off
reproduces the bare student.

```bash
# SD3.5-Medium DMD student + ReSeed, 4 steps
python -m reseed generate sd35m --mode sd35m-dmd \
    --model $MODEL --checkpoint $STUDENT \
    --first-interval-residual-bias-checkpoint runs/sd35m_seed42/checkpoint-2100 \
    --prompts configs/protocols/coco2017_val5k_1024_sd35m_seeded_v1.json \
    --evaluation-protocol coco2017_val5k_1024_sd35m_seeded_v1 \
    --output out/sd35m_reseed --inference-steps 4 --num-shards 4 --finalize-shards

# SenseFlow-FLUX + ReSeed
python -m reseed generate flux \
    --dev-model $FLUX_DEV --senseflow-transformer $SENSEFLOW \
    --first-interval-residual-bias-checkpoint runs/flux_seed42/checkpoint-2000 \
    --prompts $FLUX_PROMPTS --output out/flux_reseed --steps 4 --samples-per-prompt 10
```

`--mode sd35m-base` (or `sd3m-base`) draws the teacher reference. `--bias-scale`
and `--bias-shuffle-seed` in `flux_generate` support the α·bias / reversed-bias
ablations (Figure 2d).

## 3. Score quality and diversity

```bash
# quality: HPSv2.1 / LAION-AeS / FID
python -m reseed score quality suite \
    --manifest out/sd35m_reseed/generation_manifest.jsonl --output q/sd35m_reseed

# CLIP score + PickScore (MIT GAD scorer)
python -m reseed score clip-pickscore --manifest out/sd35m_reseed/generation_manifest.jsonl --output q/

# diversity: DINOv3 + CLIP (1 - mean pairwise cosine over the 10 seeds)
python -m reseed diversity feature score \
    --manifest out/sd35m_div/generation_manifest.jsonl --output div/sd35m_reseed
# diversity: LPIPS-Alex
python -m reseed diversity lpips --manifest out/sd35m_div/generation_manifest.jsonl --output div/sd35m_reseed
python -m reseed diversity summarize --root div/
```

Diversity uses the 100-prompt × 10-seed bank
(`configs/protocols/sd35m1024_diversity_100x10_v1.json`); quality uses the
5,000- or 1,000-prompt COCO banks. See [`configs/README.md`](configs/README.md).

## Evaluation contracts

Generators enforce, and record in `generation_protocol.json`:

- the protocol file's name and SHA-256, and the prompt-CSV SHA-256;
- for SD3.5-Medium, the pinned model revision and component config SHAs;
- that a student checkpoint's declared reference inference-step count matches
  the protocol's `base_inference_steps`.

A mismatch stops generation with an error.

