# ReSeed — video (Wan2.1-T2V-1.3B)

ReSeed for the released **DFD / DMD2 four-step Wan2.1-T2V-1.3B** student. The
method is identical to the image side — a zero-initialised per-block residual
bias, active only on the first student step, fitted so the student's first-step
velocity matches the teacher across the first interval — but the student lives
in the **Data-Forcing-Distillation (DFD) / fastgen** codebase, so this side ships
as a **plug-in** for a DFD checkout rather than a standalone package.

We do **not** vendor DFD or fastgen (they carry their own upstream licenses).
You need a working DFD checkout — `csy2077/data-forcing-distillation` with the
`fastgen` runtime and the released Wan2.1 DMD2 checkpoint (`dmd2_t2v_vipe/0001000`).

Full results and ablations are in [`../docs/RESULTS_video.md`](../docs/RESULTS_video.md).
Headline: +46,080 params (0.00325 % of the denoiser), 300 training steps, raises
video-level DINO diversity from 0.337 to 0.429 on 320 VBench prompts (recovering
56 % of the teacher gap) with text alignment and VBench quality held at teacher
level and +0.14 % inference latency.

## What this directory contains

```
video/
├── evaluation/
│   ├── reseed/
│   │   ├── adapter.py            WanBlockResidualBias + FirstStepGate + load/save
│   │   ├── flow.py               first-interval teacher secant on the Wan sampler
│   │   ├── train_reseed.py       the trainer (torch.distributed, FSDP)
│   │   ├── encode_prompts.py     UMT5 prompt-embedding cache builder
│   │   ├── fetch_vipe_captions.py  ViPE-Wild caption pool
│   │   ├── bench_overhead.py     latency / memory overhead measurement
│   │   ├── plot_training.py      training-curve plot
│   │   ├── test_reseed.py        adapter/flow unit checks
│   │   └── *.sbatch              reference MUSICA launchers
│   ├── metrics/
│   │   ├── diversity_math.py     video/per-frame 1 - mean pairwise cosine
│   │   ├── eval_feature_diversity.py  DINOv2 / CLIP diversity over a results tree
│   │   └── test_diversity_math.py
│   ├── tools/                    eval config loader, video probe, seed grid, report
│   ├── generate_multiseed.py     multi-seed generation driver
│   └── config.yaml               central eval config (paths, arms, sampler)
└── patches/
    └── video_model_inference.reseed.patch   makes the DFD inference entrypoint ReSeed-aware
```

## Install into a DFD checkout

```bash
DFD=/path/to/data-forcing-distillation      # the repo/ that has fastgen/ and scripts/

# 1. copy the ReSeed additions next to DFD's own evaluation code
cp -r video/evaluation/reseed   $DFD/evaluation/
cp -r video/evaluation/metrics  $DFD/evaluation/
cp -r video/evaluation/tools    $DFD/evaluation/     # merge with existing tools/
cp    video/evaluation/generate_multiseed.py $DFD/evaluation/
cp    video/evaluation/config.yaml           $DFD/evaluation/

# 2. make the released inference entrypoint ReSeed-aware (one 45-line change)
cd $DFD && git apply /path/to/video/patches/video_model_inference.reseed.patch
```

The patch adds an **opt-in** block to `scripts/inference/video_model_inference.py`:
when `FASTGEN_RESEED_ADAPTER=<checkpoint dir>` is set, it loads the adapter and
installs a first-step gate. Loading is RNG-free (`torch.zeros`, `load_state_dict`,
hook registration), so the initial noise is bit-identical with or without the
adapter — every comparison stays paired. Unset, the file behaves exactly as
upstream.

## Train

```bash
# one-time: cache UMT5 embeddings for the ViPE caption pool
python evaluation/reseed/encode_prompts.py --output reseed_data/vipe_long/umt5_embeddings.pt

# fit the first-step bias (4 GPUs shown)
python -m torch.distributed.run --standalone --nproc_per_node=4 \
    evaluation/reseed/train_reseed.py \
    --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
    --ckpt_path $DFD_CKPT_ROOT/dmd2_t2v_vipe/0001000 \
    --embeddings reseed_data/vipe_long/umt5_embeddings.pt \
    --output reseed_runs/dmd2_m13 \
    --max-steps 2000 --micro-steps 13 --save-every 100 --seed 42 \
    - trainer.fsdp=True trainer.ddp=False
```

The paper's working point is **checkpoint-300** (`reseed_runs/.../checkpoint-300/`,
which holds `residual_biases.pt`). Diversity gains arrive early; the bias
magnitude keeps growing afterwards, so early stopping matters — see the results
report.

## Generate and score

```bash
# generate a ReSeed arm: switch the adapter on via the env var the patch reads
FASTGEN_RESEED_ADAPTER=reseed_runs/dmd2_m13/checkpoint-300 \
python evaluation/generate_multiseed.py --config evaluation/config.yaml \
    --models dmd2_reseed --seeds 0 1 2 3 4 5 6 7 --output-root evaluation/generated

# video-level + per-frame diversity (DINOv2 / CLIP)
python evaluation/metrics/eval_feature_diversity.py \
    --config evaluation/config.yaml --results-root evaluation/generated --expected-seeds 8
```

`config.yaml` defines the arms (teacher / dmd2 / dfd / dmd2_reseed), the sampler
(`t_list = [0.999, 0.937, 0.833, 0.624, 0]`, no CFG, 81 frames @ 832×480), and
paths. All arms share prompts, seeds, and initial noise; the ReSeed arm differs
from `dmd2` only by the adapter switched on for step 1.

## Reference cluster scripts

`evaluation/reseed/*.sbatch` are the MUSICA launchers (`train_reseed.sbatch`,
`eval_reseed.sbatch`, `bench_overhead.sbatch`, …). `eval_reseed.sbatch` calls a
`run_all.sh` orchestrator that is part of the DFD eval harness (not vendored
here); the self-contained path above (`generate_multiseed.py` +
`eval_feature_diversity.py`) reproduces the same numbers. Adapt `#SBATCH`
headers and the `DFD_ROOT` / `HF_HOME` paths for your site.
