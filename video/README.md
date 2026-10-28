# ReSeed — video (Wan2.1-T2V-1.3B)

ReSeed for the **DFD / DMD2 four-step Wan2.1-T2V-1.3B** student. This code
integrates with a **Data-Forcing Distillation (DFD) / fastgen** checkout.
It trains zero-initialized per-block residual biases to match the teacher's
average velocity over the first interval. The student weights remain frozen,
and the biases are active only during the first step.

[Paper](https://arxiv.org/abs/2610.37110) · [Project page](https://twwinde.github.io/reseed-project-page/)

DFD and fastgen are not included. You need a working DFD checkout
(`csy2077/data-forcing-distillation`) with the
`fastgen` runtime and the released Wan2.1 DMD2 checkpoint (`dmd2_t2v_vipe/0001000`).

The adapter has 46,080 parameters (0.00325% of the denoiser). At 300 training
updates, video-level DINO diversity increases from 0.337 to 0.429 on the
320-prompt evaluation bank, recovering 56% of the gap to the teacher.
The paper reports the accompanying quality metrics and an inference latency
increase of 0.14% under its timing setup.

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
│   ├── metrics/
│   │   ├── diversity_math.py     video/per-frame 1 - mean pairwise cosine
│   │   ├── eval_feature_diversity.py  DINOv2 / CLIP diversity over a results tree
│   │   └── test_diversity_math.py
│   ├── tools/                    eval config loader, video probe, seed grid
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

# 2. add ReSeed support to the inference entrypoint
cd $DFD && git apply /path/to/video/patches/video_model_inference.reseed.patch
```

The patch adds optional adapter loading to `scripts/inference/video_model_inference.py`:
when `FASTGEN_RESEED_ADAPTER=<checkpoint dir>` is set, it loads the adapter and
installs a first-step gate. Loading is RNG-free (`torch.zeros`, `load_state_dict`,
hook registration), preserving the initial-noise draw for matched comparisons.
Without the variable, inference uses the original student.

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

The paper evaluates **checkpoint-300**, which contains `residual_biases.pt`.
The example above saves intermediate checkpoints up to 2,000 updates; use
`--max-steps 300` to stop at the reported checkpoint.

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

