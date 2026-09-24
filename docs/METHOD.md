# ReSeed: a seed-response bias for the first denoising step

ReSeed restores same-prompt diversity in an already-distilled few-step student
without retraining it. It adds a constant, zero-initialised per-channel bias to
each transformer block's output, active **only during the first denoising
step**, and fits that bias so the student's first-step velocity matches the
teacher's velocity across the same interval. Every base weight stays frozen,
and steps 2–4 run the unmodified student.

## 1. Setting

The students are rectified-flow samplers. A state at noise level `σ` is
`x_σ = (1−σ)·x₀ + σ·ε`, a network predicts the velocity `v_θ(x_σ, σ, c)`, and
the clean-latent estimate is

```
x̂₀ = x_σ − σ · v_θ(x_σ, σ, c)
```

Sampling is `K` Euler steps on a decreasing grid. All three backbones use four
student steps with shift 3, giving the boundaries

```
σ = [1.0, 0.8984375, 0.75, 0.5, 0]
```

Because the sampler is deterministic, everything that differs between two
images of the same prompt comes from the initial noise `z`.

## 2. The diagnosis ReSeed is built on

Write the four-step generator as a first step followed by a frozen suffix:

```
F_θ(z, c) = H_θ( R_θ(z, c), c )
```

`R_θ` is the first high-noise step, `H_θ` the remaining three. Two measurements
say the loss of diversity happens in `R_θ`, not in `H_θ`:

- **Teacher-prefix depth.** Replacing the student's first `k` steps with the
  teacher and leaving the rest untouched: DINOv3 same-prompt diversity goes
  `0.2142 → 0.3117` at `k=1`, then flattens (`0.3256` at `k=2`, `0.3207` at
  `k=3`). The first step alone delivers **91.5%** of the gain that replacing
  three steps delivers. Meanwhile CLIPScore does *not* move at `k=1`
  (`34.42 → 34.36`, smaller than the pooled seed standard deviation of `0.126`)
  and only rises at `k≥2`. So the first step is where diversity is decided,
  and it is not simply that more teacher is better on every axis.
- **Per prompt.** With all arms ending in the same frozen suffix, the student
  falls below the teacher-prefix diagonal for **98 of 100** prompts. ReSeed
  lifts 33 of them above it and cuts the mean shortfall `0.100 → 0.019`,
  closing **81%** of the gap without touching the suffix.

This is an oracle intervention, not a sampler: it uses more network calls and a
different solver history.

## 3. The method

### Parameterisation

For SD3 block `b`, add per-channel vectors `β_b ∈ R^1536` to the block output of
both streams:

```
h_b^img ← h_b^img + β_b^img
h_b^ctx ← h_b^ctx + β_b^ctx
```

The bias is broadcast over batch and token dimensions and starts at zero, so an
untrained adapter reproduces the frozen student exactly. The last block carries
only an image bias, giving 24 image and 23 context vectors:

```
N_bias = 72,192 parameters = 0.00356% of the 2,028,328,000-parameter student
```

Unlike BitFit, ReSeed does not unfreeze existing model biases; it adds new
zero-initialised vectors whose activity is gated by the solver interval.

### Objective

```
L_DCB = || v_S(b) − sg(v_T) ||²
```

`v_S(b)` is the student's first-step velocity with the bias active; `v_T` is the
teacher's velocity across the same interval, taken with a stop-gradient.

### Building the teacher target

`v_T` is the **secant** over the first student interval, not a single teacher
call:

```
v_T = (x_end − x₀) / Δσ
```

`x_end` comes from running the teacher for `m` micro-steps on a grid built by
interpolating uniformly in **unshifted base-sigma space** between the student
boundaries. One student interval is exactly ¼ of that space, so `m` micro-steps
reproduce a global `4m`-step teacher exactly:

| backbone | student steps | `m` | equivalent teacher | teacher CFG |
|---|---|---|---|---|
| SD3-Medium | 4 | 7 | 28-step | 7 |
| SD3.5-Medium | 4 | 10 | 40-step | 4.5 |

Getting `m` wrong is silent: the run trains fine and regresses onto the wrong
reference.

## 4. Why a constant bias can help

The bias carries no seed and no prompt information, so it cannot store a
per-sample correction. It works through the frozen network's curvature.

Let `F(z; β)` be the first-step output for noise `z` and
`J_β(z) = ∂F(z; β)/∂β` at `β = 0`. A local expansion gives

```
F(z; β) ≈ F(z; 0) + J_β(z)·β
```

so for two seeds the change in their difference is `[J_β(z_i) − J_β(z_j)]·β`.
That term varies with the seed even though `β` is fixed. Two consequences worth
keeping straight:

- A seed-independent shift **cancels at first order** in every seed difference.
  Whatever diversity it creates comes from curvature, which is why two shifts of
  equal norm can have very different effects, and why the learned direction
  matters rather than just its size.
- The bias does not add seed information; it **repositions** the first-step
  states so the frozen suffix becomes more sensitive to the noise it already
  receives.

## 5. Training recipe

- Student frozen; only `β` is trained. No target images, no perceptual loss, no
  adversarial loss.
- Fewer than **8,000 prompt–noise presentations** (JourneyDB prompts).
- Blocks 0–23, interval 0 only.
- Reference operating point: 2,000 updates (SD3-M reports 1,900; SD3.5-M 2,100).
  Diversity is mostly recovered in the first 300–400 updates and then flattens.

## 6. Inference

The bias is active in step 1 and off in steps 2–4.

| student | before → after (DINOv3) | gain | peak activation | latency overhead |
|---|---|---|---|---|
| SD3-Medium | 0.2142 → 0.2926 | +36.6% | +0.00 MiB | +0.410% |
| SD3.5-Medium | 0.2111 → 0.3106 | +47.2% | +0.00 MiB | +0.194% |
| SenseFlow-FLUX | 0.1575 → 0.2336 | +48.3% | +1.00 MiB | +0.250% |

(These are the fixed-student repair rows; the matched-student rows in the main
table use different checkpoints.)

## 7. Honest limitations

- **Compositional accuracy falls.** On the GenEval 10-seed matrix, every
  diversity-raising method loses compositional accuracy, and ReSeed loses the
  most: on SD3.5-Medium overall `0.6872 → 0.5896`, counting `0.652 → 0.383`,
  colour-attribute `0.539 → 0.410` — measured on the same images that give the
  highest DINOv3 diversity. This is not in the paper and should be confronted
  rather than omitted.
- **The teacher-prefix arms exceed the teacher.** The 28-step SD3-M teacher's
  own same-prompt DINOv3 diversity is `0.2686`, *below* the prefix arms
  (`0.312 / 0.326 / 0.321`). A hybrid of teacher-early and student-late steps is
  off-schedule; more variation there is not automatically better variation, and
  it costs alignment (`34.36` vs the teacher's `35.24`).
- **First-order saturation.** In the frozen-suffix sensitivity test, `‖ΔH‖` was
  constant across ε (96.9 / 97.9 / 99.7) and the per-element perturbation
  (9.8e-5) sits below bf16 resolution, so the absolute gains are not local
  derivatives. The stable ratio (ReSeed/student ≈ 1.20) is still meaningful.
- The suffix is held frozen throughout. Whether ReSeed composes with methods
  that change distillation itself (GAD, DP-DMD) is untested.

## 8. Where things live

| what | path |
|---|---|
| trainer | `src/unified_distill/trainers/sd3m_first_interval_residual_bias.py` |
| adapter | `src/unified_distill/models/sd3_residual_bias.py` |
| flow helpers (`boundary_aligned_teacher_sigmas`) | `src/unified_distill/diffusion/flow_matching.py` |
| training job | `clusters/musica/slurm/sd3m1024_first_interval_residual_bias_train.sbatch` |
| per-checkpoint evaluation | `clusters/musica/slurm/first_interval_training_length_eval.sbatch` |
| teacher-CFG ablation | `clusters/musica/submit_sd3m_reseed_teacher_cfg_ablation.sh` |
| first-interval figures | `tools/plot_first_interval_panels.py`, `tools/plot_first_step_mode_coverage.py` |

Metrics: DINOv3 / CLIP feature cosine distance and LPIPS-Alex over 100 prompts
× 10 seeds (45 pairs per prompt) for diversity; CLIP, PickScore, HPSv2.1,
LAION-AeS and FID for quality.
