# ReSeed

**A First-Step Bias Restores Lost Diversity in Diffusion Model Distillation**

Wenwu Tang, Dong Wang, Yun Cheng, and Olga Saukh

[Paper](https://arxiv.org/abs/2610.37110) · [Project page](https://twwinde.github.io/reseed-project-page/) · [Image models](image/README.md) · [Video model](video/README.md)

ReSeed improves diversity in a frozen, few-step distilled model by training
small residual biases that are active only in the first denoising step.
The biases align the student's first-step velocity with the teacher's average
velocity over the same interval. The original student weights and all later
steps remain unchanged. Training requires prompt–noise pairs, but no target
images, perceptual losses, or adversarial losses.

<p align="center">
  <img src="assets/teaser_diversity.png" width="100%"
       alt="SD3.5-Medium samples from the same prompt and 25 matched initial noises: teacher, four-step DMD student, and DMD with ReSeed.">
</p>
<p align="center"><em>SD3.5-Medium with the same prompt and 25 matched initial noises.
Left: teacher. Middle: four-step DMD student. Right: DMD + ReSeed.</em></p>

## Method

Our SD3-Medium experiments test where diversity can be recovered after
distillation. Replacing the first student step with the teacher's trajectory
over the same interval restores much of the output diversity, even when the
final three student steps stay unchanged. This motivates a correction applied
only at the first step.

ReSeed adds a zero-initialized, per-channel residual bias to each selected
transformer block's output. The biases are shared across prompts and noise
seeds. We fit them to the teacher's first-interval velocity while keeping the
teacher and student backbones frozen. At inference, the biases are enabled
only for the first step and require no extra denoising evaluations.

We evaluate ReSeed on SD3-Medium, SD3.5-Medium, SenseFlow-FLUX, and
Wan2.1-T2V-1.3B. For SenseFlow-FLUX, the adapter adds 0.00196% trainable
parameters relative to the student denoiser and improves DINO diversity by
48.3%. See the [paper](https://arxiv.org/abs/2610.37110) for evaluation details
and the quality–diversity trade-off.

## Examples

### Text-to-image: SenseFlow-FLUX

<p align="center">
  <img src="assets/flux_diversity_grid.jpg" width="100%"
       alt="Nine matched seeds for a cat resting on an open laptop computer. Left: FLUX.1-dev teacher. Middle: SenseFlow-FLUX. Right: SenseFlow-FLUX with ReSeed.">
</p>
<p align="center"><em>“A cat resting on an open laptop computer,” with nine matched seeds.
Left: FLUX.1-dev teacher. Middle: SenseFlow-FLUX. Right: SenseFlow-FLUX + ReSeed.</em></p>

### Text-to-video: Wan2.1-T2V-1.3B

Each row uses the same prompt with different noise seeds. Corresponding
columns use the same initial noise across methods. The top, middle, and
bottom rows show the teacher, the four-step DMD2 student, and DMD2 + ReSeed.

<p align="center">
  <img src="assets/reseed_video_diversity_latte.gif" width="100%"
       alt="Wan2.1 latte-art videos across matched seeds. Top: teacher. Middle: DMD2. Bottom: DMD2 with ReSeed.">
</p>
<p align="center"><em>Pouring latte art.</em></p>

<p align="center">
  <img src="assets/reseed_video_diversity_sparkler.gif" width="100%"
       alt="Wan2.1 videos of a child with a sparkler, across matched seeds. Top: teacher. Middle: DMD2. Bottom: DMD2 with ReSeed.">
</p>
<p align="center"><em>A child with a sparkler at night.</em></p>

## Installation and usage

The image code is a Python package. The video code integrates with the
Data-Forcing Distillation / fastgen framework.

| Models | Code | Instructions |
| --- | --- | --- |
| SD3-Medium, SD3.5-Medium, SenseFlow-FLUX | `image/` | [Training, generation, and evaluation](image/README.md) |
| Wan2.1-T2V-1.3B | `video/` | [DFD installation and ReSeed usage](video/README.md) |

To install the image package:

```bash
git clone https://github.com/TWWinde/ReSeed.git
cd ReSeed/image
pip install -e .
python -m reseed --help
```

For image evaluation dependencies, use `pip install -e ".[eval]"`. Follow the
per-model instructions above to prepare teacher and student checkpoints,
train an adapter, and generate matched-noise samples.

This repository contains ReSeed training, inference, and evaluation code;
it does not train the distilled students. Supply the teacher and student
checkpoints separately. Model weights and the full evaluation prompt banks
are not included in this repository.

## Evaluation

Image quality is evaluated on COCO caption banks, and diversity on a shared
100-prompt × 10-seed bank. Evaluation configurations are in
[`image/configs/`](image/configs/README.md). They record prompt-bank hashes,
sampler settings, and model requirements so that comparisons use the same
prompts and initial noises.

Video evaluation uses DINOv2/CLIP diversity and VBench quality metrics.
See [`video/README.md`](video/README.md) for the generation settings and
evaluation commands.

## Citation

```bibtex
@misc{tang2026reseed,
  title={A First-Step Bias Restores Lost Diversity in Diffusion Model Distillation},
  author={Wenwu Tang and Dong Wang and Yun Cheng and Olga Saukh},
  year={2026},
  eprint={2610.37110},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.37110}
}
```

## License and acknowledgments

ReSeed is released under the [MIT License](LICENSE). The CLIP/PickScore
evaluation script is adapted from GAD. The video code requires DFD / fastgen,
which are not included here. See [NOTICE.md](NOTICE.md) for third-party
attribution and model and dataset references. Model weights remain subject
to their respective licenses.
