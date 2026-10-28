# Provenance and third-party notices

This repository contains ReSeed training, evaluation, and inference code for
SD3-Medium, SD3.5-Medium, SenseFlow-FLUX, and Wan2.1-T2V-1.3B.
Teacher and distilled student checkpoints are supplied separately. The code
trains only ReSeed adapters, not the distilled students.

## Origin of the code

- **`image/`** — the `reseed` package, developed from the project's internal
  `unified_distill` package. It contains flow solvers, adapters, trainers, and
  evaluation code. The third-party scorer is listed separately below.
- **`video/`** — the ReSeed additions written on top of the Data-Forcing
  Distillation (DFD) / fastgen stack. This directory is a plug-in: DFD and
  fastgen themselves are **not** vendored here. The one modification to the DFD
  inference entrypoint is provided as a patch (`video/patches/`), not as a copy
  of the upstream file.

## Vendored third-party file

- `image/tools/score_clip_pickscore.py` — the CLIP-score + PickScore evaluator,
  vendored from the **GAD** project (`Hannah1102/GAD`), MIT License,
  Copyright (c) 2026 Hannah Huang. The license text is kept alongside it as
  `image/tools/LICENSE.gad-scorer`. Only the hard-coded PickScore cache path was
  generalised to `${HF_HOME}`.

## Upstream models and datasets (cite in any use)

- SD3-Medium, SD3.5-Medium — Stability AI (Esser et al., 2024).
- FLUX.1-dev — Black Forest Labs (2024); SenseFlow student — Ge et al., 2026.
- Wan2.1-T2V-1.3B — Wan et al., 2025; DMD2 — Yin et al., 2024; DFD — Chen et al., 2026.
- DMD — Yin et al., 2024. GAD — Huang et al., 2026. DP-DMD — Wu et al., 2026.
- Prompts/datasets: JourneyDB (Sun et al., 2023), ViPE-Wild-1M (Huang et al.,
  2025), COCO-2017 (Lin et al., 2014), VBench (Huang et al., 2024).
- Scorers: CLIP (Radford et al., 2021), PickScore (Kirstain et al., 2023),
  HPSv2.1 (Wu et al., 2023), LAION aesthetic predictor (Schuhmann et al., 2022),
  LPIPS (Zhang et al., 2018), DINOv3/DINOv2 (Siméoni et al., 2025; Oquab et al.,
  2024), FID (Heusel et al., 2017).

## License

This repository is released under the MIT License (see [`LICENSE`](LICENSE)),
which is compatible with the MIT-licensed vendored scorer above. When combining
it with other frameworks (DFD/fastgen on the video side, Diffusers on the image
side), also observe their upstream terms and the licenses of any model weights
you download.
