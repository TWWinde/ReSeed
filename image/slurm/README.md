# Reference cluster launchers (MUSICA)

These are the original SLURM scripts used to train and evaluate the three image
models on the MUSICA H100 cluster. They are included for provenance and as
templates, **not** as turnkey scripts: the `#SBATCH` headers (account,
partition, QOS), the `source clusters/musica/env.sh` line, the `$CRRA_PYTHON`
interpreters, and the Hugging Face hub paths are all site-specific. Run them
from the `image/` directory after editing those for your environment. The
portable commands are in [`../README.md`](../README.md); these scripts just wrap
the same `python -m reseed ...` entrypoints for a scheduler.

| script | what it does |
|---|---|
| `sd3m1024_first_interval_residual_bias_train.sbatch` | train the SD3-Medium bias adapter |
| `sd3m1024_first_interval_residual_bias_seed_train.sbatch` | same, multi-seed |
| `flux_first_interval_residual_bias_cache.sbatch` | build the FLUX teacher-target cache |
| `flux_first_interval_residual_bias_train.sbatch` | train the SenseFlow-FLUX bias adapter |
| `submit_residual_bias_multiseed_coco5k.sh` | orchestrate the 3-seed training sweep |
| `submit_first_interval_training_length_coco5k.sh` | training-length sweep |
| `submit_senseflow_flux_residual_bias_training_length.sh` | FLUX training-length sweep |
| `flux_baseline_quality_diversity.sbatch` | generate + score FLUX (quality + diversity) |
| `first_interval_training_length_coco5k_eval.sbatch` | evaluate checkpoints on COCO-5k |
| `first_interval_training_length_coco5k_summary.sbatch` | summarise the sweep |
| `sd35m1024_portrait_teacher60_dmd.sbatch` | Figure 1 portrait panels (60-step teacher) |

Module and dispatcher names were rewritten from the internal `unified_distill`
package to the `reseed` CLI shipped here; the SLURM/env scaffolding was left as
found.
