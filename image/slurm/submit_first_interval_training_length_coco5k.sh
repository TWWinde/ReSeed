#!/bin/bash
# Submit 40 matched COCO5K jobs, then aggregate completed results and curves.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/data/fs201205/wt93384/project/CRRA}"
cd "$PROJECT_ROOT"
STUDY_ROOT="${TRAINING_LENGTH_ROOT:-$PROJECT_ROOT/evaluations/first_interval_residual_bias_training_length_100step_v1}"
mkdir -p "$PROJECT_ROOT/slurm" "$STUDY_ROOT/summary_coco5k"

ARRAY_JOB=$(sbatch --parsable --array=0-39 \
  --export="ALL,TRAINING_LENGTH_ROOT=$STUDY_ROOT" \
  clusters/musica/slurm/first_interval_training_length_coco5k_eval.sbatch)
SUMMARY_JOB=$(sbatch --parsable --dependency="afterany:$ARRAY_JOB" \
  --export="ALL,TRAINING_LENGTH_ROOT=$STUDY_ROOT" \
  clusters/musica/slurm/first_interval_training_length_coco5k_summary.sbatch)

printf 'coco5k_array=%s\nsummary_job=%s\nstudy_root=%s\n' \
  "$ARRAY_JOB" "$SUMMARY_JOB" "$STUDY_ROOT"
