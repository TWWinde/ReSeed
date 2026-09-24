#!/bin/bash
# SenseFlow-FLUX first-interval residual-bias study. Teacher endpoints are
# cached first; the training job then submits one COCO5K + diversity job as
# soon as each atomic 100-step checkpoint appears.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/fs201205/wt93384/project/CRRA}"
cd "$PROJECT_ROOT"
export CRRA_ROOT="$PROJECT_ROOT"
source clusters/musica/env.sh

HUB="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
FLUX_DEV="${CRRA_FLUX_DEV_MODEL:-$HUB/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21}"
SENSEFLOW="${CRRA_SENSEFLOW_FLUX_TRANSFORMER:-/data/fs201205/wt93384/models/SenseFlow/SenseFlow-FLUX}"
SEED="${SEED:-42}"
STUDY_ROOT="${TRAINING_LENGTH_ROOT:-$PROJECT_ROOT/evaluations/senseflow_flux_residual_bias_training_length_100step_v1}"
TRAIN_ROOT="${TRAINING_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/senseflow_flux_residual_bias_training_length_100step_v1}"
CACHE="$TRAIN_ROOT/teacher_cache_seed_${SEED}"
OUTPUT="$TRAIN_ROOT/seed_${SEED}"
HOOK="$PROJECT_ROOT/tools/submit_first_interval_training_length_eval.py"

mkdir -p "$STUDY_ROOT/submission_ledgers" "$TRAIN_ROOT" "$PROJECT_ROOT/slurm"
[[ -f "$FLUX_DEV/scheduler/scheduler_config.json" ]] || { echo "missing FLUX.1-dev components: $FLUX_DEV" >&2; exit 2; }
[[ -f "$SENSEFLOW/config.json" ]] || { echo "missing SenseFlow transformer: $SENSEFLOW" >&2; exit 2; }

CACHE_JOB=$(sbatch --parsable --job-name=sf-bias-cache \
  --export="ALL,SEED=$SEED,MAX_STEPS=2000,CACHE=$CACHE,STUDENT_COMPONENTS_MODEL=$FLUX_DEV" \
  clusters/musica/slurm/flux_first_interval_residual_bias_cache.sbatch)

TRAIN_JOB=$(sbatch --parsable --job-name=sf-bias-train --dependency="afterok:$CACHE_JOB" \
  --export="ALL,SEED=$SEED,MAX_STEPS=2000,SAVE_EVERY=100,KEEP_LAST=20,CACHE=$CACHE,OUTPUT=$OUTPUT,STUDENT_COMPONENTS_MODEL=$FLUX_DEV,STUDENT_TRANSFORMER_MODEL=$SENSEFLOW,STUDENT_MAX_SEQUENCE_LENGTH=512,CHECKPOINT_HOOK=$HOOK,TRAINING_LENGTH_FAMILY=senseflow,TRAINING_LENGTH_ROOT=$STUDY_ROOT" \
  clusters/musica/slurm/flux_first_interval_residual_bias_train.sbatch)

printf 'cache_job=%s\ntrain_job=%s\ncache=%s\noutput=%s\nstudy=%s\n' \
  "$CACHE_JOB" "$TRAIN_JOB" "$CACHE" "$OUTPUT" "$STUDY_ROOT"
