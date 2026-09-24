#!/bin/bash
# Submit seed 43/44 training-length studies for SD3M, SD3.5M, and SenseFlow.
# Every atomic 100-step checkpoint immediately dispatches one full-node job
# containing matched COCO5K quality followed by 100x10 diversity evaluation.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/fs201205/wt93384/project/CRRA}"
cd "$PROJECT_ROOT"
export CRRA_ROOT="$PROJECT_ROOT"
source clusters/musica/env.sh

HUB="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
FLUX_DEV="${CRRA_FLUX_DEV_MODEL:-$HUB/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21}"
SENSEFLOW="${CRRA_SENSEFLOW_FLUX_TRANSFORMER:-/data/fs201205/wt93384/models/SenseFlow/SenseFlow-FLUX}"
SD3_STUDENT="$PROJECT_ROOT/outputs/musica_sd3m1024_tdm_teacher_cfg_scale_v1/sd3m_tdm_teacher_cfg_w7_1024/seed_42/checkpoint-2000"
SD35_STUDENT="$PROJECT_ROOT/outputs/musica_sd35m1024_tdm_teacher_cfg_scale_v1/sd35m_tdm_teacher_cfg_w4p5/seed_42/checkpoint-2000"
ROOT_TAG="${ROOT_TAG:-first_interval_residual_bias_training_length_multiseed_coco5k_v1}"
HOOK="$PROJECT_ROOT/tools/submit_first_interval_training_length_eval.py"

for path in \
  "$FLUX_DEV/model_index.json" \
  "$SENSEFLOW/config.json" \
  "$SD3_STUDENT/manifest.json" \
  "$SD35_STUDENT/manifest.json" \
  "$CRRA_COCO5K_PROMPTS"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done

for seed in 43 44; do
  STUDY_ROOT="$PROJECT_ROOT/evaluations/$ROOT_TAG/seed_$seed"
  TRAIN_ROOT="$PROJECT_ROOT/outputs/$ROOT_TAG/seed_$seed"
  mkdir -p "$STUDY_ROOT/submission_ledgers" "$TRAIN_ROOT" "$PROJECT_ROOT/slurm"
  COMMON="MAX_STEPS=2000,SAVE_EVERY=100,KEEP_LAST=20,SEED=$seed,CHECKPOINT_HOOK=$HOOK,TRAINING_LENGTH_ROOT=$STUDY_ROOT,TRAINING_LENGTH_QUALITY_DATASET=coco5k"

  SD3_JOB=$(sbatch --parsable --job-name="tl-sd3-s${seed}" \
    --export="ALL,$COMMON,TRAINING_LENGTH_FAMILY=sd3m,STUDENT_CHECKPOINT=$SD3_STUDENT,OUTPUT=$TRAIN_ROOT/sd3m" \
    clusters/musica/slurm/sd3m1024_first_interval_residual_bias_train.sbatch)

  SD35_JOB=$(sbatch --parsable --job-name="tl-s35-s${seed}" \
    --export="ALL,$COMMON,TRAINING_LENGTH_FAMILY=sd35m,MODEL=$CRRA_SD35M_MODEL,TEACHER_CFG=4.5,STUDENT_CHECKPOINT=$SD35_STUDENT,OUTPUT=$TRAIN_ROOT/sd35m" \
    clusters/musica/slurm/sd3m1024_first_interval_residual_bias_train.sbatch)

  CACHE="$TRAIN_ROOT/senseflow_teacher_cache"
  SF_OUTPUT="$TRAIN_ROOT/senseflow"
  SF_CACHE_JOB=$(sbatch --parsable --job-name="tl-sf-cache-s${seed}" \
    --export="ALL,SEED=$seed,MAX_STEPS=2000,CACHE=$CACHE,STUDENT_COMPONENTS_MODEL=$FLUX_DEV" \
    clusters/musica/slurm/flux_first_interval_residual_bias_cache.sbatch)
  SF_TRAIN_JOB=$(sbatch --parsable --job-name="tl-sf-s${seed}" \
    --dependency="afterok:$SF_CACHE_JOB" \
    --export="ALL,$COMMON,TRAINING_LENGTH_FAMILY=senseflow,CACHE=$CACHE,OUTPUT=$SF_OUTPUT,STUDENT_COMPONENTS_MODEL=$FLUX_DEV,STUDENT_TRANSFORMER_MODEL=$SENSEFLOW,STUDENT_MAX_SEQUENCE_LENGTH=512" \
    clusters/musica/slurm/flux_first_interval_residual_bias_train.sbatch)

  "$CRRA_PYTHON" - "$STUDY_ROOT/submission.json" "$seed" \
    "$SD3_JOB" "$SD35_JOB" "$SF_CACHE_JOB" "$SF_TRAIN_JOB" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "protocol": "first_interval_residual_bias_training_length_multiseed_coco5k_v1",
    "seed": int(sys.argv[2]),
    "max_steps": 2000,
    "save_every": 100,
    "quality": "COCO5K, one matched image per prompt",
    "diversity": "shared 100 prompts x 10 seeds bank",
    "training_jobs": {
        "sd3m": sys.argv[3],
        "sd35m": sys.argv[4],
        "senseflow_cache": sys.argv[5],
        "senseflow": sys.argv[6],
    },
}, indent=2) + "\n", encoding="utf-8")
PY
  printf 'seed=%s sd3m=%s sd35m=%s senseflow_cache=%s senseflow=%s\n' \
    "$seed" "$SD3_JOB" "$SD35_JOB" "$SF_CACHE_JOB" "$SF_TRAIN_JOB"
done
