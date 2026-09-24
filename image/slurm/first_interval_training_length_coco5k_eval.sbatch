#!/bin/bash
# Re-evaluate the 20 SD3M and 20 SD3.5M residual-bias checkpoints on matched
# COCO5K. Diversity is reused from the completed shared 100x10 bank.
# Submit with --array=0-39; every task receives one full four-H100 node.
#SBATCH --job-name=tl-coco5k
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=/data/fs201205/wt93384/project/CRRA/slurm/training-length-coco5k-%A_%a.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/data/fs201205/wt93384/project/CRRA}"
cd "$PROJECT_ROOT"
export CRRA_ROOT="$PROJECT_ROOT"
source clusters/musica/env.sh
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

INDEX="${SLURM_ARRAY_TASK_ID:?submit as an array}"
STUDY_ROOT="${TRAINING_LENGTH_ROOT:-$PROJECT_ROOT/evaluations/first_interval_residual_bias_training_length_100step_v1}"
TRAIN_ROOT="${TRAINING_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/first_interval_residual_bias_training_length_100step_v1}"
if (( INDEX < 20 )); then
  FAMILY=sd3m
  STEP=$(( (INDEX + 1) * 100 ))
else
  FAMILY=sd35m
  STEP=$(( (INDEX - 19) * 100 ))
fi

BIAS_CHECKPOINT="$TRAIN_ROOT/$FAMILY/checkpoint-$STEP"
EVAL_ROOT="$STUDY_ROOT/evaluations/$FAMILY/checkpoint-$STEP"
[[ -f "$BIAS_CHECKPOINT/manifest.json" ]] || {
  echo "missing atomic checkpoint: $BIAS_CHECKPOINT" >&2
  exit 2
}
mkdir -p "$EVAL_ROOT"

if [[ "$FAMILY" == sd3m ]]; then
  env PHASE=quality \
    EVALUATION_ROOT="$EVAL_ROOT" \
    FIRST_INTERVAL_RESIDUAL_BIAS_CHECKPOINT="$BIAS_CHECKPOINT" \
    QUALITY_PROMPTS="$CRRA_COCO5K_PROMPTS" QUALITY_EXPECTED_COUNT=5000 \
    QUALITY_DIR_NAME=coco5k QUALITY_STAGE_LABEL=coco5k \
    QUALITY_PROTOCOL="sd3m_first_interval_residual_bias_step_${STEP}_coco5k_v1" \
    RUN_LABEL="SD3M residual bias step $STEP COCO5K" \
    bash clusters/musica/slurm/sd3m1024_first_interval_lora_quality_diversity.sbatch
else
  SD35_STUDENT="${SD35_STUDENT_CHECKPOINT:-$PROJECT_ROOT/outputs/musica_sd35m1024_tdm_teacher_cfg_scale_v1/sd35m_tdm_teacher_cfg_w4p5/seed_42/checkpoint-2000}"
  env MODE=tdm CHECKPOINT="$SD35_STUDENT" TEACHER_CFG=4.5 CHECKPOINT_TEACHER_CFG=4.5 \
    EVALUATION_ROOT="$EVAL_ROOT" \
    FIRST_INTERVAL_RESIDUAL_BIAS_CHECKPOINT="$BIAS_CHECKPOINT" \
    COCO_PROMPTS="$CRRA_COCO5K_PROMPTS" COCO_EXPECTED_COUNT=5000 \
    COCO_DIR_NAME=coco5k COCO_STAGE_LABEL=coco5k \
    COCO_PROTOCOL="coco2017_val5k_1024_sd35m_seeded_v1" \
    RUN_COCO5K=1 RUN_DIVERSITY=0 \
    bash clusters/musica/slurm/sd35m1024_quality_diversity.sbatch
fi

required_outputs=(
  "$EVAL_ROOT/coco5k/generation_manifest.jsonl"
  "$EVAL_ROOT/coco5k/metrics.json"
  "$EVAL_ROOT/coco5k/extended_metrics/metrics_hps.json"
  "$EVAL_ROOT/coco5k/extended_metrics/metrics_aes.json"
  "$EVAL_ROOT/coco5k/extended_metrics/metrics_fid.json"
)
for output in "${required_outputs[@]}"; do
  [[ -s "$output" ]] || {
    echo "required COCO5K output missing or empty: $output" >&2
    exit 1
  }
done
[[ "$(wc -l < "$EVAL_ROOT/coco5k/generation_manifest.jsonl")" -eq 5000 ]] || {
  echo "COCO5K manifest does not contain 5000 records" >&2
  exit 1
}

echo "completed family=$FAMILY step=$STEP root=$EVAL_ROOT"
