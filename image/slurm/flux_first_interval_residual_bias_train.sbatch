#!/bin/bash
# Phase 2: train zero-initialised per-channel residual biases on all 19 double
# and 38 single FLUX.1-schnell blocks (233,472 fp32 parameters) against the
# cached FLUX.1-dev teacher endpoints.  The teacher is never loaded here -- the
# cache from flux_first_interval_residual_bias_cache.sbatch supplies the targets,
# which is what keeps the job inside 80 GB at 1024px.  SEED must match the seed
# the cache was built with, or the trainer's (seed, rank, step) check will fail.
#SBATCH --job-name=flux-s0-bias
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/data/fs201205/wt93384/project/CRRA/slurm/flux-first-interval-bias-train-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/data/fs201205/wt93384/project/CRRA}"
[[ -f "$PROJECT_ROOT/pyproject.toml" ]] || { echo "invalid PROJECT_ROOT: $PROJECT_ROOT" >&2; exit 2; }
cd "$PROJECT_ROOT"
export CRRA_ROOT="$PROJECT_ROOT"
source clusters/musica/env.sh
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING=1

HUB="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
TEACHER="${CRRA_FLUX_DEV_MODEL:-$HUB/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21}"
STUDENT="${STUDENT_COMPONENTS_MODEL:-${CRRA_FLUX_SCHNELL_MODEL:-$HUB/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9}}"
STUDENT_TRANSFORMER="${STUDENT_TRANSFORMER_MODEL:-$STUDENT}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-2000}"
CACHE="${CACHE:-$PROJECT_ROOT/outputs/flux_first_interval_residual_bias_v1/teacher_cache_seed_${SEED}}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/outputs/flux_first_interval_residual_bias_v1/seed_${SEED}}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SAVE_EVERY="${SAVE_EVERY:-250}"
KEEP_LAST="${KEEP_LAST:-2}"
TEACHER_GUIDANCE="${TEACHER_GUIDANCE:-3.5}"
TEACHER_MICRO_STEPS="${TEACHER_MICRO_STEPS:-7}"
STUDENT_MAX_SEQUENCE_LENGTH="${STUDENT_MAX_SEQUENCE_LENGTH:-256}"
CHECKPOINT_HOOK="${CHECKPOINT_HOOK:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ ! -f "$STUDENT_TRANSFORMER/config.json" && ! -f "$STUDENT_TRANSFORMER/transformer/config.json" ]]; then
  echo "missing FLUX student transformer: $STUDENT_TRANSFORMER" >&2
  exit 2
fi
[[ -f "$CACHE/cache_manifest.json" ]] || { echo "missing teacher-target cache: $CACHE" >&2; exit 2; }
[[ -f "$CRRA_JOURNEYDB_TRAIN" ]] || { echo "missing prompt file: $CRRA_JOURNEYDB_TRAIN" >&2; exit 2; }

mkdir -p "$OUTPUT"
echo "phase=train student=$STUDENT_TRANSFORMER blocks=all(19 double + 38 single) steps=$MAX_STEPS lr=$LEARNING_RATE seed=$SEED cache=$CACHE"
TRAIN_ARGS=(
  -m reseed.trainers.flux_first_interval_residual_bias \
  --teacher-target-cache "$CACHE" \
  --teacher-model "$TEACHER" \
  --student-model "$STUDENT" \
  --student-transformer-model "$STUDENT_TRANSFORMER" \
  --prompt-file "$CRRA_JOURNEYDB_TRAIN" \
  --output "$OUTPUT" \
  --resolution 1024 \
  --student-steps 4 \
  --max-steps "$MAX_STEPS" \
  --batch-size 1 \
  --teacher-micro-steps "$TEACHER_MICRO_STEPS" \
  --teacher-guidance-scale "$TEACHER_GUIDANCE" \
  --teacher-max-sequence-length 512 \
  --student-max-sequence-length "$STUDENT_MAX_SEQUENCE_LENGTH" \
  --learning-rate "$LEARNING_RATE" \
  --gradient-checkpointing \
  --save-every "$SAVE_EVERY" --keep-last "$KEEP_LAST" \
  --seed "$SEED"
)
[[ -z "$CHECKPOINT_HOOK" ]] || TRAIN_ARGS+=(--checkpoint-hook "$CHECKPOINT_HOOK")
[[ -z "$RESUME_FROM_CHECKPOINT" ]] || TRAIN_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
exec "$CRRA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 "${TRAIN_ARGS[@]}"
