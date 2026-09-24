#!/bin/bash
# Phase 1: build the FLUX.1-dev teacher-endpoint cache for the schnell student's
# first interval.  The teacher is the ONLY transformer loaded here -- no student,
# no optimizer -- because dev + schnell + T5-XXL with a 1024px backward pass does
# not fit comfortably on an 80 GB H100.  Four ranks, one GPU each, every rank
# writing its own shard under CACHE/rank_NN/.
#SBATCH --job-name=flux-s0-cache
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=/data/fs201205/wt93384/project/CRRA/slurm/flux-first-interval-bias-cache-%j.out

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
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-2000}"
CACHE="${CACHE:-$PROJECT_ROOT/outputs/flux_first_interval_residual_bias_v1/teacher_cache_seed_${SEED}}"
TEACHER_GUIDANCE="${TEACHER_GUIDANCE:-3.5}"
TEACHER_MICRO_STEPS="${TEACHER_MICRO_STEPS:-7}"

[[ -f "$TEACHER/transformer/config.json" ]] || { echo "missing FLUX.1-dev teacher: $TEACHER" >&2; exit 2; }
[[ -f "$STUDENT/scheduler/scheduler_config.json" ]] || { echo "missing FLUX student scheduler: $STUDENT" >&2; exit 2; }
[[ -f "$CRRA_JOURNEYDB_TRAIN" ]] || { echo "missing prompt file: $CRRA_JOURNEYDB_TRAIN" >&2; exit 2; }

mkdir -p "$CACHE"
echo "phase=cache teacher=dev guidance=$TEACHER_GUIDANCE micro_steps=$TEACHER_MICRO_STEPS steps=$MAX_STEPS seed=$SEED cache=$CACHE"
exec "$CRRA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m reseed.trainers.flux_first_interval_residual_bias \
  --build-cache-only \
  --teacher-target-cache "$CACHE" \
  --teacher-model "$TEACHER" \
  --student-model "$STUDENT" \
  --prompt-file "$CRRA_JOURNEYDB_TRAIN" \
  --resolution 1024 \
  --student-steps 4 \
  --max-steps "$MAX_STEPS" \
  --batch-size 1 \
  --teacher-micro-steps "$TEACHER_MICRO_STEPS" \
  --teacher-guidance-scale "$TEACHER_GUIDANCE" \
  --teacher-max-sequence-length 512 \
  --seed "$SEED" \
  --resume
