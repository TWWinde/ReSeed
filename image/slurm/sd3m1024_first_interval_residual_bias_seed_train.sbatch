#!/bin/bash
# Train zero-initialized per-channel residual-output biases for interval zero.
# All frozen S7 weights and all three suffix intervals remain unchanged.
# Seed-parameterized variant of sd3m1024_first_interval_residual_bias_train.sbatch:
# SEED is required and selects the trainer seed (prompt stream + rank-offset noise
# generator) and the default output directory.
#SBATCH --job-name=sd3-s0-bias-seed
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/data/fs201205/wt93384/project/CRRA/slurm/sd3m1024-first-interval-residual-bias-seed-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/data/fs201205/wt93384/project/CRRA}"
cd "$PROJECT_ROOT"
export CRRA_ROOT="$PROJECT_ROOT"
source clusters/musica/env.sh
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING=1

MODEL="$CRRA_SD3M_MODEL"
STUDENT="$PROJECT_ROOT/outputs/musica_sd3m1024_tdm_teacher_cfg_scale_v1/sd3m_tdm_teacher_cfg_w7_1024/seed_42/checkpoint-2000"
SEED="${SEED:?SEED must be set (e.g. SEED=43)}"
MAX_STEPS="${MAX_STEPS:-2000}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/outputs/sd3m1024_s7_first_interval_residual_bias_blocks0_23_2000_seed${SEED}_v1}"
BLOCK_START="${BLOCK_START:-0}"
BLOCK_END="${BLOCK_END:-23}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SAVE_EVERY="${SAVE_EVERY:-250}"
KEEP_LAST="${KEEP_LAST:-2}"

[[ -f "$STUDENT/student/config.json" ]] || { echo "missing S7 checkpoint: $STUDENT" >&2; exit 2; }
[[ -f "$CRRA_JOURNEYDB_TRAIN" ]] || { echo "missing prompt file: $CRRA_JOURNEYDB_TRAIN" >&2; exit 2; }

echo "base=S7 adapter=residual_bias blocks=$BLOCK_START..$BLOCK_END max_steps=$MAX_STEPS lr=$LEARNING_RATE seed=$SEED"
exec "$CRRA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m reseed.trainers.sd3m_first_interval_residual_bias \
  --model "$MODEL" \
  --student-checkpoint "$STUDENT" \
  --prompt-file "$CRRA_JOURNEYDB_TRAIN" \
  --output "$OUTPUT" \
  --resolution 1024 \
  --max-steps "$MAX_STEPS" \
  --batch-size 1 \
  --block-start "$BLOCK_START" --block-end "$BLOCK_END" \
  --teacher-micro-steps 7 \
  --teacher-guidance-scale 7 \
  --teacher-cfg-threshold 0.96 \
  --learning-rate "$LEARNING_RATE" \
  --save-every "$SAVE_EVERY" --keep-last "$KEEP_LAST" \
  --seed "$SEED"
