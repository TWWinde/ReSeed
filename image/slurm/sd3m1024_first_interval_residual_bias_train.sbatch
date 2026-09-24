#!/bin/bash
# Train zero-initialized per-channel residual-output biases for interval zero.
# All frozen S7 weights and all three suffix intervals remain unchanged.
#SBATCH --job-name=sd3-s0-bias
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/data/fs201205/wt93384/project/CRRA/slurm/sd3m1024-first-interval-residual-bias-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/data/fs201205/wt93384/project/CRRA}"
cd "$PROJECT_ROOT"
export CRRA_ROOT="$PROJECT_ROOT"
source clusters/musica/env.sh
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING=1

# MODEL/TEACHER_CFG are env switches so the same driver runs the SD3.5-Medium
# arm: MODEL=$CRRA_SD35M_MODEL TEACHER_CFG=4.5. Block count (24), shift (3) and
# the bf16 boundaries [1, 0.8984375, 0.75, 0.5, 0] are identical on both models.
MODEL="${MODEL:-$CRRA_SD3M_MODEL}"
TEACHER_CFG="${TEACHER_CFG:-7}"
STUDENT="${STUDENT_CHECKPOINT:-$PROJECT_ROOT/outputs/musica_sd3m1024_tdm_teacher_cfg_scale_v1/sd3m_tdm_teacher_cfg_w7_1024/seed_42/checkpoint-2000}"
MAX_STEPS="${MAX_STEPS:-2000}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/outputs/sd3m1024_s7_first_interval_residual_bias_blocks0_23_2000_v1}"
BLOCK_START="${BLOCK_START:-0}"
BLOCK_END="${BLOCK_END:-23}"
# Student interval the bias is trained on: 0..3, or "all" to sample k uniformly
# at every step. For k > 0 the input state is a frozen adapter-free S7 rollout.
INTERVAL_INDEX="${INTERVAL_INDEX:-0}"
[[ "$INTERVAL_INDEX" =~ ^([0-3]|all)$ ]] || {
  echo "INTERVAL_INDEX must be 0, 1, 2, 3 or all, got $INTERVAL_INDEX" >&2
  exit 2
}
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SAVE_EVERY="${SAVE_EVERY:-250}"
KEEP_LAST="${KEEP_LAST:-2}"
CHECKPOINT_HOOK="${CHECKPOINT_HOOK:-}"
SEED="${SEED:-42}"
# Teacher sub-steps inside the one student interval. The grid is built by
# uniform interpolation in unshifted base-sigma space between the student
# boundaries, and a student interval is exactly 1/STUDENT_STEPS of that
# space, so micro-steps m reproduces a global STUDENT_STEPS*m-step teacher
# exactly. SD3-M distils and evaluates against a 28-step teacher, hence 7.
# SD3.5-M uses a 40-step reference teacher and must pass 10.
TEACHER_MICRO_STEPS="${TEACHER_MICRO_STEPS:-7}"

[[ -f "$STUDENT/student/config.json" || -f "$STUDENT/transformer/config.json" ]] || {
  echo "missing student checkpoint (student/ or transformer/): $STUDENT" >&2
  exit 2
}
[[ -f "$CRRA_JOURNEYDB_TRAIN" ]] || { echo "missing prompt file: $CRRA_JOURNEYDB_TRAIN" >&2; exit 2; }

echo "base=S7 adapter=residual_bias blocks=$BLOCK_START..$BLOCK_END max_steps=$MAX_STEPS lr=$LEARNING_RATE interval_index=$INTERVAL_INDEX teacher_micro_steps=$TEACHER_MICRO_STEPS teacher_cfg=$TEACHER_CFG"
TRAIN_ARGS=(
  --model "$MODEL"
  --student-checkpoint "$STUDENT"
  --prompt-file "$CRRA_JOURNEYDB_TRAIN"
  --output "$OUTPUT"
  --resolution 1024
  --max-steps "$MAX_STEPS"
  --batch-size 1
  --block-start "$BLOCK_START" --block-end "$BLOCK_END"
  --interval-index "$INTERVAL_INDEX"
  --teacher-micro-steps "$TEACHER_MICRO_STEPS"
  --teacher-guidance-scale "$TEACHER_CFG"
  --teacher-cfg-threshold 0.96
  --learning-rate "$LEARNING_RATE"
  --save-every "$SAVE_EVERY" --keep-last "$KEEP_LAST"
  --student-steps "${STUDENT_STEPS:-4}"
  --seed "$SEED"
)
[[ -z "$CHECKPOINT_HOOK" ]] || TRAIN_ARGS+=(--checkpoint-hook "$CHECKPOINT_HOOK")
exec "$CRRA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m reseed.trainers.sd3m_first_interval_residual_bias \
  "${TRAIN_ARGS[@]}"
