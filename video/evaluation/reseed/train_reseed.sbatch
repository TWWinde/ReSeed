#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --job-name=reseed-dmd2
#SBATCH --requeue
# NODE_FAIL happens here (29/79 nodes were down at last count). With --requeue
# SLURM restarts the script on another node and train_reseed.py resumes from
# the latest checkpoint-N automatically. Pass EXCLUDE=n3002-001,... via
# --export to steer around nodes that already failed.
#
# ReSeed training, 4 ranks (one sample each per update, global batch 4 --
# the same presentation budget CRRA used). Budget: ~26 teacher forwards per
# sample at ~1.07 s each dominates, so expect ~30 s per update on H100 and
# ~17 h for 2000 updates. CRRA reports the diversity gain saturating by
# update 300-400; checkpoints every 100 let us stop early after evaluating.
#
#   sbatch evaluation/reseed/train_reseed.sbatch
#   sbatch --export="ALL,MAX_STEPS=3,MICRO_STEPS=2,OUT_SUFFIX=_smoke" ...   # plumbing check
set -euo pipefail

DFD_ROOT="${DFD_ROOT:-/data/fs201205/wt93384/project/DFD}"
REPO="$DFD_ROOT/repo"
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HOME="${HF_HOME:-/data/fs201205/wt93384/huggingface}"
export HF_HUB_DISABLE_XET=1 LOCAL_FILES_ONLY=true TOKENIZERS_PARALLELISM=false
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-$REPO/evaluation/fastgen_output}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export NCCL_TIMEOUT=1800

PY="$DFD_ROOT/venv/bin/python"
STUDENT="${STUDENT:-dmd2}"
CKPT="${CKPT:-$DFD_ROOT/checkpoints/dmd2_t2v_vipe/0001000}"
EMB="${EMB:-$DFD_ROOT/reseed_data/vipe_long/umt5_embeddings.pt}"
OUT="${OUT:-$DFD_ROOT/reseed_runs/${STUDENT}_m${MICRO_STEPS:-13}${OUT_SUFFIX:-}}"
NPROC="${NPROC:-4}"

for p in "$CKPT.net_model/.metadata" "$CKPT.pth" "$EMB"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p" >&2; exit 2; }
done
mkdir -p "$OUT"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} student=$STUDENT out=$OUT nproc=$NPROC"

"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  evaluation/reseed/train_reseed.py \
  --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
  --ckpt_path "$CKPT" \
  --embeddings "$EMB" \
  --output "$OUT" \
  --max-steps "${MAX_STEPS:-2000}" \
  --micro-steps "${MICRO_STEPS:-13}" \
  --save-every "${SAVE_EVERY:-100}" \
  --log-every 5 \
  --seed "${SEED:-42}" \
  ${EXTRA_ARGS:-} \
  - trainer.fsdp=True trainer.ddp=False "log_config.name=reseed_${STUDENT}"
