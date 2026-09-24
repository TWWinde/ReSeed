#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=48
#SBATCH --time=00:40:00
#SBATCH --job-name=reseed-bench
#SBATCH --requeue
# Inference overhead of the ReSeed adapter (evaluation/reseed/bench_overhead.py).
#   ADAPTER=<checkpoint-N dir> sbatch --export=ALL evaluation/reseed/bench_overhead.sbatch
set -euo pipefail
DFD_ROOT="${DFD_ROOT:-/data/fs201205/wt93384/project/DFD}"
REPO="$DFD_ROOT/repo"; cd "$REPO"
ADAPTER="${ADAPTER:-$DFD_ROOT/reseed_runs/dmd2_m13/checkpoint-300}"
export PYTHONPATH="$REPO" HF_HOME="${HF_HOME:-/data/fs201205/wt93384/huggingface}"
export HF_HUB_DISABLE_XET=1 LOCAL_FILES_ONLY=true TOKENIZERS_PARALLELISM=false
export FASTGEN_OUTPUT_ROOT="$REPO/evaluation/fastgen_output" CUDA_VISIBLE_DEVICES=0
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29777
OUT="$DFD_ROOT/reseed_runs/bench_$(basename "$ADAPTER").json"
echo "host=$(hostname) adapter=$ADAPTER -> $OUT"
"$DFD_ROOT/venv/bin/python" evaluation/reseed/bench_overhead.py \
  --ckpt_path "$DFD_ROOT/checkpoints/dmd2_t2v_vipe/0001000" \
  --adapter "$ADAPTER" \
  --embeddings "$DFD_ROOT/reseed_data/vipe_long/umt5_embeddings.pt" \
  --output "$OUT" --n-prompts 8 --warmup 2 \
  --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
  - trainer.fsdp=True trainer.ddp=False log_config.name=reseed_bench
echo DONE
