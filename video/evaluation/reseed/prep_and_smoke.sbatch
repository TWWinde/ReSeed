#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=48
#SBATCH --time=01:30:00
#SBATCH --job-name=reseed-prep
#
# 1. Encode the ViPE captions with UMT5 once (GPU) -> umt5_embeddings.pt
# 2. 3-update smoke of the trainer on 2 ranks with m=2: exercises the
#    identity check, the bias-gradient check, DDP all-reduce and checkpoint
#    save. If this passes, the 17 h run is just the same thing with the
#    real budget.
set -euo pipefail
DFD_ROOT="${DFD_ROOT:-/data/fs201205/wt93384/project/DFD}"
REPO="$DFD_ROOT/repo"; cd "$REPO"
export PYTHONPATH="$REPO" HF_HOME="${HF_HOME:-/data/fs201205/wt93384/huggingface}"
export HF_HUB_DISABLE_XET=1 LOCAL_FILES_ONLY=true TOKENIZERS_PARALLELISM=false
export FASTGEN_OUTPUT_ROOT="$REPO/evaluation/fastgen_output" OMP_NUM_THREADS=10
PY="$DFD_ROOT/venv/bin/python"
DATA="$DFD_ROOT/reseed_data/vipe_long"
EMB="$DATA/umt5_embeddings.pt"

echo "### 1/2 encode captions"
[[ -s "$DATA/captions.txt" ]] || { echo "ERROR: $DATA/captions.txt missing" >&2; exit 2; }
if [[ -s "$EMB" ]]; then echo "  reusing $EMB"; else
  CUDA_VISIBLE_DEVICES=0 "$PY" evaluation/reseed/encode_prompts.py --captions "$DATA/captions.txt" --out "$EMB" --batch-size 8
fi

echo "### 2/2 smoke training: 3 updates, m=2, 2 ranks"
OUT="$DFD_ROOT/reseed_runs/dmd2_smoke"; rm -rf "$OUT"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 evaluation/reseed/train_reseed.py \
  --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
  --ckpt_path "$DFD_ROOT/checkpoints/dmd2_t2v_vipe/0001000" \
  --embeddings "$EMB" --output "$OUT" \
  --max-steps 3 --micro-steps 2 --save-every 3 --log-every 1 \
  - trainer.fsdp=True trainer.ddp=False log_config.name=reseed_smoke
echo "### smoke artefacts"; find "$OUT" -type f | sort; cat "$OUT/metrics.jsonl"
