#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=48
#SBATCH --time=00:50:00
#SBATCH --job-name=reseed-peek
#
# Sanity check on an intermediate ReSeed checkpoint: a few videos, with and
# without the adapter, same seeds. Answers "did training collapse?" without
# waiting for the full pipeline.
#   CK=<checkpoint-N dir>   (default: dmd2_m13/checkpoint-100; waits for it)
set -euo pipefail
DFD_ROOT="${DFD_ROOT:-/data/fs201205/wt93384/project/DFD}"
REPO="$DFD_ROOT/repo"; cd "$REPO"
CK="${CK:-$DFD_ROOT/reseed_runs/dmd2_m13/checkpoint-100}"
for _ in $(seq 1 80); do [[ -f "$CK/residual_biases.pt" ]] && break; sleep 30; done
[[ -f "$CK/residual_biases.pt" ]] || { echo "ERROR: $CK never appeared" >&2; exit 2; }
echo "checkpoint: $CK"; cat "$CK/manifest.json" | tr -d '\n' | cut -c1-400; echo

export PYTHONPATH="$REPO" HF_HOME="${HF_HOME:-/data/fs201205/wt93384/huggingface}"
export HF_HUB_DISABLE_XET=1 LOCAL_FILES_ONLY=true TOKENIZERS_PARALLELISM=false
export FASTGEN_OUTPUT_ROOT="$REPO/evaluation/fastgen_output" CUDA_VISIBLE_DEVICES=0
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
PY="$DFD_ROOT/venv/bin/python"
OUT="$DFD_ROOT/reseed_peek/$(basename "$CK")"; rm -rf "$OUT"; mkdir -p "$OUT"
head -n 3 web_dfd_eval_prompts.txt > "$OUT/peek_prompts.txt"     # first 3 lines are unique

port=29600
for SEED in 0 1; do for ARM in control reseed; do
  port=$((port + 1))
  ADAPTER_ENV=(); [[ $ARM == reseed ]] && ADAPTER_ENV=(FASTGEN_RESEED_ADAPTER="$CK")
  echo "### $ARM seed=$SEED"
  env MASTER_PORT=$port FASTGEN_EVAL_PROBE="$OUT/$ARM/seed$SEED/probe.json" "${ADAPTER_ENV[@]}" \
    "$PY" scripts/inference/video_model_inference.py \
      --ckpt_path "$DFD_ROOT/checkpoints/dmd2_t2v_vipe/0001000" \
      --do_student_sampling True --do_teacher_sampling False \
      --prompt_file "$OUT/peek_prompts.txt" \
      --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
      --save_as_gif False --video_save_dir "$OUT/$ARM/seed$SEED" - \
      trainer.seed=$SEED trainer.fsdp=True trainer.ddp=False log_config.name=reseed_peek \
      2>&1 | grep -E "ReSeed adapter installed|Student sampling time|Saved video|Error|Traceback" | cut -c1-220
done; done

echo "### compare"
"$PY" - "$OUT" <<'PY'
import json, sys, numpy as np, imageio.v3 as iio
from pathlib import Path
out = Path(sys.argv[1]); ok = True
for seed in (0, 1):
    pc = json.load(open(out/f"control/seed{seed}/probe.json")); pr = json.load(open(out/f"reseed/seed{seed}/probe.json"))
    same = pc["initial_noise"]["sha256"] == pr["initial_noise"]["sha256"]
    ok &= same
    print(f"seed {seed}: noise bit-identical control vs reseed = {same}   adapter step={pr.get('reseed_adapter',{}).get('step')} "
          f"bias_rms={pr.get('reseed_adapter',{}).get('bias_rms'):.4g} abs_max={pr.get('reseed_adapter',{}).get('bias_abs_max'):.4g}")
    for i in range(3):
        vc = iio.imread(out/f"control/seed{seed}/peek_prompts/student_step4_{i:04d}_seed{seed}.mp4").astype(np.float32)
        vr = iio.imread(out/f"reseed/seed{seed}/peek_prompts/student_step4_{i:04d}_seed{seed}.mp4").astype(np.float32)
        mae = np.abs(vc - vr).mean()
        flag = "" if 10 < vr.std() < 90 else "  <-- DEGENERATE?"
        print(f"  prompt {i}: control std={vc.std():5.1f}  reseed std={vr.std():5.1f}  MAE(control,reseed)={mae:5.2f}{flag}")
        ok &= vr.std() > 10
print("PEEK:", "OK" if ok else "PROBLEM")
PY
