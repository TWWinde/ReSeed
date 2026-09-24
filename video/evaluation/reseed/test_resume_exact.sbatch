#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=48
#SBATCH --time=00:40:00
#SBATCH --job-name=reseed-rexact
#
# Does resuming add drift beyond what two uninterrupted runs already show?
#   A1, A2: steps 1..7 straight through, same seed  (run-to-run baseline)
#   B:      steps 1..4, stop, resume 5..7 from checkpoint-4
# The first version of this test compared only A vs B and "failed" -- but the
# divergence began at step 2, before any resume. Two fresh runs with the same
# seed are NOT bit-identical here: the forward is (step-1 loss matches to the
# last digit), the backward is not (flash-attention atomics, cudnn.benchmark,
# TF32). So the question is whether |A1-B| looks like |A1-A2|.
set -euo pipefail
DFD_ROOT="${DFD_ROOT:-/data/fs201205/wt93384/project/DFD}"
REPO="$DFD_ROOT/repo"; cd "$REPO"
export PYTHONPATH="$REPO" HF_HOME="${HF_HOME:-/data/fs201205/wt93384/huggingface}"
export HF_HUB_DISABLE_XET=1 LOCAL_FILES_ONLY=true TOKENIZERS_PARALLELISM=false
export FASTGEN_OUTPUT_ROOT="$REPO/evaluation/fastgen_output" OMP_NUM_THREADS=10
PY="$DFD_ROOT/venv/bin/python"
BASE="$DFD_ROOT/reseed_runs/rexact"; rm -rf "$BASE"; mkdir -p "$BASE"
COMMON=(evaluation/reseed/train_reseed.py --config fastgen/configs/experiments/WanT2V/config_dmd2.py
        --ckpt_path "$DFD_ROOT/checkpoints/dmd2_t2v_vipe/0001000"
        --embeddings "$DFD_ROOT/reseed_data/vipe_long/umt5_embeddings.pt"
        --micro-steps 2 --log-every 1 --seed 42 --skip-identity-check)
run() { "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 "${COMMON[@]}" "$@" \
        - trainer.fsdp=True trainer.ddp=False log_config.name=reseed_rexact; }

echo "### A1: 1..7 uninterrupted"; run --output "$BASE/A1" --max-steps 7 --save-every 7
echo "### A2: 1..7 uninterrupted"; run --output "$BASE/A2" --max-steps 7 --save-every 7
echo "### B1: 1..4";               run --output "$BASE/B"  --max-steps 4 --save-every 4
echo "### B2: resume 5..7";        run --output "$BASE/B"  --max-steps 7 --save-every 7

echo "### compare"
"$PY" - <<'PY'
import json, torch
from pathlib import Path
base = Path("/data/fs201205/wt93384/project/DFD/reseed_runs/rexact")
def bias(name): return torch.load(base/name/"checkpoint-7/residual_biases.pt", weights_only=True)["state_dict"]
def seq(name):
    rows=[json.loads(l) for l in (base/name/"metrics.jsonl").read_text().splitlines() if l.strip() and '"step"' in l]
    return [(r["step"], r["prompt_idx"], r["loss"]) for r in rows]
def maxdiff(x, y): return max(float((x[k].float()-y[k].float()).abs().max()) for k in x)
def rms(x, y): return float(torch.cat([(x[k].float()-y[k].float()).flatten() for k in x]).pow(2).mean().sqrt())
A1, A2, B = bias("A1"), bias("A2"), bias("B")
sA1, sA2, sB = seq("A1"), seq("A2"), seq("B")
print("prompt sequences identical (A1==A2==B):", [x[:2] for x in sA1]==[x[:2] for x in sA2]==[x[:2] for x in sB])
print("step-1 loss  A1 %.6f  A2 %.6f  B %.6f" % (sA1[0][2], sA2[0][2], sB[0][2]))
first_div = next((i+1 for i,(a,b) in enumerate(zip(sA1,sA2)) if a[2]!=b[2]), None)
print("A1 vs A2 first differing loss at step:", first_div, " (resume in B happens at 4->5)")
d_aa, d_ab1, d_ab2 = maxdiff(A1,A2), maxdiff(A1,B), maxdiff(A2,B)
r_aa, r_ab1, r_ab2 = rms(A1,A2), rms(A1,B), rms(A2,B)
print("checkpoint-7 bias max|diff|:  A1-A2 %.3e   A1-B %.3e   A2-B %.3e" % (d_aa, d_ab1, d_ab2))
print("checkpoint-7 bias rms diff :  A1-A2 %.3e   A1-B %.3e   A2-B %.3e" % (r_aa, r_ab1, r_ab2))
verdict = "PASS" if (d_aa > 0 and max(d_ab1, d_ab2) <= 2.0*d_aa) or (d_aa == 0 and max(d_ab1,d_ab2) == 0) else "FAIL"
print("RESUME_DRIFT_WITHIN_BASELINE:", verdict, " (resume-vs-fresh drift <= 2x fresh-vs-fresh drift)")
PY
