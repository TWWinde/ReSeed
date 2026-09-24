#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --job-name=reseed-eval
#SBATCH --requeue
#
# Evaluate one ReSeed adapter checkpoint as a third arm next to dmd2 and dfd.
#
#   ADAPTER=/data/.../reseed_runs/dmd2_m13/checkpoint-500 \
#       sbatch --export=ALL evaluation/reseed/eval_reseed.sbatch
#
# Generation runs ONLY for the new arm (same 71 prompts x 8 seeds, identical
# noise to the dmd2 arm because the adapter install is RNG-free); the existing
# dmd2/dfd videos and their metadata are kept, and stages 3-6 then score all
# three arms together: DINO/CLIP diversity + CLIP text adherence + VBench.
# Pass NUM_PROMPTS=5 for a quick look.
set -euo pipefail
DFD_ROOT="${DFD_ROOT:-/data/fs201205/wt93384/project/DFD}"
REPO="$DFD_ROOT/repo"; cd "$REPO"
[[ -n "${ADAPTER:-}" && -f "$ADAPTER/residual_biases.pt" ]] || { echo "ERROR: ADAPTER must point to a checkpoint-N dir with residual_biases.pt" >&2; exit 2; }

export DFD_RESEED_ADAPTER="$ADAPTER"
export PYTHON="$DFD_ROOT/venv/bin/python" DFD_PYTHON="$DFD_ROOT/venv/bin/python"
export VBENCH_PYTHON="$DFD_ROOT/venv_vbench/bin/python"
export DFD_ROOT DFD_CKPT_ROOT="$DFD_ROOT/checkpoints"
export HF_HOME="${HF_HOME:-/data/fs201205/wt93384/huggingface}" HF_HUB_DISABLE_XET=1
export FASTGEN_OUTPUT_ROOT="$REPO/evaluation/fastgen_output" TORCH_HOME=/data/fs201205/wt93384/cache/torch
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false

# One arm per checkpoint: checkpoint-500 -> arm `dmd2_reseed_ckpt500`, which
# inherits the dmd2_reseed spec (eval_config.model_spec) but has its own video
# directory, metadata records and VBench raw dir. Two checkpoints therefore
# never collide in generated/dmd2_reseed/, and the resume logic can never
# mistake one checkpoint's videos for another's.
STEP="$(basename "$ADAPTER")"                 # checkpoint-500
ARM="dmd2_reseed_${STEP//checkpoint-/ckpt}"   # dmd2_reseed_ckpt500
export RESULTS_TAG="reseed_${STEP}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} adapter=$ADAPTER arm=$ARM"

ARGS=(--models "$ARM")
[[ -n "${NUM_PROMPTS:-}" ]] && ARGS+=(--num-prompts "$NUM_PROMPTS")
bash evaluation/scripts/run_all.sh "${ARGS[@]}"

# Keep a copy of the three-arm results keyed by adapter step.
DST="$REPO/evaluation/results_${RESULTS_TAG}"; mkdir -p "$DST"
cp -r "$REPO/evaluation/results/." "$DST/"
cp "$ADAPTER/manifest.json" "$DST/adapter_manifest.json" 2>/dev/null || true
echo "results copied to $DST"
