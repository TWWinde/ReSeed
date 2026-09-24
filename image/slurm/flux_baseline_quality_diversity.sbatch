#!/bin/bash
# Matched FLUX.1 evaluation (COCO5K quality or 100x10 diversity) on four H100s.
# MODEL=dev|schnell|senseflow selects the transformer, step count, guidance mode and
# the T5 sequence length; PHASE=quality|diversity selects the metric family.
# Submit the two phases as independent jobs so either stays valid if the other
# environment fails.  An optional BIAS_CHECKPOINT applies the learned
# first-interval residual bias on solver step zero only.
#SBATCH --job-name=flux-eval
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/data/fs201205/wt93384/project/CRRA/slurm/flux-baseline-eval-%j.out

set -uo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/data/fs201205/wt93384/project/CRRA}"
[[ -f "$PROJECT_ROOT/pyproject.toml" ]] || { echo "invalid PROJECT_ROOT: $PROJECT_ROOT" >&2; exit 2; }
cd "$PROJECT_ROOT"
export CRRA_ROOT="$PROJECT_ROOT"
source clusters/musica/env.sh
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

MODEL="${MODEL:?set MODEL=dev, MODEL=schnell, or MODEL=senseflow}"
case "$MODEL" in dev|schnell|senseflow) ;; *) echo "invalid MODEL=$MODEL" >&2; exit 2 ;; esac
PHASE="${PHASE:?set PHASE=quality or PHASE=diversity}"
case "$PHASE" in quality|diversity) ;; *) echo "invalid PHASE=$PHASE" >&2; exit 2 ;; esac

HUB="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
FLUX_DEV_MODEL="${CRRA_FLUX_DEV_MODEL:-$HUB/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21}"
FLUX_SCHNELL_MODEL="${CRRA_FLUX_SCHNELL_MODEL:-$HUB/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9}"
SENSEFLOW_TRANSFORMER="${CRRA_SENSEFLOW_FLUX_TRANSFORMER:-/data/fs201205/wt93384/models/SenseFlow/SenseFlow-FLUX}"
if [[ "$MODEL" == dev ]]; then
  MODEL_PATH="$FLUX_DEV_MODEL"
  STEPS="${STEPS:-28}"
  GUIDANCE="${GUIDANCE:-3.5}"
  MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
elif [[ "$MODEL" == schnell ]]; then
  MODEL_PATH="$FLUX_SCHNELL_MODEL"
  STEPS="${STEPS:-4}"
  GUIDANCE=""
  MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-256}"
else
  # The public SenseFlow release contains the distilled transformer only.
  # FLUX.1-dev supplies the tokenizer/text encoders, VAE, and scheduler.
  MODEL_PATH="$FLUX_DEV_MODEL"
  STEPS="${STEPS:-4}"
  GUIDANCE=""
  MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
fi

BIAS_CHECKPOINT="${BIAS_CHECKPOINT:-}"
BIAS_SCALE="${BIAS_SCALE:-1.0}"
BIAS_RANDOM_SEED="${BIAS_RANDOM_SEED:-}"
BIAS_SHUFFLE_SEED="${BIAS_SHUFFLE_SEED:-}"
if [[ -n "$BIAS_RANDOM_SEED" && -n "$BIAS_SHUFFLE_SEED" ]]; then
  echo "BIAS_RANDOM_SEED and BIAS_SHUFFLE_SEED are mutually exclusive" >&2
  exit 2
fi
if [[ -z "$BIAS_CHECKPOINT" && ( -n "$BIAS_RANDOM_SEED" || -n "$BIAS_SHUFFLE_SEED" ) ]]; then
  echo "bias variant options require BIAS_CHECKPOINT" >&2
  exit 2
fi

ROOT="${EVALUATION_ROOT:-$PROJECT_ROOT/evaluations/flux_${MODEL}_baseline_metrics_v1}"
QUALITY_PROMPTS="${QUALITY_PROMPTS:-$CRRA_COCO5K_PROMPTS}"
QUALITY_EXPECTED_COUNT="${QUALITY_EXPECTED_COUNT:-5000}"
QUALITY_DIR_NAME="${QUALITY_DIR_NAME:-coco5k}"
QUALITY_STAGE_LABEL="${QUALITY_STAGE_LABEL:-coco5k}"
COCO_OUT="$ROOT/$QUALITY_DIR_NAME"
DIV_OUT="$ROOT/diversity_100x10"
DIV_CASES="${DIV_CASES:-$PROJECT_ROOT/evaluations/musica_frequency_checkpoint_diversity_100x10_v1/inputs/sd3m_prompt_seed_cases.csv}"
DIV_EXPECTED_COUNT="${DIV_EXPECTED_COUNT:-1000}"
STATUS="$ROOT/evaluation_status_${PHASE}.tsv"
QUALITY_PROTOCOL="${QUALITY_PROTOCOL:-flux_${MODEL}_${STEPS}step_1024_coco5k_v1}"
DIV_PROTOCOL="${DIV_PROTOCOL:-flux_${MODEL}_${STEPS}step_1024_diversity_100x10_v1}"
RUN_LABEL="${RUN_LABEL:-FLUX.1-$MODEL baseline}"

[[ "$QUALITY_EXPECTED_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  echo "QUALITY_EXPECTED_COUNT must be a positive integer, got $QUALITY_EXPECTED_COUNT" >&2
  exit 2
}

mkdir -p "$ROOT"
printf 'stage\tstatus\tdetail\n' > "$STATUS"
record_status() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$STATUS"; }

required=("$MODEL_PATH/model_index.json" "$MODEL_PATH/scheduler/scheduler_config.json")
if [[ "$MODEL" == senseflow ]]; then
  required+=("$SENSEFLOW_TRANSFORMER/config.json" "$SENSEFLOW_TRANSFORMER/diffusion_pytorch_model.safetensors.index.json")
else
  required+=("$MODEL_PATH/transformer/config.json")
fi
if [[ -n "$BIAS_CHECKPOINT" ]]; then
  required+=("$BIAS_CHECKPOINT/residual_biases.pt")
fi
if [[ "$PHASE" == quality ]]; then
  required+=("$QUALITY_PROMPTS" "$CRRA_COCO5K_FID_REFERENCE")
else
  required+=("$DIV_CASES" "$CRRA_DINOV3_MODEL" "$CRRA_CLIP_VIT_LARGE_MODEL")
fi
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    record_status preflight failed "missing $path"
    echo "missing required input: $path" >&2
    exit 2
  fi
done
record_status preflight completed "model=$MODEL phase=$PHASE steps=$STEPS run=$RUN_LABEL"

common_args=(
  --model "$MODEL" --dev-model "$FLUX_DEV_MODEL" --schnell-model "$FLUX_SCHNELL_MODEL"
  --senseflow-transformer "$SENSEFLOW_TRANSFORMER"
  --steps "$STEPS" --max-sequence-length "$MAX_SEQUENCE_LENGTH" --resolution 1024
)
if [[ -n "$GUIDANCE" ]]; then
  common_args+=(--guidance "$GUIDANCE")
else
  common_args+=(--no-guidance)
fi
if [[ -n "$BIAS_CHECKPOINT" ]]; then
  common_args+=(--first-interval-residual-bias-checkpoint "$BIAS_CHECKPOINT" --bias-scale "$BIAS_SCALE")
  [[ -n "$BIAS_RANDOM_SEED" ]] && common_args+=(--bias-random-seed "$BIAS_RANDOM_SEED")
  [[ -n "$BIAS_SHUFFLE_SEED" ]] && common_args+=(--bias-shuffle-seed "$BIAS_SHUFFLE_SEED")
fi

generate_four_shards() {
  local prompts="$1" output="$2" expected="$3" protocol="$4" label="$5" samples="$6"
  mkdir -p "$output"
  local pids=()
  for shard in 0 1 2 3; do
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=12 \
      --mem=180G --cpu-bind=none \
      "$CRRA_PYTHON" tools/flux_generate.py "${common_args[@]}" \
      --prompts "$prompts" --output "$output" --evaluation-protocol "$protocol" \
      --expected-count "$expected" --samples-per-prompt "$samples" \
      --num-shards 4 --shard-id "$shard" --resume \
      > "$ROOT/${label}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || return 1
  "$CRRA_PYTHON" tools/flux_generate.py "${common_args[@]}" \
    --prompts "$prompts" --output "$output" --evaluation-protocol "$protocol" \
    --expected-count "$expected" --samples-per-prompt "$samples" \
    --num-shards 4 --finalize-shards
}

if [[ "$PHASE" == quality ]]; then
  if generate_four_shards "$QUALITY_PROMPTS" "$COCO_OUT" "$QUALITY_EXPECTED_COUNT" "$QUALITY_PROTOCOL" "$QUALITY_STAGE_LABEL" 1; then
    record_status "${QUALITY_STAGE_LABEL}_generation" completed "${QUALITY_EXPECTED_COUNT}_images"
  else
    record_status "${QUALITY_STAGE_LABEL}_generation" failed see_shard_logs
  fi
  if [[ -f "$COCO_OUT/generation_manifest.jsonl" ]]; then
    if "$CRRA_PYTHON" -m reseed score clip-pickscore \
      --manifest "$COCO_OUT/generation_manifest.jsonl" --output "$COCO_OUT" \
      --expected-count "$QUALITY_EXPECTED_COUNT" --clip-batch-size 4 --pickscore-batch-size 2 \
      --evaluation-protocol "$QUALITY_PROTOCOL"; then
      record_status clip_pickscore completed "n_${QUALITY_EXPECTED_COUNT}"
    else
      record_status clip_pickscore failed scorer_error
    fi
    if "$CRRA_PYTHON" -m reseed score quality suite \
      --manifest "$COCO_OUT/generation_manifest.jsonl" \
      --output "$COCO_OUT/extended_metrics" --expected-count "$QUALITY_EXPECTED_COUNT" \
      --evaluation-protocol "$QUALITY_PROTOCOL" --batch-size 8 --num-workers 8 \
      --metrics hps aes fid --hps-root "$CRRA_HPS_ROOT" \
      --hps-checkpoint "$CRRA_HPS_CHECKPOINT" --pyiqa-root "$CRRA_PYIQA_ROOT" \
      --python-deps "$CRRA_EVAL_PYTHON_DEPS" \
      --fid-reference-dir "$CRRA_COCO5K_FID_REFERENCE"; then
      record_status hps_aes_fid completed suite_finished
    else
      record_status hps_aes_fid partial_or_failed inspect_extended_metric_status
    fi
  else
    record_status clip_pickscore skipped missing_coco_manifest
    record_status hps_aes_fid skipped missing_coco_manifest
  fi
else
  if generate_four_shards "$DIV_CASES" "$DIV_OUT" "$DIV_EXPECTED_COUNT" "$DIV_PROTOCOL" diversity 10; then
    record_status diversity_generation completed "${DIV_EXPECTED_COUNT}_images"
  else
    record_status diversity_generation failed see_shard_logs
  fi
  if [[ -f "$DIV_OUT/generation_manifest.jsonl" ]]; then
    if "$CRRA_DIVERSITY_PYTHON" tools/feature_diversity_eval.py score \
      --manifest "$DIV_OUT/generation_manifest.jsonl" \
      --output-dir "$DIV_OUT/diversity_metrics" --samples-per-prompt 10 \
      --dino-model "$CRRA_DINOV3_MODEL" --clip-model "$CRRA_CLIP_VIT_LARGE_MODEL" \
      --feature-batch-size 16 --dtype fp16 --dino-dtype fp32 --clip-dtype fp16; then
      record_status diversity_dino_clip completed prompts_100_samples_10
    else
      record_status diversity_dino_clip failed scorer_error
    fi
    if "$CRRA_DIVERSITY_PYTHON" tools/prompt_lpips_diversity.py \
      --manifest "$DIV_OUT/generation_manifest.jsonl" \
      --output-dir "$DIV_OUT/lpips_diversity" --samples-per-prompt 10 \
      --net alex --batch-size 8; then
      record_status diversity_lpips completed prompts_100_samples_10
    else
      record_status diversity_lpips failed scorer_error
    fi
  else
    record_status diversity_dino_clip skipped missing_diversity_manifest
    record_status diversity_lpips skipped missing_diversity_manifest
  fi
fi

if [[ "$PHASE" == quality ]]; then
  if "$CRRA_PYTHON" tools/prune_completed_evaluation_images.py \
    --root "$ROOT" --quality-dir "$QUALITY_DIR_NAME" --keep 100; then
    record_status quality_image_prune completed kept_100
  else
    record_status quality_image_prune skipped_or_failed inspect_missing_metric_report
  fi
fi

cat "$STATUS"
