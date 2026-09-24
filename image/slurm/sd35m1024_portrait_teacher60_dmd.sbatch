#!/bin/bash
# Figure 1 panels on the genuine DMD student: the 75 portrait cases of
# sd35m1024_portrait_3prompts_25seeds_v1 (same prompts, seeds and latents) drawn by
#   (a) the SD3.5-Medium teacher, 60 steps at CFG 4.5
#   (b) the four-step DMD student (objective dmd, seed 42)
#   (c) the same DMD student with its paired ReSeed adapter (seed 42, 2,100
#       updates, the checkpoint the SD3.5-Medium DMD + ReSeed table row uses)
# The earlier portrait run used a TDM student under a caption that says DMD.
#SBATCH --job-name=sd35-portrait-dmd
#SBATCH --account=p201205
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/data/fs201205/wt93384/project/CRRA/slurm/sd35m1024-portrait-teacher60-dmd-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/data/fs201205/wt93384/project/CRRA}"
cd "$PROJECT_ROOT"
source clusters/musica/env.sh
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

MODEL="${MODEL:-$CRRA_SD35M_MODEL}"
SEED="${SEED:-42}"
STUDENT="$PROJECT_ROOT/outputs/musica_sd35m1024_dmd_only_unified_3seed_v1/sd35m_dmd_teacher_cfg_w4p5/seed_$SEED/checkpoint-2000"
BIAS="$PROJECT_ROOT/outputs/musica_sd35m1024_dmd_reseed_3seed_v1/seed_$SEED/checkpoint-2100"
CASES="$PROJECT_ROOT/configs/evaluation/flux_portrait_3prompts_25seeds_v1.csv"
ROOT="${EVALUATION_ROOT:-$PROJECT_ROOT/evaluations/sd35m1024_portrait_teacher60_dmd_v1}"
# the teacher samples under the 60-step protocol; the students keep the 40-step
# one, because each checkpoint records the 40-step reference it was distilled
# against and the strict check refuses any other. Both protocols share the
# prompt CSV, seeds and latents, so the arms still match case by case.
TEACHER_PROTOCOL=sd35m1024_portrait_3prompts_25seeds_teacher60_v1
STUDENT_PROTOCOL=sd35m1024_portrait_3prompts_25seeds_v1
COUNT=75

for path in "$MODEL/model_index.json" "$STUDENT/manifest.json" "$BIAS/residual_biases.pt" "$CASES"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
"$CRRA_PYTHON" - "$STUDENT/manifest.json" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1])).get("objective", {})
obj = obj.get("kind", obj) if isinstance(obj, dict) else obj
if obj != "dmd":
    raise SystemExit(f"refusing: {sys.argv[1]} reports objective {obj!r}, expected 'dmd'")
PY
mkdir -p "$ROOT"

generate_arm() {
  local label="$1" mode="$2" steps="$3" output="$4" protocol="$5"
  shift 5
  local command=(
    "$CRRA_PYTHON" -m reseed generate sd35m
    --mode "$mode" --model "$MODEL" "$@"
    --prompts "$CASES" --output "$output" --evaluation-protocol "$protocol"
    --expected-count "$COUNT" --inference-steps "$steps"
    --teacher-guidance-scale 4.5 --student-guidance-scale 1
    --resolution 1024 --max-sequence-length 256 --batch-size 1 --num-shards 4
  )
  mkdir -p "$output"
  local pids=()
  for shard in 0 1 2 3; do
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=12 --mem=180G --cpu-bind=none \
      "${command[@]}" --shard-id "$shard" --resume > "$ROOT/${label}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "$label generation failed; see $ROOT/${label}_shard_*.log" >&2; exit 1; }
  "${command[@]}" --finalize-shards
}

generate_arm teacher sd35m-base 60 "$ROOT/teacher_60step_cfg4p5" "$TEACHER_PROTOCOL"
generate_arm dmd sd35m-dmd 4 "$ROOT/dmd_seed${SEED}_4step" "$STUDENT_PROTOCOL" --checkpoint "$STUDENT"
generate_arm dmd_reseed sd35m-dmd 4 "$ROOT/dmd_reseed_seed${SEED}_2100_4step" "$STUDENT_PROTOCOL" \
  --checkpoint "$STUDENT" --first-interval-residual-bias-checkpoint "$BIAS"

# every arm must have drawn every case from the same prompt and latent
"$CRRA_PYTHON" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
arms = sorted(p for p in root.iterdir() if (p / "generation_manifest.jsonl").is_file())
tables = {a.name: {int(r["case_number"]): r for r in map(json.loads, (a / "generation_manifest.jsonl").read_text().splitlines()) if r}
          for a in arms}
assert len(tables) == 3 and all(len(t) == 75 for t in tables.values()), {k: len(v) for k, v in tables.items()}
for case in range(75):
    assert len({t[case]["latent_sha256"] for t in tables.values()}) == 1, case
    assert len({t[case]["prompt"] for t in tables.values()}) == 1, case
print("matched prompt and latent_sha256 for all 75 cases across", list(tables))
PY
echo "completed $ROOT"
