#!/bin/bash
# Aggregate all completed COCO5K checkpoint metrics and redraw the curves.
#SBATCH --job-name=tl-c5k-summary
#SBATCH --partition=zen4_0768
#SBATCH --qos=zen4_0768
#SBATCH --nodes=1
#SBATCH --ntasks=190
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/data/fs201205/wt93384/project/CRRA/slurm/training-length-coco5k-summary-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/data/fs201205/wt93384/project/CRRA}"
cd "$PROJECT_ROOT"
export CRRA_ROOT="$PROJECT_ROOT"
source clusters/musica/env.sh
STUDY_ROOT="${TRAINING_LENGTH_ROOT:-$PROJECT_ROOT/evaluations/first_interval_residual_bias_training_length_100step_v1}"
"$CRRA_PYTHON" tools/summarize_first_interval_training_length_coco5k.py --root "$STUDY_ROOT"
