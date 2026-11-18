#!/bin/bash -l
#SBATCH --job-name=fill-serialize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=01:00:00
#SBATCH --output=fill-serialize-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
ROOT="$SLURM_SUBMIT_DIR"
OUT="$ROOT/results/$SLURM_JOB_ID/fill-serialization"
OVERLAY="${AWKWARD_OVERLAY:-$ROOT/results/16639248/context-bounded/candidate-overlay}"
test -f "$OVERLAY/awkward/_connect/cuda/_indexed_fill.py"
mkdir -p "$OUT"
export PYTHONPATH="$OVERLAY" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset SCALAR_DIAGNOSTICS DIAGNOSTIC_PHASE
hostname > "$OUT/hostname.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
conda list --explicit > "$OUT/conda-explicit.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
sha256sum "$OVERLAY/awkward/_connect/cuda/_indexed_fill.py" > "$OUT/helper.sha256"
export CUPY_CACHE_DIR="$OUT/cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"
python "$ROOT/ewk-fill-serialization/replay.py" \
 --cache-dir "$OUT/artifacts" --output "$OUT/replay" 2>&1 | tee "$OUT/replay.log"
cat "$OUT/replay/comparison.md"
