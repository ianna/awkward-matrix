#!/bin/bash -l
#SBATCH --job-name=indexed-fill-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=00:30:00
#SBATCH --output=indexed-fill-cache-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
ROOT="$SLURM_SUBMIT_DIR"
OUT="$ROOT/results/$SLURM_JOB_ID/indexed-fill-cache"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/results/16474896/context-comparison/metadata"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CUPY_CACHE_DIR="$OUT/cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"
hostname > "$OUT/hostname.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
python "$ROOT/indexed-fill-cache/replay.py" --output "$OUT/replay" \
    2>&1 | tee "$OUT/replay.log"
cat "$OUT/replay/comparison.md"
