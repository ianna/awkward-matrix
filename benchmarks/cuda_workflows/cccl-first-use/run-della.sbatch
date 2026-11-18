#!/bin/bash -l
#SBATCH --job-name=cccl-first-use
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=01:00:00
#SBATCH --output=cccl-first-use-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ROOT="$SLURM_SUBMIT_DIR"
OUT="$ROOT/results/$SLURM_JOB_ID/cccl-first-use"
mkdir -p "$OUT"
hostname > "$OUT/hostname.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
conda list --explicit > "$OUT/conda-explicit.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
python "$ROOT/cccl-first-use/repro.py" --directory "$OUT/timings" \
    --processes 3 --size 4096 2>&1 | tee "$OUT/timings.txt"
# Separate diagnostic processes: do not merge these with the timings above.
python "$ROOT/cccl-first-use/repro.py" --directory "$OUT/profiles" \
    --processes 2 --size 4096 --profile 2>&1 | tee "$OUT/profiles.txt"
echo "Results: $OUT"
