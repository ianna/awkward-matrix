#!/bin/bash -l
# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
#SBATCH --job-name=awkward-public-diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=01:00:00
#SBATCH --output=awkward-public-diagnostics-%j.log

# No concurrent installs: this job mutates the shared Conda environment.
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 gcc-toolset/14 cmake/3.30.8 cudatoolkit/12.9
conda activate awkward-cuda-bench
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT="$SLURM_SUBMIT_DIR"
HARNESS="$ROOT/awkward-matrix/benchmarks/cuda_migrations/public_listarray_diagnostics.py"
RESULTS="$ROOT/results/$SLURM_JOB_ID/public-listarray-diagnostics"
mkdir -p "$RESULTS"
hostname
nvidia-smi
for variant in baseline listarray; do
    python -m pip install --no-index --no-deps --force-reinstall "$ROOT/wheels/$variant/"*.whl
    python -m pip install --no-deps --no-build-isolation -e "$ROOT/$variant"
    python "$HARNESS" run --checkout "$ROOT/$variant" --output "$RESULTS/$variant" "$@"
done
python "$HARNESS" compare --baseline "$RESULTS/baseline/diagnostics.json" \
  --candidate "$RESULTS/listarray/diagnostics.json" --output "$RESULTS/comparison"
cat "$RESULTS/comparison.md"
