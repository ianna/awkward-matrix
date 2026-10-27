#!/bin/bash -l
# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
#SBATCH --job-name=awkward-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=04:00:00
#SBATCH --output=awkward-matrix-%j.log

# One job at a time: installation mutates the shared Conda environment.
# First argument is the candidate checkout directory (e.g. listarray).
# Remaining arguments are passed to `matrix.py plan` (family/kernel/op selectors).
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 gcc-toolset/14 cmake/3.30.8 cudatoolkit/12.9
conda activate awkward-cuda-bench
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT="$SLURM_SUBMIT_DIR"
CANDIDATE="${1:?Pass candidate directory, e.g. listarray}"
shift
HARNESS="$ROOT/awkward-matrix/benchmarks/cuda_matrix/matrix.py"
RESULTS="$ROOT/results/$SLURM_JOB_ID/matrix"
mkdir -p "$RESULTS"
hostname
nvidia-smi
# Never regenerate inside workers: all three consume this exact manifest.
python "$HARNESS" plan --checkout "$ROOT/$CANDIDATE" --output "$RESULTS/manifest.json" "$@"
for mode in cpu raw compute; do
    variant="$CANDIDATE"
    if [ "$mode" = raw ]; then variant=baseline; fi
    python -m pip install --no-index --no-deps --force-reinstall "$ROOT/wheels/$variant/"*.whl
    python -m pip install --no-deps --no-build-isolation -e "$ROOT/$variant"
    python "$HARNESS" run --checkout "$ROOT/$variant" --manifest "$RESULTS/manifest.json" \
      --mode "$mode" --output "$RESULTS/$mode.json"
done
python "$HARNESS" compare --cpu "$RESULTS/cpu.json" --raw "$RESULTS/raw.json" \
  --compute "$RESULTS/compute.json" --output "$RESULTS/comparison"
