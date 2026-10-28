#!/bin/bash -l
# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
#SBATCH --job-name=awkward-indexed-stages
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=01:00:00
#SBATCH --output=awkward-indexed-stages-%j.log

# Submit from CUDA_BENCH with --export=NONE. Do not run alongside another job
# that installs into the same awkward-cuda-bench environment.
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 gcc-toolset/14 cmake/3.30.8 cudatoolkit/12.9
conda activate awkward-cuda-bench
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BENCH_ROOT="$SLURM_SUBMIT_DIR"
CANDIDATE="$BENCH_ROOT/indexed"
HARNESS="$BENCH_ROOT/awkward-matrix/benchmarks/cuda_migrations/indexed_stages.py"
RESULTS="$BENCH_ROOT/results/$SLURM_JOB_ID/indexed-stages"
mkdir -p "$RESULTS"
hostname
nvidia-smi
python -m pip install --no-index --no-deps --force-reinstall \
  "$BENCH_ROOT/wheels/indexed/"*.whl
python "$BENCH_ROOT/awkward-matrix/benchmarks/prepare_cuda.py" "$CANDIDATE"
python -m pip install --no-deps --no-build-isolation -e "$CANDIDATE"
python "$HARNESS" --checkout "$CANDIDATE" --output "$RESULTS" \
  --warmup 20 --iterations 100 --repeats 5 --profile-iterations 100 "$@"
