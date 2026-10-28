#!/bin/bash -l
# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
#SBATCH --job-name=awkward-public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=01:00:00
#SBATCH --output=awkward-public-%j.log

# Submit from CUDA_BENCH with --export=NONE. This job installs into the shared
# environment: wait for other CUDA benchmark jobs before submitting it.
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 gcc-toolset/14 cmake/3.30.8 cudatoolkit/12.9
conda activate awkward-cuda-bench
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BENCH_ROOT="$SLURM_SUBMIT_DIR"
HARNESS="$BENCH_ROOT/awkward-matrix/benchmarks/cuda_migrations/public_operations.py"
SUITE="${1:-listoffsetarray}"
case "$SUITE" in listoffsetarray|listarray|indexed|regulararray) ;; *) echo "Unknown suite: $SUITE" >&2; exit 2 ;; esac
RESULTS="$BENCH_ROOT/results/$SLURM_JOB_ID/public-$SUITE"
mkdir -p "$RESULTS"
hostname
nvidia-smi
for variant in baseline "$SUITE"; do
    python -m pip install --no-index --no-deps --force-reinstall \
      "$BENCH_ROOT/wheels/$variant/"*.whl
    python "$BENCH_ROOT/awkward-matrix/benchmarks/prepare_cuda.py" "$BENCH_ROOT/$variant"
    python -m pip install --no-deps --no-build-isolation -e "$BENCH_ROOT/$variant"
    python "$HARNESS" run --checkout "$BENCH_ROOT/$variant" \
      --suite "$SUITE" --output "$RESULTS/$variant.json" --warmup 20 --iterations 50 --repeats 5
done
python "$HARNESS" compare --baseline "$RESULTS/baseline.json" \
  --candidate "$RESULTS/$SUITE.json" --output "$RESULTS/comparison"
cat "$RESULTS/comparison.md"
