#!/bin/bash -l
# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
#SBATCH --job-name=awkward-union-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=02:00:00
#SBATCH --output=awkward-union-final-%j.log

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

ROOT="$SLURM_SUBMIT_DIR"
TOOLS="$ROOT/awkward-matrix/benchmarks/cuda_migrations"
RESULTS="$ROOT/results/$SLURM_JOB_ID/unionarray-final"
export PATH="/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64:$PATH"
mkdir -p "$RESULTS"
hostname
nvidia-smi
nsys --version
for variant in baseline unionarray; do
    python -m pip install --no-index --no-deps --force-reinstall "$ROOT/wheels/$variant/"*.whl
    python "$ROOT/awkward-matrix/benchmarks/prepare_cuda.py" "$ROOT/$variant"
    python -m pip install --no-deps --no-build-isolation -e "$ROOT/$variant"
    python "$TOOLS/measure.py" run --checkout "$ROOT/$variant" \
      --contracts "$ROOT/unionarray" --suite unionarray --nsys --warmup 20 \
      --output "$RESULTS/kernels/$variant/report.json"
    python "$TOOLS/public_operations.py" run --checkout "$ROOT/$variant" \
      --suite unionarray --warmup 20 --iterations 50 --repeats 5 \
      --output "$RESULTS/public/$variant.json"
done
python "$TOOLS/measure.py" compare --baseline "$RESULTS/kernels/baseline/report.json" \
  --candidate "$RESULTS/kernels/unionarray/report.json" --output "$RESULTS/kernels/comparison"
python "$TOOLS/public_operations.py" compare --baseline "$RESULTS/public/baseline.json" \
  --candidate "$RESULTS/public/unionarray.json" --output "$RESULTS/public/comparison"
cat "$RESULTS/kernels/comparison.md"
cat "$RESULTS/public/comparison.md"
# Enforce the kernel gate after both reports have been written.
python "$TOOLS/measure.py" compare --baseline "$RESULTS/kernels/baseline/report.json" \
  --candidate "$RESULTS/kernels/unionarray/report.json" --output "$RESULTS/kernels/comparison" --require-pass
