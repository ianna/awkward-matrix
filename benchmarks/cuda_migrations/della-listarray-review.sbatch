#!/bin/bash -l
# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
#SBATCH --job-name=awkward-listarray-review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=03:00:00
#SBATCH --output=awkward-listarray-review-%j.log

# Submit from CUDA_BENCH with --export=NONE. This installs into the shared
# environment; wait for other benchmark jobs before submitting.
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 gcc-toolset/14 cmake/3.30.8 cudatoolkit/12.9
conda activate awkward-cuda-bench
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PATH="/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64:$PATH"
ROOT="$SLURM_SUBMIT_DIR"
TOOLS="$ROOT/awkward-matrix/benchmarks/cuda_migrations"
RESULTS="$ROOT/results/$SLURM_JOB_ID/listarray-review"
mkdir -p "$RESULTS"
hostname
nvidia-smi
python -m pip freeze > "$RESULTS/environment.txt"
git -C "$ROOT/baseline" rev-parse HEAD > "$RESULTS/baseline-sha.txt"
git -C "$ROOT/listarray" rev-parse HEAD > "$RESULTS/candidate-sha.txt"
python -m pip install 'scikit-build-core>=1.0' 'pybind11>=3' pyyaml
# Conda activation or an earlier wheel build may name a compiler that is not
# installed on this compute node. Resolve the module's tools from PATH and do
# not reuse the checkout's build/cpython-* CMake cache.
export CC="$(type -P gcc || type -P cc)"
export CXX="$(type -P g++ || type -P c++)"
if [ ! -x "$CC" ] || [ ! -x "$CXX" ]; then
    echo "No working C/C++ compiler found on the compute node" >&2
    exit 1
fi
{
    printf 'CC=%s\nCXX=%s\n' "$CC" "$CXX"
    "$CC" --version
    "$CXX" --version
} > "$RESULTS/compilers.txt"
cat "$RESULTS/compilers.txt"
# Fail before profiling the baseline if the candidate cannot be built. Keep
# the wheel and complete build log with this job's results for diagnosis.
python -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "$RESULTS/wheels" \
    --config-settings="build-dir=$RESULTS/cpu-build" \
    --config-settings="cmake.define.CMAKE_C_COMPILER=$CC" \
    --config-settings="cmake.define.CMAKE_CXX_COMPILER=$CXX" \
    "$ROOT/listarray/awkward-cpp" 2>&1 | tee "$RESULTS/cpu-build.log"
for variant in baseline listarray; do
    if [ "$variant" = baseline ]; then
        python -m pip install --no-index --no-deps --force-reinstall "$ROOT/wheels/baseline/"*.whl
    else
        # The padding/size fixes require freshly built CPU kernels, not the old wheel.
        python -m pip install --no-index --no-deps --force-reinstall "$RESULTS/wheels/"*.whl
    fi
    python "$ROOT/awkward-matrix/benchmarks/prepare_cuda.py" "$ROOT/$variant"
    python -m pip install --no-deps --no-build-isolation -e "$ROOT/$variant"
    if [ "$variant" = listarray ]; then
        (
            cd "$ROOT/listarray"
            python -m pytest -q tests/test_4481_listarray_compute.py \
                tests/test_4481_cuda_error_context.py tests-cuda/test_4481_listarray_compute.py \
                tests-cuda/test_4481_cuda_error_context.py tests-cuda/test_4481_indexed_nullcount_stream.py \
                --junitxml="$RESULTS/contracts.xml"
            python -m pytest -q tests-cuda-kernels-explicit tests-cuda-kernels-explicit-parametrized \
                -k ListArray --junitxml="$RESULTS/generated-cuda.xml"
        )
    fi
    python "$TOOLS/listarray_review.py" run --checkout "$ROOT/$variant" --output "$RESULTS/$variant-review.json"
    python "$TOOLS/measure.py" run --checkout "$ROOT/$variant" --contracts "$ROOT/listarray" \
        --suite listarray --nsys --warmup 20 --output "$RESULTS/kernels/$variant/report.json"
    python "$TOOLS/public_operations.py" run --checkout "$ROOT/$variant" --suite listarray \
        --warmup 20 --iterations 50 --repeats 5 --output "$RESULTS/public/$variant.json"
done
python "$TOOLS/listarray_review.py" compare --baseline "$RESULTS/baseline-review.json" \
    --candidate "$RESULTS/listarray-review.json" --output "$RESULTS/review.md"
python "$TOOLS/measure.py" compare --baseline "$RESULTS/kernels/baseline/report.json" \
    --candidate "$RESULTS/kernels/listarray/report.json" --output "$RESULTS/kernels/comparison"
python "$TOOLS/public_operations.py" compare --baseline "$RESULTS/public/baseline.json" \
    --candidate "$RESULTS/public/listarray.json" --output "$RESULTS/public/comparison"
cat "$RESULTS/review.md" "$RESULTS/kernels/comparison.md" "$RESULTS/public/comparison.md"
