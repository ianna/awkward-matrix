#!/bin/bash -l
#SBATCH --job-name=ewk-basic-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=02:00:00
#SBATCH --output=ewk-basic-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
ROOT="$SLURM_SUBMIT_DIR"
CODE="$ROOT/ewk-persistent"
OUT="${SCALAR_OUT:-$ROOT/results/$SLURM_JOB_ID/basic-diagnostics}"
OVERLAY="${AWKWARD_OVERLAY:-$ROOT/results/16609714/indexed-fill-validation/candidate-overlay}"
INPUT="$ROOT/data/rep_100k-local.json"
REPRO="$ROOT/columnar_gpu/ewkcoffea_on_gpu/reproduce"
mkdir -p "$OUT"
export DIAGNOSTIC_PHASE=obj_basic_selection
# Existing unprofiled data: no instrumentation in these six passes.
for pair in 1 2 3; do
    for variant in baseline candidate; do
        python "$CODE/phases.py" "$ROOT/results/16610845/indexed-fill-repeat/pair-$pair/$variant" \
            > "$OUT/phases-pair-$pair-$variant.md"
    done
done
export SCALAR_DIAGNOSTICS=host PERSISTENT_OUT="$OUT/host"
# Profile only object selection in chunk two, after actual chunk-one warmup.
bash "$CODE/run-della-persistent.sbatch" "$INPUT" 20000 3 "$OVERLAY" \
    2>&1 | tee "$OUT/host.log"
# Separate trace process, with the same first-chunk warmup and input ranges.
mkdir -p "$OUT/trace"
ln -s "$OUT/host/cpu" "$OUT/trace/cpu"
export SCALAR_DIAGNOSTICS=trace EWKCOFFEA_BACKEND=cuda
export PYTHONPATH="$OVERLAY:$CODE" PERSISTENT_RESULTS="$OUT/trace/cuda"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export EWKCOFFEA_SKIP_MATERIALIZE=0
export CUPY_CACHE_DIR="$OUT/trace-cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"
export PATH="/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64:$PATH"
nsys --version > "$OUT/nsys-version.txt"
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-event-trace=false \
    --capture-range=cudaProfilerApi --capture-range-end=stop --wait=all \
    --force-overwrite=true --output="$OUT/basic" \
    python "$CODE/run_analysis_persistent.py" \
        --analysis "$REPRO/src/ewkcoffea/analysis/vbs_vvh/run_analysis.py" \
        --input "$OUT/host/input.json" --output "$OUT/trace/cuda" \
        --chunksize 20000 --chunks 3 2>&1 | tee "$OUT/trace.log"
python "$CODE/compare.py" "$OUT/trace" | tee "$OUT/trace-parity.md"
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    "$OUT/basic.nsys-rep" > "$OUT/basic-nsys-summary.txt" 2>&1
echo "Results: $OUT"
