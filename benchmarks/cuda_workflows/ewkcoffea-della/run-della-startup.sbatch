#!/bin/bash -l
#SBATCH --job-name=ewk-startup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=01:00:00
#SBATCH --output=ewk-startup-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${3:-ewkcoffea_della_repro}"
ROOT="$SLURM_SUBMIT_DIR"
OUT="$ROOT/results/$SLURM_JOB_ID/startup"
PROBE="$ROOT/columnar_gpu/ewkcoffea_on_gpu/reproduce/probe/triplets_probe.py"
SIZE="${1:-1000000}"
ORDER="${2:-big_first}"
mkdir -p "$OUT"
test "$(git -C "$ROOT/columnar_gpu" rev-parse HEAD)" = 7737ea0ba73c09c3345e43f65a17a3bc43d439a6
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
export PATH="/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64:$PATH"
hostname > "$OUT/hostname.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
conda list --explicit > "$OUT/conda-explicit.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
nsys --version > "$OUT/nsys-version.txt"
# One cache controlled here: all other compiler caches remain as configured.
export CUPY_CACHE_DIR="$OUT/cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"
python "$ROOT/probe_della.py" cpu "$ORDER" "$OUT/cpu.npy" --probe "$PROBE" --events "$SIZE" > "$OUT/cpu.log" 2>&1
for pass in empty-cache reused-cache host-profile cuda-trace; do
    case "$pass" in
        empty-cache|reused-cache) MODE=plain;;
        host-profile) MODE=profile;;
        cuda-trace) MODE=trace;;
    esac
    ARGS=(python "$ROOT/startup_profile.py" --probe "$PROBE" --output "$OUT/$pass" --events "$SIZE" --order "$ORDER" --mode "$MODE")
    if [ "$MODE" = trace ]; then
        nsys profile --trace=cuda,nvtx,osrt --sample=none \
            --cuda-event-trace=false --capture-range=cudaProfilerApi \
            --capture-range-end=stop --force-overwrite=true \
            --output="$OUT/startup" "${ARGS[@]}" 2>&1 | tee "$OUT/$pass.log"
        nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,osrt_sum \
            "$OUT/startup.nsys-rep" > "$OUT/nsys-summary.txt" 2>&1
    else
        "${ARGS[@]}" 2>&1 | tee "$OUT/$pass.log"
    fi
    python - "$OUT" "$pass" <<'PY' | tee -a "$OUT/parity.txt"
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
a = np.load(root / 'cpu.npy', allow_pickle=False)
b = np.load(root / sys.argv[2] / 'result.npy', allow_pickle=False)
assert a.shape == b.shape and a.dtype == b.dtype
np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-6, equal_nan=True)
print('PARITY PASSED', sys.argv[2])
PY
done
echo "Results: $OUT"
