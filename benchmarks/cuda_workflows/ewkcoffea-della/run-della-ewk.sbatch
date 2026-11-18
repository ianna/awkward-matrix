#!/bin/bash -l
# Della adaptation; upstream sources are cloned and pinned, not modified.
#SBATCH --job-name=ewkcoffea-della
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=03:00:00
#SBATCH --output=ewkcoffea-%j.log

set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
ROOT="$SLURM_SUBMIT_DIR"
MODE="${1:-probe}"
SIZE="${2:-1000000}"
ENV_NAME="${3:-ewkcoffea_della_repro}"
INPUT_JSON="${4:-}"
REVISION=7737ea0ba73c09c3345e43f65a17a3bc43d439a6
REPO="$ROOT/columnar_gpu"
REPRO="$REPO/ewkcoffea_on_gpu/reproduce"
OUT="$ROOT/results/$SLURM_JOB_ID"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Do not accidentally import the other benchmark checkout or sitecustomize.
unset PYTHONPATH

if [ "$MODE" = setup ]; then
    if [ ! -d "$REPO/.git" ]; then
        git clone https://github.com/kmohrman/columnar_gpu.git "$REPO"
        git -C "$REPO" switch --detach "$REVISION"
    fi
    if [ "$(git -C "$REPO" rev-parse HEAD)" != "$REVISION" ]; then
        echo "columnar_gpu is not at the reviewed revision; inspect before changing it" >&2
        exit 1
    fi
    bash "$REPRO/setup.sh" "$ENV_NAME"
    conda activate "$ENV_NAME"
    # Lock the released Awkward baseline explicitly; upstream only asks for >=2.14.
    python -m pip install 'awkward==2.14.0'
else
    conda activate "$ENV_NAME"
fi

test "$(git -C "$REPO" rev-parse HEAD)" = "$REVISION"
hostname | tee "$OUT/hostname.txt"
nvidia-smi | tee "$OUT/nvidia-smi.txt"
lscpu > "$OUT/cpu.txt"
conda list --explicit > "$OUT/conda-explicit.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
git -C "$REPO" rev-parse HEAD > "$OUT/reproducer-sha.txt"
for repo in coffea topcoffea ewkcoffea; do
    git -C "$REPRO/src/$repo" rev-parse HEAD > "$OUT/$repo-sha.txt"
done
python - <<'PY' | tee "$OUT/runtime.txt"
import awkward as ak
import cupy as cp
print("awkward", ak.__version__, ak.__file__)
print("CUDA driver", cp.cuda.runtime.driverGetVersion())
print("CUDA runtime", cp.cuda.runtime.runtimeGetVersion())
assert cp.cuda.runtime.getDeviceCount() == 1, "Expected one allocated visible GPU"
assert int(cp.arange(10).sum().get()) == 45
PY

if [ "$MODE" = setup ]; then
    echo "Setup complete: $ENV_NAME"
    exit 0
fi

nvidia-smi --query-gpu=timestamp,memory.used,memory.total \
    --format=csv -l 1 > "$OUT/gpu-memory.csv" 2>/dev/null &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

case "$MODE" in
    probe)
        for order in big_first small_first; do
            python "$ROOT/probe_della.py" cpu "$order" "$OUT/cpu-$order.npy" \
                --probe "$REPRO/probe/triplets_probe.py" --events "$SIZE" \
                2>&1 | tee "$OUT/cpu-$order.log"
            # Separate initially empty CuPy disk cache per order. Each repeat
            # is a fresh process; repeats 2/3 reuse that order's disk cache.
            export CUPY_CACHE_DIR="$OUT/cupy-cache-$order"
            mkdir -p "$CUPY_CACHE_DIR"
            for repeat in 1 2 3; do
                python "$ROOT/probe_della.py" cuda "$order" "$OUT/cuda-$order-$repeat.npy" \
                    --probe "$REPRO/probe/triplets_probe.py" --events "$SIZE" \
                    2>&1 | tee "$OUT/cuda-$order-$repeat.log"
            done
        done
        python - "$OUT" <<'PY' | tee "$OUT/parity.txt"
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
for order in ("big_first", "small_first"):
    cpu = np.load(root / f"cpu-{order}.npy", allow_pickle=False)
    for repeat in (1, 2, 3):
        gpu = np.load(root / f"cuda-{order}-{repeat}.npy", allow_pickle=False)
        assert gpu.shape == cpu.shape
        assert gpu.dtype == cpu.dtype
        np.testing.assert_allclose(gpu, cpu, rtol=1e-9, atol=1e-6, equal_nan=True)
        print(f"PARITY PASSED {order} repeat={repeat}")
PY
        ;;
    full)
        case "$SIZE" in 1k|100k|1m|5m) ;; *) echo "Use full 1k|100k|1m|5m" >&2; exit 2;; esac
        # Give every job its own upstream logs/outputs, so summaries do not mix jobs.
        mkdir -p "$OUT/full/input_jsons"
        cp "$REPRO/run_repro.sh" "$OUT/full/run_repro.sh"
        cp "$REPRO/input_jsons/"*.json "$OUT/full/input_jsons/"
        export EWKCOFFEA_DIR="$REPRO/src/ewkcoffea"
        if [ -n "$INPUT_JSON" ]; then
            case "$INPUT_JSON" in /*) export REPRO_INPUT="$INPUT_JSON";; *) export REPRO_INPUT="$ROOT/$INPUT_JSON";; esac
        fi
        cd "$OUT/full"
        for repeat in 1 2 3; do
            # Consistent cold CuPy disk-cache policy for the full workflow.
            # Alternate CPU/GPU order to reduce systematic host-cache bias.
            export CUPY_CACHE_DIR="$OUT/cupy-cache-full-$repeat"
            mkdir -p "$CUPY_CACHE_DIR"
            if [ "$repeat" = 2 ]; then backends="cuda cpu"; else backends="cpu cuda"; fi
            for backend in $backends; do
                bash ./run_repro.sh "$backend" "$SIZE"
            done
        done
        python "$REPRO/summarize.py" --logs "$OUT/full/logs" --out "$OUT/full"
        ;;
    *) echo "Use setup, probe or full" >&2; exit 2 ;;
esac
echo "Results: $OUT"
