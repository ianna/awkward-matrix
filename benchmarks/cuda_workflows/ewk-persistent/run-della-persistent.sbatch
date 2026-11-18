#!/bin/bash -l
#SBATCH --job-name=ewk-persistent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=02:00:00
#SBATCH --output=ewk-persistent-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
ROOT="$SLURM_SUBMIT_DIR"
INPUT="${1:?Supply a local input JSON as argument one}"
CHUNKSIZE="${2:-20000}"
CHUNKS="${3:-5}"
OVERLAY="${4:-}"
OUT="${PERSISTENT_OUT:-$ROOT/results/$SLURM_JOB_ID/persistent}"
CODE="$ROOT/ewk-persistent"
REPRO="$ROOT/columnar_gpu/ewkcoffea_on_gpu/reproduce"
mkdir -p "$OUT"
unset PYTHONPATH
export PYTHONPATH="$CODE" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ -n "$OVERLAY" ]; then
    test -f "$OVERLAY/awkward/_errors.py"
    export PYTHONPATH="$OVERLAY:$CODE"
fi
export EWKCOFFEA_SKIP_MATERIALIZE=0
test "$(git -C "$ROOT/columnar_gpu" rev-parse HEAD)" = 7737ea0ba73c09c3345e43f65a17a3bc43d439a6
test "$(git -C "$REPRO/src/ewkcoffea" rev-parse HEAD)" = f3fc44625f5420813bed8bb4ca5c0a1be50a4403
# Validate and normalize file paths without accessing the network.
python - "$INPUT" "$OUT/input.json" <<'PY'
import json
from pathlib import Path
import sys
source = Path(sys.argv[1]).resolve()
data = json.loads(source.read_text())
assert isinstance(data.get('files'), list) and data['files'], 'Expected upstream sample JSON with a files list'
files = []
for file in data['files']:
    assert '://' not in file, 'Use local ROOT input for this job'
    p = Path(data.get('path', '')) / file
    if not p.is_absolute():
        p = source.parent / p
    p = p.resolve()
    assert p.is_file(), f'Missing ROOT input: {p}'
    files.append(str(p))
data['path'], data['files'] = '', files
Path(sys.argv[2]).write_text(json.dumps(data, indent=2))
PY
hostname > "$OUT/hostname.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
conda list --explicit > "$OUT/conda-explicit.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
for repo in coffea topcoffea ewkcoffea; do
    git -C "$REPRO/src/$repo" rev-parse HEAD > "$OUT/$repo-sha.txt"
    git -C "$REPRO/src/$repo" diff > "$OUT/$repo-local.diff"
done
python - <<'PY' > "$OUT/awkward.txt"
import awkward as ak
print(ak.__version__, ak.__file__)
PY
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 1 > "$OUT/gpu-memory.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT
export CUPY_CACHE_DIR="$OUT/cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"
for backend in cpu cuda; do
    export EWKCOFFEA_BACKEND="$backend"
    python "$CODE/run_analysis_persistent.py" \
        --analysis "$REPRO/src/ewkcoffea/analysis/vbs_vvh/run_analysis.py" \
        --input "$OUT/input.json" --output "$OUT/$backend" \
        --chunksize "$CHUNKSIZE" --chunks "$CHUNKS" \
        2>&1 | tee "$OUT/$backend.log"
done
python "$CODE/compare.py" "$OUT" | tee "$OUT/comparison.md"
echo "Results: $OUT"
