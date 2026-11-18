#!/bin/bash -l
#SBATCH --job-name=fill-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=02:00:00
#SBATCH --output=fill-cache-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
ROOT="$SLURM_SUBMIT_DIR"
CODE="$ROOT/ewk-fill-cache-integration"
OUT="$ROOT/results/$SLURM_JOB_ID/fill-cache"
BASELINE="$ROOT/results/16639248/context-bounded/candidate-overlay"
OVERLAY="$OUT/overlay"
mkdir -p "$OUT"
python - "$BASELINE" "$OVERLAY" "$CODE" <<'PY'
from pathlib import Path
import shutil
import sys
baseline,overlay,code=map(Path,sys.argv[1:])
shutil.copytree(baseline/'awkward',overlay/'awkward',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('_indexed_fill.py','_compute_cache.py'):
 shutil.copyfile(code/name,overlay/'awkward/_connect/cuda'/name)
PY
export PYTHONPATH="$OVERLAY" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset SCALAR_DIAGNOSTICS DIAGNOSTIC_PHASE AWKWARD_CUDA_CACHE_DIR
hostname > "$OUT/hostname.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
conda list --explicit > "$OUT/conda-explicit.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
export CUPY_CACHE_DIR="$OUT/replay-cupy-cache"
python "$CODE/validate.py" --cache "$OUT/artifacts" --output "$OUT/replay" \
 2>&1 | tee "$OUT/replay.log"
# Unprofiled fresh-worker analysis pair, with prepopulated fill artifacts.
for mode in disabled enabled; do
 if [ "$mode" = enabled ]; then export AWKWARD_CUDA_CACHE_DIR="$OUT/artifacts"; else unset AWKWARD_CUDA_CACHE_DIR; fi
 PERSISTENT_OUT="$OUT/full-$mode" bash "$ROOT/ewk-persistent/run-della-persistent.sbatch" \
  "$ROOT/data/rep_100k-local.json" 20000 5 "$OVERLAY" 2>&1 | tee "$OUT/full-$mode.log"
done
python - "$OUT" <<'PY' | tee "$OUT/full-comparison.md"
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
print('| Cache | First chunk s | Warmed median s | Whole runner s |')
print('| --- | ---: | ---: | ---: |')
for mode in ('disabled','enabled'):
 s=json.loads((root/f'full-{mode}/cuda/summary.json').read_text())
 print(f"| {mode} | {s['first_chunk_s']:.3f} | {s['later_median_s']:.3f} | {s['wall_s']:.3f} |")
print('\nBoth passes completed CPU histogram parity. One sequential pair; only fill artifacts persisted.')
PY
