#!/bin/bash -l
#SBATCH --job-name=indexed-fill-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=03:00:00
#SBATCH --output=indexed-fill-validate-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
ROOT="$SLURM_SUBMIT_DIR"
CODE="$ROOT/indexed-fill-runtime-validation"
OUT="$ROOT/results/$SLURM_JOB_ID/indexed-fill-validation"
BASELINE="$ROOT/results/16474896/context-comparison/metadata"
CANDIDATE="$OUT/candidate-overlay"
mkdir -p "$OUT"
unset PYTHONPATH SCALAR_DIAGNOSTICS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python - "$CODE" "$BASELINE" "$CANDIDATE" <<'PY'
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
code,baseline,candidate = map(Path,sys.argv[1:])
assert (baseline/'awkward/_errors.py').is_file()
shutil.copytree(baseline/'awkward',candidate/'awkward',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
source = candidate/'awkward/_connect/cuda/_compute.py'
text = source.read_text()
node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='awkward_IndexedArray_fill')
lines = text.splitlines(keepends=True)
replacement = (code/'compute_entry.py').read_text()
source.write_text(''.join(lines[:node.lineno-1])+replacement+''.join(lines[node.end_lineno:]))
shutil.copyfile(code/'_indexed_fill.py',source.with_name('_indexed_fill.py'))
(candidate/'provenance.json').write_text(json.dumps(dict(
    source_branch='ianna/cuda-indexed-fill-runtime-base',source_commit='7bcadcd8b',
    baseline=str(baseline),
    helper_sha256=hashlib.sha256((code/'_indexed_fill.py').read_bytes()).hexdigest(),
    scope='Only IndexedArray_fill entry point and new helper differ from metadata-only release overlay'),indent=2))
PY
export PYTHONPATH="$CANDIDATE" CUPY_CACHE_DIR="$OUT/validation-cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"
python "$CODE/validate.py" "$OUT/cache-replay" | tee "$OUT/cache-replay.md"
# Unprofiled paired full-analysis passes on this same allocation, before diagnostics.
for variant in baseline candidate; do
    if [ "$variant" = baseline ]; then overlay="$BASELINE"; else overlay="$CANDIDATE"; fi
    PERSISTENT_OUT="$OUT/full-$variant" \
        bash "$ROOT/ewk-persistent/run-della-persistent.sbatch" \
        "$ROOT/data/rep_100k-local.json" 20000 5 "$overlay" \
        2>&1 | tee "$OUT/full-$variant.log"
done
python - "$OUT" <<'PY' | tee "$OUT/full-comparison.md"
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
print('# Unprofiled full-analysis GPU tasks\n')
print('| Variant | First chunk s | Later median s | Whole runner s |')
print('| --- | ---: | ---: | ---: |')
for variant in ('baseline','candidate'):
    s=json.loads((root/f'full-{variant}/cuda/summary.json').read_text())
    print(f"| {variant} | {s['first_chunk_s']:.3f} | {s['later_median_s']:.3f} | {s['wall_s']:.3f} |")
print('\nBoth variants passed CPU histogram parity. One sequential pair on one allocation; inspect per-chunk samples.')
PY
# Separate warmed profile/trace pass against the exact same candidate overlay.
SCALAR_OUT="$OUT/scalar-diagnostics" AWKWARD_OVERLAY="$CANDIDATE" \
    bash "$ROOT/ewk-persistent/run-della-scalar.sbatch" 2>&1 | tee "$OUT/scalar.log"
echo "Results: $OUT"
