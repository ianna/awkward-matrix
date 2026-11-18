#!/bin/bash -l
#SBATCH --job-name=ewk-context-bounded
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=03:00:00
#SBATCH --output=ewk-context-bounded-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
ROOT="$SLURM_SUBMIT_DIR"
CODE="$ROOT/ewk-context-bounded"
OUT="$ROOT/results/$SLURM_JOB_ID/context-bounded"
BASELINE="$ROOT/results/16609714/indexed-fill-validation/candidate-overlay"
CANDIDATE="$OUT/candidate-overlay"
mkdir -p "$OUT"
unset SCALAR_DIAGNOSTICS DIAGNOSTIC_PHASE PYTHONPATH
python - "$BASELINE" "$CANDIDATE" "$CODE" <<'PY'
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path
baseline,candidate,code = map(Path,sys.argv[1:])
assert (baseline/'awkward/_connect/cuda/_indexed_fill.py').is_file()
shutil.copytree(baseline/'awkward',candidate/'awkward',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=candidate/'awkward/_errors.py'
s=p.read_text(); tree=ast.parse(s)
n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cuda_argument')
lines=s.splitlines(keepends=True)
replacement=(code/'cuda_argument.py').read_text()
p.write_text(''.join(lines[:n.lineno-1])+replacement+''.join(lines[n.end_lineno:]))
# Ensure the entire change is confined to this function.
a=ast.parse(s); b=ast.parse(p.read_text())
for t in (a,b):
 t.body=[n for n in t.body if not(isinstance(n,ast.FunctionDef) and n.name=='_cuda_argument')]
assert ast.dump(a)==ast.dump(b)
(candidate/'provenance.json').write_text(json.dumps(dict(baseline=str(baseline),scope='_cuda_argument only',sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))
PY
# Unprofiled CPU/CUDA full-analysis parity precedes the separate diagnostic pass.
for variant in baseline candidate; do
 if [ "$variant" = baseline ]; then overlay="$BASELINE"; else overlay="$CANDIDATE"; fi
 PERSISTENT_OUT="$OUT/full-$variant" bash "$ROOT/ewk-persistent/run-della-persistent.sbatch" \
   "$ROOT/data/rep_100k-local.json" 20000 5 "$overlay" 2>&1 | tee "$OUT/full-$variant.log"
 python "$ROOT/ewk-persistent/phases.py" "$OUT/full-$variant" > "$OUT/phases-$variant.md"
done
python - "$OUT" <<'PY' | tee "$OUT/comparison.md"
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]); timings={}
print('| Variant | First chunk s | Warmed median s | Whole runner s |')
print('| --- | ---: | ---: | ---: |')
for variant in ('baseline','candidate'):
 s=json.loads((root/f'full-{variant}/cuda/summary.json').read_text())
 timings[variant]=s['later_median_s']
 print(f"| {variant} | {s['first_chunk_s']:.3f} | {s['later_median_s']:.3f} | {s['wall_s']:.3f} |")
print(f"\nWarmed speedup: {timings['baseline']/timings['candidate']:.2f}x")
print('\nBoth full analyses passed CPU histogram parity. One sequential pair, not repeated trials.')
PY
SCALAR_OUT="$OUT/basic-diagnostics" AWKWARD_OVERLAY="$CANDIDATE" \
 bash "$ROOT/ewk-persistent/run-della-basic.sbatch" 2>&1 | tee "$OUT/basic.log"
echo "Results: $OUT"
