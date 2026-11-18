#!/bin/bash -l
#SBATCH --job-name=ewk-context
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=01:00:00
#SBATCH --output=ewk-context-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
ROOT="$SLURM_SUBMIT_DIR"
OUT="$ROOT/results/$SLURM_JOB_ID/context-comparison"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
hostname > "$OUT/hostname.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
test "$(git -C "$ROOT/columnar_gpu" rev-parse HEAD)" = 7737ea0ba73c09c3345e43f65a17a3bc43d439a6
python - "$ROOT" "$OUT" <<'PY'
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import awkward as ak
assert ak.__version__ == '2.14.0', ak.__version__
root, out = map(Path, sys.argv[1:])
source = Path(ak.__file__).parent
for name in ('released', 'metadata'):
    shutil.copytree(source, out / name / 'awkward', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
candidate = out / 'metadata' / 'awkward'
shutil.copyfile(root / 'context_errors.py', candidate / '_errors.py')
p = candidate / '_kernels.py'
s = p.read_text()
old = 'error_context=ak._errors.ErrorContext.primary(),'
assert s.count(old) == 2, 'Release capture sites changed; inspect before patching'
p.write_text(s.replace(old, 'error_context=ak._errors.ErrorContext.cuda_primary(),'))
for name in ('_errors.py', '_kernels.py'):
    diff = difflib.unified_diff((source / name).read_text().splitlines(True),
                               (candidate / name).read_text().splitlines(True),
                               fromfile='released/' + name, tofile='metadata/' + name)
    (out / (name + '.diff')).write_text(''.join(diff))
(out / 'provenance.json').write_text(json.dumps({
    'release': ak.__version__, 'installed_source': str(source),
    'context_source_revision': '9cd52dcd',
    'sha256': {label: hashlib.sha256(path.read_bytes()).hexdigest()
               for label, path in [('release_errors', source / '_errors.py'),
                                   ('metadata_errors', candidate / '_errors.py'),
                                   ('release_kernels', source / '_kernels.py'),
                                   ('metadata_kernels', candidate / '_kernels.py')]},
    'scope': 'Error-context module refactor and two deferred CUDA capture sites only. All algorithm implementations and dependencies identical.'
}, indent=2))
PY
# Fresh process per trial; same device, dependencies and job-local CuPy cache.
# Each process warms both workloads before collecting any samples.
export CUPY_CACHE_DIR="$OUT/cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"
trial=0
for variant in released metadata metadata released released metadata; do
    trial=$((trial + 1))
    export CONTEXT_OVERLAY="$OUT/$variant" PYTHONPATH="$OUT/$variant"
    python "$ROOT/warmed_context.py" \
        --probe "$ROOT/columnar_gpu/ewkcoffea_on_gpu/reproduce/probe/triplets_probe.py" \
        --output "$OUT/trial-$trial-$variant" --events "${1:-1000000}" \
        2>&1 | tee "$OUT/trial-$trial-$variant.log"
done
unset PYTHONPATH
python - "$OUT" <<'PY' | tee "$OUT/comparison.md"
import json
from pathlib import Path
import statistics
import sys
root = Path(sys.argv[1])
print('# Released versus metadata-only CUDA context capture\n')
print('All trials use Awkward 2.14.0 algorithms; metadata variant changes only error context.\n')
print('Unprofiled completed-call medians, with workload JIT warmup and parity checks outside timing.\n')
print('| Collection | Released ms | Metadata ms | Metadata / released |')
print('| --- | ---: | ---: | ---: |')
for name in ('big', 'small'):
    values = {}
    for variant in ('released', 'metadata'):
        reports = [json.loads(p.read_text()) for p in sorted(root.glob('trial-*-'+variant+'/report.json'))]
        assert len(reports) == 3 and all(r['parity'] for r in reports)
        values[variant] = statistics.median(r['median_s'][name] for r in reports)
    a, b = values['released'], values['metadata']
    print(f'| {name} | {a*1000:.3f} | {b*1000:.3f} | {b/a:.3f} |')
print('\nParity passed: all six processes, both collections, every timed sample.')
print('Per-trial samples, first-use costs and separate warm profiles are retained; profiles are not benchmark timings.')
PY
echo "Results: $OUT"
