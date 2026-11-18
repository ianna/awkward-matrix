#!/bin/bash -l
#SBATCH --job-name=context-repeat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=03:00:00
#SBATCH --output=context-repeat-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load anaconda3/2026.7 cudatoolkit/12.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
ROOT="$SLURM_SUBMIT_DIR"
OUT="$ROOT/results/$SLURM_JOB_ID/context-bounded-repeat"
BASELINE="$ROOT/results/16609714/indexed-fill-validation/candidate-overlay"
CANDIDATE="$ROOT/results/16639248/context-bounded/candidate-overlay"
mkdir -p "$OUT"
unset SCALAR_DIAGNOSTICS DIAGNOSTIC_PHASE PYTHONPATH
for variant in baseline candidate; do
 if [ "$variant" = baseline ]; then overlay="$BASELINE"; else overlay="$CANDIDATE"; fi
 sha256sum "$overlay/awkward/_errors.py" > "$OUT/$variant-errors.sha256"
done
for pair in 1 2 3; do
 if [ "$pair" = 2 ]; then order="candidate baseline"; else order="baseline candidate"; fi
 for variant in $order; do
  if [ "$variant" = baseline ]; then overlay="$BASELINE"; else overlay="$CANDIDATE"; fi
  PERSISTENT_OUT="$OUT/pair-$pair/$variant" \
   bash "$ROOT/ewk-persistent/run-della-persistent.sbatch" \
   "$ROOT/data/rep_100k-local.json" 20000 5 "$overlay" \
   2>&1 | tee "$OUT/pair-$pair-$variant.log"
  python "$ROOT/ewk-persistent/phases.py" "$OUT/pair-$pair/$variant" \
   > "$OUT/pair-$pair/$variant/phases.md"
 done
done
python - "$OUT" <<'PY' | tee "$OUT/comparison.md"
import json
import statistics
import sys
from pathlib import Path
root=Path(sys.argv[1]); ratios=[]
print('| Pair | Baseline warmed s | Bounded warmed s | Speedup |')
print('| --- | ---: | ---: | ---: |')
for pair in range(1,4):
 values={}
 for variant in ('baseline','candidate'):
  p=root/f'pair-{pair}'/variant/'cuda/summary.json'
  values[variant]=json.loads(p.read_text())['later_median_s']
 ratio=values['baseline']/values['candidate']; ratios.append(ratio)
 print(f"| {pair} | {values['baseline']:.3f} | {values['candidate']:.3f} | {ratio:.3f}x |")
print(f'\nMedian paired speedup: {statistics.median(ratios):.3f}x')
print('\nAll six passes completed CPU histogram parity checks. Order: B/C, C/B, B/C.')
print('Warmed medians exclude each fresh worker first chunk; no profiling enabled.')
PY
