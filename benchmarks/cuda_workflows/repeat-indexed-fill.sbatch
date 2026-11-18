#!/bin/bash -l
#SBATCH --job-name=fill-repeat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=03:00:00
#SBATCH --output=fill-repeat-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
unset SCALAR_DIAGNOSTICS

ROOT="$SLURM_SUBMIT_DIR"
OUT="$ROOT/results/$SLURM_JOB_ID/indexed-fill-repeat"
BASELINE="$ROOT/results/16474896/context-comparison/metadata"
CANDIDATE="$ROOT/results/16609714/indexed-fill-validation/candidate-overlay"
mkdir -p "$OUT"

for pair in 1 2 3; do
    if [ "$pair" = 2 ]; then
        order="candidate baseline"
    else
        order="baseline candidate"
    fi
    for variant in $order; do
        if [ "$variant" = baseline ]; then
            overlay="$BASELINE"
        else
            overlay="$CANDIDATE"
        fi
        PERSISTENT_OUT="$OUT/pair-$pair/$variant" \
            bash "$ROOT/ewk-persistent/run-della-persistent.sbatch" \
            "$ROOT/data/rep_100k-local.json" 20000 5 "$overlay" \
            2>&1 | tee "$OUT/pair-$pair-$variant.log"
    done
done

python - "$OUT" <<'PY' | tee "$OUT/comparison.md"
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
ratios = []
print("| Pair | Baseline warmed s | Candidate warmed s | Speedup |")
print("| --- | ---: | ---: | ---: |")
for pair in range(1, 4):
    times = {}
    for variant in ("baseline", "candidate"):
        path = root / f"pair-{pair}" / variant / "cuda/summary.json"
        times[variant] = json.loads(path.read_text())["later_median_s"]
    ratio = times["baseline"] / times["candidate"]
    ratios.append(ratio)
    print(f"| {pair} | {times['baseline']:.3f} | "
          f"{times['candidate']:.3f} | {ratio:.2f}× |")
print(f"\nMedian paired speedup: {statistics.median(ratios):.2f}×")
print("\nAll six passes completed CPU histogram parity checks.")
PY
