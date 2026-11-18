#!/bin/bash -l
#SBATCH --job-name=ewk-chunk-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=04:00:00
#SBATCH --output=ewk-chunk-sweep-%j.log
set -eo pipefail
export SLURM_EXPORT_ENV=ALL
ROOT="$SLURM_SUBMIT_DIR"
INPUT="$ROOT/data/rep_1m-local.json"
OVERLAY="$ROOT/results/16474896/context-comparison/metadata"
OUT="$ROOT/results/$SLURM_JOB_ID/persistent-sweep"
CODE="$ROOT/ewk-persistent"
test -f "$INPUT"
test -f "$OVERLAY/awkward/_errors.py"
mkdir -p "$OUT"
for size in 100000 200000; do
    export PERSISTENT_OUT="$OUT/$size"
    # All entries, both backends, one worker each; same allocation for both sizes.
    bash "$CODE/run-della-persistent.sbatch" "$INPUT" "$size" 0 "$OVERLAY" \
        2>&1 | tee "$OUT/$size.log"
    python "$CODE/phases.py" "$PERSISTENT_OUT" | tee "$PERSISTENT_OUT/phase-comparison.md"
done
echo "Results: $OUT"
