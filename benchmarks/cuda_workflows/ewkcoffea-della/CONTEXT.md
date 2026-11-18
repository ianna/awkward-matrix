# Warmed released-versus-metadata context comparison

Uses the existing `ewkcoffea_della_repro` environment and the pinned upstream
probe. No package installs or network access occur in the batch job.

```bash
cd /home/iosborne/Projects/ewkcoffea-della
unzip -o ewkcoffea-context.zip
sbatch --export=NONE run-della-context.sbatch 1000000
```

Two job-local copies of installed Awkward 2.14.0 are created. The released copy
is unchanged. The metadata copy receives `_errors.py` from PR #4481 revision
9cd52dcd and exactly two `_kernels.py` substitutions, switching deferred CUDA
capture from `primary()` to `cuda_primary()`. This includes the lazy error-context
refactor required by metadata capture, but no migrated kernels or algorithm
changes. The exact diff and hashes are saved. This isolates context handling
from the other performance changes on that PR. It is not a comparison of the
entire PR against the release.

Six fresh processes run in R/M/M/R/R/M order, three per variant, on the same
allocated GPU with the same dependencies and a shared job-local CuPy cache.
Both collection workloads receive first-use and additional warmup calls before
nine unprofiled samples each. Collection order alternates within samples.
Times include the call, stream completion and deferred-error draining. Input
preparation, CPU reference calculation and parity checks are outside timing.
All warmup results and all sampled results are compared to CPU results.

Separate cProfile passes run after timing and write `warm_big` and `warm_small`
profiles. Use them to confirm that array pretty-printing/context formatting
has been removed from the candidate hot path. The profile passes are not
included in benchmark statistics. Other compiler caches are uncontrolled;
first-use timings are recorded for context, not treated as cold-cache benchmarks.

After completion, replace JOBID:

```bash
cat results/JOBID/context-comparison/comparison.md
head -n 60 results/JOBID/context-comparison/trial-1-released/warm_big-cumulative.txt
head -n 60 results/JOBID/context-comparison/trial-2-metadata/warm_big-cumulative.txt
```

`comparison.md` reports the median of the three process medians for each variant.
Inspect the per-process `report.json` samples before interpreting small changes.
A timing improvement is not inferred from profiling alone.

Local validation: shell syntax, Python syntax, lint, and five context tests
against the release source with this exact overlay pass. GPU execution remains
for Della. This measures valid triplet outputs; it is not a new exhaustive test
of exception messages.
