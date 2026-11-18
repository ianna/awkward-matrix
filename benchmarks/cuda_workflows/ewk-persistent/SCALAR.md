# Warmed scalar_sums diagnostic pass

From the existing ewkcoffea-della directory, in the activated environment:

```bash
sbatch --export=NONE ewk-persistent/run-della-scalar.sbatch
```

Uses the existing local 100k ROOT sample and metadata-only context overlay.
No larger input is required. It does not replace the unprofiled chunk sweep.

## Boundaries

Three chunks run through one persistent process per backend. Only the second
chunk's scalar_sums block is instrumented; chunk one warms the actual workflow.
The code verifies exact names/order of all eleven assignments in the pinned
processor, then inserts context managers around their original text at import
time. Original sources are untouched. The generated file is saved for review.
Assignments remain in the original function scope. This block includes mass,
delta-R, one explicit lepton sum, scalar additions, and the vector sum's pt.
It is broader than the name scalar_sums suggests.

A boundary fence/drain before and after the whole block excludes pending prior
work and includes completion. Inner statement durations are host submission
intervals including internal waits; they are not independently fenced device
times and must not be interpreted as kernel durations.

1. **host** runs CPU/CUDA with cProfile only inside that block on chunk two.
   Tables include full call counts and exclusive/cumulative host time. Focus
   categories identify compilation, construction, extraction/transfer, formatting,
   synchronization and reductions heuristically; the full function table remains
   available. No compile cache is cleared: new compile calls here would be real
   second-chunk work, potentially from data-dependent specialization.
2. **trace** is a separate fresh CUDA worker with the same three input ranges.
   Chunk one warms it. cudaProfilerStart/Stop enclose only chunk two's target
   block, with a parent NVTX range and eleven named statement ranges. Nsight
   records CUDA APIs, kernel names/counts and copy sizes/directions. Native CPU
   stacks are not sampled, so a long native compiler call may need a deeper
   follow-up. API durations can include waits; GPU times use the device trace.

Both passes compare merged histograms to the same CPU run, aligning categorical
labels. Host profiling and NVTX instrumentation perturb execution: these are
attribution diagnostics, never performance-gate timings. Failures retain earlier
files and abort the job; do not interpret a partial trace as a parity pass.

## Results

Replace JOBID:

```bash
cat results/JOBID/scalar-diagnostics/host/comparison.md
cat results/JOBID/scalar-diagnostics/trace-parity.md
cat results/JOBID/scalar-diagnostics/host/cuda/scalar-focus.txt
cat results/JOBID/scalar-diagnostics/host/cuda/scalar-statements.json
head -n 120 results/JOBID/scalar-diagnostics/host/cuda/scalar-profile.txt
cat results/JOBID/scalar-diagnostics/scalar-nsys-summary.txt
```

For per-statement device activity, open `scalar.nsys-rep` in Nsight Systems and
use the `scalar_sums/<statement>` NVTX ranges plus CUDA correlation IDs.
Cumulative cProfile entries and nested NVTX ranges overlap; do not sum them.

Local validation: instrumentation compiles against the exact pinned processor;
eleven harness tests pass, including statement/scope preservation and selecting
only chunk two. CUDA/Nsight execution remains to be verified on Della.
