# Runtime-base IndexedArray fill validation

Production fix: ianna/awkward branch `ianna/cuda-indexed-fill-runtime-base`, commit
`7bcadcd8b`, based on upstream main. No canonical signature, YAML semantics, native
kernels, dependencies or awkward-cpp pin change.

The prepared transform consumes input values directly. It captures a metadata
proxy for an invocation-local int64 device parameter array instead of a Python
base scalar. A bounded cache specializes by device/context and input/output dtype,
not base value/type, pointers or length. Host rebinding/launch is locked; scratch
is per invocation and the current CUDA stream is passed explicitly. This adds an
8-byte parameter upload per nonempty invocation; measure that cost rather than
assuming constant-base calls improve. No host synchronization is introduced.

The batch creates a NEW overlay from the already-tested metadata-only release
copy and replaces only the IndexedArray_fill entry point plus its new helper.
It does not install packages or alter the old overlay. The exact helper hash and
branch revision are saved. This isolates this fix from other changes on main.

## Run on Della

Unpack both supplied folders into the existing ewkcoffea-della directory, then:

```bash
sbatch --export=NONE indexed-fill-runtime-validation/run-della.sbatch
```

No new ROOT sample required; uses the existing 100k input and local JSON.
The job performs sequentially:

1. GPU replay across all three canonical input dtypes. It requires observed cold
   builds/compiles, then asserts ZERO additional builds/compiles as base values,
   Python/NumPy scalar types, pointers and lengths vary. Checks negative handling,
   output casts, untouched prefix/tails and three interleaved nondefault streams.
2. Unprofiled full analysis, five 20k chunks per backend: old metadata overlay then
   fixed overlay, on the same allocation. Each variant must pass CPU histogram
   parity, entry-range and persistent-worker checks. These are one sequential
   before/after pair, not a statistically balanced repeated benchmark. First-use
   costs and later-chunk medians are reported separately.
3. Separate warmed scalar_sums host profile and CUDA trace against the fixed
   overlay, selecting only chunk two after actual first-chunk warmup. Host and
   trace results must pass histogram parity. These instrumented times are never
   merged with unprofiled timings.

A failed assertion stops the pipeline and preserves earlier output. No claim
of zero GPU builds or full-analysis speedup is made until this batch passes.
Local verification: 74 callback/contract cases, Ruff, pylint 10/10. GPU regression
tests on the branch additionally check no rebuild after warmup, normal backend
routing against CPU kernels and concurrent streams. Device validation is pending.

## Inspect

```bash
cat results/JOBID/indexed-fill-validation/cache-replay.md
cat results/JOBID/indexed-fill-validation/full-comparison.md
cat results/JOBID/indexed-fill-validation/scalar-diagnostics/host/cuda/scalar-focus.txt
cat results/JOBID/indexed-fill-validation/scalar-diagnostics/host/cuda/scalar-statements.json
cat results/JOBID/indexed-fill-validation/scalar-diagnostics/trace-parity.md
```

Nsight details remain in `scalar-diagnostics/scalar.nsys-rep` and
`scalar-nsys-summary.txt`. Compare against the original run 16606949, accounting
for instrumentation perturbation. Preserve current original unprofiled results
16605355 and cache-isolation run 16608260 as references.
