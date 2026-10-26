# CUDA migration validation (#4463)

This harness compares the eleven migration candidates on the same GPU,
Python environment, dependency versions, CUDA runtime and driver. It does not
supply benchmark results: real GPU execution is required. The two obsolete
sorting-range kernels are excluded (removal PR #4488).

| Suite | Candidate branch in ianna/awkward | PR |
| --- | --- | --- |
| indexed | ianna/cuda-compute-indexed-simplify | #4484 |
| listarray | ianna/cuda-compute-listarray | #4481 |
| listoffsetarray | ianna/cuda-compute-listoffsetarray | #4483 |
| regulararray | ianna/cuda-compute-regulararray | #4485 |
| unionarray | ianna/cuda-compute-unionarray | #4486 |

Upstream issue and PR numbers in this document refer to scikit-hep/awkward.

## Run on the GPU runner

The manually dispatched **CUDA migration comparison** workflow takes a suite,
candidate repository/ref, and baseline ref. Run each suite against its branch
above. The default baseline is `7c403344b574980bbb020f98dc2a7b1bd5f80324`.
The workflow installs each checkout's native library before running it, so CPU
parity uses matching source and native kernels. Dependencies stay unchanged
between runs; comparisons reject environment mismatches. It uploads JSON,
Nsight `.nsys-rep` and SQLite exports even if a comparison fails.

The self-hosted runner needs Nsight Systems (`nsys`) installed and permission to
trace CUDA. Use an otherwise idle GPU. Workflow concurrency serializes its own
runs, but does not exclude unrelated GPU jobs. The workflow must be merged into awkward-matrix’s default branch before normal
manual dispatch is possible, and a self-hosted GPU runner must be registered.

For local execution, keep this harness at a fixed absolute location outside
both checkouts. Install the existing GPU development dependencies and PyYAML.
Activate the **same environment** for both runs:

```bash
# Set these to absolute checkout paths.
HARNESS=/path/to/validation/benchmarks/cuda_migrations/measure.py
BASELINE=/path/to/baseline
CANDIDATE=/path/to/candidate

python -m pip install --no-deps --force-reinstall "$BASELINE/awkward-cpp"
python -m pip install --no-deps -e "$BASELINE"
python "$HARNESS" run --checkout "$BASELINE" --contracts "$CANDIDATE" \
  --suite listoffsetarray --nsys --output results/baseline/report.json

python -m pip install --no-deps --force-reinstall "$CANDIDATE/awkward-cpp"
python -m pip install --no-deps -e "$CANDIDATE"
python "$HARNESS" run --checkout "$CANDIDATE" --contracts "$CANDIDATE" \
  --suite listoffsetarray --nsys --output results/candidate/report.json

python "$HARNESS" compare --baseline results/baseline/report.json \
  --candidate results/candidate/report.json --output results/comparison.json \
  --require-pass
```

Use the candidate's contracts for **both** runs so they exercise the same edge
cases and dtype inventory. SHA-256 fingerprints reject differing spec/case data.
The workflow also runs the candidate PR's focused GPU tests, which cover public
operations and additional malformed/sliced inputs beyond the canonical cases.

## Measurements and interpretation

* **Parity:** canonical explicit cases across every supported specialization,
  comparing every buffer against the compiled CPU implementation, including
  unchanged inputs, output tails, in/out scratch, and partial writes on error.
  Returned errors compare message, id and attempt; legacy asynchronous exceptions
  expose only the message, and the report labels that limitation. Failed parity
  prevents performance comparisons for that workload. Canonical tests may expose
  known baseline defects; their failures remain visible in the baseline JSON.
* **Host latency:** wall time inside one warmed kernel call, including any
  synchronization that the implementation performs. Resetting inputs, external
  fences, compilation, validation, tracing, and allocation hooks are excluded.
* **Stream elapsed time:** CUDA events surrounding the call on a nondefault
  stream. This interval includes GPU idle gaps while Python submits work; it is
  not the sum of kernel execution times. Nsight separately reports the sum of
  GPU kernel durations (`kernel_busy_ns`) in an instrumented pass.
* **Total latency:** wall time through completion and the explicit asynchronous
  error check. The driver retains every repeat and reports medians, rather than
  hiding variability behind a single measurement.
* **Synchronization and transfers:** a fresh warmed process captures a single
  call inside the `measure` NVTX range. An outer `capture` range stays open
  until the stream finishes so asynchronous GPU activity is not truncated.
  API and allocation timestamps are restricted to `measure`, excluding that
  completion fence. Nsight reports runtime/driver API counts
  and durations, explicit synchronize calls, blocking memcpy API calls, and
  actual DToH transfer bytes (including array/UVM transfers). Harness reset,
  validation copies and final fences are outside capture. Runtime and driver
  layers may describe the same operation; counts are not unique barriers.
  Query polling and implicit synchronization in allocation/free APIs require
  inspecting the retained full API list/trace. A missing trace is unmeasured,
  never zero. CUDA event timings and API durations are recorded in separate
  passes and must not be added together.
* **Temporary allocations:** a separate warmed CuPy MemoryHook pass records
  requested counts/bytes (including pool hits), driver-backed pool growth,
  peak live newly requested pool blocks, and retained blocks. Inputs/outputs
  were allocated before this pass. Nsight memory tracing separately records
  driver allocation events, allocated bytes, peak live allocations originating
  inside capture, and their retained bytes, including allocations outside CuPy.
  Pool reuse can give zero driver allocations while still requesting temporary
  buffers. Do not add the pool and driver numbers: their storage can overlap.
  Neither peak is the process's total memory high-water mark. Unknown memory
  schemas/operations make allocation accounting incomplete rather than zero.

Workloads include 32 short rows, 4096 short rows, and 32 long rows. Warmup defaults
to 3 calls; timing defaults to 5 repeats of 20 independently reset calls.
Mutating scratch is reset before **every** measurement (especially UnionArray
`tmpstarts`), avoiding progressive offsets and out-of-bounds benchmark writes.
Nonlocal reduction timing uses prefix occupancy so the baseline's known
count-versus-last-position defect does not yield a misleading fast comparison;
canonical/focused parity tests independently exercise holes.

The comparison gate requires candidate parity, all workload timings/traces,
complete allocation accounting, and no timing median over the configurable
`--max-regression` ratio (default 1.10). This threshold flags investigation; it
is not a statistical significance test or automatic merge approval. Review
synchronization/space changes and repeat with more samples on an idle GPU. Keep
the raw implementation if representative workloads regress. Do not close #4463
merely because the tooling or implementation PRs exist.

## References

The implementation follows the documented [CuPy memory-hook callbacks](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryHook.html),
[Nsight capture and memory-tracing options](https://docs.nvidia.com/nsight-systems/UserGuide/),
and [Nsight SQLite schema](https://docs.nvidia.com/nsight-systems/AnalysisGuide/).
It avoids deprecated `cupyx.allow_synchronize`, which can miss synchronization
outside CuPy.

The deprecated `awkward_ListOffsetArray_reduce_local_nextparents_64` converter has no live layout callers and is excluded from parity/performance gates. Its raw ABI remains available in Awkward.

## Locate warmed host dispatch overhead

`dispatch.py` diagnoses `flatten_offsets` on the compute candidate. It is a
separate diagnostic, not a replacement for baseline/candidate performance gates.
All levels use the same preallocated buffers, nonzero-origin reverse gather,
item count, and nondefault stream:

1. Backend lookup plus kernel call.
2. A previously looked-up backend kernel object.
3. The `_compute` wrapper directly.
4. The `_listoffsetarray` implementation directly.
5. `cuda.compute.unary_transform` with a stable iterator/operator/stream adapter.
6. A retained object from `make_unary_transform`, using the same iterator/op.

Preparation/factory calls and 20 warmup calls per level precede timing. The
unprofiled pass rotates level order between repeats. Each host interval includes
only the call; CUDA events measure stream elapsed time, including submission
gaps. Completion/error-context draining is outside the host interval and happens
after every invocation, preventing accumulated error contexts from triggering
periodic internal fences. Completion latency is reported separately.

A subsequent cProfile pass saves both `.pstats` and readable cumulative/self-time
tables for each level. Completion fences are outside profiling as well. Timing
numbers do not include cProfile overhead. Inspect these profiles for time in
backend bookkeeping, argument materialization, op adapters, stream validation,
iterator state conversion, cache lookup, and native launch calls. Profiles can
also reveal compilation during the supposedly warmed phase; warmup alone is
not proof of cache stability. This diagnostic does not instrument compiler-cache
misses or implement fusion.

The retained direct operator has different setup/reuse behavior from the real
implementation, which constructs its gather closure and view inside each call.
Differences between levels are diagnostic, not additive isolated cost estimates
or a production optimization. Even the prepared API still performs Python-side
argument/state handling. State buffers stay alive for the entire run; reusing
these objects with different buffers would require explicit state updates.

Run inside an allocated GPU job after installing the matching candidate and
native wheel:

```bash
python benchmarks/cuda_migrations/dispatch.py \
  --checkout /path/to/candidate --output results/dispatch \
  --warmup 20 --iterations 100 --repeats 5 --profile-iterations 100
```

The output contains `dispatch.md`, `dispatch.json` (including environment and
repeat samples), and per-level `.pstats`/`.txt` profiles. No Nsight installation
is needed. The prepared-call signature was checked against CCCL Python 1.1.1;
the script fails explicitly if the factory is absent rather than substituting
a different measurement.

On Della, using the previously prepared CUDA_BENCH checkouts, wheels and Conda
environment, clone this repository there as `awkward-matrix`, then submit:

```bash
cd /home/iosborne/Projects/CUDA_BENCH
git clone --branch ianna/cuda-migration-validation \
  https://github.com/ianna/awkward-matrix.git awkward-matrix
sbatch --export=NONE \
  awkward-matrix/benchmarks/cuda_migrations/della-dispatch.sbatch
```

If that clone already exists, use `git -C awkward-matrix pull --ff-only` instead.
The Della script installs the matching existing candidate wheel and checkout,
uses the site's default partition selection, and writes to
`results/JOBNUMBER/dispatch/`. Do not run it concurrently with another job that
reinstalls packages in `awkward-cuda-bench`.

## Completed public operations

`public_operations.py` adds a separate warmed comparison of `ak.to_regular`
(including irregular inputs that must raise) and `ak.drop_none(axis=1)` on the
same three shapes as the kernel suite. CPU expected values and input transfer
are outside timing. Every measured call includes public result construction,
stream completion and Awkward deferred-error draining, even when the call itself
raises. Both paths must match CPU values or the expected error before a report
is accepted. Errors are classified after timing, with call-versus-drain counts
reported separately. JIT warmup is excluded; inputs remain alive and each call
creates a fresh result.

The existing kernel `total_us` already includes completion and error draining.
This comparison adds the surrounding public operation and its metadata/scalar
reads. It does not assume the synchronous candidate will catch up, replace the
kernel timing/trace/allocation reports, or relax their review gate. The separate
public report flags regressions in completed latency; host submission latency
remains visible. CUDA-event times include host submission gaps.

On Della, **after other jobs using the shared Conda environment finish**:

```bash
cd /home/iosborne/Projects/CUDA_BENCH
git -C awkward-matrix pull --ff-only
sbatch --export=NONE \
  awkward-matrix/benchmarks/cuda_migrations/della-public.sbatch
```

The job installs each existing wheel and checkout sequentially, runs separate
Python workers, and writes environment metadata, samples and comparison reports
to `results/JOBNUMBER/public/`. No Nsight installation is required. The current
candidate checkout revision is recorded; update that checkout before submitting
if necessary.

Elsewhere, install each checkout and its matching native wheel before its worker:

```bash
python benchmarks/cuda_migrations/public_operations.py run \
  --checkout /path/to/baseline --output results/public/baseline.json
# Install candidate and its matching native wheel, then:
python benchmarks/cuda_migrations/public_operations.py run \
  --checkout /path/to/candidate --output results/public/candidate.json
python benchmarks/cuda_migrations/public_operations.py compare \
  --baseline results/public/baseline.json --candidate results/public/candidate.json \
  --output results/public/comparison
```

## Regular-validation chunk sweep

`regular_chunks.py` compares chunks of 1, 2, 4, 8, 16, 32, 64 and 128 rows,
for 32, 4096 and 131072 rows and all three offset dtypes. It checks valid,
early-error and late-error cases against the CPU kernel before timing. Each
thread emits its first error only; partial final chunks are supported. A single
chunk writes status directly, avoiding reduction scratch and its launch.

The Awkward candidate now defaults to chunk 64, selected from the A100 sweep
on revision `7a465ee94`. Compared with chunk 1, completed latency improved by
about 37% at 32 rows and 6% at 4096 and 131072 rows, with substantially lower
pool peaks. Chunk 128 slowed the larger workloads. The sweep continues to
include chunk 1 as its explicit reference. This sweep times the internal validation call,
including its 8-byte host status read and completion. It excludes JIT warmup and
uses a separate allocation-hook pass for actual pool temporary peaks, including
CCCL workspace. The theoretical error-buffer byte count excludes status and
workspace. It does not substitute for public-operation or kernel-contract tests.

Update the candidate checkout to the chunk-capable revision, wait for other
jobs using the shared environment, then submit from CUDA_BENCH:

```bash
git -C awkward-matrix pull --ff-only
sbatch --export=NONE \
  awkward-matrix/benchmarks/cuda_migrations/della-regular-chunks.sbatch
```

Results are `results/JOBNUMBER/regular-chunks/regular-chunks.json` and `.md`.
Compare completed latency and pool peaks across both short and long inputs
before choosing a default; reductions in scratch bytes alone are insufficient.
