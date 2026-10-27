# CPU / RawKernel / cuda.compute matrix

This is independent of `cuda_migrations/`: it freezes a candidate's canonical
kernel inventory, imports its generated fixtures, and runs the same plan in three
**separate processes** against CPU, legacy GPU, and candidate GPU implementations.
Copyright 2026 Ianna Osborne; BSD-3-Clause; developed with AI assistance.

## Layout

- `catalog.py`: dispatch/spec discovery, generated automatic **and explicit** CPU
  fixture import, deterministic fixtures for kernels without generated examples.
- `operations.py`: public slicing, reductions/statistics, sorting, masks,
  broadcasting, combinations, unions/records, strings and layout transformations.
- `runner.py`: isolated backend execution, completion, parity snapshots and timing.
- `report.py`: three-way comparison, coverage gaps and completed latency ratios.
- `matrix.py`: CLI with independent kernel/family and public-operation selectors.
- `della-matrix.sbatch`: sequential installations and workers on Della.

The initial candidate inventory has 119 canonical kernel names, 429 dtype
specializations, and 12,543 kernel fixtures. These counts are **discovered**, not
hard-coded guarantees for other branches. Dedicated fixtures cover argsort,
fused real statistics, and padding where generated cases are absent. New
uncovered kernels appear as `missing_fixture`, never an empty successful test.
Generated files must already exist in the candidate checkout. Generate them with
Awkward's preparation workflow before planning; do not regenerate in each worker.

## Della: start with a selected family

Wait for other jobs: these installations share your `awkward-cuda-bench` Conda
environment. Do not modify checkouts while the job is running. The native wheel
must match each checkout; no version changes or wheel builds occur here.

From `/home/iosborne/Projects/CUDA_BENCH`:

```bash
git -C awkward-matrix pull --ff-only
# Candidate is an existing checkout directory, not a branch name.
sbatch --export=NONE awkward-matrix/benchmarks/cuda_matrix/della-matrix.sbatch \
  listarray --scope kernels --family ListArray
```

Specific kernel (repeat `--kernel` to select several):

```bash
sbatch --export=NONE awkward-matrix/benchmarks/cuda_matrix/della-matrix.sbatch \
  listarray --scope kernels --kernel 'awkward_ListArray_getitem_next_array*'
```

Public ListArray slicing, all three standard shapes:

```bash
sbatch --export=NONE awkward-matrix/benchmarks/cuda_matrix/della-matrix.sbatch \
  listarray --scope operations --operation 'slice_*' --layout list --dtype int64
```

Other examples: `--family 'reduce*'`, `--operation 'sum*'`, `--operation '*sort*'`,
`--layout option`, `--layout byte_mask`, `--layout bit_mask`, `--layout ragged`,
`--layout empty_lists`, `--layout nonzero_offsets`, `--layout categorical`,
`--dtype float32`, and repeated `--shape 32:8 --shape 4096:8 --shape 32:4096`.
Selectors use case-sensitive shell globs; quote them. Kernel selectors select
low-level cases; operation selectors independently select public workloads.
Unknown selectors fail. `--scope` selects `kernels`, `operations`, or `all`.

To plan everything for one candidate, omit selectors. This is a large run: split
by family/layout on a cluster rather than assume the four-hour limit suffices.
A candidate branch only contains migrations merged into that branch; use a
combined candidate checkout if you want to measure all pending migrations.
There is no automatic cherry-picking of other PRs.

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -n 80 awkward-matrix-JOB_ID.log
less results/JOB_ID/matrix/comparison.md
```

Results include the frozen manifest, CPU/raw/compute JSON reports, per-case
NumPy buffer archives, incremental `.jsonl` journals, and JSON/Markdown comparison.
Keep the array subdirectories with the JSON reports; comparisons read those
artifacts. Partial worker reports have `complete: false` and are rejected.

## Portable/manual execution

Install this repo's `requirements.txt` plus matching Awkward/awkward-cpp. GPU
workers additionally need CuPy, cuda.compute/numba-cuda, CUDA and an NVIDIA GPU.
String transformations may require PyArrow. Missing dependencies are visible
errors, not parity successes. Nsight is **not** required for this timing harness.

```bash
python benchmarks/cuda_matrix/matrix.py plan --checkout /path/to/candidate \
  --output results/manifest.json --family RegularArray --scope kernels

# Install candidate CPU/native packages in this interpreter first.
python benchmarks/cuda_matrix/matrix.py run --checkout /path/to/candidate \
  --manifest results/manifest.json --mode cpu --output results/cpu.json
# Install legacy revision and matching native package in another env/process.
python benchmarks/cuda_matrix/matrix.py run --checkout /path/to/legacy \
  --manifest results/manifest.json --mode raw --output results/raw.json
# Install candidate and matching native package.
python benchmarks/cuda_matrix/matrix.py run --checkout /path/to/candidate \
  --manifest results/manifest.json --mode compute --output results/compute.json
python benchmarks/cuda_matrix/matrix.py compare --cpu results/cpu.json \
  --raw results/raw.json --compute results/compute.json --output results/comparison
```

Worker options: `--warmup 5 --iterations 10 --repeats 3`. `--max-fixtures N`
limits fixtures per specialization for smoke tests; reports flag truncation and
cannot claim complete parity. Defaults execute every selected fixture.
Comparison is exact by default. Use explicit `--rtol 1e-6 --atol 1e-8` when an
application permits floating reduction differences; integer outputs/indices,
dtypes, shapes and boolean values remain exact. `--require-complete` returns a
nonzero exit code for incomplete or failed selected three-way parity.

## What is measured and checked

- Fixtures, GPU index transfers and CPU references are outside timing. JIT warmup
  is excluded. Fixed kernel buffers are restored and fenced before each call.
- Host latency includes the public call or cached kernel wrapper. Completed
  latency also includes stream completion and deferred error draining, even when
  the call itself raises. CUDA event time includes host submission gaps and is
  **not** pure kernel execution time.
- Correctness snapshots and stability checks are outside timing. Every warmup
  and measured call is checked against its preflight result/error outcome.
- Kernel parity compares **all** argument buffers, including untouched tails,
  mutated input buffers and partial output writes on errors. Pointer-table
  arguments retain their owners and compare pointee buffers, not addresses.
- Public parity compares result values/types; expected bounds/regularity failures
  compare semantic messages while preserving call/finish phase in the reports.
  Matching generic exceptions do not count as successful public parity.
- CPU is the candidate's compiled canonical implementation. The harness does not
  substitute Python mocks or change `kernel-specification.yml`.
- Raw mode requires real `CupyKernel` implementations and rejects compute
  fallback/direct compute APIs. If a kernel was never implemented as RawKernel,
  that leg is **unavailable**. No substitute is invented.
- Compute mode forces advertised compute implementations for low-level calls;
  other kernels used by a public operation keep their production dispatch.
  A separate, untimed pass records actually invoked kernel names and wrapper
  classes. A family benchmark therefore need not be a single-kernel measurement.

The catalogue has 132 workloads spanning 13 layouts and five dtypes. Numeric reducers additionally
exercise axes 0, 1, -1 and all axes; sort/argsort include descending and unstable
variants. Default shapes include small, many-short-list and few-long-list cases.
Combinatorial public fixtures are bounded to eight elements per list to avoid
quadratic memory growth. Some operation/layout/dtype combinations are invalid;
they remain visible as `not_applicable` or incomplete, not silently removed.

## Local validation

The initial implementation was tested with the candidate's native CPU library:
all 12,543 kernel fixtures completed (one warmup and one measured repetition),
and all 8,580 public workload/layout/dtype combinations ran at shape 3:4. Of those,
7,468 returned results; 1,112 raised errors, including invalid combinations,
expected bounds failures and unavailable optional dependencies. Those errors
remain explicit; this is not a claim of GPU parity. Unit/integration tests also
exercise completion on exceptions, dtype ABI mapping, pointer-table ownership,
RawKernel purity, selectors and fail-closed comparisons.

## Coverage is audited, not assumed

Discovery covers kernel-spec-backed dispatch entries in the selected candidate.
Direct high-level algorithms without a canonical kernel entry require public
workloads. Aliases without their own specification are listed separately.
`public_operations_without_fixture` lists public APIs outside the workload
catalogue, including metadata, I/O and third-party conversions: this harness does
not claim every public API is already benchmarked. Extend `operations.py` with
its required fixtures/dependencies to add one. Runtime coverage reports identify
selected kernels not reached by public workloads. Neither a family selection nor
one successful layout implies exhaustive parameter/axis/physics-workflow coverage.

Three-way selected-case parity, public-kernel coverage, missing fixtures,
truncation and dependency failures are distinct report fields. Timing regressions
do not change correctness outcomes. These timings do not measure synchronization
API counts, DToH volume, driver allocations or kernel busy time: use the separate
Nsight-capable `cuda_migrations` harness for those trace metrics. Never interpret
unmeasured metrics as zero.
