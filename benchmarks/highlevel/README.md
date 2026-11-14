# High-level API benchmark coverage

Run the inventory from the repository root with the Python interpreter for the
Awkward release you want to check. To compare coverage across releases, run the
same commands in each release's environment and use a different output path.

```bash
python -m pip install "awkward==2.15.0"
python -m benchmarks.highlevel.coverage --output coverage-2.15.0.json
python -m benchmarks.highlevel.coverage --smoke --size 32 \
  --output smoke-2.15.0.json
```

The inventory discovers the Awkward-owned functions exported by `ak` and
`ak.str`, plus the constructors, public members, and selected documented Python
and interoperability protocols of `ak.Array`, `ak.Record`, and
`ak.ArrayBuilder`. It inspects classes without calling their properties.
Aliases have separate entries so a removed or newly exposed spelling is visible.

| Status | Meaning |
| --- | --- |
| `mapped` | The release exports the API and the registry defines a workload. |
| `smoke_passed` | The workload ran once and returned without raising. |
| `absent` | The registry has a workload for an API this release does not export. |
| `unavailable` | A package required by the workload could not be imported, or CUDA was unavailable. |
| `missing_case` | The release exports an API without a registered workload. |
| `failed` | Setup or execution raised an exception. The report includes its type and message. |

The command writes JSON to standard output by default. It exits with status 1
for missing cases or failed workloads and still saves the report when an
output path is supplied. Install optional packages to exercise their cases.
For example, PyArrow 20 works with Awkward 2.8.0, 2.14.0, and 2.15.0 in
the smoke environment. The optional cases also include Pandas, Fastavro, Safetensors,
Numba, cppyy, JAX, PyTorch, TensorFlow, ROOT, CuPy, and cuDF workloads.

The Awkward 2.15.0 smoke run passed 268 workloads. Twelve required unavailable
optional dependencies, and four registry entries were absent from the release.
All APIs discovered in 2.15.0 had workloads.

Add workloads to `cases.py` using the qualified API name. The inventory is
independent of the registry, so discovering a new API does not silently assign
a generic workload. Each case binds a zero-argument callable with deterministic
fixtures and a scratch directory. Bind a fresh case before each invocation of a
stateful workload, such as a builder mutation or a record field deletion.
The smoke check exercises the workloads once and records exceptions. It does
not measure runtime, memory, or numerical parity.

## Measure across releases

Use a separate environment for each release. Install matching dependency
versions in each environment so the results can be compared. The runner records
the package versions and sets `matching_environments` to false when Python,
platform, thread settings, or installed package versions differ. Awkward and
awkward-cpp versions are recorded separately.

```bash
python -m benchmarks.highlevel.measure \
  --python /path/to/2.14.0/bin/python \
  --python /path/to/2.15.0/bin/python \
  --size 1024 --output measurements.json
```

Use repeated `--case` globs and `--size` values to select workloads. The runner
measures one child process at a time and rotates the first release for each
case and size. Timing samples exclude fixture preparation and use three
warmups and seven trials by default. The JSON retains the individual timings
and the median. CUDA cases appear as unavailable in this CPU runner.

Memory samples use a fresh process for each trial and include native allocations
in process RSS. On Linux, when permitted, the runner resets the RSS high-water
mark to current RSS after setup and subtracts that baseline from the operation's
peak. Otherwise it reports the increase over the setup high-water mark. A zero
in the latter mode means the operation did not exceed that mark. Compare memory
samples only when their methods match. Use `--memory-repeats` to collect more
than the default one trial per workload.

The runner updates the output after each workload across all releases. It
preserves unavailable, absent, failed, and partially measured cases. A missing
case or measurement failure produces a nonzero exit status.

## Save an HTML comparison

```bash
python -m benchmarks.highlevel.report measurements.json --output comparison.html
```

The oldest release is the default baseline. Use `--baseline 2.15.0` to compare
against a different release in the run. The HTML displays coverage, absolute
values, observed timing ranges, sample counts, and comparable percentage changes.
Keep the raw JSON with the HTML to preserve individual measurements and their
environment metadata.

## Explore a saved run in Marimo

From the repository root, install the notebook dependencies in an environment
separate from the benchmark environments.

```bash
python -m venv .venv
.venv/bin/python -m pip install marimo plotly
.venv/bin/python -m marimo run benchmarks/highlevel/notebook.py -- \
  --measurements results/highlevel/2.14.0-2.15.0/measurements.json
```

The notebook can also open a measurement JSON file through its file picker.
Select an operation, fixture size, and baseline release. Switch between
absolute measurements and percentage changes. The plots and tables use the
same comparison logic as the saved HTML report.

In **Improved and degraded functions**, choose a comparison release, fixture
size, and metric, then switch between improved and degraded operations. The
table ranks the largest changes first and defaults to a 5% minimum absolute
change. Select a row to open that operation and fixture size in the plots.
