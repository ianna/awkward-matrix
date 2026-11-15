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
  --output measurements.json
```

Each case runs with three fixture sizes by default:

| Size | Fixture rows |
| --- | ---: |
| Small | 32 |
| Medium | 1,024 |
| Large | 32,768 |

Fixture row counts scale array inputs. Some public APIs have fixed-size inputs
or no array argument, such as an empty builder constructor. These retain their
natural workload at each size.

Use repeated `--case` globs and `--size` values to override the defaults. The runner
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

The default baseline is the release recorded in the saved run, or the oldest
release if the run does not specify one. Use `--baseline 2.15.0` to compare
against a different release in the run. The HTML displays coverage, absolute
values, observed timing ranges, sample counts, and comparable percentage changes.
Keep the raw JSON with the HTML to preserve individual measurements and their
environment metadata.

## Explore a saved run in Marimo

From the repository root, use [Pixi](https://pixi.sh) to launch the notebook
in an environment separate from the benchmark workers.

```bash
pixi run --locked -e notebook notebook \
  --measurements results/highlevel/2.6.0-2.15.0/measurements.json
```

The notebook can also open a measurement JSON file through its file picker.
Select an operation, fixture size, and baseline release. Switch between
absolute measurements and percentage changes. The plots and tables use the
same comparison logic as the saved HTML report.

In **Improved and degraded functions**, choose a comparison release, fixture
size, and metric, then switch between improved and degraded operations. The
table ranks the largest changes first and defaults to a 5% minimum absolute
change. Select a row to open that operation and fixture size in the plots.

The saved comparison covers 2.6.0, the first release of each minor version from
2.8 through 2.15, and 2.9.1 at all three fixture sizes. It uses 2.14.0 as the
default baseline.
Awkward 2.6.0 requires the NumPy 1.x environment used for its measurements.
Its absolute values remain visible when compared with releases measured under
NumPy 2.x. The report and notebook omit percentages across those environments.

The **Release by release** table summarizes consecutive releases in the loaded
run. Its counts use the selected metric, fixture size, and cutoff. Pairs with
different environments show no improvement or degradation counts.


## Run with Pixi or GitHub Actions

The Pixi task prepares isolated Python environments for the selected releases
and measures them sequentially on one machine. If selected, it clones
`scikit-hep/awkward` at `main`, builds the Python and native packages from the
same checkout, and records the exact commit in the results. Local source
builds require Git and a C++ compiler. The default dependency set uses
NumPy 1.26.4 to support Awkward 2.6.0 and records the resolved versions for
each target. Cases that require other optional dependencies remain visible
as unavailable.

```bash
pixi run --locked benchmark --releases 2.14.0,2.15.0 --profile full \
  --output results/highlevel/new-run
pixi run --locked -e notebook notebook \
  --measurements results/highlevel/new-run/measurements.json
```

Use `--no-include-main` for a release-only run, or `--baseline` to specify
the reference release. The quick profile uses five cases and the full profile
runs the API inventory. Both use all three sizes by default.

The **High-level Awkward benchmarks** workflow runs on request under GitHub
Actions. It takes the same release list, baseline, profile, and main toggle.
Download its artifact for the raw JSON, comparison HTML, and run log. CI and
local results are separate runs and can differ in hardware and RSS method.
