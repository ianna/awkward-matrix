# High-level API benchmark coverage

Run the inventory from the repository root with the Python interpreter for the
Awkward release you want to check. To compare coverage across releases, run the
same commands in each release's environment and use a different output path.

```bash
python -m pip install "awkward==2.14.0"
python -m benchmarks.highlevel.coverage --output coverage-2.14.0.json
python -m benchmarks.highlevel.coverage --smoke --size 32 \
  --output smoke-2.14.0.json
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
For example, PyArrow 20 works with Awkward 2.8.0 and 2.14.0 in the smoke
environment. The optional cases also include Pandas, Fastavro, Safetensors,
Numba, cppyy, JAX, PyTorch, TensorFlow, ROOT, CuPy, and cuDF workloads.

Add workloads to `cases.py` using the qualified API name. The inventory is
independent of the registry, so discovering a new API does not silently assign
a generic workload. Each case binds a zero-argument callable with deterministic
fixtures and a scratch directory. Bind a fresh case before each invocation of a
stateful workload, such as a builder mutation or a record field deletion.
The smoke check exercises the workloads once and records exceptions. It does
not measure runtime, memory, or numerical parity.
