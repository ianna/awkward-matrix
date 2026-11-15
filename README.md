# awkward-matrix

Cross-backend correctness and performance benchmarks for Awkward Array.

Created and maintained by Ianna Osborne. The goal is to compare Awkward revisions,
backends, hardware, and dependency versions using reproducible workloads and
explicit correctness checks.

Measurements cover host latency, CUDA event time, GPU kernel activity,
synchronization, data transfers, and temporary allocations. Reports should
preserve the source revisions and environment information needed to reproduce
results. Benchmark tooling alone does not establish a speedup or backend parity.

## Explore release benchmarks in Marimo

Open the [Marimo notebook](benchmarks/highlevel/notebook.py) to compare runtime
and peak RSS across ten Awkward releases. The saved run includes small (32),
medium (1,024), and large (32,768) fixture sizes.

From the repository root, use Python 3.10 or later to set up and launch the notebook.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "marimo==0.25.2" "plotly==7.1.0"
.venv/bin/python -m marimo run benchmarks/highlevel/notebook.py
```

Open the local URL printed by Marimo. The notebook loads the saved measurements
automatically. Pick a fixture size and metric to see which functions improved
or degraded between consecutive releases. Select a ranked function to see its
plots, and click a release label on either plot to open its release notes.
Use **Open measurement JSON** to explore another run.

Browse the [saved HTML comparison](results/highlevel/2.6.0-2.15.0/comparison.html)
or [download the measurement JSON](results/highlevel/2.6.0-2.15.0/measurements.json).
The [benchmark instructions](benchmarks/highlevel/README.md) explain the coverage
checks, measurement method, and how to reproduce a run.

## CUDA migration validation

The initial harness compares eleven CUDA migration candidates between
an Awkward baseline and candidate checkout. It includes full-buffer parity,
warmed timing, Nsight tracing, and temporary-allocation accounting. The capture
includes GPU completion while excluding the harness completion fence from
measured API counts.

```bash
python -m pip install -r requirements.txt
python benchmarks/cuda_migrations/measure.py --help
```

The measured environment also needs the matching Awkward/native checkout,
CuPy, CUDA compute dependencies, and an NVIDIA GPU. Full trace measurements
require Nsight Systems. See the [run instructions and methodology](benchmarks/cuda_migrations/README.md)
for environment setup, baseline/candidate installation, and report comparison.

The manual GPU workflow requires a self-hosted GPU runner registered to this
repository with `nsys` available. It is not automatically executable merely
because the repository exists.

## Full CUDA kernel matrix

The separate [`benchmarks/cuda_matrix`](benchmarks/cuda_matrix/README.md) harness
compares canonical CPU kernels, legacy RawKernel implementations, and candidate
cuda.compute implementations. It discovers the candidate's dispatch inventory,
imports generated fixtures, measures completed public operations, and supports
kernel/family/operation/layout/dtype selectors. Missing implementations and
coverage gaps remain explicit.

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```

Native workload and integration checks additionally need a matching installed
Awkward checkout. Point them at its source tree:

```bash
AWKWARD_MATRIX_CHECKOUT=/path/to/awkward python -m pytest -q tests
```

GPU execution and published results are separate from implementation and local
tests. No performance improvement is claimed without measurements.

## Authorship

Created by Ianna Osborne for awkward-matrix, with AI assistance. Development
began on the `ianna/cuda-migration-validation` branch of her Awkward fork; these
benchmark files are maintained in this repository and are not being contributed
to Awkward. References such as #4463 refer to scikit-hep/awkward.

## License

BSD 3-Clause, copyright © 2026 Ianna Osborne. See [LICENSE](LICENSE).
