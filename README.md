# awkward-matrix

Cross-backend correctness and performance benchmarks for Awkward Array.

Created and maintained by Ianna Osborne. The goal is to compare Awkward revisions,
backends, hardware, and dependency versions using reproducible workloads and
explicit correctness checks.

Measurements cover host latency, CUDA event time, GPU kernel activity,
synchronization, data transfers, and temporary allocations. Reports should
preserve the source revisions and environment information needed to reproduce
results. Benchmark tooling alone does not establish a speedup or backend parity.

## CUDA migration validation

The initial harness compares twelve active CUDA migration candidates between
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

## Provenance

Imported from `ianna/awkward`'s `ianna/cuda-migration-validation` branch at
`59fafa29c`, with small adaptations for this standalone repository. Upstream
references such as #4463 refer to scikit-hep/awkward. See [NOTICE](NOTICE) for
retained upstream license and attribution. AI assistance was used in development.


## License

BSD 3-Clause, copyright © 2026 Ianna Osborne. See [LICENSE](LICENSE).
Imported code retains its original attribution and applicable license notices.
