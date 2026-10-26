# awkward-matrix

Cross-backend correctness and performance benchmarks for Awkward Array.

Created and maintained by Ianna Osborne. The goal is to compare Awkward revisions,
backends, hardware, and dependency versions using reproducible workloads and
explicit correctness checks.

Measurements cover host latency, CUDA event time, GPU kernel activity,
synchronization, data transfers, and temporary allocations. Reports should
preserve the source revisions and environment information needed to reproduce
results. Benchmark tooling alone does not establish a speedup or backend parity.

The first implementation is proposed in a pull request importing the CUDA
migration validation harness. GPU execution and published results are separate
from implementation and local tests.

## License

BSD 3-Clause, copyright © 2026 Ianna Osborne. See [LICENSE](LICENSE).
Imported code retains its original attribution and applicable license notices.
