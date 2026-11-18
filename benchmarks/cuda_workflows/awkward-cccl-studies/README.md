# Earlier Awkward CCCL studies

These are unchanged source snapshots of the tracked studies/cccl files from
scikit-hep/awkward at 7c91097ef05b3df7fb37ce2d23268d45465551d0. Source paths and SHA-256 hashes are recorded in
../source-snapshots.json.

- bench_reduce_bool.py measures exposed cuda.compute kernels.
- benchmark.py compares physics-analysis implementations from playground.py.
- profile_cccl.py drives the same workloads under a host/CUDA profiler.
- probe_fusion_mechanism.py compares primitive dispatch and composed operations.
- bench_lazy_fusion.py is a CPU timing and launch-count proxy for a lazy-fusion
  prototype. It requires the corresponding prototype implementation.
- The remaining files supply segment algorithms, helpers and playground code.

Use the original compatible Awkward/prototype checkout, copying these files to
its studies/cccl directory when their instructions depend on that location.
Some scripts use experimental cuda.compute interfaces. Their original
dependencies and layout are required; this archive does not adapt them to the
current API.

The files retain their original bytes and formatting. They were syntax-checked
during publication, but their GPU execution was not repeated.
