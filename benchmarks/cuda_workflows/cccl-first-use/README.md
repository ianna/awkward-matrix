# cuda.compute first-use constructor reproducer

Standalone Python script; no Awkward/Coffea dependency or input data. Requires
CuPy and cuda.compute (tested API source: cccl-python 1.3.0). GPU execution verified on Della A100 in run 16475350: all cases passed NumPy
parity in unprofiled and separate profiled processes.

```bash
python repro.py --directory results --processes 3 --size 4096
# One algorithm only:
python repro.py --case stateful_transform --directory transform-only
# Separate profiling pass (perturbed timings):
python repro.py --directory profiles --processes 2 --profile
```

Every algorithm/process pair uses a fresh Python interpreter. Each case has a
new CuPy cache directory, reused by later processes of that case. Other compiler
caches are left untouched. No CCCL algorithm is called by the parent or by the
worker's generic CUDA/context warmup.

The four cases are stateless increment, increment capturing a one-element device
array, inclusive PLUS scan, and PLUS reduction. All use int64. Results are checked
against NumPy after execution. Allocations and validation stay outside build and
execution timing. Adapter construction, first factory build, cached factory call,
workspace query, first execution and five warmed executions are reported
separately. If build work is deferred until execution, first-execution timing
will expose it. cProfile covers only first factory construction, not imports,
adapter creation or execution. JSON includes package versions, device and runtime.
Do not interpret a cached same-process factory call as a disk-cache hit.

Results directories must be new; this prevents silently overwriting/reusing a
supposedly empty CuPy cache. Profile and unprofiled runs have distinct directories.
The small input isolates setup from kernel workload: this is a diagnostic,
not a representative throughput benchmark or an assertion of an upstream bug.

## Della

Unpack the folder inside your existing `/home/iosborne/Projects/ewkcoffea-della`:

```bash
cd /home/iosborne/Projects/ewkcoffea-della
unzip -o cccl-first-use.zip
sbatch --export=NONE cccl-first-use/run-della.sbatch
```

After completion:

```bash
cat results/JOBID/cccl-first-use/timings.txt
head -n 100 results/JOBID/cccl-first-use/profiles/stateful_transform-2-profile.txt
```

The batch uses the existing environment without changing packages or requiring
network access. Attach `repro.py`, the unprofiled JSONs and the second-process
profiles when sharing results. `MESSAGE.md` contains the workflow evidence; it
does not claim this standalone script has already reproduced those timings.
