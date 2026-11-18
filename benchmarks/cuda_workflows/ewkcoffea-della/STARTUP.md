# Residual startup diagnostics on Della

This adds profiling to the existing ewkcoffea-della installation. It does not
install or update packages and does not edit the upstream checkout.

From the Della `ewkcoffea-della` directory:

```bash
sbatch --export=NONE run-della-startup.sbatch 1000000 big_first
```

The environment is `ewkcoffea_della_repro` by default. An optional third argument
selects another existing environment. Do not run installers against that
environment while this job runs.

Four fresh GPU processes run sequentially:

1. **empty-cache**: unprofiled, initially empty job-specific CuPy disk cache.
2. **reused-cache**: unprofiled, reuses that CuPy cache.
3. **host-profile**: reuses it; cProfile separately captures first, second and
   repeated-first calls, with stream completion and deferred-error draining.
4. **cuda-trace**: reuses it; Nsight captures those three calls with NVTX labels,
   CUDA API/kernel activity and OS runtime waits. Input preparation and generic
   warmup are outside the CUDA profiler capture range.

Only CuPy's cache is controlled. Other compiler caches are untouched and may be
populated by earlier processes. A reused CuPy cache does not establish that
CCCL/MLIR or other JITs are warm. Profiles are separate diagnostic passes, not
latency benchmarks. Unprofiled timings also include deferred-error draining,
slightly extending the original upstream probe's boundary.

The source functions are loaded from the pinned upstream probe without editing
their bodies; its input generation, CPU combination counts and generic warmup
are reproduced. CPU/GPU final-output parity is checked after each GPU pass.
This is final-output parity only, not intermediate-result or full-analysis parity.

## Inspect the result

Replace JOBID with the submitted job number:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
cat results/JOBID/startup/parity.txt
cat results/JOBID/startup/reused-cache/timings.json
head -n 65 results/JOBID/startup/host-profile/first-cumulative.txt
head -n 65 results/JOBID/startup/host-profile/first-tottime.txt
head -n 65 results/JOBID/startup/host-profile/repeat_first-cumulative.txt
cat results/JOBID/startup/host-profile/subprocesses.jsonl
cat results/JOBID/startup/nsys-summary.txt
```

Compare `first` against `repeat_first`, and inspect compiler/import/cache/build
frames. Cumulative profile entries overlap: do not add their times together.
`self_s` in the full `*-functions.json` tables is exclusive host time. Long
native calls include time inside their libraries; this profile does not resolve
native compiler stacks. Subprocess audit records include executable and launch
time only; many compilers run in-process, so an empty file does not rule out JIT.
Nsight API duration can include waits and is not automatically kernel time.

Download the reports for further inspection:

```bash
# Run on your Mac, replacing JOBID:
scp -r iosborne@della-gpu.princeton.edu:/home/iosborne/Projects/ewkcoffea-della/results/JOBID/startup ~/Downloads/ewk-startup-JOBID
```

The `.nsys-rep` can be opened in Nsight Systems. A second job with `small_first`
checks whether the same expensive stack follows first use rather than input size:

```bash
sbatch --export=NONE run-della-startup.sbatch 1000000 small_first
```

If Nsight fails, the earlier unprofiled and host-profile reports are retained.
Use `ewk-startup-JOBID.log` to diagnose the trace failure.
