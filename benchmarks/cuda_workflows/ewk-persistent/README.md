# Several full-analysis chunks in one persistent worker

This uses the pinned CMS analysis, not the synthetic triplet probe. Requires a
local CMS ROOT file and its sample JSON, plus the existing
`ewkcoffea_della_repro` environment. No downloads or installs run on compute nodes.

Unpack in `/home/iosborne/Projects/ewkcoffea-della`, then submit:

```bash
sbatch --export=NONE ewk-persistent/run-della-persistent.sbatch \
  /absolute/path/to/local-sample.json 20000 5
```

This requests up to five chunks of about 20k events each. Start with the 100k
input; actual Coffea entry ranges are recorded and checked. Input must provide
at least three chunks, otherwise the job fails rather than reporting a warmed
comparison from too little data. The JSON schema is upstream's single-sample
schema with `files` and optional `path`. Remote URLs are rejected deliberately:
compute-node network access/credentials are not configured by this job.

To use the already-tested metadata overlay, pass its absolute directory as a
fourth argument (no packages are modified):

```bash
sbatch --export=NONE ewk-persistent/run-della-persistent.sbatch \
  /absolute/path/to/local-sample.json 20000 5 \
  /home/iosborne/Projects/ewkcoffea-della/results/16474896/context-comparison/metadata
```

## Execution and measurements

- Runs CPU and CUDA over the same input/ranges, in that order, in separate
  processes. This is an exploratory run; CPU/GPU order and host file caches
  aren't balanced as in a repeated production comparison.
- Uses Coffea's FuturesExecutor public `pool` argument with a single spawned
  ProcessPoolExecutor worker. Standard executor has no automatic worker recycle
  policy configured here. This does not alter Loky's memory-leak threshold or
  enable the upstream `sitecustomize` workaround. The original merge policy is
  retained; no iterative-executor substitution.
- Verifies at least three completed chunks under exactly one PID per backend,
  no duplicate input ranges, and identical CPU/GPU entry ranges.
- Records completed worker task time, including I/O/materialization, analysis,
  histogram filling and task compression. Generic CUDA import/context setup is
  outside the first task timer but inside the whole-run wall time. First-use
  analysis JIT remains in the first chunk; later chunks are summarized separately.
- Captures upstream phase prints per entry range. Those upstream timers aren't
  all independently fenced: use them as diagnostic attribution, not pure device
  time or a sum of independently measured stages.
- Records per-chunk RSS, cumulative peak RSS, CuPy pool usage/reservation and
  a one-second device-memory trace. These are memory observations, not a guarantee
  of absence of a leak. A worker failure aborts the run; no automatic restart.
- Aligns categorical bins by label (rejecting missing labels and numeric bin changes), then compares merged CPU/CUDA histogram axes, values, weighted variances and flow
  bins (rtol=1e-6, atol=1e-8). Mismatches abort; do not label throughput as parity
  validated until comparison passes. This is parity for `njets` and `njets_counts`
  requested by the original reproducer, not every possible analysis histogram.

Inspect:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
cat results/JOBID/persistent/comparison.md
cat results/JOBID/persistent/cuda/summary.json
```

Detailed logs and merged histograms live under the CPU/CUDA directories. The
whole-run wall time includes runner setup, metadata preparation, merging and
output serialization; it excludes environment activation and imports preceding
runner execution. Code does not edit any upstream source files. It records local
source diffs and SHAs for provenance.

Local validation: real spawned-process/cloudpickle persistence test, histogram
value/variance checks, shell syntax and Python lint pass. Full Coffea/CUDA
execution still needs Della and the input data.
