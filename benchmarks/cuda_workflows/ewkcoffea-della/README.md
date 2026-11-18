# Run the ewkcoffea reproducer on Della

Reviewed source revision: `kmohrman/columnar_gpu@7737ea0ba73c09c3345e43f65a17a3bc43d439a6`.

This bundle contains new Della wrappers, not copies of the analysis. Setup clones
the reviewed repository and runs its setup script. The authors pin coffea to
`b910c2fe`, topcoffea to `285a46f`, and ewkcoffea to `f3fc446`.
The wrapper pins the released Awkward control to **2.14.0**, records resolved
dependency versions, and uses a dedicated environment, `ewkcoffea_della_repro`.
It does not install into `awkward-cuda-bench`.

These are the historical wrappers used for the Della investigation. See the
[parent instructions](../README.md) for installation from awkward-matrix and
the setup procedure for a network-enabled login host. The validation described
below reflects the original preparation of the bundle.

## Review findings

1. **Good workload for a real-operation check.** The standalone probe exercises
   padded ragged jets, three-way combinations, Lorentz-vector arithmetic,
   `argmin`, jagged selection, flattening and missing values. Reversing collection
   order and repeating the first collection helps separate first-use costs from
   work proportional to input size.
2. **The probe is not the entire analysis.** It uses synthetic jets and does not
   execute ROOT input, selections, DNN inference, weights or histogram filling.
   The full study uses real CMS data and modified coffea/analysis forks.
3. **The unaccounted startup cost may still contain compilation.** Warming
   `CUPY_CACHE_DIR` controls CuPy's disk cache; it does not establish that CCCL,
   MLIR, Python initialization and other per-process work are warm. The source
   does not yet isolate these costs. The published 13x result is a particular
   warmed block against one CPU core, not whole-workflow throughput.
4. **Parity needs an additional check.** The upstream probe reports timings but
   does not compare CPU/GPU outputs. `probe_della.py` saves the final repeated
   collection outside timing. Both execution orders cover both collections, and
   the batch script checks all three GPU repeats against CPU. This checks the
   final mass output, not every intermediate or the full analysis histograms.
5. **Timing boundaries differ.** The probe fences the default stream before and
   after each timed block. Deferred-error draining is added outside those timed
   blocks by this wrapper. Full-analysis subphases fence at their ends, but the
   first subphase lacks an explicit starting fence and can inherit previous
   queued work. Top-level phase timers are not consistently fenced. Nsight or
   cProfile passes must remain separate from latency measurements.
6. **Single GPU / single worker first.** The supplied analysis runs
   `FuturesExecutor` with one worker. It does not distribute one workload across
   multiple GPUs. Its one-chunk runs do not test worker recycling. For a separate
   multi-chunk investigation, verify worker PIDs and loky's threshold explicitly;
   the supplied `sitecustomize.py` suppresses import errors and changes a private
   global threshold. Do not enable it silently for baseline runs.
7. **One analysis detail needs checking before treating it as a physics oracle.**
   In the pinned processor, selection `2lOSSF_1fjx_ejj3_fj0noMatch` is not gated
   by `isSig_mask`, unlike neighboring truth-matching selections. Its category
   entry is commented out in the pinned source, so this alone does not establish
   an effect on the shipped histograms. Check it before enabling that category.
8. **Environment is not fully locked upstream.** The source repositories are
   pinned, but several Conda/pip requirements are broad. Keep the exported
   manifests and compare configurations with the same dependency stack.

Sources:
- [Workflow README](https://github.com/kmohrman/columnar_gpu/blob/7737ea0ba73c09c3345e43f65a17a3bc43d439a6/ewkcoffea_on_gpu/README.md)
- [Probe](https://github.com/kmohrman/columnar_gpu/blob/7737ea0ba73c09c3345e43f65a17a3bc43d439a6/ewkcoffea_on_gpu/reproduce/probe/triplets_probe.py)
- [Reproduction instructions](https://github.com/kmohrman/columnar_gpu/blob/7737ea0ba73c09c3345e43f65a17a3bc43d439a6/ewkcoffea_on_gpu/reproduce/README.md)
- [Timing patch](https://github.com/kmohrman/columnar_gpu/blob/7737ea0ba73c09c3345e43f65a17a3bc43d439a6/ewkcoffea_on_gpu/reproduce/patches/0003-Add-per-call-sub-phase-timing-inside-obj_combinatori.patch)
- [Pinned processor](https://github.com/cmstas/ewkcoffea/blob/f3fc44625f5420813bed8bb4ca5c0a1be50a4403/analysis/vbs_vvh/analysis_processor_semilep.py#L1247)

## 1. Copy the scripts from awkward-matrix

Follow the [parent installation instructions](../README.md). They install the
probe and batch wrappers in the original workspace layout.

## 2. Unpack on Della

```bash
cd /home/iosborne/Projects
unzip ewkcoffea-della.zip
cd ewkcoffea-della
```

All following `sbatch` commands must be submitted from this directory. The
wrapper uses `SLURM_SUBMIT_DIR` to locate its scripts and save results.
It intentionally omits `--partition=gpu` and uses the working Della allocation
pattern: one GPU, `nomig&gpu40`, and `--export=NONE`.

## 3. Queue setup and the standalone probe

Setup creates a separate environment and downloads the analysis dependencies
(allow roughly 15 GB plus checkout/results space). It runs on a compute node.
The probe job waits for setup to succeed:

```bash
SETUP_JOB=$(sbatch --parsable --export=NONE run-della-ewk.sbatch setup)
echo "Setup job: $SETUP_JOB"

sbatch --export=NONE --dependency=afterok:${SETUP_JOB%%;*} \
  run-della-ewk.sbatch probe 1000000
```

For a smaller smoke test use `probe 10000`. The default one-million-event probe
is the workload used in the upstream startup discussion. No CMS files or grid
credentials are required for this mode.

The job runs CPU controls and both GPU execution orders. Each GPU order uses a
separate initially empty CuPy cache directory. Repeats 2 and 3 start fresh
processes but reuse that order's disk cache. Within each process the upstream
probe also repeats the first collection. These are distinct kinds of warmup;
do not average them together.

Check status, substituting the actual probe job ID:

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
tail -F ewkcoffea-JOBID.log
```

Results:

```bash
cat results/JOBID/parity.txt
grep '^RESULT' results/JOBID/cpu-*.log results/JOBID/cuda-*.log
cat results/JOBID/runtime.txt
```

Expect six `PARITY PASSED` lines. Version manifests, repository SHAs, hardware
details, GPU memory samples, raw logs and NumPy result arrays are retained in
the job directory. These wrappers do not apply your Awkward PR automatically.
Parity requires matching shapes and dtypes, with floating values compared using
`rtol=1e-9`, `atol=1e-6` and matching NaNs.

## 4. Prepare local input for the full workflow

Start with the **1k** file (4.3 MB). Obtain `mega_rep_1k.root` from the author or
copy it from CMS storage using your existing authorized credentials. Put it in
`/home/iosborne/Projects/ewkcoffea-della/data/mega_rep_1k.root`.
Remote xrootd URLs require working CMS credentials on the compute node; those
credentials are not set up or forwarded by this wrapper.

On Della:

```bash
cd /home/iosborne/Projects/ewkcoffea-della
mkdir -p data

module load anaconda3/2026.7
conda activate ewkcoffea_della_repro
python - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
template = root / 'columnar_gpu/ewkcoffea_on_gpu/reproduce/input_jsons/rep_1k.json'
sample = root / 'data/mega_rep_1k.root'
assert sample.is_file(), f'Place the input here first: {sample}'
data = json.loads(template.read_text())
data['path'] = ''
data['files'] = [str(sample.resolve())]
(root / 'data/rep_1k-local.json').write_text(json.dumps(data, indent=2) + '\n')
PY

sbatch --export=NONE run-della-ewk.sbatch \
  full 1k ewkcoffea_della_repro data/rep_1k-local.json
```

The job runs three CPU/GPU pairs on the same host, one worker each, with CPU/GPU
order alternated. Every GPU run uses a new CuPy cache directory, so the full
workflow's cache policy is consistently cold. These are cold-start analysis
measurements, not warmed service throughput. The synthetic probe above is the
separate warmup study.

Results:

```bash
cat results/JOBID/full/summary.md
ls results/JOBID/full/logs/
ls results/JOBID/full/outputs/
```

`summary.md` and plots describe timings. Inspect output histograms, categories,
counts and weighted sums against CPU before treating a speedup as validated.
The wrapper does not assert full-analysis histogram parity.

Move to `100k`, then `1m`, only after the 1k run succeeds. Obtain each corresponding
ROOT file and make its matching local JSON. The size argument selects chunking;
it must agree with the actual input file. Do not assume `5m` fits the 40 GB A100;
the upstream large-card result is not a memory guarantee for Della.

## 5. Comparing your migration branch later

First keep this released control intact. Use a second environment for a candidate,
with the same dependency versions and pinned analysis checkouts. Install the
candidate's matching `awkward-cpp` build and editable Python package there. The
wrapper accepts the environment name as argument three:

```bash
sbatch --export=NONE run-della-ewk.sbatch \
  probe 1000000 ewkcoffea_della_candidate
```

The runtime record includes `awkward.__file__`, since both a release and your
branch can report version `2.14.0`. Record the candidate Git SHA as well.
Compare CPU/GPU results and the same timing/cache boundaries. Do not mix a
dependency upgrade with an Awkward branch comparison.

Della's newer NVIDIA driver can run the older CUDA runtime selected by the
authors' environment; no driver downgrade is indicated. See NVIDIA's
[backward compatibility explanation](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html).

No files or comments have been posted to the author's repository.
