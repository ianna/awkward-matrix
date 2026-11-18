# Standalone CUDA workflow benchmarks

These are the Della harnesses used alongside the kernel migration benchmarks.
The sources were recovered from the saved commands that created and updated the
original bundles. Formatting and packaging were checked locally. GPU measurements
have not been rerun as part of this publication.

The directories retain their bundle names because the batch scripts refer to
one another through the existing Della workspace layout.

| Directory or job | Purpose |
| --- | --- |
| `awkward-cccl-studies` | Earlier kernel, physics-query and fusion studies from the Awkward checkout |
| `ewkcoffea-della` | Pinned triplet probe, startup profiling and released/metadata context comparison |
| `cccl-first-use` | Transform, stateful transform, scan and reduction preparation across fresh processes |
| `ewk-persistent` | Full analysis in a persistent worker, histogram parity, phase reports, chunk sweep and scalar/basic-selection diagnostics |
| `indexed-fill-cache` | Callback cache-key replay varying pointers, lengths, dtypes and captured base values |
| `indexed-fill-runtime-validation` | Prepared runtime-base fill validation and paired full-analysis comparison |
| `repeat-indexed-fill.sbatch` | Three unprofiled baseline/candidate pairs in alternating order |
| `ewk-context-bounded` | Bounded error-context summaries with full-analysis and diagnostic comparisons |
| `context-bounded-repeat.sbatch` | Alternating unprofiled bounded-context comparisons |
| `ewk-fill-serialization` | Fresh-process prepared-fill build/serialize/deserialize replay |
| `ewk-fill-cache-integration` | Historical opt-in production artifact-cache replay and full-analysis comparison |

The existing [kernel migration suite](../cuda_migrations/README.md) covers
ListArray, IndexedArray, RegularArray, ListOffsetArray and UnionArray work.
The [CUDA matrix](../cuda_matrix/README.md) provides selectable CPU, RawKernel
and compute parity cases. The standalone
[dispatch reproducer](../../reproducers/README.md) is also published separately.

## Install the workflow scripts in an existing Della workspace

From the awkward-matrix checkout on Della:

```bash
WORKFLOW_ROOT=/home/iosborne/Projects/ewkcoffea-della
mkdir -p "$WORKFLOW_ROOT"
cp -R benchmarks/cuda_workflows/ewkcoffea-della/. "$WORKFLOW_ROOT/"
for suite in cccl-first-use ewk-persistent indexed-fill-cache \
    indexed-fill-runtime-validation ewk-context-bounded \
    ewk-fill-serialization ewk-fill-cache-integration; do
    cp -R "benchmarks/cuda_workflows/$suite" "$WORKFLOW_ROOT/"
done
cp benchmarks/cuda_workflows/*repeat*.sbatch "$WORKFLOW_ROOT/"
```

Read each suite's README before submitting its jobs. The full analysis needs a
local ROOT file, a local sample JSON and the pinned upstream analysis/environment.
The wrappers record dependency versions and source revisions in their results.

Several comparison wrappers default to overlays or phase reports from specific
historical Slurm jobs. Those job directories must exist, or their configured
paths must be updated to equivalent overlays before running. They are recorded
experiment dependencies, not files supplied by this repository. Scalar/basic
diagnostics target the pinned processor's statement blocks and fail when the
expected source structure changes.

Setup requires network access. Della compute nodes previously failed to resolve
GitHub. Run setup on the network-enabled login host with the workspace variables:

```bash
cd "$WORKFLOW_ROOT"
SLURM_SUBMIT_DIR="$PWD" \
SLURM_JOB_ID="setup-login-$(date +%Y%m%d-%H%M%S)" \
bash run-della-ewk.sbatch setup
```

The setup wrapper selects the site's Anaconda and CUDA modules and performs a
CUDA smoke test. Their availability must match the host. Subsequent GPU jobs use
`sbatch --export=NONE`, as shown in the suite READMEs. Use a local sample JSON
instead of the illustrative placeholder in the persistent-worker instructions.

## Source snapshots

`source-snapshots.json` records the original Awkward commits, source paths and
SHA-256 hashes of the copied helpers. These files preserve experimental
implementations used by the harnesses. The cache integration snapshot predates
the split of the disk cache from the focused runtime-base PR.

The copied Awkward sources retain their BSD license headers, and the upstream
license is included in `LICENSE.awkward`.

## Local validation

```bash
python -m pip install pytest numpy cloudpickle hist awkward
python -m pytest benchmarks/cuda_workflows
python -m compileall -q benchmarks/cuda_workflows
find benchmarks/cuda_workflows -name '*.sbatch' -exec bash -n {} \;
```

The CPU tests check worker persistence, instrumentation boundaries, histogram
comparison, serialization metadata, bounded summaries and harness control flow.
They use substitutes for GPU operations where applicable. CUDA compilation,
device execution and timing comparisons require a GPU allocation.

Generated ROOT files, environments, compiled caches, Nsight traces and large
result directories remain external inputs/outputs.

Codex assisted with recovery, packaging, documentation and local validation.
