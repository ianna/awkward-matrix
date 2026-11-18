# Production opt-in IndexedArray_fill artifact-cache validation

Implementation: #4503 commit6c88db479. This bundle includes its exact
_indexed_fill.py and _compute_cache.py, plus a production-path replay and Della
batch wrapper. No installed environment or existing overlay is changed.

Submit from /home/iosborne/Projects/ewkcoffea-della:

    sbatch --export=NONE ewk-fill-cache-integration/run-della.sbatch

The wrapper copies the bounded-context overlay from run16639248 into the job
results and replaces only these two helpers. It creates a fresh artifact cache
under results/JOBID/fill-cache/artifacts. Context/bounded behavior is held fixed.

Production flag:

    export AWKWARD_CUDA_CACHE_DIR=/absolute/path/to/a/cache-you-control

The replay uses the actual helper.fill production entry point; it does not
manually assign deserialized objects. For each canonical dtype it runs fresh
processes with caching disabled, first enabled build, then two enabled loads.
All calls are on nondefault streams, with different lengths, pointers and bases;
output/prefix/tail parity is required. Loaded workers must have zero native
builds and operator compiles, including the first invocation. No zero-build
claim is made until the replay passes. Instrumented completed-call timings
exclude initial CUDA context setup; per-stage count data remain in result.json.
The cache is not a replacement for runtime state management or in-process reuse.

After replay, the batch runs separate, unprofiled 5x20k CPU/CUDA full analyses
with cache disabled then enabled (prepopulated from replay). Both must pass
histogram parity. Other kernels still compile normally. This is one sequential
pair; repeat with alternating order before making a stable performance claim.
The existing ewk-persistent harness already on Della is required.

Read:

    cat results/JOBID/fill-cache/replay/comparison.md
    cat results/JOBID/fill-cache/full-comparison.md

Artifacts include source/version/dtype/architecture compatibility keys, are
written atomically, and can be deleted. Missing serialization support, rejected
blobs and cache filesystem failures fall back to normal compilation in production.
The strict replay intentionally fails if that fallback occurs in a load worker,
so an unavailable feature cannot masquerade as a successful cache hit.

Scope: IndexedArray_fill only. This is GPU-built code persisted across processes,
not GPU-free ProxyArray AOT and not an all-kernel disk cache. Global shared prepared
objects remain protected by the existing host lock; parameter storage remains
invocation-local. GPU validation of the production integration is pending.
