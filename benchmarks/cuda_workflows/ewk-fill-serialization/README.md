# Fresh-process serialization of Awkward's prepared IndexedArray_fill

Adapts Ashwin Srinath's stateful cuda.compute serialization example to the actual
#4503 `_PreparedFill` operator and its invocation-local parameter binding. This
is a standalone validation harness, not a production disk-cache integration.
No environment or checkout is modified, and no upgrade is performed.

## Della

From /home/iosborne/Projects/ewkcoffea-della:

    sbatch --export=NONE ewk-fill-serialization/run-della.sbatch

Uses ewkcoffea_della_repro and the overlay from run16639248, which already includes
#4503. The batch runs three new processes per canonical input dtype (int32,
int64, uint32): one build/serialize worker, then two deserialize workers. Each
retains its own CUDA arrays and algorithm. Blob and CuPy caches live within the
new job output directory. Preparation and the first completed invocation are
measured separately; all subsequent stages are saved in per-worker result.json.
Context initialization is excluded; first native loading during invocation is
included. Timers have build-counter instrumentation and synchronization overhead;
they are not a full-analysis performance benchmark.

Every call passes the actual helper's op adapter to bind fresh device state.
Checks cover changing Python/NumPy scalar bases, lengths and pointers, negative
indices, untouched output prefix/tail, and interleaved nondefault streams.
Submission to the shared prepared object is sequential; this does not claim
thread-safe concurrent mutation. The zero-length case follows production's
no-launch behavior. Output buffers remain alive throughout pointer-reuse checks.

Cold workers must observe native builds and operator compilation, establishing
that the counters work. Deserialized workers must observe zero native builds AND
zero operator compiles, including adapter creation and their first launch. A
failure saves completed per-worker evidence and aborts rather than printing a
successful summary. If serialization or instrumentation APIs are absent/changed,
the harness fails explicitly; inspect replay.log and pip-freeze.txt. We have not
yet established availability in Della's installed cccl-python version.

Cache manifests include the helper source hash, algorithm/array/state descriptors,
GPU major/minor capability, CUDA runtime, Python version and CCCL/MLIR/CuPy package
versions. They exclude runtime values, shapes and pointers. Blobs are local test
artifacts, not a portable long-term storage format. Each worker uses a single
thread; deserialized objects must not be shared concurrently between threads.

This tests GPU-built persisted artifacts. It does not yet test GPU-free AOT builds
with ProxyArray/compute_capability, or a full-analysis cache integration.

## Results

    cat results/JOBID/fill-serialization/replay/comparison.md
    cat results/JOBID/fill-serialization/replay/int64-1-load/result.json

If the loaded workers compile, report the stage counts rather than assuming the
API meets the requirement. Only after this replay passes should a production
cache and an uninstrumented fresh-worker/full-analysis comparison be attempted.

Local tests exercise cache invalidation and build-counter/timing attribution,
including cleanup on exceptions. GPU correctness/serialization awaits Della.
