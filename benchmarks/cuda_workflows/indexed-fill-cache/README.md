# IndexedArray_fill callback/cache-key isolation

Runs the actual installed `awkward_IndexedArray_fill` implementation, not a
reimplementation of its callback. No CMS input data needed. The script observes
the exact arguments to `cuda.compute.algorithms._transform.make_unary_transform`
and compares its real specialization keys by equality. Device/context/thread are
held fixed. It counts calls to CCCL's native build entry and stateful operator
compilation. Private hooks are diagnostic-only and restored afterward.

Cases: baseline first/repeat, identical inputs, different input pointer, different
output pointer, declared lengths 32/96 with original buffers, input view with same
pointer but different shape, output offset, input dtype, output dtype, captured
base values 11/12, and Python-int versus numpy-int64 base. Each variant is repeated
and followed by a return to baseline. Buffers remain alive throughout to prevent
allocator pointer recycling from hiding a pointer change. Source values are the
same for pointer-only cases. Dtype cases necessarily use new allocations; their
pointer-only controls are separate.

Callback observations record captured scalar values/types and array metadata,
source hash and package versions. Output checks include untouched prefix/tail.
Timing includes observation overhead and is diagnostic; build/compile counts and
key equality are the primary evidence. No caches are cleared during replay.
A new CuPy disk cache is used, while other compiler caches are untouched.

## Della

Unpack under the existing ewkcoffea-della directory:

```bash
sbatch --export=NONE indexed-fill-cache/run-della.sbatch
```

After completion:

```bash
cat results/JOBID/indexed-fill-cache/replay/comparison.md
cat results/JOBID/indexed-fill-cache/replay/environment.json
```

`steps.json` holds full key-group comparisons and captured closure metadata.
Group IDs are local to this process, not portable hash fingerprints. A fresh
callback object alone is not considered a changed key: actual equality decides.
If the observed factory is bypassed by a different installed API, the script
fails rather than claiming a hit. This does not change the installed environment.

## Already checked locally

`static_check.py` executes the real source definitions for CachableFunction and
its closure keying, plus the actual kernel callback body, with CPU array stand-ins.
Using cccl-python 1.3.0 source and Awkward 2.14.0 source, it finds unchanged callback
keys for identity/pointer/length/shape/offset cases; input/output dtype and base
value/type changes alter them. CPU callback outputs pass parity. This is evidence
about source-level keys only, not GPU build behavior. GPU replay remains pending.

The callback captures `base` as a scalar constant. Changes to per-chunk offsets
are therefore a specific candidate for repeated builds, but the replay and real
workflow argument history must confirm the cause before changing production code.
