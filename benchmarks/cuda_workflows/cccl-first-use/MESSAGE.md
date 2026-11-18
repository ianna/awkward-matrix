Hi Ashwin and team — the standalone reproducer now confirms the fresh-process constructor cost without Awkward or Coffea (Della A100 40 GB, cccl-python 1.3.0, numba-cuda-mlir 0.5.4, CuPy 14.2.0, CUDA runtime 12.9).

Across three fresh processes per case, median first factory construction takes:
- stateless transform: 1.736 s
- stateful transform (captures one device array): 1.786 s
- inclusive PLUS scan: 3.770 s
- PLUS reduction: 2.421 s

The same factory call in the same process takes only 65–143 microseconds, with warmed completed execution around 15–27 microseconds. All cases pass NumPy output parity. Each case shares its CuPy disk cache across processes; other compiler caches are untouched. One stateful-transform process took 3.81 s, so there is some build-time variability.

In a separate second-process stateful-transform profile, 1.634 s of a 1.740 s build is exclusive time in DeviceUnaryTransform’s native binding constructor; stateful operator compilation is about 0.105 s cumulatively. We have not separated compilation/linking inside the native constructor.

Attached repro.py separates adapter creation, first build, same-process cached factory lookup, workspace sizing and first/warmed execution. Its --profile option runs separate diagnostics. Run: python repro.py --directory results --processes 3

This matches the real triplet workflow’s startup pattern: 29.4 s first use versus 0.124 s warmed, despite a reused CuPy cache. Which constructor artifacts can be cached across processes, and is there a supported persistent-cache or ahead-of-time route we should use?
