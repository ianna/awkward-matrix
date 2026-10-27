# Standalone cuda.compute transform dispatch reproducer

`cuda_compute_transform_dispatch.py` needs only a working CuPy, NumPy,
`cuda.compute` and numba-cuda environment. It does not import Awkward or the
benchmark harness. The prepared-transform and adapter APIs match CCCL Python
1.1.1. Run inside an allocated GPU job:

```bash
python reproducers/cuda_compute_transform_dispatch.py \
  --rows 32 --output transform-32.json
python reproducers/cuda_compute_transform_dispatch.py \
  --rows 4096 --output transform-4096.json
```

The four paths perform the same stateful gather using fixed buffers, a stable
callable, one counting iterator and one nondefault stream:

- `convenience_callable`: `unary_transform(..., op=gather)`.
- `prepared_callable`: retained `make_unary_transform` result, `op=gather`.
- `convenience_adapter`: `unary_transform` with retained `make_op_adapter(gather)`.
- `prepared_adapter`: both prepared transform and adapter retained.

Preparation and 20 warmup calls per path precede timing. Five repeats rotate
path order; each repeat reports the median of 100 calls. Completion fences are
outside the host interval. CUDA-event times include submission gaps and must
not be interpreted as pure device execution. Correctness checks are outside
timing. JSON includes all repeat samples and GPU/runtime/package information.
The script intentionally retains all captured arrays through the entire run;
it does not demonstrate safe rebinding or concurrent use of prepared objects.

For the cuda.compute team, the question is whether repeated calls with the same
callable could reuse its adapter/state description more cheaply, and whether
retaining an adapter alongside a prepared transform is the recommended public
usage. Compare all four paths before attributing the cost to factory cache-key
construction versus callable adaptation. A timing difference alone does not
show repeated JIT compilation.

This reproducer is ready to attach to an upstream issue, but has not itself been
GPU-validated here. Attach the generated JSON from both sizes when reporting it.
