"""Run the pinned upstream probe unchanged, then save its final output for parity."""

import argparse
import os
import runpy
import sys
from pathlib import Path

import awkward as ak
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("cpu", "cuda"))
    parser.add_argument("order", choices=("big_first", "small_first"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--events", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.events <= 0:
        parser.error("--events must be positive")
    os.environ["PROBE_NEVT"] = str(args.events)
    sys.argv = [str(args.probe), args.backend, args.order]
    namespace = runpy.run_path(str(args.probe), run_name="__main__")
    # At the pinned revision, '_' holds triplets #3's output, i.e. the first
    # collection repeated. Running both orders saves both collections.
    result = namespace["_"]
    if not isinstance(result, ak.Array):
        raise TypeError("Upstream probe changed: final output is no longer an Array")
    if args.backend == "cuda":
        import cupy as cp
        from awkward._connect import cuda

        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        if stream.ptr in cuda.cuda_streamptr_to_contexts:
            cuda.synchronize_cuda(stream)
    # Transfers, error draining, saving and CPU/GPU comparisons stay outside
    # upstream's timing regions. This is not a completed-public-call benchmark.
    values = ak.to_numpy(ak.to_backend(result, "cpu"))
    if np.ma.isMaskedArray(values) and np.any(values.mask):
        raise AssertionError("Unexpected missing values after fill_none")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.asarray(values), allow_pickle=False)
    print(f"SAVED {args.output} shape={values.shape} dtype={values.dtype}", flush=True)


if __name__ == "__main__":
    main()
