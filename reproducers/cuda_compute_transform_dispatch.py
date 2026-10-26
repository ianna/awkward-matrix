# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Warmed stateful unary-transform dispatch overhead; no Awkward dependency.

python cuda_compute_transform_dispatch.py --rows 32 --output transform-32.json
python cuda_compute_transform_dispatch.py --rows 4096 --output transform-4096.json

All modes use identical live buffers, callable, iterator and nondefault stream.
JIT/preparation and warmup are excluded. Completion fences are outside host
latency but inside completed latency. CUDA-event time includes submission gaps.
"""

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path


def operations(compute, make_op_adapter, outer, inner, output, stream, np):
    iterator = compute.CountingIterator(np.int64(0))

    def gather(i):
        return inner[outer[i]]

    adapter = make_op_adapter(gather)
    # Factories may return the same cached object; calls are strictly sequential.
    prepared = compute.make_unary_transform(d_in=iterator, d_out=output, op=adapter)
    arguments = dict(d_in=iterator, d_out=output, num_items=len(outer), stream=stream)
    return {
        "convenience_callable": partial(
            compute.unary_transform, op=gather, **arguments
        ),
        "prepared_callable": partial(prepared, op=gather, **arguments),
        "convenience_adapter": partial(
            compute.unary_transform, op=adapter, **arguments
        ),
        "prepared_adapter": partial(prepared, op=adapter, **arguments),
    }


def measure(cp, stream, call, iterations):
    start, end = cp.cuda.Event(), cp.cuda.Event()
    samples = []
    for _ in range(iterations):
        stream.synchronize()
        start.record(stream)
        before = time.perf_counter_ns()
        call()
        submitted = time.perf_counter_ns()
        end.record(stream)
        end.synchronize()
        completed = time.perf_counter_ns()
        samples.append(
            {
                "host_us": (submitted - before) / 1000,
                "completion_us": (completed - before) / 1000,
                "stream_us": cp.cuda.get_elapsed_time(start, end) * 1000,
            }
        )
    return {key: statistics.median(x[key] for x in samples) for key in samples[0]}


def main():
    import cupy as cp
    import numpy as np
    from cuda import compute
    from cuda.compute.op import make_op_adapter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.warmup, args.iterations, args.repeats) < 1:
        parser.error("All counts must be positive")
    stream = cp.cuda.Stream(non_blocking=True)

    class StreamAdapter:
        def __cuda_stream__(self):
            return (0, stream.ptr)

    with stream:
        host_outer = np.arange(args.rows + 1, dtype=np.int64)[::-1].copy() + 1
        host_inner = np.arange(args.rows + 3, dtype=np.int64) * 8
        outer, inner = cp.asarray(host_outer), cp.asarray(host_inner)
        output = cp.empty(args.rows + 1, dtype=cp.int64)
        calls = operations(
            compute, make_op_adapter, outer, inner, output, StreamAdapter(), np
        )
        for call in calls.values():
            for _ in range(args.warmup):
                call()
                stream.synchronize()
            np.testing.assert_array_equal(
                output.get(stream=stream), host_inner[host_outer]
            )
        samples = {name: [] for name in calls}
        names = list(calls)
        for repeat in range(args.repeats):
            shift = repeat % len(names)
            for name in names[shift:] + names[:shift]:
                samples[name].append(measure(cp, stream, calls[name], args.iterations))
                np.testing.assert_array_equal(
                    output.get(stream=stream), host_inner[host_outer]
                )
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if any(
            word in d.metadata["Name"].lower()
            for word in ("cccl", "cupy", "numba", "cuda-bindings")
        )
    }
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "packages": packages,
        "device": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)[
            "name"
        ].decode(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "settings": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "results": {
            name: {
                "samples": values,
                "median": {
                    key: statistics.median(x[key] for x in values) for key in values[0]
                },
            }
            for name, values in samples.items()
        },
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
