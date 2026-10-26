# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Locate warmed flatten_offsets host overhead; profile in a separate pass."""

import argparse
import cProfile
import json
import pstats
import statistics
import time
from pathlib import Path

import numpy as np
from measure import environment

NAME = "awkward_ListOffsetArray_flatten_offsets"


class StreamAdapter:
    def __init__(self, stream):
        self.stream = stream

    def __cuda_stream__(self):
        return (0, self.stream.ptr)


def levels(backend, wrapper, implementation, compute, stream, output, outer, inner):
    """Keep every buffer alive, with one stable op/iterator for direct calls."""
    key = (NAME, np.int64, np.int64, np.int64)
    args = (output, outer, len(outer), inner)
    kernel = backend[key]
    iterator = compute.CountingIterator(np.int64(0))
    adapter = StreamAdapter(stream)
    selected = outer[: len(outer)]

    def gather(i):
        return inner[selected[i]]

    before = time.perf_counter_ns()
    prepared = compute.make_unary_transform(d_in=iterator, d_out=output, op=gather)
    factory_ns = time.perf_counter_ns() - before
    operations = {
        "backend_lookup": lambda: backend[key](*args),
        "cached_backend_kernel": lambda: kernel(*args),
        "compute_wrapper": lambda: wrapper(*args),
        "compute_implementation": lambda: implementation(*args),
        "direct_unary_transform": lambda: compute.unary_transform(
            d_in=iterator, d_out=output, op=gather, num_items=len(outer), stream=adapter
        ),
        "prepared_transform": lambda: prepared(
            d_in=iterator, d_out=output, op=gather, num_items=len(outer), stream=adapter
        ),
    }
    return operations, factory_ns


def profile_calls(profiler, call, finish, count):
    for _ in range(count):
        profiler.enable()
        try:
            call()
        finally:
            profiler.disable()
        # Error-context draining and completion fences must not dominate the
        # profile or change when the backend's invocation-list bound is hit.
        finish()


def measure_call(cp, stream, call, finish):
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    before = time.perf_counter_ns()
    call()
    submitted = time.perf_counter_ns()
    end.record(stream)
    end.synchronize()
    finish()
    completed = time.perf_counter_ns()
    return {
        "host_us": (submitted - before) / 1000,
        "completion_us": (completed - before) / 1000,
        "stream_us": cp.cuda.get_elapsed_time(start, end) * 1000,
    }


def run(args):
    import awkward as ak
    import cupy as cp
    from awkward._backends.cupy import CupyBackend
    from awkward._connect import cuda
    from awkward._connect.cuda import _compute, _listoffsetarray
    from awkward._kernels import CudaComputeKernel
    from cuda import compute

    checkout = Path(args.checkout).resolve()
    if checkout not in Path(ak.__file__).resolve().parents:
        raise ValueError("Install the candidate checkout before running diagnostics")
    backend = CupyBackend.instance()
    if not isinstance(backend[(NAME, np.int64, np.int64, np.int64)], CudaComputeKernel):
        raise ValueError(
            "This diagnostic requires the cuda.compute candidate, not the raw baseline"
        )
    if not hasattr(compute, "make_unary_transform"):
        raise ValueError(
            "Installed cuda.compute lacks make_unary_transform; no prepared-call comparison is possible"
        )
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stream = cp.cuda.Stream(non_blocking=True)
    report = {
        "environment": environment(cp, checkout),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "profile_iterations": args.profile_iterations,
        "cases": [],
    }
    with stream:
        for rows in args.rows:
            # Nonzero origin and reverse gather prevent an identity-copy test.
            host_outer = np.arange(rows + 1, dtype=np.int64)[::-1].copy() + 1
            host_inner = np.arange(rows + 3, dtype=np.int64) * 8
            expected = np.full(rows + 5, -999, dtype=np.int64)
            expected[: rows + 1] = host_inner[host_outer]
            outer, inner = cp.asarray(host_outer), cp.asarray(host_inner)
            output = cp.full(rows + 5, -999, dtype=cp.int64)
            calls, factory_ns = levels(
                backend,
                _compute.awkward_ListOffsetArray_flatten_offsets,
                _listoffsetarray.flatten_offsets,
                compute,
                stream,
                output,
                outer,
                inner,
            )

            def finish():
                cuda.synchronize_cuda(stream)

            finish()
            for call in calls.values():
                for _ in range(args.warmup):
                    call()
                    finish()
                np.testing.assert_array_equal(output.get(stream=stream), expected)
            samples = {name: [] for name in calls}
            names = list(calls)
            # Rotate order between repeats; all compilation/preparation and
            # warmup preceded this unprofiled timing pass.
            for repeat in range(args.repeats):
                for name in names[repeat % len(names) :] + names[: repeat % len(names)]:
                    measurements = [
                        measure_call(cp, stream, calls[name], finish)
                        for _ in range(args.iterations)
                    ]
                    samples[name].append(
                        {
                            key: statistics.median(x[key] for x in measurements)
                            for key in measurements[0]
                        }
                    )
            case = {
                "rows": rows,
                "factory_setup_us_excluded": factory_ns / 1000,
                "levels": {},
            }
            # Separate profiling pass: do not wrap timed samples in cProfile.
            for name, call in calls.items():
                profiler = cProfile.Profile()
                profile_calls(profiler, call, finish, args.profile_iterations)
                np.testing.assert_array_equal(output.get(stream=stream), expected)
                prefix = directory / f"flatten-{rows}-{name}"
                profiler.dump_stats(str(prefix) + ".pstats")
                with open(str(prefix) + ".txt", "w") as stream_file:
                    stats = pstats.Stats(profiler, stream=stream_file).strip_dirs()
                    stats.sort_stats("cumulative").print_stats(50)
                    stats.sort_stats("tottime").print_stats(50)
                case["levels"][name] = {
                    "samples": samples[name],
                    "median": {
                        key: statistics.median(x[key] for x in samples[name])
                        for key in samples[name][0]
                    },
                    "profile": prefix.name + ".pstats",
                }
            report["cases"].append(case)
            (directory / "dispatch.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
    lines = [
        "# Warmed flatten_offsets dispatch diagnostics",
        "",
        "Each level uses the same preallocated buffers, gather, item count and nondefault stream. cProfile runs separately. Stream time includes submission gaps; it is not pure kernel time.",
        "",
        "| Rows | Call level | Host µs | Completion µs | Stream µs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        for name, entry in case["levels"].items():
            m = entry["median"]
            lines.append(
                f"| {case['rows']} | {name} | {m['host_us']:.3f} | {m['completion_us']:.3f} | {m['stream_us']:.3f} |"
            )
    (directory / "dispatch.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[32, 4096])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-iterations", type=int, default=100)
    args = parser.parse_args()
    if (
        min(
            *args.rows,
            args.warmup,
            args.iterations,
            args.repeats,
            args.profile_iterations,
        )
        <= 0
    ):
        parser.error("All counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
