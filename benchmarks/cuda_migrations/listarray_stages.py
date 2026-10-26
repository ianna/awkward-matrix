# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Separate warmed ListArray validation, reduction and writing costs."""

import argparse
import cProfile
import json
import pstats
import statistics
from pathlib import Path

import numpy as np
from dispatch import measure_call, profile_calls
from measure import environment


def inputs(kind, rows, width, dtype):
    starts = (np.arange(rows) * width).astype(dtype)
    stops = (starts + width).astype(dtype)
    if kind == "advanced":
        return (
            [
                np.full(rows + 3, 123, dtype=np.int64),
                np.full(rows + 3, 123, dtype=np.int64),
                starts,
                stops,
                np.arange(width, dtype=np.int64),
                (np.arange(rows) % width).astype(np.int64),
            ],
            rows * width,
            "awkward_ListArray_getitem_next_array_advanced",
        )
    output = np.full(rows * width + 3, 123, dtype=np.int64)
    return (
        [
            output.copy(),
            output.copy(),
            np.arange(width + 1, dtype=np.int64),
            output.copy(),
            starts,
            stops,
        ],
        width,
        "awkward_ListArray_getitem_jagged_expand",
    )


def stages(setup, first, status, partials, storage, stream):
    return {
        "validation": lambda: setup.transform(
            "check", setup.check, partials, setup.chunks, stream
        ),
        "reduction": lambda: setup.reducer(
            temp_storage=storage,
            d_in=partials,
            d_out=first,
            num_items=setup.chunks,
            op=setup.reduce_operation,
            h_init=setup.initial,
            stream=stream,
        ),
        "write_and_status": lambda: setup.transform(
            "write",
            setup.write,
            setup.discard,
            getattr(setup, "items", setup.length),
            stream,
        ),
    }


def run(args):
    import awkward as ak
    import cupy as cp
    from awkward._backends.numpy import NumpyBackend
    from awkward._connect.cuda import _listarray_prepared as prepared
    from awkward._connect.cuda._listarray import _Stream

    checkout = Path(args.checkout).resolve()
    if checkout not in Path(ak.__file__).resolve().parents:
        raise ValueError("Imported awkward does not belong to --checkout")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    stream = cp.cuda.Stream(non_blocking=True)
    report = {
        "environment": environment(cp, checkout),
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key not in ("checkout", "output")
        },
        "cases": [],
    }
    dtype = np.dtype(args.dtype).type
    for kind in ("advanced", "jagged"):
        host, parameter, name = inputs(kind, args.rows, args.width, dtype)
        expected = [array.copy() for array in host]
        key = (name, *(array.dtype.type for array in host))
        error = (
            NumpyBackend.instance()[key](*expected, args.rows, parameter)
            if kind == "advanced"
            else NumpyBackend.instance()[key](*expected, parameter, args.rows)
        )
        if error.str:
            raise AssertionError("Stage fixture unexpectedly fails the CPU kernel")
        with stream, prepared._lock:
            arrays = tuple(cp.asarray(array) for array in host)
            setup = prepared._setup(
                cp.cuda.runtime.getDevice(),
                cp.cuda.driver.ctxGetCurrent(),
                kind,
                tuple(array.dtype.str for array in arrays),
                args.rows,
                parameter,
            )
            first, status = cp.empty(1, dtype=cp.int64), cp.empty(2, dtype=cp.int64)
            partials = cp.empty(setup.chunks, dtype=cp.int64)
            setup.bind((*arrays, first, status))
            try:
                # Build/load the reduction and initialize first, outside timing.
                setup.find(first, _Stream())
                stream.synchronize()
                storage_bytes = setup.reducer(
                    temp_storage=None,
                    d_in=partials,
                    d_out=first,
                    num_items=setup.chunks,
                    op=setup.reduce_operation,
                    h_init=setup.initial,
                    stream=_Stream(),
                )
                storage = cp.empty(storage_bytes, dtype=cp.uint8)
                calls = stages(setup, first, status, partials, storage, _Stream())
                for call in calls.values():
                    for _ in range(args.warmup):
                        call()
                        stream.synchronize()
                samples = {stage: [] for stage in calls}
                names = list(calls)
                for repeat in range(args.repeats):
                    shift = repeat % len(names)
                    for stage in names[shift:] + names[:shift]:
                        measured = [
                            measure_call(cp, stream, calls[stage], stream.synchronize)
                            for _ in range(args.iterations)
                        ]
                        samples[stage].append(
                            {
                                key: statistics.median(x[key] for x in measured)
                                for key in measured[0]
                            }
                        )
                for stage, call in calls.items():
                    profiler = cProfile.Profile()
                    profile_calls(
                        profiler, call, stream.synchronize, args.profile_iterations
                    )
                    prefix = directory / f"{kind}-{args.rows}-{args.width}-{stage}"
                    profiler.dump_stats(str(prefix) + ".pstats")
                    with open(str(prefix) + ".txt", "w") as stream_file:
                        stats = pstats.Stats(profiler, stream=stream_file).strip_dirs()
                        stats.sort_stats("cumulative").print_stats(40)
                        stats.sort_stats("tottime").print_stats(40)
                    report["cases"].append(
                        {
                            "kind": kind,
                            "stage": stage,
                            "samples": samples[stage],
                            "median": {
                                key: statistics.median(x[key] for x in samples[stage])
                                for key in samples[stage][0]
                            },
                        }
                    )
                for i in (0, 1) if kind == "advanced" else (0, 1, 3):
                    np.testing.assert_array_equal(
                        arrays[i].get(stream=stream), expected[i]
                    )
                np.testing.assert_array_equal(
                    status.get(stream=stream), [args.rows, 9223372036854775807]
                )
            finally:
                setup.clear()
    (directory / "listarray-stages.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    lines = [
        "# Warmed ListArray stage diagnostics",
        "",
        "Fixed buffers and workspace are preallocated; creation, host status transfer and production buffer rebinding are excluded. Each stage is fenced independently: do not sum its completion times to estimate whole-call latency. CUDA-event times include host submission gaps. cProfile runs separately; end-to-end comparisons remain in the kernel/public harnesses.",
        "",
        "| Operation | Stage | Host µs | Completed µs | Stream µs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        median = case["median"]
        lines.append(
            f"| {case['kind']} | {case['stage']} | {median['host_us']:.3f} | {median['completion_us']:.3f} | {median['stream_us']:.3f} |"
        )
    (directory / "listarray-stages.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=["int32", "uint32", "int64"], default="int64"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-iterations", type=int, default=100)
    args = parser.parse_args()
    if (
        args.rows <= 64
        or min(
            args.width,
            args.warmup,
            args.iterations,
            args.repeats,
            args.profile_iterations,
        )
        < 1
    ):
        parser.error("Stage profiling requires rows > 64 and positive counts")
    run(args)


if __name__ == "__main__":
    main()
