# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Independently fenced prepared IndexedArray simplification stage diagnostics."""

import argparse
import cProfile
import json
import pstats
import statistics
from pathlib import Path

import numpy as np
from dispatch import measure_call, profile_calls
from measure import environment


def inputs(rows, width, error=False):
    outer = np.arange(rows, dtype=np.int64) % width
    outer[::5] = -1
    if error:
        outer[rows // 2] = width
    inner = np.arange(width, dtype=np.int64)[::-1].copy()
    inner[::3] = -1
    return np.full(rows + 3, 123, dtype=np.int64), outer, inner


def stages(cp, setup, arrays, status, positions, storage, stream):
    states = (setup.output, setup.outer, setup.inner, setup.status)

    def bind():
        for state, array in zip(states, (*arrays, status), strict=True):
            state.bind(array)

    def clear():
        for state in states:
            state.clear()

    def bind_clear():
        bind()
        clear()

    def allocations():
        # Deliberately include return-to-pool release; no caller buffers or
        # prepared state reference these temporary allocations.
        if hasattr(setup, "reduce_storage_bytes"):
            scratch = (
                cp.empty(2 + setup.chunks, dtype=cp.int64),
                cp.empty(storage.nbytes, dtype=cp.uint8),
            )
        else:
            scratch = (
                cp.empty(2, dtype=cp.int64),
                cp.empty(setup.chunks, dtype=cp.int64),
                cp.empty(storage.nbytes, dtype=cp.uint8),
            )
        del scratch

    def workspace_query():
        return setup.reducer(
            temp_storage=None,
            d_in=positions,
            d_out=status[:1],
            num_items=setup.chunks,
            op=setup.reduce_operation,
            h_init=setup.initial,
            stream=stream,
        )

    calls = {
        "validation": lambda: setup.transform(
            "check", setup.check, positions, setup.chunks, stream
        ),
        "reduction": lambda: setup.reducer(
            temp_storage=storage,
            d_in=positions,
            d_out=status[:1],
            num_items=setup.chunks,
            op=setup.reduce_operation,
            h_init=setup.initial,
            stream=stream,
        ),
        "gather_and_attempt": lambda: setup.transform(
            "write", setup.write, setup.discard, setup.length, stream
        ),
        "workspace_query": workspace_query,
        "allocation_and_release": allocations,
        "bind_and_clear": bind_clear,
        # This is a standalone 16-byte transfer on an otherwise fenced stream.
        # It does not include waiting for a preceding validation/gather chain.
        "status_transfer": lambda: status.get().tolist(),
    }
    return calls, bind, clear


def run(args):
    import awkward as ak
    import cupy as cp
    from awkward._backends.numpy import NumpyBackend
    from awkward._connect.cuda import _indexedarray as prepared

    checkout = Path(args.checkout).resolve()
    if checkout not in Path(ak.__file__).resolve().parents:
        raise ValueError("Imported awkward does not belong to --checkout")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": environment(cp, checkout),
        "settings": {
            k: v for k, v in vars(args).items() if k not in ("checkout", "output")
        },
        "complete": False,
        "cases": [],
    }
    stream = cp.cuda.Stream(non_blocking=True)
    for invalid in (False, True):
        host = inputs(args.rows, args.width, invalid)
        expected = host[0].copy()
        error = NumpyBackend.instance()[
            "awkward_IndexedArray_simplify", np.int64, np.int64, np.int64
        ](expected, host[1], args.rows, host[2], args.width)
        with stream, prepared._lock:
            arrays = tuple(cp.asarray(x) for x in host)
            setup = prepared._setup(
                cp.cuda.runtime.getDevice(),
                cp.cuda.driver.ctxGetCurrent(),
                *(x.dtype.str for x in arrays),
                args.rows,
                args.width,
            )
            # Warm the actual pipeline and check full outputs before isolation.
            result = setup(*arrays, prepared._Stream())
            row, attempt = result.get().tolist()
            expected_status = (
                [error.id, error.attempt]
                if error.str
                else [args.rows, np.iinfo(np.int64).max]
            )
            assert [row, attempt] == expected_status
            np.testing.assert_array_equal(arrays[0].get(), expected)
            status = cp.asarray(expected_status, dtype=cp.int64)
            positions = cp.empty(setup.chunks, dtype=cp.int64)
            arguments = dict(
                d_in=positions,
                d_out=status[:1],
                num_items=setup.chunks,
                op=setup.reduce_operation,
                h_init=setup.initial,
                stream=prepared._Stream(),
            )
            storage_bytes = setup.reducer(temp_storage=None, **arguments)
            storage = cp.empty(storage_bytes, dtype=cp.uint8)
            calls, bind, clear = stages(
                cp, setup, arrays, status, positions, storage, prepared._Stream()
            )
            kind = "bounds" if invalid else "valid"
            try:
                # Initialize reduction input and first-error state in order.
                bind()
                calls["validation"]()
                calls["reduction"]()
                calls["gather_and_attempt"]()
                stream.synchronize()
                samples = {name: [] for name in calls}
                for name, call in calls.items():
                    for _ in range(args.warmup):
                        bind()
                        call()
                        stream.synchronize()
                names = list(calls)
                for repeat in range(args.repeats):
                    shift = repeat % len(names)
                    for name in names[shift:] + names[:shift]:
                        measurements = []
                        for _ in range(args.iterations):
                            bind()  # preparation outside the measured boundary
                            measurements.append(
                                measure_call(
                                    cp, stream, calls[name], stream.synchronize
                                )
                            )
                        samples[name].append(
                            {
                                k: statistics.median(x[k] for x in measurements)
                                for k in measurements[0]
                            }
                        )
                for name, call in calls.items():
                    bind()
                    profiler = cProfile.Profile()
                    profile_calls(
                        profiler, call, stream.synchronize, args.profile_iterations
                    )
                    prefix = directory / f"indexed-{kind}-{name}"
                    profiler.dump_stats(str(prefix) + ".pstats")
                    with open(str(prefix) + ".txt", "w") as output:
                        stats = pstats.Stats(profiler, stream=output).strip_dirs()
                        stats.sort_stats("cumulative").print_stats(40)
                        stats.sort_stats("tottime").print_stats(40)
                    report["cases"].append(
                        {
                            "kind": kind,
                            "stage": name,
                            "samples": samples[name],
                            "median": {
                                k: statistics.median(x[k] for x in samples[name])
                                for k in samples[name][0]
                            },
                        }
                    )
                np.testing.assert_array_equal(arrays[0].get(), expected)
                assert status.get().tolist() == expected_status
            finally:
                clear()
    report["workspace_size_cached_in_production"] = hasattr(
        setup, "reduce_storage_bytes"
    )
    report["complete"] = True
    (directory / "indexed-stages.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Warmed IndexedArray stage diagnostics",
        "",
        "Each stage is fenced independently; do not sum completion times. Validation/reduction/gather use preallocated buffers and workspace. Allocation includes pooled allocation/release; binding includes descriptor updates/clear. Status transfer is a standalone 16-byte get/tolist after prior work is fenced, not total pipeline waiting. Rebinding preparation is outside each measured boundary. CUDA-event times include host submission gaps; cProfile runs separately. Backend dispatch and error-context capture are excluded; use the public benchmark for end-to-end latency.",
        "",
        f"Production workspace sizing cached: {report['workspace_size_cached_in_production']}. The isolated workspace_query stage measures query cost even when production caches it.",
        "",
        "| Input | Stage | Host µs | Completed µs | Stream µs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        m = case["median"]
        lines.append(
            f"| {case['kind']} | {case['stage']} | {m['host_us']:.3f} | {m['completion_us']:.3f} | {m['stream_us']:.3f} |"
        )
    (directory / "indexed-stages.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=8)
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
        parser.error("rows must exceed 64 and all other counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
