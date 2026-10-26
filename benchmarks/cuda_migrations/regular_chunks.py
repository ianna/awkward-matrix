# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Sweep bounded regular-validation chunks before changing the production default."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from measure import allocation_hook, environment
from public_operations import measure_completed


def offsets_for(rows, dtype, pattern):
    if rows < 2:
        raise ValueError("At least two rows are required")
    offsets = np.arange(rows + 1, dtype=dtype) * 4
    if pattern == "early_error":
        offsets[2:] += 1
    elif pattern == "late_error":
        offsets[-1] += 1
    elif pattern != "regular":
        raise ValueError(f"Unknown pattern: {pattern}")
    return offsets


def expected_status(error):
    if not error.str:
        return 0
    code = error.id + 1
    return -code if b"monotonically" in error.str else code


def write_report(directory, report):
    (directory / "regular-chunks.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Warmed toRegularArray chunk sweep",
        "",
        f"Candidate: `{report['environment']['git_sha']}`. No production chunk size is selected by this benchmark.",
        "",
        "Each call includes prepared validation/reduction and the canonical 8-byte status read. Stream completion follows every call. Warmup, CPU parity checks and allocation profiling are outside timing; allocations use a separate pass. Size/output buffers are preallocated. This is the internal validation boundary, not the public-operation benchmark.",
        "",
        "| Rows | Dtype | Pattern | Chunk | Error-buffer bytes | Host µs | Completed µs | Completed / chunk 1 | Pool temporary peak bytes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    references = {
        (entry["rows"], entry["dtype"], entry["pattern"]): entry["median"][
            "completion_us"
        ]
        for entry in report["cases"]
        if entry["chunk_size"] == 1
    }
    for entry in report["cases"]:
        median = entry["median"]
        reference = references.get((entry["rows"], entry["dtype"], entry["pattern"]))
        ratio = f"{median['completion_us'] / reference:.3f}" if reference else "pending"
        lines.append(
            f"| {entry['rows']} | {entry['dtype']} | {entry['pattern']} | {entry['chunk_size']} | {entry['error_buffer_bytes']} | {median['host_us']:.3f} | {median['completion_us']:.3f} | {ratio} | {entry['pool_temporary_peak_bytes']} |"
        )
    (directory / "regular-chunks.md").write_text("\n".join(lines) + "\n")


def run(args):
    import awkward as ak
    import cupy as cp
    from awkward._backends.numpy import NumpyBackend
    from awkward._connect.cuda._listoffsetarray import _Stream
    from awkward._connect.cuda._prepared_gather import regular_status

    checkout = Path(args.checkout).resolve()
    if checkout not in Path(ak.__file__).resolve().parents:
        raise ValueError("Imported awkward does not belong to --checkout")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": environment(cp, checkout),
        "settings": {
            "rows": args.rows,
            "chunks": args.chunks,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
        },
        "cases": [],
    }
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for rows in args.rows:
            for dtype in (np.int32, np.uint32, np.int64):
                for pattern in ("regular", "early_error", "late_error"):
                    host = offsets_for(rows, dtype, pattern)
                    expected = np.full(3, 123, dtype=np.int64)
                    error = NumpyBackend.instance()[
                        "awkward_ListOffsetArray_toRegularArray", np.int64, dtype
                    ](expected, host, len(host))
                    code = expected_status(error)
                    source = cp.asarray(host)
                    calls, sizes = {}, {}
                    for chunk in args.chunks:
                        sizes[chunk] = cp.full(3, 123, dtype=cp.int64)

                        def call(chunk=chunk):
                            status = regular_status(
                                sizes[chunk],
                                source,
                                len(host),
                                _Stream(),
                                chunk_size=chunk,
                            )
                            return int(status.get(stream=stream)[0])

                        calls[chunk] = call
                    stream.synchronize()
                    for chunk, call in calls.items():
                        for _ in range(args.warmup):
                            value = call()
                            stream.synchronize()
                            if value != code:
                                raise AssertionError(
                                    f"Incorrect status for chunk {chunk}: {value} != {code}"
                                )
                            np.testing.assert_array_equal(
                                sizes[chunk].get(stream=stream), expected
                            )
                    samples = {chunk: [] for chunk in args.chunks}
                    # Rotate chunk order to avoid always favoring one warm state.
                    for repeat in range(args.repeats):
                        shift = repeat % len(args.chunks)
                        for chunk in args.chunks[shift:] + args.chunks[:shift]:
                            measurements = []
                            for _ in range(args.iterations):
                                metrics, value, errors = measure_completed(
                                    cp, stream, calls[chunk], stream.synchronize
                                )
                                if errors or value != code:
                                    raise AssertionError(
                                        f"Chunk {chunk} failed: {errors}, status={value}"
                                    )
                                measurements.append(metrics)
                            samples[chunk].append(
                                {
                                    key: statistics.median(x[key] for x in measurements)
                                    for key in measurements[0]
                                }
                            )
                    for chunk, call in calls.items():
                        hook = allocation_hook(cp)
                        with hook:
                            value = call()
                        stream.synchronize()
                        if value != code:
                            raise AssertionError("Allocation pass failed parity")
                        np.testing.assert_array_equal(
                            sizes[chunk].get(stream=stream), expected
                        )
                        parts = (rows + chunk - 1) // chunk
                        report["cases"].append(
                            {
                                "rows": rows,
                                "dtype": np.dtype(dtype).name,
                                "pattern": pattern,
                                "chunk_size": chunk,
                                "chunks": parts,
                                "status": "passed",
                                "error_buffer_bytes": 0 if parts == 1 else 8 * parts,
                                "pool_temporary_peak_bytes": hook.peak_live_bytes,
                                "allocation_requests": hook.requests,
                                "requested_bytes": hook.requested_bytes,
                                "samples": samples[chunk],
                                "median": {
                                    key: statistics.median(
                                        x[key] for x in samples[chunk]
                                    )
                                    for key in samples[chunk][0]
                                },
                            }
                        )
                    write_report(directory, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[32, 4096, 131072])
    parser.add_argument(
        "--chunks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if (
        min(args.rows) < 2
        or min(*args.chunks, args.warmup, args.iterations, args.repeats) < 1
    ):
        parser.error("Rows must be at least 2; other counts must be positive")
    if len(set(args.chunks)) != len(args.chunks) or 1 not in args.chunks:
        parser.error("Chunk sizes must be unique and include the chunk-1 reference")
    run(args)


if __name__ == "__main__":
    main()
