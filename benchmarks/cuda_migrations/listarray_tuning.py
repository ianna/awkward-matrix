# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Compare ListArray tuning candidates without changing large-input defaults."""

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import listarray_stages
import numpy as np
from measure import allocation_hook, environment
from public_operations import measure_completed

MESSAGES = {
    "advanced": (
        "stops[i] < starts[i]",
        "stops[i] > len(content)",
        "index out of range",
    ),
    "jagged": ("stops[i] < starts[i]", "cannot fit jagged slice into nested list"),
}


def operation(prepared, decode, kind, arrays, length, parameter, stream, variant, name):
    def call():
        options = (
            {"spread_lanes": variant} if kind == "spread" else {"chunk_size": variant}
        )
        status = prepared.run(kind, arrays, length, parameter, stream, **options)
        return (
            None if kind == "spread" else decode(status, length, MESSAGES[kind], name)
        )

    return call


def write_report(directory, report):
    (directory / "listarray-tuning.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    lines = [
        "# ListArray tuning candidates",
        "",
        f"Candidate: `{report['environment']['git_sha']}`. Large-input defaults are not selected automatically.",
        "",
        "Validation variants are chunk sizes (reference 64); spread variants are lane counts (reference 0, current lookup). Fixed input/output buffers are preallocated, but temporary allocations, rebinding and canonical host status reads are included. JIT warmup and CPU parity checks are excluded. Allocation hooks run in a separate pass. This isolates the implementation, excluding backend dispatch/error-context bookkeeping.",
        "",
        "| Workload | Variant | Host µs | Completed µs | Completed / reference | Pool temporary peak bytes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        reference = 0 if case["kind"] == "spread" else 64
        base = case["variants"][str(reference)]["median"]["completion_us"]
        for variant, entry in case["variants"].items():
            m = entry["median"]
            lines.append(
                f"| {case['name']} | {variant} | {m['host_us']:.3f} | {m['completion_us']:.3f} | {m['completion_us'] / base:.3f} | {entry['pool_temporary_peak_bytes']} |"
            )
    (directory / "listarray-tuning.md").write_text("\n".join(lines) + "\n")


def run(args):
    import awkward as ak
    import cupy as cp
    from awkward._backends.numpy import NumpyBackend
    from awkward._connect.cuda import _listarray_prepared as prepared
    from awkward._connect.cuda._listarray import _prepared_error, _Stream

    checkout = Path(args.checkout).resolve()
    if checkout not in Path(ak.__file__).resolve().parents:
        raise ValueError("Imported awkward does not belong to --checkout")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": environment(cp, checkout),
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key not in ("checkout", "output")
        },
        "cases": [],
    }
    tasks = []
    for kind in ("advanced", "jagged"):
        arrays, parameter, name = listarray_stages.inputs(kind, 4096, 8, np.int64)
        tasks.append(
            (kind, arrays, 4096, parameter, name, args.chunks, f"{kind}:4096:8")
        )
    for rows, width in ((32, 8), (4096, 8), (32, 4096)):
        arrays = [
            np.full(rows * width + 3, 123, dtype=np.int64),
            np.arange(rows, dtype=np.int64) + 7,
            np.arange(rows + 1, dtype=np.int64) * width,
        ]
        tasks.append(
            (
                "spread",
                arrays,
                rows,
                0,
                "awkward_ListArray_getitem_next_range_spreadadvanced",
                args.spread_lanes,
                f"spread:{rows}:{width}",
            )
        )
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for kind, host, length, parameter, name, variants, label in tasks:
            expected = [array.copy() for array in host]
            key = (name, *(array.dtype.type for array in host))
            scalars = (
                (length,)
                if kind == "spread"
                else (
                    (length, parameter) if kind == "advanced" else (parameter, length)
                )
            )
            error = NumpyBackend.instance()[key](*expected, *scalars)
            if error.str:
                raise AssertionError("Tuning fixture unexpectedly fails CPU validation")
            arrays = tuple(cp.asarray(array) for array in host)
            calls = {
                variant: operation(
                    prepared,
                    _prepared_error,
                    kind,
                    arrays,
                    length,
                    parameter,
                    _Stream(),
                    variant,
                    name,
                )
                for variant in variants
            }
            for call in calls.values():
                for _ in range(args.warmup):
                    # Reset outputs so a skipped write cannot pass parity.
                    for i in (
                        (0,)
                        if kind == "spread"
                        else ((0, 1) if kind == "advanced" else (0, 1, 3))
                    ):
                        arrays[i].fill(123)
                    stream.synchronize()
                    result = call()
                    stream.synchronize()
                    if result is not None:
                        raise AssertionError(f"Unexpected kernel error: {result}")
                    for i in (
                        (0,)
                        if kind == "spread"
                        else ((0, 1) if kind == "advanced" else (0, 1, 3))
                    ):
                        np.testing.assert_array_equal(
                            arrays[i].get(stream=stream), expected[i]
                        )
            samples = {variant: [] for variant in variants}
            for repeat in range(args.repeats):
                shift = repeat % len(variants)
                for variant in variants[shift:] + variants[:shift]:
                    measured = []
                    for _ in range(args.iterations):
                        metrics, result, errors = measure_completed(
                            cp, stream, calls[variant], stream.synchronize
                        )
                        if errors or result is not None:
                            raise AssertionError(f"Tuning failed: {errors}, {result}")
                        measured.append(metrics)
                    samples[variant].append(
                        {
                            key: statistics.median(x[key] for x in measured)
                            for key in measured[0]
                        }
                    )
            case = {"kind": kind, "name": label, "variants": {}}
            for variant, call in calls.items():
                hook = allocation_hook(cp)
                with hook:
                    result = call()
                stream.synchronize()
                if result is not None:
                    raise AssertionError("Allocation pass failed")
                for i in (
                    (0,)
                    if kind == "spread"
                    else ((0, 1) if kind == "advanced" else (0, 1, 3))
                ):
                    np.testing.assert_array_equal(
                        arrays[i].get(stream=stream), expected[i]
                    )
                case["variants"][str(variant)] = {
                    "samples": samples[variant],
                    "median": {
                        key: statistics.median(x[key] for x in samples[variant])
                        for key in samples[variant][0]
                    },
                    "pool_temporary_peak_bytes": hook.peak_live_bytes,
                }
            report["cases"].append(case)
            write_report(directory, report)
    if args.stages:
        for chunk in args.chunks:
            listarray_stages.run(
                SimpleNamespace(
                    checkout=args.checkout,
                    output=directory / f"stages-chunk-{chunk}",
                    rows=4096,
                    width=8,
                    dtype="int64",
                    chunk_size=chunk,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    profile_iterations=args.profile_iterations,
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunks", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--spread-lanes", nargs="+", type=int, default=[0, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-iterations", type=int, default=100)
    parser.add_argument("--stages", action="store_true")
    args = parser.parse_args()
    if (
        min(
            *args.chunks,
            args.warmup,
            args.iterations,
            args.repeats,
            args.profile_iterations,
        )
        < 1
        or min(args.spread_lanes) < 0
    ):
        parser.error("Counts must be positive; spread lanes may also be zero")
    if (
        64 not in args.chunks
        or 0 not in args.spread_lanes
        or len(set(args.chunks)) != len(args.chunks)
        or len(set(args.spread_lanes)) != len(args.spread_lanes)
    ):
        parser.error("Variants must be unique and include references 64 and 0")
    run(args)


if __name__ == "__main__":
    main()
