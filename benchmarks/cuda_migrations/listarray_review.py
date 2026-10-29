# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""First-use/varying-shape and warmed long-index regression measurements.

First use includes process-local JIT/setup, but may use compiler disk caches.
Warmed measurements exclude preparation, resets and parity comparisons. Each
completed sample includes the call, stream completion and deferred error drain.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def shapes():
    return [(rows, 2 + rows % 7) for rows in range(2, 42)]


def compilation_probe(jit, call):
    """Count stateful operator compiles in a separate diagnostic call."""
    original = jit._StatefulOp.compile
    compiled = []

    def track(operator, *args, **kwargs):
        compiled.append(operator.func.__qualname__)
        return original(operator, *args, **kwargs)

    jit._StatefulOp.compile = track
    try:
        result = call()
    finally:
        jit._StatefulOp.compile = original
    return compiled, result


def run(args):
    import awkward as ak
    import cupy as cp
    from awkward._backends.cupy import CupyBackend
    from awkward._connect import cuda
    from cuda.compute import _jit

    checkout = args.checkout.resolve()
    if not Path(ak.__file__).resolve().is_relative_to(checkout):
        raise ValueError("Install --checkout before running")
    stream = cp.cuda.Stream(non_blocking=True)
    results = []

    def finish():
        stream.synchronize()
        if stream.ptr in cuda.cuda_streamptr_to_contexts:
            cuda.synchronize_cuda(stream)

    def invoke(kernel, values):
        error = None
        begin = time.perf_counter_ns()
        try:
            error = kernel(*values)
        except ValueError as caught:
            error = caught
        submitted = time.perf_counter_ns()
        try:
            finish()
        except ValueError as caught:
            error = caught
        ended = time.perf_counter_ns()
        return error, (submitted - begin) / 1000, (ended - begin) / 1000

    with stream:
        # More than 32 lengths exposes setup eviction/shape specializations.
        # All arrays and expected values exist before measuring each call.
        key = ("awkward_ListArray_getitem_jagged_expand", *([np.int64] * 6))
        kernel = CupyBackend.instance()[key]
        for rows, width in [*shapes(), shapes()[0]]:
            host_starts = np.arange(rows, dtype=np.int64) * width
            output = cp.empty(rows * width, dtype=cp.int64)
            values = [
                cp.empty_like(output),
                cp.empty_like(output),
                cp.arange(width + 1, dtype=cp.int64),
                output,
                cp.asarray(host_starts),
                cp.asarray(host_starts + width),
                width,
                rows,
            ]
            finish()
            error, host, completed = invoke(kernel, values)
            if error is not None and getattr(error, "str", None) is not None:
                raise AssertionError(error)
            if isinstance(error, Exception):
                raise error
            np.testing.assert_array_equal(cp.asnumpy(output), np.arange(rows * width))
            results.append(
                dict(
                    case=f"varying_jagged:{rows}:{width}",
                    host_us=host,
                    completed_us=completed,
                )
            )

        key = ("awkward_ListArray_getitem_next_array", *([np.int64] * 5))
        kernel = CupyBackend.instance()[key]
        for rows, width in [(32, 8), (4096, 8), (4, 1048576)]:
            starts = np.arange(rows, dtype=np.int64) * width
            indices = np.arange(width, dtype=np.int64)
            output = cp.empty(rows * width, dtype=cp.int64)
            advanced = cp.empty_like(output)
            values = [
                output,
                advanced,
                cp.asarray(starts),
                cp.asarray(starts + width),
                cp.asarray(indices),
                rows,
                width,
                rows * width,
            ]
            for _ in range(args.warmup):
                invoke(kernel, values)
            samples = []
            for _ in range(args.repeats):
                error, host, completed = invoke(kernel, values)
                if isinstance(error, Exception) or (
                    error is not None and getattr(error, "str", None) is not None
                ):
                    raise AssertionError(error)
                samples.append((host, completed))
            np.testing.assert_array_equal(cp.asnumpy(output), np.arange(rows * width))
            np.testing.assert_array_equal(cp.asnumpy(advanced), np.tile(indices, rows))
            # Instrument only after all timing samples and parity comparisons.
            # A warmed repeat should not recompile a stateful iterator/operator.
            compiled, (error, _, _) = compilation_probe(
                _jit, lambda: invoke(kernel, values)
            )
            if isinstance(error, Exception) or (
                error is not None and getattr(error, "str", None) is not None
            ):
                raise AssertionError(error)
            results.append(
                dict(
                    case=f"long_next_array:{rows}:{width}",
                    host_us=statistics.median(x[0] for x in samples),
                    completed_us=statistics.median(x[1] for x in samples),
                    stateful_compiles_in_untimed_repeat=compiled,
                )
            )
    cache = {}
    try:
        from awkward._connect.cuda import _listarray_prepared, _listarray_slicing

        cache = {
            "prepared": _listarray_prepared._setup.cache_info()._asdict(),
            "slicing": _listarray_slicing._setup.cache_info()._asdict(),
        }
    except ImportError:
        pass  # RawKernel baseline has no prepared ListArray module.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(checkout=str(checkout), cases=results, setup_cache=cache), indent=2
        )
        + "\n"
    )


def compare(args):
    baseline, candidate = (
        json.loads(path.read_text()) for path in (args.baseline, args.candidate)
    )
    if [x["case"] for x in baseline["cases"]] != [
        x["case"] for x in candidate["cases"]
    ]:
        raise ValueError("Case inventories differ")
    text = [
        "# ListArray review regression measurements",
        "",
        "First-use rows include setup/JIT and may use disk caches; long_next_array rows are warmed. Completed time includes stream completion/error draining. Parity is checked outside timing. Compilation counts come from a separate untimed repeat after warmup, and do not perturb timing samples.",
        "",
        "| Case | Host µs baseline → candidate | Completed µs baseline → candidate | Untimed repeat compiles baseline → candidate |",
        "| --- | --- | --- | --- |",
    ]
    for left, right in zip(baseline["cases"], candidate["cases"], strict=True):
        counts = [
            str(len(case["stateful_compiles_in_untimed_repeat"]))
            if "stateful_compiles_in_untimed_repeat" in case
            else "—"
            for case in (left, right)
        ]
        text.append(
            f"| {left['case']} | {left['host_us']:.3f} → {right['host_us']:.3f} | {left['completed_us']:.3f} → {right['completed_us']:.3f} | {counts[0]} → {counts[1]} |"
        )
    text += ["", f"Candidate setup cache: `{candidate['setup_cache']}`", ""]
    args.output.write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("run")
    worker.add_argument("--checkout", required=True, type=Path)
    worker.add_argument("--output", required=True, type=Path)
    worker.add_argument("--warmup", type=int, default=20)
    worker.add_argument("--repeats", type=int, default=25)
    report = commands.add_parser("compare")
    report.add_argument("--baseline", required=True, type=Path)
    report.add_argument("--candidate", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (run if args.command == "run" else compare)(args)


if __name__ == "__main__":
    main()
