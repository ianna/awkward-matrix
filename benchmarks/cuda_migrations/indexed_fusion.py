# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Experimental fused validation/reduction; never changes production dispatch."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from dispatch import measure_call
from indexed_stages import inputs
from measure import environment


def error_operator(outer, length, innerlength):
    # g(row) is row for an invalid index and length otherwise. g(g(row))=g(row).
    # f(a,b)=min(g(a),g(b)) is associative/commutative on rows plus the sentinel.
    # Keeping accumulator values in that same domain avoids a TransformIterator.
    def earliest(left, right):
        a = np.int64(length)
        if left < length:
            index = np.int64(outer[left])
            if index >= 0 and index >= innerlength:
                a = left
        b = np.int64(length)
        if right < length:
            index = np.int64(outer[right])
            if index >= 0 and index >= innerlength:
                b = right
        return a if a < b else b

    return earliest


def prototype_class(prepared, cp, make_reduce_into, make_op_adapter):
    class Fused(prepared._Simplify):
        def __init__(self, output_dtype, outer_dtype, inner_dtype, length, innerlength):
            if length <= 64:
                raise ValueError("Fusion prototype requires more than 64 rows")
            super().__init__(
                output_dtype, outer_dtype, inner_dtype, length, innerlength
            )
            self.fused_operation = error_operator(self.outer, length, innerlength)
            self.workspace_bytes = None

        def __call__(self, output, outer, inner, stream):
            status = cp.empty(2, dtype=cp.int64)
            states = (self.output, self.outer, self.inner, self.status)
            for state, array in zip(
                states, (output, outer, inner, status), strict=True
            ):
                state.bind(array)
            try:
                first = status[:1]
                if self.reducer is None:
                    self.reduce_operation = make_op_adapter(self.fused_operation)
                    self.reducer = make_reduce_into(
                        d_in=self.iterator,
                        d_out=first,
                        op=self.reduce_operation,
                        h_init=self.initial,
                    )
                arguments = dict(
                    d_in=self.iterator,
                    d_out=first,
                    num_items=self.length,
                    op=self.reduce_operation,
                    h_init=self.initial,
                    stream=stream,
                )
                if self.workspace_bytes is None:
                    self.workspace_bytes = self.reducer(temp_storage=None, **arguments)
                storage = cp.empty(self.workspace_bytes, dtype=cp.uint8)
                self.reducer(temp_storage=storage, **arguments)
                # Reuse production's guarded parallel gather and attempt packing.
                self.transform("write", self.write, self.discard, self.length, stream)
            finally:
                for state in states:
                    state.clear()
            return status

    return Fused


def run(args):
    import awkward as ak
    import cupy as cp
    from awkward._backends.numpy import NumpyBackend
    from awkward._connect.cuda import _indexedarray as prepared
    from cuda.compute import make_reduce_into
    from cuda.compute.op import make_op_adapter

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

    def checkpoint():
        (directory / "indexed-fusion.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    Fused = prototype_class(prepared, cp, make_reduce_into, make_op_adapter)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream, prepared._lock:
        for kind, error_row in (
            ("valid", None),
            ("bounds_first", 0),
            ("bounds_middle", args.rows // 2),
            ("bounds_last", args.rows - 1),
        ):
            host = inputs(args.rows, args.width)
            if error_row is not None:
                host[1][error_row] = args.width
            expected = host[0].copy()
            error = NumpyBackend.instance()[
                "awkward_IndexedArray_simplify", np.int64, np.int64, np.int64
            ](expected, host[1], args.rows, host[2], args.width)
            expected_status = (
                [error.id, error.attempt]
                if error.str
                else [args.rows, np.iinfo(np.int64).max]
            )
            arrays = tuple(cp.asarray(x) for x in host)
            dtype_arguments = (*(x.dtype.str for x in arrays), args.rows, args.width)
            variants = {
                "production": prepared._Simplify(*dtype_arguments),
                "fused": Fused(*dtype_arguments),
            }
            if hasattr(variants["production"], "combine"):
                raise ValueError(
                    "Production is already fused; use checkout 712d13f0b for the historical prototype comparison"
                )
            samples = {name: [] for name in variants}
            errors = {}

            def call(setup):
                status = setup(*arrays, prepared._Stream())
                observed = status.get().tolist()
                return observed

            # Any compile/launch failure aborts this worker: continuing after a
            # CUDA launch failure risks trusting a poisoned device context.
            for name, setup in variants.items():
                try:
                    for _ in range(args.warmup):
                        arrays[0].fill(123)
                        observed = call(setup)
                        assert observed == expected_status
                        np.testing.assert_array_equal(arrays[0].get(), expected)
                except Exception as failure:
                    errors[name] = {
                        "type": type(failure).__name__,
                        "message": str(failure),
                    }
                    report["cases"].append({"kind": kind, "errors": errors})
                    checkpoint()
                    raise
            names = list(variants)
            for repeat in range(args.repeats):
                for name in names if repeat % 2 == 0 else names[::-1]:
                    measurements = []
                    for _ in range(args.iterations):
                        arrays[0].fill(123)
                        stream.synchronize()
                        observed = []

                        def measured():
                            observed.extend(call(variants[name]))

                        measurements.append(
                            measure_call(cp, stream, measured, stream.synchronize)
                        )
                        assert observed == expected_status
                        np.testing.assert_array_equal(arrays[0].get(), expected)
                    samples[name].append(
                        {
                            k: statistics.median(x[k] for x in measurements)
                            for k in measurements[0]
                        }
                    )
            medians = {
                name: {
                    k: statistics.median(x[k] for x in samples[name])
                    for k in samples[name][0]
                }
                for name in variants
            }
            report["cases"].append(
                {
                    "kind": kind,
                    "samples": samples,
                    "median": medians,
                    "completion_ratio": medians["fused"]["completion_us"]
                    / medians["production"]["completion_us"],
                    "parity_passed": True,
                }
            )
            checkpoint()
    report["complete"] = True
    checkpoint()
    lines = [
        "# Experimental IndexedArray validation/reduction fusion",
        "",
        "Prepared calls plus 16-byte status transfer, not public operations. Both variants include per-call scratch, rebinding and stream completion. Inputs, resets, parity checks and JIT warmup are outside timing. Order alternates by repeat. No backend dispatch/error-context capture. No production dispatch changes. Stream time includes submission gaps and is not pure device time.",
        "",
        "| Input | Production completed µs | Fused completed µs | Fused / production |",
        "| --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        m = case["median"]
        lines.append(
            f"| {case['kind']} | {m['production']['completion_us']:.3f} | {m['fused']['completion_us']:.3f} | {case['completion_ratio']:.3f} |"
        )
    (directory / "indexed-fusion.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if (
        args.rows <= 64
        or min(args.width, args.warmup, args.iterations, args.repeats) < 1
    ):
        parser.error("rows must exceed 64; all other counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
