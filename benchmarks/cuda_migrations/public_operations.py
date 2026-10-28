# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Compare warmed public CUDA operations through completion and error checking."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from dispatch import finish_stream
from measure import environment
from workloads import SHAPES

ENVIRONMENT_FIELDS = (
    "device",
    "packages",
    "python",
    "cuda_runtime",
    "cuda_driver",
    "execution_environment",
)


def fixture_arrays(rows, width, irregular=False):
    offsets = np.arange(rows + 1, dtype=np.int64) * width
    if irregular:
        if rows < 2 or width < 1:
            raise ValueError("Irregular fixture requires two nonempty rows")
        offsets[rows // 2] += 1
    content = np.arange(rows * width, dtype=np.int64)
    index = content.copy()
    index[::3] = -1
    return offsets, content, index


def cases(ak, rows, width):
    def array(irregular=False, option=False):
        offsets, content, index = fixture_arrays(rows, width, irregular)
        layout = ak.contents.NumpyArray(content)
        if option:
            layout = ak.contents.IndexedOptionArray(ak.index.Index64(index), layout)
        return ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(offsets), layout))

    yield "to_regular", array(), lambda x: ak.to_regular(x, axis=1)
    yield "drop_none", array(option=True), lambda x: ak.drop_none(x, axis=1)
    if rows >= 2 and width >= 1:
        yield (
            "to_regular_irregular",
            array(irregular=True),
            lambda x: ak.to_regular(x, axis=1),
        )


LISTARRAY_CASES = (
    "slice_advanced",
    "slice_jagged",
    "slice_spread",
    "slice_advanced_bounds",
    "slice_jagged_bounds",
)


def listarray_cases(ak, rows, width, backend="cpu"):
    # Use ListArray explicitly: constructing from Python lists would select
    # ListOffsetArray and could bypass the kernels under investigation.
    offsets = np.arange(rows + 1, dtype=np.int64) * width
    values = np.arange(rows * width * 3, dtype=np.int64)
    inner = ak.contents.RegularArray(ak.contents.NumpyArray(values), 3)
    source = ak.Array(
        ak.contents.ListArray(
            ak.index.Index64(offsets[:-1]),
            ak.index.Index64(offsets[1:]),
            inner,
        )
    )
    row_index = np.arange(rows, dtype=np.int64)
    column_index = np.zeros(rows, dtype=np.int64)
    inner_index = np.ones(rows, dtype=np.int64)
    jagged = ak.Array([[0, -1] for _ in range(width)])
    bad_jagged = ak.Array([[3] if i == width // 2 else [0] for i in range(width)])
    bad_column = column_index.copy()
    if rows:
        bad_column[rows // 2] = width
    source = ak.to_backend(source, backend)
    row_index, column_index, inner_index, bad_column, jagged, bad_jagged = (
        ak.to_backend(ak.Array(index), backend)
        for index in (
            row_index,
            column_index,
            inner_index,
            bad_column,
            jagged,
            bad_jagged,
        )
    )
    yield "slice_advanced", source, lambda x: x[row_index, column_index, inner_index]
    yield "slice_jagged", source, lambda x: x[:, jagged]
    yield "slice_spread", source, lambda x: x[row_index, :]
    yield (
        "slice_advanced_bounds",
        source,
        lambda x: x[row_index, bad_column, inner_index],
    )
    yield "slice_jagged_bounds", source, lambda x: x[:, bad_jagged]


INDEXED_CASES = tuple(
    f"field_{outer}_{inner}{suffix}"
    for outer in ("indexed", "option")
    for inner in ("indexed", "option")
    for suffix in ("", "_bounds")
)


def indexed_cases(ak, rows, width, backend="cpu"):
    if rows < 1 or width < 1:
        raise ValueError("Indexed fixtures require positive rows and width")
    # Indexed nodes cannot directly contain other indexed nodes. A record field
    # holds the inner indirection; public field selection composes the indices.
    for outer_option in (False, True):
        for inner_option in (False, True):
            inner_index = np.arange(width - 1, -1, -1, dtype=np.int64)
            if inner_option:
                inner_index[::3] = -1
            inner_class = (
                ak.contents.IndexedOptionArray
                if inner_option
                else ak.contents.IndexedArray
            )
            inner = inner_class(
                ak.index.Index64(inner_index),
                ak.contents.NumpyArray(np.arange(width, dtype=np.int64)),
            )
            record = ak.contents.RecordArray([inner], ["x"])
            outer_index = np.arange(rows, dtype=np.int64) % width
            if outer_option:
                outer_index[::5] = -1
            outer_class = (
                ak.contents.IndexedOptionArray
                if outer_option
                else ak.contents.IndexedArray
            )
            for invalid in (False, True):
                index = outer_index.copy()
                if invalid:
                    index[rows // 2] = width
                source = ak.Array(outer_class(ak.index.Index64(index), record))
                source = ak.to_backend(source, backend)
                name = (
                    f"field_{'option' if outer_option else 'indexed'}_"
                    f"{'option' if inner_option else 'indexed'}"
                    f"{'_bounds' if invalid else ''}"
                )
                yield name, source, lambda x: x["x"]


REGULARARRAY_CASES = (
    "regular_at",
    "regular_range",
    "regular_reverse",
    "regular_advanced",
    "regular_spread",
    "regular_broadcast",
    "regular_broadcast_ragged",
    "regular_bounds",
)


def regulararray_cases(ak, rows, width, backend="cpu"):
    if rows < 1 or width < 1:
        raise ValueError("RegularArray fixtures require positive dimensions")
    source = ak.to_backend(
        ak.Array(
            ak.contents.RegularArray(
                ak.contents.NumpyArray(np.arange(rows * width, dtype=np.int64)), width
            )
        ),
        backend,
    )
    columns = ak.to_backend(ak.Array(np.array([0, -1], dtype=np.int64)), backend)
    row_index = ak.to_backend(ak.Array(np.arange(rows, dtype=np.int64)), backend)
    singleton = ak.to_backend(
        ak.Array(
            ak.contents.RegularArray(
                ak.contents.NumpyArray(np.arange(rows, dtype=np.int64)), 1
            )
        ),
        backend,
    )
    yield "regular_at", source, lambda x: x[:, -1]
    yield "regular_range", source, lambda x: x[:, 1::2]
    yield "regular_reverse", source, lambda x: x[:, ::-1]
    yield "regular_advanced", source, lambda x: x[:, columns]
    yield "regular_spread", source, lambda x: x[row_index, :]
    yield "regular_broadcast", source, lambda x: ak.broadcast_arrays(x, singleton)[1]
    yield (
        "regular_broadcast_ragged",
        source,
        lambda x: ak.broadcast_arrays(ak.from_regular(x), singleton)[1],
    )
    yield "regular_bounds", source, lambda x: x[:, width]


def case_names(suite, rows, width):
    if suite == "listarray":
        return LISTARRAY_CASES
    if suite == "indexed":
        return INDEXED_CASES
    if suite == "regulararray":
        return REGULARARRAY_CASES
    return ("to_regular", "drop_none") + (
        ("to_regular_irregular",) if rows >= 2 and width >= 1 else ()
    )


def expectation(ak, call):
    try:
        return {"values": ak.to_list(call()), "error": None}
    except (ValueError, IndexError) as error:
        # Keep a stable semantic fragment, excluding backend-specific context.
        if isinstance(error, IndexError) and "index out of range" in str(error):
            return {"error": "index out of range", "error_type": "IndexError"}
        if (
            isinstance(error, ValueError)
            and "index out of range" in str(error)
            and "compiled code" in str(error)
        ):
            return {"error": "index out of range", "error_type": "ValueError"}
        if "not regular" not in str(error):
            raise
        return {"error": "not regular"}


def validate_errors(errors, expected):
    if expected["error"] is None:
        if errors:
            raise AssertionError(f"Unexpected CUDA error(s): {errors!r}")
    else:

        def matches(stage, error):
            message = str(error)
            if expected["error"] not in message:
                return False
            if expected.get("error_type") == "IndexError":
                # Legacy CUDA error draining wraps kernel bounds failures in
                # ValueError. Preserve the phase and require its CUDA marker;
                # an unrelated ValueError must not pass as an indexing error.
                return isinstance(error, IndexError) or (
                    stage == "finish"
                    and isinstance(error, ValueError)
                    and "index out of range in compiled CUDA code (awkward_" in message
                )
            return isinstance(error, ValueError)

        if not errors or any(not matches(stage, error) for stage, error in errors):
            raise AssertionError(f"Expected {expected['error']!r}, got {errors!r}")


def measure_completed(cp, stream, call, finish):
    """Always drain deferred errors, including when the public call raises."""
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    errors = []
    result = None
    before = time.perf_counter_ns()
    try:
        result = call()
    except Exception as error:
        errors.append(("call", error))
    submitted = time.perf_counter_ns()
    end.record(stream)
    try:
        finish()
    except Exception as error:
        errors.append(("finish", error))
    finally:
        end.synchronize()
    completed = time.perf_counter_ns()
    return (
        {
            "host_us": (submitted - before) / 1000,
            "completion_us": (completed - before) / 1000,
            "stream_us": cp.cuda.get_elapsed_time(start, end) * 1000,
        },
        result,
        errors,
    )


def run(args):
    import awkward as ak
    import cupy as cp
    from awkward._connect import cuda

    checkout = Path(args.checkout).resolve()
    if checkout not in Path(ak.__file__).resolve().parents:
        raise ValueError("Imported awkward does not belong to --checkout")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": environment(cp, checkout),
        "settings": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "shapes": SHAPES,
            "suite": getattr(args, "suite", "listoffsetarray"),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "measurements": [],
    }
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for rows, width in SHAPES:
            suite = report["settings"]["suite"]
            factory = {
                "listarray": listarray_cases,
                "indexed": indexed_cases,
                "regulararray": regulararray_cases,
            }.get(suite, cases)
            device_cases = (
                {
                    name: (source, operation)
                    for name, source, operation in factory(ak, rows, width, "cuda")
                }
                if suite in ("listarray", "indexed", "regulararray")
                else {}
            )
            for name, host, operation in factory(ak, rows, width):
                expected = expectation(ak, lambda: operation(host))
                if device_cases:
                    source, operation = device_cases[name]
                else:
                    source = ak.to_backend(host, "cuda")

                def finish():
                    finish_stream(cuda, stream)

                def call():
                    return operation(source)

                finish()
                case_id = f"{name}:{rows}:{width}"
                print(f"Checking {case_id}", flush=True)
                failure = None
                observed = []
                for _ in range(args.warmup):
                    _, result, errors = measure_completed(cp, stream, call, finish)
                    try:
                        validate_errors(errors, expected)
                        if expected["error"] is None:
                            actual = ak.to_list(ak.to_backend(result, "cpu"))
                            if actual != expected["values"]:
                                raise AssertionError(
                                    "Output differs from CPU reference"
                                )
                    except AssertionError as error:
                        failure = str(error)
                        observed = [
                            {
                                "stage": stage,
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                            for stage, error in errors
                        ]
                    del result
                    finish()
                    if failure is not None:
                        break
                if failure is not None:
                    report["measurements"].append(
                        {
                            "case": case_id,
                            "status": "parity_failed",
                            "failure": failure,
                            "expected_error": expected["error"],
                            "expected_error_type": expected.get("error_type"),
                            "observed_errors": observed,
                            "error_stages": {
                                stage: sum(e["stage"] == stage for e in observed)
                                for stage in ("call", "finish")
                            },
                            "samples": [],
                            "median": None,
                        }
                    )
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"PARITY FAILED {case_id}: {failure}", flush=True)
                    continue
                samples = []
                stages = {"call": 0, "finish": 0}
                for _ in range(args.repeats):
                    measurements = []
                    for _ in range(args.iterations):
                        metrics, result, errors = measure_completed(
                            cp, stream, call, finish
                        )
                        validate_errors(errors, expected)
                        for stage, _ in errors:
                            stages[stage] += 1
                        measurements.append(metrics)
                        del result
                    samples.append(
                        {
                            key: statistics.median(x[key] for x in measurements)
                            for key in measurements[0]
                        }
                    )
                report["measurements"].append(
                    {
                        "case": f"{name}:{rows}:{width}",
                        "status": "passed",
                        "expected_error": expected["error"],
                        "expected_error_type": expected.get("error_type"),
                        "error_stages": stages,
                        "samples": samples,
                        "median": {
                            key: statistics.median(x[key] for x in samples)
                            for key in samples[0]
                        },
                    }
                )
                output.write_text(json.dumps(report, indent=2) + "\n")


def comparison(baseline, candidate, max_regression=1.1):
    for field in ENVIRONMENT_FIELDS:
        if baseline["environment"][field] != candidate["environment"][field]:
            raise ValueError(f"Incomparable environments: {field}")
    if baseline["settings"] != candidate["settings"]:
        raise ValueError("Mismatched settings or harness")
    old = {x["case"]: x for x in baseline["measurements"]}
    new = {x["case"]: x for x in candidate["measurements"]}
    expected_cases = {
        f"{name}:{rows}:{width}"
        for rows, width in baseline["settings"]["shapes"]
        for name in case_names(
            baseline["settings"].get("suite", "listoffsetarray"), rows, width
        )
    }
    if old.keys() != expected_cases or new.keys() != expected_cases:
        raise ValueError("Missing or mismatched public workload inventory")
    entries = []
    for name, before in old.items():
        after = new[name]
        if before["status"] not in ("passed", "parity_failed") or after[
            "status"
        ] not in ("passed", "parity_failed"):
            raise ValueError(f"Unvalidated public workload: {name}")
        if (before["expected_error"], before.get("expected_error_type")) != (
            after["expected_error"],
            after.get("expected_error_type"),
        ):
            raise ValueError(f"Mismatched error expectation: {name}")
        validated = before["status"] == after["status"] == "passed"
        ratio = (
            after["median"]["completion_us"] / before["median"]["completion_us"]
            if validated
            else None
        )
        entries.append(
            {
                "case": name,
                "baseline": before,
                "candidate": after,
                "completion_ratio": ratio,
                "completion_regression": ratio > max_regression if validated else None,
                "parity_passed": validated,
            }
        )
    return {
        "baseline_sha": baseline["environment"]["git_sha"],
        "candidate_sha": candidate["environment"]["git_sha"],
        "max_regression": max_regression,
        "public_parity_passed": all(x["parity_passed"] for x in entries),
        "no_completion_regressions": all(x["parity_passed"] for x in entries)
        and not any(x["completion_regression"] for x in entries),
        "cases": entries,
    }


def compare(args):
    report = comparison(
        json.loads(Path(args.baseline).read_text()),
        json.loads(Path(args.candidate).read_text()),
        args.max_regression,
    )
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Completed public CUDA operations",
        "",
        f"Public parity passed: **{report['public_parity_passed']}**.",
        "",
        f"Baseline: `{report['baseline_sha']}`; candidate: `{report['candidate_sha']}`.",
        "",
        "Inputs and CPU reference results are prepared outside timing. JIT warmup is excluded. Each sample includes the public call, result construction, stream completion and deferred-error draining. Error classification and output comparisons are outside timing. Stream time includes submission gaps; it is not pure kernel time.",
        "",
        "Existing kernel total timings already include completion/error draining. This adds the public-operation boundary; it does not replace the kernel comparison or its review gate.",
        "",
        "| Public workload | Host µs, baseline → candidate | Completed µs, baseline → candidate | Completion ratio | Error stages, baseline → candidate |",
        "| --- | --- | --- | --- | --- |",
    ]
    failures = []
    for entry in report["cases"]:
        before, after = entry["baseline"], entry["candidate"]
        if not entry["parity_passed"]:
            lines.append(
                f"| {entry['case']} | — | — | PARITY FAILED | {before['status']} → {after['status']} |"
            )
            for label, row in (("baseline", before), ("candidate", after)):
                if row["status"] == "parity_failed":
                    failures.append(f"{entry['case']} ({label}): {row['failure']}")
            continue
        lines.append(
            f"| {entry['case']} | {before['median']['host_us']:.3f} → {after['median']['host_us']:.3f} | {before['median']['completion_us']:.3f} → {after['median']['completion_us']:.3f} | {entry['completion_ratio']:.3f} | {before['error_stages']} → {after['error_stages']} |"
        )
    lines.extend(["", *failures])
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("run")
    worker.add_argument(
        "--suite",
        choices=("listoffsetarray", "listarray", "indexed", "regulararray"),
        default="listoffsetarray",
    )
    worker.add_argument("--checkout", required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument("--warmup", type=int, default=20)
    worker.add_argument("--iterations", type=int, default=50)
    worker.add_argument("--repeats", type=int, default=5)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--max-regression", type=float, default=1.1)
    args = parser.parse_args()
    if args.command == "run":
        if min(args.warmup, args.iterations, args.repeats) <= 0:
            parser.error("All counts must be positive")
        run(args)
    else:
        if args.max_regression < 1:
            parser.error("--max-regression must be at least 1")
        compare(args)


if __name__ == "__main__":
    main()
