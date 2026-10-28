# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Untimed dispatch trace and separate cProfile passes for public ListArray slices."""

import argparse
import cProfile
import hashlib
import json
import pstats
import statistics
import time
import traceback
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from dispatch import finish_stream
from measure import environment
from public_operations import expectation, indexed_cases, listarray_cases
from workloads import SHAPES

OPERATIONS = ("slice_jagged", "slice_spread")
INDEXED_OPERATIONS = ("field_indexed_indexed", "field_indexed_option")


def argument_metadata(value):
    """Only host metadata: never fetch GPU values/pointers or call array repr."""
    if hasattr(value, "shape") and hasattr(value, "dtype") and len(value.shape) > 0:
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device_array": hasattr(value, "__cuda_array_interface__"),
        }
    if hasattr(value, "__cuda_array_interface__"):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device_array": True,
        }
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    return {"type": type(value).__name__}


class Recorder:
    def __init__(self):
        self.events = []
        self.active = []
        self.phase = "call"
        self.iteration = 0

    def invoke(self, kind, name, implementation, call, arguments=(), key=()):
        index = len(self.events)
        event = {
            "kind": kind,
            "name": name,
            "implementation": implementation,
            "phase": self.phase,
            "iteration": self.iteration,
            "parent": self.active[-1] if self.active else None,
            "signature": [np.dtype(x).str for x in key],
            "arguments": [argument_metadata(x) for x in arguments],
            "stack": [
                {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_stack(limit=10)[:-2]
            ],
        }
        self.events.append(event)
        self.active.append(index)
        before = time.perf_counter_ns()
        try:
            return call()
        except Exception as error:
            event["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            event["host_ns"] = time.perf_counter_ns() - before
            self.active.pop()

    @contextmanager
    def install(self, backend_class, cuda=None, error_context=None):
        original = backend_class.__getitem__
        recorder = self

        class Traced:
            def __init__(self, kernel, key):
                self.kernel, self.key = kernel, key

            def __getattr__(self, name):
                return getattr(self.kernel, name)

            def __call__(self, *args):
                return recorder.invoke(
                    "kernel",
                    self.key[0],
                    type(self.kernel).__name__,
                    lambda: self.kernel(*args),
                    args,
                    self.key[1:],
                )

        def lookup(backend, key):
            return Traced(original(backend, key), key)

        with ExitStack() as patches:
            patches.enter_context(patch.object(backend_class, "__getitem__", lookup))
            if error_context is not None:
                for name in ("frozen_primary", "cuda_primary"):
                    if hasattr(error_context, name):
                        original_capture = getattr(error_context, name)

                        def capture(cls, _original=original_capture, _name=name):
                            return recorder.invoke(
                                "error_context", _name, "Awkward", _original
                            )

                        patches.enter_context(
                            patch.object(error_context, name, classmethod(capture))
                        )
            if cuda is None:
                yield
            else:
                synchronize = cuda.synchronize_cuda

                def sync(*args, **kwargs):
                    return recorder.invoke(
                        "synchronize_cuda",
                        "synchronize_cuda",
                        "Awkward",
                        lambda: synchronize(*args, **kwargs),
                    )

                with patch.object(cuda, "synchronize_cuda", sync):
                    yield


def aggregate(events, iterations):
    groups = {}
    for event in events:
        key = (event["kind"], event["name"], event["implementation"], event["phase"])
        groups.setdefault(key, []).append(event)
    return [
        {
            "kind": kind,
            "name": name,
            "implementation": implementation,
            "phase": phase,
            "calls": len(group),
            "calls_per_operation": len(group) / iterations,
            "instrumented_host_us_per_operation": sum(e["host_ns"] for e in group)
            / iterations
            / 1000,
            "median_instrumented_call_us": statistics.median(
                e["host_ns"] for e in group
            )
            / 1000,
        }
        for (kind, name, implementation, phase), group in groups.items()
    ]


def profile_calls(call, finish, count, include_finish):
    profiler = cProfile.Profile()
    for _ in range(count):
        result = None
        try:
            profiler.enable()
            try:
                result = call()
            finally:
                if include_finish:
                    finish()
        finally:
            profiler.disable()
            if not include_finish:
                finish()
        del result
    return profiler


def profile_records(profiler):
    return [
        {
            "file": file,
            "line": line,
            "function": function,
            "primitive_calls": primitive,
            "calls": calls,
            "self_seconds": own,
            "cumulative_seconds": cumulative,
        }
        for (file, line, function), (
            primitive,
            calls,
            own,
            cumulative,
            _,
        ) in pstats.Stats(profiler).stats.items()
    ]


def check(ak, result, expected):
    actual = ak.to_list(ak.to_backend(result, "cpu"))
    if actual != expected["values"]:
        raise AssertionError("Public slice differs from CPU reference")


def run(args):
    import awkward as ak
    import cupy as cp
    from awkward._backends.backend import Backend
    from awkward._connect import cuda
    from awkward._errors import ErrorContext

    suite = getattr(args, "suite", "listarray")
    cases = indexed_cases if suite == "indexed" else listarray_cases
    checkout = Path(args.checkout).resolve()
    if checkout not in Path(ak.__file__).resolve().parents:
        raise ValueError("Imported awkward does not belong to --checkout")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": environment(cp, checkout),
        "complete": False,
        "settings": {
            "suite": suite,
            "shapes": args.shapes,
            "operations": args.operation,
            "warmup": args.warmup,
            "trace_iterations": args.trace_iterations,
            "profile_iterations": args.profile_iterations,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "cases": [],
    }
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for rows, width in args.shapes:
            cpu = {name: (source, op) for name, source, op in cases(ak, rows, width)}
            gpu = {
                name: (source, op)
                for name, source, op in cases(ak, rows, width, "cuda")
            }
            for name in args.operation:
                case_id = f"{name}:{rows}:{width}"
                print(f"Diagnosing {case_id}", flush=True)
                host, host_op = cpu[name]
                expected = expectation(ak, lambda: host_op(host))
                source, operation = gpu[name]

                def call():
                    return operation(source)

                def finish():
                    finish_stream(cuda, stream)

                finish()
                for _ in range(args.warmup):
                    result = call()
                    finish()
                    check(ak, result, expected)
                    del result
                recorder = Recorder()
                for iteration in range(args.trace_iterations):
                    recorder.iteration = iteration
                    with recorder.install(Backend, cuda, ErrorContext):
                        recorder.phase = "call"
                        result = call()
                        recorder.phase = "finish"
                        finish()
                    # CPU copies/validation must not appear as slice dispatch.
                    check(ak, result, expected)
                    del result
                row = {
                    "case": case_id,
                    "events": recorder.events,
                    "dispatch": aggregate(recorder.events, args.trace_iterations),
                    "profiles": {},
                }
                for boundary in ("call_only", "completed"):
                    profiler = profile_calls(
                        call, finish, args.profile_iterations, boundary == "completed"
                    )
                    prefix = output / f"{name}-{rows}-{width}-{boundary}"
                    profiler.dump_stats(str(prefix) + ".pstats")
                    with Path(str(prefix) + ".txt").open("w") as file:
                        stats = pstats.Stats(profiler, stream=file).strip_dirs()
                        stats.sort_stats("cumulative").print_stats(80)
                        stats.sort_stats("tottime").print_stats(80)
                    row["profiles"][boundary] = profile_records(profiler)
                    result = call()
                    finish()
                    check(ak, result, expected)
                    del result
                report["cases"].append(row)
                (output / "diagnostics.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
    report["complete"] = True
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")


def normalize_profile(records):
    result = Counter()
    for record in records:
        # Preserve module paths but remove checkout/install prefixes for comparison.
        path = record["file"].replace("\\", "/")
        if "/awkward/" in path:
            path = "awkward/" + path.split("/awkward/", 1)[1]
        elif "/site-packages/" in path:
            path = path.split("/site-packages/", 1)[1]
        else:
            path = Path(path).name
        result[(path, record["function"])] += record["self_seconds"]
    return result


def compare(args):
    old, new = (
        json.loads(Path(path).read_text()) for path in (args.baseline, args.candidate)
    )
    if not old["complete"] or not new["complete"] or old["settings"] != new["settings"]:
        raise ValueError("Incomplete or mismatched diagnostic passes")
    for field in (
        "device",
        "packages",
        "python",
        "cuda_runtime",
        "cuda_driver",
        "execution_environment",
    ):
        if old["environment"][field] != new["environment"][field]:
            raise ValueError(f"Incomparable environments: {field}")
    before, after = (
        {row["case"]: row for row in report["cases"]} for report in (old, new)
    )
    expected_cases = {
        f"{name}:{rows}:{width}"
        for name in old["settings"]["operations"]
        for rows, width in old["settings"]["shapes"]
    }
    if before.keys() != expected_cases or after.keys() != expected_cases:
        raise ValueError("Incomplete case inventory")
    result = {
        "baseline_sha": old["environment"]["git_sha"],
        "candidate_sha": new["environment"]["git_sha"],
        "cases": [],
    }
    lines = [
        f"# Public {old['settings'].get('suite', 'listarray')} dispatch and host profiles",
        "",
        "Diagnostic passes only: tracing and cProfile perturb execution. These values are not benchmark latencies or pure GPU times. Kernel wrapper host durations include any internal waits. Nested events overlap; do not sum them. Profile tables use exclusive self time to avoid double-counting callers.",
        "",
    ]
    for case in before:
        a, b = before[case], after[case]
        lines.extend(
            [
                f"## {case}",
                "",
                "| Actual call | Implementation | Phase | Calls/op baseline → candidate | Instrumented host µs/op baseline → candidate |",
                "| --- | --- | --- | --- | --- |",
            ]
        )

        def dispatch_map(row):
            return {
                (x["kind"], x["name"], x["implementation"], x["phase"]): x
                for x in row["dispatch"]
            }

        ad, bd = dispatch_map(a), dispatch_map(b)
        for key in sorted(ad.keys() | bd.keys()):
            _, name, implementation, phase = key
            lines.append(
                f"| {name} | {implementation} | {phase} | {ad.get(key, {}).get('calls_per_operation', 0):g} → {bd.get(key, {}).get('calls_per_operation', 0):g} | {ad.get(key, {}).get('instrumented_host_us_per_operation', 0):.3f} → {bd.get(key, {}).get('instrumented_host_us_per_operation', 0):.3f} |"
            )
        profiles = {}
        for boundary in ("call_only", "completed"):
            ap, bp = (normalize_profile(row["profiles"][boundary]) for row in (a, b))
            count = old["settings"]["profile_iterations"]
            changes = [
                {
                    "file": key[0],
                    "function": key[1],
                    "baseline_self_us_per_op": ap[key] * 1e6 / count,
                    "candidate_self_us_per_op": bp[key] * 1e6 / count,
                    "delta_self_us_per_op": (bp[key] - ap[key]) * 1e6 / count,
                }
                for key in ap.keys() | bp.keys()
            ]
            changes.sort(key=lambda x: x["delta_self_us_per_op"], reverse=True)
            profiles[boundary] = changes
            lines.extend(
                [
                    "",
                    f"### {boundary}: largest increases in exclusive host time",
                    "",
                    "| Function | Baseline µs/op | Candidate µs/op | Δ µs/op |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for change in changes[:20]:
                lines.append(
                    f"| {change['file']}:{change['function']} | {change['baseline_self_us_per_op']:.3f} | {change['candidate_self_us_per_op']:.3f} | {change['delta_self_us_per_op']:.3f} |"
                )
        result["cases"].append(
            {
                "case": case,
                "baseline_dispatch": a["dispatch"],
                "candidate_dispatch": b["dispatch"],
                "profile_deltas": profiles,
            }
        )
    if old["settings"].get("suite") == "indexed" and all(
        name in old["settings"]["operations"] for name in INDEXED_OPERATIONS
    ):
        result["paired_profiles"] = paired_profiles(old, new, lines)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + ".json").write_text(json.dumps(result, indent=2) + "\n")
    Path(str(prefix) + ".md").write_text("\n".join(lines) + "\n")


def paired_profiles(old, new, lines):
    """Compare option minus indexed within each revision, using exclusive time."""
    pairs = []
    count = old["settings"]["profile_iterations"]
    for rows, width in old["settings"]["shapes"]:
        names = [f"{name}:{rows}:{width}" for name in INDEXED_OPERATIONS]
        for boundary in ("call_only", "completed"):
            profiles = []
            for report in (old, new):
                cases = {row["case"]: row for row in report["cases"]}
                profiles.append(
                    [
                        normalize_profile(cases[name]["profiles"][boundary])
                        for name in names
                    ]
                )
            keys = set().union(*(p.keys() for pair in profiles for p in pair))
            changes = []
            for key in keys:
                values = [p[key] * 1e6 / count for pair in profiles for p in pair]
                changes.append(
                    {
                        "file": key[0],
                        "function": key[1],
                        "baseline_indexed_us": values[0],
                        "baseline_option_us": values[1],
                        "candidate_indexed_us": values[2],
                        "candidate_option_us": values[3],
                        "candidate_option_minus_indexed_us": values[3] - values[2],
                        "excess_delta_us": values[3]
                        - values[2]
                        - values[1]
                        + values[0],
                    }
                )
            changes.sort(
                key=lambda x: x["candidate_option_minus_indexed_us"], reverse=True
            )
            pairs.append(
                {"shape": [rows, width], "boundary": boundary, "functions": changes}
            )
            lines.extend(
                [
                    "",
                    f"## Paired indexed→option versus indexed→indexed: {rows}:{width}, {boundary}",
                    "",
                    "Exclusive host time only. Status get/tolist, dispatch and context functions remain individual rows; these profiles do not measure DToH bytes.",
                    "",
                    "| Function | Baseline indexed / option µs | Candidate indexed / option µs | Candidate option − indexed µs | Change in gap µs |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            # Include focused functions even when absent from the largest increases.
            focused = [
                x
                for x in changes
                if any(
                    token in (x["file"] + x["function"])
                    for token in (
                        "_errors.py",
                        "_kernels.py",
                        "_indexedarray.py",
                        "_compute.py",
                        "'get'",
                        "'tolist'",
                        "frozen_primary",
                        "cuda_primary",
                        "__getitem__",
                    )
                )
            ]
            selected = changes[:20] + [x for x in focused if x not in changes[:20]]
            for x in selected:
                lines.append(
                    f"| {x['file']}:{x['function']} | {x['baseline_indexed_us']:.3f} / {x['baseline_option_us']:.3f} | {x['candidate_indexed_us']:.3f} / {x['candidate_option_us']:.3f} | {x['candidate_option_minus_indexed_us']:.3f} | {x['excess_delta_us']:.3f} |"
                )
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("run")
    worker.add_argument("--checkout", required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument(
        "--suite", choices=("listarray", "indexed"), default="listarray"
    )
    worker.add_argument(
        "--operation", choices=OPERATIONS + INDEXED_OPERATIONS, action="append"
    )
    worker.add_argument("--shape", action="append")
    worker.add_argument("--warmup", type=int, default=20)
    worker.add_argument("--trace-iterations", type=int, default=3)
    worker.add_argument("--profile-iterations", type=int, default=100)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        allowed = INDEXED_OPERATIONS if args.suite == "indexed" else OPERATIONS
        args.operation = args.operation or list(allowed)
        if any(name not in allowed for name in args.operation):
            parser.error("Operation does not belong to selected suite")
        args.shapes = (
            [tuple(map(int, s.split(":"))) for s in args.shape]
            if args.shape
            else SHAPES
        )
        if min(args.warmup, args.trace_iterations, args.profile_iterations) <= 0 or any(
            len(s) != 2 or min(s) <= 0 for s in args.shapes
        ):
            parser.error("Counts and ROWS:WIDTH dimensions must be positive")
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
