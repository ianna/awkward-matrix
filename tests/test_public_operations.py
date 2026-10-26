# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks/cuda_migrations"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for name in ("workloads", "measure", "dispatch"):
    load(name)
public = load("public_operations")


@pytest.mark.parametrize("error_stage", [None, "call", "finish", "both"])
def test_completed_boundary_always_drains_errors(monkeypatch, error_stage):
    clock = iter([1000, 4000, 10000])
    monkeypatch.setattr(public.time, "perf_counter_ns", lambda: next(clock))
    events = []

    class Event:
        def record(self, stream):
            events.append("record")

        def synchronize(self):
            events.append("wait")

    def call():
        events.append("call")
        if error_stage in ("call", "both"):
            raise ValueError("not regular")
        return "result"

    def finish():
        events.append("finish")
        if error_stage in ("finish", "both"):
            raise ValueError("not regular")

    cp = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, get_elapsed_time=lambda a, b: 0.005)
    )
    metrics, result, errors = public.measure_completed(cp, None, call, finish)
    assert metrics == {"host_us": 3.0, "completion_us": 9.0, "stream_us": 5.0}
    assert events == ["record", "call", "record", "finish", "wait"]
    assert result == (None if error_stage in ("call", "both") else "result")
    assert [stage for stage, _ in errors] == {
        None: [],
        "call": ["call"],
        "finish": ["finish"],
        "both": ["call", "finish"],
    }[error_stage]
    public.validate_errors(errors, {"error": "not regular" if error_stage else None})


@pytest.mark.parametrize(
    "errors,expected",
    [
        ([], "not regular"),
        ([("finish", RuntimeError("CUDA launch failed"))], "not regular"),
        ([("call", ValueError("wrong error"))], "not regular"),
        ([("finish", ValueError("not regular"))], None),
    ],
)
def test_unexpected_or_missing_error_is_not_success(errors, expected):
    with pytest.raises(AssertionError):
        public.validate_errors(errors, {"error": expected})


def reports():
    cases = [
        {
            "case": f"{name}:2:3",
            "status": "passed",
            "median": {"completion_us": 100, "host_us": 10},
            "expected_error": "not regular" if "irregular" in name else None,
            "error_stages": {"call": 0, "finish": 0},
        }
        for name in ("to_regular", "drop_none", "to_regular_irregular")
    ]
    baseline = {
        "environment": {field: "same" for field in public.ENVIRONMENT_FIELDS},
        "settings": {"shapes": [[2, 3]], "warmup": 20, "iterations": 50},
        "measurements": cases,
    }
    baseline["environment"]["git_sha"] = "baseline"
    candidate = copy.deepcopy(baseline)
    candidate["environment"]["git_sha"] = "candidate"
    return baseline, candidate


def test_public_completion_gate_does_not_hide_host_cost():
    baseline, candidate = reports()
    candidate["measurements"][0]["median"] = {"completion_us": 90, "host_us": 50}
    result = public.comparison(baseline, candidate)
    assert result["no_completion_regressions"]
    assert result["cases"][0]["completion_ratio"] == 0.9
    assert result["cases"][0]["candidate"]["median"]["host_us"] == 50
    candidate["measurements"][0]["median"]["completion_us"] = 150
    assert not public.comparison(baseline, candidate)["no_completion_regressions"]


@pytest.mark.parametrize("field", public.ENVIRONMENT_FIELDS)
def test_reject_incomparable_environment(field):
    baseline, candidate = reports()
    candidate["environment"][field] = "different"
    with pytest.raises(ValueError, match="Incomparable environments"):
        public.comparison(baseline, candidate)


@pytest.mark.parametrize(
    "mutation", ["missing", "both_missing", "settings", "error", "status"]
)
def test_reject_incomplete_or_unvalidated_reports(mutation):
    baseline, candidate = reports()
    if mutation == "missing":
        candidate["measurements"].pop()
    elif mutation == "both_missing":
        baseline["measurements"].pop()
        candidate["measurements"].pop()
    elif mutation == "settings":
        candidate["settings"]["iterations"] = 1
    elif mutation == "error":
        candidate["measurements"][0]["expected_error"] = "other"
    else:
        candidate["measurements"][0]["status"] = "failed"
    with pytest.raises(ValueError):
        public.comparison(baseline, candidate)


@pytest.mark.parametrize("rows,width", [(2, 3), (0, 8), (3, 0)])
def test_public_fixtures_match_independent_cpu_expectations(rows, width):
    ak = pytest.importorskip("awkward")
    items = list(public.cases(ak, rows, width))
    for name, source, operation in items:
        expected = public.expectation(ak, lambda: operation(source))
        if name == "to_regular_irregular":
            assert expected == {"error": "not regular"}
        else:
            values = [list(range(i * width, (i + 1) * width)) for i in range(rows)]
            if name == "drop_none":
                values = [[x for x in row if x % 3 != 0] for row in values]
            assert expected == {"values": values, "error": None}
            if name == "to_regular":
                assert operation(source).layout.is_regular
    offsets, _, index = public.fixture_arrays(rows, width)
    assert offsets.size == rows + 1
    np.testing.assert_array_equal(index[::3], -1)


def test_worker_warmup_is_excluded_and_complete_reports_compare(tmp_path, monkeypatch):
    import json

    ak = pytest.importorskip("awkward")
    calls = []

    class Stream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=Stream, Event=Event, get_elapsed_time=lambda a, b: 0.001
            )
        ),
    )
    monkeypatch.setattr(ak, "to_backend", lambda source, backend: source)
    monkeypatch.setattr(
        public, "finish_stream", lambda cuda, stream: calls.append("finish")
    )
    monkeypatch.setattr(public, "SHAPES", [(2, 3), (0, 8), (3, 0)])
    env = {field: "same" for field in public.ENVIRONMENT_FIELDS}
    env["git_sha"] = "revision"
    monkeypatch.setattr(public, "environment", lambda cp, checkout: env)
    output = tmp_path / "report.json"
    public.run(
        SimpleNamespace(
            checkout=Path(ak.__file__).resolve().parents[2],
            output=output,
            warmup=2,
            iterations=2,
            repeats=2,
        )
    )
    report = json.loads(output.read_text())
    assert len(report["measurements"]) == 7
    assert len(calls) == 7 * (1 + 2 * 2 + 2 * 2)
    for case in report["measurements"]:
        assert case["status"] == "passed"
        assert len(case["samples"]) == 2
        if case["expected_error"]:
            # Only measured iterations contribute; warmup is excluded.
            assert case["error_stages"] == {"call": 4, "finish": 0}
        else:
            assert case["error_stages"] == {"call": 0, "finish": 0}
    assert public.comparison(report, report)["no_completion_regressions"]
    public.compare(
        SimpleNamespace(
            baseline=output,
            candidate=output,
            output=tmp_path / "comparison",
            max_regression=1.1,
        )
    )
    assert (tmp_path / "comparison.json").exists()
    assert (
        "Completed public CUDA operations" in (tmp_path / "comparison.md").read_text()
    )


def test_public_fixtures_reach_the_migrated_kernels(monkeypatch):
    ak = pytest.importorskip("awkward")
    from awkward._backends.numpy import NumpyBackend

    seen = set()
    original = NumpyBackend.__getitem__

    def lookup(self, key):
        seen.add(key[0])
        return original(self, key)

    monkeypatch.setattr(NumpyBackend, "__getitem__", lookup)
    for _, source, operation in public.cases(ak, 3, 4):
        public.expectation(ak, lambda: operation(source))
    assert {
        "awkward_ListOffsetArray_drop_none_indexes",
        "awkward_ListOffsetArray_toRegularArray",
    } <= seen
