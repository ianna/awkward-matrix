# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
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


for name in ("workloads", "measure", "dispatch", "public_operations"):
    load(name)
diag = load("public_listarray_diagnostics")


def test_trace_metadata_does_not_read_device_values():
    class DeviceArray:
        shape = (32,)
        dtype = np.dtype("int64")
        __cuda_array_interface__ = {}

        def get(self):
            raise AssertionError("Device read")

        def __repr__(self):
            raise AssertionError("Device repr")

    assert diag.argument_metadata(DeviceArray()) == {
        "shape": [32],
        "dtype": "int64",
        "device_array": True,
    }
    assert diag.argument_metadata(np.int64(4)) == 4


def test_actual_invocations_not_lookups_and_hooks_restore():
    class Kernel:
        special = "preserved"

        def __call__(self, *args):
            return 7

    class Backend:
        def __getitem__(self, key):
            return Kernel()

    original = Backend.__getitem__
    calls = []
    cuda = SimpleNamespace(synchronize_cuda=lambda *args: calls.append("finish"))
    sync = cuda.synchronize_cuda
    recorder = diag.Recorder()
    with recorder.install(Backend, cuda):
        backend = Backend()
        unused = backend[("unused", np.int64)]
        assert unused.special == "preserved"
        for _ in range(2):
            assert backend[("actual", np.int64)](np.arange(3), 3) == 7
        recorder.phase = "finish"
        cuda.synchronize_cuda(None)
    assert Backend.__getitem__ is original and cuda.synchronize_cuda is sync
    assert [e["name"] for e in recorder.events] == [
        "actual",
        "actual",
        "synchronize_cuda",
    ]
    summary = diag.aggregate(recorder.events, 2)
    assert summary[0]["calls_per_operation"] == 1
    assert summary[1]["phase"] == "finish"
    assert calls == ["finish"]


def test_trace_nested_events_and_exceptions_restore_hooks():
    recorder = diag.Recorder()

    def outer():
        return recorder.invoke("kernel", "inner", "Compute", lambda: 42)

    assert recorder.invoke("kernel", "outer", "Raw", outer) == 42
    assert recorder.events[1]["parent"] == 0

    def fail():
        raise RuntimeError("test")

    with pytest.raises(RuntimeError):
        recorder.invoke("kernel", "error", "Raw", fail)
    assert recorder.active == []
    assert recorder.events[-1]["error"]["type"] == "RuntimeError"


@pytest.mark.parametrize("include_finish", [False, True])
def test_profile_completion_boundary_and_dispatch_are_separate(include_finish):
    events = []

    def public_call():
        events.append("call")

    def drain():
        events.append("finish")

    profiler = diag.profile_calls(public_call, drain, 3, include_finish)
    assert events == ["call", "finish"] * 3
    names = {r["function"] for r in diag.profile_records(profiler)}
    assert "public_call" in names
    assert ("drain" in names) == include_finish


def reports():
    settings = {
        "operations": ["slice_jagged"],
        "shapes": [[2, 3]],
        "profile_iterations": 10,
    }
    env = {
        field: "same"
        for field in (
            "device",
            "packages",
            "python",
            "cuda_runtime",
            "cuda_driver",
            "execution_environment",
        )
    }
    env["git_sha"] = "revision"
    profile = [
        {
            "file": "/checkout/src/awkward/contents/listarray.py",
            "function": "slice",
            "self_seconds": 0.001,
        }
    ]
    return {
        "settings": settings,
        "environment": env,
        "complete": True,
        "cases": [
            {
                "case": "slice_jagged:2:3",
                "dispatch": [],
                "profiles": {"call_only": profile, "completed": profile},
            }
        ],
    }


@pytest.mark.parametrize(
    "invalid", [None, "missing", "settings", "environment", "incomplete"]
)
def test_comparison_is_complete_and_uses_exclusive_times(tmp_path, invalid):
    baseline, candidate = reports(), reports()
    if invalid == "missing":
        candidate["cases"] = []
    elif invalid == "settings":
        candidate["settings"]["profile_iterations"] = 9
    elif invalid == "environment":
        candidate["environment"]["device"] = "other"
    elif invalid == "incomplete":
        candidate["complete"] = False
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(baseline))
    b.write_text(json.dumps(candidate))
    args = SimpleNamespace(baseline=a, candidate=b, output=tmp_path / "comparison")
    if invalid:
        with pytest.raises(ValueError):
            diag.compare(args)
    else:
        diag.compare(args)
        result = json.loads((tmp_path / "comparison.json").read_text())
        row = result["cases"][0]["profile_deltas"]["call_only"][0]
        assert row["baseline_self_us_per_op"] == 100
        assert row["delta_self_us_per_op"] == 0
        assert "not benchmark latencies" in (tmp_path / "comparison.md").read_text()


def test_cpu_dispatch_reaches_additional_kernels():
    ak = pytest.importorskip("awkward")
    from awkward._backends.backend import Backend

    for name, source, operation in diag.listarray_cases(ak, 3, 4):
        if name not in diag.OPERATIONS:
            continue
        recorder = diag.Recorder()
        with recorder.install(Backend):
            result = operation(source)
        assert ak.to_list(result)
        kernels = {event["name"] for event in recorder.events}
        assert len(kernels) > 1
        assert (
            "awkward_ListArray_getitem_jagged_expand"
            if name == "slice_jagged"
            else "awkward_ListArray_getitem_next_range_spreadadvanced"
        ) in kernels


@pytest.mark.parametrize("include_finish", [False, True])
def test_profile_drains_even_if_public_call_raises(include_finish):
    drained = []

    def fail():
        raise ValueError("unexpected failure")

    with pytest.raises(ValueError, match="unexpected failure"):
        diag.profile_calls(fail, lambda: drained.append(True), 1, include_finish)
    assert drained == [True]


@pytest.mark.parametrize("suite", ["listarray", "indexed", "regulararray"])
def test_worker_roundtrip_without_gpu(tmp_path, monkeypatch, suite):
    ak = pytest.importorskip("awkward")

    class Stream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(Stream=Stream))
    )
    monkeypatch.setattr(ak, "to_backend", lambda source, backend, **kwargs: source)
    monkeypatch.setattr(diag, "finish_stream", lambda *args: None)
    env = {
        field: "same"
        for field in (
            "device",
            "packages",
            "python",
            "cuda_runtime",
            "cuda_driver",
            "execution_environment",
        )
    }
    env["git_sha"] = "revision"
    monkeypatch.setattr(diag, "environment", lambda *args: env)
    diag.run(
        SimpleNamespace(
            checkout=Path(ak.__file__).resolve().parents[2],
            output=tmp_path,
            shapes=[(2, 3)],
            suite=suite,
            operation=list(
                {
                    "indexed": diag.INDEXED_OPERATIONS,
                    "regulararray": diag.REGULAR_OPERATIONS,
                }.get(suite, diag.OPERATIONS)
            ),
            warmup=1,
            trace_iterations=2,
            profile_iterations=2,
        )
    )
    path = tmp_path / "diagnostics.json"
    data = json.loads(path.read_text())
    assert data["complete"] and len(data["cases"]) == 2
    assert len(list(tmp_path.glob("*.pstats"))) == 4
    assert all(row["events"] and row["profiles"]["completed"] for row in data["cases"])
    diag.compare(
        SimpleNamespace(baseline=path, candidate=path, output=tmp_path / "comparison")
    )
    assert (tmp_path / "comparison.md").exists()


def test_context_capture_is_traced_and_restored():
    class Context:
        @classmethod
        def frozen_primary(cls):
            return "context"

    class Backend:
        def __getitem__(self, key):
            raise AssertionError("No kernel requested")

    original = Context.__dict__["frozen_primary"]
    recorder = diag.Recorder()
    with recorder.install(Backend, error_context=Context):
        assert Context.frozen_primary() == "context"
    assert Context.__dict__["frozen_primary"] is original
    assert recorder.events[0]["kind"] == "error_context"
    assert recorder.events[0]["name"] == "frozen_primary"


def test_indexed_paired_profile_subtracts_common_work():
    def report(indexed, option):
        return {
            "settings": {"profile_iterations": 10, "shapes": [[4096, 8]]},
            "cases": [
                {
                    "case": f"{name}:4096:8",
                    "profiles": {
                        boundary: [
                            {
                                "file": "/tmp/awkward/_errors.py",
                                "function": "freeze",
                                "self_seconds": seconds,
                            }
                        ]
                        for boundary in ("call_only", "completed")
                    },
                }
                for name, seconds in zip(diag.INDEXED_OPERATIONS, (indexed, option))
            ],
        }

    lines = []
    paired = diag.paired_profiles(report(0.001, 0.002), report(0.003, 0.005), lines)
    row = paired[0]["functions"][0]
    assert row["candidate_option_minus_indexed_us"] == pytest.approx(200)
    assert row["excess_delta_us"] == pytest.approx(100)
    assert any("_errors.py:freeze" in line for line in lines)
