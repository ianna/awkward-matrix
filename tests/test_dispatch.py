# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks/cuda_migrations"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("workloads")
load("measure")
dispatch = load("dispatch")


def test_levels_same_gather_and_prepare_once():
    output = np.full(8, -999, dtype=np.int64)
    outer = np.array([4, 2, 1], dtype=np.int64)
    inner = np.arange(8, dtype=np.int64) * 8
    factories = []
    streams = []

    class Backend:
        def __getitem__(self, key):
            assert key == (dispatch.NAME, np.int64, np.int64, np.int64)
            return wrapper

    def wrapper(target, index, length, source):
        target[:length] = source[index[:length]]

    def transform(**kw):
        assert kw["d_in"] == 0
        streams.append(kw["stream"].__cuda_stream__())
        for i in range(kw["num_items"]):
            kw["d_out"][i] = kw["op"](i)

    def make(**kw):
        factories.append(kw)
        return transform

    calls, _ = dispatch.levels(
        Backend(),
        wrapper,
        wrapper,
        SimpleNamespace(
            CountingIterator=lambda x: x,
            make_unary_transform=make,
            unary_transform=transform,
        ),
        SimpleNamespace(ptr=123),
        output,
        outer,
        inner,
    )
    assert len(calls) == 6
    for call in calls.values():
        for _ in range(3):
            output[:] = -999
            call()
            np.testing.assert_array_equal(
                output, [32, 16, 8, -999, -999, -999, -999, -999]
            )
    assert len(factories) == 1
    assert streams == [(0, 123)] * 6


def test_profile_excludes_completion():
    events = []
    profiler = SimpleNamespace(
        enable=lambda: events.append("enable"), disable=lambda: events.append("disable")
    )
    dispatch.profile_calls(
        profiler, lambda: events.append("call"), lambda: events.append("finish"), 2
    )
    assert events == ["enable", "call", "disable", "finish"] * 2


def test_timing_excludes_final_fence_from_host(monkeypatch):
    clock = iter([1000, 4000, 10000])
    monkeypatch.setattr(dispatch.time, "perf_counter_ns", lambda: next(clock))
    events = []

    class Event:
        def record(self, stream):
            events.append("record")

        def synchronize(self):
            events.append("wait")

    cp = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, get_elapsed_time=lambda a, b: 0.005)
    )
    result = dispatch.measure_call(
        cp, None, lambda: events.append("call"), lambda: events.append("finish")
    )
    assert result == {"host_us": 3.0, "completion_us": 9.0, "stream_us": 5.0}
    assert events == ["record", "call", "record", "wait", "finish"]


def test_finish_before_awkward_context_exists():
    calls = []
    stream = SimpleNamespace(ptr=123, synchronize=lambda: calls.append("stream"))
    cuda = SimpleNamespace(
        cuda_streamptr_to_contexts={},
        synchronize_cuda=lambda s: calls.append("awkward"),
    )
    dispatch.finish_stream(cuda, stream)
    assert calls == ["stream"]
    cuda.cuda_streamptr_to_contexts[123] = object()
    dispatch.finish_stream(cuda, stream)
    assert calls == ["stream", "awkward"]
