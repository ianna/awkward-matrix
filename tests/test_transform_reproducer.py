# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PATH = (
    Path(__file__).resolve().parents[1]
    / "reproducers/cuda_compute_transform_dispatch.py"
)
spec = importlib.util.spec_from_file_location("transform_reproducer", PATH)
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


def test_four_modes_share_buffers_iterator_and_retained_adapter():
    output = np.full(3, -1, dtype=np.int64)
    outer = np.asarray([4, 2, 1], dtype=np.int64)
    inner = np.arange(8, dtype=np.int64) * 8
    factories, adapters, invocations = [], [], []
    stream = object()

    class Adapter:
        def __init__(self, function):
            self.function = function

        def __call__(self, i):
            return self.function(i)

    def make_adapter(function):
        adapter = Adapter(function)
        adapters.append(adapter)
        return adapter

    def transform(**kwargs):
        invocations.append(kwargs)
        for i in range(kwargs["num_items"]):
            kwargs["d_out"][i] = kwargs["op"](i)

    def prepare(**kwargs):
        factories.append(kwargs)
        return transform

    compute = SimpleNamespace(
        CountingIterator=lambda start: object(),
        make_unary_transform=prepare,
        unary_transform=transform,
    )
    operations = repro.operations(
        compute, make_adapter, outer, inner, output, stream, np
    )
    assert set(operations) == {
        "convenience_callable",
        "prepared_callable",
        "convenience_adapter",
        "prepared_adapter",
    }
    for call in operations.values():
        output[:] = -1
        call()
        np.testing.assert_array_equal(output, [32, 16, 8])
    assert len(factories) == len(adapters) == 1
    assert len({id(x["d_in"]) for x in invocations}) == 1
    assert all(x["d_out"] is output and x["stream"] is stream for x in invocations)
    assert invocations[0]["op"] is invocations[1]["op"]
    assert invocations[2]["op"] is invocations[3]["op"] is adapters[0]


def test_host_timing_excludes_waits(monkeypatch):
    times = iter([1000, 4000, 10000])
    monkeypatch.setattr(repro.time, "perf_counter_ns", lambda: next(times))
    events = []

    class Event:
        def record(self, stream):
            events.append("record")

        def synchronize(self):
            events.append("completed")

    cp = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, get_elapsed_time=lambda a, b: 0.005)
    )
    stream = SimpleNamespace(synchronize=lambda: events.append("ready"))
    result = repro.measure(cp, stream, lambda: events.append("call"), 1)
    assert result == {"host_us": 3, "completion_us": 9, "stream_us": 5}
    assert events == ["ready", "record", "call", "record", "completed"]
