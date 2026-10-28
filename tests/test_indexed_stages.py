# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

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
indexed = load("indexed_stages")


@pytest.mark.parametrize("rows,width", [(65, 8), (4096, 8), (129, 4096)])
@pytest.mark.parametrize("invalid", [False, True])
def test_inputs_match_cpu_and_preserve_capacity(rows, width, invalid):
    pytest.importorskip("awkward")
    from awkward._backends.numpy import NumpyBackend

    output, outer, inner = indexed.inputs(rows, width, invalid)
    error = NumpyBackend.instance()[
        "awkward_IndexedArray_simplify", np.int64, np.int64, np.int64
    ](output, outer, rows, inner, width)
    boundary = rows // 2 if invalid else rows
    assert bool(error.str) == invalid
    if invalid:
        assert (error.id, error.attempt) == (boundary, width)
    expected = [(-1 if outer[i] < 0 else int(inner[outer[i]])) for i in range(boundary)]
    np.testing.assert_array_equal(output[:boundary], expected)
    np.testing.assert_array_equal(output[boundary:], np.full(rows + 3 - boundary, 123))


@pytest.mark.parametrize("compact", [False, True])
def test_stage_boundaries_fixed_buffers_and_stream(compact):
    events = []

    class State:
        def bind(self, array):
            events.append(("bind", array))

        def clear(self):
            events.append(("clear",))

    class Status:
        def __getitem__(self, item):
            return "first"

        def get(self):
            events.append(("get",))
            return np.array([129, np.iinfo(np.int64).max])

    def empty(length, dtype):
        events.append(("allocate", length, dtype))
        return np.empty(length, dtype=dtype)

    cp = SimpleNamespace(empty=empty, int64=np.int64, uint8=np.uint8)
    setup = SimpleNamespace(
        output=State(),
        outer=State(),
        inner=State(),
        status=State(),
        chunks=3,
        length=129,
        check="check_op",
        write="write_op",
        discard="discard",
        reduce_operation="minimum",
        initial=np.array([129]),
        transform=lambda *args: events.append(("transform", args)),
        reducer=lambda **kwargs: events.append(("reduce", kwargs)),
    )
    if compact:
        setup.reduce_storage_bytes = 64
    arrays = tuple(object() for _ in range(3))
    status, partials, stream = Status(), object(), object()
    storage = np.empty(64, dtype=np.uint8)
    calls, bind, clear = indexed.stages(
        cp, setup, arrays, status, partials, storage, stream
    )
    assert list(calls) == [
        "validation",
        "reduction",
        "gather_and_attempt",
        "workspace_query",
        "allocation_and_release",
        "bind_and_clear",
        "status_transfer",
    ]
    bind()
    assert [e[1] for e in events] == [*arrays, status]
    events.clear()
    calls["validation"]()
    assert events == [("transform", ("check", "check_op", partials, 3, stream))]
    events.clear()
    calls["reduction"]()
    kwargs = events[0][1]
    assert kwargs["temp_storage"] is storage
    assert kwargs["d_in"] is partials and kwargs["d_out"] == "first"
    assert kwargs["stream"] is stream and kwargs["num_items"] == 3
    events.clear()
    calls["gather_and_attempt"]()
    assert events == [("transform", ("write", "write_op", "discard", 129, stream))]
    events.clear()
    calls["workspace_query"]()
    assert events[0][1]["temp_storage"] is None
    assert events[0][1]["stream"] is stream
    events.clear()
    calls["allocation_and_release"]()
    assert events == (
        [("allocate", 5, np.int64), ("allocate", 64, np.uint8)]
        if compact
        else [
            ("allocate", 2, np.int64),
            ("allocate", 3, np.int64),
            ("allocate", 64, np.uint8),
        ]
    )
    events.clear()
    calls["bind_and_clear"]()
    assert [e[0] for e in events] == ["bind"] * 4 + ["clear"] * 4
    events.clear()
    calls["status_transfer"]()
    assert events == [("get",)]
    clear()
    assert events[-4:] == [("clear",)] * 4


@pytest.mark.parametrize(
    "arguments", [["--rows", "64"], ["--width", "0"], ["--iterations", "0"]]
)
def test_cli_rejects_small_or_invalid_stage_workloads(monkeypatch, arguments):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "indexed_stages",
            "--checkout",
            "/tmp/checkout",
            "--output",
            "/tmp/out",
            *arguments,
        ],
    )
    with pytest.raises(SystemExit) as error:
        indexed.main()
    assert error.value.code == 2


def test_fused_stages_need_no_chunk_buffer():
    events = []
    state = SimpleNamespace(bind=lambda array: None, clear=lambda: None)
    setup = SimpleNamespace(
        output=state,
        outer=state,
        inner=state,
        status=state,
        combine=object(),
        iterator=object(),
        length=129,
        reduce_operation=object(),
        initial=np.array([129]),
        reducer=lambda **kwargs: events.append(kwargs),
        transform=lambda *args: events.append(args),
        write=object(),
        discard=object(),
        reduce_storage_bytes=64,
    )
    allocations = []

    def empty(length, dtype):
        allocations.append((length, dtype))
        return np.empty(length, dtype=dtype)

    cp = SimpleNamespace(empty=empty, int64=np.int64, uint8=np.uint8)
    calls, _, _ = indexed.stages(
        cp,
        setup,
        (object(), object(), object()),
        np.array([129, 0]),
        None,
        np.empty(64, dtype=np.uint8),
        object(),
    )
    assert "validation" not in calls and "reduction" not in calls
    calls["validation_and_reduction"]()
    assert events[0]["d_in"] is setup.iterator
    assert events[0]["num_items"] == 129
    calls["allocation_and_release"]()
    assert allocations == [(2, np.int64), (64, np.uint8)]
