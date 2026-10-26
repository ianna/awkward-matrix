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
stages = load("listarray_stages")


@pytest.mark.parametrize("kind", ["advanced", "jagged"])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_stage_fixtures_reach_cpu_kernels(kind, dtype):
    pytest.importorskip("awkward")
    from awkward._backends.numpy import NumpyBackend

    rows, width = 65, 8
    arrays, parameter, name = stages.inputs(kind, rows, width, dtype)
    key = (name, *(array.dtype.type for array in arrays))
    scalars = (rows, parameter) if kind == "advanced" else (parameter, rows)
    error = NumpyBackend.instance()[key](*arrays, *scalars)
    assert not error.str
    if kind == "advanced":
        np.testing.assert_array_equal(
            arrays[0][:rows], np.arange(rows) * width + np.arange(rows) % width
        )
        np.testing.assert_array_equal(arrays[1][:rows], np.arange(rows))
        outputs = (0, 1)
        length = rows
    else:
        np.testing.assert_array_equal(
            arrays[0][: rows * width], np.tile(np.arange(width), rows)
        )
        np.testing.assert_array_equal(
            arrays[1][: rows * width], np.tile(np.arange(1, width + 1), rows)
        )
        np.testing.assert_array_equal(
            arrays[3][: rows * width], np.arange(rows * width)
        )
        outputs = (0, 1, 3)
        length = rows * width
    for i in outputs:
        np.testing.assert_array_equal(arrays[i][length:], [123, 123, 123])


@pytest.mark.parametrize("kind", ["advanced", "jagged"])
def test_stage_calls_use_compact_counts_and_fixed_workspace(kind):
    events = []
    setup = SimpleNamespace(
        chunks=3,
        length=129,
        check=object(),
        write=object(),
        discard=object(),
        reduce_operation=object(),
        initial=np.asarray(129, dtype=np.int64),
        transform=lambda *args: events.append(("transform", args)),
        reducer=lambda **kwargs: events.append(("reduce", kwargs)),
    )
    if kind == "jagged":
        setup.items = 129 * 8
    first, status, partials, storage, stream = (object() for _ in range(5))
    calls = stages.stages(setup, first, status, partials, storage, stream)
    for call in calls.values():
        call()
    assert events[0] == ("transform", ("check", setup.check, partials, 3, stream))
    reduction = events[1][1]
    assert reduction["temp_storage"] is storage
    assert reduction["d_in"] is partials and reduction["d_out"] is first
    assert reduction["num_items"] == 3
    assert reduction["stream"] is stream
    length = 129 if kind == "advanced" else 129 * 8
    assert events[2] == (
        "transform",
        ("write", setup.write, setup.discard, length, stream),
    )
