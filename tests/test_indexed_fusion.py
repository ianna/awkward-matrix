# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks/cuda_migrations"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for name in ("workloads", "measure", "dispatch", "indexed_stages"):
    load(name)
fusion = load("indexed_fusion")


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize(
    "values", [[0, 1, 2, 3, 9, 4], [9, 9, 9, 9, 9, 9], [0, 1, 2, 0, 1, 2]]
)
def test_reduction_operator_associativity_and_commutativity(dtype, values):
    outer = np.asarray(values, dtype=dtype)
    if dtype is not np.uint32:
        outer[1] = -1
    operation = fusion.error_operator(outer, len(outer), 3)
    domain = range(len(outer) + 1)
    for left, right in itertools.product(domain, repeat=2):
        assert operation(left, right) == operation(right, left)
    for a, b, c in itertools.product(domain, repeat=3):
        assert operation(operation(a, b), c) == operation(a, operation(b, c))
    # The sentinel remains the result if there are no invalid indices, including
    # negative outer indices, rather than accidentally selecting the first row.
    expected = next((i for i, x in enumerate(outer) if x >= 3), len(outer))
    actual = len(outer)
    for row in domain:
        actual = operation(actual, row)
    assert actual == expected


@pytest.mark.parametrize("outer_dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("inner_dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("error_row", [None, 0, 64, 128])
def test_arbitrary_reduction_trees_match_native_first_error(
    outer_dtype, inner_dtype, error_row
):
    pytest.importorskip("awkward")
    from awkward._backends.numpy import NumpyBackend

    length, width = 129, 3
    outer = (np.arange(length) % width).astype(outer_dtype)
    inner = np.arange(width, dtype=inner_dtype)
    if outer_dtype is not np.uint32:
        outer[::5] = -1
    if inner_dtype is not np.uint32:
        inner[0] = -7
    if error_row is not None:
        outer[error_row] = width
        outer[-1] = width + 1
    output = np.full(length + 3, 123, dtype=np.int64)
    error = NumpyBackend.instance()[
        "awkward_IndexedArray_simplify", np.int64, outer_dtype, inner_dtype
    ](output, outer, length, inner, width)
    expected = error.id if error.str else length
    op = fusion.error_operator(outer, length, width)
    rng = np.random.default_rng(4358)
    for _ in range(5):
        queue = [*rng.permutation(length), length]
        while len(queue) > 1:
            queue = [
                op(queue[i], queue[i + 1]) for i in range(0, len(queue) - 1, 2)
            ] + (queue[-1:] if len(queue) % 2 else [])
        assert queue == [expected]
        if error.str:
            assert int(outer[queue[0]]) == error.attempt
        assert np.all(output[expected:] == 123)


def test_prototype_uses_one_reduction_and_existing_gather():
    from types import SimpleNamespace

    events = []

    class State:
        def bind(self, array):
            self.array = array

        def clear(self):
            self.array = None

        def __getitem__(self, index):
            return self.array[index]

    class Base:
        def __init__(self, *args):
            self.output, self.outer, self.inner, self.status = (
                State() for _ in range(4)
            )
            self.length = args[-2]
            self.initial = np.asarray(self.length, dtype=np.int64)
            self.iterator = object()
            self.reducer = None
            self.write, self.discard = object(), object()

        def transform(self, name, operation, output, length, stream):
            assert (name, operation, output, length) == (
                "write",
                self.write,
                self.discard,
                self.length,
            )
            events.append("gather")

    def make_reducer(**kwargs):
        events.append("prepare")

        def reduce(*, temp_storage, **kwargs):
            if temp_storage is None:
                events.append("query")
                return 64
            events.append("reduce")
            result = int(kwargs["h_init"])
            for row in range(kwargs["num_items"]):
                result = kwargs["op"](result, row)
            kwargs["d_out"][0] = result

        return reduce

    cls = fusion.prototype_class(
        SimpleNamespace(_Simplify=Base), np, make_reducer, lambda op: op
    )
    instance = cls("i8", "i8", "i8", 129, 3)
    outer = np.zeros(129, dtype=np.int64)
    outer[64] = 3
    for _ in range(2):
        status = instance(np.empty(129, dtype=np.int64), outer, np.arange(3), object())
        assert status.shape == (2,)
        assert status[0] == 64
        assert all(
            state.array is None
            for state in (
                instance.output,
                instance.outer,
                instance.inner,
                instance.status,
            )
        )
    assert events == ["prepare", "query", "reduce", "gather", "reduce", "gather"]
    with pytest.raises(ValueError, match="more than 64"):
        cls("i8", "i8", "i8", 32, 3)
