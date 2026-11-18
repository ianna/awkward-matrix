import ast
from pathlib import Path

import awkward as ak
import pytest


@pytest.fixture
def summarize():
    path = Path(__file__).with_name("cuda_argument.py")
    namespace = {}
    exec(compile(ast.parse(path.read_text()), str(path), "exec"), namespace)
    return namespace["_cuda_argument"]


def test_no_recursive_type_form_behavior_or_value_formatting(summarize, monkeypatch):
    array = ak.Array([{f"field{i}": [1, 2] for i in range(1000)}])
    record = array[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic capture must use outer metadata only")

    for cls in (ak.Array, ak.Record):
        monkeypatch.setattr(cls, "type", property(forbidden))
        monkeypatch.setattr(cls, "__repr__", forbidden)
        monkeypatch.setattr(cls, "__str__", forbidden)
    monkeypatch.setattr(ak.contents.RecordArray, "form", property(forbidden))
    assert summarize(array) == "<Array layout=RecordArray length=1>"
    assert summarize(record) == "<Record layout=RecordArray>"
    assert summarize({"pt": array}) == "<dict>"
    assert summarize([array]) == "[<Array layout=RecordArray length=1>]"


def test_summary_does_not_retain_buffers(summarize):
    import gc
    import weakref

    array = ak.Array([[1], []])
    ref = weakref.ref(array)
    result = summarize(array)
    del array
    gc.collect()
    assert ref() is None
    assert result == "<Array layout=ListOffsetArray length=2>"
