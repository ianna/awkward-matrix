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
chunks = load("regular_chunks")


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("rows", [2, 33, 4096])
def test_sweep_fixtures_have_the_intended_first_error(dtype, rows):
    pytest.importorskip("awkward")
    from awkward._backends.numpy import NumpyBackend

    for pattern, code in (("regular", 0), ("early_error", 2), ("late_error", rows)):
        offsets = chunks.offsets_for(rows, dtype, pattern)
        output = np.full(3, 123, dtype=np.int64)
        error = NumpyBackend.instance()[
            "awkward_ListOffsetArray_toRegularArray", np.int64, dtype
        ](output, offsets, len(offsets))
        assert chunks.expected_status(error) == code
        np.testing.assert_array_equal(output, [4, 123, 123])


def test_sweep_report_keeps_each_reference_and_scratch_count(tmp_path):
    entries = []
    for pattern in ("regular", "late_error"):
        for chunk, latency in ((1, 100), (32, 50)):
            entries.append(
                {
                    "rows": 4096,
                    "dtype": "int64",
                    "pattern": pattern,
                    "chunk_size": chunk,
                    "error_buffer_bytes": 8 * ((4096 + chunk - 1) // chunk),
                    "pool_temporary_peak_bytes": 1024,
                    "median": {"completion_us": latency, "host_us": latency - 10},
                }
            )
    report = {"environment": {"git_sha": "revision"}, "cases": entries}
    chunks.write_report(tmp_path, report)
    assert json.loads((tmp_path / "regular-chunks.json").read_text()) == report
    text = (tmp_path / "regular-chunks.md").read_text()
    assert "0.500" in text
    assert "32768" in text and "1024" in text
    assert "No production chunk size is selected" in text


def test_sweep_worker_forwards_chunks_and_excludes_warmup(tmp_path, monkeypatch):
    ak = pytest.importorskip("awkward")
    from awkward._backends.numpy import NumpyBackend

    calls = []

    class Array(np.ndarray):
        def get(self, **kwargs):
            return np.asarray(self)

    class Stream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def synchronize(self):
            pass

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

    class Hook:
        peak_live_bytes = 512
        requested_bytes = 8
        requests = 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def regular_status(size, source, length, stream, *, chunk_size):
        calls.append(chunk_size)
        error = NumpyBackend.instance()[
            "awkward_ListOffsetArray_toRegularArray", np.int64, source.dtype.type
        ](size, source, length)
        return np.asarray([chunks.expected_status(error)], dtype=np.int64).view(Array)

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            asarray=lambda x: np.asarray(x).view(Array),
            full=lambda *a, **kw: np.full(*a, **kw).view(Array),
            int64=np.int64,
            cuda=SimpleNamespace(
                Stream=Stream, Event=Event, get_elapsed_time=lambda a, b: 0.001
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "awkward._connect.cuda._listoffsetarray",
        SimpleNamespace(_Stream=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "awkward._connect.cuda._prepared_gather",
        SimpleNamespace(regular_status=regular_status),
    )
    monkeypatch.setattr(
        chunks, "environment", lambda cp, checkout: {"git_sha": "revision"}
    )
    monkeypatch.setattr(chunks, "allocation_hook", lambda cp: Hook())
    chunks.run(
        SimpleNamespace(
            checkout=Path(ak.__file__).resolve().parents[2],
            output=tmp_path,
            rows=[4],
            chunks=[1, 8],
            warmup=1,
            iterations=1,
            repeats=2,
        )
    )
    report = json.loads((tmp_path / "regular-chunks.json").read_text())
    assert len(report["cases"]) == 18
    assert calls.count(1) == calls.count(8) == 9 * 4
    for entry in report["cases"]:
        assert entry["status"] == "passed"
        assert len(entry["samples"]) == 2
        assert entry["error_buffer_bytes"] == (32 if entry["chunk_size"] == 1 else 0)
        assert entry["pool_temporary_peak_bytes"] == 512
