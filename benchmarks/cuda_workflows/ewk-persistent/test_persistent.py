import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from compare import compare
from persistent_worker import PersistentPool


def test_one_spawned_process_runs_multiple_cloudpickled_tasks(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSISTENT_RESULTS", str(tmp_path))
    monkeypatch.setenv("EWKCOFFEA_BACKEND", "cpu")
    bias = 5

    def task(item):
        print("CHUNK_TIMING phase=example seconds=0.01", flush=True)
        return os.getpid(), item.entrystart + bias

    with PersistentPool() as pool:
        futures = [
            pool.submit(
                task,
                SimpleNamespace(
                    filename="test.root",
                    dataset="test",
                    entrystart=i * 10,
                    entrystop=(i + 1) * 10,
                ),
            )
            for i in range(4)
        ]
        outputs = [future.result(timeout=30) for future in futures]
    assert len({pid for pid, _ in outputs}) == 1
    assert outputs[0][0] != os.getpid()
    assert [value for _, value in outputs] == [5, 15, 25, 35]
    events = [
        json.loads(line)
        for path in tmp_path.glob("worker-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert sum(e["event"] == "chunk_end" for e in events) == 4
    assert sum(e["event"] == "upstream_phase" for e in events) == 4


class Histogram:
    axes = ("x",)

    def __init__(self, value=1, variance=2):
        self.value_array = np.asarray([0, value, 0])
        self.variance_array = np.asarray([0, variance, 0])

    def values(self, flow):
        assert flow
        return self.value_array

    def variances(self, flow):
        assert flow
        return self.variance_array


def test_histogram_values_and_variances_are_checked():
    compare({"hist": Histogram()}, {"hist": Histogram()})
    with pytest.raises(AssertionError):
        compare({"hist": Histogram()}, {"hist": Histogram(value=3)})
    with pytest.raises(AssertionError):
        compare({"hist": Histogram()}, {"hist": Histogram(variance=3)})


@pytest.mark.parametrize("growth", [True, False])
def test_category_order_aligns_values_variances_and_flow(growth):
    import hist

    left = hist.Hist(
        hist.axis.StrCategory(["a", "b"], growth=growth, name="process"),
        hist.axis.Regular(2, 0, 2, name="x"),
        storage=hist.storage.Weight(),
    )
    right = hist.Hist(
        hist.axis.StrCategory(["b", "a"], growth=growth, name="process"),
        hist.axis.Regular(2, 0, 2, name="x"),
        storage=hist.storage.Weight(),
    )
    for h in (left, right):
        h.fill(process=["a", "b", "a", "b"], x=[-1, 0.5, 1.5, 3], weight=[1, 2, 3, 4])
        if not growth:
            h.fill(process=["unknown"], x=[3], weight=[5])
    compare(left, right)
    right.fill(process=["a"], x=[-1], weight=[2])
    with pytest.raises(AssertionError):
        compare(left, right)


def test_category_set_difference_is_rejected():
    import hist

    left = hist.Hist(hist.axis.StrCategory(["a", "b"], growth=True, name="process"))
    right = hist.Hist(hist.axis.StrCategory(["a", "c"], growth=True, name="process"))
    with pytest.raises(AssertionError, match="category sets differ"):
        compare(left, right)


def test_numeric_binning_difference_is_rejected():
    import hist

    left = hist.Hist(hist.axis.Regular(2, 0, 2, name="x"))
    right = hist.Hist(hist.axis.Regular(2, 0, 3, name="x"))
    with pytest.raises(AssertionError, match="axis 0 differs"):
        compare(left, right)


def test_reordered_category_variance_difference_is_rejected():
    import hist

    a = hist.Hist(
        hist.axis.StrCategory(["a", "b"], name="p"), storage=hist.storage.Weight()
    )
    b = hist.Hist(
        hist.axis.StrCategory(["b", "a"], name="p"), storage=hist.storage.Weight()
    )
    a.fill(p=["a"], weight=[2])
    b.fill(p=["a", "a"], weight=[1, 1])
    with pytest.raises(AssertionError, match="variance"):
        compare(a, b)


def test_phase_report_excludes_first_and_separates_nested_timers(tmp_path):
    from phases import read_backend

    chunks = [
        dict(pid=12, start=i * 10, stop=(i + 1) * 10, at=i, seconds=v)
        for i, v in enumerate([100, 8, 10])
    ]
    (tmp_path / "summary.json").write_text(json.dumps({"chunks": chunks}))
    events = []
    for i, seconds in enumerate([99, 2, 4]):
        for kind, name in [("phase", "objects"), ("subphase", "triplets")]:
            events.append(
                dict(
                    event="upstream_phase",
                    pid=12,
                    chunk=[i * 10, (i + 1) * 10],
                    line=f"CHUNK_TIMING {kind}={name} seconds={seconds}",
                )
            )
    (tmp_path / "worker-12.jsonl").write_text("\n".join(map(json.dumps, events)))
    result = read_backend(tmp_path)
    assert result["task_median_s"] == 9
    assert result["phases"]["phase:objects"]["median_s"] == 3
    assert result["phases"]["subphase:triplets"]["median_s"] == 3
    assert result["phases"]["phase:objects"]["observed_chunks"] == 2


def test_phase_report_rejects_duplicate_records(tmp_path):
    from phases import read_backend

    chunks = [dict(pid=12, start=i, stop=i + 1, at=i, seconds=1) for i in range(3)]
    (tmp_path / "summary.json").write_text(json.dumps({"chunks": chunks}))
    event = dict(
        event="upstream_phase",
        pid=12,
        chunk=[1, 2],
        line="CHUNK_TIMING phase=objects seconds=1",
    )
    (tmp_path / "worker-12.jsonl").write_text(
        json.dumps(event) + "\n" + json.dumps(event)
    )
    with pytest.raises(AssertionError, match="Repeated phase"):
        read_backend(tmp_path)


def test_scalar_instrumentation_preserves_statements_and_scope():
    from scalar_diagnostics import LABELS, instrument

    source = "class AnalysisProcessor:\n    def process(self):\n"
    source += "".join(f"        {label} = {i}\n" for i, label in enumerate(LABELS))
    source += "        return [" + ",".join(LABELS) + "]\n"
    namespaces = [{}, {}]
    exec(source, namespaces[0])
    exec(instrument(source), namespaces[1])
    assert (
        namespaces[0]["AnalysisProcessor"]().process()
        == namespaces[1]["AnalysisProcessor"]().process()
    )


def test_scalar_profile_only_selects_second_chunk(monkeypatch, tmp_path):
    import scalar_diagnostics as diagnostic
    from scalar_diagnostics import LABELS, instrument

    monkeypatch.setattr(diagnostic, "_ordinal", 0)
    monkeypatch.setenv("PERSISTENT_RESULTS", str(tmp_path))
    monkeypatch.setenv("EWKCOFFEA_BACKEND", "cpu")
    monkeypatch.setenv("SCALAR_DIAGNOSTICS", "host")
    source = "class AnalysisProcessor:\n    def process(self):\n"
    source += "".join(f"        {label} = {i}\n" for i, label in enumerate(LABELS))
    source += "        return 42\n"
    namespace = {}
    exec(instrument(source), namespace)
    for i in range(3):
        diagnostic.chunk_start(dict(start=i * 10, stop=(i + 1) * 10))
        assert namespace["AnalysisProcessor"]().process() == 42
        diagnostic.chunk_end()
        if i == 0:
            assert not (tmp_path / "scalar.pstats").exists()
    report = json.loads((tmp_path / "scalar-chunk.json").read_text())
    assert report["ordinal"] == 2 and report["start"] == 10
    rows = json.loads((tmp_path / "scalar-statements.json").read_text())
    assert [r["statement"] for r in rows] == [*LABELS, "__total__"]
    assert (tmp_path / "scalar-focus.txt").exists()
