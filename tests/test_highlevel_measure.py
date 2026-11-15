from types import SimpleNamespace

from benchmarks.highlevel import measure
from benchmarks.highlevel.cases import Case


def test_timing_rebinds_mutating_case_for_each_trial(monkeypatch):
    executed = []

    def prepare(fixtures, target):
        assert fixtures.size == 8
        assert target is len
        pending = [1]

        def run():
            executed.append(pending.pop())

        return run

    monkeypatch.setattr(measure, "CASES", {"ak.test": Case(prepare)})
    monkeypatch.setattr(measure, "inventory", lambda _: {"ak.test": len})
    result = measure.worker(None, "ak.test", 8, "timing", warmups=2, repeats=3)
    assert result["status"] == "measured"
    assert len(result["samples_ns"]) == 3
    assert executed == [1] * 5


def test_failed_memory_worker_preserves_timing(monkeypatch):
    def invoke(*args, **kwargs):
        if args[3] == "timing":
            return {"status": "measured", "samples_ns": [10, 20, 30]}
        return {"status": "failed", "detail": "worker timeout"}

    monkeypatch.setattr(measure, "invoke", invoke)
    args = SimpleNamespace(
        timeout=5, warmups=1, repeats=3, timing_batches=1, memory_repeats=1
    )
    row = measure.measure_case("python", "ak.sum", 8, args)
    assert row["status"] == "partial"
    assert row["timing"]["median_ns"] == 20
    assert row["memory"][0]["detail"] == "worker timeout"


def test_cli_uses_three_fixture_sizes_by_default(monkeypatch, tmp_path):
    sizes = []

    def collect(args):
        sizes.append(args.size)
        return 0

    monkeypatch.setattr(measure, "measure", collect)
    arguments = [
        "--python",
        "one",
        "--python",
        "two",
        "--output",
        str(tmp_path / "out"),
    ]
    assert measure.main(arguments) == 0
    assert measure.main([*arguments, "--size", "128"]) == 0
    assert sizes == [[32, 1024, 32768], [128]]


def test_main_keeps_its_identity_when_package_version_matches(monkeypatch, tmp_path):
    import json

    environment = {"awkward": "2.15.0", "awkward_cpp": "58", "apis": ["ak.sum"]}
    monkeypatch.setattr(measure, "invoke", lambda *args, **kwargs: environment.copy())
    monkeypatch.setattr(
        measure,
        "measure_case",
        lambda python, name, size, _: {"api": name, "size": size, "status": "measured"},
    )
    output = tmp_path / "measurements.json"
    args = SimpleNamespace(
        python=["release-python", "main-python"],
        main_commit="a" * 40,
        baseline=None,
        case=["ak.sum"],
        size=[32],
        warmups=1,
        repeats=1,
        timing_batches=1,
        memory_repeats=1,
        timeout=5,
        output=output,
    )
    assert measure.measure(args) == 0
    report = json.loads(output.read_text())
    release, main = report["releases"]
    assert release["awkward"] == main["awkward_version"] == "2.15.0"
    assert main["awkward"] == "main"
    assert main["source_commit"] == "a" * 40
    assert report["matching_environments"]


def test_measure_case_keeps_independent_batches(monkeypatch):
    batches = iter([[10, 11], [100, 101], [12, 13]])
    modes = []

    def invoke(*args, **kwargs):
        mode = args[3]
        modes.append(mode)
        if mode == "timing":
            return {"status": "measured", "samples_ns": next(batches)}
        return {"status": "measured", "sample": {"delta_bytes": 100}}

    monkeypatch.setattr(measure, "invoke", invoke)
    args = SimpleNamespace(
        timeout=5, warmups=1, repeats=2, timing_batches=3, memory_repeats=5
    )
    row = measure.measure_case("python", "ak.sum", 32, args)
    assert modes == ["timing"] * 3 + ["memory"] * 5
    assert row["timing"]["median_ns"] == 12.5
    assert len(row["timing"]["batches"]) == 3
    assert len(row["memory"]) == 5
