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
    args = SimpleNamespace(timeout=5, warmups=1, repeats=3, memory_repeats=1)
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
