import importlib.util
import pstats
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "startup_profile", Path(__file__).with_name("startup_profile.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("mode", ["plain", "profile", "trace"])
def test_measure_boundaries_and_outputs(tmp_path, mode):
    events = []

    class NVTX:
        def RangePush(self, label):
            events.append("push")

        def RangePop(self):
            events.append("pop")

    def operation():
        events.append("operation")
        return 42

    result, elapsed = module.measure(
        "first",
        operation,
        lambda: events.append("sync"),
        tmp_path,
        mode,
        NVTX() if mode == "trace" else None,
    )
    assert result == 42 and elapsed >= 0
    assert events == (
        ["sync", "push", "operation", "sync", "pop"]
        if mode == "trace"
        else ["sync", "operation", "sync"]
    )
    if mode == "profile":
        assert pstats.Stats(str(tmp_path / "first.pstats")).total_calls > 0
        assert (tmp_path / "first-functions.json").exists()
    else:
        assert not list(tmp_path.glob("*.pstats"))


def test_definitions_leave_main_unexecuted(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def triplets(x): return x + 1\n"
        "def make_jets(): pass\n"
        "def timeit(): pass\n"
        "def sync(): pass\n"
        'print("main")\n'
        'raise RuntimeError("main executed")\n'
    )
    namespace = module.load_definitions(probe)
    assert namespace["triplets"](1) == 2


def test_changed_probe_rejected(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text('print("changed")\n')
    with pytest.raises(RuntimeError, match="changed"):
        module.load_definitions(probe)
