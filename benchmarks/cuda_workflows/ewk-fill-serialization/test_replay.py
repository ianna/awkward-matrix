import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "replay", Path(__file__).with_name("replay.py")
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_key_changes_for_code_dtype_architecture_and_versions():
    original = replay.key_metadata(b"callback", "<i8", [8, 0], {"cccl": "1"}, 12090)
    changed = [
        replay.key_metadata(b"new callback", "<i8", [8, 0], {"cccl": "1"}, 12090),
        replay.key_metadata(b"callback", "<i4", [8, 0], {"cccl": "1"}, 12090),
        replay.key_metadata(b"callback", "<i8", [9, 0], {"cccl": "1"}, 12090),
        replay.key_metadata(b"callback", "<i8", [8, 0], {"cccl": "2"}, 12090),
        replay.key_metadata(b"callback", "<i8", [8, 0], {"cccl": "1"}, 13000),
    ]
    assert all(key != original for key in changed)
    assert (
        replay.key_metadata(b"callback", "<i8", [8, 0], {"cccl": "1"}, 12090)
        == original
    )


def test_counters_restore_after_failure_and_count_first_execution():
    def build():
        return "compiled"

    class Op:
        def compile(self):
            return "operator"

    interop = SimpleNamespace(call_build=build)
    jit = SimpleNamespace(_StatefulOp=Op)
    original = Op.compile
    with pytest.raises(ValueError, match="launch"):
        with replay.counters(interop, jit) as counts:
            assert interop.call_build() == "compiled"
            assert Op().compile() == "operator"
            assert counts == {"builds": 1, "operator_compiles": 1}
            raise ValueError("launch")
    assert interop.call_build is build
    assert Op.compile is original


def test_stage_records_only_its_own_compilation_and_waits():
    waits = []
    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            get_current_stream=lambda: SimpleNamespace(
                synchronize=lambda: waits.append(1)
            )
        )
    )
    counts = {"builds": 9, "operator_compiles": 4}
    rows = []

    def first_launch():
        counts["operator_compiles"] += 1
        return 42

    assert replay.measured(rows, "invoke_0", counts, first_launch, cp) == 42
    assert rows[0]["builds"] == 0
    assert rows[0]["operator_compiles"] == 1
    assert len(waits) == 2
