"""CPU substitutes check harness boundaries, not CUDA compilation or runtime."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "repro", Path(__file__).with_name("repro.py")
)
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


@pytest.mark.parametrize("case", repro.CASES)
@pytest.mark.parametrize("profile", [False, True])
def test_worker_phases_and_result_checks(monkeypatch, tmp_path, case, profile):
    class Buffer(np.ndarray):
        def get(self):
            return np.asarray(self)

    def buffer(x):
        return np.asarray(x).view(Buffer)

    events = []
    stream = SimpleNamespace(synchronize=lambda: None)
    cp = ModuleType("cupy")
    cp.arange = lambda n, dtype: buffer(np.arange(n, dtype=dtype))
    cp.ones = lambda n, dtype: buffer(np.ones(n, dtype=dtype))
    cp.empty = lambda n, dtype: buffer(np.empty(n, dtype=dtype))
    cp.int64, cp.uint8 = np.int64, np.uint8
    cp.asnumpy = np.asarray
    cp.__version__ = "test-double"
    cp.cuda = SimpleNamespace(
        get_current_stream=lambda: stream,
        runtime=SimpleNamespace(
            getDeviceProperties=lambda _: {"name": b"test"},
            getDevice=lambda: 0,
            runtimeGetVersion=lambda: 0,
            driverGetVersion=lambda: 0,
        ),
    )
    cc = ModuleType("cuda.compute")
    cc.OpKind = SimpleNamespace(PLUS="plus")

    def factory(**ignored):
        events.append("factory")

        def launch(**kw):
            if "temp_storage" in kw and kw["temp_storage"] is None:
                events.append("query")
                return 32
            events.append("execution")
            x, y = kw["d_in"], kw["d_out"]
            if case == "scan":
                y[:] = np.cumsum(x)
            elif case == "reduce":
                y[0] = np.sum(x)
            else:
                y[:] = [kw["op"](value) for value in x]

        return launch

    cc.make_unary_transform = cc.make_inclusive_scan = cc.make_reduce_into = factory
    op = ModuleType("cuda.compute.op")
    op.make_op_adapter = lambda x: x
    cuda = ModuleType("cuda")
    cuda.compute = cc
    for name, module in [
        ("cupy", cp),
        ("cuda", cuda),
        ("cuda.compute", cc),
        ("cuda.compute.op", op),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    output = tmp_path / "report.json"
    repro.worker(SimpleNamespace(size=17, case=case, profile=profile, output=output))
    report = json.loads(output.read_text())
    assert report["parity"]
    assert len(report["warm_executions"]) == 5
    assert events.count("factory") == 2
    assert events.count("execution") == 6
    assert events.count("query") == (case in ("scan", "reduce"))
    assert output.with_suffix(".pstats").exists() == profile
