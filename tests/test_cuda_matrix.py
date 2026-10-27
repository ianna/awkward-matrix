# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks/cuda_matrix"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "matrix_" + name, DIRECTORY / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The CLI modules are intentionally standalone (no installation required).
with_path = list(sys.path)
sys.path.insert(0, str(DIRECTORY))
catalog, runner, operations, report = (
    load(n) for n in ("catalog", "runner", "operations", "report")
)
sys.path[:] = with_path


def test_generated_fixture_parser_does_not_execute(tmp_path):
    path = tmp_path / "test.py"
    path.write_text(
        "def test_one():\n    out = [123, 123]\n    out = dangerous(out)\n    length = 2\n    raise RuntimeError()\n"
    )
    assert catalog.literal_inputs(path, [{"name": "out"}, {"name": "length"}]) == [
        {"id": "test_one", "values": [[123, 123], 2]}
    ]
    assert catalog.literal_inputs(tmp_path / "missing", []) == []


def test_discovery_uses_dispatch_and_spec_and_reports_gaps(tmp_path):
    backend = tmp_path / "src/awkward/_backends/cupy.py"
    backend.parent.mkdir(parents=True)
    backend.write_text(
        'def _get_cuda_compute_impl(name):\n    return {"awkward_ListArray_at": 1, "awkward_alias": 2}\n'
    )
    (tmp_path / "kernel-specification.yml").write_text(
        'kernels:\n- name: awkward_ListArray_at\n  specializations:\n  - name: specialized\n    args: [{name: out, type: "List[int64_t]", dir: out}]\n'
    )
    result = catalog.inventory(tmp_path)
    assert result["kernels"][0]["coverage"] == "missing_fixture"
    assert result["dispatch_without_selected_spec"] == ["awkward_alias"]
    assert not catalog.inventory(tmp_path, families=["RegularArray"])["kernels"]
    assert catalog.dtype("Const[List[uint32_t]]") == "uint32"


@pytest.mark.parametrize("phase", ["call", "finish", "both", None])
def test_completion_always_drains(monkeypatch, phase):
    calls = []

    class Event:
        def record(self, stream):
            calls.append("record")

        def synchronize(self):
            calls.append("wait")

    cp = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, get_elapsed_time=lambda a, b: 0.002)
    )

    def finish(cp, stream):
        calls.append("finish")
        if phase in ("finish", "both"):
            raise ValueError("index out of range")

    def call():
        if phase in ("call", "both"):
            raise IndexError("index out of range")

    monkeypatch.setattr(runner, "finish_cuda", finish)
    metrics, _, errors = runner.completed(call, cp, object())
    assert calls == ["record", "record", "finish", "wait"]
    assert metrics["completion_us"] >= metrics["host_us"]
    assert len(errors) == (2 if phase == "both" else int(phase is not None))


def test_measure_resets_excluded_and_warmup_not_counted():
    count = {"call": 0, "reset": 0}

    def call():
        count["call"] += 1

    def reset():
        count["reset"] += 1

    result = runner.measure(call, reset, {"warmup": 2, "repeats": 2, "iterations": 3})
    assert count == {"call": 8, "reset": 8}
    assert len(result["samples"]) == 2
    assert result["error_stages"] == {"call": 0, "finish": 0}


def test_compare_exact_indices_tolerant_floats_and_incomplete(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, x=np.array([1, 2]), y=np.array([1.0, np.nan]))
    np.savez(b, x=np.array([1, 2]), y=np.array([1.000001, np.nan]))
    cpu = {"status": "measured", "errors": [], "arrays": str(a)}
    other = dict(cpu, arrays=str(b))
    assert report.compare_rows(cpu, other)[0] == "failed"
    assert report.compare_rows(cpu, other, rtol=1e-5)[0] == "passed"
    np.savez(b, x=np.array([1, 3]), y=np.array([1.0, np.nan]))
    assert report.compare_rows(cpu, other, rtol=1)[0] == "failed"
    assert report.compare_rows(cpu, {"status": "unavailable"})[0] == "incomplete"


def test_unexpected_errors_do_not_become_public_parity():
    cpu = {"status": "measured", "errors": [{"message": "unsupported type"}]}
    assert report.compare_rows(cpu, cpu)[0] == "not_applicable"
    gpu = {"status": "measured", "errors": [{"message": "CUDA launch failed"}]}
    assert report.compare_rows(cpu, gpu)[0] == "failed"
    assert runner.encode(complex(np.nan, np.inf)) == {
        "complex": [{"float": "nan"}, {"float": "inf"}]
    }


def test_comparison_missing_case_fails_closed(tmp_path):
    env = {"git_sha": "sha", "device": "GPU", "cuda_driver": 1, "cuda_runtime": 1}
    paths = {}
    for mode in ("cpu", "raw", "compute"):
        paths[mode] = tmp_path / f"{mode}.json"
        runner.save(
            paths[mode],
            {
                "mode": mode,
                "complete": True,
                "expected_case_count": 1 if mode != "raw" else 0,
                "manifest_sha256": "hash",
                "settings": {},
                "environment": env,
                "cases": [
                    {
                        "id": "op",
                        "status": "measured",
                        "result": [1],
                        "errors": [],
                        "median": {"completion_us": 10},
                    }
                ]
                if mode != "raw"
                else [],
            },
        )
    result = report.compare(
        SimpleNamespace(**paths, output=tmp_path / "comparison", rtol=0, atol=0)
    )
    assert not result["three_way_parity_complete"]
    assert result["cases"][0]["modes"]["raw"]["parity"] == "incomplete"


@pytest.mark.parametrize("layout", operations.LAYOUTS)
@pytest.mark.parametrize("dtype", operations.DTYPES)
def test_cpu_public_catalogue(layout, dtype):
    ak = pytest.importorskip("awkward")
    fixtures = list(
        operations.cases(
            ak,
            layout,
            dtype,
            3,
            4,
            "cpu",
            ["num", "slice_advanced", "slice_spread", "sum"],
        )
    )
    assert len(fixtures) == 4
    for _, call in fixtures:
        _, result, errors = runner.completed(call)
        if not errors:
            json.dumps(runner.snapshot(ak, result), allow_nan=False)


def test_native_kernel_manifest_worker_smoke(tmp_path, monkeypatch):
    pytest.importorskip("awkward")
    checkout = os.environ.get("AWKWARD_MATRIX_CHECKOUT")
    if not checkout:
        pytest.skip("Needs matching generated Awkward tree")
    # One deterministic valid fixture: do not time a hand-waved synthetic call.
    entry = {
        "kernel": "awkward_RegularArray_reduce_local_nextparents_64",
        "specialization": "awkward_RegularArray_reduce_local_nextparents_64",
        "arguments": [
            {"name": "out", "type": "List[int64_t]", "dir": "out"},
            {"name": "size", "type": "int64_t", "dir": "in"},
            {"name": "length", "type": "int64_t", "dir": "in"},
        ],
        "fixtures": [{"id": "parents", "values": [[123] * 6, 3, 2]}],
    }
    call, reset, arrays = runner.kernel_case(entry, entry["fixtures"][0], "cpu", None)
    reset()
    _, _, errors = runner.completed(call)
    assert not errors
    np.testing.assert_array_equal(arrays[0], [0, 0, 0, 1, 1, 1])


def test_float_spec_is_float32_not_numpy_float64():
    assert catalog.dtype("Const[List[float]]") == "float32"
    assert catalog.dtype("List[double]") == "float64"


def test_raw_dispatch_rejects_compute_fallback_and_direct_apis(monkeypatch):
    import types

    import awkward._kernels as kernels

    class Raw:
        pass

    class Compute:
        def __init__(self, impl=None, key=None):
            self.impl = impl

    class FakeBackend:
        _kernels = {}

        def _new_kernel(self, key):
            return Raw() if key[0] == "raw" else Compute()

        def _get_cuda_compute_impl(self, name):
            return (lambda: None) if name == "compute" else None

        @classmethod
        def instance(cls):
            return instance

    instance = FakeBackend()
    monkeypatch.setattr(kernels, "CupyKernel", Raw)
    monkeypatch.setattr(kernels, "CudaComputeKernel", Compute)
    fake_backend = types.ModuleType("awkward._backends.cupy")
    fake_backend.CupyBackend = FakeBackend
    monkeypatch.setitem(sys.modules, "awkward._backends.cupy", fake_backend)
    cuda = types.ModuleType("cuda")
    cuda.__path__ = []
    cccl = types.ModuleType("cuda.compute")
    cuda.compute = cccl

    def transform():
        return "original"

    transform.__module__ = "cuda.compute.algorithms"
    cccl.unary_transform = transform
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.compute", cccl)
    with runner.dispatch("raw", {"compute"}, set()):
        assert isinstance(instance._new_kernel(("raw",)), Raw)
        with pytest.raises(runner.Unavailable):
            instance._new_kernel(("compute",))
        with pytest.raises(runner.Unavailable):
            cccl.unary_transform()
    assert cccl.unary_transform() == "original"
    with runner.dispatch("compute", {"compute"}, set()):
        assert isinstance(instance._new_kernel(("compute",)), Compute)


def test_pointer_table_uses_pointee_outputs_not_addresses():
    pytest.importorskip("awkward")
    entry = {
        "kernel": "awkward_ListArray_combinations",
        "specialization": "awkward_ListArray64_combinations_64",
        "arguments": [
            {"name": n, "type": t}
            for n, t in (
                ("tocarry", "List[List[int64_t]]"),
                ("toindex", "List[int64_t]"),
                ("fromindex", "List[int64_t]"),
                ("n", "int64_t"),
                ("replacement", "bool"),
                ("starts", "List[int64_t]"),
                ("stops", "List[int64_t]"),
                ("length", "int64_t"),
            )
        ],
    }
    fixture = {"values": [[[123], [123]], [123, 123], [0, 0], 2, False, [0], [2], 1]}
    call, reset, arrays = runner.kernel_case(entry, fixture, "cpu", None)
    reset()
    assert not runner.completed(call)[2]
    np.testing.assert_array_equal(arrays[0], [0])
    np.testing.assert_array_equal(arrays[1], [1])
    reset()
    np.testing.assert_array_equal(arrays[0], [123])


def test_all_current_specializations_have_fixtures():
    checkout = os.environ.get("AWKWARD_MATRIX_CHECKOUT")
    if not checkout:
        pytest.skip("Needs generated candidate tree")
    result = catalog.inventory(checkout)
    assert result["kernels"]
    assert all(entry["fixtures"] for entry in result["kernels"])
    ids = [
        fixture["id"] for entry in result["kernels"] for fixture in entry["fixtures"]
    ]
    assert len(ids) == len(set(ids))


def test_native_float32_recipe_preserves_buffer_abi():
    pytest.importorskip("awkward")
    entry = {
        "kernel": "awkward_reduce_sum",
        "specialization": "awkward_reduce_sum_float32_float32_64",
        "arguments": [
            {"name": "toptr", "type": "List[float]"},
            {"name": "fromptr", "type": "Const[List[float]]"},
            {"name": "offsets", "type": "List[int64_t]"},
            {"name": "outlength", "type": "int64_t"},
        ],
    }
    call, reset, buffers = runner.kernel_case(
        entry, {"values": [[123, 123], [1.25, 2.5, 3.75], [0, 2, 3], 2]}, "cpu", None
    )
    reset()
    assert not runner.completed(call)[2]
    assert buffers[0].dtype == np.dtype("float32")
    np.testing.assert_array_equal(buffers[0], [3.75, 3.75])


def test_validation_is_outside_timing_and_checks_every_iteration():
    seen = []
    runner.measure(
        lambda: 42,
        lambda: None,
        {"warmup": 2, "iterations": 3, "repeats": 2},
        validate=lambda value, errors: seen.append((value, errors)),
    )
    assert seen == [(42, [])] * 8


def test_native_worker_to_comparison_roundtrip(tmp_path):
    ak = pytest.importorskip("awkward")
    checkout = Path(ak.__file__).resolve().parents[2]
    if not (checkout / "kernel-specification.yml").exists():
        pytest.skip("Needs a source checkout")
    entry = {
        "kernel": "awkward_RegularArray_reduce_local_nextparents_64",
        "specialization": "awkward_RegularArray_reduce_local_nextparents_64",
        "arguments": [
            {"name": "out", "type": "List[int64_t]"},
            {"name": "size", "type": "int64_t"},
            {"name": "length", "type": "int64_t"},
        ],
        "fixtures": [{"id": "parents", "values": [[123] * 6, 3, 2]}],
    }
    manifest, output = tmp_path / "manifest.json", tmp_path / "cpu.json"
    runner.save(
        manifest,
        {
            "kernels": [entry],
            "scope": "kernels",
            "operations": [],
            "layouts": [],
            "dtypes": [],
            "shapes": [],
        },
    )
    runner.run(
        SimpleNamespace(
            checkout=checkout,
            manifest=manifest,
            output=output,
            mode="cpu",
            scope=None,
            warmup=1,
            iterations=2,
            repeats=2,
            max_fixtures=0,
        )
    )
    measured = json.loads(output.read_text())
    assert measured["complete"] and measured["expected_case_count"] == 1
    assert len(output.with_suffix(".jsonl").read_text().splitlines()) == 1
    # Reuse CPU artifacts only to test report plumbing; no GPU is simulated.
    paths = {"cpu": output}
    for mode in ("raw", "compute"):
        replica = json.loads(json.dumps(measured))
        replica["mode"] = mode
        paths[mode] = tmp_path / f"{mode}.json"
        runner.save(paths[mode], replica)
    result = report.compare(
        SimpleNamespace(**paths, output=tmp_path / "comparison", rtol=0, atol=0)
    )
    assert result["three_way_parity_complete"]
    measured["settings"]["max_fixtures"] = 1
    runner.save(output, measured)
    with pytest.raises(ValueError, match="settings"):
        report.compare(
            SimpleNamespace(**paths, output=tmp_path / "other", rtol=0, atol=0)
        )
