# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks/cuda_migrations"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for name in (
    "workloads",
    "measure",
    "dispatch",
    "public_operations",
    "listarray_stages",
):
    load(name)
tuning = load("listarray_tuning")


@pytest.mark.parametrize(
    "kind,variant", [("advanced", 8), ("jagged", 16), ("spread", 32)]
)
def test_tuning_forwards_variant_and_decodes_status(kind, variant):
    events = []
    arrays, stream, status = object(), object(), object()

    def run(*args, **kwargs):
        events.append(("run", args, kwargs))
        return status

    def decode(*args):
        events.append(("decode", args))
        return "error result"

    call = tuning.operation(
        SimpleNamespace(run=run),
        decode,
        kind,
        arrays,
        129,
        12,
        stream,
        variant,
        "kernel",
    )
    assert call() == (None if kind == "spread" else "error result")
    assert events[0] == (
        "run",
        (kind, arrays, 129, 12, stream),
        {"spread_lanes" if kind == "spread" else "chunk_size": variant},
    )
    if kind != "spread":
        assert events[1] == ("decode", (status, 129, tuning.MESSAGES[kind], "kernel"))
    else:
        assert len(events) == 1


def test_tuning_report_uses_the_correct_reference(tmp_path):
    cases = []
    for kind, variants in (("advanced", [8, 64]), ("spread", [0, 32])):
        cases.append(
            {
                "kind": kind,
                "name": kind,
                "variants": {
                    str(variant): {
                        "median": {
                            "host_us": 10,
                            "completion_us": 50 if variant in (8, 32) else 100,
                        },
                        "pool_temporary_peak_bytes": 512,
                    }
                    for variant in variants
                },
            }
        )
    report = {"environment": {"git_sha": "revision"}, "cases": cases}
    tuning.write_report(tmp_path, report)
    assert json.loads((tmp_path / "listarray-tuning.json").read_text()) == report
    text = (tmp_path / "listarray-tuning.md").read_text()
    assert text.count("0.500") == 2
    assert text.count("1.000") == 2
    assert "not selected automatically" in text
