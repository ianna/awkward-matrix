# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import json
from types import SimpleNamespace

import pytest

from benchmarks.cuda_migrations.listarray_review import compare, shapes


def test_shapes_exceed_previous_cache_capacity():
    assert len({rows for rows, _ in shapes()}) > 32
    assert all(rows > 0 and width > 0 for rows, width in shapes())


def test_comparison_preserves_order_and_cache_metadata(tmp_path):
    paths = [tmp_path / name for name in ("baseline.json", "candidate.json")]
    for path, duration in zip(paths, [10, 5], strict=True):
        path.write_text(
            json.dumps(
                dict(
                    cases=[
                        dict(
                            case="long_next_array:4:1048576",
                            host_us=duration,
                            completed_us=duration + 1,
                        )
                    ],
                    setup_cache=dict(misses=1),
                )
            )
        )
    output = tmp_path / "review.md"
    args = SimpleNamespace(baseline=paths[0], candidate=paths[1], output=output)
    compare(args)
    assert "10.000 → 5.000" in output.read_text()
    assert "misses" in output.read_text()
    paths[1].write_text(json.dumps(dict(cases=[], setup_cache={})))
    with pytest.raises(ValueError, match="inventories differ"):
        compare(args)


def test_batch_ignores_stale_compiler_and_fails_before_profiling(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    root = tmp_path / "CUDA BENCH"
    root.mkdir()
    binaries = tmp_path / "tools"
    binaries.mkdir()
    calls = tmp_path / "calls.txt"
    setup = tmp_path / "setup.sh"
    setup.write_text("module() { :; }\nconda() { :; }\n")
    scripts = {
        "gcc": "echo gcc-test\n",
        "g++": "echo g++-test\n",
        "nvidia-smi": "exit 0\n",
        "git": "echo 0123456789abcdef\n",
        "python": (
            'printf "%s\\n" "$@" >> "$CALLS"\n'
            'case "$*" in "-m pip wheel "*) exit 19;; esac\n'
        ),
    }
    for name, body in scripts.items():
        path = binaries / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{binaries}:/usr/bin:/bin",
        "BASH_ENV": str(setup),
        "SLURM_SUBMIT_DIR": str(root),
        "SLURM_JOB_ID": "42",
        "CC": "/missing/compiler/gcc",
        "CXX": "/opt/rh/gcc-toolset-14/root/usr/bin/g++",
        "CALLS": str(calls),
    }
    script = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/cuda_migrations/della-listarray-review.sbatch"
    )
    result = subprocess.run(
        ["bash", str(script)], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 19, result.stdout + result.stderr
    arguments = calls.read_text().splitlines()
    results = root / "results/42/listarray-review"
    assert f"--config-settings=build-dir={results}/cpu-build" in arguments
    assert (
        f"--config-settings=cmake.define.CMAKE_CXX_COMPILER={binaries}/g++" in arguments
    )
    assert (
        f"--config-settings=cmake.define.CMAKE_C_COMPILER={binaries}/gcc" in arguments
    )
    assert not any("/opt/rh/" in arg for arg in arguments)
    assert not any("measure.py" in arg for arg in arguments)
    assert (results / "cpu-build.log").is_file()


def test_compilation_probe_records_calls_and_restores_on_error():
    from benchmarks.cuda_migrations.listarray_review import compilation_probe

    class Stateful:
        def __init__(self):
            self.func = shapes

        def compile(self, value):
            return value + 1

    jit = SimpleNamespace(_StatefulOp=Stateful)
    original = Stateful.compile
    compiled, value = compilation_probe(jit, lambda: Stateful().compile(41))
    assert compiled == [shapes.__qualname__]
    assert value == 42
    assert Stateful.compile is original

    def failing():
        raise RuntimeError("diagnostic failure")

    with pytest.raises(RuntimeError, match="diagnostic failure"):
        compilation_probe(jit, failing)
    assert Stateful.compile is original
