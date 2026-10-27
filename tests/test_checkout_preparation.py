# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import subprocess
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/prepare_cuda.py"
spec = importlib.util.spec_from_file_location("prepare_cuda", PATH)
prepare_cuda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_cuda)


def test_repeated_checkout_updates_regenerate_ignored_table(tmp_path):
    generator = tmp_path / "dev/generate-kernel-signatures.py"
    generator.parent.mkdir()
    generator.write_text(
        'from pathlib import Path\nPath("ignored-signatures.py").write_text(Path("revision.txt").read_text())\n'
    )
    source = tmp_path / "revision.txt"
    source.write_text("legacy kernel and helpers")
    prepare_cuda.prepare(tmp_path)
    target = tmp_path / "ignored-signatures.py"
    assert target.read_text() == source.read_text()
    source.write_text("migrated kernel; no legacy helpers")
    prepare_cuda.prepare(tmp_path)
    assert target.read_text() == source.read_text()


def test_generation_failure_stops_preparation(tmp_path):
    generator = tmp_path / "dev/generate-kernel-signatures.py"
    generator.parent.mkdir()
    generator.write_text("raise SystemExit(9)\n")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        prepare_cuda.prepare(tmp_path)
    assert caught.value.returncode == 9


def test_missing_checkout_generator_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="signature generator"):
        prepare_cuda.prepare(tmp_path)


def test_real_generator_removes_deleted_rawkernel_and_helpers(tmp_path):
    import ast
    import os
    import shutil

    import numpy as np

    checkout = os.environ.get("AWKWARD_MATRIX_CHECKOUT")
    if not checkout:
        pytest.skip("Needs Awkward source checkout")
    root = Path(checkout)
    name = "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64"
    sources = Path("src/awkward/_connect/cuda/cuda_kernels")
    if (root / sources / (name + ".cu")).exists():
        pytest.skip("Requires the checkout with the migrated RawKernel removed")
    shutil.copytree(root / sources, tmp_path / sources)
    for relative in ("dev/generate-kernel-signatures.py", "kernel-specification.yml"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    for relative in ("awkward-cpp/include/awkward", "awkward-cpp/src/awkward_cpp"):
        (tmp_path / relative).mkdir(parents=True)
    target = tmp_path / "src/awkward/_connect/cuda/_kernel_signatures.py"
    target.write_text(f"# stale generated wrappers for {name}, {name}_a, {name}_b\n")
    prepare_cuda.prepare(tmp_path)
    source = target.read_text()
    assert name + "_a" not in source and name + "_b" not in source
    # Execute only generated function declarations with NumPy dtype bindings;
    # CuPy imports and CUDA compilation are intentionally unnecessary here.
    namespace = dict(vars(np), cupy=np)
    body = [
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
    ]
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(target), "exec"), namespace
    )
    table = namespace["by_signature"](None)
    assert table[(name, np.int64, np.int64, np.int64)] is None
    assert not any(key[0] in (name + "_a", name + "_b") for key in table)
