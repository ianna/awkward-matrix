"""Replay the installed IndexedArray_fill callback and observe CCCL's real keys."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import time
from pathlib import Path


class Groups:
    def __init__(self):
        self.values = {}

    def group(self, key):
        return self.values.setdefault(key, len(self.values) + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("indexed-fill-cache-results")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    import numpy as np
    from awkward._connect.cuda import _compute
    from cuda.compute import _caching, _cccl_interop, _jit
    from cuda.compute.algorithms import _transform

    for module, attribute in (
        (_caching, "_make_cache_key_from_args"),
        (_caching, "_key_for"),
        (_cccl_interop, "call_build"),
        (_jit, "_StatefulOp"),
    ):
        assert hasattr(module, attribute), f"CCCL diagnostic API changed: {attribute}"
    stream = cp.cuda.get_current_stream()
    original_factory = _transform.make_unary_transform
    original_build = _cccl_interop.call_build
    original_compile = _jit._StatefulOp.compile
    keys, operator_keys = Groups(), Groups()
    observations = []
    counters = dict(builds=0, compiles=0)

    def factory(*positional, **keywords):
        # This is the specialization portion of the ACTUAL decorated factory
        # key. Device/context/thread remain constant throughout this replay.
        key = _caching._make_cache_key_from_args(*positional, **keywords)
        adapter = keywords["op"]
        op_key = _caching._key_for(adapter)
        function = getattr(adapter, "func", None)
        closure = {}
        if function is not None and function.__closure__:
            for name, cell in zip(
                function.__code__.co_freevars, function.__closure__, strict=True
            ):
                value = cell.cell_contents
                if hasattr(value, "__cuda_array_interface__"):
                    closure[name] = dict(
                        kind="device_array",
                        dtype=str(value.dtype),
                        shape=list(value.shape),
                        pointer=int(value.data.ptr),
                    )
                else:
                    closure[name] = dict(kind=type(value).__name__, value=repr(value))
        observations.append(
            dict(
                factory_key_group=keys.group(key),
                operator_key_group=operator_keys.group(op_key),
                closure=closure,
            )
        )
        return original_factory(*positional, **keywords)

    def build(*a, **kw):
        counters["builds"] += 1
        return original_build(*a, **kw)

    def compile_op(self, *a, **kw):
        counters["compiles"] += 1
        return original_compile(self, *a, **kw)

    source = cp.asarray(np.resize(np.array([-1, 0, 2, -3, 5], dtype=np.int64), 128))
    output = cp.empty(160, dtype=cp.int64)
    source_copy, output_copy = source.copy(), cp.empty_like(output)
    source32, output32 = source.astype(cp.int32), cp.empty(160, dtype=cp.int32)
    # Keep all buffers alive so pool pointer reuse cannot hide a pointer change.
    baseline = (source, output, 64, 3, 10)
    variants = [
        ("identical", baseline),
        ("input_pointer_only", (source_copy, output, 64, 3, 10)),
        ("output_pointer_only", (source, output_copy, 64, 3, 10)),
        ("declared_length_32", (source, output, 32, 3, 10)),
        ("declared_length_96", (source, output, 96, 3, 10)),
        ("input_shape_same_pointer", (source[:64], output, 64, 3, 10)),
        ("output_offset_7", (source, output, 64, 7, 10)),
        ("input_dtype_int32", (source32, output, 64, 3, 10)),
        ("output_dtype_int32", (source, output32, 64, 3, 10)),
        ("base_value_11", (source, output, 64, 3, 11)),
        ("base_value_12", (source, output, 64, 3, 12)),
        ("base_type_numpy_int64", (source, output, 64, 3, np.int64(10))),
    ]
    results = []
    _transform.make_unary_transform = factory
    _cccl_interop.call_build = build
    _jit._StatefulOp.compile = compile_op
    try:
        steps = [("baseline_first", baseline), ("baseline_repeat", baseline)]
        for label, case in variants:
            steps.extend(
                [
                    (label, case),
                    (label + "_repeat", case),
                    ("baseline_after_" + label, baseline),
                ]
            )
        for label, (src, dst, length, offset, base) in steps:
            dst.fill(123)
            stream.synchronize()
            before = counters.copy()
            n = len(observations)
            start = time.perf_counter()
            _compute.awkward_IndexedArray_fill(dst, offset, src, length, base)
            host = time.perf_counter() - start
            stream.synchronize()
            completed = time.perf_counter() - start
            assert len(observations) == n + 1, (
                "Actual callback did not reach the observed factory"
            )
            actual = cp.asnumpy(dst)
            values = cp.asnumpy(src[:length]).astype(np.int64)
            expected = np.full(dst.size, 123, dtype=dst.dtype)
            expected[offset : offset + length] = np.where(
                values < 0, -1, values + int(base)
            )
            np.testing.assert_array_equal(actual, expected)
            row = dict(
                label=label,
                length=length,
                offset=offset,
                base=int(base),
                base_type=type(base).__name__,
                input_dtype=str(src.dtype),
                output_dtype=str(dst.dtype),
                input_shape=list(src.shape),
                input_pointer=int(src.data.ptr),
                output_pointer=int(dst.data.ptr),
                builds=counters["builds"] - before["builds"],
                compiles=counters["compiles"] - before["compiles"],
                host_s=host,
                completed_s=completed,
                parity=True,
                **observations[-1],
            )
            results.append(row)
            # Persist after every step so a later diagnostic failure retains evidence.
            (args.output / "steps.json").write_text(json.dumps(results, indent=2))
            print(
                f"{label}: key={row['factory_key_group']} builds={row['builds']} "
                f"compiles={row['compiles']} host={host:.6f}s parity=passed",
                flush=True,
            )
    finally:
        _transform.make_unary_transform = original_factory
        _cccl_interop.call_build = original_build
        _jit._StatefulOp.compile = original_compile
    versions = {}
    for package in (
        "awkward",
        "cccl-python",
        "cuda-cccl",
        "numba-cuda",
        "numba-cuda-mlir",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    text = inspect.getsource(_compute.awkward_IndexedArray_fill)
    (args.output / "callback.py").write_text(text)
    (args.output / "environment.json").write_text(
        json.dumps(
            dict(
                versions=versions,
                cupy=cp.__version__,
                source_file=_compute.__file__,
                source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                device=cp.cuda.runtime.getDevice(),
                context=int(cp.cuda.driver.ctxGetCurrent()),
                driver=cp.cuda.runtime.driverGetVersion(),
                runtime=cp.cuda.runtime.runtimeGetVersion(),
                cupy_cache=os.environ.get("CUPY_CACHE_DIR"),
                note=(
                    "Actual specialization keys compared by equality in one process. "
                    "No cache clears. Private CCCL APIs are used only for observation; "
                    "timing is diagnostic."
                ),
            ),
            indent=2,
        )
    )
    with (args.output / "comparison.md").open("w") as report:
        report.write(
            "| Case | Factory key group | Native builds | "
            "Operator compiles | Host ms |\n"
        )
        report.write("| --- | ---: | ---: | ---: | ---: |\n")
        for r in results:
            report.write(
                f"| {r['label']} | {r['factory_key_group']} | {r['builds']} | "
                f"{r['compiles']} | {r['host_s'] * 1000:.3f} |\n"
            )
        report.write("\nAll cases passed output and untouched-prefix/tail checks.\n")


if __name__ == "__main__":
    main()
