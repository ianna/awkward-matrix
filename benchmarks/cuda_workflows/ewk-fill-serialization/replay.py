"""Serialize the actual #4503 prepared fill, then rebind state in fresh workers."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def key_metadata(helper_bytes, dtype, capability, versions, runtime):
    return {
        "schema": 1,
        "operation": "awkward_IndexedArray_fill/unary_transform",
        "helper_sha256": hashlib.sha256(helper_bytes).hexdigest(),
        "input_dtype": dtype,
        "output_dtype": "<i8",
        "input_rank": 1,
        "output_rank": 1,
        "input_contiguous": True,
        "output_contiguous": True,
        "state_dtype": "<i8",
        "state_rank": 1,
        "capability": capability,
        "versions": versions,
        "cuda_runtime": runtime,
        "python": list(sys.version_info[:3]),
    }


@contextmanager
def counters(interop, jit):
    counts = {"builds": 0, "operator_compiles": 0}
    if not hasattr(interop, "call_build") or not hasattr(jit, "_StatefulOp"):
        raise RuntimeError(
            "Build instrumentation changed; adapt counters before measuring"
        )
    build = interop.call_build
    compile_op = jit._StatefulOp.compile

    def counted_build(*args, **kwargs):
        counts["builds"] += 1
        return build(*args, **kwargs)

    def counted_compile(*args, **kwargs):
        counts["operator_compiles"] += 1
        return compile_op(*args, **kwargs)

    interop.call_build = counted_build
    jit._StatefulOp.compile = counted_compile
    try:
        yield counts
    finally:
        interop.call_build = build
        jit._StatefulOp.compile = compile_op


def measured(rows, stage, counts, function, cp):
    before = counts.copy()
    cp.cuda.get_current_stream().synchronize()
    start = time.perf_counter()
    result = function()
    cp.cuda.get_current_stream().synchronize()
    rows.append(
        {
            "stage": stage,
            "completed_s": time.perf_counter() - start,
            **{key: counts[key] - before[key] for key in before},
        }
    )
    return result


def worker(args):
    import cuda.compute as cc
    import cupy as cp
    import numpy as np
    from awkward._connect.cuda import _indexed_fill as helper
    from cuda.compute import _cccl_interop, _jit

    serialize = getattr(cc, "serialize", None)
    deserialize = getattr(cc, "deserialize", None)
    if serialize is None or deserialize is None:
        from cuda.compute import algorithms

        serialize = getattr(algorithms, "serialize", None)
        deserialize = getattr(algorithms, "deserialize", None)
    if serialize is None or deserialize is None:
        raise RuntimeError(
            "Installed cuda.compute lacks serialization APIs. "
            "Use a separate compatible environment; "
            "do not upgrade the benchmark env in place."
        )
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    capability = [properties["major"], properties["minor"]]
    versions = {
        name: version(name)
        for name in (
            "cuda-cccl",
            "cccl-python",
            "numba-cuda-mlir",
            "numba-cuda",
            "cupy",
        )
    }
    metadata = key_metadata(
        Path(helper.__file__).read_bytes(),
        np.dtype(args.dtype).str,
        capability,
        versions,
        cp.cuda.runtime.runtimeGetVersion(),
    )
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    blob = args.cache_dir / (digest + ".bin")
    manifest = blob.with_suffix(".json")
    args.output.mkdir(parents=True, exist_ok=False)
    source = cp.arange(64, dtype=args.dtype)
    target = cp.empty(64, dtype=cp.int64)
    cp.asnumpy(source[:1])  # Context initialization excluded from all stage timers.
    rows = []
    retained = []
    with counters(_cccl_interop, _jit) as counts:
        prepared = measured(
            rows,
            "adapter",
            counts,
            lambda: helper._PreparedFill(np.dtype(np.int64)),
            cp,
        )
        parameter = cp.asarray([10], dtype=cp.int64)
        prepared.base.bind(parameter)
        try:
            if args.phase == "build":
                if blob.exists():
                    raise AssertionError("Cold phase requires an empty artifact cache")
                prepared.transform = measured(
                    rows,
                    "build",
                    counts,
                    lambda: cc.make_unary_transform(
                        d_in=source, d_out=target, op=prepared.operation
                    ),
                    cp,
                )
                payload = measured(
                    rows, "serialize", counts, lambda: serialize(prepared.transform), cp
                )
                temporary = blob.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_bytes(payload)
                temporary.replace(blob)
                manifest.write_text(json.dumps(metadata, indent=2))
            else:
                assert json.loads(manifest.read_text()) == metadata
                prepared.transform = measured(
                    rows,
                    "deserialize",
                    counts,
                    lambda: deserialize(blob.read_bytes()),
                    cp,
                )
        finally:
            prepared.base.clear()

        for iteration, (length, base) in enumerate(
            ((64, 7), (129, 100), (32, np.int64(55)), (1, -10), (0, 3), (96, 2**40))
        ):
            host = np.arange(length, dtype=args.dtype)
            if np.dtype(args.dtype).kind == "i":
                host[::3] = -1
            device = cp.asarray(host)
            output = cp.full(length + 7, 123, dtype=cp.int64)
            retained.append((device, output))  # Ensure pointer changes are real.
            if length:
                measured(
                    rows,
                    f"invoke_{iteration}",
                    counts,
                    lambda: prepared(device, output[3 : 3 + length], base),
                    cp,
                )
            else:
                # Production fill exits before preparing/launching for zero length.
                rows.append(
                    dict(
                        stage=f"invoke_{iteration}",
                        completed_s=0.0,
                        builds=0,
                        operator_compiles=0,
                    )
                )
            expected = np.full(length + 7, 123, dtype=np.int64)
            expected[3 : 3 + length] = np.where(
                host < 0, -1, host.astype(np.int64) + int(base)
            )
            np.testing.assert_array_equal(cp.asnumpy(output), expected)
            assert prepared.base.__cuda_array_interface__["data"][0] == 0

        stream_cases = []
        before = counts.copy()
        for base in (11, 22, 33):
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                host = (
                    np.array([-1, 0, 5], dtype=args.dtype)
                    if args.dtype != "uint32"
                    else np.array([0, 1, 5], dtype=args.dtype)
                )
                device = cp.asarray(host)
                output = cp.full(7, 123, dtype=cp.int64)
                prepared(device, output[2:5], base)
            stream_cases.append((stream, host, device, output, base))
        for stream, host, device, output, base in stream_cases:
            stream.synchronize()
            expected = np.full(7, 123, dtype=np.int64)
            expected[2:5] = np.where(host < 0, -1, host.astype(np.int64) + base)
            np.testing.assert_array_equal(cp.asnumpy(output), expected)
        stream_counts = {key: counts[key] - before[key] for key in before}
        assert all(value == 0 for value in stream_counts.values()), stream_counts

    result = dict(
        phase=args.phase,
        pid=os.getpid(),
        dtype=args.dtype,
        metadata=metadata,
        stages=rows,
        stream_counts=stream_counts,
        parity=True,
        helper=str(helper.__file__),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    if args.phase == "build":
        assert sum(row["builds"] for row in rows) > 0, (
            "Native build counter did not fire"
        )
        assert sum(row["operator_compiles"] for row in rows) > 0, (
            "Op counter did not fire"
        )
        checked = [row for row in rows if row["stage"].startswith("invoke_")]
    else:
        checked = rows
    assert all(row["builds"] == row["operator_compiles"] == 0 for row in checked), (
        "Unexpected compilation, including first execution after deserialize",
        checked,
    )
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("build", "load"))
    parser.add_argument("--dtype", choices=("int32", "int64", "uint32"))
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.phase:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for dtype in ("int32", "int64", "uint32"):
        for trial, phase in enumerate(("build", "load", "load")):
            out = args.output / f"{dtype}-{trial}-{phase}"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--phase",
                    phase,
                    "--dtype",
                    dtype,
                    "--cache-dir",
                    str(args.cache_dir.resolve()),
                    "--output",
                    str(out.resolve()),
                ],
                check=True,
            )
            results.append(json.loads((out / "result.json").read_text()))
    assert len({r["pid"] for r in results}) == len(results), (
        "Expected fresh worker processes"
    )
    lines = [
        "# Fresh-process IndexedArray_fill serialization",
        "",
        "| Dtype | Phase | Preparation ms | First call ms | Builds | Op compiles |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        stages = r["stages"]
        prep = sum(
            s["completed_s"] for s in stages if not s["stage"].startswith("invoke_")
        )
        first = next(s["completed_s"] for s in stages if s["stage"] == "invoke_0")
        lines.append(
            f"| {r['dtype']} | {r['phase']} | {prep * 1000:.3f} | {first * 1000:.3f} | "
            f"{sum(s['builds'] for s in stages)} | "
            f"{sum(s['operator_compiles'] for s in stages)} |"
        )
    lines += [
        "",
        "PASSED: output/prefix/tail and interleaved nondefault-stream parity;",
        "zero builds/operator compiles in loaded workers, including first invocation.",
        "Preparation includes adapters plus build/serialization or deserialization;",
        "excludes blob filesystem write and initial CUDA context setup.",
        "Instrumented stage timings, not uninstrumented full-analysis measurements.",
    ]
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
