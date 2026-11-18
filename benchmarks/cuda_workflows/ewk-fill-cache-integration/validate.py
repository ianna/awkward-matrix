"""Exercise #4503's production disk-cache path in independent workers."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def worker(args):
    import cupy as cp
    import numpy as np
    from awkward._connect.cuda import _indexed_fill as helper
    from cuda.compute import _cccl_interop, _jit

    assert hasattr(helper, "load_or_build"), "Expected production cache integration"
    if args.phase == "disabled":
        os.environ.pop("AWKWARD_CUDA_CACHE_DIR", None)
    else:
        os.environ["AWKWARD_CUDA_CACHE_DIR"] = str(args.cache.resolve())
    cp.zeros(1).get()  # Context setup is not included in the call timer.
    counts = dict(builds=0, compiles=0)
    build = _cccl_interop.call_build
    compile_op = _jit._StatefulOp.compile

    def counted_build(*a, **kw):
        counts["builds"] += 1
        return build(*a, **kw)

    def counted_compile(*a, **kw):
        counts["compiles"] += 1
        return compile_op(*a, **kw)

    _cccl_interop.call_build = counted_build
    _jit._StatefulOp.compile = counted_compile
    rows = []
    buffers = []
    try:
        for ordinal, base in enumerate((7, np.int64(100), -10)):
            length = (64, 129, 32)[ordinal]
            host = np.arange(length, dtype=args.dtype)
            if args.dtype != "uint32":
                host[::3] = -1
            source = cp.asarray(host)
            target = cp.full(length + 7, 123, dtype=cp.int64)
            buffers.append((source, target))
            stream = cp.cuda.Stream(non_blocking=True)
            cp.cuda.get_current_stream().synchronize()
            before = counts.copy()
            with stream:
                start = time.perf_counter()
                helper.fill(target, 3, source, length, base)
            stream.synchronize()
            elapsed = time.perf_counter() - start
            rows.append(
                dict(seconds=elapsed, **{k: counts[k] - before[k] for k in counts})
            )
            expected = np.full(length + 7, 123, dtype=np.int64)
            expected[3 : 3 + length] = np.where(
                host < 0, -1, host.astype(np.int64) + int(base)
            )
            np.testing.assert_array_equal(cp.asnumpy(target), expected)
    finally:
        _cccl_interop.call_build = build
        _jit._StatefulOp.compile = compile_op
    result = dict(
        phase=args.phase,
        dtype=args.dtype,
        pid=os.getpid(),
        rows=rows,
        counts=counts,
        parity=True,
        helper=str(helper.__file__),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    if args.phase == "load":
        assert counts == dict(builds=0, compiles=0), result
    else:
        assert counts["builds"] > 0 and counts["compiles"] > 0, result
    assert all(row["builds"] == row["compiles"] == 0 for row in rows[1:]), rows
    if args.phase != "disabled":
        assert list(args.cache.rglob("*.bin")), "No artifact was written"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("disabled", "build", "load"))
    parser.add_argument("--dtype", choices=("int32", "int64", "uint32"))
    args = parser.parse_args()
    if args.phase:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.mkdir(parents=True, exist_ok=False)
    results = []
    for dtype in ("int32", "int64", "uint32"):
        for trial, phase in enumerate(("disabled", "build", "load", "load")):
            out = args.output / f"{dtype}-{trial}-{phase}"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--cache",
                    str(args.cache.resolve()),
                    "--output",
                    str(out.resolve()),
                    "--phase",
                    phase,
                    "--dtype",
                    dtype,
                ],
                check=True,
            )
            results.append(json.loads((out / "result.json").read_text()))
    assert len({r["pid"] for r in results}) == len(results)
    lines = [
        "# Production fill artifact-cache replay",
        "",
        "| Dtype | Phase | First completed call ms | Builds | Compiles |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r['dtype']} | {r['phase']} | {r['rows'][0]['seconds'] * 1000:.3f} | "
            f"{r['counts']['builds']} | {r['counts']['compiles']} |"
        )
    lines += [
        "",
        "PASSED: all outputs/prefix/tail; nondefault streams;",
        "all loaded workers have zero builds/compiles, including their first call.",
        "Instrumented completed calls exclude initial CUDA context setup.",
    ]
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
