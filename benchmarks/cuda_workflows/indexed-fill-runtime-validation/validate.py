"""Assert zero warmed build/compile calls while runtime base and buffers change."""

import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from awkward._connect.cuda import _compute, _indexed_fill
from cuda.compute import _cccl_interop, _jit


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    builds, compiles = [], []
    original_build = _cccl_interop.call_build
    original_compile = _jit._StatefulOp.compile

    def build(*args, **kwargs):
        builds.append(time.perf_counter())
        return original_build(*args, **kwargs)

    def compile_op(self, *args, **kwargs):
        compiles.append(time.perf_counter())
        return original_compile(self, *args, **kwargs)

    _cccl_interop.call_build = build
    _jit._StatefulOp.compile = compile_op
    keepalive, rows = [], []
    try:
        for dtype in (np.int32, np.int64, np.uint32):
            for i, (length, base) in enumerate(
                [
                    (64, 10),
                    (64, 11),
                    (32, np.int64(12)),
                    (129, -10),
                    (1, 2**40),
                    (0, 100),
                    (96, 1001),
                    (64, 10),
                    (64, 2**63 - 1024),
                ]
            ):
                host = np.resize(
                    np.asarray(
                        [0, 1, 2, 3, 5] if dtype == np.uint32 else [-1, 0, 2, -3, 5],
                        dtype=dtype,
                    ),
                    max(1, length),
                )
                source = cp.asarray(host)
                target = cp.full(length + 9, 123, dtype=cp.int64)
                keepalive.extend((source, target))
                cp.cuda.get_current_stream().synchronize()
                before = len(builds), len(compiles)
                start = time.perf_counter()
                _compute.awkward_IndexedArray_fill(target, 3, source, length, base)
                elapsed = time.perf_counter() - start
                cp.cuda.get_current_stream().synchronize()
                expected = np.full(length + 9, 123, dtype=np.int64)
                x = host[:length].astype(np.int64)
                expected[3 : 3 + length] = np.where(x < 0, -1, x + int(base))
                np.testing.assert_array_equal(cp.asnumpy(target), expected)
                delta = len(builds) - before[0], len(compiles) - before[1]
                if i == 0:
                    assert delta[0] >= 1 and delta[1] >= 1, (
                        "Cold counters were not observed"
                    )
                else:
                    assert delta == (0, 0), (
                        f"Unexpected warmed build/compile: {dtype} {length} "
                        f"{base} {delta}"
                    )
                rows.append(
                    dict(
                        dtype=np.dtype(dtype).name,
                        length=length,
                        base=int(base),
                        base_type=type(base).__name__,
                        builds=delta[0],
                        compiles=delta[1],
                        host_ms=elapsed * 1000,
                        parity=True,
                    )
                )
        # Interleave current-stream launches before waiting, with distinct bases.
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
        pending = []
        before = len(builds), len(compiles)
        for repeat in range(3):
            for i, stream in enumerate(streams):
                with stream:
                    source = cp.arange(4096, dtype=cp.int64)
                    target = cp.empty_like(source)
                    base = 100 * repeat + i
                    _compute.awkward_IndexedArray_fill(
                        target, 0, source, source.size, base
                    )
                    keepalive.extend((source, target))
                    pending.append((stream, target, base))
        for stream, target, base in pending:
            stream.synchronize()
            np.testing.assert_array_equal(cp.asnumpy(target), np.arange(4096) + base)
        assert (len(builds), len(compiles)) == before, (
            "Concurrent-stream replay rebuilt"
        )
    finally:
        _cccl_interop.call_build = original_build
        _jit._StatefulOp.compile = original_compile
    (output / "validation.json").write_text(
        json.dumps(
            dict(
                cases=rows,
                concurrent_stream_parity=True,
                warmed_builds=0,
                warmed_compiles=0,
                helper_file=_indexed_fill.__file__,
                compute_file=_compute.__file__,
            ),
            indent=2,
        )
    )
    print(
        "PASSED: all output/prefix/tail checks; zero warmed builds/compiles; "
        "concurrent streams passed"
    )
    print("| dtype | length | base | builds | compiles | host ms |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['dtype']} | {row['length']} | {row['base']} | "
            f"{row['builds']} | {row['compiles']} | {row['host_ms']:.3f} |"
        )


if __name__ == "__main__":
    main()
