"""Isolate cuda.compute first-use builds in fresh processes. No Awkward needed."""

import argparse
import cProfile
import importlib.metadata
import json
import os
import pstats
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

CASES = ("transform", "stateful_transform", "scan", "reduce")


def worker(args):
    # Only worker processes import GPU libraries. No cross-process Python cache.
    import cuda.compute as cc
    import cupy as cp
    import numpy as np
    from cuda.compute.op import make_op_adapter

    stream = cp.cuda.get_current_stream()
    x = cp.arange(args.size, dtype=cp.int64)
    y = cp.empty(1 if args.case == "reduce" else args.size, dtype=cp.int64)
    bias = cp.ones(1, dtype=cp.int64)
    initial = np.asarray(0, dtype=np.int64)
    # Generic context/memory warmup only; no cuda.compute algorithms called yet.
    assert int(x[:8].sum().get()) == sum(range(min(args.size, 8)))
    stream.synchronize()

    def timed(fn):
        stream.synchronize()
        start = time.perf_counter()
        result = fn()
        host = time.perf_counter() - start
        stream.synchronize()
        return result, dict(host_s=host, completed_s=time.perf_counter() - start)

    def increment(value):
        return value + 1

    def captured_increment(value):
        return value + bias[0]

    raw_op = (
        captured_increment
        if args.case == "stateful_transform"
        else increment
        if args.case == "transform"
        else cc.OpKind.PLUS
    )
    adapter, adapter_time = timed(lambda: make_op_adapter(raw_op))
    common = dict(d_in=x, d_out=y, op=adapter)
    if args.case in ("transform", "stateful_transform"):
        factory = partial(cc.make_unary_transform, **common)
    elif args.case == "scan":
        factory = partial(cc.make_inclusive_scan, **common, init_value=None)
    else:
        factory = partial(cc.make_reduce_into, **common, h_init=initial)

    profile = cProfile.Profile() if args.profile else None
    if profile is not None:
        profile.enable()
    try:
        prepared, first_build = timed(factory)
    finally:
        if profile is not None:
            profile.disable()
    if profile is not None:
        prefix = args.output.with_suffix("")
        profile.dump_stats(str(prefix) + ".pstats")
        with Path(str(prefix) + "-profile.txt").open("w") as output:
            stats = pstats.Stats(profile, stream=output)
            stats.sort_stats("cumulative").print_stats(40)
            stats.sort_stats("tottime").print_stats(40)
    # This repeats the same factory key in the same process.
    cached, cached_factory = timed(factory)
    launch_args = dict(common, num_items=args.size, stream=stream)
    query = None
    workspace_bytes = 0
    if args.case in ("scan", "reduce"):
        launch_args["init_value" if args.case == "scan" else "h_init"] = (
            None if args.case == "scan" else initial
        )
        workspace_bytes, query = timed(
            lambda: prepared(temp_storage=None, **launch_args)
        )
        # Allocation outside execution timing.
        launch_args["temp_storage"] = cp.empty(workspace_bytes, dtype=cp.uint8)
    _, first_execution = timed(lambda: prepared(**launch_args))
    repeats = [timed(lambda: cached(**launch_args))[1] for _ in range(5)]
    values = cp.asnumpy(y)
    expected = (
        np.array([args.size * (args.size - 1) // 2], dtype=np.int64)
        if args.case == "reduce"
        else np.cumsum(np.arange(args.size, dtype=np.int64))
        if args.case == "scan"
        else np.arange(args.size, dtype=np.int64) + 1
    )
    np.testing.assert_array_equal(values, expected)
    versions = {}
    for package in (
        "cccl-python",
        "cuda-cccl",
        "numba-cuda",
        "numba-cuda-mlir",
        "numpy",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = properties["name"]
    report = dict(
        case=args.case,
        size=args.size,
        profiled=args.profile,
        adapter=adapter_time,
        first_build=first_build,
        cached_factory=cached_factory,
        workspace_query=query,
        workspace_bytes=int(workspace_bytes),
        first_execution=first_execution,
        warm_executions=repeats,
        parity=True,
        versions=versions,
        cupy=cp.__version__,
        gpu=name.decode() if isinstance(name, bytes) else name,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(),
        cupy_cache=os.environ.get("CUPY_CACHE_DIR"),
        python=sys.version,
        executable=sys.executable,
        note=(
            "Only CuPy cache controlled. Other caches untouched. "
            "No build occurs in parent process."
        ),
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(
        f"{args.case}: build={first_build['host_s']:.6f}s "
        f"cached={cached_factory['host_s']:.6f}s parity=passed",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument(
        "--directory", type=Path, default=Path("cccl-first-use-results")
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Diagnostic build profiles; timings are perturbed",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.size < 1 or args.processes < 1:
        parser.error("size and processes must be positive")
    if args.worker:
        if args.case == "all" or args.output is None:
            parser.error("worker requires one case and output")
        worker(args)
        return
    args.directory.mkdir(parents=True, exist_ok=False)
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        cache = (args.directory / f"cupy-cache-{case}").resolve()
        cache.mkdir()
        for repeat in range(1, args.processes + 1):
            output = args.directory / f"{case}-{repeat}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--case",
                case,
                "--size",
                str(args.size),
                "--output",
                str(output),
            ]
            if args.profile:
                command.append("--profile")
            subprocess.run(
                command, env={**os.environ, "CUPY_CACHE_DIR": str(cache)}, check=True
            )
    print("\nCase | process | first build s | cached factory s | warm execution us")
    for case in cases:
        for repeat in range(1, args.processes + 1):
            data = json.loads((args.directory / f"{case}-{repeat}.json").read_text())
            warm = statistics.median(x["completed_s"] for x in data["warm_executions"])
            print(
                f"{case} | {repeat} | {data['first_build']['host_s']:.6f} | "
                f"{data['cached_factory']['host_s']:.6f} | {warm * 1e6:.3f}"
            )


if __name__ == "__main__":
    main()
