"""Profile the pinned triplets probe without changing its operation sequence."""

import argparse
import ast
import cProfile
import json
import os
import pstats
import sys
import time
from pathlib import Path

REVISION = "7737ea0ba73c09c3345e43f65a17a3bc43d439a6"


def measure(label, operation, sync, output, mode, nvtx=None):
    sync()
    profiler = cProfile.Profile() if mode == "profile" else None
    if nvtx is not None:
        nvtx.RangePush(label)
    start = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    try:
        result = operation()
        sync()
    finally:
        if profiler is not None:
            profiler.disable()
        elapsed = time.perf_counter() - start
        if nvtx is not None:
            nvtx.RangePop()
    if profiler is not None:
        profiler.dump_stats(str(output / f"{label}.pstats"))
        for key in ("cumulative", "tottime"):
            with (output / f"{label}-{key}.txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                    key
                ).print_stats(100)
        # Machine-readable full table retains paths and avoids rounding short calls.
        rows = []
        for (filename, line, function), (
            primitive,
            calls,
            own,
            cumulative,
            _,
        ) in pstats.Stats(profiler).stats.items():
            rows.append(
                dict(
                    file=filename,
                    line=line,
                    function=function,
                    primitive_calls=primitive,
                    calls=calls,
                    self_s=own,
                    cumulative_s=cumulative,
                )
            )
        (output / f"{label}-functions.json").write_text(json.dumps(rows, indent=2))
    print(f"STARTUP mode={mode} call={label} seconds={elapsed:.6f}", flush=True)
    return result, elapsed


def load_definitions(path):
    # Execute the upstream imports, configuration and function definitions only.
    # The first print starts its main program; reject a changed source layout.
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
        ):
            break
        body.append(node)
    names = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    if not {"triplets", "make_jets", "timeit", "sync"} <= names:
        raise RuntimeError("Upstream probe definitions changed")
    namespace = {"__name__": "startup_probe", "__file__": str(path)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=1_000_000)
    parser.add_argument(
        "--order", choices=("big_first", "small_first"), default="big_first"
    )
    parser.add_argument(
        "--mode", choices=("plain", "profile", "trace"), default="profile"
    )
    args = parser.parse_args()
    if args.events < 1:
        parser.error("--events must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["PROBE_NEVT"] = str(args.events)
    sys.argv = [str(args.probe), "cuda", args.order]
    ns = load_definitions(args.probe)
    ak, np = ns["ak"], ns["np"]
    import cupy as cp
    from awkward._connect import cuda

    # Keep the upstream preparation, including its CPU combinations/counts.
    big = ns["make_jets"](args.events, 8, seed=1)
    small = ns["make_jets"](args.events, 4, seed=2)
    counts = {
        name: int(ak.sum(ak.num(ak.combinations(array, 3))))
        for name, array in (("big", big), ("small", small))
    }
    big, small = ak.to_backend(big, "cuda"), ak.to_backend(small, "cuda")
    ns["sync"]()
    ns["timeit"]("warmup (ak.num)", lambda a: ak.sum(ak.num(a)), big)

    def finish():
        ns["sync"]()
        stream = cp.cuda.get_current_stream()
        if stream.ptr in cuda.cuda_streamptr_to_contexts:
            cuda.synchronize_cuda(stream)

    finish()
    first, second = (big, small) if args.order == "big_first" else (small, big)
    times = {}
    # Audit only subprocess identity/time, not environments or full command lines.
    # Many JIT compilers run in-process, so no subprocess does not mean no JIT.
    with (args.output / "subprocesses.jsonl").open("w") as audit:

        def on_audit(event, values):
            if event == "subprocess.Popen" and not audit.closed:
                audit.write(
                    json.dumps(
                        {
                            "at_s": time.perf_counter() - origin,
                            "executable": str(values[0]),
                        }
                    )
                    + "\n"
                )
                audit.flush()

        origin = time.perf_counter()
        sys.addaudithook(on_audit)
        tracing = args.mode == "trace"
        if tracing:
            cp.cuda.runtime.profilerStart()
        try:
            for label, array in (
                ("first", first),
                ("second", second),
                ("repeat_first", first),
            ):
                result, times[label] = measure(
                    label,
                    lambda: ns["triplets"](array),
                    finish,
                    args.output,
                    args.mode,
                    cp.cuda.nvtx if tracing else None,
                )
        finally:
            if tracing:
                cp.cuda.runtime.profilerStop()
    # Final output parity is checked by the batch wrapper outside all profiles.
    values = ak.to_numpy(ak.to_backend(result, "cpu"))
    if np.ma.isMaskedArray(values) and np.any(values.mask):
        raise AssertionError("Unexpected missing values")
    np.save(args.output / "result.npy", np.asarray(values), allow_pickle=False)
    (args.output / "timings.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "order": args.order,
                "events": args.events,
                "counts": counts,
                "seconds": times,
                "awkward": ak.__version__,
                "awkward_path": ak.__file__,
                "cupy": cp.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cupy_cache": os.environ.get("CUPY_CACHE_DIR"),
                "note": (
                    "Completed calls including deferred-error draining; "
                    "profiled values are diagnostic, not benchmark latencies."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
