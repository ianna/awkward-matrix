"""Completed warmed triplets measurements; output checks are outside timing."""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from startup_profile import load_definitions, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=1_000_000)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    if min(args.events, args.samples) < 1:
        parser.error("events and samples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["PROBE_NEVT"] = str(args.events)
    sys.argv = [str(args.probe), "cuda", "big_first"]
    ns = load_definitions(args.probe)
    ak, np = ns["ak"], ns["np"]
    import cupy as cp
    from awkward._connect import cuda

    assert ak.__version__ == "2.14.0", ak.__version__
    expected_root = Path(os.environ["CONTEXT_OVERLAY"]).resolve()
    assert Path(ak.__file__).resolve().is_relative_to(expected_root), ak.__file__
    cpu = {
        name: ns["make_jets"](args.events, width, seed)
        for name, width, seed in [("big", 8, 1), ("small", 4, 2)]
    }
    expected = {name: ak.to_numpy(ns["triplets"](jets)) for name, jets in cpu.items()}
    gpu = {name: ak.to_backend(jets, "cuda") for name, jets in cpu.items()}

    def finish():
        cp.cuda.get_current_stream().synchronize()
        stream = cp.cuda.get_current_stream()
        if stream.ptr in cuda.cuda_streamptr_to_contexts:
            cuda.synchronize_cuda(stream)

    def parity(name, result):
        values = ak.to_numpy(ak.to_backend(result, "cpu"))
        assert (
            values.shape == expected[name].shape
            and values.dtype == expected[name].dtype
        )
        np.testing.assert_allclose(
            values, expected[name], rtol=1e-9, atol=1e-6, equal_nan=True
        )

    finish()
    ns["timeit"]("generic warmup", lambda a: ak.sum(ak.num(a)), gpu["big"])
    cold = {}
    # Workload-specific first uses and additional warmup excluded from samples.
    for name in ("big", "small"):
        result, cold[name] = measure(
            "first_use_" + name,
            lambda: ns["triplets"](gpu[name]),
            finish,
            args.output,
            "plain",
        )
        parity(name, result)
        result = ns["triplets"](gpu[name])
        finish()
        parity(name, result)
    samples = {"big": [], "small": []}
    for repeat in range(args.samples):
        for name in ("big", "small") if repeat % 2 == 0 else ("small", "big"):
            result, elapsed = measure(
                f"{name}_{repeat}",
                lambda: ns["triplets"](gpu[name]),
                finish,
                args.output,
                "plain",
            )
            samples[name].append(elapsed)
            parity(name, result)
    # Separate diagnostic pass: never included in the unprofiled statistics.
    for name in ("big", "small"):
        result, _ = measure(
            "warm_" + name,
            lambda: ns["triplets"](gpu[name]),
            finish,
            args.output,
            "profile",
        )
        parity(name, result)
    report = dict(
        awkward=ak.__version__,
        path=ak.__file__,
        events=args.events,
        first_use_s=cold,
        samples_s=samples,
        median_s={name: statistics.median(values) for name, values in samples.items()},
        parity=True,
        note=(
            "Completed warmed calls. CPU reference, warmup, parity "
            "and cProfile are outside samples."
        ),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("PARITY PASSED both collections, all samples")


if __name__ == "__main__":
    main()
