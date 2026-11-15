"""Collect comparable timing and peak RSS samples across Awkward releases."""

from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
from contextlib import redirect_stdout
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from .cases import CASES, FIXTURE_SIZES, Fixtures, Unavailable
from .coverage import inventory

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "JAX_PLATFORMS": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONHASHSEED": "0",
}
PACKAGE_EXCLUSIONS = {"awkward", "awkward-cpp"}
WORKER_MODULE = "benchmarks.highlevel.measure"


def environment(ak):
    packages = {
        re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower(): dist.version
        for dist in distributions()
    }
    return {
        "awkward": ak.__version__,
        "awkward_cpp": packages.get("awkward-cpp"),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": {name: os.environ.get(name) for name in THREAD_ENV},
        "packages": {
            k: v for k, v in sorted(packages.items()) if k not in PACKAGE_EXCLUSIONS
        },
        "apis": sorted(inventory(ak)),
    }


def completed(run):
    result = run()
    ready = getattr(type(result), "block_until_ready", None)
    if ready is not None:
        ready(result)
    return result


def peak_rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * 1024 if sys.platform.startswith("linux") else peak


def linux_peak_rss():
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise OSError("Linux /proc status has no VmHWM field")


def memory_sample(run):
    read_peak = peak_rss
    method = "process_high_water_delta"
    baseline = read_peak()
    if sys.platform.startswith("linux"):
        try:
            Path("/proc/self/clear_refs").write_text("5", encoding="ascii")
            baseline = linux_peak_rss()
            read_peak = linux_peak_rss
            method = "linux_vmhwm_delta"
        except OSError:
            pass
    result = completed(run)
    peak = read_peak()
    del result
    return {
        "baseline_bytes": baseline,
        "peak_bytes": peak,
        "delta_bytes": max(0, peak - baseline),
        "method": method,
    }


def worker(ak, name, size, mode, warmups, repeats):
    entries = inventory(ak)
    if name not in entries:
        return {"status": "absent"}
    if name not in CASES:
        return {"status": "missing_case"}
    case = CASES[name]
    if case.cuda:
        return {"status": "unavailable", "detail": "CUDA case outside CPU measurements"}

    with TemporaryDirectory(prefix="awkward-measure-") as directory:
        fixtures = Fixtures(ak, size, Path(directory))

        def prepare():
            return case.bind(fixtures, entries[name])

        try:
            if mode == "memory":
                return {"status": "measured", "sample": memory_sample(prepare())}

            samples = []
            for index in range(warmups + repeats):
                # Bind every time for destructive methods, builders, and input iterators.
                run = prepare()
                start = perf_counter_ns()
                result = completed(run)
                elapsed = perf_counter_ns() - start
                if index >= warmups:
                    samples.append(elapsed)
                del result, run
            return {"status": "measured", "samples_ns": samples}
        except Unavailable as error:
            return {"status": "unavailable", "detail": str(error)}
        except Exception as error:
            return {"status": "failed", "detail": f"{type(error).__name__}: {error}"}


def invoke(python, *args, timeout):
    try:
        process = subprocess.run(
            [python, "-m", WORKER_MODULE, "_worker", *map(str, args)],
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, **THREAD_ENV},
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "failed", "detail": str(error)}
    if process.returncode == 0:
        try:
            return json.loads(process.stdout)
        except json.JSONDecodeError:
            pass
    return {
        "status": "failed",
        "detail": f"Worker exit {process.returncode}\n{(process.stderr or process.stdout)[-4000:]}",
    }


def measure_case(python, name, size, args):
    options = {"timeout": args.timeout}
    timing_batches = []
    for _ in range(args.timing_batches):
        timing = invoke(
            python, name, size, "timing", args.warmups, args.repeats, **options
        )
        timing_batches.append(timing)
        if len(timing_batches) == 1 and timing["status"] != "measured":
            return {
                "api": name,
                "size": size,
                "status": timing["status"],
                "timing": timing,
            }

    samples = [
        value for batch in timing_batches for value in batch.get("samples_ns", [])
    ]
    timing = {
        "status": "measured",
        "samples_ns": samples,
        "median_ns": statistics.median(samples),
        "batches": timing_batches,
    }
    memory = [
        invoke(python, name, size, "memory", 0, 1, **options)
        for _ in range(args.memory_repeats)
    ]
    status = (
        "measured"
        if all(sample["status"] == "measured" for sample in [*timing_batches, *memory])
        else "partial"
    )
    row = {
        "api": name,
        "size": size,
        "status": status,
        "timing": timing,
        "memory": memory,
    }
    return row


def compatibility_key(env):
    return {
        k: v
        for k, v in env.items()
        if k
        not in {"awkward", "awkward_cpp", "awkward_version", "source_commit", "apis"}
    }


def measure(args):
    envs = [invoke(python, timeout=args.timeout) for python in args.python]
    for python, env in zip(args.python, envs):
        if env.get("status") == "failed":
            raise ValueError(f"{python}: {env['detail']}")
    if main_commit := getattr(args, "main_commit", None):
        head = envs[-1]
        head.update(
            awkward_version=head["awkward"], awkward="main", source_commit=main_commit
        )
    versions = [env["awkward"] for env in envs]
    if len(set(versions)) != len(versions):
        raise ValueError("Provide one interpreter per distinct Awkward release")
    comparable = all(
        compatibility_key(env) == compatibility_key(envs[0]) for env in envs
    )
    names = set(CASES).union(*(env["apis"] for env in envs))
    selected = sorted(
        name
        for name in names
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in args.case)
    )
    if not selected:
        raise ValueError("No workloads matched --case")
    report = {
        "schema_version": 1,
        "matching_environments": comparable,
        "settings": {
            "sizes": args.size,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timing_batches": args.timing_batches,
            "memory_repeats": args.memory_repeats,
            "timeout_seconds": args.timeout,
        },
        "releases": [{**env, "results": []} for env in envs],
    }

    if baseline := getattr(args, "baseline", None):
        report["default_baseline"] = baseline

    def save():
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)

    save()
    for index, (name, size) in enumerate(itertools.product(selected, args.size)):
        # Rotate which release runs first and keep measurements on the same worker.
        for offset in range(len(envs)):
            i = (index + offset) % len(envs)
            env = envs[i]
            if name in env["apis"]:
                row = measure_case(args.python[i], name, size, args)
            else:
                row = {"api": name, "size": size, "status": "absent"}
            report["releases"][i]["results"].append(row)
        save()
    return int(
        any(
            row["status"] in {"missing_case", "failed", "partial"}
            for release in report["releases"]
            for row in release["results"]
        )
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_worker"]:
        with redirect_stdout(sys.stderr):
            import awkward as ak

            if len(argv) == 1:
                result = environment(ak)
            else:
                _, name, size, mode, warmups, repeats = argv
                result = worker(ak, name, int(size), mode, int(warmups), int(repeats))
        print(json.dumps(result))
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", action="append", required=True, help="release interpreter"
    )
    parser.add_argument("--case", action="append", help="qualified API name or glob")
    parser.add_argument(
        "--size", type=int, action="append", help="number of fixture rows"
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timing-batches", type=int, default=3)
    parser.add_argument("--memory-repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.size = list(dict.fromkeys(args.size or FIXTURE_SIZES))
    args.case = args.case or ["*"]
    if len(args.python) < 2:
        parser.error("provide at least two release interpreters")
    if (
        min(
            *args.size,
            args.repeats,
            args.timing_batches,
            args.memory_repeats,
            args.timeout,
        )
        <= 0
    ):
        parser.error("size, repeats, memory repeats, and timeout must be positive")
    if args.warmups < 0:
        parser.error("warmups must be non-negative")
    try:
        return measure(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
