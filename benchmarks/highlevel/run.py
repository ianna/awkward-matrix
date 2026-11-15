"""Prepare isolated Awkward environments, then measure and report on them."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .cases import FIXTURE_SIZES
from .measure import measure
from .report import build_comparison, render_html
from .summary import write_comparisons

UPSTREAM = "https://github.com/scikit-hep/awkward.git"
# NumPy 1.x also supports Awkward 2.6, which cannot import with NumPy 2.
DEPENDENCIES = (
    "numpy==1.26.4",
    "packaging==26.3",
    "fsspec==2026.9.0",
    "pyarrow==20.0.0",
    "pandas==2.3.3",
    "fastavro==1.13.1",
    "safetensors==0.8.0",
)
QUICK_CASES = ["ak.Array", "ak.flatten", "ak.sum", "ak.to_list", "ak.str.length"]


def release_list(value):
    releases = [version.strip() for version in value.split(",")]
    if not all(
        re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) for version in releases
    ):
        raise argparse.ArgumentTypeError("expected comma-separated release versions")
    if len(set(releases)) != len(releases):
        raise argparse.ArgumentTypeError("release versions must be distinct")
    return releases


def create_environment(directory, *packages):
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(directory)], check=True
    )
    interpreter = directory / "bin" / "python"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(interpreter),
            *DEPENDENCIES,
            *packages,
        ],
        check=True,
        env={
            **os.environ,
            "CMAKE_BUILD_PARALLEL_LEVEL": os.environ.get(
                "CMAKE_BUILD_PARALLEL_LEVEL", "2"
            ),
        },
    )
    return str(interpreter)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    measurements = args.output / "measurements.json"
    report = args.output / "comparison.html"
    comparisons = args.output / "comparisons.csv"
    if any(path.exists() for path in (measurements, report, comparisons)):
        raise ValueError("Choose a fresh output directory for this run")

    with TemporaryDirectory(prefix="awkward-matrix-run-") as temporary:
        root = Path(temporary)
        args.python = [
            create_environment(root / version, f"awkward=={version}")
            for version in args.releases
        ]
        args.main_commit = None
        if args.include_main:
            source = root / "awkward"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--depth",
                    "1",
                    "--recurse-submodules",
                    "--shallow-submodules",
                    "--branch",
                    "main",
                    UPSTREAM,
                    str(source),
                ],
                check=True,
            )
            args.main_commit = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            print(f"Benchmarking Awkward main at {args.main_commit}", flush=True)
            args.python.append(
                create_environment(
                    root / "main", str(source), str(source / "awkward-cpp")
                )
            )

        args.output = measurements
        args.case = args.case or (QUICK_CASES if args.profile == "quick" else ["*"])
        args.size = list(dict.fromkeys(args.size or FIXTURE_SIZES))
        args.warmups, args.repeats, args.timeout = 3, 7, 120
        status = measure(args)

    data = json.loads(measurements.read_text(encoding="utf-8"))
    comparison = build_comparison(data, args.baseline)
    report.write_text(
        render_html(data, comparison, measurements.name), encoding="utf-8"
    )
    with comparisons.open("w", encoding="utf-8", newline="") as stream:
        write_comparisons(comparison, stream)
    print(f"Saved {measurements}, {report}, and {comparisons}", flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releases", type=release_list, default="2.15.0")
    parser.add_argument(
        "--include-main", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--baseline", help="release or main, defaults to oldest release"
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--case", action="append", help="override workloads with API globs"
    )
    parser.add_argument(
        "--size", type=int, action="append", help="override fixture sizes"
    )
    parser.add_argument("--timing-batches", type=int, default=3)
    parser.add_argument("--memory-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)
    targets = [*args.releases, *(["main"] if args.include_main else [])]
    if len(targets) < 2:
        parser.error("select at least two targets")
    if args.baseline is not None and args.baseline not in targets:
        parser.error("the baseline must be a selected target")
    if min(args.timing_batches, args.memory_repeats) < 1:
        parser.error("trial counts must be positive")
    if args.size and min(args.size) < 1:
        parser.error("size must be positive")
    try:
        return run(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
