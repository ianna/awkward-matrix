#!/usr/bin/env python
# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Discover, select, measure and compare the complete CUDA migration inventory."""

import argparse
from pathlib import Path

from catalog import selected, write_manifest
from operations import DTYPES, LAYOUTS, NAMES
from report import compare
from runner import run, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser(
        "plan", help="Discover kernels and freeze a shared fixture manifest"
    )
    plan.add_argument(
        "--scope", choices=("all", "kernels", "operations"), default="all"
    )
    plan.add_argument("--checkout", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--kernel", action="append", default=[], help="Repeatable glob")
    plan.add_argument(
        "--family",
        action="append",
        default=[],
        help="Repeatable family glob, e.g. ListArray or reduce",
    )
    plan.add_argument(
        "--operation", action="append", default=[], help="Repeatable glob"
    )
    plan.add_argument("--layout", action="append", choices=LAYOUTS)
    plan.add_argument("--dtype", action="append", choices=DTYPES)
    plan.add_argument("--shape", action="append", help="ROWS:WIDTH; repeatable")
    worker = sub.add_parser("run")
    worker.add_argument("--checkout", required=True)
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument("--mode", choices=("cpu", "raw", "compute"), required=True)
    worker.add_argument(
        "--scope", choices=("all", "kernels", "operations"), default=None
    )
    worker.add_argument("--warmup", type=int, default=5)
    worker.add_argument("--iterations", type=int, default=10)
    worker.add_argument("--repeats", type=int, default=3)
    worker.add_argument(
        "--max-fixtures",
        type=int,
        default=0,
        help="0 = all; truncated coverage is recorded",
    )
    comparison = sub.add_parser("compare")
    for mode in ("cpu", "raw", "compute"):
        comparison.add_argument("--" + mode, required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--rtol", type=float, default=0)
    comparison.add_argument("--atol", type=float, default=0)
    comparison.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        shapes = [
            tuple(map(int, x.split(":")))
            for x in (args.shape or ["32:8", "4096:8", "32:4096"])
        ]
        if any(len(s) != 2 or min(s) <= 0 for s in shapes):
            parser.error("Shapes must contain positive ROWS:WIDTH")
        manifest = write_manifest(args.checkout, args.output, args.kernel, args.family)
        if (args.kernel or args.family) and not manifest["kernels"]:
            parser.error("No kernels match the selectors")
        if args.operation and not any(selected(n, args.operation) for n in NAMES):
            parser.error("No operations match the selectors")
        manifest.update(
            scope=args.scope,
            operations=[n for n in NAMES if selected(n, args.operation)],
            layouts=args.layout or list(LAYOUTS),
            dtypes=args.dtype or list(DTYPES),
            shapes=shapes,
        )
        public = {
            p.stem.removeprefix("ak_")
            for p in (Path(args.checkout) / "src/awkward/operations").glob("ak_*.py")
        }
        manifest["public_operations_without_fixture"] = sorted(public - set(NAMES))
        save(args.output, manifest)
        print(
            f"{len(manifest['kernels'])} specializations, {sum(len(k['fixtures']) for k in manifest['kernels'])} kernel fixtures, {len(manifest['operations'])} public workloads"
        )
    elif args.command == "run":
        if (
            min(args.warmup, args.iterations, args.repeats) <= 0
            or args.max_fixtures < 0
        ):
            parser.error("Counts must be positive; max-fixtures must be nonnegative")
        run(args)
    else:
        if min(args.rtol, args.atol) < 0:
            parser.error("Tolerances must be nonnegative")
        result = compare(args)
        if args.require_complete and not result["three_way_parity_complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
