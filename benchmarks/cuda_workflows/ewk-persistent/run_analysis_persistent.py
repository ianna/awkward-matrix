"""Run the pinned analysis with its FuturesExecutor in one persistent process."""

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

from persistent_worker import PersistentPool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--chunks", type=int, default=5)
    args = parser.parse_args()
    if (args.chunks != 0 and args.chunks < 3) or args.chunksize < 1:
        parser.error(
            "Use chunks=0 for the entire file, or at least three chunks, "
            "and a positive chunk size"
        )
    args.output = args.output.resolve()
    args.input = args.input.resolve()
    args.analysis = args.analysis.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["PERSISTENT_RESULTS"] = str(args.output)
    # Keep the upstream runner/processor unchanged. Only supply the pool through
    # the pinned FuturesExecutor's public pool argument; keep its merging policy.
    from coffea import processor

    original = processor.FuturesExecutor
    with PersistentPool() as pool:

        def executor(*positional, **options):
            options["pool"] = pool
            options["workers"] = 1
            return original(*positional, **options)

        processor.FuturesExecutor = executor
        sys.path.insert(0, str(args.analysis.parent))
        os.chdir(args.analysis.parent)
        sys.argv = [
            str(args.analysis),
            str(args.input),
            "-x",
            "futures",
            "-n",
            "1",
            "-s",
            str(args.chunksize),
            "-o",
            "histograms",
            "--outpath",
            str(args.output),
            "-p",
            "semilep",
            "--hist-list",
            "njets",
            "njets_counts",
        ]
        if args.chunks:
            sys.argv.extend(["-c", str(args.chunks)])
        start = time.perf_counter()
        try:
            runpy.run_path(str(args.analysis), run_name="__main__")
        finally:
            processor.FuturesExecutor = original
        elapsed = time.perf_counter() - start
    events = [
        json.loads(line)
        for p in args.output.glob("worker-*.jsonl")
        for line in p.read_text().splitlines()
    ]
    chunks = sorted(
        (e for e in events if e["event"] == "chunk_end"), key=lambda e: e["at"]
    )
    assert len(chunks) >= 3, (
        f"Only {len(chunks)} chunks completed; use smaller chunks/more input"
    )
    assert len({c["pid"] for c in chunks}) == 1, "Worker PID changed"
    keys = [(c["filename"], c["start"], c["stop"]) for c in chunks]
    assert len(set(keys)) == len(keys), "Repeated chunks; investigate retries"
    if args.chunks == 0:
        import uproot

        files = {}
        for chunk in chunks:
            files.setdefault(chunk["filename"], []).append(
                (chunk["start"], chunk["stop"])
            )
        for filename, ranges in files.items():
            ranges.sort()
            assert ranges[0][0] == 0, "Missing beginning of input"
            assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])), (
                "Gap/overlap in input"
            )
            with uproot.open(filename) as source:
                assert ranges[-1][1] == source["Events"].num_entries, (
                    "Input not fully covered"
                )
    import statistics

    summary = dict(
        backend=os.environ["EWKCOFFEA_BACKEND"],
        wall_s=elapsed,
        worker_pid=chunks[0]["pid"],
        full_input=args.chunks == 0,
        events=sum(c["stop"] - c["start"] for c in chunks),
        later_median_events_per_s=statistics.median(
            (c["stop"] - c["start"]) / c["seconds"] for c in chunks[1:]
        ),
        chunks=chunks,
        first_chunk_s=chunks[0]["seconds"],
        later_median_s=statistics.median(c["seconds"] for c in chunks[1:]),
        note=(
            "Completed worker tasks include chunk I/O/materialization, analysis, "
            "histogram filling and task serialization; merging is outside. "
            "Generic CUDA context initialization precedes the first task timer. "
            "Upstream phase timers are diagnostic and not all independently fenced."
        ),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
