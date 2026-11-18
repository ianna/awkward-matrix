"""Compare saved warmed task/phase records; no GPU or histogram loading needed."""

import argparse
import json
import re
import statistics
from pathlib import Path

PATTERN = re.compile(r"(?:^|\s)(phase|subphase)=(\S+)\s+seconds=([0-9.eE+-]+)")


def read_backend(root):
    summary = json.loads((root / "summary.json").read_text())
    chunks = sorted(summary["chunks"], key=lambda c: c["at"])
    assert len(chunks) >= 3 and len({c["pid"] for c in chunks}) == 1
    keys = [(c["pid"], c["start"], c["stop"]) for c in chunks]
    assert len(set(keys)) == len(keys), (
        "Ambiguous ranges across files; inspect separately"
    )
    warmed = set(keys[1:])
    groups = {}
    for path in root.glob("worker-*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "upstream_phase":
                continue
            key = (event["pid"], *event["chunk"])
            if key not in warmed:
                continue
            match = PATTERN.search(event["line"])
            if match:
                kind, phase, seconds = match.groups()
                groups.setdefault((kind, phase), {}).setdefault(key, []).append(
                    float(seconds)
                )
    stats = {}
    for (kind, phase), values in groups.items():
        # Multiple records with an identical name are not silently summed.
        assert all(len(v) == 1 for v in values.values()), f"Repeated phase {phase}"
        stats[kind + ":" + phase] = dict(
            median_s=statistics.median(v[0] for v in values.values()),
            observed_chunks=len(values),
            expected_chunks=len(warmed),
            values_s=[values[k][0] if k in values else None for k in keys[1:]],
        )
    return dict(
        task_median_s=statistics.median(c["seconds"] for c in chunks[1:]),
        chunks=chunks,
        phases=stats,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    results = {b: read_backend(args.results / b) for b in ("cpu", "cuda")}
    for b, report in results.items():
        print(
            f"{b}: warmed task median {report['task_median_s']:.3f} s; "
            f"{len(report['chunks']) - 1} later chunks"
        )
    print(
        "\nUpstream diagnostic timers, not independently fenced device times. "
        "Nested phases overlap; do not sum rows. Missing observations are shown.\n"
    )
    for kind in ("phase", "subphase"):
        print(f"## {kind}\n")
        print(
            "| Name | CPU median s | CUDA median s | CUDA − CPU s | "
            "Observed chunks CPU/CUDA |"
        )
        print("| --- | ---: | ---: | ---: | --- |")
        names = set().union(*(r["phases"] for r in results.values()))

        def delta(name):
            a, b = (results[v]["phases"].get(name) for v in ("cpu", "cuda"))
            return b["median_s"] - a["median_s"] if a and b else float("-inf")

        for name in sorted(
            (n for n in names if n.startswith(kind + ":")), key=delta, reverse=True
        ):
            a, b = (results[v]["phases"].get(name) for v in ("cpu", "cuda"))
            av = f"{a['median_s']:.4f}" if a else "missing"
            bv = f"{b['median_s']:.4f}" if b else "missing"
            difference = f"{delta(name):+.4f}" if a and b else "—"
            observed = "/".join(str(x["observed_chunks"]) if x else "0" for x in (a, b))
            print(
                f"| {name.split(':', 1)[1]} | {av} | {bv} | {difference} | {observed} |"
            )
        print()
    (args.results / "phase-comparison.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
