"""Render a saved benchmark run as a GitHub Actions job summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import STATUS_LABELS, build_comparison, release_url

MIN_CHANGE = 5
TOP_CHANGES = 3


def markdown_summary(run):
    comparison = build_comparison(run)
    baseline = comparison["baseline"]
    settings = run["settings"]
    sizes = ", ".join(f"{size:,}" for size in settings["sizes"])
    releases = {
        release["awkward"]: release.get("source_commit") for release in run["releases"]
    }
    labels = {
        version: f"[{version}]({release_url(version, commit)})"
        for version, commit in releases.items()
    }
    statuses = list(STATUS_LABELS)
    lines = [
        "## Awkward benchmark results",
        "",
        f"Baseline {labels[baseline]} · Fixture rows {sizes} · "
        f"{settings['repeats']} timing trials per workload",
        "",
        "### Coverage",
        "",
        "| Release | Matches baseline | " + " | ".join(STATUS_LABELS.values()) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in statuses) + " |",
    ]
    for version, counts in comparison["coverage"].items():
        matches = comparison["environments_match_baseline"][version]
        values = " | ".join(str(counts[status]) for status in statuses)
        lines.append(f"| {labels[version]} | {'Yes' if matches else 'No'} | {values} |")

    lines += [
        "",
        f"### Changes of at least {MIN_CHANGE}%",
        "",
        "Counts are per API and fixture size. Comparisons require matching "
        "environments. RSS comparisons also require matching methods and a "
        "nonzero baseline. Changes are observations, without a significance test.",
        "",
        "| Release | Metric | Improved | Degraded | Comparable |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    top = []
    for version in releases:
        if version == baseline:
            continue
        for metric, label in (("time", "Runtime"), ("rss", "Peak RSS delta")):
            field = f"{metric}_change_pct"
            rows = [
                (row[field], row["api"], row["size"])
                for row in comparison["rows"]
                if row["release"] == version and row[field] is not None
            ]
            improved = sorted(row for row in rows if row[0] <= -MIN_CHANGE)
            degraded = sorted(
                (row for row in rows if row[0] >= MIN_CHANGE), reverse=True
            )
            counts = f"{len(improved)} | {len(degraded)}" if rows else "— | —"
            lines.append(f"| {labels[version]} | {label} | {counts} | {len(rows)} |")
            for direction, changes in (("Improved", improved), ("Degraded", degraded)):
                for change, api, size in changes[:TOP_CHANGES]:
                    top.append(
                        f"| {labels[version]} | {label} | {direction} | "
                        f"`{api}` | {size:,} | {change:+.2f}% |"
                    )
    if top:
        lines += [
            "",
            "### Largest observed changes",
            "",
            "| Release | Metric | Direction | API | Fixture rows | Change |",
            "| --- | --- | --- | --- | ---: | ---: |",
            *top,
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    args = parser.parse_args()
    print(
        markdown_summary(json.loads(args.measurements.read_text(encoding="utf-8"))),
        end="",
    )


if __name__ == "__main__":
    main()
