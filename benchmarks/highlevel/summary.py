"""Render a saved benchmark run as a GitHub Actions job summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .report import STATUS_LABELS, build_comparison, release_url

MIN_CHANGE = 5
TOP_CHANGES = 3


METRICS = (
    ("time", "Runtime", "time_ns", "range_ns", 1000, "µs"),
    ("rss", "Peak RSS delta", "rss_bytes", "rss_range_bytes", 1024, "KiB"),
)


def write_comparisons(comparison, stream):
    """Write every metric and workload, including unavailable comparisons."""
    writer = csv.writer(stream)
    writer.writerow(
        [
            "baseline",
            "release",
            "api",
            "fixture_rows",
            "metric",
            "unit",
            "baseline_median",
            "baseline_min",
            "baseline_max",
            "release_median",
            "release_min",
            "release_max",
            "change_pct",
            "classification",
            "comparison_detail",
        ]
    )
    baseline = comparison["baseline"]
    references = {
        (row["api"], row["size"]): row
        for row in comparison["rows"]
        if row["release"] == baseline
    }
    for row in comparison["rows"]:
        if row["release"] == baseline:
            continue
        reference = references[row["api"], row["size"]]
        for metric, label, field, extent, divisor, unit in METRICS:
            change = row[f"{metric}_change_pct"]
            classification = "Unavailable"
            if change is not None:
                if change < -MIN_CHANGE:
                    classification = "Improved"
                elif change > MIN_CHANGE:
                    classification = "Degraded"
                else:
                    classification = "Within ±5%"
            values = []
            for item in (reference, row):
                low, high = item[extent] or (None, None)
                values.extend(
                    value / divisor if value is not None else None
                    for value in (item[field], low, high)
                )
            writer.writerow(
                [
                    baseline,
                    row["release"],
                    row["api"],
                    row["size"],
                    label,
                    unit,
                    *values,
                    change,
                    classification,
                    row[f"{metric}_change_reason"],
                ]
            )


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
        f"{settings.get('timing_batches', 1)} timing processes × {settings['repeats']} trials "
        f"· {settings.get('memory_repeats', 1)} RSS processes per workload",
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
        "### Changes relative to the baseline",
        "",
        "Counts are per API and fixture size. Comparisons require matching "
        "environments. RSS comparisons also require matching methods and a "
        "nonzero baseline. Changes are observations, without a significance test. "
        "See the full HTML report or comparisons.csv for every change and its observed range.",
        "",
        f"| Release | Metric | Improved >{MIN_CHANGE}% | Within ±{MIN_CHANGE}% | Degraded >{MIN_CHANGE}% | Comparable |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    reference = {
        (row["api"], row["size"]): row
        for row in comparison["rows"]
        if row["release"] == baseline
    }
    top = []
    within_rows = []
    for version in releases:
        if version == baseline:
            continue
        for metric, label, value_field, _, divisor, unit in METRICS:
            field = f"{metric}_change_pct"
            rows = sorted(
                (
                    row
                    for row in comparison["rows"]
                    if row["release"] == version and row[field] is not None
                ),
                key=lambda row: row[field],
            )
            improved = [row for row in rows if row[field] < -MIN_CHANGE]
            degraded = [row for row in reversed(rows) if row[field] > MIN_CHANGE]
            within = [row for row in rows if abs(row[field]) <= MIN_CHANGE]
            counts = (
                f"{len(improved)} | {len(within)} | {len(degraded)}"
                if rows
                else "— | — | —"
            )
            lines.append(f"| {labels[version]} | {label} | {counts} | {len(rows)} |")

            def format_row(row):
                api, size = row["api"], row["size"]
                before = reference[api, size][value_field] / divisor
                after = row[value_field] / divisor
                return (
                    f"| {labels[version]} | {label} | `{api}` | {size:,} | "
                    f"{before:,.3f} → {after:,.3f} {unit} | {row[field]:+.2f}% |"
                )

            top.extend(format_row(row) for row in improved[:TOP_CHANGES])
            top.extend(format_row(row) for row in degraded[:TOP_CHANGES])
            within_rows.extend(format_row(row) for row in within)
    columns = [
        "| Release | Metric | API | Fixture rows | Baseline → release | Change |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    if top:
        lines += ["", "### Largest observed changes", "", *columns, *top]
    if within_rows:
        lines += ["", "### Changes within ±5%", ""]
        if len(within_rows) <= 100:
            lines += [*columns, *within_rows]
        else:
            lines.append(
                f"All {len(within_rows):,} rows are in `comparisons.csv`. "
                "Filter the classification column to **Within ±5%**."
            )
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
