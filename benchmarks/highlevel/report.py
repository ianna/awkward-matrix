"""Save an HTML comparison of a high-level Awkward benchmark run."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from html import escape
from pathlib import Path
from statistics import median
from urllib.parse import quote

ENVIRONMENT_FIELDS = (
    "python",
    "implementation",
    "platform",
    "machine",
    "threads",
    "packages",
)
STATUS_LABELS = {
    "measured": "Measured",
    "partial": "Partial",
    "unavailable": "Unavailable",
    "absent": "Absent",
    "failed": "Failed",
    "missing_case": "Missing case",
    "no_result": "No result",
}


def summarize(row):
    timing = row.get("timing", {})
    times = timing.get("samples_ns", [])
    memory = row.get("memory", [])
    samples = [item["sample"] for item in memory if item["status"] == "measured"]
    deltas = [sample["delta_bytes"] for sample in samples]
    methods = sorted({sample["method"] for sample in samples})
    return {
        "status": row["status"],
        "time_ns": median(times) if times else None,
        "range_ns": (min(times), max(times)) if times else None,
        "time_samples": len(times),
        "rss_bytes": median(deltas) if deltas else None,
        "rss_methods": methods,
        "rss_samples": len(deltas),
        "rss_trials": len(memory),
        "details": [
            entry["detail"] for entry in [row, timing, *memory] if "detail" in entry
        ],
    }


def percent_change(value, reference, compatible, reason):
    if reference is None or value is None:
        return None, "Measurement unavailable"
    if not compatible:
        return None, reason
    if reference == 0:
        return None, "Baseline is zero"
    return (value - reference) / reference * 100, ""


def build_comparison(run, baseline=None):
    """Align workload keys and calculate changes for compatible samples."""
    if run.get("schema_version") != 1:
        raise ValueError("Unsupported measurement schema")
    releases = {release["awkward"]: release for release in run["releases"]}
    if not releases or len(releases) != len(run["releases"]):
        raise ValueError("Expected nonempty, distinct Awkward releases")
    versions = list(releases)
    if baseline is None:
        try:
            baseline = min(
                versions, key=lambda version: tuple(map(int, version.split(".")))
            )
        except ValueError as error:
            raise ValueError(
                "Specify --baseline when comparing non-final versions"
            ) from error
    if baseline not in releases:
        raise ValueError(f"Baseline {baseline!r} is absent from the run")
    environments = {
        version: all(
            release.get(field) == releases[baseline].get(field)
            for field in ENVIRONMENT_FIELDS
        )
        for version, release in releases.items()
    }
    results = {
        version: {
            (row["api"], row["size"]): summarize(row) for row in release["results"]
        }
        for version, release in releases.items()
    }
    workloads = sorted(set().union(*(entries.keys() for entries in results.values())))
    coverage = {version: Counter() for version in versions}
    rows = []
    for api, size in workloads:
        key = (api, size)
        empty = summarize({"status": "no_result"})
        base = results[baseline].get(key, empty)
        for version in versions:
            sample = results[version].get(key, empty)
            match = environments[version]
            time_pct, time_reason = percent_change(
                sample["time_ns"], base["time_ns"], match, "Environment differs"
            )
            same_rss_method = (
                len(sample["rss_methods"]) == 1
                and sample["rss_methods"] == base["rss_methods"]
            )
            rss_pct, rss_reason = percent_change(
                sample["rss_bytes"],
                base["rss_bytes"],
                match and same_rss_method,
                "Environment differs"
                if not match
                else "RSS methods differ or are mixed",
            )
            coverage[version][sample["status"]] += 1
            rows.append(
                {
                    "api": api,
                    "size": size,
                    "release": version,
                    **sample,
                    "time_change_pct": time_pct,
                    "time_change_reason": time_reason,
                    "rss_change_pct": rss_pct,
                    "rss_change_reason": rss_reason,
                }
            )
    return {
        "baseline": baseline,
        "environments_match_baseline": environments,
        "coverage": coverage,
        "rows": rows,
    }


def change_html(value, reason):
    if value is None:
        return f'<span title="{escape(reason)}">—</span>'
    return f"{value:+.2f}%"


def number(value, divisor):
    return "—" if value is None else f"{value / divisor:,.3f}"


def render_html(run, comparison, raw_link):
    baseline = escape(comparison["baseline"])
    coverage_rows = []
    for release, counts in comparison["coverage"].items():
        cells = [f"<td>{counts[status]}</td>" for status in STATUS_LABELS]
        matches = comparison["environments_match_baseline"][release]
        coverage_rows.append(
            f'<tr><th scope="row">{escape(release)}</th>'
            f"<td>{'Yes' if matches else 'No'}</td>{''.join(cells)}</tr>"
        )

    result_rows = []
    for row in comparison["rows"]:
        time_range = "—"
        if row["range_ns"] is not None:
            low, high = row["range_ns"]
            time_range = f"{number(low, 1_000)}–{number(high, 1_000)}"
        status = escape(STATUS_LABELS.get(row["status"], row["status"]))
        if row["details"]:
            details = escape("\n".join(row["details"]))
            status = (
                f"<details><summary>{status}</summary><pre>{details}</pre></details>"
            )
        method = ", ".join(row["rss_methods"])
        rss = number(row["rss_bytes"], 1024)
        rss = f'<span title="{escape(method)}">{rss}</span>'
        time_change = change_html(row["time_change_pct"], row["time_change_reason"])
        rss_change = change_html(row["rss_change_pct"], row["rss_change_reason"])
        if row["release"] == comparison["baseline"]:
            time_change = rss_change = "Baseline"
        result_rows.append(
            f'<tr><th scope="row">{escape(row["api"])}</th><td>{row["size"]}</td>'
            f"<td>{escape(row['release'])}</td><td>{status}</td>"
            f"<td>{number(row['time_ns'], 1_000)}</td><td>{time_range}</td>"
            f"<td>{row['time_samples']}</td><td>{time_change}</td><td>{rss}</td>"
            f"<td>{row['rss_samples']}/{row['rss_trials']}</td><td>{rss_change}</td></tr>"
        )

    metadata = {key: run.get(key) for key in ("schema_version", "settings")}
    metadata["releases"] = [
        {key: value for key, value in release.items() if key not in {"apis", "results"}}
        for release in run["releases"]
    ]
    details = escape(json.dumps(metadata, indent=2))
    coverage_head = "".join(f"<th>{label}</th>" for label in STATUS_LABELS.values())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Awkward release comparison</title>
<style>
body {{ font: 1rem/1.5 system-ui, sans-serif; margin: 2rem auto; padding: 0 1rem;
       max-width: 90rem; color: #1b2634; background: #fff; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: .45rem .65rem; text-align: left; border-bottom: 1px solid #dce3ea; }}
thead {{ background: #edf2f7; }}
tbody tr:nth-child(even) {{ background: #f7f9fb; }}
pre {{ white-space: pre-wrap; min-width: 15rem; }}
</style>
</head>
<body>
<h1>Awkward release comparison</h1>
<p>Baseline <strong>{baseline}</strong>. <a href="{escape(raw_link)}">Raw measurement JSON</a></p>
<h2>Coverage</h2>
<div class="scroll"><table>
<thead><tr><th>Release</th><th>Environment matches baseline</th>{coverage_head}</tr></thead>
<tbody>{"".join(coverage_rows)}</tbody></table></div>
<h2>Measurements</h2>
<p>Changes are relative to the same workload and size in the baseline.
Positive changes mean higher time or RSS. The time range shows the observed
minimum and maximum. Percentages require matching environments, and RSS
percentages also require matching measurement methods and a nonzero baseline.</p>
<p>RSS delta is the increase above the setup baseline. For process high-water
measurements, zero means the operation did not exceed the setup high-water mark.
Memory sample counts show successful trials out of attempted trials.</p>
<div class="scroll"><table><thead><tr>
<th>API</th><th>Rows</th><th>Release</th><th>Status</th>
<th>Median time (µs)</th><th>Time range (µs)</th><th>Time samples</th><th>Time change</th>
<th>Median RSS delta (KiB)</th><th>RSS samples</th><th>RSS change</th>
</tr></thead><tbody>{"".join(result_rows)}</tbody></table></div>
<h2>Run metadata</h2>
<details><summary>Settings and release environments</summary><pre>{details}</pre></details>
</body></html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="measurement JSON")
    parser.add_argument("--baseline", help="release to compare against")
    parser.add_argument("--output", type=Path, required=True, help="HTML report path")
    args = parser.parse_args(argv)
    try:
        if args.measurements.resolve() == args.output.resolve():
            raise ValueError("The HTML report and measurement JSON need distinct paths")
        run = json.loads(args.measurements.read_text(encoding="utf-8"))
        report = build_comparison(run, args.baseline)
        raw_link = quote(os.path.relpath(args.measurements, args.output.parent))
        args.output.write_text(render_html(run, report, raw_link), encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
