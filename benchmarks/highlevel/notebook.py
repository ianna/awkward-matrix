import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Awkward release comparison")


@app.cell
def _():
    import json
    from html import escape
    from pathlib import Path
    from urllib.parse import quote

    import marimo as mo
    import plotly.graph_objects as go
    from packaging.version import Version

    from benchmarks.highlevel.cases import fixture_size_label
    from benchmarks.highlevel.report import STATUS_LABELS, build_comparison

    return (
        Path,
        STATUS_LABELS,
        Version,
        build_comparison,
        escape,
        go,
        json,
        mo,
        quote,
        fixture_size_label,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Awkward release comparison
    """)
    return


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=[".json"], label="Open measurement JSON")
    upload
    return (upload,)


@app.cell
def _(Path, build_comparison, json, mo, upload):
    _default = Path("results/highlevel/2.6.0-2.15.0/measurements.json")
    if not _default.is_file():
        _default = Path("results/highlevel/2.14.0-2.15.0/measurements.json")
    _path = Path(mo.cli_args().get("measurements", str(_default)))
    source_name = upload.name() if upload.value else str(_path)
    try:
        run = json.loads(upload.contents() if upload.value else _path.read_bytes())
        initial = build_comparison(run)
    except (OSError, ValueError, TypeError, KeyError) as _error:
        mo.stop(
            True,
            mo.callout(
                mo.md(f"Open a measurement JSON file to continue.\n\n`{_error}`"),
                kind="warn",
            ),
        )
    mo.stop(
        not initial["rows"],
        mo.callout("The run contains no completed workload entries.", kind="warn"),
    )
    mo.md(f"Measurements from **{source_name}**")
    return initial, run


@app.cell
def _(Version, initial, mo):
    versions = sorted(initial["coverage"], key=Version)
    _apis = sorted({row["api"] for row in initial["rows"]})
    baseline = mo.ui.dropdown(versions, value=initial["baseline"], label="Baseline")
    operation = mo.ui.dropdown(
        _apis, value="ak.sum" if "ak.sum" in _apis else _apis[0], label="Operation"
    )
    display_mode = mo.ui.radio(
        ["Absolute", "Change (%)"], value="Absolute", inline=True, label="Display"
    )
    mo.hstack([operation, baseline, display_mode], justify="start", wrap=True)
    return baseline, display_mode, operation, versions


@app.cell
def _(baseline, build_comparison, run):
    comparison = build_comparison(run, baseline.value)
    return (comparison,)


@app.cell
def _(baseline, initial, mo, fixture_size_label, versions):
    _candidates = [version for version in versions if version != baseline.value]
    _sizes = sorted({row["size"] for row in initial["rows"]})
    candidate = mo.ui.dropdown(
        _candidates,
        value=_candidates[-1] if _candidates else None,
        label="Compare release",
    )
    summary_size = mo.ui.dropdown(
        {fixture_size_label(size): size for size in _sizes},
        value=fixture_size_label(1024 if 1024 in _sizes else _sizes[0]),
        label="Fixture size",
    )
    metric = mo.ui.dropdown(
        {"Runtime": "time", "Peak RSS": "rss"}, value="Runtime", label="Metric"
    )
    direction = mo.ui.radio(
        ["Improved", "Degraded"], value="Improved", inline=True, label="Show"
    )
    threshold = mo.ui.number(
        start=0, step=1, value=5, label="Minimum absolute change (%)"
    )
    mo.vstack(
        [
            mo.md("## Improved and degraded functions"),
            mo.hstack(
                [candidate, summary_size, metric, direction, threshold],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return candidate, direction, metric, summary_size, threshold


@app.cell
def _():
    def group_changes(comparison, release, size, metric, cutoff):
        field = f"{metric}_change_pct"
        rows = [
            row
            for row in comparison["rows"]
            if row["release"] == release
            and row["size"] == size
            and row[field] is not None
        ]
        improved = [row for row in rows if row[field] < 0 and -row[field] >= cutoff]
        degraded = [row for row in rows if row[field] > 0 and row[field] >= cutoff]
        return rows, improved, degraded

    return (group_changes,)


@app.cell
def _(build_comparison, run, versions):
    consecutive = [
        (previous, current, build_comparison(run, previous))
        for previous, current in zip(versions, versions[1:])
    ]
    return (consecutive,)


@app.cell
def _(consecutive, group_changes, metric, mo, summary_size, threshold):
    _summary = []
    for _previous, _current, _comparison in consecutive:
        _rows, _improved, _degraded = group_changes(
            _comparison,
            _current,
            summary_size.value,
            metric.value,
            threshold.value or 0,
        )
        _compatible = _comparison["environments_match_baseline"][_current]
        _summary.append(
            {
                "From": _previous,
                "To": _current,
                "Improved": len(_improved) if _rows else None,
                "Degraded": len(_degraded) if _rows else None,
                "Below cutoff or unchanged": len(_rows)
                - len(_improved)
                - len(_degraded)
                if _rows
                else None,
                "Comparable": len(_rows),
                "Status": "Comparable"
                if _rows
                else "No comparable measurements"
                if _compatible
                else "Different environments",
            }
        )
    mo.vstack(
        [
            mo.md("### Release by release"),
            mo.md(
                "Each row compares a release with the previous release in this run. "
                "The metric, fixture size, and cutoff above apply to every pair."
            ),
            mo.ui.table(_summary, selection=None, page_size=8)
            if _summary
            else mo.md("Load at least two releases to compare them."),
        ]
    )
    return


@app.cell
def _(
    candidate, comparison, direction, group_changes, metric, mo, summary_size, threshold
):
    _field = f"{metric.value}_change_pct"
    _rows, _improved, _degraded = group_changes(
        comparison,
        candidate.value,
        summary_size.value,
        metric.value,
        threshold.value or 0,
    )
    _show_improved = direction.value == "Improved"
    _ranked = sorted(
        _improved if _show_improved else _degraded,
        key=lambda row: -abs(row[_field]),
    )
    ranking = mo.ui.table(
        [
            {"Function": row["api"], "Change (%)": round(row[_field], 2)}
            for row in _ranked
        ],
        selection="single",
        page_size=8,
    )
    mo.vstack(
        [
            mo.md(
                f"### {comparison['baseline']} → {candidate.value or 'No comparison release'}\n\n"
                f"**{len(_improved)} improved · {len(_degraded)} degraded** "
                f"across {len(_rows)} comparable workloads. "
                "Select a row to inspect its plots. The cutoff filters observed "
                "changes and does not test statistical significance."
            ),
            mo.callout(
                "The selected release and baseline use different environments. "
                "Use the absolute plots to inspect these measurements.",
                kind="warn",
            )
            if candidate.value
            and not comparison["environments_match_baseline"][candidate.value]
            else ranking
            if _ranked
            else mo.md("No functions match this selection."),
        ]
    )
    return (ranking,)


@app.cell
def _(initial, mo, operation, ranking, fixture_size_label, summary_size):
    active_api = ranking.value[0]["Function"] if ranking.value else operation.value
    _sizes = sorted(
        {row["size"] for row in initial["rows"] if row["api"] == active_api}
    )
    sizes = mo.ui.multiselect(
        {fixture_size_label(size): size for size in _sizes},
        value=[fixture_size_label(summary_size.value)]
        if ranking.value
        else [fixture_size_label(size) for size in _sizes[:4]],
        label="Sizes to plot",
    )
    mo.vstack([mo.md(f"## {active_api}"), sizes])
    return active_api, sizes


@app.cell
def _(active_api, comparison, sizes, versions):
    selected = sorted(
        (
            row
            for row in comparison["rows"]
            if row["api"] == active_api and row["size"] in sizes.value
        ),
        key=lambda row: (row["size"], versions.index(row["release"])),
    )
    return (selected,)


@app.cell
def _(escape, fixture_size_label, go, quote):
    def make_plot(rows, versions, baseline, *, metric, relative):
        time = metric == "time"
        unit = "%" if relative else "µs" if time else "KiB"
        field = "time_ns" if time else "rss_bytes"
        divisor = 1_000 if time else 1_024
        change = f"{metric}_change_pct"
        fig = go.Figure()
        for size in sorted({row["size"] for row in rows}):
            series = [row for row in rows if row["size"] == size]
            values = [
                row[change]
                if relative
                else None
                if row[field] is None
                else row[field] / divisor
                for row in series
            ]
            if all(value is None for value in values):
                continue
            error = None
            if time and not relative:
                error = {
                    "type": "data",
                    "symmetric": False,
                    "array": [
                        (row["range_ns"][1] - row[field]) / divisor
                        if row["range_ns"]
                        else None
                        for row in series
                    ],
                    "arrayminus": [
                        (row[field] - row["range_ns"][0]) / divisor
                        if row["range_ns"]
                        else None
                        for row in series
                    ],
                }
            fig.add_scatter(
                x=[row["release"] for row in series],
                y=values,
                name=fixture_size_label(size),
                mode="lines+markers",
                connectgaps=False,
                error_y=error,
                hovertemplate=f"Awkward %{{x}}<br>%{{y:,.3f}} {unit}<extra>%{{fullData.name}}</extra>",
            )
        if not fig.data:
            return None
        title = "Runtime" if time else "Peak RSS delta"
        axis = f"Change from baseline ({unit})" if relative else f"Median ({unit})"
        release_notes = "https://github.com/scikit-hep/awkward/releases/tag"
        release_links = [
            f'<a href="{release_notes}/v{quote(version, safe="")}" target="_blank">'
            f"<u>{escape(version)}</u></a>"
            for version in versions
        ]
        fig.update_layout(
            title=title,
            xaxis={
                "title": "Awkward release",
                "type": "category",
                "categoryorder": "array",
                "categoryarray": versions,
                "tickmode": "array",
                "tickvals": versions,
                "ticktext": release_links,
                "automargin": True,
            },
            yaxis_title=axis,
            height=340,
            margin={"t": 45, "l": 65, "r": 25, "b": 55},
            legend_title="Fixture size",
        )
        if relative:
            fig.add_hline(y=0, line_dash="dot")
            fig.update_layout(title=f"{title} relative to {baseline}")
        return fig

    return (make_plot,)


@app.cell
def _(baseline, display_mode, make_plot, mo, selected, versions):
    _relative = display_mode.value == "Change (%)"
    _runtime = make_plot(
        selected, versions, baseline.value, metric="time", relative=_relative
    )
    _memory = make_plot(
        selected, versions, baseline.value, metric="rss", relative=_relative
    )
    mo.vstack(
        [
            mo.ui.plotly(_runtime)
            if _runtime
            else mo.callout("No runtime values for this selection.", kind="info"),
            mo.ui.plotly(_memory)
            if _memory
            else mo.callout(
                "No comparable RSS values for this selection.", kind="info"
            ),
            mo.md(
                "Select a release label to open its release notes. "
                "Runtime whiskers show the observed minimum and maximum. "
                "A zero RSS delta can mean the operation stayed below the setup "
                "high-water mark. Positive changes mean higher time or RSS."
            ),
        ]
    )
    return


@app.cell
def _(STATUS_LABELS, comparison, mo, selected):
    def _change(row, metric):
        value = row[f"{metric}_change_pct"]
        return f"{value:+.2f}%" if value is not None else row[f"{metric}_change_reason"]

    def _units(value, divisor):
        return round(value / divisor, 3) if value is not None else None

    _details = [
        {
            "Awkward": row["release"],
            "Rows": row["size"],
            "Status": STATUS_LABELS.get(row["status"], row["status"]),
            "Runtime (µs)": _units(row["time_ns"], 1_000),
            "Runtime change": _change(row, "time"),
            "RSS delta (KiB)": _units(row["rss_bytes"], 1_024),
            "RSS change": _change(row, "rss"),
            "Details": " | ".join(row["details"]),
        }
        for row in selected
    ]
    _coverage = [
        {
            "Awkward": release,
            "Matches baseline": comparison["environments_match_baseline"][release],
            **{label: counts[status] for status, label in STATUS_LABELS.items()},
        }
        for release, counts in comparison["coverage"].items()
    ]
    mo.vstack(
        [
            mo.md("## Selected workload"),
            mo.ui.table(_details, selection=None)
            if _details
            else mo.md("Select a fixture size to inspect measurements."),
            mo.md("## Run coverage"),
            mo.ui.table(_coverage, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
