import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Awkward release comparison")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go
    from packaging.version import Version

    from benchmarks.highlevel.report import STATUS_LABELS, build_comparison

    return Path, STATUS_LABELS, Version, build_comparison, go, json, mo


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
    _default = "results/highlevel/2.14.0-2.15.0/measurements.json"
    _path = Path(mo.cli_args().get("measurements", _default))
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
def _(baseline, initial, mo, versions):
    _candidates = [version for version in versions if version != baseline.value]
    _sizes = sorted({row["size"] for row in initial["rows"]})
    candidate = mo.ui.dropdown(
        _candidates,
        value=_candidates[-1] if _candidates else None,
        label="Compare release",
    )
    summary_size = mo.ui.dropdown(
        {f"{size:,}": size for size in _sizes},
        value=f"{_sizes[0]:,}",
        label="Fixture rows",
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
def _(candidate, comparison, direction, metric, mo, summary_size, threshold):
    _field = f"{metric.value}_change_pct"
    _rows = [
        row
        for row in comparison["rows"]
        if row["release"] == candidate.value
        and row["size"] == summary_size.value
        and row[_field] is not None
    ]
    _cutoff = threshold.value or 0
    _improved = [row for row in _rows if row[_field] < 0 and -row[_field] >= _cutoff]
    _degraded = [row for row in _rows if row[_field] > 0 and row[_field] >= _cutoff]
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
                f"**{len(_improved)} improved · {len(_degraded)} degraded** "
                f"across {len(_rows)} comparable workloads. "
                "Select a row to inspect its plots. The cutoff filters observed "
                "changes and does not test statistical significance."
            ),
            ranking if _ranked else mo.md("No functions match this selection."),
        ]
    )
    return (ranking,)


@app.cell
def _(initial, mo, operation, ranking, summary_size):
    active_api = ranking.value[0]["Function"] if ranking.value else operation.value
    _sizes = sorted(
        {row["size"] for row in initial["rows"] if row["api"] == active_api}
    )
    sizes = mo.ui.multiselect(
        {f"{size:,}": size for size in _sizes},
        value=[f"{summary_size.value:,}"]
        if ranking.value
        else [f"{size:,}" for size in _sizes[:4]],
        label="Rows to plot",
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
def _(go):
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
                name=f"{size:,} rows",
                mode="lines+markers",
                connectgaps=False,
                error_y=error,
                hovertemplate=f"Awkward %{{x}}<br>%{{y:,.3f}} {unit}<extra>%{{fullData.name}}</extra>",
            )
        if not fig.data:
            return None
        title = "Runtime" if time else "Peak RSS delta"
        axis = f"Change from baseline ({unit})" if relative else f"Median ({unit})"
        fig.update_layout(
            title=title,
            xaxis={
                "title": "Awkward release",
                "type": "category",
                "categoryorder": "array",
                "categoryarray": versions,
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
