from benchmarks.highlevel.report import build_comparison


def result(times, rss, method="process_high_water_delta"):
    return {
        "api": "ak.sum",
        "size": 32,
        "status": "measured",
        "timing": {"samples_ns": times},
        "memory": [
            {"status": "measured", "sample": {"delta_bytes": rss, "method": method}}
        ],
    }


def run(reference, candidate, *, same_environment=True):
    return {
        "schema_version": 1,
        "releases": [
            {"awkward": "2.15.0", "python": "3.12", "results": [candidate]},
            {
                "awkward": "2.9.0",
                "python": "3.12" if same_environment else "3.11",
                "results": [reference],
            },
        ],
    }


def test_comparison_uses_medians_and_numeric_release_order():
    comparison = build_comparison(run(result([10, 90, 20], 200), result([10, 10], 250)))
    assert comparison["baseline"] == "2.9.0"
    candidate = comparison["rows"][0]
    assert candidate["time_change_pct"] == -50
    assert candidate["rss_change_pct"] == 25
    assert comparison["rows"][1]["range_ns"] == (10, 90)


def test_comparison_preserves_absolute_values_when_environments_differ():
    comparison = build_comparison(
        run(result([10], 0), result([20], 100), same_environment=False)
    )
    candidate = comparison["rows"][0]
    assert candidate["time_ns"] == 20
    assert candidate["rss_bytes"] == 100
    assert candidate["time_change_pct"] is None
    assert candidate["rss_change_reason"] == "Environment differs"


def test_rss_comparison_requires_matching_methods_and_nonzero_baseline():
    candidate = result([20], 100)
    zero = build_comparison(run(result([10], 0), candidate))["rows"][0]
    mismatch = build_comparison(run(result([10], 10, "other"), candidate))["rows"][0]
    assert zero["rss_change_reason"] == "Baseline is zero"
    assert mismatch["rss_change_reason"] == "RSS methods differ or are mixed"
    assert mismatch["time_change_pct"] == 100


def test_saved_baseline_can_be_overridden():
    data = run(result([10], 100), result([20], 100))
    data["default_baseline"] = "2.15.0"
    assert build_comparison(data)["baseline"] == "2.15.0"
    assert build_comparison(data, "2.9.0")["baseline"] == "2.9.0"
