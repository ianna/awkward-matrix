# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Fail-closed three-way parity and completed-latency reporting."""

import json
from pathlib import Path

import numpy as np
from runner import save


def equal(a, b, rtol, atol):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError("Different result keys/types")
        for key in a:
            equal(a[key], b[key], rtol, atol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError("Different result lengths")
        for x, y in zip(a, b, strict=True):
            equal(x, y, rtol, atol)
    elif (
        isinstance(a, (float, int))
        and not isinstance(a, bool)
        and isinstance(b, (float, int))
        and not isinstance(b, bool)
    ):
        # Integer indices/values remain exact even with floating tolerances.
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                raise AssertionError(f"{a} != {b}")
        else:
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    elif type(a) is not type(b) or a != b:
        raise AssertionError(f"Different values: {a!r}, {b!r}")


def semantic(errors):
    def message(error):
        text = error["message"]
        if "index out of range" in text:
            return "index out of range"
        if "not regular" in text:
            return "not regular"
        return text.split(" in compiled")[0].strip()

    return sorted(message(e) for e in errors)


def compare_rows(cpu, other, rtol=0, atol=0):
    if cpu["status"] != "measured" or other["status"] != "measured":
        return "incomplete", "Missing/failed/unavailable measurement"
    try:
        if semantic(cpu["errors"]) != semantic(other["errors"]):
            raise AssertionError("Error outcomes/messages differ")
        if "arrays" in cpu:
            with (
                np.load(cpu["arrays"], allow_pickle=False) as a,
                np.load(other["arrays"], allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files):
                    raise AssertionError("Different buffers")
                for key in a.files:
                    if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape:
                        raise AssertionError(f"Different buffer dtype/shape: {key}")
                    if a[key].dtype.kind in "fc":
                        np.testing.assert_allclose(
                            a[key], b[key], rtol=rtol, atol=atol, equal_nan=True
                        )
                    else:
                        np.testing.assert_array_equal(a[key], b[key])
            # Compare structured error coordinates when both APIs expose them.
            for a, b in zip(cpu["errors"], other["errors"], strict=True):
                for field in ("id", "attempt"):
                    if field in a and field in b and a[field] != b[field]:
                        raise AssertionError(f"Different error {field}")
        elif not cpu["errors"]:
            equal(cpu["result"], other["result"], rtol, atol)
        else:
            # Only recognized public bounds/regularity failures are parity cases.
            if any(
                x not in ("index out of range", "not regular")
                for x in semantic(cpu["errors"])
            ):
                return (
                    "not_applicable",
                    "CPU public operation rejects this layout/dtype",
                )
        return "passed", ""
    except (AssertionError, ValueError, KeyError, OSError) as error:
        return "failed", str(error)


def compare(args):
    reports = {
        mode: json.loads(Path(getattr(args, mode)).read_text())
        for mode in ("cpu", "raw", "compute")
    }
    if {report["mode"] for report in reports.values()} != {"cpu", "raw", "compute"}:
        raise ValueError("Require one CPU, RawKernel, and compute report")
    for mode, report in reports.items():
        if report.get("complete") is not True:
            raise ValueError(f"Incomplete worker report: {mode}")
        if report.get("expected_case_count") != len(report["cases"]):
            raise ValueError(f"Incomplete workload inventory: {mode}")
        if report.get("harness_sha256") != reports["cpu"].get("harness_sha256"):
            raise ValueError("Reports used different harness code")
        if report["mode"] != mode:
            raise ValueError(f"Mislabeled report: {mode}")
        if (
            report["manifest_sha256"] != reports["cpu"]["manifest_sha256"]
            or report["settings"] != reports["cpu"]["settings"]
        ):
            raise ValueError("Reports used different manifests/settings")
    for mode, current in reports.items():
        if current["environment"].get("python") != reports["cpu"]["environment"].get(
            "python"
        ):
            raise ValueError(f"Different Python environment: {mode}")
        packages = {
            k: v
            for k, v in current["environment"].get("packages", {}).items()
            if k not in ("awkward", "awkward-cpp")
        }
        reference = {
            k: v
            for k, v in reports["cpu"]["environment"].get("packages", {}).items()
            if k not in ("awkward", "awkward-cpp")
        }
        if packages != reference:
            raise ValueError(f"Different dependency environment: {mode}")
    for field in ("device", "cuda_driver", "cuda_runtime"):
        if reports["raw"]["environment"].get(field) != reports["compute"][
            "environment"
        ].get(field):
            raise ValueError(f"Different GPU environment: {field}")
    maps = {
        mode: {row["id"]: row for row in report["cases"]}
        for mode, report in reports.items()
    }
    if any(len(rows) != len(reports[mode]["cases"]) for mode, rows in maps.items()):
        raise ValueError("Duplicate case identifiers")
    ids = set().union(*(rows.keys() for rows in maps.values()))
    entries = []
    for case in sorted(ids):
        cpu = maps["cpu"].get(case, {"status": "missing"})
        row = {"id": case, "modes": {}, "cpu": cpu}
        for mode in ("raw", "compute"):
            other = maps[mode].get(case, {"status": "missing"})
            state, reason = compare_rows(cpu, other, args.rtol, args.atol)
            row["modes"][mode] = {
                "parity": state,
                "reason": reason,
                "measurement": other,
            }
        raw, compute = (maps[m].get(case, {}) for m in ("raw", "compute"))
        if raw.get("median") and compute.get("median"):
            row["compute_over_raw_completion"] = (
                compute["median"]["completion_us"] / raw["median"]["completion_us"]
            )
        entries.append(row)
    coverage = {}
    for mode, rows in maps.items():
        coverage[mode] = sorted(
            {kernel for row in rows.values() for kernel, _ in row.get("dispatch", [])}
        )
    coverage_limits = {m: r.get("coverage", {}) for m, r in reports.items()}
    truncated = any(r["settings"].get("max_fixtures", 0) for r in reports.values())
    expected_kernels = {
        row["kernel"]
        for rows in maps.values()
        for row in rows.values()
        if "kernel" in row
    }
    expected_kernels.update(
        reports["compute"].get("coverage", {}).get("planned_kernels", [])
    )
    counts = {
        mode: {
            state: sum(e["modes"][mode]["parity"] == state for e in entries)
            for state in ("passed", "failed", "incomplete", "not_applicable")
        }
        for mode in ("raw", "compute")
    }
    result = {
        "counts": counts,
        "coverage_limits": coverage_limits,
        "truncated": truncated,
        "kernels_without_public_workload": sorted(
            expected_kernels - set(coverage["compute"])
        ),
        "schema": 1,
        "environments": {m: r["environment"] for m, r in reports.items()},
        "rtol": args.rtol,
        "atol": args.atol,
        "cases": entries,
        "operation_kernel_coverage": coverage,
        "three_way_parity_complete": bool(entries)
        and not truncated
        and all(
            e["modes"][m]["parity"] == "passed"
            for e in entries
            for m in ("raw", "compute")
        ),
    }
    save(str(args.output) + ".json", result)
    lines = [
        "# CPU / RawKernel / cuda.compute matrix",
        "",
        f"Parity counts: `{counts}`.",
        "",
        f"Complete three-way parity: **{result['three_way_parity_complete']}**.",
        "",
        "Timings exclude fixture construction/transfers, resets and JIT warmup. Completed time includes result construction, stream completion and deferred errors. Stream time includes submission gaps. No profiler is active.",
        "",
        "| Case | CPU µs | Raw µs | Compute µs | Compute / raw | CPU↔raw | CPU↔compute |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in entries:
        measurements = [
            row["cpu"],
            *(row["modes"][m]["measurement"] for m in ("raw", "compute")),
        ]
        times = [
            f"{r['median']['completion_us']:.3f}" if r.get("median") else "—"
            for r in measurements
        ]
        ratio = (
            f"{row['compute_over_raw_completion']:.3f}"
            if "compute_over_raw_completion" in row
            else "—"
        )
        lines.append(
            f"| {row['id']} | {' | '.join(times)} | {ratio} | {row['modes']['raw']['parity']} | {row['modes']['compute']['parity']} |"
        )
    Path(str(args.output) + ".md").write_text("\n".join(lines) + "\n")
    return result
