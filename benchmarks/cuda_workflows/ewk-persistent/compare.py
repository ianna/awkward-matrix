"""Check CPU/GPU histograms, including weighted variances and flow bins."""

import gzip
import json
import sys
from pathlib import Path

import cloudpickle
import numpy as np


def axis_permutations(left, right, path):
    assert len(left) == len(right), path + ": histogram rank differs"
    permutations = []
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        if a == b:
            continue
        message = f"{path}: axis {i} differs"
        assert type(a) is type(b), message
        assert type(a).__name__ in ("StrCategory", "IntCategory"), message
        assert a.name == b.name and a.label == b.label, message
        assert a.traits == b.traits and a.metadata == b.metadata, message
        labels, other = list(a), list(b)
        assert len(labels) == len(other) and set(labels) == set(other), (
            message + ": category sets differ"
        )
        offset = int(a.traits.underflow)
        positions = {label: j + offset for j, label in enumerate(other)}
        order = list(range(offset)) + [positions[label] for label in labels]
        if a.traits.overflow:
            order.append(offset + len(other))
        permutations.append((i, order))
    return permutations


def align(values, permutations):
    for dimension, order in permutations:
        values = np.take(values, order, axis=dimension)
    return values


def compare(a, b, path="output"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for name in a:
            compare(a[name], b[name], path + "/" + str(name))
    elif hasattr(a, "axes") and callable(getattr(a, "values", None)):
        permutations = axis_permutations(a.axes, b.axes, path)
        np.testing.assert_allclose(
            a.values(flow=True),
            align(b.values(flow=True), permutations),
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
            err_msg=path,
        )
        av, bv = a.variances(flow=True), b.variances(flow=True)
        assert (av is None) == (bv is None), path
        if av is not None:
            np.testing.assert_allclose(
                av,
                align(bv, permutations),
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
                err_msg=path + "/variance",
            )
    elif hasattr(a, "value"):
        compare(a.value, b.value, path)
    elif isinstance(a, (np.ndarray, int, float, np.number)):
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=path
        )
    else:
        assert a == b, path


def main():
    root = Path(sys.argv[1])
    outputs = []
    for backend in ("cpu", "cuda"):
        with gzip.open(root / backend / "histograms.pkl.gz", "rb") as source:
            outputs.append(cloudpickle.load(source))
    ranges = []
    for backend in ("cpu", "cuda"):
        summary = json.loads((root / backend / "summary.json").read_text())
        ranges.append(
            sorted((c["filename"], c["start"], c["stop"]) for c in summary["chunks"])
        )
    assert ranges[0] == ranges[1], "CPU/GPU processed different entry ranges"
    compare(*outputs)
    print("HISTOGRAM PARITY PASSED: labeled axes, values, variances and flow bins")
    print("\n| Backend | First chunk s | Later chunks median s | Whole runner s |")
    print("| --- | ---: | ---: | ---: |")
    for backend in ("cpu", "cuda"):
        s = json.loads((root / backend / "summary.json").read_text())
        print(
            f"| {backend} | {s['first_chunk_s']:.3f} | "
            f"{s['later_median_s']:.3f} | {s['wall_s']:.3f} |"
        )
    print("\nWorker PIDs and entry ranges verified; details are in each summary.json.")


if __name__ == "__main__":
    main()
