# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Read the candidate's canonical inventory without importing its GPU backend."""

import ast
import fnmatch
import hashlib
import json
from pathlib import Path


def selected(name, patterns):
    return not patterns or any(fnmatch.fnmatchcase(name, p) for p in patterns)


def family(name):
    return name.removeprefix("awkward_").split("_", 1)[0]


def compute_names(checkout):
    path = Path(checkout) / "src/awkward/_backends/cupy.py"
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_get_cuda_compute_impl",
            "_supports_cuda_compute",
        ):
            names.update(
                x.value
                for x in ast.walk(node)
                if isinstance(x, ast.Constant)
                and isinstance(x.value, str)
                and x.value.startswith("awkward_")
            )
    return names


def dtype(typename):
    primitive = (
        typename.replace("Const[", "")
        .replace("List[", "")
        .replace("]", "")
        .removesuffix("_t")
    )
    return {"float": "float32", "double": "float64", "long double": "longdouble"}.get(
        primitive, primitive
    )


def literal_inputs(path, arguments):
    """Read generated CPU fixtures, never execute their test bodies or assertions."""
    if not path.exists():
        return []
    fixtures = []
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        values = {}
        for statement in node.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id not in values:
                    try:
                        values[target.id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        pass
        if all(a["name"] in values for a in arguments):
            fixtures.append(
                {"id": node.name, "values": [values[a["name"]] for a in arguments]}
            )
    return fixtures


def dedicated_fixtures(name, arguments):
    """Bounded deterministic cases for canonical kernels lacking generated data."""
    values = None
    if name in (
        "awkward_reduce_sumofsquares",
        "awkward_reduce_sumofpowers",
        "awkward_reduce_centered_sumofsquares",
    ):
        values = dict(
            toptr=[123, 123, 123],
            fromptr=[1, 2, 3, 4, 5],
            offsets=[0, 2, 2, 5],
            outlength=3,
            n=3,
            means=[1.5, 0, 4],
        )
    elif name == "awkward_argsort":
        values = dict(
            toptr=[123] * 6,
            fromptr=[3, 1, 1, 2, 5, 4],
            length=6,
            offsets=[0, 3, 3, 6],
            offsetslength=4,
            ascending=True,
            stable=True,
        )
    elif name == "awkward_ListOffsetArray_rpad_and_clip_axis1":
        values = dict(toindex=[123] * 12, fromoffsets=[0, 2, 2, 5], length=3, target=4)
    if values is None or not all(a["name"] in values for a in arguments):
        return []
    return [
        {
            "id": "matrix_"
            + name
            + "_"
            + "_".join(dtype(a["type"]) for a in arguments)
            + ("_descending_unstable" if descending else "_valid"),
            "values": [
                False
                if descending and a["name"] in ("ascending", "stable")
                else values[a["name"]]
                for a in arguments
            ],
        }
        for descending in ([False, True] if name == "awkward_argsort" else [False])
    ]


def inventory(checkout, kernels=(), families=()):
    import yaml

    root = Path(checkout)
    names = compute_names(root)
    specification = yaml.safe_load((root / "kernel-specification.yml").read_text())
    entries = []
    for kernel in specification["kernels"]:
        name = kernel["name"]
        if (
            name not in names
            or not selected(name, kernels)
            or not selected(family(name), families)
        ):
            continue
        for specialization in kernel["specializations"]:
            arguments = specialization["args"]
            path = (
                root
                / "awkward-cpp/tests-cpu-kernels"
                / f"test_cpu{specialization['name']}.py"
            )
            fixtures = literal_inputs(path, arguments)
            explicit = (
                root
                / "awkward-cpp/tests-cpu-kernels-explicit"
                / f"test_unit_cpu{specialization['name']}.py"
            )
            fixtures += literal_inputs(explicit, arguments)
            if not fixtures:
                fixtures = dedicated_fixtures(name, arguments)
                for fixture in fixtures:
                    fixture["id"] = specialization["name"] + fixture["id"]
            entries.append(
                {
                    "kernel": name,
                    "family": family(name),
                    "specialization": specialization["name"],
                    "arguments": arguments,
                    "fixtures": fixtures,
                    "coverage": "generated" if fixtures else "missing_fixture",
                }
            )
    known = {e["kernel"] for e in entries}
    aliases = sorted(
        n
        for n in names - known
        if selected(n, kernels) and selected(family(n), families)
    )
    return {
        "schema": 1,
        "spec_sha256": hashlib.sha256(
            (root / "kernel-specification.yml").read_bytes()
        ).hexdigest(),
        "kernels": entries,
        "dispatch_without_selected_spec": aliases,
    }


def write_manifest(checkout, output, kernels=(), families=()):
    result = inventory(checkout, kernels, families)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result
