"""Inventory and smoke-check high-level workloads in one installed release."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from .cases import CASES, Fixtures, Unavailable

# Public protocols documented by Awkward. Ignore Python implementation hooks.
PROTOCOLS = {
    "__array__",
    "__array_function__",
    "__array_ufunc__",
    "__arrow_array__",
    "__cast_cpp__",
    "__delitem__",
    "__dir__",
    "__getattr__",
    "__getitem__",
    "__iter__",
    "__len__",
    "__setattr__",
    "__setitem__",
}
CLASSES = ("Array", "Record", "ArrayBuilder")


def inventory(ak):
    """Return public names and targets without evaluating instance descriptors."""
    entries = {}
    for prefix, namespace in (("ak", ak), ("ak.str", getattr(ak, "str", None))):
        if namespace is None:
            continue
        for name, target in vars(namespace).items():
            if name.startswith("_") or not inspect.isroutine(target):
                continue
            module = getattr(target, "__module__", "")
            if module == ak.__name__ or module.startswith(f"{ak.__name__}."):
                entries[f"{prefix}.{name}"] = target

    for name in CLASSES:
        cls = getattr(ak, name, None)
        if cls is None:
            continue
        entries[f"ak.{name}"] = cls
        for member, target in vars(cls).items():
            if (member in PROTOCOLS and inspect.isroutine(target)) or (
                not member.startswith("_")
                and (
                    inspect.isroutine(target)
                    or inspect.isdatadescriptor(target)
                    or inspect.isclass(target)
                    or isinstance(target, classmethod)
                )
            ):
                entries[f"ak.{name}.{member}"] = target
    return entries


def coverage(ak, *, size=32, smoke=False, cases=CASES):
    """Describe coverage and optionally execute each available workload once."""
    if size < 1:
        raise ValueError("size must be positive")
    entries = inventory(ak)
    rows = []
    for name in sorted(entries.keys() | cases.keys()):
        row = {"api": name, "case": name if name in cases else None}
        if name not in entries:
            row["status"] = "absent"
        elif name not in cases:
            row["status"] = "missing_case"
        elif not smoke:
            row["status"] = "mapped"
        else:
            try:
                with TemporaryDirectory(prefix="awkward-matrix-") as directory:
                    fixtures = Fixtures(ak, size, Path(directory))
                    with warnings.catch_warnings(record=True) as caught:
                        run = cases[name].bind(fixtures, entries[name])
                        run()
                    if caught:
                        row["warnings"] = list(
                            dict.fromkeys(str(warning.message) for warning in caught)
                        )
                row["status"] = "smoke_passed"
            except Unavailable as error:
                row.update(status="unavailable", detail=str(error))
            except Exception as error:
                row.update(status="failed", detail=f"{type(error).__name__}: {error}")
        rows.append(row)
    return {"awkward_version": ak.__version__, "size": size, "coverage": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="execute each mapped case")
    parser.add_argument("--size", type=int, default=32, help="number of fixture rows")
    parser.add_argument("--output", type=Path, help="write JSON to this file")
    args = parser.parse_args(argv)
    if args.size < 1:
        parser.error("--size must be positive")

    import awkward as ak

    report = coverage(ak, size=args.size, smoke=args.smoke)
    content = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(content)
    else:
        args.output.write_text(content, encoding="utf-8")
    return int(
        any(row["status"] in {"missing_case", "failed"} for row in report["coverage"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
