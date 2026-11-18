"""Diagnostic-only hooks for the pinned scalar_sums block; no source edits."""

import ast
import cProfile
import importlib.abc
import importlib.machinery
import json
import os
import pstats
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LABELS = (
    "mass_l0l1",
    "dr_l0l1",
    "scalarptsum_lep",
    "scalarptsum_lepmet",
    "scalarptsum_lepmetFJ0",
    "scalarptsum_lepmetFJ01",
    "scalarptsum_lepmetalljets",
    "scalarptsum_lepmetcentjets",
    "scalarptsum_lepmetfwdjets",
    "scalarptsum_lepmetvbsFJ0",
    "vectorsum_lepmetvbsFJ0_pt",
)
PHASE = os.environ.get("DIAGNOSTIC_PHASE", "scalar_sums")
assert PHASE in ("scalar_sums", "obj_basic_selection")
PREFIX = "basic" if PHASE == "obj_basic_selection" else "scalar"
_ordinal = 0
_active = False
_installed = False
_rows = []


def instrument(source):
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AnalysisProcessor"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process"
    )

    def label(n):
        return (
            n.targets[0].id
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            else None
        )

    if PHASE == "obj_basic_selection":

        def marker(node, name):
            return isinstance(node, ast.Expr) and any(
                isinstance(part, ast.Constant)
                and isinstance(part.value, str)
                and f"CHUNK_TIMING phase={name} seconds=" in part.value
                for part in ast.walk(node)
            )

        start = (
            next(i for i, n in enumerate(method.body) if marker(n, "data_load_to_gpu"))
            + 1
        )
        stop = next(
            i for i, n in enumerate(method.body) if marker(n, "obj_basic_selection")
        )
        block = method.body[start:stop]
        assert len(block) > 30, "Pinned object-selection block changed"
        labels = tuple(
            f"line_{n.lineno}: " + ast.unparse(n).splitlines()[0][:180] for n in block
        )
    else:
        start = next(i for i, n in enumerate(method.body) if label(n) == LABELS[0])
        block = method.body[start : start + len(LABELS)]
        assert tuple(label(n) for n in block) == LABELS, (
            "Pinned scalar_sums block changed"
        )
        labels = LABELS
    lines = source.splitlines(keepends=True)
    indent = " " * block[0].col_offset
    replacement = [
        indent + "from scalar_diagnostics import region as _scalar_region\n",
        indent + 'with _scalar_region("__total__"):\n',
    ]
    for name, statement in zip(labels, block, strict=True):
        replacement.append(indent + f"    with _scalar_region({name!r}):\n")
        for line in lines[statement.lineno - 1 : statement.end_lineno]:
            replacement.append("        " + line)
    return "".join(
        lines[: block[0].lineno - 1] + replacement + lines[block[-1].end_lineno :]
    )


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode("utf-8")
        patched = instrument(source)
        target = (
            Path(os.environ["PERSISTENT_RESULTS"])
            / f"instrumented_{PREFIX}_processor.py"
        )
        target.write_text(patched)
        return compile(patched, str(target), "exec")


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "analysis_processor_semilep":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec and spec.origin, "Cannot locate pinned processor"
        spec.loader = Loader(fullname, spec.origin)
        return spec


def install():
    global _installed
    if not _installed:
        assert "analysis_processor_semilep" not in sys.modules, (
            "Processor loaded before diagnostic hook"
        )
        sys.meta_path.insert(0, Finder())
        _installed = True


def chunk_start(info):
    global _ordinal, _active, _rows
    _ordinal += 1
    _active = _ordinal == 2
    _rows = []
    if _active:
        (Path(os.environ["PERSISTENT_RESULTS"]) / f"{PREFIX}-chunk.json").write_text(
            json.dumps(dict(info, pid=os.getpid(), ordinal=_ordinal), indent=2)
        )


def chunk_end():
    global _active
    _active = False


def cuda_module():
    if os.environ.get("EWKCOFFEA_BACKEND") == "cuda":
        import cupy as cp

        return cp
    return None


def finish(cp):
    if cp is not None:
        from awkward._connect import cuda

        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        if stream.ptr in cuda.cuda_streamptr_to_contexts:
            cuda.synchronize_cuda(stream)


@contextmanager
def region(name):
    if not _active:
        yield
        return
    cp = cuda_module()
    mode = os.environ["SCALAR_DIAGNOSTICS"]
    outer = name == "__total__"
    trace = mode == "trace" and cp is not None
    profile = cProfile.Profile() if outer and mode == "host" else None
    if outer:
        finish(cp)
        if trace:
            cp.cuda.runtime.profilerStart()
    if trace:
        cp.cuda.nvtx.RangePush(PHASE + "/" + name)
    if profile:
        profile.enable()
    start = time.perf_counter()
    try:
        yield
    finally:
        if outer:
            finish(cp)
        elapsed = time.perf_counter() - start
        if profile:
            profile.disable()
        if trace:
            cp.cuda.nvtx.RangePop()
            if outer:
                cp.cuda.runtime.profilerStop()
        _rows.append(dict(statement=name, host_s=elapsed))
        if outer:
            root = Path(os.environ["PERSISTENT_RESULTS"])
            (root / f"{PREFIX}-statements.json").write_text(json.dumps(_rows, indent=2))
            if profile:
                profile.dump_stats(str(root / f"{PREFIX}.pstats"))
                with (root / f"{PREFIX}-profile.txt").open("w") as stream:
                    stats = pstats.Stats(profile, stream=stream)
                    stats.sort_stats("cumulative").print_stats(100)
                    stats.sort_stats("tottime").print_stats(100)
                    for pattern in ("make_", "call_build", "compile", "item", "get"):
                        stats.sort_stats("cumulative").print_callers(pattern)
                rows = []
                for (f, line, function), (
                    primitive,
                    calls,
                    own,
                    total,
                    _,
                ) in pstats.Stats(profile).stats.items():
                    rows.append(
                        dict(
                            file=f,
                            line=line,
                            function=function,
                            calls=calls,
                            primitive_calls=primitive,
                            self_s=own,
                            cumulative_s=total,
                        )
                    )
                (root / f"{PREFIX}-functions.json").write_text(
                    json.dumps(rows, indent=2)
                )
                groups = dict(
                    compilation=[],
                    construction=[],
                    extraction_transfer=[],
                    formatting=[],
                    synchronization=[],
                    reductions=[],
                )
                for row in rows:
                    f, fun = row["file"].lower(), row["function"].lower()
                    if (
                        "compil" in fun
                        or "compile" in f
                        or ("_jit" in f and fun == "compile")
                    ):
                        groups["compilation"].append(row)
                    if ("cuda/compute" in f or "compute/_bindings" in f) and (
                        fun.startswith("make_") or fun == "__init__"
                    ):
                        groups["construction"].append(row)
                    if any(
                        word in fun
                        for word in ("asnumpy", "'get'", "'item'", "to_backend")
                    ):
                        groups["extraction_transfer"].append(row)
                    if "synchroniz" in fun:
                        groups["synchronization"].append(row)
                    if "_reduce" in f or "reduce" in fun or "'sum'" in fun:
                        groups["reductions"].append(row)
                    if "prettyprint" in f or "_errors.py" in f:
                        groups["formatting"].append(row)
                with (root / f"{PREFIX}-focus.txt").open("w") as stream:
                    stream.write(
                        "Diagnostic cumulative times overlap; do not sum rows.\n"
                    )
                    for group, values in groups.items():
                        stream.write("\n" + group + "\n")
                        for row in sorted(
                            values, key=lambda r: r["cumulative_s"], reverse=True
                        )[:25]:
                            stream.write(json.dumps(row) + "\n")
