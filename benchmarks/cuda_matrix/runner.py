# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Isolated CPU, legacy RawKernel, and candidate cuda.compute workers."""

import contextlib
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from catalog import dtype
from operations import cases


class Unavailable(Exception):
    pass


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def encode(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return {"complex": [encode(value.real), encode(value.imag)]}
    if isinstance(value, float) and not np.isfinite(value):
        return {"float": str(value)}
    if isinstance(value, dict):
        return {key: encode(x) for key, x in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(x) for x in value]
    return value


def snapshot(ak, value):
    if isinstance(value, ak.forms.Form):
        return value.to_dict()
    if hasattr(value, "__cuda_array_interface__"):
        import cupy as cp

        value = cp.asnumpy(value)
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": encode(value.tolist()),
        }
    if isinstance(value, dict):
        return {key: snapshot(ak, x) for key, x in value.items()}
    if isinstance(value, (ak.Array, ak.Record)):
        return {
            "type": str(ak.type(value)),
            "values": encode(ak.to_list(ak.to_backend(value, "cpu"))),
        }
    if isinstance(value, (list, tuple)):
        return [snapshot(ak, x) for x in value]
    return encode(value)


def error_record(error, phase):
    return {"type": type(error).__name__, "message": str(error), "phase": phase}


def finish_cuda(cp, stream):
    from awkward._connect import cuda

    if stream.ptr in cuda.cuda_streamptr_to_contexts:
        cuda.synchronize_cuda(stream)
    stream.synchronize()


def completed(call, cp=None, stream=None):
    start = cp.cuda.Event() if cp else None
    end = cp.cuda.Event() if cp else None
    if cp:
        start.record(stream)
    before = time.perf_counter_ns()
    result, pending = None, []
    try:
        result = call()
    except Exception as error:
        pending.append(("call", error))
    submitted = time.perf_counter_ns()
    if cp:
        end.record(stream)
        try:
            finish_cuda(cp, stream)
        except Exception as error:
            pending.append(("finish", error))
        finally:
            end.synchronize()
    stopped = time.perf_counter_ns()
    # Error classification/formatting is outside both timing boundaries.
    errors = [error_record(error, phase) for phase, error in pending]
    if hasattr(result, "filename") and getattr(result, "str", None):
        errors.append(
            {
                "type": "KernelError",
                "message": result.str.decode(errors="replace"),
                "phase": "call",
                "id": int(result.id),
                "attempt": int(result.attempt),
            }
        )
        result = None
    metrics = {
        "host_us": (submitted - before) / 1000,
        "completion_us": (stopped - before) / 1000,
    }
    if cp:
        metrics["stream_us"] = cp.cuda.get_elapsed_time(start, end) * 1000
    return metrics, result, errors


def measure(call, reset, settings, cp=None, stream=None, validate=None):
    # Reset all caller buffers outside timing. A fence prevents a queued reset
    # from being charged to kernel completion/event measurements.
    expected_errors = None
    for _ in range(settings["warmup"]):
        reset()
        if cp:
            finish_cuda(cp, stream)
        _, result, errors = completed(call, cp, stream)
        if validate:
            validate(result, errors)
        del result
        expected_errors = bool(errors)
    samples, stages = [], {"call": 0, "finish": 0}
    for _ in range(settings["repeats"]):
        sample = []
        for _ in range(settings["iterations"]):
            reset()
            if cp:
                finish_cuda(cp, stream)
            metrics, result, errors = completed(call, cp, stream)
            if validate:
                validate(result, errors)
            del result
            if bool(errors) != expected_errors:
                raise AssertionError("Error outcome changed after warmup")
            for error in errors:
                stages[error["phase"]] += 1
            sample.append(metrics)
        samples.append(
            {key: statistics.median(x[key] for x in sample) for key in sample[0]}
        )
    return {
        "samples": samples,
        "median": {
            key: statistics.median(x[key] for x in samples) for key in samples[0]
        },
        "error_stages": stages,
    }


@contextlib.contextmanager
def dispatch(mode, names, observed):
    """Force compute only for registered names; require real RawKernel otherwise.

    Logging is enabled only for a separate correctness/dispatch pass.
    """
    from awkward._backends.backend import Backend
    from awkward._backends.cupy import CupyBackend
    from awkward._kernels import CudaComputeKernel, CupyKernel

    original_new, original_get = CupyBackend._new_kernel, Backend.__getitem__

    def new(self, key):
        if mode == "compute" and key[0] in names:
            impl = self._get_cuda_compute_impl(key[0])
            if impl is None:
                raise Unavailable(f"No compute implementation: {key[0]}")
            return CudaComputeKernel(impl, key)
        try:
            kernel = original_new(self, key)
        except (KeyError, NotImplementedError) as error:
            raise Unavailable(f"No legacy GPU implementation: {key}") from error
        if mode == "raw" and not isinstance(kernel, CupyKernel):
            raise Unavailable(f"Not a RawKernel: {key[0]} ({type(kernel).__name__})")
        return kernel

    patches = []
    if mode == "raw":
        import cuda.compute as cccl

        # Guard both public APIs and aliases already imported by Awkward.
        # Legacy public reducers can call compute directly, bypassing kernel dispatch.
        for module in [
            cccl,
            *(
                m
                for n, m in list(sys.modules.items())
                if n.startswith("awkward._connect.cuda") and m is not None
            ),
        ]:
            for name, value in list(vars(module).items()):
                if callable(value) and getattr(value, "__module__", "").startswith(
                    "cuda.compute"
                ):

                    def forbidden(*args, _name=name, **kwargs):
                        raise Unavailable(
                            f"Direct cuda.compute API on RawKernel path: {_name}"
                        )

                    patches.append((module, name, value))
                    setattr(module, name, forbidden)
    CupyBackend._new_kernel = new
    # Cached wrappers from any input construction must not bypass mode forcing.
    CupyBackend.instance()._kernels.clear()
    try:
        yield
    finally:
        for module, name, value in reversed(patches):
            setattr(module, name, value)
        CupyBackend._new_kernel = original_new
        Backend.__getitem__ = original_get
        CupyBackend.instance()._kernels.clear()


def kernel_case(entry, fixture, mode, cp):
    from awkward._backends.numpy import NumpyBackend
    from awkward_cpp.cpu_kernels import kernel as signatures

    matches = [
        (key, func)
        for key, func in signatures.items()
        if func.__name__ == entry["specialization"]
    ]
    if not matches:
        raise Unavailable("Specialization absent from installed awkward-cpp")
    key, _ = matches[0]
    if mode == "cpu":
        kernel = NumpyBackend.instance()[key]
    else:
        from awkward._backends.cupy import CupyBackend

        kernel = CupyBackend.instance()[key]
    args, reset_pairs, snapshots = [], [], []
    for arg, value in zip(entry["arguments"], fixture["values"], strict=True):
        if "List[List" in arg["type"]:
            pointees = []
            for row in value:
                host = np.array(row, dtype=dtype(arg["type"]))
                array = cp.asarray(host) if cp else host.copy()
                pointees.append(array)
                reset_pairs.append((array, cp.asarray(host) if cp else host))
                snapshots.append(array)
            pointers = np.array(
                [x.data.ptr if cp else x.ctypes.data for x in pointees], dtype=np.uintp
            )
            args.append(cp.asarray(pointers) if cp else pointers)
        elif "List[" in arg["type"]:
            host = np.array(value, dtype=dtype(arg["type"]))
            array = cp.asarray(host) if cp else host.copy()
            args.append(array)
            reset_pairs.append((array, cp.asarray(host) if cp else host))
            snapshots.append(array)
        else:
            args.append(value)

    def reset():
        for array, original in reset_pairs:
            array[...] = original

    return lambda: kernel(*args), reset, snapshots


def environment(checkout, cp):
    packages = {}
    for name in (
        "numpy",
        "awkward",
        "awkward-cpp",
        "cupy-cuda12x",
        "cupy-cuda13x",
        "cuda-cccl",
        "numba-cuda",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_sha": subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"], text=True
            )
        ),
    }
    if cp:
        result.update(
            device=str(
                cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
            ),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        )
    return result


def run(args):
    import awkward as ak
    from catalog import compute_names

    checkout = Path(args.checkout).resolve()
    if checkout not in Path(ak.__file__).resolve().parents:
        raise ValueError("Imported Awkward does not belong to --checkout")
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    args.scope = args.scope or manifest.get("scope", "all")
    cp = None
    if args.mode != "cpu":
        import cupy as cp
    stream = cp.cuda.Stream(non_blocking=True) if cp else None
    output = Path(args.output)
    artifacts = output.parent / (output.stem + "-arrays")
    artifacts.mkdir(parents=True, exist_ok=True)
    settings = {key: getattr(args, key) for key in ("warmup", "iterations", "repeats")}
    report = {
        "schema": 1,
        "complete": False,
        "mode": args.mode,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "settings": dict(settings, scope=args.scope, max_fixtures=args.max_fixtures),
        "environment": environment(checkout, cp),
        "harness_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
        "cases": [],
    }
    kernel_count = sum(
        max(1, min(len(e["fixtures"]), args.max_fixtures))
        if args.max_fixtures
        else max(1, len(e["fixtures"]))
        for e in manifest["kernels"]
    )
    operation_count = (
        len(manifest["operations"])
        * len(manifest["layouts"])
        * len(manifest["dtypes"])
        * len(manifest["shapes"])
    )
    report["expected_case_count"] = (
        kernel_count if args.scope in ("all", "kernels") else 0
    ) + (operation_count if args.scope in ("all", "operations") else 0)
    report["coverage"] = {
        "planned_specializations": len(manifest["kernels"]),
        "planned_kernels": sorted({e["kernel"] for e in manifest["kernels"]}),
        "planned_kernel_fixtures": sum(len(e["fixtures"]) for e in manifest["kernels"]),
        "kernel_fixture_limit": args.max_fixtures,
        "public_operations_without_fixture": manifest.get(
            "public_operations_without_fixture", []
        ),
        "dispatch_without_selected_spec": manifest.get(
            "dispatch_without_selected_spec", []
        ),
    }
    journal = output.with_suffix(".jsonl")
    journal.write_text("")

    def checkpoint(row):
        with journal.open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        if len(report["cases"]) % 100 == 0:
            save(output, report)
        print(row["id"], row["status"], flush=True)

    names = compute_names(checkout)
    # CPU does not import CuPy or mutate its backend.
    manager = (
        dispatch(args.mode, names, set())
        if cp
        else contextlib.nullcontext(lambda enabled: None)
    )
    with stream if cp else contextlib.nullcontext(), manager:
        if args.scope in ("all", "kernels"):
            for entry in manifest["kernels"]:
                fixtures = entry["fixtures"]
                if args.max_fixtures:
                    fixtures = fixtures[: args.max_fixtures]
                if not fixtures:
                    report["cases"].append(
                        {
                            "id": "kernel/" + entry["specialization"],
                            "kernel": entry["kernel"],
                            "status": "missing_fixture",
                        }
                    )
                for fixture in fixtures:
                    row = {"id": "kernel/" + fixture["id"], "kernel": entry["kernel"]}
                    try:
                        call, reset, buffers = kernel_case(
                            entry, fixture, args.mode, cp
                        )
                        reset()
                        if cp:
                            finish_cuda(cp, stream)
                        _, _, errors = completed(call, cp, stream)
                        if any(e["type"] == "Unavailable" for e in errors):
                            raise Unavailable(str(errors))
                        row["errors"] = errors
                        path = artifacts / (
                            hashlib.sha256(row["id"].encode()).hexdigest() + ".npz"
                        )
                        np.savez(
                            path,
                            **{
                                str(i): cp.asnumpy(a) if cp else a.copy()
                                for i, a in enumerate(buffers)
                                if hasattr(a, "dtype")
                            },
                        )
                        row["arrays"] = str(path.resolve())
                        expected_buffers = [
                            cp.asnumpy(a) if cp else a.copy() for a in buffers
                        ]
                        expected_errors = errors

                        def validate_kernel(result, errors):
                            if errors != expected_errors:
                                raise AssertionError(
                                    "Kernel error changed between samples"
                                )
                            for actual, expected in zip(
                                buffers, expected_buffers, strict=True
                            ):
                                np.testing.assert_array_equal(
                                    cp.asnumpy(actual) if cp else actual, expected
                                )

                        row.update(
                            measure(call, reset, settings, cp, stream, validate_kernel)
                        )
                        row["status"] = "measured"
                    except Unavailable as error:
                        row.update(status="unavailable", reason=str(error))
                    except Exception as error:
                        row.update(
                            status="failed",
                            error=error_record(error, "setup_or_measure"),
                        )
                    report["cases"].append(row)
                    checkpoint(row)
        if args.scope in ("all", "operations"):
            operations = manifest["operations"]
            for layout in manifest["layouts"]:
                for dtype_name in manifest["dtypes"]:
                    for rows, width in manifest["shapes"]:
                        try:
                            calls = list(
                                cases(
                                    ak,
                                    layout,
                                    dtype_name,
                                    rows,
                                    width,
                                    "cuda" if cp else "cpu",
                                    operations,
                                )
                            )
                        except Exception as error:
                            calls = [(name, None) for name in operations]
                            setup_error = error_record(error, "fixture")
                        for name, call in calls:
                            row = {
                                "id": f"operation/{name}/{layout}/{dtype_name}/{rows}:{width}"
                            }
                            observed = set()
                            try:
                                if call is None:
                                    raise Unavailable(str(setup_error))
                                # Observe actual calls, not just kernel lookups.
                                from awkward._backends.backend import Backend

                                original_get = Backend.__getitem__

                                def get(self, key):
                                    kernel = original_get(self, key)

                                    def invoke(*values):
                                        observed.add((key[0], type(kernel).__name__))
                                        return kernel(*values)

                                    return invoke

                                Backend.__getitem__ = get
                                try:
                                    _, result, errors = completed(call, cp, stream)
                                finally:
                                    Backend.__getitem__ = original_get
                                if any(e["type"] == "Unavailable" for e in errors):
                                    raise Unavailable(str(errors))
                                row["dispatch"] = sorted(observed)
                                row["errors"] = errors
                                row["result"] = (
                                    snapshot(ak, result) if not errors else None
                                )
                                expected_errors, expected_result = errors, row["result"]

                                def validate_public(result, errors):
                                    if errors != expected_errors:
                                        raise AssertionError(
                                            "Public error changed between samples"
                                        )
                                    if (
                                        not errors
                                        and snapshot(ak, result) != expected_result
                                    ):
                                        raise AssertionError(
                                            "Public result changed between samples"
                                        )

                                del result
                                row.update(
                                    measure(
                                        call,
                                        lambda: None,
                                        settings,
                                        cp,
                                        stream,
                                        validate_public,
                                    )
                                )
                                row["status"] = "measured"
                            except Unavailable as error:
                                row.update(status="unavailable", reason=str(error))
                            except Exception as error:
                                row.update(
                                    status="failed",
                                    error=error_record(error, "setup_or_measure"),
                                )
                            report["cases"].append(row)
                            checkpoint(row)
    report["complete"] = True
    save(output, report)
