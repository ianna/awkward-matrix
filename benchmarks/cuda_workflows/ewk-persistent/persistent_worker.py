"""One spawned process, no worker recycling; cloudpickle preserves Coffea callables."""

import json
import multiprocessing
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def record(event):
    event["pid"] = os.getpid()
    event["at"] = time.time()
    root = Path(os.environ["PERSISTENT_RESULTS"])
    with (root / f"worker-{os.getpid()}.jsonl").open("a") as output:
        output.write(json.dumps(event) + "\n")


def finish():
    if os.environ.get("EWKCOFFEA_BACKEND") == "cuda":
        import cupy as cp
        from awkward._connect import cuda

        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        if stream.ptr in cuda.cuda_streamptr_to_contexts:
            cuda.synchronize_cuda(stream)


def memory():
    values = {"peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                values["rss_kib"] = int(line.split()[1])
    if os.environ.get("EWKCOFFEA_BACKEND") == "cuda":
        import cupy as cp

        values.update(
            pool_used=cp.get_default_memory_pool().used_bytes(),
            pool_reserved=cp.get_default_memory_pool().total_bytes(),
        )
    return values


def work_item(args):
    for item in args:
        if all(
            hasattr(item, field) for field in ("entrystart", "entrystop", "filename")
        ):
            return item
        if isinstance(item, (list, tuple)):
            found = work_item(item)
            if found is not None:
                return found
    return None


class OutputTrace:
    def __init__(self, stream, chunk):
        self.stream, self.chunk, self.pending = stream, chunk, ""

    def write(self, text):
        self.stream.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if "CHUNK_TIMING" in line:
                record(dict(event="upstream_phase", chunk=self.chunk, line=line))
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def execute(payload):
    import cloudpickle

    diagnostic = None
    if os.environ.get("SCALAR_DIAGNOSTICS"):
        import scalar_diagnostics as diagnostic

        diagnostic.install()
    function, args, kwargs = cloudpickle.loads(payload)
    item = work_item(args)
    if item is None:
        return function(*args, **kwargs)
    info = dict(
        filename=str(item.filename),
        start=int(item.entrystart),
        stop=int(item.entrystop),
        dataset=str(item.dataset),
    )
    finish()
    before = memory()
    record(dict(event="chunk_start", **info, memory=before))
    original_stdout = sys.stdout
    sys.stdout = OutputTrace(original_stdout, [info["start"], info["stop"]])
    if diagnostic is not None:
        diagnostic.chunk_start(info)
    start = time.perf_counter()
    try:
        result = function(*args, **kwargs)
        finish()
        elapsed = time.perf_counter() - start
    except BaseException as error:
        record(dict(event="chunk_failure", **info, error=repr(error)))
        raise
    finally:
        if diagnostic is not None:
            diagnostic.chunk_end()
        sys.stdout = original_stdout
    record(dict(event="chunk_end", **info, seconds=elapsed, memory=memory()))
    return result


class PersistentPool(ProcessPoolExecutor):
    def __init__(self, max_workers=1, **kwargs):
        if max_workers != 1:
            raise ValueError("This experiment requires exactly one worker")
        super().__init__(max_workers=1, mp_context=multiprocessing.get_context("spawn"))

    def submit(self, function, /, *args, **kwargs):
        import cloudpickle

        return super().submit(execute, cloudpickle.dumps((function, args, kwargs)))
