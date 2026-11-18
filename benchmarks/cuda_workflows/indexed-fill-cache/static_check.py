"""CPU-only check of installed-source closure keys; not a GPU/cache-hit test."""

import argparse
import ast
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-source", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    args = parser.parse_args()
    globals_ = dict(
        np=np,
        is_device_array=lambda x: hasattr(x, "__cuda_array_interface__"),
        get_dtype=lambda x: x.dtype,
        get_shape=lambda x: x.shape,
    )
    tree = ast.parse(args.cache_source.read_text())
    selected = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and n.name in ("_type_fqn", "_make_hashable", "CachableFunction")
    ]
    assert len(selected) == 3
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), str(args.cache_source), "exec"
        ),
        globals_,
    )
    key = globals_["CachableFunction"]

    class Array(np.ndarray):
        @property
        def __cuda_array_interface__(self):
            return dict(
                shape=self.shape,
                strides=None,
                typestr=self.dtype.str,
                data=(self.ctypes.data, False),
                version=3,
            )

    callbacks = []

    def transform(**kw):
        callbacks.append(kw["op"])
        for i in range(kw["num_items"]):
            kw["d_out"][i] = kw["op"](i)

    kernel_globals = dict(unary_transform=transform, CountingIterator=lambda x: x)
    tree = ast.parse(args.kernel_source.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "awkward_IndexedArray_fill"
    ]
    assert len(nodes) == 1
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(args.kernel_source), "exec"
        ),
        kernel_globals,
    )
    kernel = kernel_globals["awkward_IndexedArray_fill"]
    src = np.resize(np.array([-1, 0, 2, -3, 5], dtype=np.int64), 128).view(Array)
    dst = np.empty(160, dtype=np.int64).view(Array)

    def invoke(source=src, output=dst, length=64, offset=3, base=10):
        output.fill(123)
        kernel(output, offset, source, length, base)
        expected = np.full(output.size, 123, dtype=output.dtype)
        expected[offset : offset + length] = np.where(
            source[:length] < 0, -1, source[:length] + base
        )
        np.testing.assert_array_equal(output, expected)
        return key(callbacks[-1])

    baseline = invoke()
    for description, kwargs in [
        ("identical", {}),
        ("new input pointer", dict(source=src.copy())),
        ("new output pointer", dict(output=dst.copy())),
        ("length", dict(length=32)),
        ("shape", dict(source=src[:64])),
        ("offset", dict(offset=7)),
    ]:
        assert invoke(**kwargs) == baseline, description
        print("UNCHANGED closure key:", description)
    for description, kwargs in [
        ("input dtype", dict(source=src.astype(np.int32))),
        ("output dtype", dict(output=dst.astype(np.int32))),
        ("base value", dict(base=11)),
        ("base scalar type", dict(base=np.int64(10))),
    ]:
        assert invoke(**kwargs) != baseline, description
        print("CHANGED closure key:", description)
    print(
        "Actual callback CPU output parity passed; "
        "device compilation/build behavior untested here."
    )


if __name__ == "__main__":
    main()
