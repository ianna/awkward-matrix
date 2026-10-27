# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Public-operation catalogue spanning layout families and numeric/string paths.

Each fixture is constructed once, including all GPU transfers and index arrays.
Operation dispatch is observed separately from timing; missing coverage is explicit.
"""

import numpy as np

NAMES = (
    "slice_at",
    "slice_range",
    "slice_array",
    "slice_jagged",
    "slice_advanced",
    "slice_spread",
    "slice_bounds",
    "sum",
    "prod",
    "min",
    "max",
    "argmin",
    "argmax",
    "count",
    "count_nonzero",
    "any",
    "all",
    "mean",
    "var",
    "std",
    "moment",
    "sort",
    "argsort",
    "num",
    "flatten",
    "ravel",
    "local_index",
    "pad_none",
    "pad_clip",
    "drop_none",
    "fill_none",
    "is_none",
    "to_regular",
    "from_regular",
    "concatenate",
    "broadcast_arrays",
    "combinations",
    "cartesian",
    "where",
    "mask",
    "zip",
    "unzip",
    "is_valid",
    "validity_error",
    "to_packed",
    "run_lengths",
    "firsts",
    "singletons",
    "nan_to_num",
    "values_astype",
    "strings_length",
    "strings_upper",
    "strings_sort",
    "strings_argsort",
)
NAMES += (
    "angle",
    "argcartesian",
    "argcombinations",
    "almost_equal",
    "array_equal",
    "broadcast_fields",
    "categories",
    "from_categorical",
    "is_categorical",
    "copy",
    "corr",
    "covar",
    "linear_fit",
    "ptp",
    "real",
    "imag",
    "round",
    "softmax",
    "nan_to_none",
    "full_like",
    "zeros_like",
    "ones_like",
    "isclose",
    "unflatten",
    "with_field",
    "without_field",
    "with_name",
    "with_parameter",
    "without_parameters",
    "zip_no_broadcast",
    "merge_option_of_records",
    "merge_union_of_records",
    "transform",
    "to_buffers",
    "from_buffers",
)

REDUCERS = (
    "sum",
    "prod",
    "min",
    "max",
    "argmin",
    "argmax",
    "count",
    "count_nonzero",
    "any",
    "all",
    "mean",
    "var",
    "std",
)
NAMES += tuple(
    f"{name}_{axis}" for name in REDUCERS for axis in ("axis0", "axis1", "all")
)
NAMES += ("sort_unstable", "argsort_unstable", "sort_descending", "argsort_descending")

LAYOUTS = (
    "listoffset",
    "list",
    "regular",
    "indexed",
    "option",
    "byte_mask",
    "bit_mask",
    "union",
    "record",
    "ragged",
    "empty_lists",
    "nonzero_offsets",
    "categorical",
)
DTYPES = ("int64", "float32", "float64", "complex128", "bool")


def fixture(ak, layout, dtype, rows, width, backend):
    values = np.arange(rows * width * 3).reshape(rows * width, 3) % 11 + 1
    values = values.astype(dtype)
    if dtype == "complex128":
        values += 1j * (values.real % 3)
    leaf = ak.contents.RegularArray(ak.contents.NumpyArray(values.reshape(-1)), 3)
    offsets = np.arange(rows + 1, dtype=np.int64) * width
    if layout == "ragged":
        offsets = np.concatenate(
            ([0], np.cumsum(np.arange(rows) % (width + 1)))
        ).astype(np.int64)
    elif layout == "empty_lists":
        offsets = np.zeros(rows + 1, dtype=np.int64)
    elif layout == "nonzero_offsets":
        offsets = offsets + 1
        leaf = ak.contents.RegularArray(
            ak.contents.NumpyArray(
                np.concatenate((np.zeros(3, dtype=dtype), values.reshape(-1)))
            ),
            3,
        )
    base = ak.contents.ListOffsetArray(ak.index.Index64(offsets), leaf)
    if layout == "categorical":
        base = ak.contents.IndexedArray(
            ak.index.Index64(np.arange(rows, dtype=np.int64)),
            base,
            parameters={"__array__": "categorical"},
        )
    if layout == "list":
        base = ak.contents.ListArray(
            ak.index.Index64(offsets[:-1]), ak.index.Index64(offsets[1:]), leaf
        )
    elif layout == "regular":
        base = ak.contents.RegularArray(leaf, width)
    elif layout == "indexed":
        base = ak.contents.IndexedArray(
            ak.index.Index64(np.arange(rows)[::-1].copy()), base
        )
    elif layout == "option":
        index = np.arange(rows, dtype=np.int64)
        index[::3] = -1
        base = ak.contents.IndexedOptionArray(ak.index.Index64(index), base)
    elif layout == "byte_mask":
        base = ak.contents.ByteMaskedArray(
            ak.index.Index8((np.arange(rows) % 3 != 0).astype(np.int8)),
            base,
            valid_when=True,
        )
    elif layout == "bit_mask":
        bits = np.packbits(
            (np.arange(rows) % 3 != 0).astype(np.uint8), bitorder="little"
        )
        base = ak.contents.BitMaskedArray(
            ak.index.IndexU8(bits), base, valid_when=True, length=rows, lsb_order=True
        )
    elif layout == "union":
        # Distinct types prevent eager union simplification from erasing coverage.
        other = ak.contents.NumpyArray(np.arange(rows, dtype=np.int64))
        base = ak.contents.UnionArray(
            ak.index.Index8((np.arange(rows) % 2).astype(np.int8)),
            ak.index.Index64(np.arange(rows, dtype=np.int64)),
            [base, other],
        )
    elif layout == "record":
        base = ak.contents.RecordArray([base, base], ["x", "y"])
    return ak.to_backend(ak.Array(base), backend)


def cases(ak, layout, dtype, rows, width, backend, names=NAMES):
    array = fixture(ak, layout, dtype, rows, width, backend)
    index = ak.to_backend(ak.Array(np.arange(rows, dtype=np.int64)), backend)
    zeros = ak.to_backend(ak.Array(np.zeros(rows, dtype=np.int64)), backend)
    bad = ak.to_backend(ak.Array(np.full(rows, width, dtype=np.int64)), backend)
    jagged = ak.to_backend(ak.Array([[0, -1] for _ in range(width)]), backend)
    condition = ak.to_backend(ak.Array(np.arange(rows) % 2 == 0), backend)
    strings = (
        ak.to_backend(
            ak.Array(["a", "", "é", "a\x00b"] * ((rows + 3) // 4))[:rows], backend
        )
        if any(n.startswith("strings_") for n in names)
        else None
    )
    # Prepare bounded combinatorial inputs on CPU. An unrelated GPU slice in
    # fixture construction must not make other RawKernel workloads unavailable.
    bounded = array
    if any(
        name in ("cartesian", "combinations", "argcartesian", "argcombinations")
        for name in names
    ):
        host = fixture(ak, layout, dtype, rows, width, "cpu")
        if layout == "record":
            host = ak.zip(
                {field: host[field][:, :8] for field in ak.fields(host)}, depth_limit=1
            )
        elif layout != "union":
            host = host[:, :8]
        bounded = ak.to_backend(host, backend)
    form, length, buffers = ak.to_buffers(array)
    for name in names:
        reduction, _, axis_name = name.rpartition("_")
        if reduction in REDUCERS and axis_name in ("axis0", "axis1", "all"):
            axis = {"axis0": 0, "axis1": 1, "all": None}[axis_name]
            yield (
                name,
                lambda reduction=reduction, axis=axis: getattr(ak, reduction)(
                    array, axis=axis
                ),
            )
            continue
        if name in (
            "sort_unstable",
            "argsort_unstable",
            "sort_descending",
            "argsort_descending",
        ):
            operation, variant = name.split("_")
            yield (
                name,
                lambda operation=operation, variant=variant: getattr(ak, operation)(
                    array,
                    axis=-1,
                    stable=variant != "unstable",
                    ascending=variant != "descending",
                ),
            )
            continue
        if name in (
            "sum",
            "prod",
            "min",
            "max",
            "argmin",
            "argmax",
            "count",
            "count_nonzero",
            "any",
            "all",
            "mean",
            "var",
            "std",
        ):
            yield name, lambda name=name: getattr(ak, name)(array, axis=-1)
        elif name in ("sort", "argsort"):
            yield name, lambda name=name: getattr(ak, name)(array, axis=-1, stable=True)
        else:
            calls = {
                "angle": lambda: ak.angle(array),
                "argcartesian": lambda: ak.argcartesian([bounded, bounded], axis=1),
                "argcombinations": lambda: ak.argcombinations(bounded, 2, axis=1),
                "almost_equal": lambda: ak.almost_equal(array, array),
                "array_equal": lambda: ak.array_equal(array, array),
                "broadcast_fields": lambda: ak.broadcast_fields(array, array),
                "categories": lambda: ak.categories(array),
                "from_categorical": lambda: ak.from_categorical(array),
                "is_categorical": lambda: ak.is_categorical(array),
                "copy": lambda: ak.copy(array),
                "corr": lambda: ak.corr(array, array, axis=-1),
                "covar": lambda: ak.covar(array, array, axis=-1),
                "linear_fit": lambda: ak.linear_fit(array, array, axis=-1),
                "ptp": lambda: ak.ptp(array, axis=-1),
                "real": lambda: ak.real(array),
                "imag": lambda: ak.imag(array),
                "round": lambda: ak.round(array),
                "softmax": lambda: ak.softmax(array, axis=-1),
                "nan_to_none": lambda: ak.nan_to_none(array),
                "full_like": lambda: ak.full_like(array, 2),
                "zeros_like": lambda: ak.zeros_like(array),
                "ones_like": lambda: ak.ones_like(array),
                "isclose": lambda: ak.isclose(array, array),
                "unflatten": lambda: ak.unflatten(
                    ak.flatten(array, axis=1), width, axis=0
                ),
                "with_field": lambda: ak.with_field(ak.zip({"x": array}), array, "y"),
                "without_field": lambda: ak.without_field(
                    ak.zip({"x": array, "y": array}), "y"
                ),
                "with_name": lambda: ak.with_name(ak.zip({"x": array}), "Matrix"),
                "with_parameter": lambda: ak.with_parameter(array, "matrix", True),
                "without_parameters": lambda: ak.without_parameters(array),
                "zip_no_broadcast": lambda: ak.zip_no_broadcast(
                    {"x": array, "y": array}
                ),
                "merge_option_of_records": lambda: ak.merge_option_of_records(array),
                "merge_union_of_records": lambda: ak.merge_union_of_records(array),
                "transform": lambda: ak.transform(lambda layout, **kwargs: None, array),
                "to_buffers": lambda: ak.to_buffers(array),
                "from_buffers": lambda: ak.from_buffers(
                    form, length, buffers, backend=backend
                ),
                "slice_at": lambda: array[:, 0],
                "slice_range": lambda: array[:, ::2],
                "slice_array": lambda: array[:, index[:1]],
                "slice_jagged": lambda: array[:, jagged],
                "slice_advanced": lambda: array[index, zeros, zeros],
                "slice_spread": lambda: array[index, :],
                "slice_bounds": lambda: array[index, bad, zeros],
                "moment": lambda: ak.moment(array, 3, axis=-1),
                "num": lambda: ak.num(array, axis=1),
                "flatten": lambda: ak.flatten(array, axis=1),
                "ravel": lambda: ak.ravel(array),
                "local_index": lambda: ak.local_index(array, axis=1),
                "pad_none": lambda: ak.pad_none(array, width + 2, axis=1),
                "pad_clip": lambda: ak.pad_none(
                    array, max(1, width - 1), axis=1, clip=True
                ),
                "drop_none": lambda: ak.drop_none(array, axis=0),
                "fill_none": lambda: ak.fill_none(array, 0, axis=0),
                "is_none": lambda: ak.is_none(array, axis=0),
                "to_regular": lambda: ak.to_regular(array, axis=1),
                "from_regular": lambda: ak.from_regular(array, axis=-1),
                "concatenate": lambda: ak.concatenate([array, array], axis=0),
                "broadcast_arrays": lambda: ak.broadcast_arrays(array, array),
                "combinations": lambda: ak.combinations(bounded, 2, axis=1),
                "cartesian": lambda: ak.cartesian([bounded, bounded], axis=1),
                "where": lambda: ak.where(condition, array, array),
                "mask": lambda: ak.mask(array, condition),
                "zip": lambda: ak.zip({"x": array, "y": array}),
                "unzip": lambda: ak.unzip(ak.zip({"x": array, "y": array})),
                "is_valid": lambda: ak.is_valid(array),
                "validity_error": lambda: ak.validity_error(array),
                "to_packed": lambda: ak.to_packed(array),
                "run_lengths": lambda: ak.run_lengths(array),
                "firsts": lambda: ak.firsts(array, axis=1),
                "singletons": lambda: ak.singletons(array, axis=0),
                "nan_to_num": lambda: ak.nan_to_num(array),
                "values_astype": lambda: ak.values_astype(array, np.float64),
                "strings_length": lambda: ak.num(strings, axis=-1),
                "strings_upper": lambda: ak.str.upper(strings),
                "strings_sort": lambda: ak.sort(strings),
                "strings_argsort": lambda: ak.argsort(strings),
            }
            yield name, calls[name]
