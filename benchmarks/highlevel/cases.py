"""Workloads for the public Awkward API."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from functools import cached_property, partial
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Callable


class Unavailable(RuntimeError):
    """A workload requires an optional package or runtime."""


@dataclass(frozen=True)
class Case:
    prepare: Callable
    requires: tuple[str, ...] = ()
    cuda: bool = False

    def bind(self, fixtures, target):
        for name in self.requires:
            try:
                import_module(name)
            except ImportError as error:
                raise Unavailable(f"{name}: {error}") from error
        if self.cuda:
            import cupy

            try:
                device_count = cupy.cuda.runtime.getDeviceCount()
            except cupy.cuda.runtime.CUDARuntimeError as error:
                raise Unavailable(f"CUDA: {error}") from error
            if device_count == 0:
                raise Unavailable("No CUDA device")
        return self.prepare(fixtures, target)


@dataclass
class Fixtures:
    ak: object
    size: int
    directory: Path

    @cached_property
    def np(self):
        return import_module("numpy")

    @cached_property
    def rows(self):
        return [
            [float((i + j) % 11) for j in range(i % 5 + 2)] for i in range(self.size)
        ]

    @cached_property
    def ragged(self):
        return self.ak.Array(self.rows)

    @cached_property
    def other(self):
        return self.ragged * 2 + 1

    @cached_property
    def optional(self):
        return self.ak.Array([[None, *row] for row in self.rows])

    @cached_property
    def nan(self):
        return self.ak.Array([[float("nan"), *row] for row in self.rows])

    @cached_property
    def dense(self):
        return self.np.arange(self.size * 4, dtype="int64").reshape(-1, 4)

    @cached_property
    def regular(self):
        return self.ak.Array(self.dense)

    @cached_property
    def records(self):
        return self.ak.Array({"x": self.ragged, "y": self.other})

    @cached_property
    def record(self):
        return self.records[0]

    @cached_property
    def strings(self):
        samples = ("Abc-12", " café ", "abc-34", "42", "Two words")
        return self.ak.Array([samples[i % len(samples)] for i in range(self.size)])

    @cached_property
    def categorical(self):
        categories = self.ak.Array(["abc", "42", "café"])
        index = self.ak.index.Index64(self.np.arange(self.size) % len(categories))
        layout = self.ak.contents.IndexedArray(
            index, categories.layout, parameters={"__array__": "categorical"}
        )
        return self.ak.Array(layout)

    @cached_property
    def builder(self):
        builder = self.ak.ArrayBuilder()
        builder.extend(self.dense[:, 0].tolist())
        return builder


def call(arguments, **options):
    """Resolve fixtures during setup and bind the public operation."""
    return Case(lambda f, fn: partial(fn, *arguments(f), **options))


def unary(fixture, names, requires=(), **options):
    def prepare(f, fn):
        return partial(fn, getattr(f, fixture), **options)

    return {name: Case(prepare, requires) for name in names.split()}


def pair(names, **options):
    case = call(lambda f: (f.ragged, f.other), **options)
    return dict.fromkeys(names.split(), case)


def mixin_class(f, fn):
    class Point:
        pass

    return lambda: fn({})(type("Point", (Point,), {}))


def mixin_class_method(f, fn):
    def add(self, other):
        return self.x + other.x

    return lambda: fn(f.np.add, {"Point"})(add)


def materialize(f, fn):
    source = f.np.arange(f.size)
    form = f.ak.forms.NumpyForm("int64", form_key="data")

    options = {}
    if "enable_virtualarray_caching" in signature(f.ak.from_buffers).parameters:
        options["enable_virtualarray_caching"] = False
    array = f.ak.from_buffers(form, f.size, {"data-data": lambda: source}, **options)
    return partial(fn, array)


OPERATIONS = {
    **unary(
        "ragged",
        """
        argsort backend copy flatten is_valid local_index num ones_like
        ravel round singletons sort to_buffers to_json to_layout to_list to_packed
        type validity_error zeros_like
    """,
    ),
    **unary(
        "ragged",
        """
        all any argmax argmin count count_nonzero max mean min prod ptp softmax
        std sum var sumofsquares centered_sumofsquares
    """,
        axis=-1,
    ),
    **unary(
        "nan",
        """
        nanargmax nanargmin nanmax nanmean nanmin nanprod nanstd nansum nanvar
    """,
        axis=-1,
    ),
    **unary("nan", "nan_to_none nan_to_num"),
    **unary("optional", "drop_none is_none", axis=-1),
    **unary("records", "fields parameters is_tuple unzip"),
    **unary("regular", "from_regular to_numpy"),
    **unary("dense", "from_numpy from_dlpack"),
    **unary("categorical", "categories from_categorical is_categorical"),
    **pair("almost_equal array_equal broadcast_arrays isclose"),
    **pair("corr covar linear_fit", axis=-1),
    "broadcast_fields": call(lambda f: (f.records, f.ak.Array({"z": f.other}))),
    "angle": call(lambda f: (f.ragged + 1j * f.other,)),
    "imag": call(lambda f: (f.ragged + 1j * f.other,)),
    "real": call(lambda f: (f.ragged + 1j * f.other,)),
    "argcartesian": call(lambda f: ([f.ragged, f.other],)),
    "cartesian": call(lambda f: ([f.ragged, f.other],)),
    "concatenate": call(lambda f: ([f.ragged, f.other],), axis=1),
    "argcombinations": call(lambda f: (f.ragged, 2)),
    "combinations": call(lambda f: (f.ragged, 2)),
    "enforce_type": call(lambda f: (f.ragged, "var * float32")),
    "fill_none": call(lambda f: (f.optional, 0)),
    "firsts": call(lambda f: (f.optional,)),
    "from_buffers": call(lambda f: f.ak.to_buffers(f.ragged)),
    "from_iter": call(lambda f: (f.rows,)),
    "from_json": call(lambda f: (json.dumps(f.rows),)),
    "full_like": call(lambda f: (f.ragged, 7)),
    "mask": call(lambda f: (f.ragged, f.ragged > 3)),
    "materialize": Case(materialize),
    "merge_option_of_records": call(
        lambda f: (f.ak.mask(f.records, f.np.arange(f.size) % 3 != 1),), axis=0
    ),
    "merge_union_of_records": call(
        lambda f: (f.ak.concatenate([f.records, f.ak.Array({"z": f.other})]),),
        axis=0,
    ),
    "mixin_class": Case(mixin_class),
    "mixin_class_method": Case(mixin_class_method),
    "moment": call(lambda f: (f.ragged, 2), axis=-1),
    "pad_none": call(lambda f: (f.ragged, 6)),
    "run_lengths": call(lambda f: (f.ak.Array(f.np.arange(f.size) // 3),)),
    "strings_astype": call(lambda f: (f.ak.Array(["123"] * f.size), "int64")),
    "sumofpowers": call(lambda f: (f.ragged, 3), axis=-1),
    "to_categorical": call(lambda f: (f.strings,)),
    "to_backend": call(lambda f: (f.ragged, "typetracer")),
    "to_regular": call(lambda f: (f.ak.Array(f.dense.tolist()),)),
    "transform": call(lambda f: (lambda layout, **kwargs: None, f.ragged)),
    "unflatten": call(lambda f: (f.ak.flatten(f.ragged), f.ak.num(f.ragged))),
    "values_astype": call(lambda f: (f.ragged, "float32")),
    "where": call(lambda f: (f.ragged > 3, f.ragged, f.other)),
    "with_field": call(lambda f: (f.records, f.other, "z")),
    "with_name": call(lambda f: (f.records, "Point")),
    "with_named_axis": call(lambda f: (f.ragged, ("row", "value"))),
    "with_parameter": call(lambda f: (f.ragged, "fixture", "ragged")),
    "without_field": call(lambda f: (f.records, "y")),
    "without_named_axis": call(
        lambda f: (f.ak.with_named_axis(f.ragged, ("row", "value")),)
    ),
    "without_parameters": call(
        lambda f: (f.ak.with_parameter(f.ragged, "fixture", "ragged"),)
    ),
    "zip": call(lambda f: ({"x": f.ragged, "y": f.other},)),
    "zip_no_broadcast": call(lambda f: ({"x": f.ragged, "y": f.other},)),
}


def parquet_dataset(f, fn):
    f.ak.to_parquet(
        f.records, f.directory / "part.parquet", parquet_compliant_nested=True
    )
    return partial(fn, f.directory)


def parquet_row_groups(f, fn):
    arrays = (f.records, f.records)
    return lambda: fn(iter(arrays), f.directory / "groups.parquet")


def avro_file(f, fn):
    import fastavro

    path = f.directory / "records.avro"
    schema = {
        "type": "record",
        "name": "Point",
        "fields": [{"name": "x", "type": {"type": "array", "items": "double"}}],
    }
    with path.open("wb") as stream:
        fastavro.writer(stream, schema, ({"x": row} for row in f.rows))
    return partial(fn, str(path))


def read_file(format_name):
    def prepare(f, fn):
        path = f.directory / f"input.{format_name}"
        getattr(f.ak, f"to_{format_name}")(f.records, path)
        return partial(fn, path)

    return prepare


def write_file(format_name):
    def prepare(f, fn):
        return partial(fn, f.records, f.directory / f"output.{format_name}")

    return prepare


def external_input(f, fn):
    import pyarrow

    return partial(fn, pyarrow.array(f.rows))


def arrow_schema(f, fn):
    import pyarrow

    schema = pyarrow.schema([("x", pyarrow.list_(pyarrow.float64()))])
    return partial(fn, schema)


def device_conversion(module, method, *, ragged=False):
    def prepare(f, fn):
        convert = getattr(import_module(module), method)
        return partial(fn, convert(f.rows if ragged else f.dense))

    return prepare


def jax_conversion(*, into_awkward):
    def prepare(f, fn):
        jax = import_module("jax")
        jax.config.update("jax_enable_x64", True)
        f.ak.jax.register_and_check()
        source = jax.numpy.asarray(f.dense) if into_awkward else f.regular
        return partial(fn, source)

    return Case(prepare, ("jax",))


def root_input(f, fn):
    import ROOT

    frame = ROOT.RDF.FromNumpy({"x": f.dense[:, 0].copy()})
    return partial(fn, frame, ["x"])


OPERATIONS.update(
    {
        **unary("ragged", "to_arrow", ("pyarrow",)),
        **unary("records", "to_arrow_table", ("pyarrow",)),
        **unary("records", "to_dataframe", ("pandas",)),
        "to_jax": jax_conversion(into_awkward=False),
        **unary("regular", "to_tensorflow", ("tensorflow",)),
        **unary("regular", "to_torch", ("torch",)),
        **unary("ragged", "to_raggedtensor", ("tensorflow",)),
        "from_arrow": Case(external_input, ("pyarrow",)),
        "from_arrow_schema": Case(arrow_schema, ("pyarrow",)),
        "from_avro_file": Case(avro_file, ("fastavro",)),
        "from_feather": Case(read_file("feather"), ("pyarrow",)),
        "from_parquet": Case(read_file("parquet"), ("pyarrow", "fsspec")),
        "metadata_from_parquet": Case(read_file("parquet"), ("pyarrow", "fsspec")),
        "from_safetensors": Case(read_file("safetensors"), ("safetensors",)),
        "to_feather": Case(write_file("feather"), ("pyarrow",)),
        "to_parquet": Case(write_file("parquet"), ("pyarrow", "fsspec")),
        "to_safetensors": Case(write_file("safetensors"), ("safetensors",)),
        "to_parquet_dataset": Case(parquet_dataset, ("pyarrow", "fsspec")),
        "to_parquet_row_groups": Case(parquet_row_groups, ("pyarrow", "fsspec")),
        "from_cupy": Case(device_conversion("cupy", "asarray"), ("cupy",), cuda=True),
        "to_cupy": Case(lambda f, fn: partial(fn, f.regular), ("cupy",), cuda=True),
        "from_cudf": Case(
            lambda f, fn: partial(
                fn, import_module("cudf").Series(f.dense[:, 0].copy())
            ),
            ("cupy", "cudf"),
            cuda=True,
        ),
        "to_cudf": Case(
            lambda f, fn: partial(fn, f.ragged), ("cupy", "cudf"), cuda=True
        ),
        "from_jax": jax_conversion(into_awkward=True),
        "from_torch": Case(device_conversion("torch", "tensor"), ("torch",)),
        "from_tensorflow": Case(
            device_conversion("tensorflow", "constant"), ("tensorflow",)
        ),
        "from_raggedtensor": Case(
            device_conversion("tensorflow.ragged", "constant", ragged=True),
            ("tensorflow",),
        ),
        "from_rdataframe": Case(root_input, ("ROOT",)),
        "to_rdataframe": Case(lambda f, fn: partial(fn, {"x": f.ragged}), ("ROOT",)),
    }
)


def strings(names, *args):
    def prepare(f, fn):
        return partial(fn, f.strings, *args)

    return dict.fromkeys(names.split(), Case(prepare, ("pyarrow",)))


STRINGS = {
    **strings("""
        capitalize is_alnum is_alpha is_ascii is_decimal is_digit is_lower
        is_numeric is_printable is_space is_title is_upper length lower
        ltrim_whitespace reverse rtrim_whitespace split_whitespace swapcase
        title to_categorical trim_whitespace upper
    """),
    **strings("center lpad rpad", 12),
    **strings(
        "count_substring ends_with find_substring match_substring starts_with", "bc"
    ),
    **strings(
        """
        count_substring_regex find_substring_regex match_substring_regex
        split_pattern_regex
    """,
        "[0-9]+",
    ),
    **strings("extract_regex", r"(?P<value>[0-9]+)"),
    **strings("ltrim rtrim trim", " "),
    **strings("match_like", "%bc%"),
    **strings("repeat", 2),
    **strings("replace_slice", 0, 2, "X"),
    **strings("replace_substring", "bc", "XY"),
    **strings("replace_substring_regex", "[0-9]+", "XY"),
    **strings("slice", 1, 3),
    **strings("split_pattern", "-"),
    "index_in": Case(
        lambda f, fn: partial(fn, f.strings, f.ak.Array(["42", "abc-34"])),
        ("pyarrow",),
    ),
    "is_in": Case(
        lambda f, fn: partial(fn, f.strings, f.ak.Array(["42", "abc-34"])),
        ("pyarrow",),
    ),
    "join": Case(
        lambda f, fn: partial(fn, f.ak.Array([["x", "y"]] * f.size), "-"),
        ("pyarrow",),
    ),
    "join_element_wise": Case(
        lambda f, fn: partial(fn, f.strings, f.strings, f.ak.Array(["-"] * f.size)),
        ("pyarrow",),
    ),
}


def member(fixture, name, *, method=False, requires=()):
    def prepare(f, _target):
        instance = getattr(f, fixture)
        if method:
            return getattr(instance, name)
        return partial(getattr, instance, name)

    return Case(prepare, requires)


def properties(fixture, names):
    return {name: member(fixture, name) for name in names.split()}


def write_show(fixture):
    def prepare(f, _target):
        show = getattr(f, fixture).show

        def run():
            stream = io.StringIO()
            show(stream=stream)
            return stream.getvalue()

        return run

    return Case(prepare)


def builder_method(name, *args, start=()):
    def prepare(f, _target):
        builder = f.ak.ArrayBuilder()
        for step, values in start:
            getattr(builder, step)(*values)
        return partial(getattr(builder, name), *args)

    return Case(prepare)


def temporal_builder_method(name, dtype):
    def prepare(f, _target):
        builder = f.ak.ArrayBuilder()
        return partial(getattr(builder, name), f.np.array(5, dtype=dtype)[()])

    return Case(prepare)


def context(name, *args):
    def prepare(f, _target):
        builder = f.ak.ArrayBuilder()
        manager = getattr(builder, name)
        if isinstance(manager, type):
            manager = partial(manager, builder)

        def run():
            with manager(*args):
                pass

        return run

    return Case(prepare)


LIST_STARTED = (("begin_list", ()), ("integer", (1,)))
TUPLE_STARTED = (("begin_tuple", (2,)), ("index", (0,)), ("integer", (1,)))
RECORD_STARTED = (("begin_record", ()), ("field", ("x",)), ("integer", (1,)))


MEMBERS = {
    "Array": {
        **properties(
            "records",
            "attrs behavior fields is_tuple layout nbytes ndim "
            "named_axis positional_axis type typestr",
        ),
        "mask": member("ragged", "mask"),
        "Mask": Case(lambda f, fn: partial(fn, f.ragged)),
        "to_list": member("ragged", "to_list", method=True),
        "tolist": member("ragged", "tolist", method=True),
        "to_numpy": member("regular", "to_numpy", method=True),
        "show": write_show("records"),
        "numba_type": member("ragged", "numba_type", requires=("numba",)),
        "cpp_type": member("ragged", "cpp_type", requires=("cppyy",)),
    },
    "Record": {
        **properties(
            "record",
            "attrs behavior fields is_tuple layout nbytes "
            "named_axis positional_axis type typestr",
        ),
        "to_list": member("record", "to_list", method=True),
        "tolist": member("record", "tolist", method=True),
        "show": write_show("record"),
        "numba_type": member("record", "numba_type", requires=("numba",)),
    },
    "ArrayBuilder": {
        **properties("builder", "attrs behavior type typestr"),
        "snapshot": member("builder", "snapshot", method=True),
        "to_list": member("builder", "to_list", method=True),
        "tolist": member("builder", "tolist", method=True),
        "to_numpy": member("builder", "to_numpy", method=True),
        "show": write_show("builder"),
        "numba_type": member("builder", "numba_type", requires=("numba",)),
        "null": builder_method("null"),
        "boolean": builder_method("boolean", True),
        "integer": builder_method("integer", 42),
        "real": builder_method("real", 1.5),
        "complex": builder_method("complex", 1.5 + 2j),
        "datetime": temporal_builder_method("datetime", "datetime64[ms]"),
        "timedelta": temporal_builder_method("timedelta", "timedelta64[ms]"),
        "bytestring": builder_method("bytestring", b"abc"),
        "string": builder_method("string", "café"),
        "begin_list": builder_method("begin_list"),
        "end_list": builder_method("end_list", start=LIST_STARTED),
        "begin_tuple": builder_method("begin_tuple", 2),
        "index": builder_method("index", 1, start=TUPLE_STARTED),
        "end_tuple": builder_method("end_tuple", start=TUPLE_STARTED),
        "begin_record": builder_method("begin_record"),
        "field": builder_method("field", "y", start=RECORD_STARTED),
        "end_record": builder_method("end_record", start=RECORD_STARTED),
        "append": Case(lambda f, _: partial(f.ak.ArrayBuilder().append, f.rows[0])),
        "extend": Case(lambda f, _: partial(f.ak.ArrayBuilder().extend, f.ragged)),
        "List": context("List"),
        "Tuple": context("Tuple", 2),
        "Record": context("Record", None),
        "list": context("list"),
        "tuple": context("tuple", 2),
        "record": context("record"),
    },
}


def protocol(fixture, operation, requires=()):
    def prepare(f, _target):
        instance = getattr(f, fixture)
        return partial(operation, f, instance)

    return Case(prepare, requires)


MEMBERS["Array"].update(
    {
        "__getitem__": protocol("records", lambda f, x: x["y"]),
        "__getattr__": protocol("records", lambda f, x: x.y),
        "__iter__": protocol("ragged", lambda f, x: tuple(x)),
        "__len__": protocol("ragged", lambda f, x: len(x)),
        "__array__": protocol("regular", lambda f, x: f.np.asarray(x)),
        "__array_ufunc__": protocol("ragged", lambda f, x: f.np.add(x, 1)),
        "__array_function__": protocol("ragged", lambda f, x: f.np.sum(x, axis=-1)),
        "__arrow_array__": protocol(
            "ragged", lambda f, x: import_module("pyarrow").array(x), ("pyarrow",)
        ),
        "__cast_cpp__": protocol("ragged", lambda f, x: x.__cast_cpp__(), ("cppyy",)),
    }
)


def record_ufunc(f, _target):
    absolute = f.np.absolute
    behavior = {(absolute, "Point"): lambda point: absolute(point.x)}
    point = f.ak.Record(f.record, with_name="Point", behavior=behavior)
    return partial(absolute, point)


MEMBERS["Record"].update(
    {
        "__getitem__": protocol("record", lambda f, x: x["y"]),
        "__getattr__": protocol("record", lambda f, x: x.y),
        "__array_ufunc__": Case(record_ufunc),
    }
)
MEMBERS["ArrayBuilder"].update(
    {
        "__len__": protocol("builder", lambda f, x: len(x)),
        "__array__": protocol("builder", lambda f, x: f.np.asarray(x)),
        "__arrow_array__": protocol(
            "builder", lambda f, x: import_module("pyarrow").array(x), ("pyarrow",)
        ),
    }
)


def mutate_field(fixture, name):
    def prepare(f, _target):
        # Make a distinct wrapper so mutating the case leaves fixtures intact.
        source = getattr(f, fixture)
        instance = type(source)(source)
        if name == "__setitem__":
            return partial(instance.__setitem__, "z", 7)
        return partial(instance.__delitem__, "y")

    return Case(prepare)


for name, fixture in (("Array", "records"), ("Record", "record")):
    MEMBERS[name].update(
        {
            "__setitem__": mutate_field(fixture, "__setitem__"),
            "__delitem__": mutate_field(fixture, "__delitem__"),
            "__dir__": protocol(fixture, lambda f, x: dir(x)),
            "__setattr__": protocol(fixture, lambda f, x: setattr(x, "behavior", {})),
        }
    )


CASES = {
    "ak.Array": call(lambda f: (f.rows,)),
    "ak.Record": call(lambda f: ({"x": f.rows[0], "y": f.rows[-1]},)),
    "ak.ArrayBuilder": call(lambda f: ()),
    **{f"ak.{name}": case for name, case in OPERATIONS.items()},
    **{f"ak.str.{name}": case for name, case in STRINGS.items()},
    **{
        f"ak.{owner}.{name}": case
        for owner, members in MEMBERS.items()
        for name, case in members.items()
    },
}
