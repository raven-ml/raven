# /// script
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Writes the Arrow IPC corpus of the talon.arrow suite with pinned pyarrow.

Run from this directory with `uv run arrow.py [--check]`. The fixtures are
committed, so the suite itself needs neither Python nor the network. With
--check nothing is written, and the run fails if a fixture would change.

- ../support/*.arrow and *.arrows: files and streams of every type, nulls at
  every level, batch boundaries, compression, dictionaries (replaced, delta,
  duplicate), extensions, custom metadata, and fields talon refuses.
- ../golden/values.txt: per file and column, the talon type pyarrow's type
  reads as and every value: integers and ticks in decimal, floats with %.17g,
  text and bytes in hexadecimal, null as `n`, nested values bracketed. The
  suite renders talon's columns the same way.
"""

import argparse
import filecmp
import struct
import sys
import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

HERE = Path(__file__).parent
OUT = HERE.parent / "support"
GOLDEN = HERE.parent / "golden" / "values.txt"


def nulled(values, every=5):
    return [None if i % every == 0 else v for i, v in enumerate(values)]


N = 40
ints = [(i * 3) - 40 for i in range(N)]
floats = [0.5 * i - 7.25 for i in range(N)] + [float("inf"), float("-inf")]
texts = ["", "a", "é", "日本語", "🦊", "abc"]


def text(i):
    return texts[i % len(texts)]


def bytes_of(i):
    return bytes([i % 256, (i * 7) % 256])[: (i % 3)]


def flat_table():
    d = datetime(2024, 1, 2, 3, 4, 5, 123456)
    columns = {
        "bool": pa.array(nulled([i % 3 == 0 for i in range(N)]), pa.bool_()),
        "int8": pa.array(nulled([i - 20 for i in range(N)]), pa.int8()),
        "int16": pa.array(nulled(ints), pa.int16()),
        "int32": pa.array(nulled(ints), pa.int32()),
        "int64": pa.array(nulled([i * (1 << 40) for i in ints]), pa.int64()),
        "uint8": pa.array(nulled(range(N)), pa.uint8()),
        "uint16": pa.array(nulled(range(N)), pa.uint16()),
        "uint32": pa.array(nulled(range(N)), pa.uint32()),
        "uint64": pa.array(nulled([i << 40 for i in range(N)]), pa.uint64()),
        "float16": pa.array(nulled([float(i) for i in range(N)]), pa.float16()),
        "float32": pa.array(nulled(floats[:N]), pa.float32()),
        "float64": pa.array(nulled(floats[:N]), pa.float64()),
        "utf8": pa.array(nulled([text(i) for i in range(N)]), pa.string()),
        "large_utf8": pa.array(nulled([text(i) for i in range(N)]), pa.large_string()),
        "binary": pa.array(nulled([bytes_of(i) for i in range(N)]), pa.binary()),
        "large_binary": pa.array(nulled([bytes_of(i) for i in range(N)]), pa.large_binary()),
        "fixed_binary": pa.array(nulled([b"abc" for _ in range(N)]), pa.binary(3)),
        "date32": pa.array(nulled([date(2024, 1, 1) + timedelta(i) for i in range(N)]), pa.date32()),
        "date64": pa.array(nulled([date(1969, 12, 20) + timedelta(i) for i in range(N)]), pa.date64()),
        "time32_s": pa.array(nulled([i * 60 for i in range(N)]), pa.time32("s")),
        "time32_ms": pa.array(nulled([i * 60000 for i in range(N)]), pa.time32("ms")),
        "time64_us": pa.array(nulled([i * 60000000 for i in range(N)]), pa.time64("us")),
        "time64_ns": pa.array(nulled([i * 60000000000 for i in range(N)]), pa.time64("ns")),
        "ts_s": pa.array(nulled([d + timedelta(seconds=i) for i in range(N)]), pa.timestamp("s")),
        "ts_ms": pa.array(nulled([d + timedelta(milliseconds=i) for i in range(N)]), pa.timestamp("ms")),
        "ts_us": pa.array(nulled([d + timedelta(microseconds=i) for i in range(N)]), pa.timestamp("us")),
        "ts_ns": pa.array(nulled([d + timedelta(microseconds=i) for i in range(N)]), pa.timestamp("ns")),
        "instant": pa.array(nulled([d + timedelta(minutes=i) for i in range(N)]), pa.timestamp("ms", tz="UTC")),
        "zoned": pa.array(nulled([d + timedelta(minutes=i) for i in range(N)]), pa.timestamp("us", tz="Europe/Paris")),
        "dur_s": pa.array(nulled([timedelta(seconds=i) for i in range(N)]), pa.duration("s")),
        "dur_ms": pa.array(nulled([timedelta(milliseconds=i) for i in range(N)]), pa.duration("ms")),
        "dur_us": pa.array(nulled([timedelta(microseconds=i) for i in range(N)]), pa.duration("us")),
        "dur_ns": pa.array(nulled([timedelta(microseconds=i) for i in range(N)]), pa.duration("ns")),
    }
    return pa.table(columns)


def list_cell(i):
    return [None, [], [None], [i, None, i + 1], [i], [i, i, i, i]][i % 6]


def nested_table():
    columns = {
        "list": pa.array([list_cell(i) for i in range(N)], pa.list_(pa.int32())),
        "large_list": pa.array([list_cell(i) for i in range(N)], pa.large_list(pa.int64())),
        "list_list": pa.array(
            [None if i % 7 == 0 else [list_cell(i), None] for i in range(N)],
            pa.list_(pa.list_(pa.int32())),
        ),
        "struct": pa.array(
            [
                None if i % 5 == 0 else {"a": None if i % 3 == 0 else i, "b": text(i)}
                for i in range(N)
            ],
            pa.struct([("a", pa.int32()), ("b", pa.string())]),
        ),
        "list_struct": pa.array(
            [
                None if i % 7 == 0 else [{"x": i}, None]
                for i in range(N)
            ],
            pa.list_(pa.struct([("x", pa.int64())])),
        ),
        "fixed_list": pa.array(
            [None if i % 5 == 0 else [i, i + 1, i + 2] for i in range(N)],
            pa.list_(pa.int32(), 3),
        ),
        "fixed_list_text": pa.array(
            [None if i % 5 == 0 else [text(i), None] for i in range(N)],
            pa.list_(pa.string(), 2),
        ),
        "fixed_nested": pa.array(
            [None if i % 5 == 0 else [[0.5, i * 1.0], [-1.0, 2.0], [3.0, i * 2.0]] for i in range(N)],
            pa.list_(pa.list_(pa.float32(), 2), 3),
        ),
        "map": pa.array(
            [None if i % 6 == 0 else [("k%d" % i, i), ("z", None)] for i in range(N)],
            pa.map_(pa.string(), pa.int32()),
        ),
    }
    return pa.table(columns)


def dict_table():
    columns = {
        "cat": pa.array(nulled([text(i % 4) for i in range(N)])).dictionary_encode(),
        "cat_int8": pa.DictionaryArray.from_arrays(
            pa.array(nulled([i % 3 for i in range(N)]), pa.int8()),
            pa.array(["x", "y", "z"]),
        ),
        "cat_uint32": pa.DictionaryArray.from_arrays(
            pa.array(nulled([i % 3 for i in range(N)]), pa.uint32()),
            pa.array(["x", "y", "z"], pa.large_string()),
        ),
    }
    return pa.table(columns)


def views_table():
    long_text = "a string longer than twelve bytes"
    return pa.table(
        {
            "utf8_view": pa.array(
                nulled([text(i) if i % 2 else long_text + text(i) for i in range(N)]),
                pa.string_view(),
            ),
            "binary_view": pa.array(
                nulled([bytes_of(i) for i in range(N)]), pa.binary_view()
            ),
        }
    )


class Point(pa.ExtensionType):
    def __init__(self):
        pa.ExtensionType.__init__(self, pa.struct([("x", pa.float64()), ("y", pa.float64())]), "raven.point")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()


class Kg(pa.ExtensionType):
    def __init__(self):
        pa.ExtensionType.__init__(self, pa.float64(), "raven.mass")

    def __arrow_ext_serialize__(self):
        return b"kg"

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()


def extension_table():
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [2, 2])
    permuted = pa.fixed_shape_tensor(pa.int16(), [2, 3], permutation=[1, 0])
    tensors = pa.ExtensionArray.from_storage(
        tensor_type,
        pa.array(
            [None if i % 5 == 0 else [float(i), 1.5, -2.0, float(i * 2)] for i in range(N)],
            tensor_type.storage_type,
        ),
    )
    permuted_values = pa.ExtensionArray.from_storage(
        permuted,
        pa.array(
            [[i, i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(N)],
            permuted.storage_type,
        ),
    )
    kg = pa.ExtensionArray.from_storage(Kg(), pa.array(nulled(floats[:N]), pa.float64()))
    point_storage = pa.array(
        [None if i % 5 == 0 else {"x": i * 1.0, "y": -i * 1.0} for i in range(N)],
        Point().storage_type,
    )
    points = pa.ExtensionArray.from_storage(Point(), point_storage)
    return pa.table({"tensor": tensors, "permuted": permuted_values, "kg": kg, "point": points})


def metadata_table():
    field = pa.field("x", pa.int32(), metadata={"unit": "kg", "origin": "lab:1"})
    schema = pa.schema(
        [field, pa.field("y", pa.string())],
        metadata={"schema key": "schema value", "ARROW:x": "reserved"},
    )
    return pa.table(
        {"x": pa.array([1, None, 3], pa.int32()), "y": pa.array(["a", "b", None])},
        schema=schema,
    )


# Writers

CHUNKS = [13, 0, 1, 26]


def batches(table, chunks):
    whole = table.combine_chunks().to_batches()
    if chunks is None or not whole:
        return whole
    whole, at, out = whole[0], 0, []
    for n in chunks:
        out.append(whole.slice(at, n))
        at += n
    return out


def write_file(path, table, chunks=None, options=None):
    with ipc.new_file(path, table.schema, options=options) as w:
        for b in batches(table, chunks):
            w.write_batch(b)


def write_stream(path, table, chunks=None, options=None):
    with ipc.new_stream(path, table.schema, options=options) as w:
        for b in batches(table, chunks):
            w.write_batch(b)


def two_batches(path, first, second, options=None):
    schema = pa.schema([("c", pa.dictionary(pa.int32(), pa.string()))])
    with ipc.new_stream(path, schema, options=options) as w:
        for values in (first, second):
            w.write_batch(pa.record_batch([values if isinstance(values, pa.Array) else pa.array(values).dictionary_encode()], schema=schema))


def delta_arrays():
    grow = lambda idx, d: pa.DictionaryArray.from_arrays(pa.array(idx, pa.int32()), pa.array(d))
    return (grow([0, 1, None], ["u", "v"]), grow([2, 0, 3, 1], ["u", "v", "w", "x"]))


def refused_files(out):
    tables = {
        "refused_null": pa.table({"bad": pa.nulls(3)}),
        "refused_decimal": pa.table({"bad": pa.array([Decimal("1.5")], pa.decimal128(4, 1))}),
        "refused_interval": pa.table({"bad": pa.array([(1, 2, 3)], pa.month_day_nano_interval())}),
        "refused_union": pa.table(
            {"bad": pa.UnionArray.from_sparse(pa.array([0], pa.int8()), [pa.array([1], pa.int32())])}
        ),
        "refused_ree": pa.table(
            {"bad": pa.RunEndEncodedArray.from_arrays(pa.array([3], pa.int32()), pa.array([1], pa.int32()))}
        ),
        "refused_list_view": pa.table({"bad": pa.array([[1]], pa.list_view(pa.int32()))}),
        "refused_dict_dup": pa.table({"bad": pa.DictionaryArray.from_arrays(
            pa.array([0, 1, 2], pa.int32()), pa.array(["a", "a", "b"]))}),
        "refused_dict_null": pa.table({"bad": pa.DictionaryArray.from_arrays(
            pa.array([0, 1, 2], pa.int32()), pa.array(["p", None, "q"]))}),
        "refused_dict_int": pa.table({"bad": pa.DictionaryArray.from_arrays(
            pa.array([0, 1], pa.int16()), pa.array([10, 20], pa.int64()))}),
        "refused_dict_binary": pa.table({"bad": pa.DictionaryArray.from_arrays(
            pa.array([0, 1], pa.int32()), pa.array([b"\xff", b"\x00"], pa.binary()))}),
        "refused_tensor_dict": pa.table(
            [pa.array([[0, 1]], pa.list_(pa.dictionary(pa.int32(), pa.int32()), 2))],
            schema=pa.schema([pa.field("bad", pa.list_(pa.dictionary(pa.int32(), pa.int32()), 2), metadata={
                "ARROW:extension:name": "arrow.fixed_shape_tensor",
                "ARROW:extension:metadata": '{"shape":[2]}'})])),
        "refused_nested": pa.table(
            {"outer": pa.array([{"bad": Decimal("1.5")}], pa.struct([("bad", pa.decimal128(4, 1))]))}
        ),
    }
    for name, table in tables.items():
        write_file(out / (name + ".arrow"), table)
    two_batches(out / "refused_dict_replaced.arrows", ["p", "q", None], ["r", "s", "s"])
    (out / "refused_bomb.arrows").write_bytes(schema_bomb(40))
    # A date past int32 days reads as an error, naming the row.
    days = pa.array([2**31 * 86400000], pa.date64())
    write_file(out / "refused_date_range.arrow", pa.table({"d": days}))
    # Some writers state instants, not midnights. pyarrow will not read them.
    ms = pa.array([-1, 1, -86400000, -86400001, None], pa.date64())
    write_file(out / "date64_floor.arrow", pa.table({"d": ms}))


def schema_bomb(depth):
    """A stream of a [depth]-deep struct whose children point twice at one table."""
    b = bytearray(4)
    word = lambda v: b.extend(struct.pack("<I", v))
    patch = lambda at, target: struct.pack_into("<I", b, at, target - at)

    def table(offsets, size):
        while (len(b) + 4 + 2 * len(offsets)) % 8:
            b.append(0)
        vt = len(b)
        b.extend(struct.pack("<HH", 4 + 2 * len(offsets), size))
        b.extend(struct.pack("<%dH" % len(offsets), *offsets))
        start = len(b)
        b.extend(struct.pack("<i", start - vt))
        return start

    message = table([4, 6, 8], 12)
    patch(0, message)
    b.extend(struct.pack("<hBx", 4, 1))  # V5, a Schema header.
    header = len(b)
    word(0)
    schema = table([0, 4], 8)
    patch(header, schema)
    fields = len(b)
    word(0)
    pending_types = []
    next_slots = [fields]
    for level in range(depth):
        vector = len(b)
        word(2 if level else 1)
        for at in next_slots:
            patch(at, vector)
        slots = [vector + 4 + 4 * i for i in range(2 if level else 1)]
        for _ in slots:
            word(0)
        node = table([0, 0, 4, 8, 0, 12], 16)
        for at in slots:
            patch(at, node)
        b.extend(struct.pack("<B3x", 13))  # A Struct_ type.
        pending_types.append(len(b))
        word(0)
        next_slots = [len(b)]
        word(0)
    for at in pending_types:
        target = table([], 4)
        patch(at, target)
    while len(b) % 8:
        b.append(0)
    return struct.pack("<iI", -1, len(b)) + bytes(b) + struct.pack("<II", 0xFFFFFFFF, 0)


def write_all(out):
    flat, nested, dicts = flat_table(), nested_table(), dict_table()
    write_file(out / "flat.arrow", flat, CHUNKS)
    write_stream(out / "flat.arrows", flat, CHUNKS)
    write_file(out / "flat_lz4.arrow", flat, CHUNKS, ipc.IpcWriteOptions(compression="lz4"))
    write_file(out / "flat_zstd.arrow", flat, CHUNKS, ipc.IpcWriteOptions(compression="zstd"))
    write_file(out / "nested.arrow", nested, CHUNKS)
    write_stream(out / "nested_zstd.arrows", nested, CHUNKS, ipc.IpcWriteOptions(compression="zstd"))
    write_file(out / "dicts.arrow", dicts, CHUNKS)
    write_stream(out / "views.arrows", views_table(), CHUNKS)
    write_file(out / "extension.arrow", extension_table(), CHUNKS)
    write_file(out / "metadata.arrow", metadata_table())
    write_file(out / "empty.arrow", flat.slice(0, 0))
    write_stream(out / "one_row.arrows", flat.slice(1, 1))
    first, second = delta_arrays()
    two_batches(out / "delta.arrows", first, second, ipc.IpcWriteOptions(emit_dictionary_deltas=True))
    refused_files(out)


# Golden text


def dict_reads_as(column):
    """The type a dictionary column reads as, by the rule of the suite."""
    t = column.type
    dictionaries = [column.chunk(i).dictionary for i in range(column.num_chunks)]
    first = dictionaries[-1] if dictionaries else pa.array([], t.value_type)
    same = all(d.equals(first.slice(0, len(d))) for d in dictionaries)
    text = pa.types.is_string(t.value_type) or pa.types.is_large_string(t.value_type)
    distinct = len(set(first.to_pylist())) == len(first) and first.null_count == 0
    if text and same and distinct:
        return "categorical"
    return talon_type(t.value_type)


def talon_type(t):
    if isinstance(t, pa.BaseExtensionType):
        if isinstance(t, pa.FixedShapeTensorType) and (t.permutation is None or list(t.permutation) == list(range(len(t.shape)))):
            return "tensor[%s, %s]" % (float_name(t.value_type), "×".join(str(s) for s in t.shape))
        if isinstance(t, pa.FixedShapeTensorType):
            meta = '{"shape":%s,"permutation":%s}' % (list_text(t.shape), list_text(t.permutation))
        else:
            meta = t.__arrow_ext_serialize__().decode()
        quoted = ' "%s"' % meta.replace('"', '\\"') if meta else ""
        return "ext[%s%s, %s]" % (t.extension_name, quoted, talon_type(t.storage_type))
    if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t):
        return float_name(t)
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t):
        return "string"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_binary_view(t) or pa.types.is_fixed_size_binary(t):
        return "binary"
    if pa.types.is_date(t):
        return "date"
    if pa.types.is_time(t):
        return "clock[%s]" % t.unit
    if pa.types.is_duration(t):
        return "duration[%s]" % t.unit
    if pa.types.is_timestamp(t):
        return "datetime[%s, UTC]" % t.unit if t.tz else "datetime[%s]" % t.unit
    if pa.types.is_map(t):
        return "list[record[key %s, value %s]]" % (talon_type(t.key_type), talon_type(t.item_type))
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return "list[%s]" % talon_type(t.value_type)
    if pa.types.is_fixed_size_list(t):
        e = t.value_type
        if pa.types.is_integer(e) or pa.types.is_floating(e):
            return "tensor[%s, %s]" % (float_name(e), t.list_size)
        return "list[%s]" % talon_type(e)
    if pa.types.is_struct(t):
        return "record[%s]" % ", ".join("%s %s" % (f.name, talon_type(f.type)) for f in t)
    raise ValueError(str(t))


def list_text(xs):
    return "[" + ",".join(str(x) for x in xs) + "]"


def float_name(t):
    return {pa.bool_(): "bool", pa.float16(): "float16", pa.float32(): "float32", pa.float64(): "float64"}.get(t, str(t))


def is_text(t):
    return pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t)


def is_binary(t):
    return pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_binary_view(t) or pa.types.is_fixed_size_binary(t)


def token(t, v):
    """The text of the value [v], as pyarrow's as_py gives it, of type [t]."""
    if v is None:
        return "n"
    if isinstance(t, pa.BaseExtensionType):
        return token(t.storage_type, v)
    if pa.types.is_boolean(t):
        return "1" if v else "0"
    if pa.types.is_integer(t):
        return str(int(v))
    if pa.types.is_floating(t):
        return "%.17g" % float(v)
    if is_text(t):
        return v.encode().hex() or "e"
    if is_binary(t):
        return bytes(v).hex() or "e"
    if pa.types.is_map(t):
        return "[" + " ".join("{%s %s}" % (token(t.key_type, k), token(t.item_type, i)) for k, i in v) + "]"
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return "[" + " ".join(token(t.value_type, x) for x in v) + "]"
    if pa.types.is_struct(t):
        return "{" + " ".join(token(f.type, v[f.name]) for f in t) + "}"
    raise ValueError(str(t))


def ticks_type(t):
    """The integer type the ticks of a temporal [t] read as, or None."""
    if pa.types.is_date64(t):
        return "date64"
    if pa.types.is_date32(t) or pa.types.is_time32(t):
        return pa.int32()
    if pa.types.is_time(t) or pa.types.is_timestamp(t) or pa.types.is_duration(t):
        return pa.int64()
    return None


def column_tokens(column):
    t = column.type
    if pa.types.is_dictionary(t):
        out = []
        for i in range(column.num_chunks):
            chunk = column.chunk(i)
            decoded = chunk.dictionary.take(chunk.indices)
            out.extend(token(t.value_type, v) for v in decoded.to_pylist())
        return out
    ticks = ticks_type(t)
    if ticks == "date64":
        column = column.cast(pa.date32()).cast(pa.int32())
    elif ticks is not None:
        column = column.cast(ticks)
    return [token(column.type if ticks is not None else t, v) for v in column.to_pylist()]


def read(path):
    reader = ipc.open_file if path.suffix == ".arrow" else ipc.open_stream
    with reader(str(path)) as r:
        return r.read_all()


def golden_lines(out):
    lines = []
    for path in sorted(out.glob("*.arrow*")):
        if path.name.startswith(("refused", "date64_floor")):
            continue
        table = read(path)
        for column, name in zip(table.columns, table.column_names):
            ty = dict_reads_as(column) if pa.types.is_dictionary(column.type) else talon_type(column.type)
            lines.append("|".join([path.name, name, ty, " ".join(column_tokens(column))]) + "\n")
    return "".join(lines)


def main():
    pa.register_extension_type(Kg())
    pa.register_extension_type(Point())
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        OUT.mkdir(exist_ok=True)
        write_all(OUT)
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(golden_lines(OUT), encoding="utf-8")
        return
    with tempfile.TemporaryDirectory(dir=HERE.parent) as tmp:
        tmp = Path(tmp)
        write_all(tmp)
        differ = [p.name for p in sorted(tmp.iterdir()) if not filecmp.cmp(p, OUT / p.name, shallow=False)]
        if differ or golden_lines(tmp) != GOLDEN.read_text(encoding="utf-8"):
            sys.exit("differ: %s" % differ)


if __name__ == "__main__":
    main()
