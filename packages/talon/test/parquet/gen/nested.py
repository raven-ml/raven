# /// script
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Writes the nested fixtures of the Parquet suite.

Run from this directory with `uv run nested.py [--check]`. The fixtures are
committed, so the suite itself needs neither Python nor the network. With
--check nothing is written, and the run fails if a fixture would change.

- ../support/nested_*.parquet: one table of every nesting, nulls at each level
  and empty lists, in four row groups, for each data page version and with
  dictionary or PLAIN pages, the PLAIN files in pages of a few rows.
- The files of HELD, which fixtures.py fetches, are only read.
- ../golden/nested.txt: for each file and column, as pyarrow reads it: `file
  "column" value value ...`, a value per row.
"""

import argparse
import struct
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent
OUT = HERE.parent / "support"
GOLDEN = HERE.parent / "golden" / "nested.txt"
# Files fixtures.py fetches from apache/parquet-testing or that the suite
# already holds, read here as nested columns: the legacy list and map shapes
# no current writer produces.
HELD = [
    "old_list_structure.parquet",
    "nested_lists.snappy.parquet",
    "nested_maps.snappy.parquet",
    "nullable.impala.parquet",
    "list_columns.parquet",
    "nulls.snappy.parquet",
    "repeated_primitive_no_list.parquet",
    "map_no_value.parquet",
    "incorrect_map_schema.parquet",
    "datapage_v2.snappy.parquet",
]

# The table: one row in [PERIOD] of each kind of absence, in the same rows of
# every column, so that each row group holds each kind.

PERIOD = 6
ROWS = 40
GROUP = 13


def ints(i):
    return [None, [], [None], [i, None, i + 1], [i], [i, i, i, i]][i % PERIOD]


def lists(i):
    return [None, [], [None], [[]], [[i, None], None, [], [i + 1]], [[i], [i]]][i % PERIOD]


def texts(i):
    return [None, [], [""], ["a", None, "é"], ["a", "a"], [f"s{i % 3}"]][i % PERIOD]


def people(i):
    return [
        None,
        [],
        [None],
        [{"age": None, "name": None}],
        [{"age": i, "name": "ann"}, None, {"age": None, "name": "bob"}],
        [{"age": i, "name": None}],
    ][i % PERIOD]


def record(i):
    return [
        None,
        {"x": None, "tags": None, "sub": None},
        {"x": i, "tags": [], "sub": {"y": None}},
        {"x": i, "tags": [None], "sub": {"y": float(i) / 2}},
        {"x": None, "tags": ["t", None, "u"], "sub": None},
        {"x": i, "tags": ["v"], "sub": {"y": -0.0}},
    ][i % PERIOD]


def maps(i):
    return [
        None,
        [],
        [("k", None)],
        [("k", i), ("l", None), ("m", i + 1)],
        [("k", i)],
        [(f"k{i % 2}", 7), ("z", 8)],
    ][i % PERIOD]


def required(i):
    return [[], [i], [i, i + 1], [], [i], [i]][i % PERIOD]


def flag(i):
    return [None, [], [True], [False, None, True], [True, True], [False]][i % PERIOD]


PERSON = pa.struct([("age", pa.int32()), ("name", pa.string())])
RECORD = pa.struct(
    [
        ("x", pa.int32()),
        ("tags", pa.list_(pa.string())),
        ("sub", pa.struct([("y", pa.float64())])),
    ]
)
SCHEMA = pa.schema(
    [
        ("id", pa.int32()),
        ("tag", pa.string()),
        ("ints", pa.list_(pa.int32())),
        ("lists", pa.list_(pa.list_(pa.int64()))),
        ("texts", pa.list_(pa.string())),
        ("people", pa.list_(PERSON)),
        ("record", RECORD),
        ("maps", pa.map_(pa.string(), pa.int32())),
        pa.field("required", pa.list_(pa.field("element", pa.int32(), nullable=False)), nullable=False),
        ("flags", pa.list_(pa.bool_())),
    ]
)


def table():
    columns = {
        "id": list(range(ROWS)),
        "tag": [f"t{i % 3}" for i in range(ROWS)],
        "ints": [ints(i) for i in range(ROWS)],
        "lists": [lists(i) for i in range(ROWS)],
        "texts": [texts(i) for i in range(ROWS)],
        "people": [people(i) for i in range(ROWS)],
        "record": [record(i) for i in range(ROWS)],
        "maps": [maps(i) for i in range(ROWS)],
        "required": [required(i) for i in range(ROWS)],
        "flags": [flag(i) for i in range(ROWS)],
    }
    return pa.table(columns, schema=SCHEMA)


# Bytes of a data page small enough that a row group spans several.
SMALL_PAGE = 16
SMALL_BATCH = 2

VARIANTS = [
    ("nested_v1_dict.parquet", "1.0", True),
    ("nested_v1_plain.parquet", "1.0", False),
    ("nested_v2_dict.parquet", "2.0", True),
    ("nested_v2_plain.parquet", "2.0", False),
]


def generate():
    t = table()
    for name, page, dictionary in VARIANTS:
        pq.write_table(
            t,
            OUT / name,
            version="2.6",
            data_page_version=page,
            use_dictionary=dictionary,
            compression="NONE",
            row_group_size=GROUP,
            data_page_size=SMALL_PAGE if not dictionary else None,
            write_batch_size=SMALL_BATCH,
            write_statistics=False,
            store_schema=False,
        )


# Rendering, as test_nested.ml renders talon's columns

def quote(b):
    if isinstance(b, str):
        b = b.encode()
    out = '"'
    for c in b:
        ch = chr(c)
        if 32 <= c <= 126 and ch not in '"\\':
            out += ch
        else:
            out += f"\\x{c:02x}"
    return out + '"'


def value(t, v):
    if v is None:
        return "null"
    if pa.types.is_map(t):
        names = (t.key_field.name, t.item_field.name)
        items = ", ".join(
            f"{{{names[0]}: {value(t.key_type, k)}, {names[1]}: {value(t.item_type, x)}}}" for k, x in v
        )
        return f"[{items}]"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return "[" + ", ".join(value(t.value_type, x) for x in v) + "]"
    if pa.types.is_struct(t):
        fields = ", ".join(f"{t.field(i).name}: {value(t.field(i).type, v[t.field(i).name])}" for i in range(t.num_fields))
        return "{" + fields + "}"
    if pa.types.is_boolean(t):
        return "true" if v else "false"
    if pa.types.is_float32(t):
        return f"0x{struct.unpack('<I', struct.pack('<f', v))[0]:08x}"
    if pa.types.is_float64(t):
        return f"0x{struct.unpack('<Q', struct.pack('<d', v))[0]:016x}"
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return quote(v)
    if pa.types.is_integer(t):
        return str(v)
    if pa.types.is_timestamp(t):
        return str(int(pa.scalar(v, t).cast(pa.int64()).as_py()))
    raise SystemExit(f"no rendering of {t}")


# pyarrow refuses a map whose keys are optional. These are the values of
# incorrect_map_schema.parquet in the order its PLAIN pages store them, read
# from the gunzipped pages by hand.
BY_HAND = [
    'incorrect_map_schema.parquet "my_map" '
    '[{key: "parent", value: "another"}, {key: "name", value: "report"}]'
]
# Talon reads a map's entries as records of the key_value group's fields, also
# when the group holds only the key; pyarrow reads those as a list of keys.
KEYS_ONLY = ("map_no_value.parquet", "my_map_no_v")


def goldens():
    lines = []
    for name in [n for n, _, _ in VARIANTS] + HELD:
        if name == "incorrect_map_schema.parquet":
            continue
        # pyarrow reads back neither page version of a list of booleans split
        # across pages; every variant holds one table, so it reads the
        # dictionary-encoded one, whose pages are whole.
        t = pq.ParquetFile(OUT / name.replace("_plain", "_dict")).read()
        for c in t.schema.names:
            a = t.column(c).combine_chunks()
            ty = t.schema.field(c).type
            if (name, c) == KEYS_ONLY:
                item = ty.value_type
                rows = ["[" + ", ".join(f"{{key: {value(item, k)}}}" for k in v) + "]" for v in a.to_pylist()]
            else:
                rows = [value(ty, v) for v in a.to_pylist()]
            lines.append(f"{name} {quote(c)} " + " ".join(rows))
    return "\n".join(lines + BY_HAND) + "\n"


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    if not parser.parse_args().check:
        generate()
        GOLDEN.write_text(goldens())
        return
    committed = OUT
    with tempfile.TemporaryDirectory() as tmp:
        OUT = Path(tmp)
        for name in HELD:
            (OUT / name).write_bytes((committed / name).read_bytes())
        generate()
        stale = [n for n, _, _ in VARIANTS if (OUT / n).read_bytes() != (committed / n).read_bytes()]
        text = goldens()
    OUT = committed
    if GOLDEN.read_text() != text:
        stale.append("nested.txt")
    if stale:
        raise SystemExit(f"stale: {' '.join(stale)}; run gen/nested.py")


if __name__ == "__main__":
    main()
