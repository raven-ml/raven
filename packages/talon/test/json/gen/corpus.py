# /// script
# dependencies = ["orjson==3.10.15"]
# ///
"""Writes the fixtures of the JSON suite and what Python reads from them.

Run from this directory with `uv run corpus.py [--check]`. The fixtures are
committed, so the suite itself needs neither Python nor the network. With
--check nothing is written, and the run fails if a fixture would change.

Each fixture is written by Python (orjson, or the standard json module) and
read back by the same library. ../golden/values.txt holds one line per file
and column: `file\\tcolumn\\ttype\\trow\\x1frow...`. The type is the one the
rules of Talon_json's sniffing give from Python's values, and each row is
rendered as test_json.ml renders a column.
"""

import argparse
import json
from pathlib import Path

import orjson

HERE = Path(__file__).parent
SUPPORT = HERE.parent / "support"
GOLDEN = HERE.parent / "golden" / "values.txt"

FLAT = [
    {"id": 1, "price": 1.5, "name": "alpha", "active": True},
    {"id": 2, "price": None, "name": "béta", "active": False},
    {"id": -9007199254740993, "name": "日本語 emoji \U0001F600", "active": None},
    {"id": 9223372036854775807, "price": 3.0, "name": "", "extra": "kept"},
]

NESTED = [
    {
        "user": {"name": "ana", "loc": {"lat": 45.5, "lng": -73.5}},
        "tags": ["a", "b"],
        "scores": [[1, 2], [3]],
    },
    {"user": {"name": "ben", "loc": None}, "tags": [], "scores": [[]]},
    {"user": None, "tags": ["c"], "scores": None},
    {"user": {"loc": {"lat": 1.0, "lng": 2.0}}, "scores": [[4], [], [5, 6]]},
]

NUMBERS = [
    {"i": 0, "x": 1e21, "y": 5e-324},
    {"i": -9223372036854775808, "x": 1.7976931348623157e308, "y": -0.0},
    {"i": 9223372036854775807, "x": 2.5e-8, "y": 123456.789},
]

ESCAPES = [
    {"s": 'quote " back \\ slash / nl \n tab \t bell \x07 end'},
    {"s": "emoji \U0001F600 and é"},
]


def write_ndjson(name, rows):
    data = b"".join(orjson.dumps(r) + b"\n" for r in rows)
    return name, data


def write_ascii_array(name, rows):
    text = json.dumps(rows, indent=2, ensure_ascii=True)
    return name, text.encode("ascii")


def write_compact_array(name, rows):
    text = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return name, text.encode("utf-8")


FIXTURES = [
    write_ndjson("flat.ndjson", FLAT),
    write_ndjson("nested.ndjson", NESTED),
    write_ndjson("numbers.ndjson", NUMBERS),
    write_ndjson("escapes.ndjson", ESCAPES),
    write_ascii_array("flat_ascii.json", FLAT),
    write_compact_array("nested.json", NESTED),
    write_ascii_array("escapes_ascii.json", ESCAPES),
]


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def join(a, b):
    """The sniffed type of a column, from its values, as talon's rules say."""
    if a is None:
        return b
    if b is None:
        return a
    if a == b:
        return a
    if {a, b} == {"int64", "float64"} if isinstance(a, str) and isinstance(b, str) else False:
        return "float64"
    if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] == "list":
        return ("list", join(a[1], b[1]))
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = join(out.get(k), v)
        return out
    return "string"


def type_of(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "bool"
    if is_int(v):
        return "int64"
    if isinstance(v, float):
        return "float64"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        t = None
        for x in v:
            t = join(t, type_of(x))
        return ("list", t)
    t = {}
    for k, x in v.items():
        t[k] = join(t.get(k), type_of(x))
    return t


def type_name(t):
    if t is None:
        return "string"
    if isinstance(t, str):
        return t
    if isinstance(t, tuple):
        return f"list[{type_name(t[1])}]"
    fields = ", ".join(f"{k} {type_name(v)}" for k, v in t.items())
    return f"record[{fields}]"


def escape(s):
    return (
        s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\x1f", "\\x1f")
    )


def number(v):
    r = repr(v)
    return r[:-2] if r.endswith(".0") else r


def render(v, t):
    if v is None:
        return "null"
    if isinstance(t, tuple):
        return "[" + ",".join(render(x, t[1]) for x in v) + "]"
    if isinstance(t, dict):
        parts = (f"{k}={render(v.get(k), t[k])}" for k in t)
        return "{" + ",".join(parts) + "}"
    if t == "string":
        return '"' + escape(v) + '"'
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v) if is_int(v) else number(v)


def golden_lines(name, data):
    if name.endswith(".ndjson"):
        rows = [orjson.loads(line) for line in data.splitlines() if line.strip()]
    else:
        rows = json.loads(data)
    columns = []
    types = {}
    for row in rows:
        for k, v in row.items():
            if k not in types:
                columns.append(k)
                types[k] = None
            types[k] = join(types[k], type_of(v))
    lines = []
    for c in columns:
        t = types[c]
        values = "\x1f".join(render(r.get(c), t) for r in rows)
        lines.append(f"{name}\t{c}\t{type_name(t)}\t{values}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    lines = []
    SUPPORT.mkdir(exist_ok=True)
    GOLDEN.parent.mkdir(exist_ok=True)
    failed = False
    for name, data in FIXTURES:
        path = SUPPORT / name
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                print(f"{path} would change")
                failed = True
        else:
            path.write_bytes(data)
        lines.extend(golden_lines(name, data))
    text = "\n".join(lines) + "\n"
    if args.check:
        if not GOLDEN.exists() or GOLDEN.read_text(encoding="utf-8") != text:
            print(f"{GOLDEN} would change")
            failed = True
        if failed:
            raise SystemExit(1)
    else:
        GOLDEN.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
