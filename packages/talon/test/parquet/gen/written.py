# /// script
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Reads talon's own Parquet files with pyarrow and records what pyarrow reads.

Run from this directory with `uv run written.py [--check]` after writing
../support/written/*.parquet with the executable gen_written. The files are
committed, so the suite itself needs neither Python nor the network. The files
past a data page's size are not committed: this run writes them with
gen_written under _build/default, which must be built. With --check nothing is
written, and the run fails if ../golden/written.txt would change.

It fails if a file's Parquet types, row groups or statistics break the format,
and records in ../golden/written.txt, per file and column, pyarrow's type and
every value rendered as test_writer.ml renders talon's: ints in decimal, float
bits in hexadecimal, bytes in hexadecimal, nested values bracketed. A column
past 4096 bytes of text is recorded by its MD5.
"""

import argparse
import hashlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent
WRITTEN = HERE.parent / "support" / "written"
GOLDEN = HERE.parent / "golden" / "written.txt"
GEN = HERE.parents[4] / "_build/default/packages/talon/test/parquet/gen_written.exe"

TYPES = {
    "bool": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float16": "halffloat",
    "float32": "float",
    "float64": "double",
    "string": "string",
    "binary": "binary",
    "category": "string",
    "date": "date32[day]",
    "clock_s": "time32[ms]",
    "clock_ms": "time32[ms]",
    "clock_us": "time64[us]",
    "clock_ns": "time64[ns]",
    "duration_ms": "int64",
    "wall_s": "timestamp[ms]",
    "wall_ms": "timestamp[ms]",
    "wall_us": "timestamp[us]",
    "wall_ns": "timestamp[ns]",
    "instant_ms": "timestamp[ms, tz=UTC]",
    "instant_us": "timestamp[us, tz=UTC]",
    "instant_ns": "timestamp[ns, tz=UTC]",
}


def render_scalar(t, v):
    if pa.types.is_float16(t) and isinstance(v, (bytes, bytearray)):
        v = struct.unpack("<e", bytes(v))[0]
    if pa.types.is_boolean(t):
        return "true" if v else "false"
    if pa.types.is_floating(t):
        v = float(v)
        return "nan" if v != v else "%016x" % struct.unpack(">Q", struct.pack(">d", v))[0]
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).hex()
    if isinstance(v, str):
        return v.encode().hex()
    return str(int(v))


def render(a):
    t = a.type
    if isinstance(a, pa.ChunkedArray):
        a = a.combine_chunks()
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        offsets = a.offsets.to_pylist()
        values = render(a.values)
        out = []
        for i in range(len(a)):
            if a[i].is_valid:
                o = offsets[i], offsets[i + 1]
                out.append("[" + " ".join(values[o[0]:o[1]]) + "]")
            else:
                out.append("null")
        return out
    if pa.types.is_struct(t):
        fields = [render(a.field(k)) for k in range(t.num_fields)]
        out = []
        for i in range(len(a)):
            if a[i].is_valid:
                body = ",".join(
                    f"{t.field(k).name}={fields[k][i]}" for k in range(t.num_fields)
                )
                out.append("{" + body + "}")
            else:
                out.append("null")
        return out
    if pa.types.is_date32(t) or pa.types.is_time(t) or pa.types.is_timestamp(t):
        ints = a.cast(pa.int32() if pa.types.is_date32(t) or pa.types.is_time32(t) else pa.int64())
        return ["null" if v is None else str(v) for v in ints.to_pylist()]
    return ["null" if not x.is_valid else render_scalar(t, x.as_py()) for x in a]


def utf8(b):
    b.decode("utf-8")


def check_statistics(path, pf):
    """min and max of every row group of a flat column, against its values, and
    the truncation rules on byte strings."""
    meta = pf.metadata
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        table = pf.read_row_group(g)
        for c in range(rg.num_columns):
            col = rg.column(c)
            top = col.path_in_schema.split(".")[0]
            if "." in col.path_in_schema or top not in table.column_names:
                continue
            values = table.column(top).combine_chunks()
            s = col.statistics
            assert s is not None, f"{path}: {top}: no statistics"
            nulls = sum(1 for v in values if not v.is_valid)
            assert s.null_count == nulls, f"{path}: {top}: null count {s.null_count} of {nulls}"
            if pa.types.is_temporal(values.type):
                continue
            present = [v.as_py() for v in values if v.is_valid]
            if values.type.equals(pa.string()) or values.type.equals(pa.binary()):
                as_bytes = [v.encode() if isinstance(v, str) else bytes(v) for v in present]
                if not present:
                    assert not s.has_min_max
                    continue
                lo, hi = min(as_bytes), max(as_bytes)
                if not s.has_min_max:
                    # Only a value that has no shorter upper bound: all 0xFF.
                    assert len(hi) > 64 and set(hi[:64]) <= {0xFF}, f"{path}: {top}"
                    continue
                smin = s.min if isinstance(s.min, bytes) else s.min.encode()
                assert smin <= lo, f"{path}: {top}: min {smin!r} above {lo!r}"
                if len(lo) <= 64:
                    assert smin == lo
                if s.max is not None:
                    smax = s.max if isinstance(s.max, bytes) else s.max.encode()
                    assert smax >= hi, f"{path}: {top}: max {smax!r} below {hi!r}"
                    if len(hi) <= 64:
                        assert smax == hi
                    if values.type.equals(pa.string()):
                        utf8(smin), utf8(smax)
                continue
            is_float = pa.types.is_floating(values.type)
            comparable = [v for v in present if not is_float or v == v]
            if not comparable:
                assert not s.has_min_max
                continue
            assert s.has_min_max
            import math

            if is_float:
                if pa.types.is_float16(values.type):
                    # pyarrow gives a FLOAT16 statistic as its two bytes.
                    s = type("S", (), {
                        "min": struct.unpack("<e", s.min)[0],
                        "max": struct.unpack("<e", s.max)[0],
                    })
                # Zeros: the written minimum is -0 and the maximum +0.
                assert s.min == min(comparable), f"{path}: {top}: min {s.min}"
                assert s.max == max(comparable), f"{path}: {top}: max {s.max}"
                if s.min == 0:
                    assert math.copysign(1, s.min) < 0, f"{path}: {top}: min +0"
                if s.max == 0:
                    assert math.copysign(1, s.max) > 0, f"{path}: {top}: max -0"
            elif not pa.types.is_temporal(values.type):
                assert s.min == min(comparable) and s.max == max(comparable), f"{path}: {top}"


def rendered(column):
    text = " ".join(render(column))
    if len(text) > 4096:
        return "md5:" + hashlib.md5(text.encode()).hexdigest()
    return text


def lines(large):
    out = []
    files = list(WRITTEN.glob("*.parquet")) + list(large.glob("*.parquet"))
    for path in sorted(files, key=lambda p: p.stem):
        pf = pq.ParquetFile(path)
        t = pf.read()
        meta = pf.metadata
        name = path.stem
        assert meta.format_version == "1.0", name
        out.append(f"{name} rows {t.num_rows} groups {meta.num_row_groups}")
        check_statistics(name, pf)
        for i, field in enumerate(t.schema):
            if name.startswith("flat") or name == "row_groups":
                want = TYPES[field.name]
                assert str(field.type) == want, f"{name}: {field.name}: {field.type} is not {want}"
            out.append(f"{name} column {field.name} {field.type}")
            out.append(f"{name} values {field.name} " + rendered(t.column(i)))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as large:
        subprocess.run([GEN, large, "large"], check=True)
        text = lines(Path(large))
    if args.check:
        if GOLDEN.read_text() != text:
            sys.exit("golden/written.txt would change")
        return
    GOLDEN.write_text(text)


main()
