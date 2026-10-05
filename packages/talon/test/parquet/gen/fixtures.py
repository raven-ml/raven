# /// script
# dependencies = ["pyarrow==25.0.1", "numpy==2.3.3"]
# ///
"""Writes the fixtures of the Parquet suite.

Run from this directory with `uv run fixtures.py`. The fixtures are committed,
so the suite itself needs neither Python nor the network.

- ../support/: files of apache/parquet-testing at COMMIT, checked against
  their SHA-256 (Apache-2.0, see ../support/LICENSE.txt), and files written by
  pyarrow.
- ../golden/values.txt: one line per file, row group and column, as pyarrow reads it:
  `file group "column" "type" rows md5 first-values`, or `file group "column"
  error` for a column that does not read. The type is talon's, computed here
  from the Parquet schema, and float64 for a decimal, which has none; md5 is of
  all values joined by spaces.
- ../golden/overrides.txt: the same for columns read as another type than their own, and
  for decimals of at most 18 digits read as int64.
"""

import hashlib
import json
import struct
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).parent
OUT = HERE.parent / "support"
COMMIT = "56653c437c8092f704a092d0d1d4e600124cd49f"
RAW = f"https://raw.githubusercontent.com/apache/parquet-testing/{COMMIT}/"
CORPUS = [
    ("data/alltypes_plain.parquet", "12a618d20a59ee0967fef45e7ec1ff6d451e724838edc1bbeac780ca15e8fcc4"),
    ("data/alltypes_plain.snappy.parquet", "9f8c5d74012498235eea4431035484dc61a76f8ad2b2b9cb5ac6972db43de591"),
    ("data/alltypes_dictionary.parquet", "7b58c33503858c533e1521b3022b85a0de23e5a144420d7a3c1c426929e5f6fb"),
    ("data/binary.parquet", "b48b756e48a13f58e1234a8588c507a06a7a9bcdfb63994c86fe19d22864be8b"),
    ("data/binary_truncated_min_max.parquet", "94a1e9ef0cd5104168c1e80480fac8918a962a355d9ab33bca3e13ff4402b201"),
    ("data/bson.parquet", "44b503ac1ecb70627b29fbce2d3109cd161afe5c0f72b556978b454fc64cb129"),
    ("data/json.parquet", "594f8dca52a6428e4350d12faeaca9e2155c77511abb9122b935bd5be1a26bf5"),
    ("data/byte_array_decimal.parquet", "9e3ccb253adc5881521b952f7b621954551df1e48dfda19e9b02126aca9b127d"),
    ("data/fixed_length_decimal.parquet", "67e61d18ecca6027731faf397c5981b29d863f9eef68e3644501588663e1bfd2"),
    ("data/fixed_length_decimal_legacy.parquet", "323ff9d3379903d528cbf7b00f93e4598ed71d39aa6eeff3d322fff541c1fb2a"),
    ("data/int32_decimal.parquet", "3441daea2c44032a78a3615b82373f34575ba7d820541e821f86d8cc143653f9"),
    ("data/int64_decimal.parquet", "e24dcf95589ee230636e228ad75aa0496eca7e8f97339d9bb6ec5c8f7ab0ef56"),
    ("data/byte_stream_split.zstd.parquet", "64df58cf9a13672749cf8c852840553301f48cc45604c68b62ea8dc9437671b5"),
    ("data/byte_stream_split_extended.gzip.parquet", "86a96259c2c82e50aa2826b91d6de9514e538235343eed9c2a98318424b46cd8"),
    ("data/column_chunk_key_value_metadata.parquet", "21d0e70a750f70428ddda3ea68640c3e5753eab2311fddb1a1aefcbe428e61d0"),
    ("data/concatenated_gzip_members.parquet", "92b6af9b766dc3e46413794ed4df009e0584b8fdca106ade1a9a1ed955d32771"),
    ("data/data_index_bloom_encoding_stats.parquet", "66d53151197819919343972c997837845503ce82665c9c4481c866ef57dde0eb"),
    ("data/data_index_bloom_encoding_with_length.parquet", "d4b249e678359ba10739535ea04b082b193008baef157aadc18d20435c9c5fa5"),
    ("data/datapage_v1-snappy-compressed-checksum.parquet", "f06df378ad412ace763d129f317c52236230b2fb24073c32d2c2d5fc1ef9d697"),
    ("data/datapage_v1-uncompressed-checksum.parquet", "b1d664eaba82d89b4107a2dc2b953ec33566b3bb4f902b79ed6ced7b9fff5664"),
    ("data/int32_with_null_pages.parquet", "392046fe71c7bdf7ea59e258596b5e6919f01f65f702a27ca56d8763d2e9f9b7"),
    ("data/plain-dict-uncompressed-checksum.parquet", "4c8abc17ad0354dc540b0ad2c519d998ffb4a1a5997471b4313864852d82dccc"),
    ("data/rle-dict-snappy-checksum.parquet", "bb9de5fd817da4c403992ce2ee2c9dc93d4444f936e1a64722c65dd98d208578"),
    ("data/datapage_v2_empty_datapage.snappy.parquet", "c93d4d6ace5ac92d3bc0ba04f44077f6fb7019cbe4f3982f204d666653fc0514"),
    ("data/page_v2_empty_compressed.parquet", "5d56ca84e4fc4e77fdc713dbb9aff6f3a6c4727083628945ea5cfcb39b56aa65"),
    ("data/delta_binary_packed.parquet", "d1c2173fe97255959e3d087b3fa5b7b5c27b2aac135337b2896772d7bbdc31b4"),
    ("data/delta_byte_array.parquet", "a400b789aef5cde88551f25cdd9bba8f0ff0fe01c48ddc5303c26edf119ee279"),
    ("data/delta_encoding_optional_column.parquet", "71f8f00b00ecc132a1cc5d534acca900d02d0fe4ba3525607b03b2bd06a56f1a"),
    ("data/delta_encoding_required_column.parquet", "36ddcb79799d56d5098f4cdb42777873a5ddd9ae6f40d8cfa316abafde6c658a"),
    ("data/delta_length_byte_array.parquet", "efef768997748d27596a5c001ccd24eae42adf2825ca4d909e8fa8eb1d89f386"),
    ("data/dict-page-offset-zero.parquet", "6d043ed7c37b9f00aedfd4b134f282a0fe17959e731d44650ea808257438c6e6"),
    ("data/fixed_length_byte_array.parquet", "a5a24cfabf2d8882db861502a0fe1e4539a80772f472f014637a0d01519836a7"),
    ("data/float16_nonzeros_and_nans.parquet", "d0117dd9655992b869f8207235526a7d8931e079fd68d68c88c0170faa1f11ee"),
    ("data/float16_zeros_and_nans.parquet", "4901850e7dcd64588a49391fa1dddca514b598e6f4266239033ea73513850f47"),
    ("data/floating_orders_nan_count.parquet", "17f7d7655a089b9504a828dffaccd72225a9a6fd2a697099b5336ab274386f0a"),
    ("data/nan_in_stats.parquet", "77d921ab7bed54232da778f920f423bd821075353b6147e3680f5b20c85f6337"),
    ("data/single_nan.parquet", "ea3371c44ed1794843a2f529888120537f68aedcb80d6fbe32cea1003ab5769e"),
    ("data/int96_from_spark.parquet", "e769aaa996dee57f1104470dcce16d6871cd720d314c13361db16bf9e49f5c97"),
    ("data/int96_timestamp_order.parquet", "e35f8748d286a729e719a01c5411f81c79802d61a55046d5cf9a663918a14644"),
    ("data/lz4_raw_compressed.parquet", "d509774f6ba2f7fa2984308e64509c97a7a69ab94d5ab017211219b9812ef551"),
    ("data/nation.dict-malformed.parquet", "245c025fe866c7a55612bf0848034e6cb7b33965668e9244bc007ab0eb61034d"),
    ("data/sort_columns.parquet", "6fa8ce56cf7848e5f6a07191f7a1f1520a52f9e0983fa809bf90babf32b9525b"),
    ("data/rle_boolean_encoding.parquet", "585e22b54c482befc54fc6caaea5efce788f1d0737505c2d8b121da8ac0c7d76"),
    ("data/unknown-logical-type.parquet", "7febd4a6163c591dc6e28f408c0882b011b72cfe95c8a0bc57382feefaff4e33"),
    ("data/int32_with_uuid_logical_type.parquet", "7289902a2c914e220888b57393ccf1292bc2091d1732d1b6c229098869a20f82"),
    ("data/flba12_timestamp.parquet", "0e6b5a9552969394a42b67fb9bf17031b677909a698945fcc9ea77acfb056529"),
    ("data/datapage_v1-corrupt-checksum.parquet", "b337106431c826e3326ab8fecfa5560688aa57549fd46e0fa7cfcf99cd4e2c9e"),
    ("data/rle-dict-uncompressed-corrupt-checksum.parquet", "b96f9198a18ec7a389f989c7d2a170ddad74a664ddd4e009a240f21c83cceddf"),
    ("data/hadoop_lz4_compressed.parquet", "b43a31978e5c28c251523ab4610989881175c5ea84c8e533c22286c0f193a77a"),
    ("data/non_hadoop_lz4_compressed.parquet", "32fd9bbeffcad29dbefa73f46d0a88d0abd220ad6eeb80e5090ad8fa20d2b901"),
    ("data/large_string_map.brotli.parquet", "1ce6839f093ebc0699b1e2769ed04036bab40405bacbb5dacdd376dd94c13451"),
    ("data/map_no_value.parquet", "5c4fc6c13fe7308acb2fd317a3bd59e5b9c9c206c005e863ae0a1abdbbf5e2ea"),
    ("data/incorrect_map_schema.parquet", "5591dde252b46bc238a88e9c02e35780c5eb086e2677df105aaa91ff1fde8fba"),
    ("data/datapage_v2.snappy.parquet", "44f29191b5fa8cfe0ab848495bd8ef89344ac0d8f87b3dff12e267631e2b5c03"),
    ("data/list_columns.parquet", "5988ab91b6cb7efa7bf6a77f789b40929212280519be6c9daad56e01d5ceb218"),
    ("data/nulls.snappy.parquet", "40192e879fe7905d1341b495d06f8470e2fd02608bf8f9e6a71b2b774acc5252"),
    ("data/repeated_primitive_no_list.parquet", "fcd6152058b8b8259a516105da5919b23cb8ccfc42258de0fe20e3107f8ef809"),
    ("data/uniform_encryption.parquet.encrypted", "b61382c4e9515b970f9aad8136b0e4c17b7803f8ab5e0f368a33b8d0d5ea04bf"),
    ("data/encrypt_columns_plaintext_footer.parquet.encrypted", "81c55445badc672251616ce3b57663b65a6fa45990c029a72fef3d247f0e0eeb"),
    ("bad_data/ARROW-GH-43605.parquet", "9f81fe00d9b9732ecb17375c6997c85f8ad52a90709ef3cd72dfcdb458fd7248"),
    ("bad_data/ARROW-GH-47662.parquet", "00804313753fa04c3ef662e436da221e9a61d1732db020cdfe351c188f240391"),
    ("bad_data/ARROW-RS-GH-6229-DICTHEADER.parquet", "cbc54d6738ea0bf9c941151b5b59cd71597beb0517d949539db59f99c18e9998"),
    ("bad_data/PARQUET-1481.parquet", "65dfd7e8fa9284a3e4c79bf72bb73e4297f457069df9f47cff5880701da6d71c"),
]


def fetch():
    (OUT / "bad_data").mkdir(parents=True, exist_ok=True)
    for path, sha in CORPUS + [("LICENSE.txt", None)]:
        name = path.removeprefix("data/")
        with urllib.request.urlopen(RAW + path) as r:
            data = r.read()
        if sha and hashlib.sha256(data).hexdigest() != sha:
            raise SystemExit(f"{path}: SHA-256 mismatch")
        (OUT / name).write_bytes(data)
    names = "\n".join(f"- `{p.removeprefix('data/')}`" for p, _ in CORPUS)
    (OUT / "README.md").write_text(
        f"Files of [apache/parquet-testing](https://github.com/apache/parquet-testing) at commit\n"
        f"`{COMMIT}`, under the Apache License 2.0 (`LICENSE.txt`):\n\n{names}\n\n"
        "The other files are written by `../gen/fixtures.py` with pyarrow.\n"
    )


# Generated files

ROWS = 300
rng = np.random.default_rng(7)


def nulls(a, p=0.25):
    """[a] with about [p] of its values null, alone and in runs."""
    mask = rng.random(ROWS) < p / 2
    for start in rng.integers(0, ROWS - 8, 4):
        mask[start : start + 8] = True
    if isinstance(a, pa.ExtensionArray):
        return pa.ExtensionArray.from_storage(a.type, nulls(a.storage, p))
    return pc.if_else(pa.array(mask), pa.nulls(ROWS, a.type), a)


def types_table():
    ints = lambda lo, hi, t: pa.array(rng.integers(lo, hi, ROWS, dtype=np.int64), t)
    us = rng.integers(-(2**50), 2**50, ROWS)
    cols = {
        "bool": pa.array(rng.random(ROWS) < 0.5),
        "i8": ints(-128, 128, pa.int8()),
        "i16": ints(-(2**15), 2**15, pa.int16()),
        "i32": ints(-(2**31), 2**31, pa.int32()),
        "i64": pa.array(rng.integers(-(2**63), 2**63 - 1, ROWS), pa.int64()),
        "u8": ints(0, 256, pa.uint8()),
        "u16": ints(0, 2**16, pa.uint16()),
        "u32": ints(0, 2**32, pa.uint32()),
        "u64": pa.array(rng.integers(0, 2**63, ROWS, dtype=np.uint64) * 2 + 1, pa.uint64()),
        "f16": pa.array(rng.standard_normal(ROWS).astype(np.float16)),
        "f32": pa.array(np.r_[[np.nan, -0.0, np.inf], rng.standard_normal(ROWS - 3)].astype(np.float32)),
        "f64": pa.array(np.r_[[-np.inf, 0.0, np.nan], rng.standard_normal(ROWS - 3)]),
        "d9": pa.array([Decimal(int(x)).scaleb(-2) for x in rng.integers(-(10**9) + 1, 10**9, ROWS)], pa.decimal128(9, 2)),
        "d18": pa.array([Decimal(int(x)).scaleb(-6) for x in rng.integers(-(10**18) + 1, 10**18, ROWS)], pa.decimal128(18, 6)),
        "date": pa.array(rng.integers(-800_000, 2_900_000, ROWS).astype(np.int32), pa.date32()),
        "clock_ms": pa.array(rng.integers(0, 86_400_000, ROWS).astype(np.int32), pa.time32("ms")),
        "clock_us": pa.array(rng.integers(0, 86_400_000_000, ROWS), pa.time64("us")),
        "clock_ns": pa.array(rng.integers(0, 86_400_000_000_000, ROWS), pa.time64("ns")),
        "ts_ms": pa.array(us // 1000, pa.timestamp("ms")),
        "ts_us_utc": pa.array(us, pa.timestamp("us", tz="UTC")),
        "ts_us_paris": pa.array(us, pa.timestamp("us", tz="Europe/Paris")),
        "ts_ns": pa.array(us * 7, pa.timestamp("ns")),
        "string": pa.array([f"v{rng.integers(0, 40)}é\n\"" for _ in range(ROWS)]),
        "binary": pa.array([bytes(rng.integers(0, 256, rng.integers(0, 9), dtype=np.uint8)) for _ in range(ROWS)]),
        "fsb": pa.array([bytes(rng.integers(0, 256, 5, dtype=np.uint8)) for _ in range(ROWS)], pa.binary(5)),
        "uuid": pa.ExtensionArray.from_storage(pa.uuid(), pa.array([bytes(rng.integers(0, 256, 16, dtype=np.uint8)) for _ in range(ROWS)], pa.binary(16))),
    }
    cols = {k: nulls(v) for k, v in cols.items()}
    cols["all_null"] = pa.nulls(ROWS, pa.int32())
    cols["no_null"] = pa.array(rng.integers(0, 10, ROWS), pa.int64())
    fields = [pa.field(k, v.type) for k, v in cols.items()]
    fields += [pa.field("req_i32", pa.int32(), nullable=False), pa.field("req_string", pa.string(), nullable=False)]
    cols["req_i32"] = pa.array(rng.integers(-5, 5, ROWS), pa.int32())
    cols["req_string"] = pa.array([f"r{i % 7}" for i in range(ROWS)])
    return pa.table(list(cols.values()), schema=pa.schema(fields))


def decimals_table():
    """Decimals of 9, 18 and 38 digits at scale 0 and above, with the extremes of
    each precision, zero, a value of each sign and nulls."""
    r = np.random.default_rng(11)

    def drawn(p):
        digits = "".join(str(d) for d in r.integers(0, 10, int(r.integers(1, p + 1))))
        return int(digits) * (1 if r.random() < 0.5 else -1)

    def column(p, s):
        top = 10**p - 1
        units = [top, -top, 0, 1, -1, None] + [drawn(p) for _ in range(ROWS - 6)]
        values = [None if u is None else Decimal(f"{u}E-{s}") for u in units]
        return pa.array(values, pa.decimal128(p, s))

    return pa.table({f"d{p}_{s}": column(p, s) for p, s in [(9, 0), (9, 2), (18, 0), (18, 6), (38, 0), (38, 10)]})


BIT_ROWS = 2001


def bits_table():
    """Columns whose levels and booleans are bit-packed: runs of each length
    around a byte and a word, alternating valid and null, then random rows, a
    run of nulls and a run of values longer than a page; booleans in runs of
    the same lengths, then random. An own generator keeps the other fixtures'
    draws."""
    r = np.random.default_rng(23)
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65, 127, 128, 129]

    def runs(first):
        return np.concatenate([np.full(n, (k % 2 == 0) == first) for k, n in enumerate(lengths)])

    ran = runs(True)
    tail = BIT_ROWS - len(ran) - 600
    valid = np.concatenate([ran, r.random(tail) < 0.5, np.zeros(300, bool), np.ones(300, bool)])
    values = np.concatenate([runs(False), r.random(BIT_ROWS - len(ran)) < 0.5])
    mask = pa.array(~valid)
    masked = lambda a: pc.if_else(mask, pa.nulls(BIT_ROWS, a.type), a)
    return pa.table(
        {
            "b": masked(pa.array(values)),
            "f": masked(pa.array(r.standard_normal(BIT_ROWS))),
            "s": masked(pa.array([f"s{k}" for k in r.integers(0, 50, BIT_ROWS)])),
            "b_none": pa.nulls(BIT_ROWS, pa.bool_()),
            "b_full": pa.array(values),
            "b_req": pa.array(values),
        },
        schema=pa.schema(
            [pa.field("b", pa.bool_()), pa.field("f", pa.float64()), pa.field("s", pa.string()),
             pa.field("b_none", pa.bool_()), pa.field("b_full", pa.bool_()),
             pa.field("b_req", pa.bool_(), nullable=False)]
        ),
    )


def write(name, table, **kw):
    kw = dict(store_schema=False, row_group_size=100, write_batch_size=16, data_page_size=256) | kw
    pq.write_table(table, OUT / name, **kw)


def generate():
    t = types_table()
    write("types.parquet", t, data_page_version="1.0", compression="snappy")
    write("types_v2_zstd.parquet", t, data_page_version="2.0", compression="zstd", write_page_checksum=True)
    codecs = t.select(["bool", "i32", "f64", "d18", "string", "fsb", "req_string"])
    write("types_gzip.parquet", codecs, data_page_version="1.0", compression="gzip")
    write("types_lz4.parquet", codecs, data_page_version="2.0", compression="lz4")
    write("types_plain.parquet", t, use_dictionary=False, compression="none", store_decimal_as_integer=True)
    enc = t.select(["bool", "i8", "i32", "i64", "u64", "f16", "f32", "f64", "d9", "string", "fsb", "req_string"])
    column_encoding = {
        "i8": "DELTA_BINARY_PACKED", "i32": "DELTA_BINARY_PACKED", "i64": "DELTA_BINARY_PACKED",
        "u64": "BYTE_STREAM_SPLIT", "f16": "BYTE_STREAM_SPLIT", "f32": "BYTE_STREAM_SPLIT",
        "f64": "BYTE_STREAM_SPLIT", "d9": "BYTE_STREAM_SPLIT", "string": "DELTA_BYTE_ARRAY",
        "fsb": "DELTA_BYTE_ARRAY", "req_string": "DELTA_LENGTH_BYTE_ARRAY",
    }
    for name, compression in [("encodings.parquet", "zstd"), ("encodings_uncompressed.parquet", "none")]:
        write(
            name, enc, data_page_version="2.0", use_dictionary=False, compression=compression,
            store_decimal_as_integer=True, column_encoding=column_encoding,
        )
    write("fallback.parquet", t.select(["i8", "f64", "string", "binary"]), dictionary_pagesize_limit=64, compression="lz4")
    names = pa.table({"a b": [1, 2], '"q"': [3, 4], "é": [5, 6], "": [7, 8]})
    write("names.parquet", names)
    write("empty.parquet", t.slice(0, 0))
    write("decimals_int.parquet", decimals_table(), store_decimal_as_integer=True)
    write("decimals_bytes.parquet", decimals_table())
    write("decimal38.parquet", pa.table({"d": pa.array([Decimal("1.5")], pa.decimal128(38, 10))}))
    write("brotli.parquet", t.select(["i32"]), compression="brotli")
    write("int96.parquet", t.select(["ts_ns"]), use_deprecated_int96_timestamps=True)
    write("bad_encoding.parquet", t.select(["i32"]), use_dictionary=False, compression="none")
    bad_encoding()
    bits = bits_table()
    # Odd row groups and pages that start inside a byte of the validity.
    small = dict(row_group_size=731, write_batch_size=13, data_page_size=16)
    write("bits_v1.parquet", bits, data_page_version="1.0", use_dictionary=False, compression="none", **small)
    booleans = {c: "RLE" for c in ["b", "b_none", "b_full", "b_req"]}
    write(
        "bits_v2_rle.parquet", bits, data_page_version="2.0", use_dictionary=False, compression="none",
        column_encoding=booleans, **small,
    )
    write("bits_dict.parquet", bits, data_page_version="1.0", compression="snappy", **small)
    assert set(encodings("bits_v1.parquet", "b")) == {"PLAIN"}
    assert set(encodings("bits_v2_rle.parquet", "b")) == {"RLE"}
    assert "RLE_DICTIONARY" in encodings("bits_dict.parquet", "s")
    assert set(encodings("fallback.parquet", "i8")) >= {"DICT", "RLE_DICTIONARY", "PLAIN"}
    assert "DELTA_BYTE_ARRAY" in encodings("encodings.parquet", "fsb")


# A Thrift compact reader, enough to walk page headers.


class Reader:
    def __init__(self, b, p):
        self.b, self.p, self.marks = b, p, {}

    def byte(self):
        self.p += 1
        return self.b[self.p - 1]

    def varint(self):
        v = shift = 0
        while True:
            c = self.byte()
            v |= (c & 0x7F) << shift
            shift += 7
            if c < 0x80:
                return v

    def struct(self, path=()):
        d, last = {}, 0
        while (h := self.byte()) != 0:
            fid = last + (h >> 4) if h >> 4 else self.zigzag()
            last = fid
            self.marks[path + (fid,)] = self.p
            d[fid] = self.value(h & 0xF, path + (fid,))
        return d

    def zigzag(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)

    def value(self, t, path):
        if t in (1, 2):
            return t == 1
        if t == 3:
            return self.byte()
        if t in (4, 5, 6):
            return self.zigzag()
        if t == 7:
            self.p += 8
            return None
        if t == 8:
            n = self.varint()
            self.p += n
            return self.b[self.p - n : self.p]
        if t in (9, 10):
            h = self.byte()
            n = self.varint() if h >> 4 == 15 else h >> 4
            return [self.value(h & 0xF, path) for _ in range(n)]
        if t == 12:
            return self.struct(path)
        raise ValueError(f"wire type {t}")


ENCODINGS = {0: "PLAIN", 2: "PLAIN_DICTIONARY", 3: "RLE", 5: "DELTA_BINARY_PACKED",
             6: "DELTA_LENGTH_BYTE_ARRAY", 7: "DELTA_BYTE_ARRAY", 8: "RLE_DICTIONARY", 9: "BYTE_STREAM_SPLIT"}


def pages(name, column):
    """The page headers of [column] in the first row group of [name], with the
    position of each header's encoding."""
    b = (OUT / name).read_bytes()
    md = pq.ParquetFile(OUT / name).metadata.row_group(0)
    c = next(md.column(i) for i in range(md.num_columns) if md.column(i).path_in_schema == column)
    p = c.dictionary_page_offset or c.data_page_offset
    while p < (c.dictionary_page_offset or c.data_page_offset) + c.total_compressed_size:
        r = Reader(b, p)
        h = r.struct()
        yield h, r.marks
        p = r.p + h[3]


def encodings(name, column):
    out = []
    for h, _ in pages(name, column):
        if h[1] == 2:
            out.append("DICT")
        else:
            header = h.get(5) or h.get(8)
            out.append(ENCODINGS[header[2 if 5 in h else 4]])
    return out


def bad_encoding():
    """bad_encoding.parquet with the encoding of its first data page set to 15,
    which Parquet does not define."""
    b = bytearray((OUT / "bad_encoding.parquet").read_bytes())
    h, marks = next(pages("bad_encoding.parquet", "i32"))
    pos = marks[(5, 2)]
    assert b[pos] == 0  # PLAIN, zigzag-encoded in one byte
    b[pos] = 30
    (OUT / "bad_encoding.parquet").write_bytes(bytes(b))


# Expected values


def quote(b):
    if isinstance(b, str):
        b = b.encode()
    return '"' + "".join(chr(c) if 0x20 <= c <= 0x7E and c not in b'"\\' else f"\\x{c:02x}" for c in b) + '"'


UNITS = {"milliseconds": "ms", "microseconds": "us", "nanoseconds": "ns"}


def talon_type(c):
    """The talon type of the Parquet column [c], or None if talon refuses it."""
    phys = c.physical_type
    lt = json.loads(c.logical_type.to_json()) if c.logical_type.type != "NONE" else {"Type": None}
    kind = lt["Type"]
    if kind in ("String", "Enum", "JSON") and phys == "BYTE_ARRAY":
        return "string"
    if kind == "Decimal":
        p, s = lt["precision"], lt["scale"]
        ok = {"INT32": p <= 9, "INT64": p <= 18, "BYTE_ARRAY": True, "FIXED_LEN_BYTE_ARRAY": True}.get(phys, False)
        if ok and 1 <= p and 0 <= s <= p:
            return "float64"
    if kind == "Date" and phys == "INT32":
        return "date"
    if kind == "Time":
        u = UNITS[lt["timeUnit"]]
        if (u == "ms") == (phys == "INT32") and phys in ("INT32", "INT64"):
            return f"clock[{u}]"
    if kind == "Timestamp" and phys == "INT64":
        u = UNITS[lt["timeUnit"]]
        return f"datetime[{u}, UTC]" if lt["isAdjustedToUTC"] else f"datetime[{u}]"
    if kind == "Int":
        w, signed = lt["bitWidth"], lt["isSigned"]
        if (w == 64) == (phys == "INT64") and phys in ("INT32", "INT64"):
            return f"{'' if signed else 'u'}int{w}"
    if kind == "Float16" and phys == "FIXED_LEN_BYTE_ARRAY" and c.length == 2:
        return "float16"
    return {"BOOLEAN": "bool", "INT32": "int32", "INT64": "int64", "INT96": "datetime[ns]",
            "FLOAT": "float32", "DOUBLE": "float64"}.get(phys, "binary")


def render(ty, a):
    """The values of the pyarrow array [a] as talon stores [ty]."""
    if isinstance(a, pa.ExtensionArray):
        a = a.storage
    if pa.types.is_decimal(a.type):
        if ty == "int64":
            return [None if v is None else str(int(v.scaleb(a.type.scale))) for v in a.to_pylist()]
        bits = lambda v: struct.unpack("<Q", struct.pack("<d", float(v)))[0]
        return [None if v is None else f"0x{bits(v):016x}" for v in a.to_pylist()]
    if ty.startswith(("date", "clock", "datetime")):
        a = a.view(pa.int32() if a.type.bit_width == 32 else pa.int64())
    if ty.startswith("float"):
        bits = {"float16": np.uint16, "float32": np.uint32, "float64": np.uint64}[ty]
        width = {"float16": 4, "float32": 8, "float64": 16}[ty]
        vals = a.to_numpy(zero_copy_only=False)
        mask = a.is_null().to_numpy(zero_copy_only=False)
        return [None if m else f"0x{int(v):0{width}x}" for v, m in zip(vals.view(bits) if vals.dtype != object else vals, mask)]
    out = []
    for v in a.to_pylist():
        if v is None:
            out.append(None)
        elif isinstance(v, bool):
            out.append("true" if v else "false")
        elif isinstance(v, (bytes, str)):
            out.append(quote(v))
        else:
            out.append(str(v))
    return out


def line(name, g, column, ty, values):
    vals = ["null" if v is None else v for v in values]
    md5 = hashlib.md5(" ".join(vals).encode()).hexdigest()
    return f"{name} {g} {quote(column)} {quote(ty)} {len(vals)} {md5} {' '.join(vals[:8])}".rstrip()


def outside_ns(path, g, column):
    us = pq.ParquetFile(path, coerce_int96_timestamp_unit="us").read_row_group(g, columns=[column])
    return any(v is not None and not -(2**63) <= v * 1000 < 2**63 for v in us.column(0).cast(pa.int64()).to_pylist())


def expect(name):
    path = OUT / name
    pf = pq.ParquetFile(path, page_checksum_verification=True)
    md = pf.metadata
    lines = []
    for g in range(md.num_row_groups):
        for i in range(md.num_columns):
            c = md.schema.column(i)
            ty = talon_type(c)
            try:
                a = pf.read_row_group(g, columns=[c.name]).column(0).combine_chunks()
                if ty == "datetime[ns]" and c.physical_type == "INT96" and outside_ns(path, g, c.name):
                    raise ValueError("outside datetime[ns]")
            except (pa.ArrowException, OSError, ValueError):
                lines.append(f"{name} {g} {quote(c.name)} error")
            else:
                lines.append(line(name, g, c.name, ty, render(ty, a)))
    return lines


READ = [p.removeprefix("data/") for p, _ in CORPUS if not p.endswith(".encrypted")] + [
    "types.parquet", "types_v2_zstd.parquet", "types_gzip.parquet", "types_lz4.parquet",
    "types_plain.parquet", "encodings.parquet", "encodings_uncompressed.parquet", "fallback.parquet", "names.parquet", "empty.parquet",
    "int96.parquet", "decimals_int.parquet", "decimals_bytes.parquet", "decimal38.parquet",
    "bits_v1.parquet", "bits_v2_rle.parquet", "bits_dict.parquet",
]
# Refused by talon when sniffed; read with errors by talon where pyarrow guesses
# (nation.dict-malformed's chunks are longer than their metadata say); or not
# opened by pyarrow and computed by hand.
SKIP = {
    "bad_data/ARROW-RS-GH-6229-DICTHEADER.parquet", "nation.dict-malformed.parquet",
    "hadoop_lz4_compressed.parquet", "non_hadoop_lz4_compressed.parquet", "large_string_map.brotli.parquet",
    "map_no_value.parquet", "incorrect_map_schema.parquet", "datapage_v2.snappy.parquet",
    "list_columns.parquet", "nulls.snappy.parquet", "repeated_primitive_no_list.parquet",
    "bad_data/PARQUET-1481.parquet", "int32_with_uuid_logical_type.parquet", "flba12_timestamp.parquet",
}


def by_hand():
    lines = [line("int32_with_uuid_logical_type.parquet", 0, "int32_uuid", "int32", [str(i) for i in range(10)])]
    seconds = [0, 1, -1, 9_223_372_036, 253_402_300_799, -62_135_596_800]
    for column, scale in [("timestamp_millis", 10**3), ("timestamp_micros", 10**6), ("timestamp_nanos", 10**9)]:
        values = [quote((s * scale).to_bytes(12, "little", signed=True)) for s in seconds]
        lines.append(line("flba12_timestamp.parquet", 0, column, "binary", values))
    return lines


def unscaled(name):
    """The lines of the decimals of at most 18 digits of [name], read as int64."""
    pf = pq.ParquetFile(OUT / name)
    md = pf.metadata
    lines = []
    for g in range(md.num_row_groups):
        for i in range(md.num_columns):
            c = md.schema.column(i)
            lt = json.loads(c.logical_type.to_json()) if c.logical_type.type != "NONE" else {"Type": None}
            if talon_type(c) == "float64" and lt["Type"] == "Decimal" and lt["precision"] <= 18:
                a = pf.read_row_group(g, columns=[c.name]).column(0).combine_chunks()
                lines.append(line(name, g, c.name, "int64", render("int64", a)))
    return lines


def overrides():
    us = pq.ParquetFile(OUT / "alltypes_plain.parquet", coerce_int96_timestamp_unit="us").read()
    plain = pq.read_table(OUT / "alltypes_plain.parquet")
    column = lambda t, c: t.column(c).combine_chunks()
    return [
        line("alltypes_plain.parquet", 0, "timestamp_col", "datetime[us]", render("datetime[us]", column(us, "timestamp_col"))),
        line("alltypes_plain.parquet", 0, "string_col", "string", render("string", column(plain, "string_col"))),
    ] + [l for name in READ if name not in SKIP for l in unscaled(name)]


def main():
    fetch()
    generate()
    lines = [l for name in READ if name not in SKIP for l in expect(name)] + by_hand()
    (HERE.parent / "golden" / "values.txt").write_text("\n".join(lines) + "\n")
    (HERE.parent / "golden" / "overrides.txt").write_text("\n".join(overrides()) + "\n")


if __name__ == "__main__":
    main()
