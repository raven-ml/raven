# /// script
# requires-python = ">=3.11"
# dependencies = ["xxhash==3.5.0"]
# ///
"""Writes golden/hash.txt, the test vectors of Expr.hash.

    uv run packages/talon/test/gen/hash.py [--check]

With --check nothing is written, and the run fails if golden/hash.txt would
change.

Each line is `kind value seed result`, the expected Expr.hash of one row as
Expr.hash's definition in talon.mli states it: the reference xxHash64 of the
row's encoded value, seeded, shifted right by two bits. `value` is, per kind:
a decimal integer; true or false; a float's binary64 bits in hex; a string's
bytes in hex; `null` for a null; ticks for temporal kinds; a record's fields
joined by `,`, nested fields in `(...)`.
"""

import argparse
import struct
import sys
from pathlib import Path

import xxhash

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "hash.txt"

U64 = 2**64
NULL_WORD = 0x9E3779B185EBCA87
CANONICAL_NAN = 0x7FF8000000000000
NS_PER_DAY = 86_400_000_000_000


def seed_word(seed):
    return seed % U64


def word(v, seed):
    return xxhash.xxh64((v % U64).to_bytes(8, "little"), seed=seed_word(seed)).intdigest()


def hash_bytes(data, seed):
    return xxhash.xxh64(data, seed=seed_word(seed)).intdigest()


def float_word(bits):
    value = struct.unpack("<d", struct.pack("<Q", bits))[0]
    if value != value:
        return CANONICAL_NAN
    if value == 0.0:
        return 0
    return bits


def null_hash(seed):
    return word(NULL_WORD, seed)


# Scalars as (kind, value text, 64-bit hash of seed).
scalars = []


def ints(kind, values):
    for v in values:
        scalars.append((kind, str(v), lambda s, v=v: word(v, s)))


ints("int8", [0, 1, -1, 127, -128])
ints("int16", [-32768, 32767, 1000])
ints("int32", [-2147483648, 2147483647, 12345678])
ints("int64", [0, 1, -1, 2**53, -(2**63), 2**63 - 1, 42])
ints("uint8", [0, 255])
ints("uint16", [0, 65535])
ints("uint32", [0, 4294967295])
ints("uint64", [0, 2**63, 2**64 - 1])

for b in [False, True]:
    v = 1 if b else 0
    scalars.append(("bool", "true" if b else "false", lambda s, v=v: word(v, s)))

for kind, bits in [
    ("float64", 0x0000000000000000),  # 0.
    ("float64", 0x8000000000000000),  # -0.
    ("float64", 0x3FF8000000000000),  # 1.5
    ("float64", 0x7FF0000000000000),  # infinity
    ("float64", 0xFFF0000000000000),  # neg_infinity
    ("float64", 0x7FF8000000000000),  # NaN
    ("float64", 0x7FF8000000000123),  # NaN with a payload
    ("float64", 0xFFF8000000000000),  # NaN with a sign
    ("float64", 0x0000000000000001),  # least subnormal
    ("float32", 0x3FB99999A0000000),  # 0.1f, as float64 bits
    ("float32", 0x8000000000000000),  # -0.
    ("float32", 0x7FF8000000000000),  # NaN
    ("float16", 0x3FF8000000000000),  # 1.5
    ("float16", 0x8000000000000000),  # -0.
    ("float16", 0x7FF8000000000000),  # NaN
]:
    scalars.append((kind, f"{bits:x}", lambda s, b=bits: word(float_word(b), s)))

lengths = [31, 32, 33, 40, 64, 100]
for text in [b"", b"a", b"abcd", b"abcdefgh", b"hello world, this is 32+ bytes long!!",
             "héllo wörld ∑ — 日本語".encode()] + [bytes(range(n)) for n in lengths]:
    for kind in ["string", "binary"]:
        scalars.append((kind, text.hex(), lambda s, t=text: hash_bytes(t, s)))
    scalars.append(("categorical", text.hex(), lambda s, t=text: hash_bytes(t, s)))
scalars.append(("binary", b"\x00\xff\xfe\x00".hex(),
                lambda s: hash_bytes(b"\x00\xff\xfe\x00", s)))

# Temporal kinds by value.
for days in [0, 1, -1, 19800]:
    scalars.append(("date", str(days), lambda s, v=days: word(v, s)))
for ticks in [0, 1_000_000_000, -5, 3_600]:
    for unit, per in [("ns", 1), ("us", 1000), ("ms", 1_000_000), ("s", 1_000_000_000)]:
        for kind in ["datetime", "duration", "clock"]:
            if kind == "clock" and (ticks < 0 or ticks * per >= NS_PER_DAY):
                continue
            scalars.append((f"{kind}_{unit}", str(ticks),
                            lambda s, v=ticks * per: word(v, s)))

# A tick past int64's nanoseconds wraps.
scalars.append(("wrap_s", "20000000000",
                lambda s: word(20000000000 * 1_000_000_000, s)))

scalars.append(("null", "null", null_hash))

# Records: [("a", int64), ("b", string)], and a record of it and an int32.
ab = [(7, b"x"), (None, b""), (-1, None), (None, None)]


def field_int(v, s):
    return null_hash(s) if v is None else word(v, s)


def field_text(t, s):
    return null_hash(s) if t is None else hash_bytes(t, s)


def rec_ab_hash(a, b, s):
    words = struct.pack("<QQ", field_int(a, s), field_text(b, s))
    return hash_bytes(words, s)


def ab_text(a, b):
    a_t = "null" if a is None else str(a)
    b_t = "null" if b is None else b.hex()
    return a_t + "," + b_t


records = []
for a, b in ab:
    records.append(("record", ab_text(a, b), lambda s, a=a, b=b: rec_ab_hash(a, b, s)))
for (a, b), c in [((1, b"z"), 5), ((None, None), None)]:
    def nested(s, a=a, b=b, c=c):
        inner = rec_ab_hash(a, b, s)
        words = struct.pack("<QQ", inner, field_int(c, s))
        return hash_bytes(words, s)
    records.append(("nested", f"({ab_text(a, b)}),{'null' if c is None else c}", nested))
# Field order matters: (a, b) and (b, a) are not one record.
records.append(("record", "1,62", lambda s: rec_ab_hash(1, b"b", s)))

# Records of no fields, and of 4, 5 and 9 integers, past xxHash64's 32-byte
# stripe.
records.append(("empty", "-", lambda s: hash_bytes(b"", s)))
for n in [4, 5, 9]:
    vs = list(range(1, n + 1))
    records.append(("wide", ",".join(map(str, vs)),
                    lambda s, vs=vs: hash_bytes(
                        struct.pack(f"<{len(vs)}Q", *(word(v, s) for v in vs)), s)))

seeds = [0, 1, -1, 2**62 - 1, -(2**62)]


def lines():
    out = []
    for kind, value, f in scalars + records:
        for seed in seeds:
            h = f(seed)
            out.append(f"{kind} {value} {seed} {h >> 2}")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    text = lines()
    if args.check:
        if GOLDEN.read_text() != text:
            sys.exit("golden/hash.txt is stale: run gen/hash.py")
        return
    GOLDEN.write_text(text)
    print(f"wrote {GOLDEN}: {text.count(chr(10))} lines")


if __name__ == "__main__":
    main()
