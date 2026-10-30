#!/usr/bin/env python3
"""Write lz's goldens with the vendors, or check lz's encoders against them.

Every golden is a compressed file a vendor wrote, listed in golden/manifest.txt
as `file codec vendor members raw`: `codec` is the lz decoder that reads
it, `members` the members or frames it holds, `raw` the input, as `kind:n`
parts joined by `+`, each the first n bytes of a corpus that
test/support/lz_support.ml builds byte for byte. The raw inputs are not
checked in.

    uv run --no-project --with deflate --with zstandard --with lz4 \
        --with python-snappy python -I packages/lz/test/gen/generate.py

With --check, it encodes every raw input at every level with lz's encoders,
through test/gen/encode.exe, and decodes each result with the vendor.
Build the executable first:

    dune build packages/lz/test/gen/encode.exe
    uv run ... python -I packages/lz/test/gen/generate.py --check

The check is run before each push; it is not part of runtest.
"""

import gzip
import os
import struct
import subprocess
import sys
import zlib

import deflate
import lz4.block
import lz4.frame
import snappy
import zstandard

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "..", "golden")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ENCODE = os.path.join(ROOT, "_build", "default", "packages", "lz", "test",
                      "gen", "encode.exe")

MASK = (1 << 64) - 1
KIB = 1024
MIB = 1024 * KIB


# Corpora: the generators of bench/reference.py, as prefixes.

def xorshift(seed):
    x = seed
    while True:
        x ^= (x << 13) & MASK
        x ^= x >> 7
        x ^= (x << 17) & MASK
        yield x


SYLLABLES = ["ka", "lo", "mi", "ne", "su", "ta", "ri", "po", "an", "el",
             "or", "is", "um", "ve", "da", "go"]


def text(n):
    gen = xorshift(0x51F15EED)
    words = []
    for _ in range(512):
        k = next(gen) % 3 + 1
        words.append("".join(SYLLABLES[next(gen) % 16] for _ in range(k)))
    out, size = [], 0
    while size < n:
        w = words[next(gen) % 512] + ("\n" if next(gen) % 12 == 0 else " ")
        out.append(w)
        size += len(w)
    return "".join(out).encode()[:n]


def columns(n):
    gen = xorshift(0x9E3779B97F4A7C15)
    key, out = 0, bytearray()
    while len(out) < n:
        key += next(gen) % 16 + 1
        out += struct.pack("<qd", key, (next(gen) % 100000) / 100)
    return bytes(out[:n])


def random_bytes(n):
    gen = xorshift(0x2545F4914F6CDD1D)
    out = bytearray()
    while len(out) < n:
        out += struct.pack("<Q", next(gen))
    return bytes(out[:n])


KINDS = {"text": text, "columns": columns, "random": random_bytes}
CACHE = {}


def corpus(kind, n):
    s = CACHE.get(kind)
    if s is None or len(s) < n:
        s = KINDS[kind](n)
        CACHE[kind] = s
    return s[:n]


def raw(spec):
    out = b""
    for part in spec.split("+"):
        kind, n = part.split(":")
        out += corpus(kind, int(n))
    return out


# Framing the specifications define, for the files no vendor call writes.

def skippable(payload, nibble=0):
    """A skippable frame (RFC 8878 section 3.1.2; LZ4 frame format)."""
    return struct.pack("<II", 0x184D2A50 + nibble, len(payload)) + payload


def gzip_member(data, ftext=False, extra=None, name=None, comment=None,
                hcrc=False):
    """A gzip member (RFC 1952 section 2.3) around libdeflate's deflate."""
    flg = ((1 if ftext else 0) | (2 if hcrc else 0) | (4 if extra else 0)
           | (8 if name else 0) | (16 if comment else 0))
    h = bytes([0x1F, 0x8B, 8, flg]) + struct.pack("<I", 0) + bytes([0, 3])
    if extra is not None:
        h += struct.pack("<H", len(extra)) + extra
    if name is not None:
        h += name + b"\0"
    if comment is not None:
        h += comment + b"\0"
    if hcrc:
        h += struct.pack("<H", zlib.crc32(h) & 0xFFFF)
    body = deflate.deflate_compress(data, 6)
    return h + body + struct.pack("<II", zlib.crc32(data), len(data) & 0xFFFFFFFF)


# Encoders: name -> (codec, vendor, function).

def raw_deflate(level):
    def f(d):
        c = zlib.compressobj(level, zlib.DEFLATED, -15)
        return c.compress(d) + c.flush()
    return f


def zstd(level, **kw):
    return zstandard.ZstdCompressor(level=level, **kw).compress


def lz4f(**kw):
    return lambda d: lz4.frame.compress(d, **kw)


# Every input of SIZES of every corpus.
CORE = {
    "zlib6": ("zlib", "zlib-6", lambda d: zlib.compress(d, 6)),
    "gz9": ("gzip", "gzip-9", lambda d: gzip.compress(d, 9, mtime=0)),
    "ld6.deflate": ("deflate", "libdeflate-6",
                    lambda d: deflate.deflate_compress(d, 6)),
    "zst3": ("zstd", "zstd-3", zstd(3, write_checksum=True)),
    "lz4b": ("lz4b", "lz4-block",
             lambda d: lz4.block.compress(d, store_size=False)),
    "lz4f": ("lz4f", "lz4-frame", lz4f()),
    "snappy": ("snappy", "snappy", snappy.compress),
}

# Every input of the small SIZES.
EXTRA = {
    "zlib1": ("zlib", "zlib-1", lambda d: zlib.compress(d, 1)),
    "zlib9": ("zlib", "zlib-9", lambda d: zlib.compress(d, 9)),
    "deflate6": ("deflate", "zlib-raw-6", raw_deflate(6)),
    "gz1": ("gzip", "gzip-1", lambda d: gzip.compress(d, 1, mtime=0)),
    "ld1.deflate": ("deflate", "libdeflate-1",
                    lambda d: deflate.deflate_compress(d, 1)),
    "ld12.deflate": ("deflate", "libdeflate-12",
                     lambda d: deflate.deflate_compress(d, 12)),
    "ld9.zlib": ("zlib", "libdeflate-9", lambda d: deflate.zlib_compress(d, 9)),
    "ld6.gz": ("gzip", "libdeflate-6", lambda d: deflate.gzip_compress(d, 6)),
    "zst1": ("zstd", "zstd-1", zstd(1, write_checksum=True)),
    "zst12": ("zstd", "zstd-12", zstd(12, write_checksum=True)),
    "zst19": ("zstd", "zstd-19", zstd(19, write_checksum=True)),
    "zst3-nocheck": ("zstd", "zstd-3-nochecksum", zstd(3)),
    "zst3-nosize": ("zstd", "zstd-3-nosize",
                    zstd(3, write_checksum=True, write_content_size=False)),
    "lz4f-independent": ("lz4f", "lz4-frame-independent",
                         lz4f(block_linked=False)),
    "lz4f-checksums": ("lz4f", "lz4-frame-checksums",
                       lz4f(block_checksum=True, content_checksum=True)),
    "lz4f-nosize": ("lz4f", "lz4-frame-nosize", lz4f(store_size=False)),
    "lz4f-hc": ("lz4f", "lz4-frame-hc9", lz4f(compression_level=9)),
}

# Goldens stop at 200000 bytes, so that the checked-in files stay small;
# the check also encodes 5 MiB inputs, whose vendor-written forms the
# benches decode.
SIZES = [0, 1, 100, 65535, 65536, 200000]
SMALL = [0, 1, 100, 65536]
CHECKED = SIZES + [5 * MIB]


def inputs(sizes, noise=max(CHECKED)):
    for kind in ("text", "columns", "random"):
        for n in sizes:
            if kind != "random" or n <= noise:
                yield "%s:%d" % (kind, n)


def specials():
    """Files that join members, frame a member by hand, or set a window,
    each with the number of members it holds."""
    t100, c64k, t200k = "text:100", "columns:65536", "text:200000"
    # Streamed, so that the frame states its 1 MiB window rather than a
    # content size.
    zw20 = zstandard.ZstdCompressor(
        compression_params=zstandard.ZstdCompressionParameters.from_level(
            3, window_log=20, write_checksum=1)).compressobj()
    return [
        ("members.gz", "gzip", "gzip-6-twice", 2, t100 + "+" + c64k,
         gzip.compress(raw(t100), mtime=0) + gzip.compress(raw(c64k), mtime=0)),
        ("fields.gz", "gzip", "rfc1952-fields", 1, t100,
         gzip_member(raw(t100), ftext=True, extra=b"lz\x04\x00ab\x00c",
                     name=b"text.txt", comment=b"made by generate.py",
                     hcrc=True)),
        ("empty-members.gz", "gzip", "gzip-6-empty-thrice", 3, "text:0",
         b"".join(gzip.compress(b"", mtime=0) for _ in range(3))),
        ("text_200000.zst1", "zstd", "zstd-1", 1, t200k,
         zstd(1, write_checksum=True)(raw(t200k))),
        ("text_200000.zst19", "zstd", "zstd-19", 1, t200k,
         zstd(19, write_checksum=True)(raw(t200k))),
        ("frames.zst", "zstd", "zstd-3-skippable", 3, t100 + "+" + c64k,
         zstd(3, write_checksum=True)(raw(t100)) + skippable(b"lz" * 10, 7)
         + zstd(3)(raw(c64k))),
        ("skippable.zst", "zstd", "skippable-only", 2, "text:0",
         skippable(b"") + skippable(b"x" * 300, 15)),
        ("window1m.zst", "zstd", "zstd-3-window-1MiB", 1, t200k,
         zw20.compress(raw(t200k)) + zw20.flush()),
        ("frames.lz4", "lz4f", "lz4-frame-twice-skippable", 3, t100 + "+" + c64k,
         lz4f()(raw(t100)) + skippable(b"lz4") + lz4f(block_linked=False)(
             raw(c64k))),
        ("linked64k.lz4", "lz4f", "lz4-frame-linked-64KiB", 1, t200k,
         lz4f(block_size=lz4.frame.BLOCKSIZE_MAX64KB, block_linked=True,
              content_checksum=True)(raw(t200k))),
        ("block4m.lz4", "lz4f", "lz4-frame-4MiB-blocks", 1, t200k,
         lz4f(block_size=lz4.frame.BLOCKSIZE_MAX4MB, block_checksum=True)(
             raw(t200k))),
    ]


def golden_name(spec, ext):
    return spec.replace(":", "_") + "." + ext


# Goldens another script writes, by manifest vendor: google_snappy.py writes
# Google's snappy 1.2 bytes, which python-snappy's Rust codec does not.
OTHERS = {"snappy-1.2"}


def write():
    os.makedirs(GOLDEN, exist_ok=True)
    manifest = os.path.join(GOLDEN, "manifest.txt")
    kept = []
    if os.path.exists(manifest):
        with open(manifest) as fh:
            kept = [l for l in fh.read().splitlines()
                    if l and l.split(" ")[2] in OTHERS]
    keep = {l.split(" ")[0] for l in kept}
    for f in os.listdir(GOLDEN):
        if f not in keep:
            os.remove(os.path.join(GOLDEN, f))
    rows = []

    def emit(name, codec, vendor, spec, data, members=1):
        with open(os.path.join(GOLDEN, name), "wb") as fh:
            fh.write(data)
        rows.append("%s %s %s %d %s" % (name, codec, vendor, members, spec))

    # Incompressible input costs its size in every file: the variants take
    # it only up to 100 bytes.
    for table, sizes, noise in ((CORE, SIZES, 200000), (EXTRA, SMALL, 100)):
        for ext, (codec, vendor, f) in table.items():
            for spec in inputs(sizes, noise):
                emit(golden_name(spec, ext), codec, vendor, spec, f(raw(spec)))
    for name, codec, vendor, members, spec, data in specials():
        emit(name, codec, vendor, spec, data, members)
    with open(manifest, "w") as fh:
        fh.write("\n".join(rows + kept) + "\n")
    verify(rows + kept)
    total = sum(os.path.getsize(os.path.join(GOLDEN, f))
                for f in os.listdir(GOLDEN))
    print("%d goldens, %.1f MiB" % (len(rows), total / MIB))


# Each golden must decode to its raw input with the vendor that reads it.

def lz4_block_decode(data, n):
    return lz4.block.decompress(data, uncompressed_size=n) if n else b""


def zstd_decode(data):
    out = b""
    with zstandard.ZstdDecompressor().stream_reader(
            data, read_across_frames=True) as r:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                return out
            out += chunk


def lz4f_decode(data):
    out, rest = b"", data
    while rest:
        magic = struct.unpack("<I", rest[:4])[0]
        if magic & 0xFFFFFFF0 == 0x184D2A50:
            n = struct.unpack("<I", rest[4:8])[0]
            rest = rest[8 + n:]
            continue
        d = lz4.frame.LZ4FrameDecompressor()
        out += d.decompress(rest)
        rest = d.unused_data
    return out


def vendor_decode(codec, data, n):
    if codec == "deflate":
        return zlib.decompress(data, -15)
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "gzip":
        return gzip.decompress(data)
    if codec == "zstd":
        return zstd_decode(data)
    if codec == "lz4b":
        return lz4_block_decode(data, n)
    if codec == "lz4f":
        return lz4f_decode(data)
    if codec == "snappy":
        return snappy.uncompress(data)
    raise ValueError(codec)


def verify(rows):
    for row in rows:
        name, codec, _, _, spec = row.split(" ")
        with open(os.path.join(GOLDEN, name), "rb") as fh:
            data = fh.read()
        expected = raw(spec)
        if vendor_decode(codec, data, len(expected)) != expected:
            sys.exit("%s does not decode to %s" % (name, spec))


# --check: lz's encoders, decoded by the vendors.

LEVELS = {
    "deflate": range(1, 10), "zlib": range(1, 10), "gzip": range(1, 10),
    "zstd": range(1, 13), "lz4b": [0], "lz4f": [0], "snappy": [0],
}


def check():
    if not os.path.exists(ENCODE):
        sys.exit("missing %s: dune build packages/lz/test/gen/encode.exe" % ENCODE)
    failed = 0
    for codec, levels in LEVELS.items():
        for level in levels:
            for spec in inputs(CHECKED):
                failed += check_one(codec, level, spec)
    print("check: %d failures" % failed)
    sys.exit(1 if failed else 0)


def check_one(codec, level, spec):
    p = subprocess.run([ENCODE, codec, str(level), spec], capture_output=True)
    if p.returncode != 0:
        print("FAIL %s %d %s: %s" % (codec, level, spec,
                                     p.stderr.decode().strip()))
        return 1
    expected = raw(spec)
    try:
        if vendor_decode(codec, p.stdout, len(expected)) == expected:
            return 0
        print("FAIL %s %d %s: the vendor decodes other bytes"
              % (codec, level, spec))
    except Exception as e:
        print("FAIL %s %d %s: the vendor refuses it: %s"
              % (codec, level, spec, e))
    return 1


if __name__ == "__main__":
    check() if "--check" in sys.argv[1:] else write()
