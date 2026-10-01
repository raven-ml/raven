#!/usr/bin/env python3
"""Generate the compress test fixtures with the reference implementations.

zlib and gzip write the deflate streams, python-snappy the Snappy blocks, lz4
the LZ4 blocks and frames, and zstandard the Zstandard frames. The inputs are
the corpora of compress_fixtures.ml, which builds the same bytes.

The golden_* files hold what this library's deflate encoder writes, which
test_deflate.ml pins; this script only checks that the references decode each
to its input. After a deliberate change to the encoder's output, rewrite each
file named in that test's "the encoder writes the pinned streams" table with
the stream its format, level and corpus give, for example from the
repository root in `dune utop otherlibs/compress`:

    Out_channel.with_open_bin
      "otherlibs/compress/test/fixtures/golden_text_level6.z" (fun oc ->
        output_string oc
          (Compress_deflate.Zlib.compress ~level:6 Compress_fixtures.text))

then run `generate.py --check`.

Usage:
    cd otherlibs/compress/test/fixtures
    uv run --with python-snappy --with lz4 --with zstandard python generate.py
"""

import gzip
import struct
import sys
import zlib

import lz4.block
import lz4.frame
import snappy
import zstandard

MASK = (1 << 64) - 1


def xorshift(seed):
    x = seed
    while True:
        x ^= (x << 13) & MASK
        x ^= x >> 7
        x ^= (x << 17) & MASK
        yield x


def lines(n):
    return "".join("line %d\n" % i for i in range(n)).encode()


def text():
    return lines(8000)


def columns():
    gen = xorshift(0x9E3779B97F4A7C15)
    key, out = 0, bytearray()
    for _ in range(2048):
        key += next(gen) % 16 + 1
        price = (next(gen) % 100000) / 100
        out += struct.pack("<qd", key, price)
    return bytes(out)


def runs():
    return bytes((i // 1000) % 256 for i in range(100000))


def random_bytes():
    gen = xorshift(0x2545F4914F6CDD1D)
    return bytes(next(gen) & 0xFF for _ in range(20000))


CORPORA = {"text": text, "columns": columns, "runs": runs, "random": random_bytes}


def write(name, data):
    with open(name, "wb") as f:
        f.write(data)


# Deflate, zlib and gzip


def gzip_member(data, header_extra=b"", flags=0, hcrc=False):
    header = b"\x1f\x8b\x08" + bytes([flags | (2 if hcrc else 0)]) + b"\0" * 4 + b"\0\xff"
    header += header_extra
    if hcrc:
        header += struct.pack("<H", zlib.crc32(header) & 0xFFFF)
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    body = c.compress(data) + c.flush()
    return header + body + struct.pack("<II", zlib.crc32(data), len(data) & 0xFFFFFFFF)


def deflate_fixtures():
    for name, data, level in [("empty", b"", 9),
                              ("fixed", b"hello, nx zlib!\n", 1),
                              ("stored", b"stored block\n", 0),
                              ("dynamic", lines(200), 9)]:
        write("zlib_%s.z" % name, zlib.compress(data, level))
    for level in (0, 1, 6, 9):
        write("zlib_level%d.z" % level, zlib.compress(text(), level))
    # Longer than a reader's 64 KiB input buffer once compressed.
    write("zlib_long.z", zlib.compress(lines(60000), 1))
    c = zlib.compressobj(9, zlib.DEFLATED, 9)
    write("zlib_window512.z", c.compress(text()) + c.flush())
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    write("deflate_text.deflate", c.compress(text()) + c.flush())
    write("gzip_python.gz", gzip.compress(text(), mtime=0))
    write("gzip_members.gz",
          gzip.compress(b"first member\n", mtime=0)
          + gzip.compress(b"second member\n", mtime=0))
    extra = struct.pack("<H", 6) + b"ab\x02\x00xy"
    write("gzip_flags.gz",
          gzip_member(b"flags\n", extra + b"name.txt\0" + b"a comment\0",
                      flags=0x04 | 0x08 | 0x10, hcrc=True))


# Snappy


def snappy_fixtures():
    for name, corpus in CORPORA.items():
        write("snappy_%s.snappy" % name, snappy.compress(corpus()))
    write("snappy_empty.snappy", snappy.compress(b""))


# LZ4


def lz4_fixtures():
    for name, corpus in CORPORA.items():
        write("lz4_%s.lz4b" % name, lz4.block.compress(corpus(), store_size=False))
    data = text()
    frames = {
        "64k_linked": dict(block_size=lz4.frame.BLOCKSIZE_MAX64KB,
                           block_linked=True, content_checksum=True,
                           store_size=True),
        "256k_independent": dict(block_size=lz4.frame.BLOCKSIZE_MAX256KB,
                                 block_linked=False, block_checksum=True),
        "1m": dict(block_size=lz4.frame.BLOCKSIZE_MAX1MB, store_size=False),
        "4m": dict(block_size=lz4.frame.BLOCKSIZE_MAX4MB, store_size=False,
                   content_checksum=True, block_checksum=True),
    }
    for name, options in frames.items():
        write("lz4_%s.lz4" % name, lz4.frame.compress(data, **options))
    write("lz4_random.lz4", lz4.frame.compress(random_bytes(), store_size=True))
    write("lz4_frames.lz4",
          lz4.frame.compress(b"first frame\n") + lz4.frame.compress(b"second frame\n"))
    write("lz4_skippable.lz4",
          struct.pack("<II", 0x184D2A53, 5) + b"skip!" + lz4.frame.compress(b"after\n"))
    write("lz4_empty.lz4", lz4.frame.compress(b""))


# Zstandard


def zstd_fixtures():
    data = text()
    for level in (-5, 1, 3, 19, 22):
        cctx = zstandard.ZstdCompressor(level=level, write_checksum=True)
        write("zstd_level%d.zst" % level, cctx.compress(data))
    cctx = zstandard.ZstdCompressor(level=3, write_checksum=False,
                                    write_content_size=False)
    write("zstd_columns.zst", cctx.compress(columns()))
    plain = zstandard.ZstdCompressor(level=3)
    write("zstd_random.zst", plain.compress(random_bytes()))
    write("zstd_rle.zst", plain.compress(b"\x07" * 300000))
    write("zstd_runs.zst", plain.compress(runs()))
    # Blocks after the first reuse its Huffman and FSE tables.
    write("zstd_blocks.zst", zstandard.ZstdCompressor(level=19).compress(lines(40000)))
    write("zstd_small.zst", plain.compress(b"hello, hello, zstandard!\n" * 4))
    params = zstandard.ZstdCompressionParameters.from_level(19, window_log=28)
    obj = zstandard.ZstdCompressor(compression_params=params).compressobj()
    write("zstd_window28.zst", obj.compress(data) + obj.flush())
    write("zstd_frames.zst",
          plain.compress(b"first frame\n") + plain.compress(b"second frame\n"))
    write("zstd_skippable.zst",
          struct.pack("<II", 0x184D2A5E, 3) + b"abc" + plain.compress(b"after\n"))
    write("zstd_empty.zst", plain.compress(b""))
    samples = [("sample %d with some shared words\n" % i).encode() * 3
               for i in range(200)]
    dictionary = zstandard.train_dictionary(1024, samples)
    with_dict = zstandard.ZstdCompressor(level=3, dict_data=dictionary)
    write("zstd_dictionary.zst", with_dict.compress(b"sample 7 with some shared words\n"))


# Coverage of the Zstandard block types, literal modes and sequence modes


def zstd_modes(frame):
    """The block types, literals types and sequence modes in [frame]."""
    seen = set()
    pos = 0
    while pos < len(frame):
        magic = struct.unpack_from("<I", frame, pos)[0]
        if magic & 0xFFFFFFF0 == 0x184D2A50:
            pos += 8 + struct.unpack_from("<I", frame, pos + 4)[0]
            continue
        fhd = frame[pos + 4]
        single = fhd >> 5 & 1
        did = [0, 1, 2, 4][fhd & 3]
        fcs = [1 if single else 0, 2, 4, 8][fhd >> 6]
        pos += 5 + (0 if single else 1) + did + fcs
        while True:
            header = int.from_bytes(frame[pos:pos + 3], "little")
            last, btype, size = header & 1, header >> 1 & 3, header >> 3
            pos += 3
            seen.add(("block", btype))
            if btype == 2:
                body = frame[pos:pos + size]
                lt = body[0] & 3
                seen.add(("literals", lt))
                sf = body[0] >> 2 & 3
                if lt < 2:
                    lsize, hl = [(body[0] >> 3, 1), (int.from_bytes(body[:2], "little") >> 4, 2),
                                 (body[0] >> 3, 1), (int.from_bytes(body[:3], "little") >> 4, 3)][sf]
                    off = hl + (lsize if lt == 0 else 1)
                else:
                    hl = [3, 3, 4, 5][sf]
                    bits = int.from_bytes(body[:hl], "little") >> 4
                    w = [10, 10, 14, 18][sf]
                    csize = bits >> w & ((1 << w) - 1)
                    seen.add(("streams", 1 if sf == 0 else 4))
                    off = hl + csize
                nseq = body[off]
                if nseq >= 128:
                    off += 2 if nseq < 255 else 3
                else:
                    off += 1
                if nseq:
                    modes = body[off]
                    for k, shift in (("ll", 6), ("of", 4), ("ml", 2)):
                        seen.add(("sequences", modes >> shift & 3))
                pos += size
            else:
                pos += 1 if btype == 1 else size
            if last:
                break
        if fhd & 4:
            pos += 4
    return seen


def check_zstd_coverage():
    import glob
    seen = set()
    for path in glob.glob("zstd_*.zst"):
        if "dictionary" in path:
            continue
        seen |= zstd_modes(open(path, "rb").read())
    wanted = {("block", 0), ("block", 1), ("block", 2), ("literals", 0),
              ("literals", 1), ("literals", 2), ("literals", 3), ("streams", 1),
              ("streams", 4), ("sequences", 0), ("sequences", 1),
              ("sequences", 2), ("sequences", 3)}
    missing = wanted - seen
    if missing:
        sys.exit("fixtures miss Zstandard modes: %s" % sorted(missing))


# Golden outputs of this library's encoders


def check_golden():
    import glob
    inputs = {"text": text(), "hello": b"hello, compress!\n", "empty": b"",
              "columns": columns(), "random": random_bytes()}
    for path in sorted(glob.glob("golden_*")):
        stem, ext = path[len("golden_"):].rsplit(".", 1)
        corpus = inputs[stem.split("_")[0]]
        data = open(path, "rb").read()
        decoded = {"z": zlib.decompress, "gz": gzip.decompress,
                   "deflate": lambda d: zlib.decompress(d, -15)}[ext](data)
        if decoded != corpus:
            sys.exit("%s does not decode to its input" % path)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check_zstd_coverage()
        check_golden()
    else:
        deflate_fixtures()
        snappy_fixtures()
        lz4_fixtures()
        zstd_fixtures()
        check_zstd_coverage()
        check_golden()
