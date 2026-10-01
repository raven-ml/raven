#!/usr/bin/env python3
"""Write the corpora that bench_compress.exe decodes, and time the references.

Three corpora of 8 MiB go to data/: [text], words drawn by a seeded xorshift
from 512 made-up words, which compresses about threefold with no long-range
repeats; [columns], int64 keys with small gaps and float64 prices of two
decimals, as a Parquet page holds them; and [random], incompressible bytes.
Each is also written compressed by every reference encoder, and the script
prints the references' throughput with the machine's load average.

Usage:
    cd otherlibs/compress/bench
    uv run --with python-snappy --with lz4 --with zstandard python reference.py
"""

import gzip
import os
import platform
import struct
import time
import zlib

import lz4
import lz4.block
import lz4.frame
import snappy
import zstandard

SIZE = 8 << 20
MASK = (1 << 64) - 1


def xorshift(seed):
    x = seed
    while True:
        x ^= (x << 13) & MASK
        x ^= x >> 7
        x ^= (x << 17) & MASK
        yield x


def text():
    gen = xorshift(0x51F15EED)
    syllables = ["ka", "lo", "mi", "ne", "su", "ta", "ri", "po", "an", "el",
                 "or", "is", "um", "ve", "da", "go"]
    words = []
    for _ in range(512):
        n = next(gen) % 3 + 1
        words.append("".join(syllables[next(gen) % 16] for _ in range(n)))
    out, size = [], 0
    while size < SIZE:
        w = words[next(gen) % 512] + ("\n" if next(gen) % 12 == 0 else " ")
        out.append(w)
        size += len(w)
    return "".join(out).encode()[:SIZE]


def columns():
    gen = xorshift(0x9E3779B97F4A7C15)
    key, out = 0, bytearray()
    while len(out) < SIZE:
        key += next(gen) % 16 + 1
        out += struct.pack("<qd", key, (next(gen) % 100000) / 100)
    return bytes(out[:SIZE])


def random_bytes():
    gen = xorshift(0x2545F4914F6CDD1D)
    return b"".join(struct.pack("<Q", next(gen)) for _ in range(SIZE // 8))


def raw_deflate(data, level):
    c = zlib.compressobj(level, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush()


ENCODERS = {
    "zlib1": lambda d: zlib.compress(d, 1),
    "zlib6": lambda d: zlib.compress(d, 6),
    "deflate6": lambda d: raw_deflate(d, 6),
    "gz": lambda d: gzip.compress(d, mtime=0),
    "snappy": snappy.compress,
    "lz4b": lambda d: lz4.block.compress(d, store_size=False),
    "lz4f": lz4.frame.compress,
    "zst3": zstandard.ZstdCompressor(level=3).compress,
    "zst19": zstandard.ZstdCompressor(level=19).compress,
}

DECODERS = {
    "zlib1": zlib.decompress,
    "zlib6": zlib.decompress,
    "deflate6": lambda z, n: zlib.decompress(z, -15),
    "gz": gzip.decompress,
    "snappy": snappy.uncompress,
    "lz4b": lambda z, n: lz4.block.decompress(z, uncompressed_size=n),
    "lz4f": lz4.frame.decompress,
    "zst3": zstandard.ZstdDecompressor().decompress,
    "zst19": zstandard.ZstdDecompressor().decompress,
}


def best(f, runs=7):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)


def decode(name, z, n):
    f = DECODERS[name]
    return (lambda: f(z, n)) if name in ("deflate6", "lz4b") else (lambda: f(z))


def main():
    os.makedirs("data", exist_ok=True)
    print("# %s · %s · Python %s · zlib %s · lz4 %s · zstandard %s"
          % (platform.machine(), platform.platform(), platform.python_version(),
             zlib.ZLIB_RUNTIME_VERSION, lz4.library_version_string(),
             zstandard.__version__))
    print("# load average %.2f %.2f %.2f" % os.getloadavg())
    print("%-8s %-9s %7s %11s %11s" % ("corpus", "codec", "ratio", "decode MB/s",
                                       "encode MB/s"))
    for corpus, make in (("text", text), ("columns", columns),
                         ("random", random_bytes)):
        data = make()
        with open("data/%s.raw" % corpus, "wb") as f:
            f.write(data)
        for name, encode in ENCODERS.items():
            z = encode(data)
            with open("data/%s.%s" % (corpus, name), "wb") as f:
                f.write(z)
            decode_s = best(decode(name, z, len(data)))
            encode_s = best(lambda: encode(data), 3) if name in ("zlib6", "snappy") else None
            print("%-8s %-9s %7.2f %11.0f %11s"
                  % (corpus, name, len(data) / len(z), len(data) / decode_s / 1e6,
                     "%.0f" % (len(data) / encode_s / 1e6) if encode_s else "-"))
    print("# load average %.2f %.2f %.2f" % os.getloadavg())


if __name__ == "__main__":
    main()
