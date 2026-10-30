#!/usr/bin/env python3
"""Write the corpora bench_lz.exe reads, and time the vendors on them.

Three synthetic corpora of 8 MiB go to data/, the generators of
test/gen/generate.py: [text], words of made-up syllables, which compresses
about twofold with no long-range repeats; [columns], int64 keys with small
gaps and float64 prices of two decimals, as a Parquet page holds them; and
[random], incompressible bytes. The Silesia corpus is fetched once and
written as [silesia], its twelve files concatenated in name order. Each
corpus is also written compressed by the vendor encoders the bench decodes.

It then prints each vendor's throughput on each corpus, single-threaded, the
best of several runs, with the load average before and after. Run it under
the timing lock on fudge, as the bench runs:

    cd packages/lz/bench
    flock -w 1800 ~/wt/TIMING.lock taskset -c 0-5 \
      uv run --no-project --with deflate --with zstandard --with lz4 \
        --with python-snappy --with xxhash python -I reference.py [--time]

Without --time it only writes data/.
"""

import gzip
import io
import os
import platform
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

import deflate
import lz4.block
import lz4.frame
import snappy
import xxhash
import zstandard

SIZE = 8 << 20
MASK = (1 << 64) - 1
SILESIA = "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"


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


def silesia():
    path = os.path.join("data", "silesia.zip")
    if not os.path.exists(path):
        print("fetching " + SILESIA, file=sys.stderr)
        with urllib.request.urlopen(SILESIA) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.rename(path + ".part", path)
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if not n.endswith("/"))
        return b"".join(z.read(n) for n in names)


def raw_deflate(level):
    def f(d):
        c = zlib.compressobj(level, zlib.DEFLATED, -15)
        return c.compress(d) + c.flush()
    return f


# The files bench_lz.exe decodes: suffix -> encoder.
ENCODINGS = {
    "zlib6": lambda d: zlib.compress(d, 6),
    "gz9": lambda d: gzip.compress(d, 9, mtime=0),
    "deflate6": lambda d: deflate.deflate_compress(d, 6),
    "zst1": zstandard.ZstdCompressor(level=1, write_checksum=True).compress,
    "zst3": zstandard.ZstdCompressor(level=3, write_checksum=True).compress,
    "zst12": zstandard.ZstdCompressor(level=12, write_checksum=True).compress,
    "zst19": zstandard.ZstdCompressor(level=19, write_checksum=True).compress,
    "lz4b": lambda d: lz4.block.compress(d, store_size=False),
    "lz4f": lz4.frame.compress,
    "snappy": snappy.compress,
}


def write():
    os.makedirs("data", exist_ok=True)
    corpora = {"text": text(SIZE), "columns": columns(SIZE),
               "random": random_bytes(SIZE), "silesia": silesia()}
    for name, data in corpora.items():
        with open(os.path.join("data", name + ".raw"), "wb") as f:
            f.write(data)
        for suffix, enc in ENCODINGS.items():
            with open(os.path.join("data", "%s.%s" % (name, suffix)), "wb") as f:
                f.write(enc(data))
        # The frame of the first 64 KiB that bench_lz.exe decodes into a
        # destination of exactly its size.
        with open(os.path.join("data", name + ".64k.zst3"), "wb") as f:
            f.write(ENCODINGS["zst3"](data[:64 << 10]))
    return corpora


# Timing

def best(f, budget=1.0, runs=5):
    """The fastest of at least [runs] calls, and as many as fit [budget]."""
    times, spent = [], 0.0
    while len(times) < runs or spent < budget:
        t0 = time.perf_counter()
        f()
        dt = time.perf_counter() - t0
        times.append(dt)
        spent += dt
        if len(times) >= 50:
            break
    return min(times)


def mbs(n, t):
    return n / t / 1e6


def zstd_d(z):
    d = zstandard.ZstdDecompressor()
    return lambda: d.decompress(z)


DECODERS = {
    "zlib6": lambda z, n: lambda: zlib.decompress(z),
    "gz9": lambda z, n: lambda: zlib.decompress(z, 31),
    "deflate6": lambda z, n: lambda: deflate.deflate_decompress(z, n),
    "zst1": lambda z, n: zstd_d(z),
    "zst3": lambda z, n: zstd_d(z),
    "zst12": lambda z, n: zstd_d(z),
    "zst19": lambda z, n: zstd_d(z),
    "lz4b": lambda z, n: lambda: lz4.block.decompress(z, uncompressed_size=n),
    "lz4f": lambda z, n: lambda: lz4.frame.decompress(z),
    "snappy": lambda z, n: lambda: snappy.uncompress(z),
}

ENCODERS = {
    "zlib 1": lambda d: zlib.compress(d, 1),
    "zlib 6": lambda d: zlib.compress(d, 6),
    "zlib 9": lambda d: zlib.compress(d, 9),
    "libdeflate 1": lambda d: deflate.deflate_compress(d, 1),
    "libdeflate 6": lambda d: deflate.deflate_compress(d, 6),
    "libdeflate 9": lambda d: deflate.deflate_compress(d, 9),
    "zstd 1": zstandard.ZstdCompressor(level=1).compress,
    "zstd 3": zstandard.ZstdCompressor(level=3).compress,
    "zstd 12": zstandard.ZstdCompressor(level=12).compress,
    "lz4 block": lambda d: lz4.block.compress(d, store_size=False),
    "snappy": snappy.compress,
}

CHECKSUMS = {
    "crc32 (zlib)": zlib.crc32,
    "crc32 (libdeflate)": deflate.crc32,
    "adler32 (zlib)": zlib.adler32,
    "xxh64": xxhash.xxh64_intdigest,
}


def load():
    return "%.2f %.2f %.2f" % os.getloadavg()


def timings(corpora):
    print("# %s · %s · Python %s · zlib %s · zstd %s · lz4 %s"
          % (platform.machine(), platform.platform(), platform.python_version(),
             zlib.ZLIB_RUNTIME_VERSION, zstandard.ZSTD_VERSION,
             lz4.library_version_string()))
    print("# load before: " + load())
    for name, data in corpora.items():
        n = len(data)
        print("\n## %s (%d bytes)\n" % (name, n))
        print("| row | MB/s | ratio |\n|---|---|---|")
        for suffix, mk in DECODERS.items():
            z = ENCODINGS[suffix](data)
            t = best(mk(z, n))
            print("| decode %s | %.0f | %.3f |" % (suffix, mbs(n, t), n / len(z)))
        for label, enc in ENCODERS.items():
            z = enc(data)
            t = best(lambda: enc(data), budget=0.5, runs=3)
            print("| encode %s | %.0f | %.3f |" % (label, mbs(n, t), n / len(z)))
        for label, f in CHECKSUMS.items():
            t = best(lambda: f(data))
            print("| %s | %.0f | |" % (label, mbs(n, t)))
    print("\n# load after: " + load())


if __name__ == "__main__":
    corpora = write()
    if "--time" in sys.argv[1:]:
        timings(corpora)
