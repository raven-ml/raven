#!/usr/bin/env python3
"""Time the vendors on the files reference.py wrote, for measurements.md.

Unlike reference.py it writes nothing: it reads DATA/<corpus>.<suffix> and
prints, per row, MB/s of decompressed (raw) bytes, MB/s of compressed bytes
and the ratio. Encoders run at the levels the lock thresholds and the
otherlibs/compress comparison need; checksums at 4 KiB, 64 KiB and 8 MiB.
Every number is the fastest of at least 5 calls (3 for an encoder) and as
many as fit a budget, with the load average before and after. Run it under
the timing lock:

    flock -w 1800 ~/wt/TIMING.lock taskset -c 0-5 \\
      uv run --no-project --with deflate --with zstandard --with lz4 \\
        --with python-snappy --with xxhash \\
        python -I vendors.py DATA [decode|encode|checksum] CORPUS...
"""

import os
import platform
import sys
import time
import zlib
from importlib.metadata import version

import deflate
import lz4
import lz4.block
import lz4.frame
import snappy
import xxhash
import zstandard


def best(f, budget, runs):
    times, spent = [], 0.0
    while len(times) < runs or spent < budget:
        t0 = time.perf_counter()
        f()
        dt = time.perf_counter() - t0
        times.append(dt)
        spent += dt
        if len(times) >= 200:
            break
    return min(times), len(times)


def zstd_d(z, n):
    d = zstandard.ZstdDecompressor()
    return lambda: d.decompress(z, max_output_size=n)


# suffix -> (label, decoder maker)
DECODERS = [
    ("zlib6", "zlib", lambda z, n: lambda: zlib.decompress(z)),
    ("zlib6", "libdeflate zlib",
     lambda z, n: lambda: deflate.zlib_decompress(z, n)),
    ("gz9", "zlib gzip", lambda z, n: lambda: zlib.decompress(z, 31)),
    ("gz9", "libdeflate gzip", lambda z, n: lambda: deflate.gzip_decompress(z)),
    ("deflate6", "zlib raw", lambda z, n: lambda: zlib.decompress(z, -15)),
    ("deflate6", "libdeflate raw",
     lambda z, n: lambda: deflate.deflate_decompress(z, n)),
    ("zst1", "zstd", zstd_d),
    ("zst3", "zstd", zstd_d),
    ("zst12", "zstd", zstd_d),
    ("zst19", "zstd", zstd_d),
    ("lz4b", "lz4 block",
     lambda z, n: lambda: lz4.block.decompress(z, uncompressed_size=n)),
    ("lz4f", "lz4 frame", lambda z, n: lambda: lz4.frame.decompress(z)),
    ("snappy", "snappy (cramjam)", lambda z, n: lambda: snappy.uncompress(z)),
]

ENCODERS = [
    ("zlib 1", lambda d: zlib.compress(d, 1)),
    ("zlib 6", lambda d: zlib.compress(d, 6)),
    ("libdeflate 1", lambda d: deflate.deflate_compress(d, 1)),
    ("libdeflate 6", lambda d: deflate.deflate_compress(d, 6)),
    ("zstd 1", zstandard.ZstdCompressor(level=1).compress),
    ("zstd 3", zstandard.ZstdCompressor(level=3).compress),
    ("lz4 block", lambda d: lz4.block.compress(d, store_size=False)),
    ("snappy (cramjam)", snappy.compress),
]

CHECKSUMS = [
    ("crc32 (zlib)", zlib.crc32),
    ("crc32 (libdeflate)", deflate.crc32),
    ("adler32 (zlib)", zlib.adler32),
    ("adler32 (libdeflate)", deflate.adler32),
    ("xxh32", xxhash.xxh32_intdigest),
    ("xxh64", xxhash.xxh64_intdigest),
]


def load():
    return "%.2f %.2f %.2f" % os.getloadavg()


def read(data, name):
    with open(os.path.join(data, name), "rb") as f:
        return f.read()


def decode(data, corpus):
    raw = read(data, corpus + ".raw")
    n = len(raw)
    print("\n## decode %s (%d bytes)\n" % (corpus, n))
    print("| file | decoder | raw MB/s | compressed MB/s | ratio | calls |")
    print("|---|---|---|---|---|---|")
    for suffix, label, mk in DECODERS:
        z = read(data, corpus + "." + suffix)
        f = mk(z, n)
        if f() != raw:
            raise SystemExit("%s.%s: %s decodes wrong bytes"
                             % (corpus, suffix, label))
        t, k = best(f, 1.0, 5)
        print("| %s | %s | %.0f | %.0f | %.3f | %d |"
              % (suffix, label, n / t / 1e6, len(z) / t / 1e6, n / len(z), k))


def encode(data, corpus):
    raw = read(data, corpus + ".raw")
    n = len(raw)
    print("\n## encode %s (%d bytes)\n" % (corpus, n))
    print("| encoder | raw MB/s | ratio | calls |")
    print("|---|---|---|---|")
    for label, enc in ENCODERS:
        z = enc(raw)
        t, k = best(lambda: enc(raw), 1.0, 3)
        print("| %s | %.0f | %.3f | %d |" % (label, n / t / 1e6, n / len(z), k))


def checksum(data, corpus):
    raw = read(data, corpus + ".raw")
    print("\n## checksums on %s\n" % corpus)
    print("| checksum | 4 KiB GB/s | 64 KiB GB/s | 8 MiB GB/s |")
    print("|---|---|---|---|")
    for label, f in CHECKSUMS:
        row = []
        for size in (4 << 10, 64 << 10, 8 << 20):
            b = raw[:size]
            t, _ = best(lambda: f(b), 0.3, 50)
            row.append("%.1f" % (size / t / 1e9))
        print("| %s | %s |" % (label, " | ".join(row)))


def main():
    data, what, corpora = sys.argv[1], sys.argv[2], sys.argv[3:]
    print("# %s · %s · Python %s · zlib %s · zstd %s · lz4 %s · "
          "deflate %s"
          % (platform.machine(), platform.platform(), platform.python_version(),
             zlib.ZLIB_RUNTIME_VERSION, zstandard.ZSTD_VERSION,
             lz4.library_version_string(), version("deflate")))
    print("# load before: " + load())
    run = {"decode": decode, "encode": encode, "checksum": checksum}[what]
    for corpus in corpora:
        run(data, corpus)
    print("\n# load after: " + load())


if __name__ == "__main__":
    main()
