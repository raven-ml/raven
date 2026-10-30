#!/usr/bin/env python3
"""Write the Snappy goldens of Google's snappy 1.2 encoder.

python-snappy, through which generate.py writes its .snappy goldens, wraps
cramjam's Rust codec, whose bytes differ from Google's on compressible
inputs of 64 KiB and more. lz's encoder writes Google's bytes, and the test
that states it reads files Google's snappy wrote: built from source, with
snappy_ref.cc beside this file, never linked into lz:

    cmake -S snappy-1.2.2 -B build -DCMAKE_BUILD_TYPE=Release \\
        -DSNAPPY_BUILD_TESTS=OFF -DSNAPPY_BUILD_BENCHMARKS=OFF
    cmake --build build --target snappy
    g++ -O2 -std=c++17 -I snappy-1.2.2 -I build \\
        packages/lz/test/gen/snappy_ref.cc build/libsnappy.a -o snappy_ref

Build it without SSE4.2 (no -msse4.2, -mavx or SNAPPY_REQUIRE_AVX): with
it, snappy hashes with the CRC32C instruction and writes other bytes. Then

    uv run --no-project --with deflate --with zstandard --with lz4 \\
        --with python-snappy python -I packages/lz/test/gen/google_snappy.py \\
        path/to/snappy_ref

writes golden/<kind>_<n>.g12.snappy, checks that python-snappy decodes each
to its input, and replaces the manifest's lines of vendor snappy-1.2.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

import snappy

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "..", "golden")
MANIFEST = os.path.join(GOLDEN, "manifest.txt")
VENDOR = "snappy-1.2"

# The inputs: every size of generate.py for the compressible corpora, where
# the encoders differ, and one random input past a fragment.
SPECS = (["text:%d" % n for n in (0, 1, 100, 65535, 65536, 200000)]
         + ["columns:%d" % n for n in (0, 1, 100, 65535, 65536, 200000)]
         + ["random:65536"])


def generator():
    spec = importlib.util.spec_from_file_location(
        "generate", os.path.join(HERE, "generate.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def compress(ref, data):
    with tempfile.TemporaryDirectory() as d:
        raw, out = os.path.join(d, "raw"), os.path.join(d, "out")
        with open(raw, "wb") as fh:
            fh.write(data)
        subprocess.run([ref, "compress", raw, out], check=True)
        with open(out, "rb") as fh:
            return fh.read()


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: google_snappy.py path/to/snappy_ref")
    ref, gen = sys.argv[1], generator()
    rows = []
    for spec in SPECS:
        data = gen.raw(spec)
        z = compress(ref, data)
        if snappy.uncompress(z) != data:
            sys.exit("%s: python-snappy decodes other bytes" % spec)
        name = spec.replace(":", "_") + ".g12.snappy"
        with open(os.path.join(GOLDEN, name), "wb") as fh:
            fh.write(z)
        rows.append("%s snappy %s 1 %s" % (name, VENDOR, spec))
    with open(MANIFEST) as fh:
        kept = [l for l in fh.read().splitlines()
                if l and l.split(" ")[2] != VENDOR]
    with open(MANIFEST, "w") as fh:
        fh.write("\n".join(kept + rows) + "\n")
    print("wrote %d goldens of %s" % (len(rows), VENDOR))


if __name__ == "__main__":
    main()
