"""Mutated goldens through lz's deflate decoder and Python's zlib.

Each case flips or replaces 1 to 3 bytes of a deflate, zlib or gzip golden
of at most 4 KiB (and cuts one in ten), then expects both decoders to
agree: the same output, both truncated, or the same zlib message. Build
decode.c first (see its header), then from the tree's root:

    uv run --no-project python -I packages/lz/bench/measure/deflate/zlib_fuzz.py \
        ./decode COUNT SEED

Three outcomes differ by design. A zlib header with FDICT cut before its
DICTID is Unsupported for lz, which declines the flag itself, and
truncated for zlib. A gzip header cut after a byte that breaks ID1, ID2
or CM is an error for lz, which checks each of those bytes as it
arrives, and truncated for zlib, which reads them in pairs. And a stream
that breaks two rules gets the message of the rule lz proves first in
the stream, while zlib, which checks code sets only after reading every
code length, may name the other, or call the stream truncated when a cut
comes first: the rules and the messages are zlib's, the choice between
two faults is not a promise. The script lists every difference for a
reader to classify.
"""

import os
import random
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "..", "..", "..", "test", "golden")
WBITS = {"deflate": -15, "zlib": 15, "gzip": 31}
WRAP = {"deflate": 0, "zlib": 1, "gzip": 2}
CAP = 300000


def vendor(codec, data):
    out, rest = b"", data
    while True:
        d = zlib.decompressobj(WBITS[codec])
        try:
            out += d.decompress(rest, CAP - len(out) + 1)
        except zlib.error as e:
            return ("error", str(e).split(": ", 1)[-1])
        if len(out) > CAP:
            return ("too long",)
        if not d.eof:
            return ("truncated",)
        # Gzip members follow each other to the input's end, and a byte
        # that begins none is an error, as lz reads them.
        rest = d.unused_data
        if codec != "gzip" or not rest:
            return ("ok", out)


def ours(decode, codec, data, path):
    with open(path, "wb") as f:
        f.write(data)
    p = subprocess.run([decode, str(WRAP[codec]), path, str(CAP)],
                       capture_output=True)
    err = p.stderr.decode()
    if p.returncode == 0:
        return ("ok", p.stdout)
    kind = err.split()[1]
    if kind == "0":
        return ("truncated",)
    if kind == "2":
        return ("too long",)
    if kind == "3":
        return ("unsupported",)
    return ("error", err.split(": ", 1)[1].strip())


def main():
    decode, count, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(seed)
    files = []
    for line in open(os.path.join(GOLDEN, "manifest.txt")):
        name, codec = line.split()[:2]
        data = open(os.path.join(GOLDEN, name), "rb").read()
        if codec in WBITS and len(data) <= 4096:
            files.append((name, codec, data))
    same = differ = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "case")
        for _ in range(count):
            name, codec, data = rng.choice(files)
            b = bytearray(data)
            for _ in range(rng.choice([1, 1, 2, 3])):
                if b:
                    i = rng.randrange(len(b))
                    if rng.random() < 0.5:
                        b[i] ^= 1 << rng.randrange(8)
                    else:
                        b[i] = rng.randrange(256)
            if b and rng.random() < 0.1:
                b = b[:rng.randrange(len(b))]
            v, o = vendor(codec, bytes(b)), ours(decode, codec, bytes(b), path)
            if v == o:
                same += 1
            else:
                differ += 1
                print("differ", name, "zlib", v[:1] + v[1:2] * (v[0] != "ok"),
                      "lz", o[:1] + o[1:2] * (o[0] != "ok"))
    print("%d agree, %d differ" % (same, differ))


if __name__ == "__main__":
    main()
