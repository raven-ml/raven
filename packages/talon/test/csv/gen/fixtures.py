# /// script
# requires-python = "==3.13.*"
# dependencies = ["zstandard==0.25.0"]
# ///
"""Writes the compressed files of the CSV suite.

Run from this directory with `uv run fixtures.py [--check]`. The fixtures are
committed, so the suite itself needs neither Python nor the network. With
--check nothing is written, and the run fails if a fixture would change.

Each ../support/ file but plain.csv is gzip or Zstandard bytes that decode to
the text it names: plain.csv, or an empty text for the empty_ files.
"""

import argparse
import gzip
import struct
import sys
import zlib
from pathlib import Path

import zstandard

HERE = Path(__file__).parent
OUT = HERE.parent / "support"

ROWS = 200


def plain():
    lines = ["id,city,score,ok,day"]
    for i in range(ROWS):
        city = '"Ban""gor, ME"' if i % 7 == 0 else f"city{i % 13}"
        score = "" if i % 11 == 0 else f"{i * 37 % 1000}.{i % 10}"
        ok = "true" if i % 3 == 0 else "false"
        lines.append(f"{i},{city},{score},{ok},2024-{i % 12 + 1:02}-{i % 28 + 1:02}")
    return ("\n".join(lines) + "\n").encode()


TEXT = plain()
CUT = len(TEXT) // 3 + 5  # Inside a record.
TINY = b"a,b\n1,x\n2,y\n"


def member(data, **kw):
    return gzip.compress(data, mtime=0, **kw)


def flagged(data):
    """A member with every optional header field (FEXTRA, FNAME, FCOMMENT,
    FHCRC) and FTEXT set, by hand: gzip.compress writes none of them."""
    flags = 0x01 | 0x02 | 0x04 | 0x08 | 0x10
    header = b"\x1f\x8b\x08" + bytes([flags]) + b"\0\0\0\0" + b"\0\xff"
    header += struct.pack("<H", 4) + b"talo"  # FEXTRA: its length, then bytes.
    header += b"plain.csv\0" + b"a comment\0"
    header += struct.pack("<H", zlib.crc32(header) & 0xFFFF)
    deflate = zlib.compressobj(wbits=-15)
    body = deflate.compress(data) + deflate.flush()
    return header + body + struct.pack("<II", zlib.crc32(data), len(data))


def frame(data, **kw):
    return zstandard.ZstdCompressor(write_checksum=True, **kw).compress(data)


def skippable(payload):
    return struct.pack("<II", 0x184D2A50, len(payload)) + payload


FILES = {
    "plain.csv": TEXT,
    "member.gz": member(TEXT),
    "members.gz": member(TEXT[:CUT]) + member(b"") + member(TEXT[CUT:]),
    "flagged.gz": flagged(TEXT),
    "frame.zst": frame(TEXT),
    "frames.zst": frame(TEXT[:CUT]) + frame(b"") + frame(TEXT[CUT:], level=19),
    "skippable.zst": skippable(b"abcd") + frame(TEXT) + skippable(b""),
    "unsized.zst": frame(TEXT, write_content_size=False),
    "empty_member.gz": member(b""),
    "empty_frame.zst": frame(b""),
    # What the name says decides nothing.
    "gzip_named.csv": member(TEXT),
    "zstd_named.csv": frame(TEXT),
    "plain_named.gz": TEXT,
    "plain_named.zst": TEXT,
    "tiny.gz": member(TINY),
    "tiny.zst": frame(TINY),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = [
        name for name, data in FILES.items()
        if not (OUT / name).exists() or (OUT / name).read_bytes() != data
    ]
    if args.check:
        if changed:
            sys.exit(f"would change: {', '.join(changed)}")
        return
    OUT.mkdir(exist_ok=True)
    for name in changed:
        (OUT / name).write_bytes(FILES[name])


if __name__ == "__main__":
    main()
