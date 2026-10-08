# /// script
# requires-python = ">=3.10"
# ///
"""Writes sections.elf: a 64-bit little-endian ELF object with no program,
whose sections are those the GSP's firmware holds its image and signatures
in, each holding its own name as bytes. Run from this directory:

  uv run sections.py
"""

import struct

SECTIONS = [".fwimage", ".fwsignature_ga10x", ".fwsignature_ad10x", ".fwsignature_gb20x"]


def main():
    names = b"\0" + b"".join(n.encode() + b"\0" for n in SECTIONS + [".shstrtab"])
    data = [n.encode() for n in SECTIONS]
    body, offsets, at = b"", [], 64
    for d in data:
        offsets.append(at + len(body))
        body += d
    shstr_at = 64 + len(body)
    body += names
    shoff = 64 + len(body)
    count = len(SECTIONS) + 2
    header = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    header += struct.pack("<HHIQQQIHHHHHH", 1, 0, 1, 0, 0, shoff, 0, 64, 0, 0, 64, count, count - 1)
    shdrs = bytes(64)
    name_at = 1
    for n, off, d in zip(SECTIONS, offsets, data):
        shdrs += struct.pack("<IIQQQQIIQQ", name_at, 1, 0, 0, off, len(d), 0, 0, 1, 0)
        name_at += len(n) + 1
    shdrs += struct.pack("<IIQQQQIIQQ", name_at, 3, 0, 0, shstr_at, len(names), 0, 0, 1, 0)
    with open("sections.elf", "wb") as f:
        f.write(header + body + shdrs)


if __name__ == "__main__":
    main()
