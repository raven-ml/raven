# /// script
# requires-python = ">=3.10"
# ///
"""Writes discovery tables, as an AMD GPU's firmware leaves them in its memory,
from a listing of the GPU's blocks, for the suites of rig_amd_abi and
rig_amd_pci. gen.sh runs it in this directory.

r9700.txt is nonnormal's Radeon AI PRO R9700 (gfx1201) as the amdgpu driver
lists it, one block instance a line: hardware ID, instance, version, harvest
bits, register segment bases. It was made with

  ssh nonnormal 'cd /sys/bus/pci/devices/0000:05:00.0/ip_discovery/die/0 &&
    for id in [0-9]*; do for i in $id/[0-9]*; do echo "$(cat $i/hw_id)
    $(cat $i/num_instance) $(cat $i/major).$(cat $i/minor).$(cat $i/revision)
    $(cat $i/harvest) $(cat $i/base_addr | tr "\\n" " ")"; done; done'

and the GC table holds what amdkfd's topology reports for that GPU
(simd_arrays_per_engine 2, array_count 8, cu_per_simd_array 8,
max_waves_per_simd 16, max_slots_scratch_cu 32, lds_size_in_kb 64).

The layouts are those of the Linux kernel's include/discovery.h (under
#pragma pack(1)); the checksums are amdgpu_discovery.c's, the 16-bit sum of
a table's bytes. Each file is 10 KiB, the bytes the driver reads:

- r9700.bin: IP table version 3, 32-bit bases, no harvest table.
- r9700_wide.bin: IP table version 4 with 64-bit bases, whose bits above the
  30 of a word address are set.
- r9700_fused.bin: as r9700.bin, with a harvest table that fuses off UMC
  instance 7 (hardware ID 150).

--check writes nothing: it fails if a committed file differs from what this
script makes, or if the C compiler ($CC, or cc) lays out discovery.h's
structures, as rig's excerpt of it holds them, otherwise than this script
packs them.
"""

import argparse
import os
import pathlib
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
EXCERPT = HERE.parents[2] / "lib/amd/gen/headers/kernel/discovery.h"
TABLE_BYTES = 10 << 10

BINARY_SIGNATURE = 0x28211407
DISCOVERY_TABLE_SIGNATURE = 0x53445049
GC_TABLE_ID = 0x4347
HARVEST_TABLE_SIGNATURE = 0x56524148
BINARY_HEADER = 60  # sizeof(binary_header): 12 bytes, then table_info[6] of 8
IP_HEADER = 80  # sizeof(ip_discovery_header)
GC_INFO_V1_0 = 88
HARVEST_TABLE = 136
# High bits a 64-bit base may carry above its word address.
HIGH = 0x1_c000_0000

# The struct.pack formats of discovery.h's structures, each up to the member
# its comment ends at.
BINARY_HEAD = "<IHHH"  # binary_header to binary_checksum
BINARY_SIZE = "<H"  # binary_size, the first bytes the checksum sums
TABLE_INFO = "<HHHH"
IP_HEAD = "<IHHIH"  # ip_discovery_header to num_dies; then die_info[16]
DIE_INFO = "<HH"
DIES = 16
IP_FLAGS = "<BB"  # the byte of base_addr_64_bit, then reserved2
DIE_HEADER = "<HH"
IP_V4 = "<HBBBBBB"  # ip_v4 to the byte of sub_revision; then the bases
GPU_INFO_HEADER = "<IHHI"
GC_WORDS = 19  # gc_info_v1_0's words, from gc_num_se
HARVEST_HEADER = "<II"
HARVEST_INFO = "<HBB"


def blocks():
    out = []
    for line in (HERE / "r9700.txt").read_text().splitlines():
        hw, inst, ver, harvest, *bases = line.split()
        major, minor, rev = map(int, ver.split("."))
        out.append((int(hw), int(inst), (major, minor, rev), int(harvest, 16), [int(b, 16) for b in bases]))
    return out


def checksum(b):
    return sum(b) & 0xffff


def ip_table(version, wide, at):
    """The IP table at byte [at] of the binary: die offsets are the binary's."""
    ips = b""
    bs = blocks()
    for hw, inst, (major, minor, rev), harvest, bases in bs:
        ips += struct.pack(IP_V4, hw, inst, len(bases), major, minor, rev, harvest & 0xf)
        for b in bases:
            ips += struct.pack("<Q", b | HIGH) if wide else struct.pack("<I", b)
    die = struct.pack(DIE_HEADER, 0, len(bs)) + ips
    size = IP_HEADER + len(die)
    dies = struct.pack(DIE_INFO, 0, at + IP_HEADER) + bytes(struct.calcsize(DIE_INFO) * (DIES - 1))
    flags = struct.pack(IP_FLAGS, 1 if wide else 0, 0)
    header = struct.pack(IP_HEAD, DISCOVERY_TABLE_SIGNATURE, version, size, 0, 1) + dies + flags
    assert len(header) == IP_HEADER
    return header + die


def gc_table():
    # gc_info_v1_0: header, then 19 words from gc_num_se.
    words = [4, 2, 2, 4, 16, 1536, 0, 0, 0, 0, 0, 32, 16, 32, 64, 1, 2, 0, 0]
    body = struct.pack(GPU_INFO_HEADER, GC_TABLE_ID, 1, 0, GC_INFO_V1_0) + struct.pack(f"<{GC_WORDS}I", *words)
    assert len(body) == GC_INFO_V1_0
    return body


def harvest_table(fused):
    entries = b"".join(struct.pack(HARVEST_INFO, hw, inst, 0) for hw, inst in fused)
    body = struct.pack(HARVEST_HEADER, HARVEST_TABLE_SIGNATURE, 0) + entries
    return body + bytes(HARVEST_TABLE - len(body))


def table(version, wide, fused):
    at = 0x40
    tables = [ip_table(version, wide, at), gc_table()] + ([harvest_table(fused)] if fused else [])
    infos, body = [], b""
    for t in tables:
        infos.append(struct.pack(TABLE_INFO, at, checksum(t), len(t), 0))
        body += bytes(at - BINARY_HEADER - len(body)) + t
        at = BINARY_HEADER + len(body)
        at = (at + 0xf) & ~0xf
    infos += [bytes(8)] * (6 - len(infos))
    size = BINARY_HEADER + len(body)
    after = struct.pack(BINARY_SIZE, size) + b"".join(infos) + body
    header = struct.pack(BINARY_HEAD, BINARY_SIGNATURE, 2, 0, checksum(after))
    t = header + after
    assert len(t) == size and size <= TABLE_BYTES
    return t + bytes(TABLE_BYTES - size)


# Checking the layouts with a C compiler

# What the excerpt takes from Linux's headers it does not copy: the fixed-width
# words, the byte order the tables are in, and Linux's flexible array in a
# union and packing.
C_PRELUDE = """\
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define LITTLEENDIAN_CPU
#define __packed __attribute__((__packed__))
#define DECLARE_FLEX_ARRAY(TYPE, NAME) struct { struct { } __empty_##NAME; TYPE NAME[]; }

/* The lowest set bit of the [n] bytes at [p], little-endian, or with [count]
   the number of set bits: a bit field's place once all its bits are set. */
static long long set_bits(const unsigned char *p, size_t n, int count) {
  long long lo = -1, k = 0;
  for (size_t i = 0; i < 8 * n; i++)
    if (p[i / 8] >> (i % 8) & 1) {
      if (lo < 0) lo = (long long)i;
      k++;
    }
  return count ? k : lo;
}
#define BITS(T, m, count) \\
  ({ T s_; memset(&s_, 0, sizeof s_); s_.m = -1; set_bits((unsigned char *)&s_, sizeof s_, count); })
"""


def offsets(fmt):
    """The byte offset of each item of the struct.pack format [fmt]."""
    return [struct.calcsize(fmt[:i]) for i in range(1, len(fmt))]


def packed(t, fmt, members, at=0):
    """(C expression, value) of [members] of [t] at the offsets [fmt] packs
    them at, from byte [at]."""
    return [(f"offsetof({t}, {m})", at + o) for m, o in zip(members, offsets(fmt))]


def layout():
    """(C expression, value) of every place and size this script packs a
    structure of discovery.h at."""
    die, info = struct.calcsize(DIE_INFO), struct.calcsize(TABLE_INFO)
    binary = BINARY_HEAD + BINARY_SIZE[1:]
    flags = struct.calcsize(IP_HEAD) + DIES * die
    harvest = struct.calcsize(HARVEST_HEADER)
    return [
        *packed("table_info", TABLE_INFO, ["offset", "checksum", "size", "padding"]),
        ("sizeof(table_info)", info),
        *packed("binary_header", binary,
                ["binary_signature", "version_major", "version_minor", "binary_checksum", "binary_size"]),
        ("offsetof(binary_header, table_list)", struct.calcsize(binary)),
        ("sizeof(((binary_header *)0)->table_list)", (BINARY_HEADER - struct.calcsize(binary))),
        ("sizeof(binary_header)", BINARY_HEADER),
        *packed("ip_discovery_header", IP_HEAD, ["signature", "version", "size", "id", "num_dies"]),
        ("offsetof(ip_discovery_header, die_info)", struct.calcsize(IP_HEAD)),
        ("sizeof(((ip_discovery_header *)0)->die_info)", DIES * die),
        ("BITS(ip_discovery_header, base_addr_64_bit, 0)", 8 * flags),
        ("offsetof(ip_discovery_header, reserved2)", flags + offsets(IP_FLAGS)[1]),
        ("sizeof(ip_discovery_header)", IP_HEADER),
        *packed("die_info", DIE_INFO, ["die_id", "die_offset"]),
        ("sizeof(die_info)", die),
        *packed("die_header", DIE_HEADER, ["die_id", "num_ips"]),
        ("sizeof(die_header)", struct.calcsize(DIE_HEADER)),
        *packed("ip_v4", IP_V4, ["hw_id", "instance_number", "num_base_address", "major", "minor", "revision"]),
        # The 4 bits the script writes the block's harvest bits to.
        ("BITS(ip_v4, sub_revision, 0)", 8 * offsets(IP_V4)[6]),
        ("BITS(ip_v4, sub_revision, 1)", 4),
        ("offsetof(ip_v4, base_address)", struct.calcsize(IP_V4)),
        ("offsetof(ip_v4, base_address_64)", struct.calcsize(IP_V4)),
        ("sizeof(ip_v4)", struct.calcsize(IP_V4)),
        *packed("struct gpu_info_header", GPU_INFO_HEADER, ["table_id", "version_major", "version_minor", "size"]),
        ("sizeof(struct gpu_info_header)", struct.calcsize(GPU_INFO_HEADER)),
        ("offsetof(struct gc_info_v1_0, gc_num_se)", struct.calcsize(GPU_INFO_HEADER)),
        ("sizeof(struct gc_info_v1_0)", GC_INFO_V1_0),
        ("sizeof(struct gc_info_v1_0)", struct.calcsize(GPU_INFO_HEADER) + 4 * GC_WORDS),
        *packed("harvest_info_header", HARVEST_HEADER, ["signature", "version"]),
        ("offsetof(harvest_table, list)", harvest),
        *packed("harvest_info", HARVEST_INFO, ["hw_id", "number_instance", "reserved"]),
        ("sizeof(harvest_info)", struct.calcsize(HARVEST_INFO)),
        ("sizeof(harvest_table)", HARVEST_TABLE),
        ("sizeof(((harvest_table *)0)->list)", HARVEST_TABLE - harvest),
    ]


def c_values(source, exprs):
    """The values of the C expressions [exprs] in the program [source], as the
    C compiler ($CC, or cc) lays it out for this machine."""
    cc = shlex.split(os.environ.get("CC") or "cc")
    if not cc or not shutil.which(cc[0]):
        sys.exit("no C compiler to check the layouts with: set CC or install cc")
    prints = "".join(f'  printf("%lld\\n", (long long)({e}));\n' for e in exprs)
    with tempfile.TemporaryDirectory() as d:
        c, exe = pathlib.Path(d) / "layouts.c", pathlib.Path(d) / "layouts"
        c.write_text(f"{source}\nint main(void) {{\n{prints}  return 0;\n}}\n", encoding="latin-1")
        r = subprocess.run([*cc, "-o", str(exe), str(c)], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{cc[0]} cannot compile the layouts:\n{r.stderr}")
        out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    return [int(v) for v in out.split()]


def check_with_cc():
    """Fails unless the C compiler lays out the excerpt's structures as this
    script packs them."""
    text = EXCERPT.read_text(encoding="latin-1")
    checks = layout()
    got = c_values(C_PRELUDE + f'#line 1 "{EXCERPT.name}"\n' + text, [e for e, _ in checks])
    wrong = [f"  {e}: discovery.py {v}, cc {g}" for (e, v), g in zip(checks, got) if v != g]
    if wrong:
        sys.exit("the C compiler lays out discovery.h otherwise:\n" + "\n".join(wrong))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    a = p.parse_args()
    files = {"r9700.bin": table(3, False, []), "r9700_wide.bin": table(4, True, []),
             "r9700_fused.bin": table(3, False, [(150, 7)])}
    if not a.check:
        for name, data in files.items():
            (HERE / name).write_bytes(data)
        return
    check_with_cc()
    stale = [n for n, data in files.items() if not (HERE / n).exists() or (HERE / n).read_bytes() != data]
    if stale:
        sys.exit("stale: " + ", ".join(stale))
    print("up to date")


if __name__ == "__main__":
    main()
