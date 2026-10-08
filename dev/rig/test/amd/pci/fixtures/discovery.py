# /// script
# requires-python = ">=3.10"
# ///
"""Writes discovery tables, as an AMD GPU's firmware leaves them in its memory,
from a listing of the GPU's blocks, for the suite of rig_amd_pci.

  uv run dev/rig/test/amd/pci/fixtures/discovery.py

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
"""

import pathlib
import struct

HERE = pathlib.Path(__file__).resolve().parent
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
        ips += struct.pack("<HBBBBBB", hw, inst, len(bases), major, minor, rev, harvest & 0xf)
        for b in bases:
            ips += struct.pack("<Q", b | HIGH) if wide else struct.pack("<I", b)
    die = struct.pack("<HH", 0, len(bs)) + ips
    size = IP_HEADER + len(die)
    dies = struct.pack("<HH", 0, at + IP_HEADER) + bytes(4 * 15)
    flags = struct.pack("<BB", 1 if wide else 0, 0)
    header = struct.pack("<IHHIH", DISCOVERY_TABLE_SIGNATURE, version, size, 0, 1) + dies + flags
    assert len(header) == IP_HEADER
    return header + die


def gc_table():
    # gc_info_v1_0: header, then 19 words from gc_num_se.
    words = [4, 2, 2, 4, 16, 1536, 0, 0, 0, 0, 0, 32, 16, 32, 64, 1, 2, 0, 0]
    body = struct.pack("<IHHI", GC_TABLE_ID, 1, 0, GC_INFO_V1_0) + struct.pack("<19I", *words)
    assert len(body) == GC_INFO_V1_0
    return body


def harvest_table(fused):
    entries = b"".join(struct.pack("<HBB", hw, inst, 0) for hw, inst in fused)
    body = struct.pack("<II", HARVEST_TABLE_SIGNATURE, 0) + entries
    return body + bytes(HARVEST_TABLE - len(body))


def table(version, wide, fused):
    at = 0x40
    tables = [ip_table(version, wide, at), gc_table()] + ([harvest_table(fused)] if fused else [])
    infos, body = [], b""
    for t in tables:
        infos.append(struct.pack("<HHHH", at, checksum(t), len(t), 0))
        body += bytes(at - BINARY_HEADER - len(body)) + t
        at = BINARY_HEADER + len(body)
        at = (at + 0xf) & ~0xf
    infos += [bytes(8)] * (6 - len(infos))
    size = BINARY_HEADER + len(body)
    after = struct.pack("<H", size) + b"".join(infos) + body
    header = struct.pack("<IHHH", BINARY_SIGNATURE, 2, 0, checksum(after))
    t = header + after
    assert len(t) == size and size <= TABLE_BYTES
    return t + bytes(TABLE_BYTES - size)


def main():
    (HERE / "r9700.bin").write_bytes(table(3, False, []))
    (HERE / "r9700_wide.bin").write_bytes(table(4, True, []))
    (HERE / "r9700_fused.bin").write_bytes(table(3, False, [(150, 7)]))


if __name__ == "__main__":
    main()
