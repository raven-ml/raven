# /// script
# requires-python = ">=3.10"
# ///
"""Writes VBIOS images for the FWSEC walk, laid out as NVIDIA's RM reads
them (open-gpu-kernel-modules 570.144: pci_exp_table.h,
kernel_gsp_vbios_tu102.c, kernel_gsp_fwsec.c, kernel_gsp_frts_tu102.c).
gen.sh runs it in this directory.

vbios.rom is three PCI expansion ROM images: the base image (2 blocks), an
EFI image (1 block, its length from NVIDIA's PCI data extension), and the
extension image (code type 0xe0), last. The base image holds the BIT table,
whose falcon data token points into the extension image: pointers there are
relative to the extension image's offset less the base image's size, 512.
The ucode table lists a debug FWSEC, then the production one, with a version
3 descriptor, two RSA-3K signatures and its image: 512 bytes of code, then
1024 of data holding the application interface, whose DMEM mapper takes the
FRTS command.

The variants each lack one thing: vbios_no_bit.rom its BIT table's checksum,
vbios_debug.rom the production FWSEC, vbios_no_mapper.rom the DMEM mapper.

--check writes nothing: it fails if a committed file differs from what this
script makes, or if the C compiler ($CC, or cc), reading rig's excerpts of
pci_exp_table.h and kernel_gsp_frts_tu102.c, places the fields of the PCI data
structures and of the FWSEC's application interface otherwise than this
script does. The BIT table's structures are left to the RM's format strings,
which this script's formats follow: their C members are of types the excerpts
do not define, and the C structures are the RM's unpacked copies.
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
EXCERPTS = HERE.parents[3] / "lib/nv/gen/headers/570"

BLOCK = 512
BASE, EFI = 2 * BLOCK, BLOCK
EXT_AT = BASE + EFI
EXT_LEN = 8 * BLOCK

BIT_AT = 0x1B0
FALCON_DATA_AT = 0x200
TABLE = BASE + 0x100  # pointers, the extension image's offset less BASE
DESC = BASE + 0x200
DESC_SIZE = 44 + 2 * 384
IMEM, DMEM = 0x200, 0x400
INTERFACE = 0x10  # in the data
MAPPER = 0x80
CMD_IN = 0x100
PKC = 0x200

# The struct.pack formats of the structures the script writes, and where it
# writes their fields.
PCIR_PTR = 0x18  # the PCI data structure's offset in the ROM header
# The PCI data structure: signature, vendor, device, 0, length, revision,
# class code, image length, 0, code type, last image.
PCIR = "<IHHHHBBBBHHBB"
NPDE = "<IHHHB"  # signature, revision, length, sub-image length, last
APP_HEADER = "<BBBB"  # version, header bytes, entry bytes, entries
APP_ENTRY = "<II"  # id, DMEM offset
MAPPER_CMD_IN = 8  # the DMEM mapper's cmd_in_buffer_offset


def image(n, code_type, last, ext_len=None):
    """An image of [n] blocks: the ROM signature, the PCI data structure at
    0x40, and NVIDIA's extension at 0x60 when [ext_len] says its length."""
    b = bytearray(n * BLOCK)
    struct.pack_into("<HH", b, 0, 0xAA55, 0)
    struct.pack_into("<H", b, PCIR_PTR, 0x40)
    pcir = struct.pack(PCIR, 0x52494350, 0x10DE, 0, 0, 0x18, 0, 0, 0, 3,
                       n if ext_len is None else n + 7, 0, code_type, 0x80 if last else 0)
    b[0x40:0x40 + len(pcir)] = pcir
    if ext_len is not None:
        struct.pack_into(NPDE, b, 0x60, 0x4544504E, 0x100, 12, ext_len, 0x80 if last else 0)
    return b


def rom(*, bit_ok=True, prod=True, mapper=True):
    base = image(2, 0x00, False)
    header = bytearray(struct.pack("<HIHBBBB", 0xB8FF, 0x00544942, 0x0100, 12, 6, 1, 0))
    if bit_ok:
        header[11] = (-sum(header)) & 0xFF
    base[BIT_AT:BIT_AT + 12] = header
    base[BIT_AT + 12:BIT_AT + 18] = struct.pack("<BBHH", 0x70, 2, 4, FALCON_DATA_AT)
    struct.pack_into("<I", base, FALCON_DATA_AT, TABLE)
    efi = image(4, 0x03, False, ext_len=1)
    ext = image(EXT_LEN // BLOCK, 0xE0, True)
    t = TABLE - BASE
    struct.pack_into("<6B", ext, t, 1, 6, 6, 2, 3, 44)
    struct.pack_into("<BBI", ext, t + 6, 0x45, 0, DESC)
    struct.pack_into("<BBI", ext, t + 12, 0x85 if prod else 0x45, 0, DESC)
    d = DESC - BASE
    stored = IMEM + DMEM
    struct.pack_into("<I", ext, d, 1 | (3 << 8) | (DESC_SIZE << 16))
    struct.pack_into("<8IHBBH", ext, d + 4, stored, PKC, INTERFACE, 0x1000, IMEM, 0x2000, 0x3000, DMEM,
                     0x400, 3, 2, 0x3)
    ext[d + 44:d + 44 + 384] = bytes([0x11]) * 384
    ext[d + 44 + 384:d + DESC_SIZE] = bytes([0x22]) * 384
    img = d + DESC_SIZE
    for i in range(stored):
        ext[img + i] = i & 0xFF
    a = img + IMEM + INTERFACE
    header, entry = struct.calcsize(APP_HEADER), struct.calcsize(APP_ENTRY)
    struct.pack_into(APP_HEADER, ext, a, 1, header, entry, 2)
    struct.pack_into(APP_ENTRY, ext, a + header, 1, 0x40)
    struct.pack_into(APP_ENTRY, ext, a + header + entry, 4 if mapper else 5, MAPPER)
    struct.pack_into("<I", ext, img + IMEM + MAPPER + MAPPER_CMD_IN, CMD_IN)
    return bytes(base + efi[:EFI] + ext)


# Checking the layouts with a C compiler

# The excerpts carry no nvtypes.h: its words, as it defines them for 64-bit
# Linux.
C_PRELUDE = """\
#include <stddef.h>
#include <stdio.h>
typedef unsigned char NvU8;
typedef unsigned short NvU16;
typedef unsigned int NvU32;
typedef unsigned long long NvU64;
"""


def offsets(fmt):
    """The byte offset of each item of the struct.pack format [fmt]."""
    return [struct.calcsize(fmt[:i]) for i in range(1, len(fmt))]


def layout():
    """(C expression, value) of every place this script writes a field the
    excerpts lay out at."""
    pcir, npde = offsets(PCIR), offsets(NPDE)
    header, entry = "FALCON_APPLICATION_INTERFACE_HEADER_V1", "FALCON_APPLICATION_INTERFACE_ENTRY_V1"
    return [
        ("OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR", PCIR_PTR),
        ("OFFSETOF_PCI_DATA_STRUCT_LEN", pcir[4]),
        ("OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN", pcir[9]),
        ("OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE", pcir[11]),
        ("OFFSETOF_PCI_DATA_STRUCT_LAST_IMAGE", pcir[12]),
        ("OFFSETOF_PCI_DATA_EXT_STRUCT_SIG", npde[0]),
        ("OFFSETOF_PCI_DATA_EXT_STRUCT_REV", npde[1]),
        ("OFFSETOF_PCI_DATA_EXT_STRUCT_LEN", npde[2]),
        ("OFFSETOF_PCI_DATA_EXT_STRUCT_SUBIMAGE_LEN", npde[3]),
        ("OFFSETOF_PCI_DATA_EXT_STRUCT_LAST_IMAGE", npde[4]),
        *((f"offsetof({header}, {m})", o)
          for m, o in zip(["version", "headerSize", "entrySize", "entryCount"], offsets(APP_HEADER))),
        (f"sizeof({header})", struct.calcsize(APP_HEADER)),
        *((f"offsetof({entry}, {m})", o) for m, o in zip(["id", "dmemOffset"], offsets(APP_ENTRY))),
        (f"sizeof({entry})", struct.calcsize(APP_ENTRY)),
        ("offsetof(FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3, cmd_in_buffer_offset)", MAPPER_CMD_IN),
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
    """Fails unless the C compiler places the excerpts' fields where this
    script writes them."""
    text = "".join(f'#line 1 "{n}"\n' + (EXCERPTS / n).read_text(encoding="latin-1")
                   for n in ["pci_exp_table.h", "kernel_gsp_frts_tu102.c"])
    checks = layout()
    got = c_values(C_PRELUDE + text, [e for e, _ in checks])
    wrong = [f"  {e}: vbios.py {v}, cc {g}" for (e, v), g in zip(checks, got) if v != g]
    if wrong:
        sys.exit("the C compiler lays out the excerpts otherwise:\n" + "\n".join(wrong))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    a = p.parse_args()
    files = {"vbios.rom": rom(), "vbios_no_bit.rom": rom(bit_ok=False), "vbios_debug.rom": rom(prod=False),
             "vbios_no_mapper.rom": rom(mapper=False)}
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
