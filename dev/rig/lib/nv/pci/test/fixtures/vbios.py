# /// script
# requires-python = ">=3.10"
# ///
"""Writes VBIOS images for the FWSEC walk, laid out as NVIDIA's RM reads
them (open-gpu-kernel-modules 570.144: pci_exp_table.h,
kernel_gsp_vbios_tu102.c, kernel_gsp_fwsec.c, kernel_gsp_frts_tu102.c).
Run from this directory:

  uv run vbios.py

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
"""

import struct

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


def image(n, code_type, last, ext_len=None):
    """An image of [n] blocks: the ROM signature, the PCI data structure at
    0x40, and NVIDIA's extension at 0x60 when [ext_len] says its length."""
    b = bytearray(n * BLOCK)
    struct.pack_into("<HH", b, 0, 0xAA55, 0)
    struct.pack_into("<H", b, 0x18, 0x40)
    pcir = struct.pack("<IHHHHBBBBHHBB", 0x52494350, 0x10DE, 0, 0, 0x18, 0, 0, 0, 3,
                       n if ext_len is None else n + 7, 0, code_type, 0x80 if last else 0)
    b[0x40:0x40 + len(pcir)] = pcir
    if ext_len is not None:
        struct.pack_into("<IHHHB", b, 0x60, 0x4544504E, 0x100, 12, ext_len, 0x80 if last else 0)
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
    struct.pack_into("<BBBB", ext, a, 1, 4, 8, 2)
    struct.pack_into("<II", ext, a + 4, 1, 0x40)
    struct.pack_into("<II", ext, a + 12, 4 if mapper else 5, MAPPER)
    struct.pack_into("<I", ext, img + IMEM + MAPPER + 8, CMD_IN)
    return bytes(base + efi[:EFI] + ext)


def main():
    for name, r in [("vbios.rom", rom()), ("vbios_no_bit.rom", rom(bit_ok=False)),
                    ("vbios_debug.rom", rom(prod=False)), ("vbios_no_mapper.rom", rom(mapper=False))]:
        with open(name, "wb") as f:
            f.write(r)


if __name__ == "__main__":
    main()
