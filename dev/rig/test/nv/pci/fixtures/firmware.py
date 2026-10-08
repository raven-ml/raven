# /// script
# requires-python = ">=3.10"
# ///
"""Writes firmware containers laid out as NVIDIA's firmware is, with bytes
of their own: the layouts of nouveau's nvfw/fw.h and nvfw/hs.h (Linux 6.18)
and of RM_RISCV_UCODE_DESC (rmRiscvUcode.h, open-gpu-kernel-modules
570.144). Run from this directory:

  uv run firmware.py

- booter.bin: a heavy-secure container (nvfw_bin_hdr, nvfw_hs_header_v2,
  nvfw_hs_load_header_v2) whose data is 0x1000 bytes of code then 0x800 of
  data, each byte its offset modulo 256. It carries two production
  signatures of 384 bytes, of 0xa1 and 0xa2 bytes, the second of which goes
  at 0x1010 of the data. Its patch location, signature index and signature
  count are words the header points to, as NVIDIA's are; its patch metadata
  names engines 0x5 and ucode ID 9.
- bootloader.bin: a container whose data is 0x3000 bytes and whose header
  is an RM_RISCV_UCODE_DESC with its manifest at 0x100, its monitor data at
  0x800 and its monitor code at 0x1000.
- fmc.elf: a 32-bit ELF object with the FMC's sections: hash (48 bytes),
  signature (96), publickey (97) and image (0x2000), each filled with one
  byte: 0x48, 0x53, 0x50, 0x49.
"""

import struct

DATA_AT = 0x400


def container(header, data):
    """nvfw_bin_hdr: bin_magic, bin_ver, bin_size, header_offset,
    data_offset, data_size; the header at 0x18, the data at DATA_AT."""
    b = bytearray(DATA_AT + len(data))
    struct.pack_into("<6I", b, 0, 0x10DE, 1, len(b), 0x18, DATA_AT, len(data))
    b[0x18:0x18 + len(header)] = header
    b[DATA_AT:] = data
    return bytes(b)


def booter():
    code, data = 0x1000, 0x800
    image = bytes(i & 0xFF for i in range(code + data))
    hs = 0x18
    sigs = hs + 0x40
    words = sigs + 2 * 384
    meta = words + 0x10
    load = meta + 0x10
    # nvfw_hs_header_v2: sig_prod_offset, sig_prod_size, patch_loc, patch_sig,
    # meta_data_offset, meta_data_size, num_sig, header_offset, header_size,
    # every offset from the container's start.
    header = bytearray(load + 0x20 - hs)
    struct.pack_into("<9I", header, 0, sigs, 2 * 384, words, words + 4, meta, 12, words + 8, load, 0x20)
    header[sigs - hs:sigs - hs + 384] = bytes([0xA1]) * 384
    header[sigs - hs + 384:sigs - hs + 768] = bytes([0xA2]) * 384
    # The words: the patch location, the signature to patch (an offset into
    # the signatures), and the number of signatures.
    struct.pack_into("<3I", header, words - hs, code + 0x10, 384, 2)
    # The patch metadata: fuse version 1, engines 0x5, ucode ID 9.
    struct.pack_into("<3I", header, meta - hs, 1, 0x5, 9)
    # nvfw_hs_load_header_v2: os_code_offset, os_code_size, os_data_offset,
    # os_data_size, num_apps, then each application's offset and size.
    struct.pack_into("<7I", header, load - hs, 0, 0x100, code, data, 1, 0x100, code - 0x100)
    return container(bytes(header), image)


def bootloader():
    desc = bytearray(84)
    struct.pack_into("<I", desc, 0x20, 0x100)
    struct.pack_into("<I", desc, 0x28, 0x800)
    struct.pack_into("<I", desc, 0x30, 0x1000)
    return container(bytes(desc), bytes(0x3000))


def elf32(sections):
    """A 32-bit little-endian ELF object with [sections], (name, bytes)."""
    names = b"\0" + b"".join(n.encode() + b"\0" for n, _ in sections) + b".shstrtab\0"
    body, at = b"", 52
    offsets = []
    for _, d in sections:
        offsets.append(at + len(body))
        body += d
    shstr = at + len(body)
    body += names
    shoff = at + len(body)
    count = len(sections) + 2
    head = b"\x7fELF" + bytes([1, 1, 1, 0]) + bytes(8)
    head += struct.pack("<HHIIIIIHHHHHH", 1, 0, 1, 0, 0, shoff, 0, 52, 0, 0, 40, count, count - 1)
    shdrs, name = bytes(40), 1
    for (n, d), off in zip(sections, offsets):
        shdrs += struct.pack("<10I", name, 1, 0, 0, off, len(d), 0, 0, 1, 0)
        name += len(n) + 1
    shdrs += struct.pack("<10I", name, 3, 0, 0, shstr, len(names), 0, 0, 1, 0)
    return head + body + shdrs


def fmc():
    return elf32([("hash", b"H" * 48), ("signature", b"S" * 96), ("publickey", b"P" * 97),
                  ("image", b"I" * 0x2000)])


def main():
    for name, data in [("booter.bin", booter()), ("bootloader.bin", bootloader()), ("fmc.elf", fmc())]:
        with open(name, "wb") as f:
            f.write(data)


if __name__ == "__main__":
    main()
