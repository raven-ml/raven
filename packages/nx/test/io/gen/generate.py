#!/usr/bin/env python3
# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.5.4", "safetensors==0.8.0", "torch==2.15.0",
#                 "pillow==13.0.0", "rasterio==1.5.2", "h5py==3.16.0"]
# ///
"""Generate nx.io's test fixtures: files other tools write.

numpy writes the NPY, NPZ and text files and the float16 SafeTensors file,
torch the other SafeTensors files and the PyTorch checkpoints pt_*.pt, and
zipfile the NPZ files numpy would not write. A checkpoint's tensors hold the
bytes io_support.ml's [pattern n seed] gives, or small ranges, and torch
reads each file back as what it saved; pt_big.pt and pt_deflated.pt are
torch's files rewritten, and the legacy file's storage keys, which torch
takes from memory addresses, are renamed in place to fixed ones. The PNG files come from the minimal writer below, which chooses the row
filters, sample depths and interlacing that an encoder picks on its own, except
png_rgba.png and png_la.png, which Pillow writes as it writes the JPEG files. Each image has a
sibling .npy of the pixels Pillow decodes from it, of shape (h, w, c), except
png_gray_trns.png and png_rgb16_trns.png, whose pixels are stated from PNG-3
where Pillow's decoding differs. No fixture is a YCCK JPEG, which Pillow does
not write, so the YCCK conversion goes untested. Python's float(), whose
reader rounds correctly on every system, gives the float64 nearest each
literal of floats.txt. The JPEG files that break T.81's rules are written
byte by byte below, or cut from Pillow's files, and have no sibling, except
jpeg_table16_coarse.jpg, whose 16-bit table Pillow writes and decodes. Nor
have jpeg_zrl_progressive.jpg and jpeg_zrl_baseline.jpg, which decode to the
same pixels. GDAL 3.12, which the rasterio wheel bundles with its libtiff,
writes the tiff_*.tif files, each with a sibling .npy of the array of shape
(h, w, c) it was given, and reads each back as that array; the tiff_*_u<n>
files hold samples of n bits, packed (GDAL's NBITS). h5py 3.16, with the
libhdf5 2.0 its wheel bundles, writes the hdf5_*.h5 files in its default
format (superblock 0, version-1 object headers, groups as symbol tables).
Each file nx.io reads whole has a sibling .npz of every dataset h5py reads
back, named by its path without the first /, an element of a datatype nx
lacks as its bytes; the others hold what nx.io refuses.

The fixtures are written into ../support, under Python 3.13, for which
every pinned package has a wheel. Pillow 13.0.0 encodes JPEG with
the libjpeg-turbo its wheel bundles, and every fixture reproduces byte for
byte with the pinned versions. With --check nothing is written, and the run
fails if a fixture would change.

Usage:
    uv run packages/nx/test/io/gen/generate.py [--check]
"""

import argparse
import filecmp
import io
import os
import pickle
import random
import re
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch


def safetensors_fixtures():
    # [+0, smallest subnormal, 1.0, +inf, NaN] at float16 and bfloat16
    f16_bits = [0x0000, 0x0001, 0x3C00, 0x7C00, 0x7E01]
    f16 = np.frombuffer(struct.pack("<5H", *f16_bits), dtype=np.float16)
    save_file({"f16_tensor": f16}, "f16_bit_exact.safetensors")
    bf16_bits = [0x0000, 0x0001, 0x3F80, 0x7F80, 0x7FC1]
    bf16 = torch.tensor(bf16_bits, dtype=torch.int16).view(torch.bfloat16)
    save_torch({"bf16_tensor": bf16}, "bf16_bit_exact.safetensors")
    # [1+2j, -0-infj, NaN+0.5j] at complex64, from the float32 bits of each part
    c64_bits = [0x3F800000, 0x40000000, 0x80000000, 0xFF800000, 0x7FC00000, 0x3F000000]
    c64 = torch.from_numpy(np.array(c64_bits, dtype="<u4").view(np.complex64))
    save_torch({"c64_tensor": c64}, "c64_bit_exact.safetensors")
    # The two FNUZ float8 formats, which nx lacks, each of the bytes [0, 1,
    # 0x80, 0x7F]: zero, the least subnormal, NaN and the largest finite value.
    fnuz = torch.tensor([0x00, 0x01, 0x80, 0x7F], dtype=torch.uint8)
    save_torch({"e4m3fnuz": fnuz.clone().view(torch.float8_e4m3fnuz),
                "e5m2fnuz": fnuz.clone().view(torch.float8_e5m2fnuz)},
               "fnuz.safetensors")


def float_literals():
    # Float literals, each with the bits of the float64 nearest it in hex: up
    # to 17 significant digits, up to 40, among the subnormals, about the
    # overflow threshold, and past the 800 digits a literal keeps.
    rng = random.Random(7)

    def literal(n, lead):
        d = str(rng.randint(1, 9)) + "".join(rng.choice("0123456789") for _ in range(n - 1))
        p = rng.randint(0, n)
        return rng.choice(["", "-", "+"]) + d[:p] + "." + d[p:] + "e" + str(lead - p + 1)

    reaches = [(300, (1, 17), (-20, 20)), (100, (18, 40), (-20, 20)),
               (100, (1, 40), (-330, -305)), (100, (1, 40), (300, 312)),
               (4, (790, 900), (-310, 300))]
    with open("floats.txt", "w") as f:
        for count, digits, lead in reaches:
            for _ in range(count):
                s = literal(rng.randint(*digits), rng.randint(*lead))
                bits = struct.unpack("<Q", struct.pack("<d", float(s)))[0]
                f.write("%s %016x\n" % (s, bits))


def numpy_fixtures():
    np.save("fortran.npy", np.asfortranarray(np.arange(1, 7, dtype="<f4").reshape(2, 3)))
    np.save("big_endian.npy", np.array([1, 256, -2], dtype=">i2"))
    for version, array in [((2, 0), np.array([7, 8, 9], dtype="<i4")),
                           ((3, 0), np.array([True, False, True]))]:
        with open("version%d.npy" % version[0], "wb") as f:
            np.lib.format.write_array(f, array, version=version)
    np.save("unicode.npy", np.array(["abc"]))
    np.savez_compressed("archive.npz", a=np.array([[1.5, -2.0]], dtype="<f4"),
                        b=np.array([3, 4, 5], dtype="<i8"))
    # Stored, as np.savez leaves its entries.
    np.savez("savez.npz", a=np.array([1, 2, 65535], dtype="<u2"),
             b=np.array([[True, False]]))
    np.savetxt("savetxt.txt",
               np.array([[0.1, 1e23, 5e-324, 2.2250738585072014e-308, 1.23456789],
                         [-0.0, np.nan, np.inf, -np.inf, 1.7976931348623157e308]]),
               delimiter=",", newline="\r\n", header="alpha beta gamma",
               footer="generated by numpy", comments="##")


def xorshift(n):
    # n bytes of noise: the low byte of each state of Marsaglia's xorshift32
    # from 2463534242, which the tests compute again.
    x, out = 2463534242, bytearray(n)
    for i in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = x & 0xFF
    return np.frombuffer(bytes(out), dtype=np.uint8)


def has_stored_block(path, payload):
    # Whether the archive's one entry holds a stored block: a LEN and its
    # complement NLEN, then LEN bytes of the entry's data. Found by search,
    # since a block of codes ends at no byte boundary.
    with zipfile.ZipFile(path) as z:
        (info,) = z.infolist()
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        name, extra = struct.unpack("<HH", local[26:30])
        f.seek(name + extra, 1)
        raw = f.read(info.compress_size)
    for k in range(len(raw) - 4):
        n, complement = struct.unpack("<HH", raw[k:k + 4])
        if n >= 256 and n ^ complement == 0xFFFF and raw[k + 4:k + 4 + n] in payload:
            return True
    return False


def stored_block_npz():
    # zlib stores a block that coding would not shrink, so noise after a run
    # of zeros puts a stored block inside the entry's deflate stream.
    w = np.concatenate([np.zeros(1024, dtype=np.uint8), xorshift(20000)])
    np.savez_compressed("stored_block.npz", w=w)
    assert has_stored_block("stored_block.npz", w.tobytes())


def unaligned_npz():
    # A deflated entry whose NPY header is not padded, so that its float64
    # data starts at no multiple of 8, as writers other than numpy leave it.
    header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (5,), }\n"
    assert (10 + len(header)) % 8 != 0
    data = struct.pack("<5d", 0.0, 1.5, 3.0, 4.5, 6.0)
    npy = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data
    # A fixed date, so that the archive is the same on every run.
    entry = zipfile.ZipInfo("w.npy", date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("unaligned.npz", "w") as z:
        z.writestr(entry, npy)


def other_names_npz():
    # A stored npy entry beside an entry of another name, which a reader of
    # npz files ignores.
    npy = io.BytesIO()
    np.lib.format.write_array(npy, np.array([1.5, -2.0], dtype="<f8"))
    with zipfile.ZipFile("other_names.npz", "w") as z:
        for name, data in [("w.npy", npy.getvalue()), ("README.txt", b"not an array\n")]:
            z.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)


def skipped_entries_npz():
    # An npy entry beside entries a reader of npz files skips without reading
    # them: a directory, which zip -r writes, and a file compressed with bzip2,
    # a method npz does not use. Then an npy entry whose name is the suffix
    # alone, which names no array.
    npy = io.BytesIO()
    np.lib.format.write_array(npy, np.array([3, -4], dtype="<i2"))
    date = (1980, 1, 1, 0, 0, 0)
    notes = zipfile.ZipInfo("weights/notes.txt", date_time=date)
    notes.compress_type = zipfile.ZIP_BZIP2
    with zipfile.ZipFile("skipped_entries.npz", "w") as z:
        z.writestr(zipfile.ZipInfo("weights/", date_time=date), b"")
        z.writestr(notes, b"not an array\n" * 8)
        z.writestr(zipfile.ZipInfo("weights/w.npy", date_time=date), npy.getvalue())
    with zipfile.ZipFile("suffix_only.npz", "w") as z:
        z.writestr(zipfile.ZipInfo(".npy", date_time=date), npy.getvalue())


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def filtered(kind, line, prev, bpp):
    out = bytearray([kind])
    for i, x in enumerate(line):
        a = line[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        out.append((x - [0, a, b, (a + b) // 2, paeth(a, b, c)][kind]) & 255)
    return bytes(out)


def packed(samples, depth):
    if depth == 16:
        return b"".join(struct.pack(">H", s) for s in samples)
    if depth == 8:
        return bytes(samples)
    per = 8 // depth
    samples = samples + [0] * (-len(samples) % per)
    return bytes(sum(s << (8 - depth * (k + 1)) for k, s in enumerate(samples[i:i + per]))
                 for i in range(0, len(samples), per))


def chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data)))


def write_png(path, pixels, depth, color, filters=(0,), palette=None,
              trns=None, interlace=False):
    """pixels: rows of pixels, each a tuple of samples."""
    height, width, channels = len(pixels), len(pixels[0]), len(pixels[0][0])
    passes = ([(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
               (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
              if interlace else [(0, 0, 1, 1)])
    bpp = max(1, depth * channels // 8)
    data = b""
    for x0, y0, dx, dy in passes:
        xs, ys = range(x0, width, dx), range(y0, height, dy)
        prev = None
        for k, y in enumerate(ys):
            if not xs:
                break
            line = packed([s for x in xs for s in pixels[y][x]], depth)
            prev = prev or bytes(len(line))
            data += filtered(filters[k % len(filters)], line, prev, bpp)
            prev = line
    header = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, int(interlace))
    body = chunk(b"IHDR", header)
    if palette:
        body += chunk(b"PLTE", bytes(v for rgb in palette for v in rgb))
    if trns:
        body += chunk(b"tRNS", bytes(trns))
    body += chunk(b"IDAT", zlib.compress(data, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + body)


def reference(path, mode):
    """The pixels Pillow decodes from [path] in [mode], of shape (h, w, c): a
    grey image keeps a channel axis of one."""
    pixels = np.asarray(Image.open(path).convert(mode))
    np.save(path.rsplit(".", 1)[0] + ".npy", pixels.reshape(pixels.shape[:2] + (-1,)))


def stated(path, pixels):
    """The pixels of [path] as PNG-3 states them, where Pillow's decoding
    differs: a tRNS colour is matched before samples are scaled to 8 bits
    (section 11.3.1.1)."""
    np.save(path.rsplit(".", 1)[0] + ".npy", np.array(pixels, dtype=np.uint8))


def extended_jpeg():
    # An extended sequential frame (SOF1) at 8 bits, which Pillow does not
    # write: the baseline file with its SOF0 marker renamed, since the two
    # frame headers are the same (T.81, B.2.2).
    with open("jpeg_baseline.jpg", "rb") as f:
        data = f.read()
    at = data.index(b"\xff\xc0")
    with open("jpeg_extended.jpg", "wb") as f:
        f.write(data[:at] + b"\xff\xc1" + data[at + 2:])


def late_table_jpeg():
    # The grey file with its table redefined after its scan: a component
    # keeps the table it had at its first scan, as libjpeg latches it, so the
    # pixels do not change.
    with open("jpeg_gray.jpg", "rb") as f:
        data = f.read()
    assert data.endswith(b"\xff\xd9")
    with open("jpeg_late_table.jpg", "wb") as f:
        f.write(data[:-2] + segment(0xDB, bytes([0x00]) + bytes([1] * 64)) + data[-2:])


def scan_spans(data):
    """The byte spans of the scans of the JPEG file [data]: each scan's SOS
    segment and its entropy-coded data, up to the next marker other than a
    restart."""
    spans, i = [], 2
    while data[i + 1] != 0xD9:
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        end = i + 2 + length
        if data[i + 1] == 0xDA:
            while not (data[end] == 0xFF and data[end + 1] != 0x00
                       and not 0xD0 <= data[end + 1] <= 0xD7):
                end += 1
            spans.append((i, end))
        i = end
    return spans


def progression_jpegs():
    # The progressive file with its first scan, DC with Al = 1, sent twice;
    # without it, so that the DC refinement refines bits never sent; and with
    # it after the second, the first AC scan of luma. Each coefficient's bits
    # are sent once, in order, and a component's DC before its AC (T.81,
    # G.1.1.1).
    with open("jpeg_progressive.jpg", "rb") as f:
        data = f.read()
    (start, end), (_, second) = scan_spans(data)[:2]
    with open("jpeg_repeated_scan.jpg", "wb") as f:
        f.write(data[:end] + data[start:end] + data[end:])
    with open("jpeg_unsent_refinement.jpg", "wb") as f:
        f.write(data[:start] + data[end:])
    with open("jpeg_ac_before_dc.jpg", "wb") as f:
        f.write(data[:start] + data[end:second] + data[start:end] + data[second:])


def zrl_jpegs():
    # Noise saved progressive and baseline, whose coefficients are the same,
    # so both decode to the same pixels. In this shape of the noise, an AC
    # refinement scan holds a run of 16 zeros (ZRL) followed by coefficients
    # already sent, whose correction bits come after the next code (T.81,
    # G.1.2.3).
    im = Image.fromarray(xorshift(128 * 64).reshape(128, 64), "L")
    im.save("jpeg_zrl_progressive.jpg", quality=50, progressive=True)
    im.save("jpeg_zrl_baseline.jpg", quality=50)


def segment(marker, data):
    return bytes([0xFF, marker]) + struct.pack(">H", len(data) + 2) + data


class Bits:
    """An entropy-coded segment, written most significant bit first, with
    each 0xFF byte stuffed with a 0x00 and the last byte padded with ones."""

    def __init__(self):
        self.out, self.acc, self.n = bytearray(), 0, 0

    def put(self, v, n):
        for i in range(n - 1, -1, -1):
            self.acc = (self.acc << 1) | ((v >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0)
                self.acc, self.n = 0, 0

    def done(self):
        while self.n:
            self.put(1, 1)
        return bytes(self.out)


def overflow_jpeg(path, side, progressive, table16):
    # A grey JPEG of [side] by [side] pixels whose every DC difference is
    # +2047, so that the DC predictor grows past 16 bits, with quantisation
    # entries of 255, or of 65535 in a 16-bit table. A progressive file sends
    # DC with Al = 13.
    jpeg = b"\xff\xd8"
    if table16:
        jpeg += segment(0xDB, bytes([0x10]) + b"\xff\xff" * 64)
    else:
        jpeg += segment(0xDB, bytes([0x00]) + b"\xff" * 64)
    sof = 0xC2 if progressive else 0xC0
    jpeg += segment(sof, bytes([8]) + struct.pack(">HH", side, side) + bytes([1, 1, 0x11, 0]))
    counts = bytes([1] + [0] * 15)
    jpeg += segment(0xC4, bytes([0x00]) + counts + bytes([11]))  # DC: code 0 is category 11
    jpeg += segment(0xC4, bytes([0x10]) + counts + bytes([0x00]))  # AC: code 0 is EOB
    bits = Bits()
    if progressive:
        jpeg += segment(0xDA, bytes([1, 1, 0x00, 0, 0, 13]))
        for _ in range((side // 8) ** 2):
            bits.put(0, 1)
            bits.put(0x7FF, 11)
    else:
        jpeg += segment(0xDA, bytes([1, 1, 0x00, 0, 63, 0]))
        for _ in range((side // 8) ** 2):
            bits.put(0, 1)
            bits.put(0x7FF, 11)
            bits.put(0, 1)
    with open(path, "wb") as f:
        f.write(jpeg + bits.done() + b"\xff\xd9")


def image_fixtures():
    rgb = [[((31 * y + 73 * x) % 256, (73 * y + 11 * x) % 256, (47 * x * y + 5) % 256)
            for x in range(3)] for y in range(5)]
    write_png("png_filters.png", rgb, 8, 2, filters=(0, 1, 2, 3, 4))
    reference("png_filters.png", "RGB")
    bits = [[((x + y) % 3 == 0,) for x in range(8)] for y in range(2)]
    write_png("png_gray1.png", [[(int(b),) for (b,) in row] for row in bits], 1, 0,
              filters=(1, 2))
    reference("png_gray1.png", "L")
    write_png("png_rgb16.png", [[(0x0000, 0xFFFF, 0x8000), (0x1234, 0xABCD, 0x01FF)]],
              16, 2, filters=(4,))
    reference("png_rgb16.png", "RGB")
    # Every 4-bit sample, two to a byte, in rows of an odd width.
    write_png("png_gray4.png", [[((5 * y + x) % 16,) for x in range(5)] for y in range(4)],
              4, 0, filters=(0, 1, 2, 3))
    reference("png_gray4.png", "L")
    indices = [[((x + 2 * y) % 4,) for x in range(5)] for y in range(5)]
    write_png("png_palette_adam7.png", indices, 2, 3, filters=(0, 1, 2, 3, 4),
              palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
              trns=[255, 128, 255, 0], interlace=True)
    reference("png_palette_adam7.png", "RGBA")
    # A tRNS chunk shorter than the palette: the entries past its end are
    # opaque.
    write_png("png_palette_short_trns.png", indices[:3], 2, 3,
              palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
              trns=[0])
    reference("png_palette_short_trns.png", "RGBA")
    rgba = np.fromfunction(lambda y, x, c: (37 * y + 59 * x + 83 * c) % 256,
                           (4, 5, 4)).astype(np.uint8)
    Image.fromarray(rgba).save("png_rgba.png")
    reference("png_rgba.png", "RGBA")
    Image.fromarray(np.ascontiguousarray(rgba[:, :, 1::2])).save("png_la.png")
    reference("png_la.png", "LA")
    # A 2-bit grey image whose tRNS value has bits above the depth set, which
    # a decoder masks: the transparent sample is 2.
    grey = [[0, 1, 2, 3], [3, 2, 1, 0]]
    write_png("png_gray_trns.png", [[(s,) for s in row] for row in grey], 2, 0,
              trns=[0xFF, 0x02])
    stated("png_gray_trns.png",
           [[(85 * s, 0 if s == 2 else 255) for s in row] for row in grey])
    # A 16-bit colour image whose second pixel differs from the tRNS colour
    # only in its low bytes, which 8 bits drop.
    write_png("png_rgb16_trns.png",
              [[(0x1234, 0x5678, 0x9ABC), (0x12FF, 0x56FF, 0x9AFF), (0, 0, 0)]],
              16, 2, trns=list(struct.pack(">HHH", 0x1234, 0x5678, 0x9ABC)))
    stated("png_rgb16_trns.png",
           [[(0x12, 0x56, 0x9A, 0), (0x12, 0x56, 0x9A, 255), (0, 0, 0, 255)]])

    def gradient(h, w, mode):
        a = np.fromfunction(lambda y, x, c: 10 + 12 * y + 8 * x + 30 * c,
                            (h, w, len(mode)))
        return Image.fromarray(a.astype(np.uint8).squeeze(), mode)

    gradient(8, 8, "RGB").save("jpeg_baseline.jpg", quality=90)
    gradient(7, 9, "RGB").save("jpeg_progressive.jpg", quality=90, progressive=True)
    gradient(6, 5, "L").save("jpeg_gray.jpg", quality=90)
    gradient(5, 6, "CMYK").save("jpeg_cmyk.jpg", quality=90)
    gradient(9, 11, "RGB").save("jpeg_restart.jpg", quality=90, restart_marker_blocks=1)
    extended_jpeg()
    # Dark content at quality 50, so that many DC coefficients are negative
    # and odd: Pillow's progressive script sends DC without its low bit, then
    # refines it.
    y, x = np.mgrid[0:64, 0:64]
    base = (45 + 30 * np.sin(x / 7.0) * np.cos(y / 5.0)).clip(0, 255)
    dark = np.stack([base, base * 0.5 + 10, 255 - base * 3], -1)
    Image.fromarray(dark.clip(0, 255).astype(np.uint8), "RGB").save(
        "jpeg_progressive_dark.jpg", quality=50, progressive=True)
    for name in ["baseline", "progressive", "progressive_dark", "cmyk", "restart",
                 "extended"]:
        reference("jpeg_%s.jpg" % name, "RGB")
    reference("jpeg_gray.jpg", "L")
    late_table_jpeg()
    reference("jpeg_late_table.jpg", "L")
    progression_jpegs()
    zrl_jpegs()
    overflow_jpeg("jpeg_dc_overflow.jpg", 128, False, False)
    overflow_jpeg("jpeg_dc_overflow_progressive.jpg", 128, True, False)
    overflow_jpeg("jpeg_table16.jpg", 32, False, True)
    overflow_jpeg("jpeg_table16_large.jpg", 512, False, True)
    # Entries above 255, which libjpeg writes in a 16-bit table, in an 8-bit
    # frame, for a coarse quality: T.81 B.2.4.1 reserves such tables for
    # 12-bit samples, but every libjpeg reader opens the file.
    coarse = [4 * q for q in [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                              14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                              18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                              92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100,
                              103, 99]]
    gradient(8, 16, "L").save("jpeg_table16_coarse.jpg", qtables=[coarse])
    reference("jpeg_table16_coarse.jpg", "L")


def pt_pattern(n, seed):
    return bytes(((37 * i) + seed) & 255 for i in range(n))


def pt_tensor(dtype, shape, seed):
    n = 1
    for d in shape:
        n *= d
    b = bytearray(pt_pattern(n * dtype.itemsize, seed))
    if dtype == torch.bool:
        b = bytearray(x & 1 for x in b)
    return torch.frombuffer(b, dtype=torch.uint8).clone().view(dtype).reshape(shape)


def pt_rewrite(src, dst, edit, compress=lambda name: zipfile.ZIP_STORED):
    # Rewrites a checkpoint's entries through zipfile, each edited.
    with zipfile.ZipFile(src) as z, zipfile.ZipFile(dst, "w") as w:
        for info in z.infolist():
            data = edit(info.filename, z.read(info.filename))
            w.writestr(zipfile.ZipInfo(info.filename), data,
                       compress_type=compress(info.filename))


def pt_same(a, b):
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or a.stride() != b.stride():
            return False
        flat = lambda t: bytes(t.contiguous().reshape(-1).view(torch.uint8).numpy())
        return a.numel() == 0 or flat(a) == flat(b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(pt_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(map(pt_same, a, b))
    return True


def pytorch_fixtures():
    import argparse as ap
    import collections
    import pickletools
    # Every dtype a storage holds, at shape (2, 3), seeded by its index.
    names = ["float64", "float32", "float16", "bfloat16", "int64", "int32",
             "int16", "int8", "uint8", "bool", "complex128", "complex64",
             "uint16", "uint32", "uint64", "float8_e4m3fn", "float8_e5m2",
             "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu",
             "float4_e2m1fn_x2", "complex32", "bits8", "bits16", "bits1x8",
             "bits2x4", "bits4x2"]
    dtypes = {n: pt_tensor(getattr(torch, n), (2, 3), i) for i, n in enumerate(names)}
    torch.save(dtypes, "pt_dtypes.pt")
    # Views of one float32 storage of 0..23.
    base = torch.arange(24, dtype=torch.float32)
    shared = torch.arange(2, dtype=torch.int64)
    views = {
        "base": base.view(4, 6), "offset": base[5:11],
        "transposed": base.view(4, 6).t(), "strided": base[1::3],
        "tied": base.view(6, 4),
        "expanded": torch.arange(3, dtype=torch.float32).expand(4, 3),
        "empty": torch.empty(0, 3), "scalar": torch.tensor(2.5),
        "param": torch.nn.Parameter(torch.ones(2)), "shared": [shared, shared],
    }
    torch.save(views, "pt_views.pt")
    # Python's data at each protocol.
    meta = {
        "none": None, "true": True, "false": False, "int": 7, "neg": -(2 ** 40),
        "big": 2 ** 70, "negbig": -(2 ** 70), "float": 1.5,
        "inf": float("inf"), "negzero": -0.0, "complex": complex(1, -2),
        "str": "\u00e9\u65e5", "bytes": b"\x00\xff", "bytearray": bytearray(b"ab"),
        "empty_bytes": b"", "tuple": (1, "a"), "list": [1, 2], "set": {3, 4},
        "frozenset": frozenset([5]), "odict": collections.OrderedDict([("a", 1)]),
        "counter": collections.Counter({"x": 2}), "size": torch.Size([2, 3]),
        "dtype": torch.float32, "namespace": ap.Namespace(a=1),
        "keys": {0: "zero", 1.5: "half", (1, 2): "pair"},
    }
    with_meta = {"w": torch.arange(4, dtype=torch.float32), "meta": meta}
    for p in [1, 2, 3, 4, 5]:
        torch.save(with_meta, f"pt_protocol{p}.pt", pickle_protocol=p)
    # The legacy format: views of one storage, and one of its own. torch names
    # storages by memory address and writes them in their names' order, so
    # each is renamed by the order the value names it in, as 0, 1, ..., and
    # the keys and storages are written again in that order.
    legacy = {"w": base.view(4, 6), "t": base.view(4, 6).t(), "o": base[5:11],
              "own": torch.arange(3, dtype=torch.int64)}
    torch.save(legacy, "pt_legacy.pt", _use_new_zipfile_serialization=False)
    with open("pt_legacy.pt", "rb") as f:
        data = f.read()
    stream = io.BytesIO(data)
    for _ in range(3):
        pickletools.dis(stream, out=io.StringIO())
    value_ops = list(pickletools.genops(stream))
    keys_at = stream.tell()
    keys = [a for op, a, _ in pickletools.genops(stream) if op.name == "BINUNICODE"]
    storages_at = stream.tell()
    # Each key's element size, from the storage class its persistent id
    # names just before it, written or fetched from the memo.
    size, last, memo = {}, None, {}
    for op, arg, _ in value_ops:
        if op.name in ("GLOBAL", "BINUNICODE"):
            if op.name == "BINUNICODE" and arg in keys and last.endswith("Storage"):
                size.setdefault(arg, torch.storage._get_dtype_from_pickle_storage_type(
                    last.split()[1]).itemsize)
            last = arg
        elif op.name == "BINPUT":
            memo[arg] = last
        elif op.name == "BINGET":
            last = memo.get(arg)
    named = [k for k in dict.fromkeys(a for op, a, _ in value_ops if op.name == "BINUNICODE") if k in keys]
    rename = {k: str(i) for i, k in enumerate(named)}
    records, pos = {}, storages_at
    for k in keys:
        n = struct.unpack("<q", data[pos:pos + 8])[0]
        records[k] = data[pos:pos + 8 + n * size[k]]
        pos += 8 + n * size[k]
    head = data[:keys_at]
    unicode = lambda k: b"X" + struct.pack("<I", len(k)) + k.encode()
    for k, new in rename.items():
        head = head.replace(unicode(k), unicode(new))
    with open("pt_legacy.pt", "wb") as f:
        f.write(head)
        pickle.dump(sorted(rename.values()), f, protocol=2)
        for k in sorted(keys, key=rename.get):
            f.write(records[k])
    # A whole module, its tensors inside its object.
    torch.manual_seed(0)
    torch.save(torch.nn.Linear(2, 3), "pt_module.pt")
    # A tensor saved with its conjugate bit set.
    torch.save({"c": torch.tensor([1 + 2j], dtype=torch.complex64).conj()},
               "pt_conj.pt")
    # CRCs off.
    torch.serialization.set_crc32_options(False)
    torch.save({"w": torch.arange(4, dtype=torch.float32)}, "pt_nocrc.pt")
    torch.serialization.set_crc32_options(True)
    # A TorchScript archive, its random serialization id zeroed. Its code lists
    # attributes in the order of a set of strings, so it is written under a
    # fixed hash seed.
    subprocess.run(
        [sys.executable, "-c",
         "import torch; torch.manual_seed(0); "
         "torch.jit.save(torch.jit.script(torch.nn.Linear(2, 2)), 'script.tmp')"],
        env={**os.environ, "PYTHONHASHSEED": "0"}, check=True,
        stderr=subprocess.DEVNULL)
    # Its debug records hold the absolute path of torch's sources, which
    # differs between environments: they are blanked.
    def script(name, data):
        if name.endswith("serialization_id"):
            return b"0" * len(data)
        return b"" if name.endswith(".debug_pkl") else data
    pt_rewrite("script.tmp", "pt_script.pt", script)
    os.remove("script.tmp")
    # A big-endian file: a big-endian host's torch writes "big" and each
    # element's bytes reversed. Storages 0, 1 and 2 hold f, c and i, of
    # components of 4, 4 and 2 bytes.
    big = {"f": torch.tensor([1.5, -2.0], dtype=torch.float32),
           "c": torch.tensor([1 + 2j], dtype=torch.complex64),
           "i": torch.tensor([1, -2], dtype=torch.int16)}
    torch.save(big, "big.tmp")
    width = {"0": 4, "1": 4, "2": 2}

    def swap(name, data):
        if name.endswith("/byteorder"):
            return b"big"
        if "/data/" in name:
            k = width[name.rsplit("/", 1)[1]]
            return b"".join(data[i:i + k][::-1] for i in range(0, len(data), k))
        return data
    pt_rewrite("big.tmp", "pt_big.pt", swap)
    os.remove("big.tmp")
    # The views' storages compressed.
    pt_rewrite("pt_views.pt", "pt_deflated.pt", lambda n, d: d,
               compress=lambda n: zipfile.ZIP_DEFLATED if "/data/" in n
               else zipfile.ZIP_STORED)
    for name, obj in [("pt_dtypes.pt", dtypes), ("pt_views.pt", views),
                      ("pt_legacy.pt", legacy), ("pt_big.pt", big),
                      ("pt_deflated.pt", views)] + \
            [(f"pt_protocol{p}.pt", with_meta) for p in [1, 2, 3, 4, 5]]:
        assert pt_same(obj, torch.load(name, weights_only=False)), name


def tiff_fixtures():
    import rasterio
    import warnings
    from rasterio.errors import NotGeoreferencedWarning

    def ramp(dtype, shape, scale):
        # A smooth ramp across rows, columns and samples, plus noise, so that
        # the predictors and the codecs have runs to find and bits to keep.
        h, w, c = shape
        i, j, k = np.meshgrid(np.arange(h), np.arange(w), np.arange(c), indexing="ij")
        noise = xorshift(h * w * c).reshape(shape).astype(np.float64)
        return ((i * 3 + j * 2 + k * 50) * scale + noise).astype(dtype)

    f32 = ramp(np.float32, (29, 41, 1), 0.25)
    f32[0, 0, 0], f32[1, 2, 0], f32[3, 4, 0], f32[5, 6, 0] = np.nan, np.inf, -np.inf, -0.0
    f64 = ramp(np.float64, (23, 19, 3), -1e-3)
    f64[2, 3, 1] = np.finfo(np.float64).tiny / 4
    cases = [
        ("tiff_lzw_u16_tiles", ramp(np.uint16, (37, 53, 2), 97),
         dict(compress="lzw", predictor=2, tiled=True, blockxsize=16, blockysize=16)),
        ("tiff_deflate_f32_bigtiff_mm", f32,
         dict(compress="deflate", predictor=3, bigtiff="YES", endianness="BIG", blockysize=4)),
        ("tiff_lzw_f64_strips", f64, dict(compress="lzw", predictor=3, blockysize=5)),
        ("tiff_deflate_i16_mm_tiles", ramp(np.int16, (40, 33, 1), -251),
         dict(compress="deflate", predictor=2, tiled=True, blockxsize=32, blockysize=16,
              endianness="BIG")),
        ("tiff_none_u32", ramp(np.uint32, (17, 13, 1), 1 << 20), dict(compress="none")),
        ("tiff_lzw_i32", ramp(np.int32, (11, 300, 1), -(1 << 18)), dict(compress="lzw", predictor=2)),
        ("tiff_lzw_u8_rgb", ramp(np.uint8, (31, 47, 3), 1), dict(compress="lzw")),
    ]
    def packed(dtype, shape, n):
        # Samples of n bits that use every bit, the largest first, and rows
        # that end inside a byte.
        a = ramp(np.int64, shape, 2654435761) % (1 << n)
        a[0, 0, 0] = (1 << n) - 1
        return a.astype(dtype)

    cases += [
        ("tiff_none_u1", packed(np.uint8, (9, 19, 1), 1), dict(compress="none", nbits=1)),
        ("tiff_none_u3_mm", packed(np.uint8, (7, 11, 2), 3),
         dict(compress="none", nbits=3, endianness="BIG")),
        ("tiff_lzw_u12_rgb", packed(np.uint16, (13, 21, 3), 12),
         dict(compress="lzw", nbits=12, blockysize=3)),
        ("tiff_deflate_u15_tiles", packed(np.uint16, (37, 45, 1), 15),
         dict(compress="deflate", nbits=15, tiled=True, blockxsize=16, blockysize=16)),
        ("tiff_deflate_u17_bigtiff", packed(np.uint32, (6, 9, 2), 17),
         dict(compress="deflate", nbits=17, bigtiff="YES", blockysize=4)),
        ("tiff_deflate_u31_mm_tiles", packed(np.uint32, (20, 17, 1), 31),
         dict(compress="deflate", nbits=31, tiled=True, blockxsize=16, blockysize=16,
              endianness="BIG")),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        for name, a, opts in cases:
            h, w, c = a.shape
            with rasterio.open(name + ".tif", "w", driver="GTiff", width=w, height=h,
                               count=c, dtype=a.dtype, **opts) as dst:
                dst.write(np.moveaxis(a, -1, 0))
            np.save(name + ".npy", a)
            with rasterio.open(name + ".tif") as src:
                back = np.moveaxis(src.read(), 0, -1)
            assert back.dtype == a.dtype and back.tobytes() == a.tobytes(), name


def hdf5_fixtures():
    import h5py

    def noise(n, dtype, shape):
        # Incompressible elements, from xorshift's bytes.
        return xorshift(n * np.dtype(dtype).itemsize).view(dtype).reshape(shape)

    def expected(name, f):
        # Every dataset h5py reads, as nx.io reads it: a datatype nx lacks as
        # its bytes, a trailing axis of its size.
        out = {}

        def visit(path, obj):
            if not isinstance(obj, h5py.Dataset) or path in out:
                return
            a = obj[()]
            native = obj.id.get_type().get_class() == h5py.h5t.COMPLEX
            if a.dtype.kind in "SV" or (a.dtype.kind == "c" and not native):
                a = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape + (a.dtype.itemsize,))
            elif a.dtype.kind == "b":
                a = a.view(np.int8)
            # Without the metadata h5py gives an enumeration's dtype.
            out[path] = np.asarray(a).view(np.dtype(a.dtype.str))

        f.visititems(visit)
        np.savez(name + ".npz", **out)

    def chunks_of(d):
        return [d.id.get_chunk_info(i) for i in range(d.id.get_num_chunks())]

    def chunk_ranges(name, f):
        # Each chunk's offset in elements and its bytes in the file, a line
        # each: path, offsets joined by commas, byte, size.
        lines = []

        def visit(path, obj):
            if isinstance(obj, h5py.Dataset) and obj.chunks:
                for c in chunks_of(obj):
                    lines.append("/%s\t%s\t%d\t%d" % (path, ",".join(map(str, c.chunk_offset)),
                                                     c.byte_offset, c.size))

        f.visititems(visit)
        with open(name + ".chunks", "w") as out:
            out.write("".join(l + "\n" for l in lines))

    def messages(path, addr, offsets=8, lengths=8):
        # The messages of the version-1 object header at [addr], its
        # continuation blocks' included, as (type, data) pairs, read after
        # FMT3 IV.A.1.a and IV.A.2.q.
        b = open(path, "rb").read()
        count, size = struct.unpack_from("<HxxxxI", b, addr + 2)
        blocks, out = [(addr + 16, size)], []
        while blocks and len(out) < count:
            at, n = blocks.pop(0)
            i = at
            while i + 8 <= at + n and len(out) < count:
                kind, length = struct.unpack_from("<HH", b, i)
                data = b[i + 8:i + 8 + length]
                if kind == 0x10:
                    blocks.append((int.from_bytes(data[:offsets], "little"),
                                   int.from_bytes(data[offsets:offsets + lengths], "little")))
                out.append((kind, data))
                i += 8 + length
        return out

    def address(obj):
        return h5py.h5o.get_info(obj.id).addr

    import ctypes
    import glob

    def libhdf5():
        # The libhdf5 h5py's wheel bundles, for the calls h5py does not make.
        libs = os.path.join(os.path.dirname(h5py.__file__), "..", "h5py.libs", "libhdf5-*")
        return ctypes.CDLL(glob.glob(libs)[0])

    def earliest(name, istore=None, sym=None, sizes=None):
        # A file in libhdf5's earliest format, with the B-tree K values and
        # the sizes of addresses and lengths given, which h5py's File does
        # not take: libhdf5's own calls set the K values.
        lib = libhdf5()
        fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
        if istore is not None:
            assert lib.H5Pset_istore_k(ctypes.c_int64(fcpl.id), ctypes.c_uint(istore)) >= 0
        if sym is not None:
            assert lib.H5Pset_sym_k(ctypes.c_int64(fcpl.id), ctypes.c_uint(sym[0]),
                                    ctypes.c_uint(sym[1])) >= 0
        if sizes is not None:
            fcpl.set_sizes(*sizes)
        fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        fapl.set_libver_bounds(h5py.h5f.LIBVER_EARLIEST, h5py.h5f.LIBVER_LATEST)
        return h5py.File(h5py.h5f.create(name.encode(), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl))

    def created(f, name, tid, shape, layout=None):
        # A dataset made through h5py's low-level calls, without the
        # modification time libhdf5 would record, as h5py's own leave out.
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_obj_track_times(False)
        if layout is not None:
            dcpl.set_layout(layout)
        return h5py.h5d.create(f.id, name.encode(), tid, h5py.h5s.create_simple(shape),
                               dcpl=dcpl)

    # Layouts, groups and links.
    with h5py.File("hdf5_layouts.h5", "w") as f:
        # Float32 elements at an offset of a multiple of 8, then three
        # bytes, so that the float64 elements after them lie at an offset
        # that is not.
        f["aligned"] = np.array([0.5, -1.0, 3.0, 1e-40], dtype="<f4")
        assert f["aligned"].id.get_offset() % 8 == 0
        f["u3"] = np.arange(3, dtype="u1")
        f["f64"] = np.array([1 / 3, -0.0, np.inf, np.nan, 5e-324], dtype="<f8")
        assert f["f64"].id.get_offset() % 8 != 0
        f["i16le"] = np.arange(-12, 12, dtype="<i2").reshape(6, 4) * 1000
        f["i16be"] = np.arange(-12, 12, dtype=">i2").reshape(6, 4) * 1000
        f["f32"] = (np.arange(24, dtype="<f4") / 7).reshape(2, 3, 4)
        f["scalar"] = np.float64(2.5)
        f.create_dataset("empty", shape=(0, 3), dtype="<f4")
        f.create_dataset("unwritten", shape=(5,), dtype="<f4", fillvalue=-1.5)
        compact = created(f, "compact", h5py.h5t.STD_I32LE, (4,), h5py.h5d.COMPACT)
        compact.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([7, -8, 9, -10], dtype="<i4"))
        c = f.create_group("chunked")
        c.create_dataset("edges", data=np.arange(70, dtype="<i4").reshape(7, 10), chunks=(3, 4))
        d = c.create_dataset("fill", shape=(9,), dtype="<f4", chunks=(4,), fillvalue=2.5)
        d[0:4] = np.arange(4, dtype="<f4")
        d = c.create_dataset("zeros", shape=(9,), dtype="<i2", chunks=(4,))
        d[4:8] = np.arange(1, 5, dtype="<i2")
        c.create_dataset("rank3", data=np.arange(210, dtype="<u2").reshape(5, 6, 7),
                         chunks=(2, 3, 4))
        # 300 chunks of one element: a B-tree of two levels, since a node
        # holds at most 64.
        c.create_dataset("deep", data=(np.arange(300) % 251).astype("i1"), chunks=(1,))
        g = f.create_group("g/h")
        g["deep"] = np.array([[1, 2], [3, 4]], dtype="i1")
        f["link"] = g["deep"]
        f["soft"] = h5py.SoftLink("/g")
        g["back"] = f["g"]
        many = f.create_group("many")
        for i in range(40):
            many["d%02d" % i] = np.int16(i)
    with h5py.File("hdf5_layouts.h5") as f:
        expected("hdf5_layouts", f)
        chunk_ranges("hdf5_layouts", f)

    # Filters, each chunk with its filter mask.
    with h5py.File("hdf5_filters.h5", "w") as f:
        f.create_dataset("deflate", data=(np.arange(600, dtype="<f4") % 37).reshape(20, 30),
                         chunks=(8, 8), compression="gzip", compression_opts=4)
        f.create_dataset("shuffle_deflate", data=np.arange(50, dtype="<i8") * 1000003,
                         chunks=(16,), shuffle=True, compression="gzip")
        f.create_dataset("lzf", data=(np.arange(400, dtype="<f8") % 11).reshape(40, 10),
                         chunks=(16, 4), compression="lzf")
        raw = np.concatenate([np.zeros(256, "<u4"), noise(256, "<u4", (256,))])
        d = f.create_dataset("lzf_raw", data=raw, chunks=(256,), compression="lzf")
        # libhdf5 stores the chunk LZF does not shrink as it is, and sets its
        # mask's bit 0.
        assert [c.filter_mask for c in chunks_of(d)] == [0, 1]
        f.create_dataset("fletcher", data=np.arange(30, dtype="<i2") - 15, chunks=(8,),
                         fletcher32=True)
        f.create_dataset("all", data=(np.arange(33 * 17, dtype=">u2") * 31).reshape(33, 17),
                         chunks=(8, 8), shuffle=True, compression="lzf", fletcher32=True)
        f.create_dataset("shuffle_deflate_fletcher", data=noise(100, "<f4", (10, 10)),
                         chunks=(4, 5), shuffle=True, compression="gzip", fletcher32=True)
    with h5py.File("hdf5_filters.h5") as f:
        expected("hdf5_filters", f)
        chunk_ranges("hdf5_filters", f)
        fletcher = chunks_of(f["fletcher"])
    # The same file with a byte of the fletcher dataset's chunk 1 flipped.
    data = bytearray(open("hdf5_filters.h5", "rb").read())
    data[fletcher[1].byte_offset + 3] ^= 0xFF
    open("hdf5_fletcher_flipped.h5", "wb").write(bytes(data))
    # The same file with each of that dataset's checksums in the byte order
    # libhdf5 wrote before 1.6.3, each half's bytes swapped, which libhdf5
    # still reads.
    data = bytearray(open("hdf5_filters.h5", "rb").read())
    for c in fletcher:
        end = c.byte_offset + c.size
        stored = data[end - 4:end]
        data[end - 4:end] = bytes([stored[1], stored[0], stored[3], stored[2]])
        assert data[end - 4:end] != stored
    open("hdf5_fletcher_reversed.h5", "wb").write(bytes(data))

    # A dataset of each datatype class nx.io reads, and attributes.
    with h5py.File("hdf5_types.h5", "w") as f:
        for code in ["i1", "u1", "<i2", ">i2", "<u2", "<i4", ">i4", "<u4", "<i8", ">u8"]:
            info = np.iinfo(code)
            name = code.lstrip("<>") + ("_be" if code[0] == ">" else "")
            f[name] = np.array([info.min, info.max, 0, 1], dtype=code)
        for code in ["<f2", ">f4", "<f8"]:
            info = np.finfo(code)
            name = code.lstrip("<>") + ("_be" if code[0] == ">" else "")
            f[name] = np.array([info.min, info.max, -0.0, np.inf, np.nan,
                                info.smallest_subnormal], dtype=code)
        f["bool"] = np.array([True, False, True])
        f.create_dataset("enum", data=np.array([1, -2, 1], dtype="<i2"),
                         dtype=h5py.enum_dtype({"a": 1, "b": -2}, basetype="<i2"))
        f["string"] = np.array([b"ab", b"abcde", b""], dtype="S5")
        f["compound"] = np.array([(1, 2.5), (-3, -0.0)], dtype=[("i", "<i4"), ("x", "<f8")])
        f["opaque"] = np.frombuffer(b"0123456789abcd", dtype="V7")
        # numpy's complex as h5py writes it, a compound type, and libhdf5's
        # complex class.
        f["complex"] = np.array([1 + 2j, -0.5 - 1j], dtype="<c8")
        c = created(f, "complex_class", h5py.h5t.COMPLEX_IEEE_F64BE, (2,))
        c.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([1.5 - 2j, np.inf + 0j], dtype="<c16"),
                mtype=h5py.h5t.COMPLEX_IEEE_F64LE)
        # Array datatypes, written element by element as their base type.
        for name, base, shape, a in [
                ("array", h5py.h5t.IEEE_F32LE, (2, 3), np.arange(24, dtype="<f4") / 4),
                ("array_be", h5py.h5t.STD_I16BE, (2,), np.arange(6, dtype="<i2") - 3)]:
            n = a.size // int(np.prod(shape))
            d = created(f, name, h5py.h5t.array_create(base, shape), (n,))
            le = h5py.h5t.IEEE_F32LE if a.dtype == np.dtype("<f4") else h5py.h5t.STD_I16LE
            d.write(h5py.h5s.ALL, h5py.h5s.ALL, a.reshape((n,) + shape),
                    mtype=h5py.h5t.array_create(le, shape))
        bits = created(f, "bitfield", h5py.h5t.STD_B16BE, (3,))
        bits.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([1, 256, 0xABCD], dtype="<u2"),
                   mtype=h5py.h5t.STD_B16LE)
        a = f["f8"].attrs
        a["scale"] = np.float64(0.25)
        a["bands"] = np.array([3, 1, 4, 1], dtype="<i4")
        a["unit"] = np.bytes_(b"nJy")
        a["name"] = "flux"
        # A name in UTF-8, which libhdf5's earliest format stores in an
        # attribute message of version 1, as any other.
        a["λ"] = np.float32(1.5)
        # An attribute created with its name's encoding set to UTF-8, which
        # libhdf5 stores in an attribute message of version 3.
        lib = libhdf5()
        lib.H5Acreate2.restype = ctypes.c_int64
        lib.H5Pcreate.restype = ctypes.c_int64
        acpl = lib.H5Pcreate(ctypes.c_int64.in_dll(lib, "H5P_CLS_ATTRIBUTE_CREATE_ID_g"))
        assert acpl >= 0
        assert lib.H5Pset_char_encoding(ctypes.c_int64(acpl), h5py.h5t.CSET_UTF8) >= 0
        scalar = h5py.h5s.create(h5py.h5s.SCALAR)
        i32 = h5py.h5t.STD_I32LE
        target = f["i1"]
        aid = lib.H5Acreate2(ctypes.c_int64(target.id.id), "ν".encode(), ctypes.c_int64(i32.id),
                             ctypes.c_int64(scalar.id), ctypes.c_int64(acpl), ctypes.c_int64(0))
        assert aid >= 0
        value = np.array(7, dtype="<i4")
        assert lib.H5Awrite(ctypes.c_int64(aid), ctypes.c_int64(i32.id),
                            value.ctypes.data_as(ctypes.c_void_p)) >= 0
        assert lib.H5Aclose(ctypes.c_int64(aid)) >= 0
        assert lib.H5Pclose(ctypes.c_int64(acpl)) >= 0
        f.flush()
        assert [d[0] for k, d in messages("hdf5_types.h5", address(f["i1"])) if k == 0x0C] == [3]
        # The attributes take the header past its first block, into
        # continuation blocks.
        f.flush()
        assert any(k == 0x10 for k, _ in messages("hdf5_types.h5", address(f["f8"])))
    with h5py.File("hdf5_types.h5") as f:
        assert f["array"].dtype.subdtype == (np.dtype("<f4"), (2, 3))
        expected("hdf5_types", f)

    # Datasets nx.io refuses when read, beside one it reads.
    import shutil
    with h5py.File("hdf5_refused.h5", "w") as f:
        f["ok"] = np.arange(3, dtype="<f4")
        f["vlen_string"] = ["a", "bc"]
        f.create_dataset("vlen_ints", data=np.array([np.arange(2), np.arange(3)], dtype=object),
                         dtype=h5py.vlen_dtype("<i4"))
        f["reference"] = np.array([f["ok"].ref], dtype=h5py.ref_dtype)
        f["type"] = np.dtype("<f4")
        f.create_dataset("committed", data=np.arange(2, dtype="<f4"), dtype=f["type"])
        f.create_dataset("scaleoffset", data=np.arange(16, dtype="<i4"), chunks=(8,),
                         scaleoffset=0)
        f.create_dataset("external", data=np.arange(4, dtype="<i4"),
                         external=[("hdf5_refused.bin", 0, 16)])
        f["null"] = h5py.Empty("<f4")
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_obj_track_times(False)
        dcpl.set_chunk((4,))
        dcpl.set_deflate(1)
        dcpl.set_filter(32000, 1, ())
        two = h5py.h5d.create(f.id, b"two_compressors", h5py.h5t.STD_I32LE,
                              h5py.h5s.create_simple((8,)), dcpl=dcpl)
        two.write(h5py.h5s.ALL, h5py.h5s.ALL, np.arange(8, dtype="<i4"))
    os.remove("hdf5_refused.bin")

    # Links: hard, soft, absolute and relative, dangling and in a cycle,
    # in a group stored as symbol tables and in one that tracks its links'
    # creation order, which libhdf5 stores as link messages in a version-2
    # object header. Its npz names each dataset by the least of its paths,
    # comparing names one by one: /a/r, a relative soft link, is less than
    # /a/x, which is less than /a-; /0 is less than /c/e.
    with h5py.File("hdf5_links.h5", "w") as f:
        f.attrs["title"] = np.bytes_(b"raven")
        a = f.create_group("a")
        a.attrs["n"] = np.int32(3)
        a["x"] = np.array([1, 2], dtype="i1")
        f["a-"] = a["x"]
        f["s"] = h5py.SoftLink("/a/x")
        a["r"] = h5py.SoftLink("x")
        c = f.create_group("c")
        c["e"] = np.array([3.5], dtype="<f8")
        f["0"] = h5py.SoftLink("/c/e")
        f["t"] = h5py.SoftLink("/c")
        f["dangling"] = h5py.SoftLink("/nowhere")
        f["loop1"] = h5py.SoftLink("/loop2")
        f["loop2"] = h5py.SoftLink("/loop1")
        o = f.create_group("ordered", track_order=True)
        o.attrs["k"] = np.int16(-4)
        o["z"] = np.arange(3, dtype="<i2")
        o["y"] = np.arange(2, dtype="<u4")
        o["rel"] = h5py.SoftLink("sub/w")
        o["up"] = h5py.SoftLink("/a/x")
        o.create_group("sub")["w"] = np.float32(0.25)
        np.savez("hdf5_links.npz", **{
            "0": f["c/e"][()], "a/r": f["a/x"][()], "ordered/rel": f["ordered/sub/w"][()],
            "ordered/y": f["ordered/y"][()], "ordered/z": f["ordered/z"][()]})
    assert open("hdf5_links.h5", "rb").read().count(b"OHDR") >= 1

    # Small B-tree K values, so that a group's B-tree and a chunk B-tree
    # each hold more than one level, in superblock version 1, which states
    # the chunk K, with addresses and lengths of 4 bytes.
    with earliest("hdf5_small_k.h5", istore=2, sym=(2, 2), sizes=(4, 4)) as f:
        g = f.create_group("many")
        for i in range(40):
            g["d%02d" % i] = np.int16(i)
        f.create_dataset("chunked", data=np.arange(100, dtype="<i4") * 3, chunks=(2,))
        f["contiguous"] = np.arange(6, dtype="<f8").reshape(2, 3)
        f["contiguous"].attrs["b"] = np.uint16(9)
    b = open("hdf5_small_k.h5", "rb").read()
    assert b[8] == 1 and b[13] == 4 and b[14] == 4
    for kind in (0, 1):
        assert re.search(re.escape(b"TREE") + bytes([kind]) + b"[\x01-\xff]", b), kind
    with h5py.File("hdf5_small_k.h5") as f:
        expected("hdf5_small_k", f)
        chunk_ranges("hdf5_small_k", f)

    # Damaged files: a shuffle filter of elements of no byte, and chunks
    # whose bytes pass any integer.
    with h5py.File("hdf5_damaged.h5", "w") as f:
        f.create_dataset("s", data=np.arange(8, dtype="<i4"), chunks=(4,), shuffle=True)
        f.create_dataset("h", data=np.zeros((2, 2), dtype="V2048"), chunks=(2, 2))
    data = bytearray(open("hdf5_damaged.h5", "rb").read())
    # The shuffle filter's description in a version-1 pipeline: its name,
    # then its one client value, the element size.
    i = data.index(b"shuffle\x00")
    assert data.count(b"shuffle\x00") == 1 and data[i + 8:i + 12] == struct.pack("<I", 4)
    data[i + 8:i + 12] = struct.pack("<I", 0)
    # The chunk dimensions of a version-3 layout message, the element's
    # bytes last.
    layout = re.compile(re.escape(b"\x03\x02\x03") + b".{8}" + re.escape(struct.pack("<3I", 2, 2, 2048)),
                        re.S)
    found = list(layout.finditer(data))
    assert len(found) == 1
    at = found[0].start() + 11
    # Under 2^52 elements, as many as an array may hold, whose 2048 bytes
    # each pass 2^62.
    data[at:at + 8] = struct.pack("<2I", 0xFFFFFFFF, 0x100000)
    open("hdf5_damaged.h5", "wb").write(bytes(data))

    # Files nx.io refuses at load.
    with h5py.File("hdf5_latest.h5", "w", libver="latest") as f:
        f["x"] = np.arange(3)
    # An external link turns its group's symbol table into link messages.
    with h5py.File("hdf5_extlink.h5", "w") as f:
        f.create_group("g")["e"] = h5py.ExternalLink("other.h5", "/x")
    with h5py.File("hdf5_userblock.h5", "w", userblock_size=512) as f:
        f["x"] = np.arange(3)


SUPPORT = Path(__file__).resolve().parent.parent / "support"


def generate():
    safetensors_fixtures()
    numpy_fixtures()
    float_literals()
    unaligned_npz()
    other_names_npz()
    skipped_entries_npz()
    stored_block_npz()
    image_fixtures()
    pytorch_fixtures()
    tiff_fixtures()
    hdf5_fixtures()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    if not parser.parse_args().check:
        os.chdir(SUPPORT)
        generate()
        return
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        generate()
        made = sorted(os.listdir(tmp))
        _, differ, missing = filecmp.cmpfiles(tmp, SUPPORT, made, shallow=False)
    stale = differ + missing + sorted(set(os.listdir(SUPPORT)) - set(made))
    if stale:
        sys.exit("stale: " + " ".join(stale))


if __name__ == "__main__":
    main()
