#!/usr/bin/env python3
"""Generate nx.image's test fixtures: images other encoders write.

The PNG files come from the minimal writer below, which chooses the row
filters, sample depths and interlacing that an encoder picks on its own, and
Pillow writes the JPEG files. Each image has a sibling .npy of the pixels
Pillow decodes from it.

Usage:
    cd dev/nx2/test/image/support
    uv run --with numpy --with pillow python ../gen/generate.py
"""

import struct
import zlib

import numpy as np
from PIL import Image


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
    np.save(path.rsplit(".", 1)[0] + ".npy", np.asarray(Image.open(path).convert(mode)))


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
    indices = [[((x + 2 * y) % 4,) for x in range(5)] for y in range(5)]
    write_png("png_palette_adam7.png", indices, 2, 3, filters=(0, 1, 2, 3, 4),
              palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
              trns=[255, 128, 255, 0], interlace=True)
    reference("png_palette_adam7.png", "RGB")

    def gradient(h, w, mode):
        a = np.fromfunction(lambda y, x, c: 10 + 12 * y + 8 * x + 30 * c,
                            (h, w, len(mode)))
        return Image.fromarray(a.astype(np.uint8).squeeze(), mode)

    gradient(8, 8, "RGB").save("jpeg_baseline.jpg", quality=90)
    gradient(7, 9, "RGB").save("jpeg_progressive.jpg", quality=90, progressive=True)
    gradient(6, 5, "L").save("jpeg_gray.jpg", quality=90)
    gradient(5, 6, "CMYK").save("jpeg_cmyk.jpg", quality=90)
    gradient(9, 11, "RGB").save("jpeg_restart.jpg", quality=90, restart_marker_blocks=1)
    for name in ["baseline", "progressive", "gray", "cmyk", "restart"]:
        reference("jpeg_%s.jpg" % name, "RGB")


if __name__ == "__main__":
    image_fixtures()
