# /// script
# requires-python = ">=3.11"
# dependencies = ["astropy==7.1.0", "numpy==2.3.2"]
# ///
"""Write the reference FITS files the ymir.fits tests read.

    uv run contrib/ymir/test/fits/gen/fixtures.py [--check]

Each image holds a pattern the tests recompute: pixel k of an image (C
order) is pattern(k) below, in the image's stored type. astropy writes the
files with checksums, so they are also the reference for DATASUM and
CHECKSUM. With --check, the files are written to a temporary directory and
compared with the committed ones.
"""

import filecmp
from fractions import Fraction
import hashlib
import pathlib
import sys
import tempfile

import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.compressed import _codecs

# gzip members carry their time of writing; pin it so the files are a
# function of this script.
_codecs.gzip_compress = lambda data: __import__("gzip").compress(data, mtime=0)

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "golden"


def pattern(n, dtype):
    k = np.arange(n, dtype=np.int64)
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    if info is not None:
        # Spread over the whole range, both ends included.
        span = int(info.max) - int(info.min)
        v = [int(info.min) + (int(x) * 7919 * (span // 97)) % (span + 1) for x in k]
        v[0] = int(info.min)
        v[-1] = int(info.max)
        return np.array(v, dtype=dtype)
    v = (k.astype(np.float64) - n / 2) * 0.375
    return v.astype(dtype)


def images(path):
    shape = (5, 7)
    n = shape[0] * shape[1]
    primary = fits.PrimaryHDU(pattern(n, np.int16).reshape(shape))
    h = primary.header
    h["OBJECT"] = ("NGC 346", "target")
    h["EXPTIME"] = (1288.4, "[s] exposure")
    h["FLAG"] = True
    h["COUNT"] = 42
    h["LONGSTR"] = "x" * 50 + " and a string that runs past one record's sixty-eight bytes"
    h["HIERARCH ESO DET DIT"] = 1.5
    h["COMMENT"] = "a comment"
    h["HISTORY"] = "made by fixtures.py"
    hdus = [primary]
    for name, dtype in [
        ("U8", np.uint8),
        ("I8", np.int8),
        ("U16", np.uint16),
        ("U32", np.uint32),
        ("I32", np.int32),
        ("I64", np.int64),
        ("U64", np.uint64),
        ("F32", np.float32),
        ("F64", np.float64),
    ]:
        hdus.append(fits.ImageHDU(pattern(n, dtype).reshape(shape), name=name))
    # A float32 image with NaN at pixels 3 and 17.
    f = pattern(n, np.float32)
    f[3] = np.nan
    f[17] = np.nan
    hdus.append(fits.ImageHDU(f.reshape(shape), name="NAN"))
    # An int16 image scaled by BSCALE 0.5 and BZERO 10, BLANK -32768 at
    # pixels 0 and 9.
    s = pattern(n, np.int16)
    s[9] = -32768
    sc = fits.ImageHDU(s.reshape(shape), name="SCALED", do_not_scale_image_data=True)
    sc.header["BSCALE"] = 0.5
    sc.header["BZERO"] = 10.0
    sc.header["BLANK"] = -32768
    hdus.append(sc)
    # A 3-axis image.
    hdus.append(fits.ImageHDU(pattern(2 * 3 * 4, np.int32).reshape((2, 3, 4)), name="CUBE"))
    write(path, hdus)
    values(path.with_suffix(".values"), hdus)
    digest(path)


def write(path, hdus):
    """Write with checksums whose comments hold no date, so the bytes are a
    function of this script."""
    hdul = fits.HDUList(hdus)
    for hdu in hdul:
        hdu.add_checksum(when="checksum")
    hdul.writeto(path, overwrite=True)


def r32(x):
    """The exact rational [x] rounded once to float32, ties to even."""
    y = np.float32(float(x))
    cands = [np.nextafter(y, np.float32(-np.inf)), y, np.nextafter(y, np.float32(np.inf))]
    return float(min(cands, key=lambda c: (abs(Fraction(float(c)) - x), int(c.view(np.uint32)) & 1)))


def noise(row):
    """cfitsio's FnNoise5_float on one row of at least 9 pixels without NaN,
    each float operation rounded once to float32 as the C source states
    (astropy's arm64 build fuses some of them), and the minimum of the
    three estimates that fits_quantize_float takes."""
    v = [Fraction(float(x)) for x in row]
    d2, d3, d5 = [], [], []
    for i in range(8, len(v)):
        v1, v2, v3, v4, v5, v6, v7, v8, v9 = v[i - 8 : i + 1]
        if not (v5 == v6 == v7):
            d2.append(abs(r32(v5 - v7)))
        if not (v3 == v4 == v5 == v6 == v7):
            d3.append(abs(r32(r32(2 * v5 - v3) - v7)))
            e = r32(r32(r32(r32(r32(6 * v5) - r32(4 * v3)) - r32(4 * v7)) + v1) + v9)
            d5.append(abs(e))
    # The 2nd order median runs over as many entries as the 3rd order's,
    # the ones past its own count being zero.
    d2 += [0.0] * (len(d3) - len(d2))
    med = lambda a: sorted(a)[(len(a) - 1) // 2]
    n2, n3, n5 = 1.0483579 * med(d2), 0.6052697 * med(d3), 0.1772048 * med(d5)
    sd = n3
    if n2 != 0 and n2 < sd:
        sd = n2
    if n5 != 0 and n5 < sd:
        sd = n5
    return sd


def ones_complement(data):
    """The ones' complement sum of big-endian 32-bit words, zero-padded."""
    data = data + b"\0" * (-len(data) % 4)
    words = np.frombuffer(data, dtype=">u4").astype(np.uint64)
    s = int(words.sum())
    while s >> 32:
        s = (s & 0xFFFFFFFF) + (s >> 32)
    return s


def digest(path):
    """The BLAKE2b-256 digest of the file's headers as stored, records, END
    and padding, in file order."""
    h = hashlib.blake2b(digest_size=32)
    with fits.open(path) as hdul:
        for hdu in hdul:
            info = hdu.fileinfo()
            with open(path, "rb") as f:
                f.seek(info["hdrLoc"])
                h.update(f.read(info["datLoc"] - info["hdrLoc"]))
    path.with_suffix(".digest").write_text("blake2b-256:" + h.hexdigest() + "\n")


def text(x):
    if isinstance(x, (float, np.floating)):
        return float(x).hex()
    return str(int(x))


def values(path, hdus, scaled=True):
    """One line per image: its name, then its pixels in C order as the
    numbers it holds (unsigned offsets applied, scaling not), then for a
    scaled image a second line of physical values."""
    lines = []
    for i, hdu in enumerate(hdus):
        if hdu.data is None:
            continue
        name = hdu.name if i > 0 else "PRIMARY"
        raw = np.asarray(hdu.data).ravel()
        lines.append(" ".join([name] + [text(x) for x in raw]))
        if scaled and "BSCALE" in hdu.header and hdu.header["BSCALE"] != 1:
            phys = 10.0 + 0.5 * raw.astype(np.float64)
            phys[raw == hdu.header["BLANK"]] = np.nan
            lines.append(" ".join([name + "/values"] + [text(x) for x in phys]))
    path.write_text("\n".join(lines) + "\n")


def tiles(path):
    """Tile-compressed images, edge tiles partial: integers Rice- and
    gzip-coded, 64-bit ones included, unsigned offsets, lossless floats,
    and quantized floats under each dither with NaN and zero pixels."""
    shape = (10, 13)
    n = shape[0] * shape[1]
    rng = np.random.default_rng(7)
    noisy = (100 + 5 * rng.standard_normal(n)).astype(np.float32)
    noisy[[4, 50]] = np.nan
    noisy[[7, 90]] = 0.0
    hdus = [fits.PrimaryHDU()]

    def comp(name, data, **kw):
        hdus.append(fits.CompImageHDU(data.reshape(shape), name=name, tile_shape=(3, 4), **kw))

    comp("RICE_I16", pattern(n, np.int16), compression_type="RICE_1")
    comp("RICE_U16", pattern(n, np.uint16), compression_type="RICE_1")
    comp("RICE_U8", pattern(n, np.uint8), compression_type="RICE_1")
    comp("RICE_I32", pattern(n, np.int32), compression_type="RICE_1")
    comp("GZIP1_I32", pattern(n, np.int32), compression_type="GZIP_1")
    comp("GZIP2_I32", pattern(n, np.int32), compression_type="GZIP_2")
    comp("GZIP2_U32", pattern(n, np.uint32), compression_type="GZIP_2")
    comp("GZIP2_I64", pattern(n, np.int64), compression_type="GZIP_2")
    comp("NOCOMP_I16", pattern(n, np.int16), compression_type="NOCOMPRESS")
    comp("GZIP2_F32", pattern(n, np.float32), compression_type="GZIP_2", quantize_level=0.0)
    comp("GZIP1_F64", pattern(n, np.float64), compression_type="GZIP_1", quantize_level=0.0)
    comp("Q_DITHER1", noisy, compression_type="RICE_1", quantize_level=4.0,
         quantize_method=1, dither_seed=17)
    comp("Q_DITHER2", noisy, compression_type="RICE_1", quantize_level=16.0,
         quantize_method=2, dither_seed=9999)
    comp("Q_NODITHER", noisy, compression_type="RICE_1", quantize_level=4.0,
         quantize_method=-1)
    comp("Q_GZIP", noisy, compression_type="GZIP_2", quantize_level=4.0,
         quantize_method=1, dither_seed=1)
    comp("Q_F64", noisy.astype(np.float64), compression_type="RICE_1", quantize_level=4.0,
         quantize_method=1, dither_seed=5)
    hdus.append(fits.CompImageHDU(pattern(2 * 5 * 7, np.int16).reshape((2, 5, 7)), name="CUBE",
                                  compression_type="RICE_1", tile_shape=(1, 2, 3)))
    # Pixels and astropy's quantization of them, by rows, with the dither
    # seed ymir derives from them: 1 plus their ones' complement sum
    # modulo 10000.
    src = (100 + 5 * rng.standard_normal(n)).astype(np.float32)
    src[[7, 90]] = 0.0
    hdus.append(fits.ImageHDU(src.reshape(shape), name="QSRC"))
    hdus.append(fits.CompImageHDU(src.reshape(shape), name="QREF", compression_type="RICE_1",
                                  tile_shape=(1, 13), quantize_level=16.0, quantize_method=2,
                                  dither_seed=1 + ones_complement(src.astype(">f4").tobytes()) % 10000))
    write(path, hdus)
    with fits.open(path) as hdul:
        values(path.with_suffix(".values"), hdul, scaled=False)
    with fits.open(path, disable_image_compression=True) as hdul:
        zscale = [noise(row) / 16.0 for row in src.reshape(shape)]
        theirs = hdul["QREF"].data["ZSCALE"]
        same = [float(a) == b for a, b in zip(theirs, zscale)]
        with open(path.with_suffix(".values"), "a") as f:
            f.write(" ".join(["QREF/zscale"] + [text(x) for x in zscale]) + "\n")
            f.write(" ".join(["QREF/same"] + ["1" if x else "0" for x in same]) + "\n")


def tables(path):
    """A binary table of every TFORM code, offsets, TNULL, scaling, cell
    shapes, text grids and heap arrays, and an ASCII table. The values file
    holds, per column, its name and its elements in C order: integers in
    decimal, floats in hex, complex as re,im, logicals as T, F or 0, bits as
    0 or 1 and text in hex."""
    n = 6
    rng = np.random.default_rng(11)
    mag = rng.normal(15, 2, n).astype(np.float32)
    mag[2] = np.nan
    q = np.array([1, -999, 3, 4, -999, 6], dtype=np.int16)
    bits = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]] * n, dtype=bool)
    bits[3] = ~bits[3]
    strings = np.array(["alpha", "", "  lead", "x" * 16, "bb", "z"])
    grid = np.array([["ab", "cde", ""], ["f", "", "gh"]] * 3)
    vla = np.array([np.arange(k, dtype=np.int32) * 7 for k in [0, 3, 1, 5, 2, 0]], dtype=object)
    vld = np.array([np.linspace(0, 1, k) for k in [2, 0, 4, 1, 1, 3]], dtype=object)
    vlt = np.array(["one", "", "three", "four four", "5", "six"], dtype=object)
    cols = [
        fits.Column("source_id", "K", array=np.arange(n, dtype=np.int64) * 10**15 + 7),
        fits.Column("ra", "D", unit="deg", array=np.linspace(0, 359.5, n)),
        fits.Column("mag", "E", unit="mag", array=mag),
        fits.Column("qual", "I", null=-999, array=q),
        fits.Column("u16", "I", bzero=32768, array=np.array([0, 1, 65535, 32768, 7, 40000], dtype=np.uint16)),
        fits.Column("i8", "B", bzero=-128, array=np.array([-128, 0, 127, 5, -5, 1], dtype=np.int8)),
        fits.Column("u32", "J", bzero=2**31, array=np.array([0, 1, 2**32 - 1, 2**31, 3, 4], dtype=np.uint32)),
        fits.Column("u8", "B", array=np.array([0, 1, 255, 128, 3, 4], dtype=np.uint8)),
        fits.Column("flag", "L", array=np.array([True, False, True, True, False, False])),
        fits.Column("bits", "11X", array=bits),
        fits.Column("name", "16A", array=strings),
        fits.Column("cplx", "C", array=(np.arange(n) + 1j * np.arange(n)[::-1]).astype(np.complex64)),
        fits.Column("dcplx", "M", array=(np.arange(n) * 0.5 - 1j).astype(np.complex128)),
        fits.Column("vec", "3E", array=np.arange(3 * n, dtype=np.float32).reshape(n, 3)),
        fits.Column("mat", "6J", dim="(3,2)", array=np.arange(6 * n, dtype=np.int32).reshape(n, 2, 3)),
        fits.Column("grid", "12A", dim="(4,3)", array=np.array([["ab", "cde", ""], ["f", "", "gh"], ["ij", "k", "l"]] * 2)),
        fits.Column("scaled", "I", array=np.array([0, 1, -2, -200, 32, 3], dtype=np.int16)),
        fits.Column("vla", "PJ()", array=vla),
        fits.Column("vld", "QD()", array=vld),
        fits.Column("vlt", "PA()", array=vlt),
    ]
    cat = fits.BinTableHDU.from_columns(cols, name="CAT")
    asc = fits.TableHDU.from_columns(
        [
            fits.Column("id", "I5", array=np.array([1, -22, 333, 0, 99999, 7]), null="-1"),
            fits.Column("x", "F8.3", array=np.array([1.5, -2.25, 0.0, 1000.125, 3.0, -0.5])),
            fits.Column("y", "E12.4", array=np.array([1.5e10, -2.5e-3, 0.0, 6.0221e23, 1.0, 2.0])),
            fits.Column("z", "D20.12", array=np.array([np.pi, -np.e, 0.0, 1e-300, 1.0, 2.0])),
            fits.Column("s", "A10", array=np.array(["one", "two", "", "four", "five", "six"])),
        ],
        name="ASC",
    )
    fits.HDUList([fits.PrimaryHDU(), cat, asc]).writeto(path, overwrite=True)
    # Values before scaling: TSCAL and TZERO enter the header afterwards, so
    # the stored integers stay those given.
    lines = []
    with fits.open(path, uint=True) as hdul:
        for hdu in hdul[1:]:
            for col in hdu.columns:
                v = hdu.data.field(col.name)
                if col.name == "i8":
                    v = v.astype(np.int64)
                lines.append(" ".join([hdu.name + "." + col.name] + [cell_text(x) for x in flat(v)]))
    with fits.open(path, mode="update") as hdul:
        h = hdul["CAT"].header
        h.insert("TFORM17", ("TSCAL17", 0.5), after=True)
        h.insert("TSCAL17", ("TZERO17", 100.0), after=True)
    # Checksums without a date: summed over the bytes as written.
    data = bytearray(path.read_bytes())
    with fits.open(path) as hdul:
        spans = [(h.fileinfo()["hdrLoc"], h.fileinfo()["datLoc"], h.fileinfo()["datSpan"]) for h in hdul]
    for hdr, dat, span in spans:
        header = fits.Header.fromstring(bytes(data[hdr:dat]).decode("ascii"))
        datasum = ones_complement(bytes(data[dat : dat + span]))
        header["DATASUM"] = (str(datasum), "checksum")
        header["CHECKSUM"] = ("0" * 16, "checksum")
        text = header.tostring().encode("ascii")
        assert len(text) == dat - hdr, "the checksum cards change the header's size"
        total = ones_complement(text) + datasum
        total = (total & 0xFFFFFFFF) + (total >> 32)
        header["CHECKSUM"] = (checksum_text(0xFFFFFFFF - total), "checksum")
        data[hdr:dat] = header.tostring().encode("ascii")
    path.write_bytes(bytes(data))
    path.with_suffix(".values").write_text("\n".join(lines) + "\n")


def checksum_text(x):
    """FITS 4.0 Appendix J's encoding of the 32-bit [x]."""
    exclude = {0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60}
    asc = [0] * 16
    for i in range(4):
        byte = (x >> (24 - 8 * i)) & 0xFF
        q, r = byte // 4 + 0x30, byte % 4
        ch = [q + r, q, q, q]
        check = True
        while check:
            check = False
            for e in sorted(exclude):
                for j in (0, 2):
                    if ch[j] == e or ch[j + 1] == e:
                        ch[j] += 1
                        ch[j + 1] -= 1
                        check = True
        for j in range(4):
            asc[4 * j + i] = ch[j]
    return "".join(chr(asc[(i + 15) % 16]) for i in range(16))


def flat(v):
    if isinstance(v, np.ndarray) and v.dtype == object:
        out = []
        for x in v:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, np.ndarray) and x.dtype.kind in "US":
                plain = x.view(np.ndarray)
                out.append(plain.tobytes().decode("ascii") if x.dtype.kind == "S" else "".join(plain.tolist()))
            else:
                out.extend(flat(np.asarray(x)))
        return out
    if isinstance(v, np.ndarray):
        return list(v.ravel()) if v.dtype.kind != "U" and v.dtype.kind != "S" else [x for x in v.ravel()]
    return [v]


def cell_text(x):
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode("ascii")
    if isinstance(x, (str, np.str_)):
        return "s" + x.encode("ascii").hex()
    if isinstance(x, (bool, np.bool_)):
        return "T" if x else "F"
    if isinstance(x, (complex, np.complexfloating)):
        return float(x.real).hex() + "," + float(x.imag).hex()
    if isinstance(x, (float, np.floating)):
        return float(x).hex()
    return str(int(x))


def write_all(directory):
    images(directory / "images.fits")
    tiles(directory / "tiles.fits")
    tables(directory / "tables.fits")


def main():
    if "--check" in sys.argv:
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            write_all(d)
            bad = [p.name for p in d.iterdir() if not filecmp.cmp(p, GOLDEN / p.name, shallow=False)]
            if bad:
                sys.exit("differ: " + ", ".join(bad))
    else:
        write_all(GOLDEN)


main()
