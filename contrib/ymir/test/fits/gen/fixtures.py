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
import hashlib
import pathlib
import sys
import tempfile

import numpy as np
from astropy.io import fits

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


def values(path, hdus):
    """One line per image: its name, then its pixels in C order as the
    numbers it holds (unsigned offsets applied, scaling not), then for a
    scaled image a second line of physical values."""
    lines = []
    for i, hdu in enumerate(hdus):
        name = hdu.name if i > 0 else "PRIMARY"
        raw = np.asarray(hdu.data).ravel()
        lines.append(" ".join([name] + [text(x) for x in raw]))
        if "BSCALE" in hdu.header and hdu.header["BSCALE"] != 1:
            phys = 10.0 + 0.5 * raw.astype(np.float64)
            phys[raw == hdu.header["BLANK"]] = np.nan
            lines.append(" ".join([name + "/values"] + [text(x) for x in phys]))
    path.write_text("\n".join(lines) + "\n")


def write_all(directory):
    images(directory / "images.fits")


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
