# /// script
# requires-python = ">=3.11"
# dependencies = ["astropy==7.1.0", "numpy==2.3.2"]
# ///
"""Check that astropy reads what ymir writes.

    dune build ./contrib/ymir/test/fits/gen/sample.exe
    ./_build/default/contrib/ymir/test/fits/gen/sample.exe out.fits
    uv run contrib/ymir/test/fits/gen/check_written.py out.fits

astropy opens the file verifying every CHECKSUM and DATASUM, each tiled
image must equal its uncompressed copy SRC_<name>, the quantized one must
lie within its quantization step, and the table must hold sample.ml's
columns. cfitsio checks it too: fitsverify must find no error, and
imcopy's decompression of each tiled image must equal astropy's, except
I64, whose 64-bit tiles cfitsio does not decompress. Both tools come with
cfitsio (brew install cfitsio).
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from astropy.io import fits

with warnings.catch_warnings():
    warnings.simplefilter("error")
    hdul = fits.open(sys.argv[1], checksum=True, uint=True)
    hdul.verify("exception")

for name in ["I16", "U16", "U8", "I8", "I32", "U32", "I64", "F32", "F64"]:
    a, b = hdul[name].data, hdul["SRC_" + name].data
    assert a.dtype.newbyteorder("=") == b.dtype.newbyteorder("="), (name, a.dtype, b.dtype)
    assert np.array_equal(a, b), name

q, src = hdul["Q"].data, hdul["SRC_Q"].data
err = np.abs(q - src).max()
assert err < 0.5, err

t = hdul["TAB"].data
assert list(t["id"]) == [1, 2, 3, 4]
assert list(t["ra"]) == [0.0, 90.5, 180.0, 359.75]
assert hdul["TAB"].columns["ra"].unit == "deg"
assert list(t["u16"]) == [0, 1, 65535, 32768]
assert list(t["flag"]) == [True, False, True, False]
assert t["bits"].shape == (4, 11)
assert t["vec"].shape == (4, 2, 3)
assert [s.rstrip() for s in t["name"]] == ["alpha", "", "  lead", "z"]
assert [list(x) for x in t["vla"]] == [[], [0, 1, 2], [3], [4, 5]]
q = t["q"]
assert q[0] == 1 and q[2] == 3 and q[1] == hdul["TAB"].header["TNULL3"]

tools = {t: shutil.which(t) for t in ("fitsverify", "imcopy")}
if None in tools.values():
    sys.exit("fitsverify and imcopy come with cfitsio: brew install cfitsio")

subprocess.run([tools["fitsverify"], "-q", "-e", sys.argv[1]], check=True)

with tempfile.TemporaryDirectory() as d:
    for name in ["I16", "U16", "U8", "I8", "I32", "U32", "F32", "F64", "Q"]:
        out = pathlib.Path(d) / f"{name}.fits"
        subprocess.run([tools["imcopy"], f"{sys.argv[1]}[{hdul.index_of(name)}]", str(out)], check=True, stdout=subprocess.DEVNULL)
        with fits.open(out, uint=True) as copy:
            a, b = copy[0].data, hdul[name].data
            assert a.dtype.newbyteorder("=") == b.dtype.newbyteorder("="), (name, a.dtype, b.dtype)
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), name

print("astropy and cfitsio read the sample:", len(hdul), "HDUs")
