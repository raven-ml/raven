# /// script
# requires-python = ">=3.13"
# dependencies = ["astropy==7.1.0", "numpy==2.3.2"]
# ///
"""Cut public archive files into the small files ymir.fits' tests read.

    uv run contrib/ymir/test/fits/gen/survey.py [--check]

Each source is fetched by HTTP range requests: its headers, then the data
spans its cut keeps, whose SHA-256 must be the recorded one, so a file the
archive reprocessed fails here. A cut keeps every header record as the
archive wrote it except the ones it must change (sizes, the reference pixel,
heap descriptors, checksums), which astropy formats, and keeps the data
bytes as stored. The generator writes survey/<name>.fits, survey/README.md
with each file's provenance and changes, and survey/values with astropy's
reading of each image and numeric column. With --check, everything is
written to a temporary directory and compared with the committed files.
"""

import filecmp
import hashlib
import math
import pathlib
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

SURVEY = pathlib.Path(__file__).resolve().parent.parent / "survey"
BLOCK = 2880
MAST = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:"


# Fetching

class Remote:
    """A file read by range requests, hashing every byte read in order."""

    def __init__(self, url):
        self.url = url
        self.sha = hashlib.sha256()
        self.size = None

    def read(self, a, n):
        n = min(n, self.size - a) if self.size is not None else n
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={a}-{a + n - 1}", "User-Agent": "ymir-fits-survey"}
        )
        with urllib.request.urlopen(req, timeout=300) as f:
            if f.status != 206:
                sys.exit(f"{self.url}: no range support (HTTP {f.status})")
            self.size = int(f.headers["Content-Range"].rsplit("/", 1)[1])
            b = f.read()
        if len(b) != n:
            sys.exit(f"{self.url}: asked {n} bytes at {a}, got {len(b)}")
        self.sha.update(b)
        return b


# Headers

def cards_of(raw):
    """The 80-byte records of a header up to END, excluded."""
    out = []
    for i in range(0, len(raw), 80):
        c = raw[i : i + 80]
        if c.rstrip() == b"END":
            return out
        out.append(c)
    raise ValueError("no END")


@dataclass
class Hdu:
    index: int
    start: int
    raw: bytes  # header records, END and padding, as stored
    size: int  # data unit bytes, padding excluded

    @property
    def header(self):
        return fits.Header.fromstring(self.raw.decode("latin-1"))

    @property
    def data_start(self):
        return self.start + len(self.raw)

    @property
    def span(self):
        return len(self.raw) + -(-self.size // BLOCK) * BLOCK


def data_size(h):
    naxis = h.get("NAXIS", 0)
    if naxis == 0:
        return 0
    groups = h.get("GROUPS", False)
    n = math.prod(h[f"NAXIS{i}"] for i in range(2 if groups else 1, naxis + 1))
    return abs(h["BITPIX"]) // 8 * h.get("GCOUNT", 1) * (h.get("PCOUNT", 0) + n)


def walk(remote, count):
    """The first [count] HDUs of [remote]."""
    hdus, pos = [], 0
    for i in range(count):
        raw = b""
        while True:
            raw += remote.read(pos + len(raw), BLOCK)
            try:
                cards_of(raw)
                break
            except ValueError:
                pass
        hdu = Hdu(i, pos, raw, 0)
        hdu.size = data_size(hdu.header)
        hdus.append(hdu)
        pos += hdu.span
    return hdus


def changed(h, values):
    """The entries of [values] that [h] does not already hold."""
    return {k: v for k, v in values.items() if h.get(k) != v}


def edited(raw, values):
    """[raw] with the cards of [values] rewritten in place by astropy, their
    comments kept."""
    keys = dict(values)
    out = []
    for c in cards_of(raw):
        k = c[:8].decode("ascii").rstrip()
        if k in keys:
            old = fits.Card.fromstring(c.decode("ascii"))
            c = fits.Card(k, keys.pop(k), old.comment).image.encode("ascii")
            assert len(c) == 80, k
        out.append(c)
    assert not keys, f"absent cards: {sorted(keys)}"
    text = b"".join(out) + b"END".ljust(80)
    return text + b" " * (len(raw) - len(text))


# Checksums (FITS 4.0 §4.4.2.7 and Appendix J)

def ones_complement(data):
    data = data + b"\0" * (-len(data) % 4)
    s = int(np.frombuffer(data, dtype=">u4").astype(np.uint64).sum())
    while s >> 32:
        s = (s & 0xFFFFFFFF) + (s >> 32)
    return s


def checksum_text(x):
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


def resummed(raw, data):
    """[raw] with its DATASUM and CHECKSUM, where present, recomputed for
    [data]."""
    keys = {c[:8].decode("ascii").rstrip() for c in cards_of(raw)}
    if "CHECKSUM" not in keys and "DATASUM" not in keys:
        return raw, []
    pad = data + b"\0" * (-len(data) % BLOCK)
    datasum = ones_complement(pad)
    sums = {"DATASUM": str(datasum)} if "DATASUM" in keys else {}
    if "CHECKSUM" not in keys:
        return edited(raw, sums), ["DATASUM"]
    zero = edited(raw, sums | {"CHECKSUM": "0" * 16})
    total = ones_complement(zero) + datasum
    total = (total & 0xFFFFFFFF) + (total >> 32)
    return edited(raw, sums | {"CHECKSUM": checksum_text(0xFFFFFFFF - total)}), sorted(sums) + ["CHECKSUM"]


# Cuts. Each returns the HDU's new header and data unit and what changed.

@dataclass
class Cut:
    origin: int
    raw: bytes
    data: bytes
    changes: list = field(default_factory=list)


def whole(remote, hdu):
    return Cut(hdu.index, hdu.raw, remote.read(hdu.data_start, hdu.size) if hdu.size else b"")


def window(remote, hdu, box):
    """The image's pixels in [box], a (start, stop) per axis in C order, the
    leading axes whole; the reference pixel moved with it."""
    h = hdu.header
    shape = [h[f"NAXIS{i}"] for i in range(h["NAXIS"], 0, -1)]
    bpp = abs(h["BITPIX"]) // 8
    (r0, r1), (c0, c1) = box[-2], box[-1]
    assert all(b == (0, n) for b, n in zip(box[:-2], shape[:-2]))
    plane = shape[-2] * shape[-1] * bpp
    row = shape[-1] * bpp
    out = []
    for p in range(math.prod(shape[:-2])):
        band = remote.read(hdu.data_start + p * plane + r0 * row, (r1 - r0) * row)
        rows = np.frombuffer(band, dtype=np.uint8).reshape(r1 - r0, row)
        out.append(rows[:, c0 * bpp : c1 * bpp].tobytes())
    data = b"".join(out)
    values = {"NAXIS1": c1 - c0, "NAXIS2": r1 - r0}
    for ax, off in ((1, c0), (2, r0)):
        if off and f"CRPIX{ax}" in h:
            values[f"CRPIX{ax}"] = h[f"CRPIX{ax}"] - off
    values = changed(h, values)
    raw, sums = resummed(edited(hdu.raw, values), data)
    where = f"rows {r0}-{r1 - 1} and columns {c0}-{c1 - 1} of each plane (0-based)"
    return Cut(hdu.index, raw, data, [where, "rewritten: " + " ".join(sorted(values) + sums)])


def groups(remote, hdu, n):
    """The first [n] random groups."""
    h = hdu.header
    size = hdu.size // h["GCOUNT"]
    data = remote.read(hdu.data_start, n * size)
    raw, sums = resummed(edited(hdu.raw, {"GCOUNT": n}), data)
    return Cut(hdu.index, raw, data, [f"groups 0-{n - 1} of {h['GCOUNT']}", "rewritten: " + " ".join(["GCOUNT"] + sums)])


ELEMENT = {"L": 1, "B": 1, "A": 1, "I": 2, "J": 4, "K": 8, "E": 4, "D": 8, "C": 8, "M": 16}


def tform(f):
    """(repeat, code, heap element) of a TFORM."""
    i = 0
    while i < len(f) and f[i].isdigit():
        i += 1
    r = int(f[:i]) if i else 1
    code = f[i]
    return r, code, f[i + 1] if code in "PQ" else None


def rows(remote, hdu, r0, r1, extra=None):
    """Rows [r0, r1) of a binary table, the heap spans their descriptors name
    moved to the start of the heap."""
    h = hdu.header
    width, nrows = h["NAXIS1"], h["NAXIS2"]
    table = bytearray(remote.read(hdu.data_start + r0 * width, (r1 - r0) * width))
    # Descriptor columns: byte offset in the row, descriptor width, element.
    desc, off = [], 0
    for i in range(1, h["TFIELDS"] + 1):
        r, code, elt = tform(h[f"TFORM{i}"].strip())
        if code in "PQ":
            desc.append((off, 4 if code == "P" else 8, elt))
            off += 8 if code == "P" else 16
        else:
            off += (r + 7) // 8 if code == "X" else r * ELEMENT[code]
    assert off == width
    dt = lambda w: ">i4" if w == 4 else ">i8"
    spans = []
    for k in range(r1 - r0):
        for o, w, elt in desc:
            n, p = np.frombuffer(table, dtype=dt(w), count=2, offset=k * width + o)
            size = (int(n) + 7) // 8 if elt == "X" else int(n) * ELEMENT[elt]
            if size:
                spans.append((int(p), int(p) + size))
    lo = min((a for a, _ in spans), default=0)
    hi = max((b for _, b in spans), default=0)
    theap = h.get("THEAP", width * nrows)
    heap = remote.read(hdu.data_start + theap + lo, hi - lo) if hi > lo else b""
    if lo:
        for k in range(r1 - r0):
            for o, w, _ in desc:
                n, p = np.frombuffer(table, dtype=dt(w), count=2, offset=k * width + o)
                if n:
                    table[k * width + o + w : k * width + o + 2 * w] = np.array([p - lo], dtype=dt(w)).tobytes()
    data = bytes(table) + heap
    values = {"NAXIS2": r1 - r0, "PCOUNT": len(heap)} | (extra or {})
    if "THEAP" in h:
        values["THEAP"] = (r1 - r0) * width
    values = changed(h, values)
    raw, sums = resummed(edited(hdu.raw, values), data)
    changes = [f"rows {r0}-{r1 - 1} of {nrows} (0-based)"]
    if lo:
        changes.append(f"heap bytes {lo}-{hi - 1} kept, descriptors moved down by {lo}")
    elif desc:
        changes.append(f"heap bytes 0-{hi - 1} kept")
    return Cut(hdu.index, raw, data, changes + ["rewritten: " + " ".join(sorted(values) + sums)])


def tiles(remote, hdu, t0, t1, axes):
    """Tiles [t0, t1) of a tile-compressed image, which must make an image of
    [axes] (ZNAXISn order); a dither seed follows its tiles."""
    h = hdu.header
    extra = {f"ZNAXIS{i + 1}": n for i, n in enumerate(axes)}
    if t0 and h.get("ZQUANTIZ", "NO_DITHER").startswith("SUBTRACTIVE_DITHER"):
        extra["ZDITHER0"] = h["ZDITHER0"] + t0
    for ax in (1, 2):
        if f"CRPIX{ax}" in h and t0:
            tile = [h[f"ZTILE{i}"] for i in (1, 2)]
            per_row = -(-h["ZNAXIS1"] // tile[0])
            off = (t0 % per_row) * tile[0] if ax == 1 else (t0 // per_row) * tile[1]
            if off:
                extra[f"CRPIX{ax}"] = h[f"CRPIX{ax}"] - off
    return rows(remote, hdu, t0, t1, extra)


# Sources

@dataclass
class Source:
    name: str
    url: str
    archive: str
    terms: str
    why: str
    sha256: str
    hdus: int
    cut: object  # remote, hdus -> list of Cut


SOURCES = [
    Source(
        "jwst-nircam-i2d",
        MAST + "JWST/product/jw02736-o001_t001_nircam_clear-f090w_i2d.fits",
        "MAST (JWST ERO 2736, SMACS 0723, NIRCam F090W mosaic)",
        "Public JWST data from MAST; NASA/ESA/CSA, acknowledgement requested.",
        "the JWST pipeline's headers, a 3-axis int32 context image, and a "
        "265-column header table written through astropy by stdatamodels",
        "48aa8ef717b677574aaf7a9c307b72574b97b6eea5c6c36b4445602a45d15c9b",
        9,
        lambda r, h: [
            whole(r, h[0]),
            window(r, h[1], [(2360, 2392), (2500, 2564)]),
            window(r, h[2], [(2360, 2392), (2500, 2564)]),
            window(r, h[3], [(0, 3), (2360, 2392), (2500, 2564)]),
            rows(r, h[8], 0, 4),
        ],
    ),
    Source(
        "hst-acs-drz",
        MAST + "HST/product/j8pu0y010_drz.fits",
        "MAST (HST ACS/WFC drizzled association j8pu0y010)",
        "Public HST data from MAST; NASA/ESA, acknowledgement requested.",
        "an 829-record primary header from CALACS and drizzlepac, the "
        "drizzle weight and context images, and a 293-column header table",
        "1da722db791a26d3f8f5506fd202a083f9a921cb53df75cfac334b2687c04ca0",
        5,
        lambda r, h: [
            whole(r, h[0]),
            window(r, h[1], [(2200, 2232), (2100, 2164)]),
            window(r, h[2], [(2200, 2232), (2100, 2164)]),
            window(r, h[3], [(2200, 2232), (2100, 2164)]),
            whole(r, h[4]),
        ],
    ),
    Source(
        "hst-wfpc2-c0f",
        MAST + "HST/product/u2ou0101t_c0f.fits",
        "MAST (HST WFPC2 calibrated exposure u2ou0101t, waivered FITS)",
        "Public HST data from MAST; NASA/ESA, acknowledgement requested.",
        "a 3-axis float primary with BSCALE and BZERO and an ASCII table "
        "of group parameters, as STSDAS wrote them",
        "29da403e340d1eab99d8bb954c86eb92d07eca1dae8e4c609d66311552137f68",
        2,
        lambda r, h: [
            window(r, h[0], [(0, 4), (380, 412), (380, 444)]),
            whole(r, h[1]),
        ],
    ),
    Source(
        "sdss-spec-lite",
        "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/0266/spec-0266-51602-0001.fits",
        "SDSS DR17 Science Archive Server (BOSS spectrum, plate 266, fibre 1)",
        "Public SDSS data; acknowledgement requested (sdss.org).",
        "binary tables written by IDL's mwrfits: a 126-column one-row "
        "table of every scalar type and fixed strings",
        "c612b8d3609cfa830c444da60575f7093034682b71dcf8be89006f9f94e7f921",
        4,
        lambda r, h: [whole(r, x) for x in h],
    ),
    Source(
        "gaia-dr1-source",
        "https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/fits/GaiaSource_000-000-000.fits",
        "ESA Gaia Archive (Gaia DR1 gaia_source, first file)",
        "ESA/Gaia/DPAC, CC BY-SA 3.0 IGO; credit ESA/Gaia/DPAC.",
        "STIL's fits-plus layout: a primary byte array holding a VOTable, "
        "then a 57-column table with NaN-filled floats",
        "a4209e82c1468408ef7e5e74fc533d5e3fb5ee310e8df8d6567f2cc0341c4618",
        2,
        lambda r, h: [whole(r, h[0]), rows(r, h[1], 0, 48)],
    ),
    Source(
        "ps1-stack-rice",
        "https://ps1images.stsci.edu/rings.v3.skycell/1784/059/rings.v3.skycell.1784.059.stk.g.unconv.fits",
        "STScI Pan-STARRS1 image archive (DR2 stack, skycell 1784.059, g)",
        "Public Pan-STARRS1 data; acknowledgement requested (panstarrs.stsci.edu).",
        "cfitsio Rice tiles of int16 scaled by BSCALE and BZERO, one row "
        "per tile, with HIERARCH records",
        "b13559677a858d816db1388dcb3ac8a7ad16c12da8e3c96fbc0a961e27dd4dfb",
        2,
        lambda r, h: [whole(r, h[0]), tiles(r, h[1], 3100, 3108, [6279, 8])],
    ),
    Source(
        "ps1-mask-gzip",
        "https://ps1images.stsci.edu/rings.v3.skycell/1784/059/rings.v3.skycell.1784.059.stk.g.unconv.mask.fits",
        "STScI Pan-STARRS1 image archive (DR2 stack mask, skycell 1784.059, g)",
        "Public Pan-STARRS1 data; acknowledgement requested (panstarrs.stsci.edu).",
        "cfitsio GZIP_1 tiles of uint16, one row per tile",
        "6e6d7f967b2e84749eaa2a0f02af354322c44f5de2e33bb987c1f8afd14b4a28",
        2,
        lambda r, h: [whole(r, h[0]), tiles(r, h[1], 3100, 3108, [6279, 8])],
    ),
    Source(
        "legacy-dr10-image",
        "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd/000/0001m002/legacysurvey-0001m002-image-g.fits.fz",
        "NERSC Legacy Surveys DR10 (south coadd, brick 0001m002, g)",
        "Public Legacy Surveys data; acknowledgement requested (legacysurvey.org).",
        "fpack's RICE_ONE float tiles of 100x100, quantized with "
        "SUBTRACTIVE_DITHER_2, with ZSCALE and ZZERO columns",
        "c2069e7fc09524300e0a75e5901844a015435538509e484411deccc599499f19",
        2,
        lambda r, h: [whole(r, h[0]), tiles(r, h[1], 18 * 36 + 17, 18 * 36 + 20, [300, 100])],
    ),
    Source(
        "legacy-dr10-maskbits",
        "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd/000/0001m002/legacysurvey-0001m002-maskbits.fits.fz",
        "NERSC Legacy Surveys DR10 (south coadd, brick 0001m002, maskbits)",
        "Public Legacy Surveys data; acknowledgement requested (legacysurvey.org).",
        "fpack's HCOMPRESS_1 tiles of int32 and uint8, a codec ymir does "
        "not decode",
        "e6a82a9cce746df80e970ca3938f1af89d3382195d402018696f7b044608a464",
        3,
        lambda r, h: [
            whole(r, h[0]),
            tiles(r, h[1], 36 + 19, 36 + 21, [200, 100]),
            tiles(r, h[2], 36 + 19, 36 + 21, [200, 100]),
        ],
    ),
    Source(
        "nicer-rmf",
        "https://heasarc.gsfc.nasa.gov/FTP/caldb/data/nicer/xti/cpf/rmf/nixtiref20170601v002.rmf",
        "HEASARC CALDB (NICER XTI response matrix)",
        "NASA HEASARC CALDB, public.",
        "an OGIP response matrix whose rows are heap arrays (1PI, 1PE), "
        "written by IDL",
        "c7ba89ae22c7953ddffe8292593c466a3ddfc13945d6361abdf695886f65938e",
        3,
        lambda r, h: [whole(r, h[0]), whole(r, h[1]), rows(r, h[2], 400, 424)],
    ),
    Source(
        "vla-uvfits",
        "https://raw.githubusercontent.com/RadioAstronomySoftwareGroup/pyuvdata/v2.4.0/pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits",
        "pyuvdata v2.4.0 test data (VLA TDEM0003, exported by CASA)",
        "pyuvdata, BSD 2-Clause; VLA data courtesy NRAO.",
        "random groups in the primary, then AIPS FQ, AN and WX tables",
        "78326b256344c17e502284425981a946d18466bfb3d22316226f0f69498fd701",
        4,
        lambda r, h: [groups(r, h[0], 12), whole(r, h[1]), whole(r, h[2]), whole(r, h[3])],
    ),
]


# Values: astropy's reading of each cut file, as digests of float64. Images
# and columns of int64, which float64 does not hold, and HCOMPRESS tiles,
# which ymir does not decode, are left out.

def digest(values):
    """MD5 of [values] as little-endian float64, every NaN as one NaN."""
    v = np.asarray(values, dtype="<f8").copy()
    v.view("<u8")[np.isnan(v)] = 0x7FF8000000000000
    return hashlib.md5(v.tobytes()).hexdigest()


def fma_scale(stored, scale, zero):
    """[zero + scale * stored] as one fma per element in float64."""
    s = np.asarray(stored, dtype=np.float64).ravel()
    return np.array([math.fma(scale, x, zero) for x in s]).reshape(np.shape(stored))


def image_lines(name, i, hdu):
    h = hdu.header
    stored = np.asarray(hdu.data)
    bitpix, blank = h["BITPIX"], h.get("BLANK")
    scale, zero = float(h.get("BSCALE", 1)), float(h.get("BZERO", 0))
    values = stored.astype(np.float64)
    if bitpix > 0:
        values = fma_scale(stored, scale, zero)
        if blank is not None:
            values[stored == blank] = np.nan
    shape = "x".join(str(n) for n in stored.shape)
    return [f"{name} {i} image {shape} {digest(values)}"]


def column_lines(name, i, hdu):
    lines = []
    raw = hdu.data.view(np.ndarray)
    for k, col in enumerate(hdu.columns, 1):
        r, code, _ = tform(str(hdu.header[f"TFORM{k}"]).strip())
        if code not in "BIJED" or r == 0:
            continue
        if [c.name for c in hdu.columns].count(col.name) != 1 or not col.name:
            continue
        stored = raw[raw.dtype.names[k - 1]]
        scale = float(hdu.header.get(f"TSCAL{k}", 1))
        zero = float(hdu.header.get(f"TZERO{k}", 0))
        if code in "BIJK":
            values = fma_scale(stored, scale, zero)
            null = hdu.header.get(f"TNULL{k}")
            if null is not None:
                values[stored == null] = np.nan
        else:
            values = fma_scale(stored, scale, zero) if (scale, zero) != (1, 0) else stored.astype(np.float64)
        shape = "x".join(str(n) for n in values.shape)
        lines.append(f"{name} {i} column {k} {shape} {digest(values)}")
    return lines


def values_lines(name, path):
    lines = []
    with fits.open(path, checksum=True, do_not_scale_image_data=True, uint=False) as hdul:
        lines.append(f"{name} hdus {len(hdul)}")
        for i, hdu in enumerate(hdul):
            h = hdu.header
            if "CHECKSUM" in h:
                lines.append(f"{name} {i} verify")
            if isinstance(hdu, fits.GroupsHDU):
                continue
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
                hcompress = getattr(hdu, "compression_type", "") == "HCOMPRESS_1"
                if hdu.data is not None and h["BITPIX"] != 64 and not hcompress:
                    lines += image_lines(name, i, hdu)
            elif isinstance(hdu, fits.BinTableHDU):
                lines += column_lines(name, i, hdu)
    return lines


# README

README_HEAD = """# Real survey files

Small cuts of public archive files, for the laws ymir.fits states over
bytes it did not write. `gen/survey.py` downloads each original by range
requests, checks the SHA-256 of the bytes it fetched, and cuts it. A cut
keeps every header record as the archive wrote it, except the rewritten
ones listed below (astropy formats them, comments kept), and keeps the data
bytes as stored. `values` holds astropy's reading of every image and
numeric binary-table column, as MD5 digests of float64.
"""


def readme(made):
    out = [README_HEAD]
    for src, cuts in made:
        out.append(f"## {src.name}.fits\n")
        out.append(f"- Source: <{src.url}>")
        out.append(f"- Archive: {src.archive}")
        out.append(f"- Terms: {src.terms}")
        out.append(f"- SHA-256 of the fetched bytes: `{src.sha256}`")
        out.append(f"- Covers: {src.why}.")
        out.append("- Cut:")
        for i, c in enumerate(cuts):
            what = "; ".join(c.changes) if c.changes else "as stored"
            out.append(f"  - HDU {i} (original HDU {c.origin}): {what}.")
        out.append("")
    return "\n".join(out)


def make(directory):
    made, lines = [], []
    for src in SOURCES:
        remote = Remote(src.url)
        hdus = walk(remote, src.hdus)
        cuts = src.cut(remote, hdus)
        got = remote.sha.hexdigest()
        if got != src.sha256:
            sys.exit(f"{src.name}: fetched bytes have SHA-256 {got}, expected {src.sha256!r}")
        path = directory / f"{src.name}.fits"
        with open(path, "wb") as f:
            for c in cuts:
                f.write(c.raw + c.data + b"\0" * (-len(c.data) % BLOCK))
        made.append((src, cuts))
        lines += values_lines(src.name, path)
        size = path.stat().st_size
        print(f"{src.name}.fits: {size} bytes")
    (directory / "README.md").write_text(readme(made))
    (directory / "values").write_text("\n".join(lines) + "\n")


def main():
    if "--check" in sys.argv:
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            make(d)
            bad = [p.name for p in d.iterdir() if not filecmp.cmp(p, SURVEY / p.name, shallow=False)]
            if bad:
                sys.exit("differ: " + ", ".join(bad))
    else:
        SURVEY.mkdir(exist_ok=True)
        make(SURVEY)


main()
