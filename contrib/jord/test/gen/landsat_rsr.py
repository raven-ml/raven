# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate test/landsat_rsr.dat, the samples of Landsat's spectral
responses around each half-maximum crossing.

    uv run contrib/jord/test/gen/landsat_rsr.py [--check]

It fetches the USGS's band-average relative spectral responses (RSR) from
the Spectral Characteristics Viewer
(https://landsat.usgs.gov/spectral-characteristics-viewer), each pinned by
its SHA-256 below; a file whose SHA-256 is not its pin fails the run.

For each band whose centre and width jord derives (TM, ETM+ and Landsat 8's
OLI), it writes the response's peak and the two samples on each side of its
first rising and its last falling crossing of half the peak, wavelengths in
micrometres. test_landsat.ml interpolates them linearly and compares the
result, rounded to 0.1 nm, with Landsat.band's values.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import hashlib
import io
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "landsat_rsr.dat"
BASE = "https://landsat.usgs.gov/landsat/spectral_viewer/bands/"
PINS = {
    "L4_TM_RSR.xlsx": "db3079622f1a7702181a16f0f0a6c7bf02a7db292c81d08ff495419fce9a8a38",
    "L5_TM_RSR.xlsx": "1d7e0bd16bf5723acee1958c0e3ae852b6452298dabfd049865d227fb34bb57c",
    "L7_ETM_RSR.xlsx": "b075f8ff6ff8fb894d45c0a2979486a77878e453e7788dc671457d3f9bc0ceee",
    "L8_OLI_RSR.xlsx": "6b8bb8bf0c6f1a3cef56eb497096283f6cd66f1af163989eb03c59aa6588cfe0",
    "HANDBOOK_LandsatThermalRSRs.spectral.dat.xlsx":
        "4d770865ad73bb850528bff19f74d7d6c144857f38b2ea3d4fe24ab1f3a22d3a",
}

HEADER = """\
# Landsat's band-average relative spectral responses (RSR) around each
# half-maximum crossing, from the USGS's Spectral Characteristics Viewer
# (https://landsat.usgs.gov/spectral-characteristics-viewer: L4_TM_RSR,
# L5_TM_RSR, L7_ETM_RSR and L8_OLI_RSR .xlsx, and
# HANDBOOK_LandsatThermalRSRs.spectral.dat.xlsx for band 6), fetched
# 2026-11-16, cut by
#   uv run contrib/jord/test/gen/landsat_rsr.py
# The derived crossings interpolate samples 1 nm apart, but 2 or 3 nm on
# TM's and ETM+'s shortwave-infrared bands and 10 to 50 nm on their band 6.
# The stated 50% points (test_landsat.ml's [stated]) are given to 0.001 nm
# for OLI-2, 0.001 um for Landsat 8's TIRS and 0.01 um for TIRS-2 (10.45
# and 11.2, whose midpoint 10.825 its file rounds to 10.83). On OLI-2's
# responses the derivation gives its stated wavelengths within 0.11 nm.
# Each row: satellite, band, the response's peak, then the two samples
# (wavelength in micrometres, response) of its first rising crossing of
# half the peak and the two of its last falling one.
"""

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def fetch(name):
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "jord-landsat-rsr"})
    with urllib.request.urlopen(req, timeout=300) as f:
        b = f.read()
    got = hashlib.sha256(b).hexdigest()
    if got != PINS[name]:
        sys.exit(f"{name}: SHA-256 {got}, pinned {PINS[name]}: source changed")
    return b


def sheets(data):
    z = zipfile.ZipFile(io.BytesIO(data))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for s in wb.find(NS + "sheets"):
        t = target[s.get(REL)].lstrip("/")
        t = t if t.startswith("xl/") else "xl/" + t
        rows = []
        for row in ET.fromstring(z.read(t)).iter(NS + "row"):
            cells = {}
            for c in row.findall(NS + "c"):
                col = re.match(r"[A-Z]+", c.get("r")).group(0)
                v = c.find(NS + "v")
                if v is not None:
                    cells[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
            rows.append(cells)
        out[s.get("name")] = rows
    return out


def curve(rows, col, start, scale):
    pts = []
    for r in rows[start:]:
        try:
            pts.append((float(r["A"]) * scale, float(r[col])))
        except (KeyError, ValueError):
            pass
    return pts


def crossings(pts):
    peak = max(r for _, r in pts)
    h = peak / 2
    rise = fall = None
    for a, b in zip(pts, pts[1:]):
        if rise is None and a[1] < h <= b[1]:
            rise = (a, b)
        if a[1] >= h > b[1]:
            fall = (a, b)
    return peak, rise, fall


def text():
    tm = [("Blue", "Blue"), ("Green", "Green"), ("Red", "Red"), ("Nir", "NIR"),
          ("Swir16", "SWIR(5)"), ("Swir22", "SWIR(7)")]
    oli = [("Coastal", "CoastalAerosol"), ("Blue", "Blue"), ("Green", "Green"),
           ("Red", "Red"), ("Nir", "NIR"), ("Swir16", "SWIR1"), ("Swir22", "SWIR2")]
    bands = []
    for sat, f, suf in [("Landsat_4", "L4_TM_RSR.xlsx", "-L4 TM"),
                        ("Landsat_5", "L5_TM_RSR.xlsx", "-L5 TM"),
                        ("Landsat_7", "L7_ETM_RSR.xlsx", "-L7")]:
        sh = sheets(fetch(f))
        for b, s in tm:
            bands.append((sat, b, curve(sh[s + suf], "B", 1, 1e-3)))
    th = sheets(fetch("HANDBOOK_LandsatThermalRSRs.spectral.dat.xlsx"))["HANDBOOK spectral_b6.dat"]
    for sat, col in [("Landsat_4", "C"), ("Landsat_5", "D"), ("Landsat_7", "B")]:
        bands.append((sat, "Lwir", curve(th, col, 3, 1.)))
    sh = sheets(fetch("L8_OLI_RSR.xlsx"))
    for b, s in oli:
        bands.append(("Landsat_8", b, curve(sh[s], "B", 1, 1e-3)))
    out = [HEADER]
    for sat, b, pts in bands:
        peak, rise, fall = crossings(pts)
        samples = [rise[0], rise[1], fall[0], fall[1]]
        cols = [repr(peak)] + [f"{w:.10g} {r!r}" for w, r in samples]
        out.append(" ".join([sat, b] + cols) + "\n")
    return "".join(out)


def main():
    t = text()
    if "--check" in sys.argv[1:]:
        if OUT.read_text() != t:
            sys.exit(f"{OUT} would change; run without --check")
    else:
        OUT.write_text(t)


if __name__ == "__main__":
    main()
