# /// script
# requires-python = ">=3.11"
# dependencies = ["pyproj==3.8.0"]
# ///
"""Generate test/wkt.dat, the WKT texts Crs.of_wkt's tests read.

    uv run contrib/jord/test/gen/wkt.py [--check]

Three sources, one text a line:

- epsg: EPSG's own WKT export of each code, from the EPSG registry's API
  (https://apps.epsg.org/api/v1/CoordRefSystem/<code>/export/?format=wkt),
  each pinned by its SHA-256 below; a text whose SHA-256 is not its pin
  fails the run with "source changed".
- proj: PROJ's WKT2 (2019) of each code Crs.of_epsg reads, by pyproj
  (CRS.from_epsg(code).to_wkt(WktVersion.WKT2_2019)), and of the mpc text.
- mpc: the proj:wkt2 of the Planetary Computer's MODIS item that
  test/stac/mpc_modis_13q1.json holds, a WKT 1 text.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pyproj
from pyproj import CRS
from pyproj.enums import WktVersion

TEST = Path(__file__).resolve().parent.parent
OUT = TEST / "wkt.dat"
EPSG = "https://apps.epsg.org/api/v1/CoordRefSystem/{}/export/?format=wkt"
ITEM = "MOD13Q1.A2023241.h18v04.061.2023258004333"
PINS = {
    4326: "8dc08ac3a39f281535da7fbe6bc78d3b093bdd4f700221053e37126103bb240c",
    4269: "53c7a7801f5f4ce999aad50169cdb7a7eddd704626888de1475495b7ed584e3d",
    26917: "a3add8c20491761eef80bb6ac17dae5b237322b9da5a5e3eef4c1236483af465",
    26904: "f141aaed2ba36e47717c89fdb6642f6ed67900188653b310b71d0baf2655a770",
    3067: "4ce1b8b70bf072ad011bddcbadaa05afad553c0c1bf1f0fa043a4ed82349ec1a",
    4167: "d8c20abe1d148c13ea7a9f770f204ae9c9746be9ce9ba4394042cbefcf033cb3",
    2193: "f23bc5564299c07643bad8c7b72a72acac6cbc6784b794a6dd99d4ef575df107",
    5070: "230ed49511755c4fe0c726c44d7d99eab758da9a6d6f9703eb03cd6e9cf75bc2",
    4283: "2d95b68dfe256e6a20af426bda72e71c406ef7c660948e0e46fa280ad7f8ec49",
    3577: "2636a63fbbcbd3bf1c040369f18671bd8635b5b7ce6299e05bccff41f807568d",
    3857: "10a102b957a42f77d930c144777a1ac86796485e8d0d672bc73f02b3f0248928",
    25835: "ea6f9f5fecbbe242618398ecb492f87ed34aaa5961e5a33c5b556f021bced135",
    4258: "cfe65694b877d6ac505268ad53bb65c780fedfc51f8a52e0917970e5ec904aa0",
}
PROJ = (4326, 32633, 32760, 4269, 26917, 10690, 3067, 4167, 2193, 5070, 4283, 3577, 3857)

HEADER = f"""\
# WKT texts for Crs.of_wkt, one a line: source, the system's code, text.
# epsg: EPSG's own export, https://apps.epsg.org/api/v1/CoordRefSystem/<code>/export/?format=wkt
#   (dataset 12.029, fetched 2026-11-16).
# proj: PROJ {pyproj.__proj_version__}'s WKT2 (pyproj {pyproj.__version__}, CRS.from_epsg(code).to_wkt(WktVersion.WKT2_2019);
#   for modis, of the mpc text).
# mpc: the proj:wkt2 of Microsoft's Planetary Computer's MODIS item
#   {ITEM} (collection modis-13Q1-061), a WKT 1 text.
# Written by
#   uv run contrib/jord/test/gen/wkt.py
# The epsg texts are the EPSG Dataset's, which IOGP owns, under its terms of
# use, https://epsg.org/terms-of-use.html.
"""


def epsg(code):
    req = urllib.request.Request(EPSG.format(code), headers={"User-Agent": "jord-wkt"})
    with urllib.request.urlopen(req, timeout=300) as f:
        b = f.read()
    got = hashlib.sha256(b).hexdigest()
    if got != PINS.get(code):
        sys.exit(f"EPSG {code}: SHA-256 {got}, pinned {PINS.get(code)}: source changed")
    return b.decode()


def mpc():
    items = json.loads((TEST / "stac" / "mpc_modis_13q1.json").read_text())["features"]
    item = next(i for i in items if i["id"] == ITEM)
    return item["properties"]["proj:wkt2"]


def text():
    out = [HEADER]
    for code in PINS:
        out.append(f"epsg {code} {epsg(code)}\n")
    for code in PROJ:
        out.append(f"proj {code} {CRS.from_epsg(code).to_wkt(WktVersion.WKT2_2019)}\n")
    m = mpc()
    out.append(f"proj modis {CRS.from_wkt(m).to_wkt(WktVersion.WKT2_2019)}\n")
    out.append(f"mpc modis {m}\n")
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
