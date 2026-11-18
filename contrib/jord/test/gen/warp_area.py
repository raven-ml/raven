# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.5", "pyproj==3.8.0", "shapely==2.2.0"]
# ///
"""Generate test/warp_area.dat, the area rules' cases: GDAL's -r average and
-r mode on same-system grids, and an exact area mean across UTM zones.

    uv run contrib/jord/test/gen/warp_area.py [--check]

The GDAL sources are the formulas of test_warp.ml's [value] and [hole],
which the OCaml test recomputes. The xzone case's source is a window of the
B04 band of Sentinel-2 L2A item ITEM, read from Microsoft's Planetary
Computer (its COG, signed by the Planetary Computer's SAS token service):
the block around a 200 x 200 grid of UTM 32N at 10 m, origin
287600/5543800, padded by 8 cells, whose values SHA256 pins; values that
differ from the pin fail the run.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds

OUT = Path(__file__).resolve().parent.parent / "warp_area.dat"
ITEM = "S2A_MSIL2A_20250702T103701_R008_T31UGR_20250702T124303"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
SHA256 = "512ccd731e069bae8dc195c211ad25424802536ef2a4a5057227e88d69df3501"
_out = []


def emit(s):
    _out.append(s + "\n")


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "jord-warp-area"})
    with urllib.request.urlopen(req, timeout=300) as f:
        return json.load(f)


def xzone_source():
    """The B04 window of ITEM around the xzone grid, its transform and the
    values' SHA-256."""
    href = get_json(STAC + ITEM)["assets"]["B04"]["href"]
    token = get_json(SAS)["token"]
    t = Affine(10, 0, 287600.0, 0, -10, 5543800.0)
    bounds = (t.c, t.f - 2000.0, t.c + 2000.0, t.f)
    with rasterio.open(href + "?" + token) as s:
        l, bo, r, tp = transform_bounds("EPSG:32632", s.crs, *bounds, densify_pts=101)
        w = from_bounds(l, bo, r, tp, s.transform).round_offsets().round_lengths()
        w = Window(w.col_off - 8, w.row_off - 8, w.width + 16, w.height + 16)
        x = s.read(1, window=w)
        return x, s.window_transform(w), hashlib.sha256(x.tobytes()).hexdigest()

def value(i, j, m):
    return (i * 7919 + j * 104729 + i * j * 31) % m + 1

def hole(i, j):
    return (i * 13 + j * 7) % 11 == 0

def grid_line(name, epsg, h, w, t):
    return "%s %d %d %d %r %r %r %r %r %r" % (name, epsg, h, w, t.c, t.f, t.b, t.e, t.a, t.d)

def rows(name, a):
    emit(name)
    for r in a:
        emit(" ".join("nan" if np.isnan(v) else repr(float(v)) for v in r))

# The exact area mean: each destination cell's footprint is the quadrilateral
# of its four corners in the source's pixel plane; a_k = area(footprint ∩ cell
# k) by the vertical-slice formula.
def m(u, v):
    both = (u >= 0) & (v >= 0)
    mixed = ((u > 0) & (v < 0)) | ((u < 0) & (v > 0))
    hi = np.maximum(u, v)
    den = np.where(mixed, np.abs(u) + np.abs(v), 1.0)
    return np.where(both, (u + v) / 2, np.where(mixed, hi * hi / (2 * den), 0.0))

def cell_area(qx, qy, x0, y0):
    tot = 0.0
    for e in range(4):
        px, py = qx[..., e], qy[..., e]
        nx_, ny = qx[..., (e + 1) % 4], qy[..., (e + 1) % 4]
        dx = nx_ - px
        lo = np.clip(np.minimum(px, nx_), x0, x0 + 1); hi = np.clip(np.maximum(px, nx_), x0, x0 + 1)
        safe = np.where(dx == 0, 1.0, dx)
        ya = py + (lo - px) / safe * (ny - py); yb = py + (hi - px) / safe * (ny - py)
        tot = tot + np.sign(dx) * (hi - lo) * (m(ya - y0, yb - y0) - m(ya - y0 - 1, yb - y0 - 1))
    return np.abs(tot)

def quads(cx, cy):
    qx = np.stack([cx[:-1, :-1], cx[:-1, 1:], cx[1:, 1:], cx[1:, :-1]], -1)
    qy = np.stack([cy[:-1, :-1], cy[:-1, 1:], cy[1:, 1:], cy[1:, :-1]], -1)
    return qx, qy

def exact_mean(qx, qy, a, ok):
    ix0 = np.floor(qx.min(-1)).astype(int); iy0 = np.floor(qy.min(-1)).astype(int)
    K = int(max((np.ceil(qx.max(-1)) - ix0).max(), (np.ceil(qy.max(-1)) - iy0).max()))
    acc = np.zeros(qx.shape[:-1]); ws = np.zeros(qx.shape[:-1])
    for dj in range(K):
        for di in range(K):
            x0 = ix0 + di; y0 = iy0 + dj
            w = cell_area(qx, qy, x0, y0)
            inside = (y0 >= 0) & (y0 < a.shape[0]) & (x0 >= 0) & (x0 < a.shape[1])
            yc = np.clip(y0, 0, a.shape[0] - 1); xc = np.clip(x0, 0, a.shape[1] - 1)
            use = inside & ok[yc, xc]
            acc += np.where(use, w * a[yc, xc], 0); ws += np.where(use, w, 0)
    return acc / ws, ws

# The same areas by shapely's polygon intersection, cell by cell: an oracle
# independent of the formula.
def shapely_mean(qx, qy, a, ok):
    out = np.zeros(qx.shape[:-1])
    for idx in np.ndindex(*qx.shape[:-1]):
        fp = shapely.Polygon(list(zip(qx[idx], qy[idx])))
        x0, y0, x1, y1 = fp.bounds
        acc = ws = 0.0
        for r in range(int(np.floor(y0)), int(np.ceil(y1))):
            for c in range(int(np.floor(x0)), int(np.ceil(x1))):
                if 0 <= r < a.shape[0] and 0 <= c < a.shape[1] and ok[r, c]:
                    w = fp.intersection(shapely.box(c, r, c + 1, r + 1)).area
                    acc += w * a[r, c]; ws += w
        out[idx] = acc / ws
    return out


# GDAL's box rule for an axis-aligned footprint [r0, r1] x [c0, c1] in source
# pixels (GWKAverageOrModeComputeSourceCoords and COMPUTE_WEIGHT,
# gdalwarpkernel.cpp:6690-6701, :6833-6856 at 3.12.2): the cells it reads are
# clamped to the source, and the first and last are weighed by their distance
# to the footprint's edge, so a footprint past the source's edge weighs the
# edge cell by 1 plus the overhang.
def gdal_box(a, ok, r0, r1, c0, c1):
    eps = 1e-10
    h, w = a.shape
    y0 = int(max(np.floor(r0 + eps), 0)); y1 = min(int(np.ceil(r1 - eps)), h)
    x0 = int(max(np.floor(c0 + eps), 0)); x1 = min(int(np.ceil(c1 - eps)), w)
    if y0 == y1 and y1 < h: y1 += 1
    if x0 == x1 and x1 < w: x1 += 1
    def weight(i, lo, hi, d0, d1):
        if i == lo: return 1.0 if lo + 1 == hi else 1 - (d0 - lo)
        return 1 - (hi - d1) if i + 1 == hi else 1.0
    acc = ws = 0.0
    for r in range(y0, y1):
        for c in range(x0, x1):
            if ok[r, c]:
                wt = weight(r, y0, y1, r0, r1) * weight(c, x0, x1, c0, c1)
                acc += wt * a[r, c]; ws += wt
    return acc / ws if ws > 0 else np.nan

def check_box(x, ok, src_t, dst_t, g):
    worst = 0.0
    for I in range(g.shape[0]):
        for J in range(g.shape[1]):
            c0, r0 = ~src_t * (dst_t * (J, I)); c1, r1 = ~src_t * (dst_t * (J + 1, I + 1))
            if np.isnan(g[I, J]): continue
            e = gdal_box(x, ok, r0, r1, c0, c1)
            worst = max(worst, abs(e - g[I, J]) / abs(g[I, J]))
    assert worst < 1e-12, worst
    return worst

box_checked = []

def case(name, epsg, h, w, src_t, H, W, dst_t, kind):
    crs = "EPSG:%d" % epsg
    lines = ["case %s" % name, grid_line("src", epsg, h, w, src_t), grid_line("dst", epsg, H, W, dst_t)]
    i, j = np.mgrid[0:h, 0:w]
    def run(x, nodata, out_nodata, method, dtype, **extra):
        o = np.full((H, W), out_nodata, dtype)
        reproject(x, o, src_transform=src_t, src_crs=crs, src_nodata=nodata,
                  dst_transform=dst_t, dst_crs=crs, dst_nodata=out_nodata,
                  resampling=method, tolerance=0, **extra)
        o = o.astype(np.float64)
        if not np.isnan(out_nodata): o[o == out_nodata] = np.nan
        return o
    outs = []
    if kind == "float":
        x = value(i, j, 1009).astype(np.float64); x[hole(i, j)] = np.nan
        outs.append(("average", run(x, np.nan, np.nan, Resampling.average, np.float64)))
        box_checked.append(check_box(x, ~hole(i, j), src_t, dst_t, outs[-1][1]))
    elif kind == "uint16":
        x = value(i, j, 12000).astype(np.uint16); x[hole(i, j)] = 0
        outs.append(("average", run(x, 0, np.nan, Resampling.average, np.float64)))
        box_checked.append(check_box(x.astype(np.float64), ~hole(i, j), src_t, dst_t, outs[-1][1]))
        outs.append(("rounded", run(x, 0, 0, Resampling.average, np.uint16)))
    else:
        x = value(i, j, 4).astype(np.uint8); x[hole(i, j)] = 0
        outs.append(("mode", run(x, 0, 0, Resampling.mode, np.uint8, MODE_TIES="MIN")))
    return lines, outs

cases = []
o = (300000.0, 5800000.0)
T = lambda d, x, y: Affine(d, 0, x, 0, -d, y)
# 10 m onto 20 m on the same lattice, NaN holes.
cases.append(case("aligned20", 32631, 12, 14, T(10, *o), 6, 7, T(20, *o), "float"))
# 10 m onto 25 m shifted by (0.37, 0.61) of a source cell, reaching past the
# source's east and south edges.
cases.append(case("shifted25", 32631, 20, 20, T(10, *o), 9, 9,
                  T(25, o[0] + 6.1, o[1] - 3.7), "float"))
# 10 m uint16 onto 30 m shifted by (0.5, 0.25) of a source cell.
cases.append(case("holes30", 32631, 21, 21, T(10, *o), 7, 7,
                  T(30, o[0] + 2.5, o[1] - 5.0), "uint16"))
# The same onto 30 m on the lattice: means of 6 and 8 cells give exact ties.
cases.append(case("ties30", 32631, 21, 21, T(10, *o), 7, 7, T(30, *o), "uint16"))
# Classes 1..4 onto 20 m on the lattice (ties), and onto 25 m shifted (areas).
cases.append(case("mode20", 32631, 12, 14, T(10, *o), 6, 7, T(20, *o), "mode"))
cases.append(case("mode25", 32631, 20, 20, T(10, *o), 9, 9,
                  T(25, o[0] + 6.1, o[1] - 3.7), "mode"))

# The xzone case.
g_t = T(10, 287600.0, 5543800.0)
r0 = c0 = 90; n = 12
dst_t = g_t * Affine.translation(c0, r0)
a_full, s_transform, sha = xzone_source()
if sha != SHA256:
    sys.exit(f"{ITEM} B04: SHA-256 {sha}, pinned {SHA256}: source changed")
tr = Transformer.from_crs("EPSG:32632", "EPSG:32631", always_xy=True)
R, C = np.meshgrid(np.arange(n + 1.0), np.arange(n + 1.0), indexing="ij")
X, Y = dst_t * (C, R)
X, Y = tr.transform(X, Y)
cx, cy = ~s_transform * (X, Y)
lo_c = int(np.floor(cx.min())) - 1; hi_c = int(np.ceil(cx.max())) + 1
lo_r = int(np.floor(cy.min())) - 1; hi_r = int(np.ceil(cy.max())) + 1
a = a_full[lo_r:hi_r, lo_c:hi_c]
src_t = s_transform * Affine.translation(lo_c, lo_r)
cx = cx - lo_c; cy = cy - lo_r
qx, qy = quads(cx, cy)
ok = a != 0
ex, ws = exact_mean(qx, qy, a.astype(np.float64), ok)
sh = shapely_mean(qx, qy, a.astype(np.float64), ok)
agree = np.abs(ex - sh).max() / np.abs(ex).max()
assert agree < 1e-12, agree

emit("""# GDAL %s's area rules (rasterio %s), and one exact area mean. For each
# GDAL case,
#   rasterio.warp.reproject(x, out, src_transform, src_crs, src_nodata,
#     dst_transform, dst_crs, dst_nodata, resampling, tolerance=0[, MODE_TIES=MIN])
# by average into float64 ([average]) and, for integer sources, into the
# source's dtype ([rounded]), and by mode into the source's dtype with
# MODE_TIES=MIN ([mode]). Every GDAL case is in one system, where GDAL's
# two-corner box is the footprint (gdalwarpkernel.cpp:6787-6793 at 3.12.2).
# Where a footprint reaches past the source's edge, GDAL weighs the edge cell
# by 1 plus the overhang (COMPUTE_WEIGHT, :6690-6701, on cells clamped to the
# source at :6839-6856): every [average] here equals that box rule to %.1e
# relative, which the generator checks.
# x is test_warp.ml's [source]: value (i, j) with m = 1009 as float64 with
# NaN at the holes; for [holes30] and [ties30], m = 12000 as uint16 with 0 at
# the holes; for the mode cases, m = 4 as uint8 with 0 at the holes. The hole
# value is src_nodata and the integer outputs' dst_nodata; every output is
# written as float, nan where GDAL wrote no data.
# Case [xzone] is not GDAL's: 12 x 12 cells of a 10 m grid of UTM 32N (from
# cell (90, 90) of the 200 x 200 grid at 287600/5543800), over the B04 band
# of S2 31UGR on 2025-07-02 (UTM 31N), whose window is the [source] block
# (uint16, 0 is nodata; contains modified Copernicus Sentinel data 2025,
# test/README.md). [average] is the exact mean over each cell's
# footprint, the quadrilateral of its corners taken into the source's pixel
# plane by pyproj %s, by the vertical-slice formula; shapely %s's polygon
# intersection gives the same means to %.1e relative.
# Written by
#   uv run contrib/jord/test/gen/warp_area.py
# A case line names the case, then the two grids as
# [src|dst epsg rows cols ox oy rx ry cx cy] (Grid.v's fields); a [source]
# block has the source's rows.""" % (
    rasterio.__gdal_version__, rasterio.__version__, max(box_checked),
    __import__("pyproj").__version__, shapely.__version__, agree))
for lines, outs in cases:
    emit("\n".join(lines))
    for name, v in outs:
        rows(name, v)
emit("case xzone")
emit(grid_line("src", 32631, a.shape[0], a.shape[1], src_t))
emit(grid_line("dst", 32632, n, n, dst_t))
rows("source", a.astype(np.float64))
rows("average", ex)


text = "".join(_out)
if "--check" in sys.argv[1:]:
    if OUT.read_text() != text:
        sys.exit(f"{OUT} would change; run without --check")
else:
    OUT.write_text(text)
