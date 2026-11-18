# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.5"]
# ///
"""Generate test/warp_gdal.dat, GDAL's warps of small rasters: gdalwarp
-et 0 (rasterio's reproject, tolerance=0), nearest and bilinear.

    uv run contrib/jord/test/gen/warp_gdal.py [--check]

Each source is the formula in [value] below with holes where [hole] holds,
the same formulas test_warp.ml computes.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform
from affine import Affine

OUT = Path(__file__).resolve().parent.parent / "warp_gdal.dat"
_out = []


def emit(s):
    _out.append(s + "\n")

def value(i, j, m):
    return (i * 7919 + j * 104729 + i * j * 31) % m + 1

def hole(i, j):
    return (i * 13 + j * 7) % 11 == 0

def source(h, w, m):
    i, j = np.mgrid[0:h, 0:w]
    return value(i, j, m).astype(np.float64), hole(i, j)

def grid_line(name, epsg, h, w, t):
    # jord's Grid: origin, row_step (b, e), col_step (a, d) of the affine map
    # x = a c + b r + c0, y = d c + e r + f0.
    return "%s %d %d %d %r %r %r %r %r %r" % (name, epsg, h, w, t.c, t.f, t.b, t.e, t.a, t.d)

def warp(src, src_t, src_crs, src_nodata, dst_shape, dst_t, dst_crs, dst_nodata, method, dtype):
    out = np.full(dst_shape, dst_nodata, dtype)
    reproject(src, out, src_transform=src_t, src_crs=src_crs, src_nodata=src_nodata,
              dst_transform=dst_t, dst_crs=dst_crs, dst_nodata=dst_nodata,
              resampling=method, tolerance=0)
    return out

def rows(name, a):
    emit(name)
    for r in a:
        emit(" ".join("nan" if np.isnan(v) else repr(float(v)) for v in r))

emit("""# GDAL %s's warps (rasterio %s), gdalwarp -et 0: for each case,
#   rasterio.warp.reproject(x, out, src_transform, src_crs, src_nodata,
#     dst_transform, dst_crs, dst_nodata, resampling, tolerance=0)
# by nearest, bilinear into float64 and, for integer sources, [rounded]:
# bilinear into the source's dtype. x is test_warp.ml's [source], value
# (i, j) with m = 1009 as float64 with NaN at the holes; for [lattice],
# m = 12000 as uint16 with 0 at the holes; for [int16lattice], m = 12000
# less 6000 as int16 with -32768 at the holes. The hole value is
# src_nodata and the integer outputs' dst_nodata; every output is written
# here as float, with nan where GDAL wrote no data.
# Every case's destination is finer than its source. Where it is not,
# GDAL widens bilinear by a scale it estimates per processing chunk, which
# jord does not: at equal resolution across UTM zones the two differ by up
# to 74 on these values, and by 1.5e-7 with XSCALE=YSCALE=1.
# Written by
#   uv run contrib/jord/test/gen/warp_gdal.py
# A case line names the case, then the two grids as
# [src|dst epsg rows cols ox oy rx ry cx cy] (Grid.v's fields).""" % (rasterio.__gdal_version__, rasterio.__version__))

def case(name, src_epsg, h, w, src_t, dst_epsg, H, W, dst_t, kind):
    emit("case %s" % name)
    emit(grid_line("src", src_epsg, h, w, src_t))
    emit(grid_line("dst", dst_epsg, H, W, dst_t))
    sc, dc = "EPSG:%d" % src_epsg, "EPSG:%d" % dst_epsg
    def run(x, nodata, out_nodata, method, dtype):
        o = warp(x, src_t, sc, nodata, (H, W), dst_t, dc, out_nodata, method, dtype)
        o = o.astype(np.float64)
        if not np.isnan(out_nodata): o[o == out_nodata] = np.nan
        return o
    if kind == "float":
        x, holes = source(h, w, 1009)
        x[holes] = np.nan
        rows("nearest", run(x, np.nan, np.nan, Resampling.nearest, np.float64))
        rows("bilinear", run(x, np.nan, np.nan, Resampling.bilinear, np.float64))
        return
    x, holes = source(h, w, 12000)
    if kind == "uint16": dtype, nodata = np.uint16, 0
    else: dtype, nodata = np.int16, -32768; x = x - 6000
    x[holes] = nodata
    x = x.astype(dtype)
    rows("nearest", run(x, nodata, nodata, Resampling.nearest, dtype))
    rows("bilinear", run(x, nodata, np.nan, Resampling.bilinear, np.float64))
    rows("rounded", run(x, nodata, nodata, Resampling.bilinear, dtype))

# Two UTM zones: a 20 m source in 32N whose north-west corner lies inside a
# 10 m tile of 33N, with NaN holes.
dst_t = Affine(10, 0, 596785.7115403633, 0, -10, 5328362.815087351)
x, y = transform("EPSG:32633", "EPSG:32632", [dst_t.c + 60], [dst_t.f - 50])
case("zones", 32632, 18, 15, Affine(20, 0, x[0] + 0.37, 0, -20, y[0] + 0.11),
     32633, 18, 16, dst_t, "float")

# TM35FIN's 16 m cells (MS-NFI's) onto a 10 m tile of UTM 35N at 27.1E, 67.8N.
x0, y0 = transform("EPSG:4326", "EPSG:32635", [27.1], [67.8])
dst_t = Affine(10, 0, round(x0[0] / 10) * 10, 0, -10, round(y0[0] / 10) * 10)
x1, y1 = transform("EPSG:32635", "EPSG:3067", [dst_t.c + 50], [dst_t.f - 40])
src_t = Affine(16, 0, np.floor(x1[0] / 16) * 16, 0, -16, np.ceil(y1[0] / 16) * 16)
case("tm35fin", 3067, 16, 15, src_t, 32635, 18, 16, dst_t, "float")

# A 20 m source in UTM 34N onto a 10 m grid of TM35FIN at 23.5E, 64N.
x0, y0 = transform("EPSG:4326", "EPSG:3067", [23.5], [64.0])
dst_t = Affine(10, 0, round(x0[0] / 10) * 10, 0, -10, round(y0[0] / 10) * 10)
x1, y1 = transform("EPSG:3067", "EPSG:32634", [dst_t.c + 40], [dst_t.f - 30])
case("to_tm35fin", 32634, 10, 9, Affine(20, 0, x1[0] + 0.21, 0, -20, y1[0] - 0.43),
     3067, 12, 12, dst_t, "float")

# A 10 m source in UTM 32N onto a 0.00005° Lonlat grid at 9.3E, 47.7N.
x1, y1 = transform("EPSG:4326", "EPSG:32632", [9.3002], [47.6998])
case("to_lonlat", 32632, 14, 14, Affine(10, 0, x1[0] + 0.3, 0, -10, y1[0] + 0.7),
     4326, 12, 12, Affine(0.00005, 0, 9.3, 0, -0.00005, 47.7), "float")

# A 20 m uint16 band onto a 10 m tile of its zone, on its lattice, with 0 as
# nodata: the weights are exact.
case("lattice", 32631, 12, 14, Affine(20, 0, 299960, 0, -20, 5800140),
     32631, 12, 24, Affine(10, 0, 300010, 0, -10, 5800010), "uint16")

# The same with int16 values of either sign: negative ties.
case("int16lattice", 32631, 12, 14, Affine(20, 0, 299960, 0, -20, 5800140),
     32631, 12, 24, Affine(10, 0, 300010, 0, -10, 5800010), "int16")

# A lon/lat source across the antimeridian, 179.998 to 180.002, onto a 10 m
# tile of UTM 1S over it.
src_t = Affine(0.0001, 0, 179.998, 0, -0.0001, -16.998)
x0, y0 = transform("EPSG:4326", "EPSG:32701", [179.9988], [-16.9988])
dst_t = Affine(10, 0, x0[0], 0, -10, y0[0])
case("antimeridian", 4326, 30, 40, src_t, 32701, 16, 20, dst_t, "float")


text = "".join(_out)
if "--check" in sys.argv[1:]:
    if OUT.read_text() != text:
        sys.exit(f"{OUT} would change; run without --check")
else:
    OUT.write_text(text)
