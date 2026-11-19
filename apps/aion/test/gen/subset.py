# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==2.5.5",
#   "safetensors==0.8.0",
#   "h5py==3.16.0",
#   "fsspec==2026.9.0",
#   "aiohttp==3.14.5",
#   "requests==2.34.2",
# ]
# ///
"""Extract the AION-1 parity subset from remote HDF5 files by HTTP Range reads.

    uv run test/gen/subset.py --list SOURCE          # dataset names of one source
    uv run test/gen/subset.py OUT_DIR [SOURCE ...]   # write OUT_DIR/<source>.safetensors

Every source file is HDF5 (the tutorial's cross-matched provabgs_desi_ls.hdf5 and the
Multimodal Universe v1 tree at Flatiron). h5py reads them through fsspec's HTTP file,
so only the byte ranges of the selected rows (and chunk indexes) are fetched.

Each output holds the selected rows of the named datasets, under their HDF5 names
with '/' replaced by '.'. Booleans are stored as uint8; fixed-width byte strings
(object ids, band names) as uint8 arrays of shape (n, width), NUL-padded. The row
indices taken are stored as "rows" (int64). Nx_io.load_safetensors reads every file.
"""
import argparse, os, sys

import fsspec, h5py, numpy as np
from safetensors.numpy import save_file

MMU = "https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1"
TUTORIAL = "https://users.flatironinstitute.org/~polymathic/data/provabgs_desi_ls.hdf5"

LS_FLUX = [f"FLUX_{b}" for b in "GRIZ"] + [f"FLUX_W{i}" for i in range(1, 5)]

# name -> (url, row selection, datasets). A selection is ("first", n) or
# ("where", n, dataset): the first n rows where that dataset is nonzero.
SOURCES = {
    # LS image, LS fluxes and EBV, DESI spectrum, cross-matched; PROVABGS redshift.
    "ls_desi": (TUTORIAL, ("first", 64), [
        "legacysurvey/object_id", "legacysurvey/ra", "legacysurvey/dec",
        "legacysurvey/image_array", "legacysurvey/image_band", "legacysurvey/EBV",
        *[f"legacysurvey/{c}" for c in LS_FLUX],
        "desi/object_id", "desi/ra", "desi/dec",
        "desi/spectrum_flux", "desi/spectrum_ivar", "desi/spectrum_lambda", "desi/spectrum_mask",
        "provabgs_desi_edr_sv3/Z_HP"]),
    # LS catalog scalars (shapes), the in-cutout catalog, and an LS image read
    # through the chunked LZF path.
    "ls_catalog": (f"{MMU}/legacysurvey/dr10_south_21/healpix=313/001-of-001.hdf5", ("first", 32), [
        "object_id", "ra", "dec", "image_array", "image_band", "EBV", *LS_FLUX,
        "SHAPE_R", "SHAPE_E1", "SHAPE_E2", "TYPE",
        "catalog_X", "catalog_Y", "catalog_SHAPE_R", "catalog_SHAPE_E1", "catalog_SHAPE_E2",
        "catalog_TYPE", *[f"catalog_FLUX_{b}" for b in "GRIZ"]]),
    "hsc": (f"{MMU}/hsc/pdr3_dud_22.5/healpix=1107/001-of-001.hdf5", ("first", 64), [
        "object_id", "ra", "dec", "image_array", "image_band",
        *[f"a_{b}" for b in "grizy"], *[f"{b}_cmodel_mag" for b in "grizy"],
        "i_sdssshape_shape11", "i_sdssshape_shape22", "i_sdssshape_shape12"]),
    "sdss": (f"{MMU}/sdss/sdss/healpix=313/001-of-001.hdf5", ("first", 64), [
        "object_id", "ra", "dec", "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
        "spectrum_mask", "Z"]),
    "desi": (f"{MMU}/desi/edr_sv3/healpix=583/001-of-001.hdf5", ("first", 64), [
        "object_id", "ra", "dec", "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
        "spectrum_mask", "Z"]),
    "gaia": (f"{MMU}/gaia/gaia/healpix=1631/001-of-001.hdf5", ("where", 256, "has_xp_continuous"), [
        "source_id", "ra", "dec", "coeff", "coeff_error", "phot_g_mean_flux",
        "phot_bp_mean_flux", "phot_rp_mean_flux", "parallax"]),
}


def open_h5(url):
    fs = fsspec.filesystem("http")
    # readahead: small metadata reads share 64 kB blocks; a large slab is one request.
    fo = fs.open(url, "rb", block_size=1 << 16, cache_type="readahead")
    return fo, h5py.File(fo, "r")


def store(x):
    if x.dtype == np.bool_:
        return x.astype(np.uint8)
    if x.dtype.kind == "S":
        w = x.dtype.itemsize
        return np.frombuffer(np.ascontiguousarray(x).tobytes(), np.uint8).reshape(*x.shape, w).copy()
    return np.ascontiguousarray(x)


def rows_of(f, sel):
    if sel[0] == "first":
        return np.arange(sel[1])
    flag = f[sel[2]][:]
    return np.flatnonzero(flag != 0)[: sel[1]]


def extract(name, out_dir):
    url, sel, names = SOURCES[name]
    fo, f = open_h5(url)
    with fo, f:
        rows = rows_of(f, sel)
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        arrays = {"rows": rows.astype(np.int64)}
        for n in names:
            if n not in f:
                print(f"{name}: no dataset {n}", file=sys.stderr)
                continue
            # Read one contiguous slab, then pick rows: one range per dataset.
            arrays[n.replace("/", ".")] = store(f[n][lo:hi][rows - lo])
    out = os.path.join(out_dir, f"{name}.safetensors")
    save_file(arrays, out, metadata={"source": url, "rows": ",".join(map(str, rows))})
    print(name, url)
    for k, v in arrays.items():
        print(f"  {k:40s} {v.dtype.str:5s} {v.shape}")


p = argparse.ArgumentParser()
p.add_argument("--list", metavar="SOURCE")
p.add_argument("out", nargs="?")
p.add_argument("sources", nargs="*")
a = p.parse_args()

if a.list:
    fo, f = open_h5(SOURCES[a.list][0])
    with fo, f:
        print(" ".join(f.keys()))
    sys.exit(0)

os.makedirs(a.out, exist_ok=True)
for s in a.sources or SOURCES:
    extract(s, a.out)
