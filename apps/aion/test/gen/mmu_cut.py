"""Cut real Multimodal Universe rows into small local HDF5 files, for parity.exe's parts
mmu and generate.

    uv run --locked mmu_cut.py OUT_DIR [GAIA]

Reads each source by HTTP Range through h5py and fsspec, so only the rows' bytes and the
files' metadata are fetched, and writes OUT_DIR/<name>.h5 with the first rows of each
dataset Survey reads, under the same names and in the same groups, at the same dtypes,
with the source's chunk shape (cut to the rows) and filter. Gaia's healpix 1631 file
(16 MB) is copied whole from GAIA when given, or fetched whole.

  tutorial  provabgs_desi_ls.hdf5, groups legacysurvey and desi, 16 rows
  ls        legacysurvey dr10_south_21 healpix 313, 8 rows (LZF chunks)
  hsc       hsc pdr3_dud_22.5 healpix 1107, 8 rows (deflate chunks)
  desi      desi edr_sv3 healpix 583, 16 rows
  sdss      sdss healpix 313, 16 rows (-1 padded spectra)
  gaia      gaia healpix 1631, whole
"""
import os, shutil, sys, urllib.request

import fsspec, h5py, numpy as np

MMU = "https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1"
TUTORIAL = "https://users.flatironinstitute.org/~polymathic/data/provabgs_desi_ls.hdf5"
GAIA = f"{MMU}/gaia/gaia/healpix=1631/001-of-001.hdf5"

LS = (["object_id", "ra", "dec", "image_array", "image_band", "EBV"]
      + [f"FLUX_{b}" for b in "GRIZ"] + [f"FLUX_W{i}" for i in range(1, 5)]
      + ["SHAPE_R", "SHAPE_E1", "SHAPE_E2"])
HSC = (["object_id", "ra", "dec", "image_array", "image_band"]
       + [f"a_{b}" for b in "grizy"] + [f"{b}_cmodel_mag" for b in "grizy"]
       + ["i_sdssshape_shape11", "i_sdssshape_shape22", "i_sdssshape_shape12"])
SPEC = ["object_id", "ra", "dec", "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
        "spectrum_mask", "Z"]

SOURCES = {
    "tutorial": (TUTORIAL, 16, {"legacysurvey": LS, "desi": SPEC}),
    "ls": (f"{MMU}/legacysurvey/dr10_south_21/healpix=313/001-of-001.hdf5", 8, {"": LS}),
    "hsc": (f"{MMU}/hsc/pdr3_dud_22.5/healpix=1107/001-of-001.hdf5", 8, {"": HSC}),
    "desi": (f"{MMU}/desi/edr_sv3/healpix=583/001-of-001.hdf5", 16, {"": SPEC}),
    "sdss": (f"{MMU}/sdss/sdss/healpix=313/001-of-001.hdf5", 16, {"": SPEC}),
}


def cut(name, out):
    url, n, groups = SOURCES[name]
    fs = fsspec.filesystem("http")
    fetched = 0
    with fs.open(url, "rb", block_size=1 << 16, cache_type="readahead") as fo, \
            h5py.File(fo, "r") as f, h5py.File(f"{out}/{name}.h5", "w") as g:
        for group, names in groups.items():
            src = f[group] if group else f
            dst = g.create_group(group) if group else g
            for d in names:
                if d not in src:
                    print(f"{name}: {group}/{d} absent", file=sys.stderr)
                    continue
                s = src[d]
                x = s[:n]
                fetched += x.nbytes
                kw = {}
                if s.chunks:
                    kw["chunks"] = (min(s.chunks[0], n),) + s.chunks[1:]
                    if s.compression:
                        kw["compression"] = s.compression
                        kw["compression_opts"] = s.compression_opts
                dst.create_dataset(d, data=x, **kw)
    print(f"{name}: {n} rows, {fetched / 1e6:.1f} MB of values, file "
          f"{os.path.getsize(f'{out}/{name}.h5') / 1e6:.1f} MB", flush=True)


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for name in SOURCES:
    cut(name, out)
gaia = f"{out}/gaia.h5"
local = sys.argv[2] if len(sys.argv) > 2 else None
if local:
    shutil.copyfile(local, gaia)
else:
    urllib.request.urlretrieve(GAIA, gaia)
print(f"gaia: whole file, {os.path.getsize(gaia) / 1e6:.1f} MB")
