# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==2.5.5",
#   "h5py==3.16.0",
#   "fsspec==2026.9.0",
#   "aiohttp==3.14.5",
#   "requests==2.34.2",
# ]
# ///
"""Fetch the Multimodal Universe's rows whole.sh runs the commands on.

    uv run apps/aion/bench/fetch.py DIR [ROWS]

Writes DIR/tutorial.h5, the first ROWS rows (256 by default) of AION-1's tutorial file
provabgs_desi_ls.hdf5, its groups legacysurvey and desi, with the datasets the commands
read, read by HTTP Range so that only those rows' bytes are fetched; and DIR/gaia.h5,
Gaia's healpix 1631 file whole (10,870 stars, 16 MB).
"""
import os, sys, urllib.request

import fsspec, h5py

DATA = "https://users.flatironinstitute.org/~polymathic/data"
TUTORIAL = f"{DATA}/provabgs_desi_ls.hdf5"
GAIA = f"{DATA}/MultimodalUniverse/v1/gaia/gaia/healpix=1631/001-of-001.hdf5"
GROUPS = {
    "legacysurvey": ["object_id", "ra", "dec", "image_array", "image_band", "EBV"]
                    + [f"FLUX_{b}" for b in "GRIZ"] + [f"FLUX_W{i}" for i in range(1, 5)],
    "desi": ["object_id", "ra", "dec", "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
             "spectrum_mask"],
}

out = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
os.makedirs(out, exist_ok=True)
fs = fsspec.filesystem("http")
with fs.open(TUTORIAL, "rb", block_size=1 << 20, cache_type="readahead") as fo, \
        h5py.File(fo, "r") as f, h5py.File(os.path.join(out, "tutorial.h5"), "w") as g:
    for group, names in GROUPS.items():
        dst = g.create_group(group)
        for d in names:
            dst.create_dataset(d, data=f[group][d][:n])
gaia = os.path.join(out, "gaia.h5")
if not os.path.exists(gaia):
    urllib.request.urlretrieve(GAIA, gaia)
for name in ("tutorial.h5", "gaia.h5"):
    print(f"{name}: {os.path.getsize(os.path.join(out, name)) / 1e6:.1f} MB", flush=True)
