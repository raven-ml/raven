# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==2.5.5",
#   "safetensors==0.8.0",
#   "h5py==3.16.0",
# ]
# ///
"""Synthetic Multimodal Universe files for test_survey and test/bin/aion.t.

    uv run test/gen/surveys.py test/mmu

DIR is test/mmu. Each file has the layout of one MMU survey's HDF5 file
(MultimodalUniverse v1 at Flatiron, read 2026-11-19: dataset names, dtypes, string
widths, chunking and filters), a few rows each, with values made up:

  legacysurvey.h5  3 rows: image_array [3;4;160;160] float32, chunks (2,1,20,40), LZF;
                   image_band S5 lower case; FLUX_*, SHAPE_*, EBV float32; object_id S16
  hsc.h5           2 rows: image_array [2;5;160;160], chunks (2,2,40,40), deflate 4;
                   a_*, *_cmodel_mag, i_sdssshape_shape* float32; object_id int64
  desi.h5          3 rows: spectrum_* [3;64], Z float64, and FLUX_G, DESI's copy of a
                   Legacy Survey column that no DESI modality reads; object_id int64
  sdss.h5          4 rows: spectrum_* [4;40], each row's wavelengths ending in a run of
                   -1 (MMU's padding to the file's longest row) of 0, 1, 10 and 39
                   pixels; Z float32; object_id S22, left-padded with spaces
  gaia.h5          4 rows: coeff [4;110] float32 (BP 0-54, RP 55-109),
                   phot_*_mean_flux and parallax float64; source_id int64, no object_id
  grouped.h5       the tutorial file's shape: groups /legacysurvey (no SHAPE_*), /desi
                   (no Z) and /provabgs_desi_edr_sv3 (Z_HP), 2 rows aligned
  sdss_gap.h5      2 rows, row 1's -1 run followed by a real pixel
  ls_bands.h5      2 rows, row 1's bands in another order

Pixels: pixel (b, y, x) of row r of a file is 1000 r + 10 (b + 1) + x mod 16, so a band
or a row read in the wrong place shows. ra and dec are float64 degrees: Legacy Survey
row k at (150 + 0.01 k, 2); Gaia rows near Legacy Survey rows 1 and 0 (0.2" and 0.5"
away), one far, one 1.5" from row 2; DESI rows 0.1" from each Legacy Survey row.
Non-finite values: legacysurvey FLUX_G row 1 is +inf, SHAPE_R row 2 NaN; gaia parallax
row 3 NaN and coeff[2, 7] NaN. Float64 columns hold values whose float32 rounding
differs from them.

Writes DIR/expected.safetensors: for each file's modality, the values Survey must give
each row as float32 (MMU's builders declare every float column float32), named
"<file>.<token key>", or "<file>/<group>.<token key>" in a group; a spectrum's flux as
"<file>.spectrum_flux" and its rows' lengths as "<file>.lengths".
Each modality's dataset is upstream's modality name for it (aion/modalities.py), the
column names MMU's files use.
"""
import os, sys

import h5py, numpy as np
from safetensors.numpy import save_file

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
expected = {}
ARCSEC = 1 / 3600


def pixels(rows, bands, side):
    r = np.arange(rows)[:, None, None, None]
    b = np.arange(bands)[None, :, None, None]
    x = np.arange(side)[None, None, None, :]
    return np.broadcast_to(1000 * r + 10 * (b + 1) + x % 16, (rows, bands, side, side)).astype(np.float32)


def strings(values, width, pad_left=False):
    return np.array([(v.rjust(width) if pad_left else v).encode() for v in values], dtype=f"S{width}")


def f64(base, n):
    # float64 values whose float32 rounding differs from them
    return base + np.arange(n) * 1.25 + 2.0 ** -30


def key_of(file, group, key):
    return f"{file}/{group}.{key}" if group else f"{file}.{key}"


def expect(file, group, key, values):
    expected[key_of(file, group, key)] = np.ascontiguousarray(np.asarray(values, dtype=np.float32))


LS_POS = [(150.0 + 0.01 * k, 2.0) for k in range(3)]


def ls_columns(f, n, shapes=True, file="legacysurvey", group="", g=None):
    g = f if g is None else g
    cols = ["FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z", "FLUX_W1", "FLUX_W2", "FLUX_W3", "FLUX_W4"]
    keys = ["tok_flux_g", "tok_flux_r", "tok_flux_i", "tok_flux_z",
            "tok_flux_w1", "tok_flux_w2", "tok_flux_w3", "tok_flux_w4"]
    if shapes:
        cols += ["SHAPE_R", "SHAPE_E1", "SHAPE_E2"]
        keys += ["tok_shape_r", "tok_shape_e1", "tok_shape_e2"]
    cols.append("EBV"); keys.append("tok_ebv")
    for i, (c, k) in enumerate(zip(cols, keys)):
        v = (np.arange(n) + 1 + 100 * i).astype(np.float32) / 7
        if c == "FLUX_G" and n > 1:
            v[1] = np.inf
        if c == "SHAPE_R" and n > 2:
            v[2] = np.nan
        g.create_dataset(c, data=v)
        expect(file, group, k, v)


# legacysurvey.h5
with h5py.File(f"{out}/legacysurvey.h5", "w") as f:
    n = 3
    f.create_dataset("object_id", data=strings([f"1500p020-{k}" for k in range(n)], 16))
    f.create_dataset("ra", data=np.array([p[0] for p in LS_POS]))
    f.create_dataset("dec", data=np.array([p[1] for p in LS_POS]))
    f.create_dataset("image_array", data=pixels(n, 4, 160), chunks=(2, 1, 20, 40), compression="lzf")
    f.create_dataset("image_band", data=np.array([strings(["des-g", "des-r", "des-i", "des-z"], 5)] * n))
    ls_columns(f, n)

# hsc.h5
HSC = ["a_g", "a_r", "a_i", "a_z", "a_y", "g_cmodel_mag", "r_cmodel_mag", "i_cmodel_mag",
       "z_cmodel_mag", "y_cmodel_mag", "i_sdssshape_shape11", "i_sdssshape_shape22",
       "i_sdssshape_shape12"]
HSC_KEYS = ["tok_a_g", "tok_a_r", "tok_a_i", "tok_a_z", "tok_a_y", "tok_mag_g", "tok_mag_r",
            "tok_mag_i", "tok_mag_z", "tok_mag_y", "tok_shape11", "tok_shape22", "tok_shape12"]
with h5py.File(f"{out}/hsc.h5", "w") as f:
    n = 2
    f.create_dataset("object_id", data=np.array([70000000001, 70000000002], dtype=np.int64))
    f.create_dataset("ra", data=np.array([34.0, 34.001]))
    f.create_dataset("dec", data=np.array([-5.0, -5.0]))
    f.create_dataset("image_array", data=pixels(n, 5, 160), chunks=(2, 2, 40, 40),
                     compression="gzip", compression_opts=4)
    f.create_dataset("image_band", data=np.array(
        [strings(["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"], 5)] * n))
    for i, (c, k) in enumerate(zip(HSC, HSC_KEYS)):
        v = (np.arange(n) + 3 + 10 * i).astype(np.float32) / 3
        f.create_dataset(c, data=v)
        expect("hsc", "", k, v)


def spectra(g, n, m, pads, file, group, key_z, z):
    lam = np.tile(3600 + 2.5 * np.arange(m, dtype=np.float32), (n, 1))
    for r, p in enumerate(pads):
        if p:
            lam[r, m - p:] = -1
    flux = (np.arange(n * m, dtype=np.float32).reshape(n, m) % 13) + 1
    ivar = np.full((n, m), 4, dtype=np.float32)
    mask = np.zeros((n, m), dtype=bool)
    mask[:, 3] = True
    g.create_dataset("spectrum_flux", data=flux)
    g.create_dataset("spectrum_ivar", data=ivar)
    g.create_dataset("spectrum_lambda", data=lam)
    g.create_dataset("spectrum_mask", data=mask)
    expected[key_of(file, group, "lengths")] = np.array([m - p for p in pads], dtype=np.int64)
    expect(file, group, "spectrum_flux", flux)
    if z is not None:
        g.create_dataset("Z", data=z)
        expect(file, group, key_z, z)


def offset(pos, east_arcsec):
    ra, dec = pos
    return (ra + east_arcsec * ARCSEC / np.cos(np.radians(dec)), dec)


# desi.h5
with h5py.File(f"{out}/desi.h5", "w") as f:
    n = 3
    pos = [offset(p, 0.1) for p in LS_POS]
    f.create_dataset("object_id", data=np.array([39627000 + k for k in range(n)], dtype=np.int64))
    f.create_dataset("ra", data=np.array([p[0] for p in pos]))
    f.create_dataset("dec", data=np.array([p[1] for p in pos]))
    spectra(f, n, 64, [0, 0, 0], "desi", "", "tok_z", f64(0.5, n))
    f.create_dataset("FLUX_G", data=np.ones(n, dtype=np.float32))

# sdss.h5
with h5py.File(f"{out}/sdss.h5", "w") as f:
    n = 4
    f.create_dataset("object_id", data=strings([str(1361217980037359616 + k) for k in range(n)], 22, pad_left=True))
    f.create_dataset("ra", data=np.array([10.0 + 0.01 * k for k in range(n)]))
    f.create_dataset("dec", data=np.full(n, 30.0))
    spectra(f, n, 40, [0, 1, 10, 39], "sdss", "", "tok_z",
            (np.arange(n, dtype=np.float32) + 1) / 9)

# sdss_gap.h5
with h5py.File(f"{out}/sdss_gap.h5", "w") as f:
    n = 2
    f.create_dataset("object_id", data=strings(["1", "2"], 22, pad_left=True))
    f.create_dataset("ra", data=np.array([10.0, 10.01]))
    f.create_dataset("dec", data=np.full(n, 30.0))
    lam = np.tile(3600 + 2.5 * np.arange(8, dtype=np.float32), (n, 1))
    lam[1, 4:6] = -1
    for name, v in [("spectrum_flux", np.ones((n, 8), np.float32)),
                    ("spectrum_ivar", np.ones((n, 8), np.float32)),
                    ("spectrum_lambda", lam), ("spectrum_mask", np.zeros((n, 8), bool))]:
        f.create_dataset(name, data=v)
    f.create_dataset("Z", data=np.zeros(n, np.float32))

# gaia.h5
with h5py.File(f"{out}/gaia.h5", "w") as f:
    pos = [offset(LS_POS[1], 0.2), offset(LS_POS[0], -0.5), (151.0, 2.0), offset(LS_POS[2], 1.5)]
    n = len(pos)
    f.create_dataset("source_id", data=np.array([4295806720 + k for k in range(n)], dtype=np.int64))
    ra = np.array([p[0] for p in pos])
    dec = np.array([p[1] for p in pos])
    f.create_dataset("ra", data=ra)
    f.create_dataset("dec", data=dec)
    coeff = (np.arange(n * 110, dtype=np.float32).reshape(n, 110) - 200) / 11
    coeff[2, 7] = np.nan
    f.create_dataset("coeff", data=coeff)
    expect("gaia", "", "tok_xp_bp", coeff[:, :55])
    expect("gaia", "", "tok_xp_rp", coeff[:, 55:])
    for c, k, base in [("phot_g_mean_flux", "tok_flux_g_gaia", 1234.5),
                       ("phot_bp_mean_flux", "tok_flux_bp_gaia", 777.0),
                       ("phot_rp_mean_flux", "tok_flux_rp_gaia", 901.25)]:
        v = f64(base, n)
        f.create_dataset(c, data=v)
        expect("gaia", "", k, v)
    par = f64(0.75, n)
    par[3] = np.nan
    f.create_dataset("parallax", data=par)
    expect("gaia", "", "tok_parallax", par)
    expect("gaia", "", "tok_ra", ra)
    expect("gaia", "", "tok_dec", dec)

# grouped.h5
with h5py.File(f"{out}/grouped.h5", "w") as f:
    n = 2
    ls = f.create_group("legacysurvey")
    ls.create_dataset("object_id", data=strings([f"1500p020-{k}" for k in range(n)], 16))
    ls.create_dataset("ra", data=np.array([p[0] for p in LS_POS[:n]]))
    ls.create_dataset("dec", data=np.array([p[1] for p in LS_POS[:n]]))
    ls_columns(f, n, shapes=False, file="grouped", group="legacysurvey", g=ls)
    desi = f.create_group("desi")
    pos = [offset(p, 0.3) for p in LS_POS[:n]]
    desi.create_dataset("object_id", data=np.array([39627000, 39627001], dtype=np.int64))
    desi.create_dataset("ra", data=np.array([p[0] for p in pos]))
    desi.create_dataset("dec", data=np.array([p[1] for p in pos]))
    spectra(desi, n, 16, [0, 0], "grouped", "desi", "tok_z", None)
    pz = f.create_group("provabgs_desi_edr_sv3")
    pz.create_dataset("Z_HP", data=np.array([0.1, 0.2]))

# ls_bands.h5
with h5py.File(f"{out}/ls_bands.h5", "w") as f:
    n = 2
    f.create_dataset("object_id", data=strings(["a", "b"], 16))
    f.create_dataset("ra", data=np.array([150.0, 150.01]))
    f.create_dataset("dec", data=np.array([2.0, 2.0]))
    f.create_dataset("image_array", data=pixels(n, 4, 96), chunks=(1, 1, 48, 48), compression="lzf")
    f.create_dataset("image_band", data=np.array(
        [strings(["des-g", "des-r", "des-i", "des-z"], 5), strings(["des-r", "des-g", "des-i", "des-z"], 5)]))

meta = {"script": "apps/aion/test/gen/surveys.py", "h5py": h5py.__version__,
        "numpy": np.__version__}
save_file(expected, f"{out}/expected.safetensors", metadata=meta)
for name in sorted(os.listdir(out)):
    print(name, os.path.getsize(f"{out}/{name}"))
