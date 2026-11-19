# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.14.1",
#   "numpy==2.5.5",
#   "safetensors==0.8.0",
#   "huggingface-hub==2.4.0",
#   "h5py==3.16.0",
#   "polymathic-aion @ git+https://github.com/PolymathicAI/AION@db7fba6df53b605e3a1a58b7f52fc33bfaaa96d3",
# ]
#
# [tool.uv]
# sources = { torch = { index = "pytorch-cpu" } }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Upstream's tokens of the real rows mmu_cut.py cut, for mmu_parity.exe.

    uv run test/gen/mmu_tokens.py DIR

DIR holds mmu_cut.py's files. Each modality of each file is built as AION's tutorial
builds it (Tutorial.ipynb cell 26): the column read with h5py and cast to float32, the
image bands named from image_band in upper case, Gaia's XP spectra from coeff's columns
0-54 and 55-109; then upstream's CodecManager (polymathic-aion at db7fba6d, aion-base's
codecs at 40541618, torch on the CPU) encodes it. Scalar classes come from upstream's
ScalarModalities, keyed by the column's name. SDSS rows are cut to the pixels before
their run of -1 wavelengths and encoded one at a time. Gaia's whole file is cut to
its first 64 rows.

Writes DIR/tokens.safetensors: "<file>[/<group>].<token key>" int64, the tokens of the
file's first rows (all of a cut, 64 of Gaia), for every modality of the file.
"""
import os, sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import h5py, numpy as np, torch
from safetensors.numpy import save_file

from aion import modalities as M
from aion.codecs import CodecManager

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

d = sys.argv[1]
cm = CodecManager(device="cpu")
out = {}

LS = [f"FLUX_{b}" for b in "GRIZ"] + [f"FLUX_W{i}" for i in range(1, 5)] + \
     ["SHAPE_R", "SHAPE_E1", "SHAPE_E2", "EBV"]
HSC = [f"a_{b}" for b in "grizy"] + [f"{b}_cmodel_mag" for b in "grizy"] + \
      ["i_sdssshape_shape11", "i_sdssshape_shape22", "i_sdssshape_shape12"]
GAIA = ["phot_g_mean_flux", "phot_bp_mean_flux", "phot_rp_mean_flux", "parallax", "ra", "dec"]


def f32(x):
    return torch.tensor(np.asarray(x, dtype=np.float32))


def put(prefix, mod):
    for k, v in cm.encode(mod).items():
        out[f"{prefix}.{k}"] = np.ascontiguousarray(v.cpu().numpy().astype(np.int64))
        print(f"  {prefix}.{k} {tuple(v.shape)}", flush=True)


def image(prefix, g, cls, n):
    bands = g["image_band"][:n]
    names = [b.decode().strip().upper() for b in bands[0]]
    assert all([b.decode().strip().upper() for b in r] == names for r in bands)
    put(prefix, cls(flux=f32(g["image_array"][:n]), bands=names))


def scalars(prefix, g, cols, n):
    for c in cols:
        if c in g:
            put(prefix, M.ScalarModalities[c](value=f32(g[c][:n])))


def spectra(prefix, g, cls, n, trim):
    lam = g["spectrum_lambda"][:n]
    fields = [g[k][:n] for k in ("spectrum_flux", "spectrum_ivar", "spectrum_mask")]
    if not trim:
        put(prefix, cls(flux=f32(fields[0]), ivar=f32(fields[1]),
                        mask=torch.tensor(fields[2].astype(bool)), wavelength=f32(lam)))
        return
    rows = []
    for i in range(n):
        neg = lam[i] < 0
        k = len(lam[i]) - int(neg.sum())
        assert neg[k:].all() and not neg[:k].any(), f"{prefix}: row {i}: -1 not a suffix"
        m = cls(flux=f32(fields[0][i:i + 1, :k]), ivar=f32(fields[1][i:i + 1, :k]),
                mask=torch.tensor(fields[2][i:i + 1, :k].astype(bool)),
                wavelength=f32(lam[i:i + 1, :k]))
        rows.append(cm.encode(m)[cls.token_key].cpu().numpy())
    out[f"{prefix}.{cls.token_key}"] = np.concatenate(rows).astype(np.int64)
    print(f"  {prefix}.{cls.token_key} {out[f'{prefix}.{cls.token_key}'].shape}", flush=True)


def rows(g):
    return g["object_id"].shape[0] if "object_id" in g else g["source_id"].shape[0]


with h5py.File(f"{d}/tutorial.h5") as f:
    n = rows(f["legacysurvey"])
    image("tutorial/legacysurvey", f["legacysurvey"], M.LegacySurveyImage, n)
    scalars("tutorial/legacysurvey", f["legacysurvey"], LS, n)
    spectra("tutorial/desi", f["desi"], M.DESISpectrum, n, trim=False)
with h5py.File(f"{d}/ls.h5") as f:
    n = rows(f)
    image("ls", f, M.LegacySurveyImage, n)
    scalars("ls", f, LS, n)
with h5py.File(f"{d}/hsc.h5") as f:
    n = rows(f)
    image("hsc", f, M.HSCImage, n)
    scalars("hsc", f, HSC, n)
with h5py.File(f"{d}/desi.h5") as f:
    n = rows(f)
    spectra("desi", f, M.DESISpectrum, n, trim=False)
    scalars("desi", f, ["Z"], n)
with h5py.File(f"{d}/sdss.h5") as f:
    n = rows(f)
    spectra("sdss", f, M.SDSSSpectrum, n, trim=True)
    scalars("sdss", f, ["Z"], n)
with h5py.File(f"{d}/gaia.h5") as f:
    n = 64
    scalars("gaia", f, GAIA, n)
    coeff = f["coeff"][:n]
    put("gaia", M.GaiaXpBp(value=f32(coeff[:, :55])))
    put("gaia", M.GaiaXpRp(value=f32(coeff[:, 55:])))

meta = {"script": "apps/aion/test/gen/mmu_tokens.py", "torch": torch.__version__}
save_file(out, f"{d}/tokens.safetensors", metadata=meta)
