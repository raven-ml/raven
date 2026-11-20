"""main.exe's three commands written with AION-1's own code, as its tutorials run it
(PolymathicAI/AION at db7fba6d, notebooks/Tutorial.ipynb and StellarTutorial.ipynb), for
whole.sh's whole commands.

    python upstream.py embed|posterior|generate FILE --rows N --batch B --device D
                       [--model base|large] --out OUT

It loads the released model with AION.from_pretrained and the codecs with
CodecManager(device=D), reads the first N rows of FILE with h5py in batches of B, as
the tutorial's format_data_modalities builds them, and writes OUT, an npz file:
- embed: the tutorial file's Legacy Survey image, g, r, i, z, W1-W4 fluxes and E(B-V),
  and DESI spectrum, encoded by the codecs, AION.encode of all their tokens and the
  mean over them;
- posterior: the redshift's softmax from AION's forward on the image and g, r, i, z;
- generate: a DESI spectrum drawn from Gaia's XP coefficients by GenerationSampler,
  ROAR in 12 linear steps at temperature 1, decoded on the codec's grid.
whole.sh runs it in a venv holding torch 2.15.0 and polymathic-aion at db7fba6d.
"""
import argparse

import h5py
import numpy as np
import torch

from aion import AION
from aion.codecs import CodecManager
from aion.fourm.generate import (GenerationSampler, build_chained_generation_schedules,
                                 init_empty_target_modality, init_full_input_modality)
from aion.modalities import (DESISpectrum, GaiaXpBp, GaiaXpRp, LegacySurveyEBV,
                             LegacySurveyFluxG, LegacySurveyFluxI, LegacySurveyFluxR,
                             LegacySurveyFluxW1, LegacySurveyFluxW2, LegacySurveyFluxW3,
                             LegacySurveyFluxW4, LegacySurveyFluxZ, LegacySurveyImage, Z)

REPOS = {"base": ("polymathic-ai/aion-base", "40541618104bab0fa85c8af68daeb867a720bb8c"),
         "large": ("polymathic-ai/aion-large", "cfdb89e8a22ddbdbbc4fe65c30fa2eb6eb7f9bcd")}
DESI = "tok_spectrum_desi"

p = argparse.ArgumentParser()
p.add_argument("command", choices=["embed", "posterior", "generate"])
p.add_argument("file")
p.add_argument("--rows", type=int, required=True)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--device", default="cuda")
p.add_argument("--model", default="base")
p.add_argument("--out", required=True)
a = p.parse_args()

torch.set_num_threads(2)
torch.set_grad_enabled(False)
device = torch.device(a.device)
repo, revision = REPOS[a.model]
model = AION.from_pretrained(repo, revision=revision).to(device).eval()
codecs = CodecManager(device=device)


def t(x, dtype="float32"):
    return torch.tensor(np.asarray(x).astype(dtype), device=device)


def photometry(ls, i, j, more):
    bands = [b.decode().upper() for b in ls["image_band"][i]]
    fluxes = [LegacySurveyFluxG, LegacySurveyFluxR, LegacySurveyFluxI, LegacySurveyFluxZ]
    names = ["FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z"]
    if more:
        fluxes += [LegacySurveyFluxW1, LegacySurveyFluxW2, LegacySurveyFluxW3,
                   LegacySurveyFluxW4, LegacySurveyEBV]
        names += ["FLUX_W1", "FLUX_W2", "FLUX_W3", "FLUX_W4", "EBV"]
    return [LegacySurveyImage(flux=t(ls["image_array"][i:j]), bands=bands)] + \
        [m(value=t(ls[n][i:j])) for m, n in zip(fluxes, names)]


def batches():
    for i in range(0, a.rows, a.batch):
        yield i, min(i + a.batch, a.rows)


out = []
with h5py.File(a.file, "r") as f:
    if a.command == "embed":
        ls, desi = f["legacysurvey"], f["desi"]
        for i, j in batches():
            spectrum = DESISpectrum(
                flux=t(desi["spectrum_flux"][i:j]), ivar=t(desi["spectrum_ivar"][i:j]),
                mask=t(desi["spectrum_mask"][i:j], "bool"),
                wavelength=t(desi["spectrum_lambda"][i:j]))
            tokens = codecs.encode(*photometry(ls, i, j, True), spectrum)
            n = sum(v.shape[1] if v.dim() == 2 else 1 for v in tokens.values())
            out.append(model.encode(tokens, num_encoder_tokens=n).mean(dim=1).cpu().numpy())
    elif a.command == "posterior":
        ls = f["legacysurvey"]
        for i, j in batches():
            logits = model(codecs.encode(*photometry(ls, i, j, False)), target_modality=Z)
            out.append(torch.softmax(logits["tok_z"], dim=-1).cpu().numpy())
    else:
        info = model.modality_info
        schedule = build_chained_generation_schedules(
            cond_domains=["tok_xp_bp", "tok_xp_rp"], target_domains=[DESI],
            tokens_per_target=[273], autoregression_schemes=["roar"], decoding_steps=[12],
            token_decoding_schedules=["linear"], temps=[1.0], temp_schedules=["constant"],
            cfg_scales=[1.0], cfg_schedules=["constant"], modality_info=info)
        sampler = GenerationSampler(model)
        coeff = f["coeff"]
        for i, j in batches():
            x = coeff[i:j]
            tokens = codecs.encode(GaiaXpBp(value=t(x[:, :55])), GaiaXpRp(value=t(x[:, 55:])))
            d = {}
            for k, v in tokens.items():
                d[k] = {"tensor": v}
                d = init_full_input_modality(d, info, k, device)
            d = init_empty_target_modality(d, info, DESI, j - i, 273, device)
            drawn = sampler.generate(d, schedule, seed=i)[DESI]["tensor"]
            out.append(codecs.decode({DESI: drawn}, DESISpectrum).flux.cpu().numpy())
np.savez(a.out, out=np.concatenate(out))
print(f"{a.command}: {sum(len(o) for o in out)} objects", flush=True)
