# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.14.1",
#   "numpy==2.5.5",
#   "safetensors==0.8.0",
#   "huggingface-hub==2.4.0",
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
"""PyTorch reference run for the AION-1 parity ladder.

Runs upstream's codecs and aion-base (polymathic-aion at db7fba6d, torch CPU) at one
precision and one thread count, and writes every quantity the ladder compares:

  goldens.safetensors  every codec on upstream's own test inputs (tests/test_data)
  codecs.safetensors   every codec on the parity subset: input as fed, the value the
                       quantizer decides on, tokens, decoded values
  model.safetensors    AION.encode embeddings; decoder logits and argmax for Z (one
                       token), a DESI spectrum (273) and an LS image (576), each from
                       upstream's forward and from the patched forward
  meta-<parts>.json    versions, settings, timings, failures, per invocation

    uv run test/gen/reference.py --precision f64 --threads 2 \
        --subset SUBSET --goldens GOLDENS --out REF64

The model part takes their input tokens from --tokens (the f64 run's codecs.safetensors)
when given, so that a legitimate codec flip does not leak into the model comparison;
otherwise from this run's own tokens.

Precision. Every input is rounded to float32 once (the data are float32 values; a few
catalog columns are stored float64 and are rounded here), then cast to the run's dtype.
In the f64 run the model and codecs are cast with .to(float64) and torch's default
dtype is float64, so constants built inside forward passes (FSQ's bounds, linspaces)
are float64 too. Weights and stored buffers are float32 values cast exactly.

Decoder mask. AION.embed_targets passes an all-zero compressed decoder attention mask, so
adapt_decoder_attention_mask masks every entry and decoder self-attention is uniform.
The patched forward puts the target count at the first decoder token of each target
modality, as 4M's training sampler does, so the targets of a modality attend to each
other. The patch lives here, never in the clone.

Batching. Codecs run in chunks of --codec-batch objects and the model in chunks of
--model-batch, the same in every run, so that run-to-run differences come from the
thread count and the precision only.
"""
import argparse, json, os, random, time, types
from collections import OrderedDict

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--subset", required=True)
p.add_argument("--goldens", required=True, help="directory of goldens as safetensors")
p.add_argument("--precision", choices=["f32", "f64"], default="f32")
p.add_argument("--threads", type=int, default=1)
p.add_argument("--parts", default="goldens,codecs,model")
p.add_argument("--tokens", help="codecs.safetensors whose tokens drive the model part")
p.add_argument("--codec-batch", type=int, default=8)
p.add_argument("--model-batch", type=int, default=2)
p.add_argument("--n-model", type=int, default=8, help="objects per source in the model part")
p.add_argument("--n-image-target", type=int, default=2, help="objects for the LS-image target")
p.add_argument("--out", required=True)
a = p.parse_args()

torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
random.seed(0)
torch.manual_seed(0)
DT = torch.float64 if a.precision == "f64" else torch.float32
if DT == torch.float64:
    torch.set_default_dtype(torch.float64)

from aion.codecs import config as codec_config                        # noqa: E402
from aion.codecs.config import HF_REPO_ID, MODALITY_CODEC_MAPPING      # noqa: E402
from aion.codecs.quantizers import FiniteScalarQuantizer                 # noqa: E402
from aion.codecs.quantizers.scalar import (                            # noqa: E402
    ScalarLogReservoirQuantizer, ScalarCompressedReservoirQuantizer,
    MultiScalarCompressedReservoirQuantizer)
from aion import modalities as M                                       # noqa: E402
from aion.model import AION                                            # noqa: E402

os.makedirs(a.out, exist_ok=True)
meta = {"precision": a.precision, "threads": a.threads, "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "default_dtype": str(torch.get_default_dtype()),
        "args": vars(a), "timings_s": {}, "notes": [], "failures": []}


def np_of(t):
    x = t.detach().cpu().numpy()
    return np.ascontiguousarray(x.astype(np.uint8) if x.dtype == np.bool_ else x)


def f(x):
    """Data value -> float32 once -> run dtype, always a fresh copy: upstream's image
    codec clamps and rescales its input in place, and torch.as_tensor would share
    memory with the loaded array, so a second encode would see the first one's writes."""
    return torch.tensor(np.asarray(x, dtype=np.float32)).to(DT)


def cat(parts):
    if isinstance(parts[0], dict):
        return {k: cat([q[k] for q in parts]) for k in parts[0]}
    return torch.cat(parts, dim=0)


def chunks(n, bs):
    return [(lo, min(n, lo + bs)) for lo in range(0, n, bs)]


# Codecs

_codecs = {}


def codec_of(mod):
    if mod not in _codecs:
        c = MODALITY_CODEC_MAPPING[mod].from_pretrained(HF_REPO_ID, modality=mod)
        _codecs[mod] = c.to(DT).eval()
    return _codecs[mod]


def decided_value(codec, x):
    """The value each quantizer compares against its edges, before rounding."""
    q = codec.quantizer
    if isinstance(codec, codec_config.ImageCodec) or isinstance(q, FiniteScalarQuantizer):
        z = codec._encode(x)
        return {"pre": z, "bounded": q._bound(z.moveaxis(1, -1))}
    if isinstance(codec, codec_config.SpectrumCodec):
        seen = {}
        nq = codec.normalization_quantizer
        orig = nq.quantize
        nq.quantize = lambda z: (seen.__setitem__("norm_pre", z.clone()), orig(z))[1]
        try:
            h, norm = codec._encode(x)
        finally:
            del nq.quantize
        return {"pre": h, "norm": norm, "norm_pre": seen["norm_pre"]}
    if isinstance(codec, codec_config.CatalogCodec):
        return {}
    v = x.value
    if isinstance(q, ScalarLogReservoirQuantizer):
        return {"pre": q._log_and_apply_min(v)}
    if isinstance(q, ScalarCompressedReservoirQuantizer):
        return {"pre": q.compress(v)}
    if isinstance(q, MultiScalarCompressedReservoirQuantizer):
        return {"pre": torch.stack([qq.compress(v[:, i]) for i, qq in enumerate(q.quantizers)], 1)}
    return {"pre": v}


def decoded_fields(d):
    if isinstance(d, M.Image):
        return {"flux": d.flux}
    if isinstance(d, M.Spectrum):
        return {"flux": d.flux, "mask": d.mask, "wavelength": d.wavelength}
    if isinstance(d, M.LegacySurveyCatalog):
        return {k: getattr(d, k) for k in ["X", "Y", "SHAPE_E1", "SHAPE_E2", "SHAPE_R"]}
    if isinstance(d, M.LegacySurveySegmentationMap):
        return {"field": d.field}
    return {"value": d.value}


def run_codec(out, prefix, mod, make, n, decode_kw=None, bs=None):
    """make(lo, hi) builds a fresh modality instance (encode mutates its input)."""
    codec = codec_of(mod)
    bs = bs or a.codec_batch
    pre, tok, dec = [], [], []
    for lo, hi in chunks(n, bs):
        pre.append(decided_value(codec, make(lo, hi)))
        t = codec.encode(make(lo, hi))
        tok.append(t)
        dec.append(decoded_fields(codec.decode(t, **(decode_kw or {}))))
    for k, v in (cat(pre).items() if pre[0] else []):
        out[f"{prefix}.{k}"] = np_of(v)
    out[f"{prefix}.tokens"] = np_of(cat(tok))
    for k, v in cat(dec).items():
        out[f"{prefix}.decoded.{k}"] = np_of(v)


def timed(name, fn):
    t0 = time.time()
    try:
        fn()
    except Exception as e:  # recorded, the run goes on
        meta["failures"].append(f"{name}: {type(e).__name__}: {e}")
        print("FAILED", name, type(e).__name__, e, flush=True)
    meta["timings_s"][name] = round(time.time() - t0, 2)
    print(f"{name}: {meta['timings_s'][name]} s", flush=True)


LS_BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]
HSC_BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]

# Goldens: upstream's own inputs (tests/codecs/*.py)

SCALAR_GOLDENS = {
    "FLUX_G": M.LegacySurveyFluxG, "FLUX_R": M.LegacySurveyFluxR, "FLUX_I": M.LegacySurveyFluxI,
    "FLUX_Z": M.LegacySurveyFluxZ, "FLUX_W1": M.LegacySurveyFluxW1, "FLUX_W2": M.LegacySurveyFluxW2,
    "FLUX_W3": M.LegacySurveyFluxW3, "FLUX_W4": M.LegacySurveyFluxW4, "SHAPE_R": M.LegacySurveyShapeR,
    "SHAPE_E1": M.LegacySurveyShapeE1, "SHAPE_E2": M.LegacySurveyShapeE2, "EBV": M.LegacySurveyEBV,
    "g_cmodel_mag": M.HSCMagG, "r_cmodel_mag": M.HSCMagR, "i_cmodel_mag": M.HSCMagI,
    "z_cmodel_mag": M.HSCMagZ, "y_cmodel_mag": M.HSCMagY, "i_sdssshape_shape11": M.HSCShape11,
    "i_sdssshape_shape22": M.HSCShape22, "i_sdssshape_shape12": M.HSCShape12,
    "a_g": M.HSCAG, "a_r": M.HSCAR, "a_i": M.HSCAI, "a_z": M.HSCAZ, "a_y": M.HSCAY,
    "phot_g_mean_flux": M.GaiaFluxG, "phot_bp_mean_flux": M.GaiaFluxBp,
    "phot_rp_mean_flux": M.GaiaFluxRp, "parallax": M.GaiaParallax, "ra": M.Ra, "dec": M.Dec,
    "bp_coefficients": M.GaiaXpBp, "rp_coefficients": M.GaiaXpRp, "Z": M.Z,
}


def golden(name):
    return load_file(os.path.join(a.goldens, name + ".safetensors"))


def part_goldens():
    out = OrderedDict()
    for name, mod in SCALAR_GOLDENS.items():
        x = golden(f"{name}_codec_input_batch")["value"]
        timed(f"goldens.{name}", lambda: run_codec(
            out, f"golden.{name}", mod, lambda lo, hi: mod(value=f(x[lo:hi])), len(x), bs=len(x)))
    img = golden("image_codec_input_batch")["image.array"][:, 5:]
    timed("goldens.image", lambda: run_codec(
        out, "golden.image", M.Image, lambda lo, hi: M.Image(flux=f(img[lo:hi]), bands=LS_BANDS),
        len(img), decode_kw={"bands": LS_BANDS}))
    s = golden("SPECTRUM_input_batch")
    timed("goldens.spectrum", lambda: run_codec(
        out, "golden.spectrum", M.Spectrum,
        lambda lo, hi: M.Spectrum(flux=f(s["spectrum.flux"][lo:hi]), ivar=f(s["spectrum.ivar"][lo:hi]),
                                  mask=torch.as_tensor(s["spectrum.mask"][lo:hi]).bool(),
                                  wavelength=f(s["spectrum.lambda"][lo:hi])),
        len(s["spectrum.flux"])))
    c = golden("catalog_codec_input_batch")
    timed("goldens.catalog", lambda: run_codec(
        out, "golden.catalog", M.LegacySurveyCatalog,
        lambda lo, hi: M.LegacySurveyCatalog(
            X=torch.as_tensor(c["X"][lo:hi]), Y=torch.as_tensor(c["Y"][lo:hi]),
            SHAPE_E1=f(c["SHAPE_E1"][lo:hi]), SHAPE_E2=f(c["SHAPE_E2"][lo:hi]),
            SHAPE_R=f(c["SHAPE_R"][lo:hi])),
        len(c["X"]), bs=len(c["X"])))
    sf = golden("scalar-field_codec_input_batch")["value"]
    timed("goldens.segmap", lambda: run_codec(
        out, "golden.segmap", M.LegacySurveySegmentationMap,
        lambda lo, hi: M.LegacySurveySegmentationMap(field=f(sf[lo:hi])), len(sf)))
    save_file(out, os.path.join(a.out, "goldens.safetensors"))


# Codecs on the subset

def subset(name):
    return load_file(os.path.join(a.subset, name + ".safetensors"))


def spectrum_maker(cls, d, pre=""):
    return lambda lo, hi: cls(
        flux=f(d[pre + "spectrum_flux"][lo:hi]), ivar=f(d[pre + "spectrum_ivar"][lo:hi]),
        mask=torch.as_tensor(d[pre + "spectrum_mask"][lo:hi]).bool(),
        wavelength=f(d[pre + "spectrum_lambda"][lo:hi]))


def scalar(out, src, mod, x):
    timed(f"codecs.{src}.{mod.token_key}", lambda: run_codec(
        out, f"{src}.{mod.token_key}", mod, lambda lo, hi: mod(value=f(x[lo:hi])), len(x), bs=len(x)))


LS_FLUX_MODS = [("FLUX_G", M.LegacySurveyFluxG), ("FLUX_R", M.LegacySurveyFluxR),
                ("FLUX_I", M.LegacySurveyFluxI), ("FLUX_Z", M.LegacySurveyFluxZ),
                ("FLUX_W1", M.LegacySurveyFluxW1), ("FLUX_W2", M.LegacySurveyFluxW2),
                ("FLUX_W3", M.LegacySurveyFluxW3), ("FLUX_W4", M.LegacySurveyFluxW4),
                ("EBV", M.LegacySurveyEBV)]


def part_codecs():
    out = OrderedDict()

    d = subset("ls_desi")
    img = d["legacysurvey.image_array"]
    timed("codecs.ls_desi.tok_image", lambda: run_codec(
        out, "ls_desi.tok_image", M.LegacySurveyImage,
        lambda lo, hi: M.LegacySurveyImage(flux=f(img[lo:hi]), bands=LS_BANDS),
        len(img), decode_kw={"bands": LS_BANDS}))
    timed("codecs.ls_desi.tok_spectrum_desi", lambda: run_codec(
        out, "ls_desi.tok_spectrum_desi", M.DESISpectrum,
        spectrum_maker(M.DESISpectrum, d, "desi."), len(img)))
    for col, mod in LS_FLUX_MODS:
        scalar(out, "ls_desi", mod, d[f"legacysurvey.{col}"])
    scalar(out, "ls_desi", M.Z, d["provabgs_desi_edr_sv3.Z_HP"])

    d = subset("ls_catalog")
    img = d["image_array"]
    timed("codecs.ls_catalog.tok_image", lambda: run_codec(
        out, "ls_catalog.tok_image", M.LegacySurveyImage,
        lambda lo, hi: M.LegacySurveyImage(flux=f(img[lo:hi]), bands=LS_BANDS),
        len(img), decode_kw={"bands": LS_BANDS}))
    for col, mod in LS_FLUX_MODS + [("SHAPE_R", M.LegacySurveyShapeR),
                                    ("SHAPE_E1", M.LegacySurveyShapeE1),
                                    ("SHAPE_E2", M.LegacySurveyShapeE2)]:
        scalar(out, "ls_catalog", mod, d[col])
    # MMU pads empty catalog slots with zeros, never the codec's 9999; upstream's own
    # golden input is padded the same way, so the slots are fed as stored.
    cat_in = {k: d[f"catalog_{k}"] for k in ["X", "Y", "SHAPE_E1", "SHAPE_E2", "SHAPE_R"]}
    timed("codecs.ls_catalog.catalog", lambda: run_codec(
        out, "ls_catalog.catalog", M.LegacySurveyCatalog,
        lambda lo, hi: M.LegacySurveyCatalog(
            X=torch.as_tensor(cat_in["X"][lo:hi]), Y=torch.as_tensor(cat_in["Y"][lo:hi]),
            SHAPE_E1=f(cat_in["SHAPE_E1"][lo:hi]), SHAPE_E2=f(cat_in["SHAPE_E2"][lo:hi]),
            SHAPE_R=f(cat_in["SHAPE_R"][lo:hi])),
        len(img), bs=len(img)))

    d = subset("hsc")
    img = d["image_array"]
    timed("codecs.hsc.tok_image_hsc", lambda: run_codec(
        out, "hsc.tok_image_hsc", M.HSCImage,
        lambda lo, hi: M.HSCImage(flux=f(img[lo:hi]), bands=HSC_BANDS),
        len(img), decode_kw={"bands": HSC_BANDS}))
    for col, mod in [("a_g", M.HSCAG), ("a_r", M.HSCAR), ("a_i", M.HSCAI), ("a_z", M.HSCAZ),
                     ("a_y", M.HSCAY), ("g_cmodel_mag", M.HSCMagG), ("r_cmodel_mag", M.HSCMagR),
                     ("i_cmodel_mag", M.HSCMagI), ("z_cmodel_mag", M.HSCMagZ),
                     ("y_cmodel_mag", M.HSCMagY), ("i_sdssshape_shape11", M.HSCShape11),
                     ("i_sdssshape_shape22", M.HSCShape22), ("i_sdssshape_shape12", M.HSCShape12)]:
        scalar(out, "hsc", mod, d[col])

    for src, cls in [("sdss", M.SDSSSpectrum), ("desi", M.DESISpectrum)]:
        d = subset(src)
        n = len(d["Z"])
        timed(f"codecs.{src}.{cls.token_key}", lambda: run_codec(
            out, f"{src}.{cls.token_key}", cls, spectrum_maker(cls, d), n))
        scalar(out, src, M.Z, d["Z"])

    d = subset("gaia")
    scalar(out, "gaia", M.GaiaXpBp, d["coeff"][:, :55])
    scalar(out, "gaia", M.GaiaXpRp, d["coeff"][:, 55:])
    for col, mod in [("phot_g_mean_flux", M.GaiaFluxG), ("phot_bp_mean_flux", M.GaiaFluxBp),
                     ("phot_rp_mean_flux", M.GaiaFluxRp), ("parallax", M.GaiaParallax),
                     ("ra", M.Ra), ("dec", M.Dec)]:
        scalar(out, "gaia", mod, d[col])
    save_file(out, os.path.join(a.out, "codecs.safetensors"))


# Model

def embed_targets_patched(self, target_mask, num_decoder_tokens=256):
    """AION.embed_targets with the training decoder mask: the count of target
    tokens sits at each target modality's first token."""
    device = next(self.parameters()).device
    mods = {}
    for mod, mask in target_mask.items():
        mask = mask.to(torch.bool).to(device)
        tensor = torch.zeros_like(mask).to(torch.long)
        compressed = torch.zeros_like(mask).to(torch.long)
        compressed[:, 0] = (~mask).sum(dim=1)
        mods[mod] = self.decoder_embeddings[mod].forward_embed(
            {"tensor": tensor, "target_mask": mask, "decoder_attention_mask": compressed})
    return self.forward_mask_decoder(mods, num_decoder_tokens)


def part_model():
    out = OrderedDict()
    toks = load_file(a.tokens or os.path.join(a.out, "codecs.safetensors"))
    meta["model_tokens_from"] = a.tokens or "this run"
    model = AION.from_pretrained("polymathic-ai/aion-base").to(DT).eval()

    def tk(src, key, n):
        t = torch.as_tensor(toks[f"{src}.{key}.tokens"][:n]).long()
        return t.reshape(n, -1) if t.dim() > 1 else t

    def inputs(src, keys, n):
        return OrderedDict((k, tk(src, k, n)) for k in keys)

    def width(d):
        return sum(v.shape[1] if v.dim() == 2 else 1 for v in d.values())

    def embed(name, d, n):
        def go():
            parts = [model.encode(OrderedDict((k, v[lo:hi]) for k, v in d.items()),
                                  num_encoder_tokens=width(d))
                     for lo, hi in chunks(n, a.model_batch)]
            out[f"embed.{name}"] = np_of(cat(parts))
        timed(f"model.embed.{name}", go)

    def logits(name, d, target, n):
        def go():
            for variant in ["upstream", "patched"]:
                if variant == "patched":
                    model.embed_targets = types.MethodType(embed_targets_patched, model)
                try:
                    parts = [model(OrderedDict((k, v[lo:hi]) for k, v in d.items()),
                                   target_modality=target)[target.token_key]
                             for lo, hi in chunks(n, a.model_batch)]
                finally:
                    if variant == "patched":
                        del model.embed_targets
                lg = cat(parts)
                out[f"logits.{name}.{variant}"] = np_of(lg)
                out[f"argmax.{name}.{variant}"] = np_of(lg.argmax(-1))
        timed(f"model.logits.{name}", go)

    n = a.n_model
    ls_scalars = ["tok_flux_g", "tok_flux_r", "tok_flux_i", "tok_flux_z", "tok_flux_w1",
                  "tok_flux_w2", "tok_flux_w3", "tok_flux_w4", "tok_ebv"]
    embed("ls_desi.image", inputs("ls_desi", ["tok_image"], n), n)
    embed("ls_desi.spectrum", inputs("ls_desi", ["tok_spectrum_desi"], n), n)
    embed("ls_desi.all", inputs("ls_desi", ["tok_image", "tok_spectrum_desi"] + ls_scalars, n), n)
    embed("ls_catalog.all", inputs("ls_catalog", ["tok_image"] + ls_scalars +
                                   ["tok_shape_r", "tok_shape_e1", "tok_shape_e2"], n), n)
    embed("hsc.all", inputs("hsc", ["tok_image_hsc", "tok_a_g", "tok_a_r", "tok_a_i", "tok_a_z",
                                    "tok_a_y", "tok_mag_g", "tok_mag_r", "tok_mag_i", "tok_mag_z",
                                    "tok_mag_y", "tok_shape11", "tok_shape22", "tok_shape12"], n), n)
    embed("sdss.all", inputs("sdss", ["tok_spectrum_sdss", "tok_z"], n), n)
    embed("desi.all", inputs("desi", ["tok_spectrum_desi", "tok_z"], n), n)
    embed("gaia.all", inputs("gaia", ["tok_xp_bp", "tok_xp_rp", "tok_flux_g_gaia",
                                      "tok_flux_bp_gaia", "tok_flux_rp_gaia", "tok_parallax",
                                      "tok_ra", "tok_dec"], n), n)

    phot = inputs("ls_desi", ["tok_image", "tok_flux_g", "tok_flux_r", "tok_flux_i", "tok_flux_z"], n)
    logits("z_from_phot", phot, M.Z, n)
    logits("z_from_desi", inputs("ls_desi", ["tok_spectrum_desi"], n), M.Z, n)
    logits("desi_from_image", inputs("ls_desi", ["tok_image"], n), M.DESISpectrum, n)
    m = a.n_image_target
    logits("image_from_desi", inputs("ls_desi", ["tok_spectrum_desi"], m), M.LegacySurveyImage, m)
    save_file(out, os.path.join(a.out, "model.safetensors"))


parts = set(a.parts.split(","))
if "goldens" in parts:
    part_goldens()
if "codecs" in parts:
    part_codecs()
if "model" in parts:
    part_model()
with open(os.path.join(a.out, f"meta-{'-'.join(sorted(parts))}.json"), "w") as fo:
    json.dump(meta, fo, indent=1)
print("failures:", len(meta["failures"]))
for x in meta["failures"]:
    print("  ", x)
