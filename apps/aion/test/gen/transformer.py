"""Upstream's AION in float64, on a narrow random network and its rounding ensemble for
test_transformer, or on AION-1's weights, with every codec, for the parity part gpu.

    uv run --locked transformer.py --weights narrow OUT [--n 32]
    uv run --locked transformer.py --weights aion-base --subset SUBSET \\
        [--precision f64] [--threads 2] [--parts codecs,model] [--tokens TOKENS] OUT

Both decode with the training decoder mask (embed_targets_patched below): the target
count at each target modality's first token, as 4M's training sampler does, so the
targets of a modality attend to each other. AION.embed_targets passes an all-zero
compressed mask, so that adapt_decoder_attention_mask masks every entry and decoder
self-attention is uniform. The patch lives here, never in the clone. Every file of the
narrow network names this script, torch's version and upstream's commit.

--weights narrow: OUT is ../upstream. Upstream's AION (polymathic-aion at db7fba6d) has
aion-base's structure (SwiGLU without biases, query-key norm, layer norms whose shift is
a buffer, at eps 1e-6, a context projection with a bias) at small sizes: width 16, 2
heads, 2 encoder and 2 decoder blocks, over four modalities (tok_xp_bp, tok_flux_g,
tok_z, tok_xp_rp). Every weight is random (torch.manual_seed(0)), designed so that each
constant matters:

- token, positional, modality and mask embeddings are scaled by 1e-3, so the first
  block's layer norm, in the encoder and in the decoder, sees a variance near its
  eps 1e-6;
- query and key projections (self- and cross-attention) are scaled by 1e-3, so the
  query-key norms see a variance near their eps, and without them the scores would be
  near zero;
- layer norms' and query-key norms' scales are 1 + U(-0.5, 0.5), and their shifts,
  buffers that AION-1's weights hold at zero but upstream's layer norm adds,
  U(-0.5, 0.5);
- the context projection's bias is U(-0.5, 0.5);
- the decoder's token tables, which no path reads, hold N(0, 1); the file keeps
  tok_z's alone.

The encoder's positional buffers are upstream's sine-cosine tables times 1e-3, and the
decoder's N(0, 1e-6), so that each table is checked where upstream reads it; the
modality embeddings are shared, and the weights file stores each under its decoder
name only, as aion-base's file does. The shifts and the decoder's tables are drawn
after every other weight, from torch's generator at seed 1.

Inputs: four objects, tok_xp_bp (55 tokens), tok_flux_g and tok_z, some absent
(upstream's input mask): object 1 lacks tok_xp_bp, object 2 lacks tok_flux_g and
tok_xp_bp's tokens 10-19, object 3 lacks tok_z. Outputs, in float64 (the reference,
torch's default dtype float64, weights cast exactly):

- embed: AION.encode of the four objects, with num_encoder_tokens the input's width;
- logits.tok_z, logits.tok_xp_rp: AION.forward of objects 1 and 2 to both targets,
  tok_xp_rp's for object 2 alone;

and for each, s.<name>: the root mean square distance from the reference of n float64
runs under perturb's dispatch mode at the order model's float32 scale. Writes:

- OUT/transformer.safetensors: the weights, float32, upstream's names.
- OUT/transformer_golden.safetensors: input.<key>.{tokens,absent} (int64 and uint8,
  [4; tokens]), f64.embed, f64.logits.<key>, s.embed, s.logits.<key> (float32), n.

--weights aion-base: upstream's codecs and aion-base at one precision and one thread
count, on the parity subset. Every input is rounded to float32 once (the data are
float32 values; a few catalog columns are stored float64 and are rounded here), then
cast to the run's dtype. In the f64 run the model and codecs are cast with .to(float64)
and torch's default dtype is float64, so constants built inside forward passes (FSQ's
bounds, linspaces) are float64 too. Codecs run in chunks of --codec-batch objects and the
model in chunks of --model-batch, the same in every run. Writes, in OUT:

  codecs.safetensors   every codec on the parity subset: input as fed, the value the
                       quantizer decides on, tokens, decoded values
  model.safetensors    AION.encode embeddings; decoder logits and argmax for Z (one
                       token), a DESI spectrum (273) and an LS image (576), each from
                       upstream's forward and from the patched forward
  meta-<parts>.json    versions, settings, timings, failures

The model part takes its input tokens from --tokens (the f64 run's codecs.safetensors)
when given, so that a legitimate codec flip does not leak into the model comparison;
otherwise from this run's own tokens.
"""
import argparse
import copy
import json
import os
import random
import time
import types
from collections import OrderedDict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from safetensors.numpy import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--weights", choices=["narrow", "aion-base"], required=True)
p.add_argument("out")
p.add_argument("--n", type=int, default=32)
p.add_argument("--subset")
p.add_argument("--precision", choices=["f32", "f64"], default="f32")
p.add_argument("--threads", type=int, default=1)
p.add_argument("--parts", default="codecs,model")
p.add_argument("--tokens", help="codecs.safetensors whose tokens drive the model part")
p.add_argument("--codec-batch", type=int, default=8)
p.add_argument("--model-batch", type=int, default=2)
p.add_argument("--n-model", type=int, default=8, help="objects per source in the model part")
p.add_argument("--n-image-target", type=int, default=2, help="objects for the LS-image target")
a = p.parse_args()

torch.set_num_threads(2 if a.weights == "narrow" else a.threads)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
random.seed(0)
torch.manual_seed(0)
DT = torch.float64 if a.weights == "aion-base" and a.precision == "f64" else torch.float32
torch.set_default_dtype(DT)

# Upstream's modules are imported at the run's default dtype.
from aion import modalities as M                                       # noqa: E402
from aion.codecs import config as codec_config                         # noqa: E402
from aion.codecs.config import HF_REPO_ID, MODALITY_CODEC_MAPPING      # noqa: E402
from aion.codecs.quantizers import FiniteScalarQuantizer               # noqa: E402
from aion.codecs.quantizers.scalar import (                            # noqa: E402
    ScalarLogReservoirQuantizer, ScalarCompressedReservoirQuantizer,
    MultiScalarCompressedReservoirQuantizer)
from aion.model import AION                                            # noqa: E402
import perturb                                                         # noqa: E402
from perturb import U32, Ensemble, spread                              # noqa: E402


def embed_targets_patched(self, target_mask, num_decoder_tokens=256):
    """AION.embed_targets with the training decoder mask: the count of target
    tokens sits at each target modality's first token."""
    mods = {}
    for mod, mask in target_mask.items():
        mask = mask.to(torch.bool)
        tensor = torch.zeros_like(mask).to(torch.long)
        compressed = torch.zeros_like(mask).to(torch.long)
        compressed[:, 0] = (~mask).sum(dim=1)
        mods[mod] = self.decoder_embeddings[mod].forward_embed(
            {"tensor": tensor, "target_mask": mask, "decoder_attention_mask": compressed})
    return self.forward_mask_decoder(mods, num_decoder_tokens)


# Narrow

def narrow():
    # An embedding lookup rounds nothing.
    perturb.NOROUND.add("embedding")
    mods = ["tok_xp_bp", "tok_flux_g", "tok_z", "tok_xp_rp"]
    inputs = ["tok_xp_bp", "tok_flux_g", "tok_z"]
    targets = [types.SimpleNamespace(token_key="tok_z", num_tokens=1),
               types.SimpleNamespace(token_key="tok_xp_rp", num_tokens=55)]
    d = 16
    config = dict(act_layer="SiLU", decoder_depth=2, encoder_depth=2, dim=d, num_heads=2,
                  mlp_ratio=4, gated_mlp=True, mlp_bias=False, norm_bias=False,
                  proj_bias=False, qk_norm=True, qkv_bias=False, domains_in=mods,
                  domains_out=mods)
    model = AION(config).eval()
    small = 1e-3

    def u(shape):
        return torch.rand(shape) - 0.5

    for name, prm in model.named_parameters():
        if name.endswith("mod_emb") or name == "mask_token" or (
                name.startswith("encoder_embeddings") and name.endswith("token_emb.weight")):
            v = torch.randn(prm.shape) * small
        elif name.startswith("decoder_embeddings") and name.endswith("token_emb.weight"):
            v = torch.randn(prm.shape)
        elif prm.ndim == 1 and "norm" in name:
            v = u(prm.shape) + 1.0
        else:
            v = u(prm.shape)
            if name.endswith("qkv.weight"):
                v[: 2 * d] *= small
            elif name.endswith("cross_attn.q.weight"):
                v *= small
            elif name.endswith("cross_attn.kv.weight"):
                v[:d] *= small
        prm.copy_(v)
    for name, buf in model.named_buffers():
        if name.endswith("pos_emb"):
            buf.mul_(small)
    buffers = torch.Generator().manual_seed(1)
    for name, buf in model.named_buffers():
        if name.startswith("decoder_embeddings") and name.endswith("pos_emb"):
            buf.copy_(torch.randn(buf.shape, generator=buffers) * small)
        elif "norm" in name and name.endswith("bias"):
            buf.copy_(torch.rand(buf.shape, generator=buffers) - 0.5)

    # The file stores each modality embedding under its decoder name only, as
    # aion-base's does, and one of the decoder's token tables, which no path reads, to
    # keep it small.
    state = {k: v.detach().clone().contiguous().numpy()
             for k, v in model.state_dict().items()
             if not (k.startswith("encoder_embeddings") and k.endswith("mod_emb"))
             and not (k.startswith("decoder_embeddings") and k.endswith("token_emb.weight")
                      and k != "decoder_embeddings.tok_z.token_emb.weight")}

    rng = np.random.default_rng(0)
    b = 4
    tokens = OrderedDict()
    absent = OrderedDict()
    for key in inputs:
        n = 55 if key == "tok_xp_bp" else 1
        vocab = 1025 if key == "tok_z" else 1024
        tokens[key] = rng.integers(0, vocab, (b, n), dtype=np.int64)
        absent[key] = np.zeros((b, n), dtype=bool)
    absent["tok_xp_bp"][1, :] = True
    absent["tok_flux_g"][2, :] = True
    absent["tok_xp_bp"][2, 10:20] = True
    absent["tok_z"][3, :] = True
    width = sum(t.shape[1] for t in tokens.values())
    fwd = [1, 2]

    def run(m, dt, copies=1, mode=None):
        """embed and logits of model m at dtype dt, each object repeated copies times."""
        torch.set_default_dtype(dt)
        rep = lambda x, rows=slice(None): torch.tensor(np.tile(x[rows], (copies, 1)))
        m.embed_targets = types.MethodType(embed_targets_patched, m)
        ctx = mode if mode is not None else torch.no_grad()
        try:
            with ctx:
                e = m.encode(OrderedDict((k, rep(v)) for k, v in tokens.items()),
                             input_mask={k: rep(v) for k, v in absent.items()},
                             num_encoder_tokens=width)
                lg = m(OrderedDict((k, rep(v, fwd)) for k, v in tokens.items()),
                       target_modality=targets,
                       input_mask={k: rep(v, fwd) for k, v in absent.items()})
        finally:
            del m.embed_targets
            torch.set_default_dtype(torch.float32)
        f = lambda x: x.detach().double().numpy()
        out = {"embed": f(e)}
        out.update({f"logits.{k}": f(v) for k, v in lg.items()})
        return out

    torch.set_default_dtype(torch.float64)
    model64 = copy.deepcopy(model).to(torch.float64).eval()
    torch.set_default_dtype(torch.float32)

    ref = run(model64, torch.float64)
    f32 = run(model, torch.float32)
    s = run(model64, torch.float64, copies=a.n, mode=Ensemble(4000, U32))
    data = {"n": np.array([a.n], np.int64)}
    for k in inputs:
        data[f"input.{k}.tokens"] = tokens[k]
        data[f"input.{k}.absent"] = absent[k].astype(np.uint8)
    # The tok_xp_rp logits are kept for the forward pass's second object (object 2) only.
    keep = {"logits.tok_xp_rp": slice(1, 2)}
    for k, r in ref.items():
        sp = spread(s[k], r)
        rows = keep.get(k, slice(None))
        r, sp, f32[k] = r[rows], sp[rows], f32[k][rows]
        data[f"f64.{k}"] = r
        data[f"s.{k}"] = sp.astype(np.float32)
        z = np.abs(f32[k] - r) / sp
        print(f"{k}: shape {r.shape}, |ref| max {np.abs(r).max():.3g}, spread median"
              f" {np.median(sp):.3g}, float32 torch max z {z.max():.2f}", flush=True)

    meta = {"script": "apps/aion/test/gen/transformer.py", "torch": torch.__version__,
            "upstream": "PolymathicAI/AION db7fba6df53b",
            "seed": "torch 0 and 1, numpy 0, random 0"}
    os.makedirs(a.out, exist_ok=True)
    save_file(state, os.path.join(a.out, "transformer.safetensors"), metadata=meta)
    save_file({k: np.ascontiguousarray(v) for k, v in data.items()},
              os.path.join(a.out, "transformer_golden.safetensors"), metadata=meta)
    print("wrote", a.out)


# AION-1

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


def aion_base():
    os.makedirs(a.out, exist_ok=True)
    parts = set(a.parts.split(","))
    if "codecs" in parts:
        part_codecs()
    if "model" in parts:
        part_model()
    with open(os.path.join(a.out, f"meta-{'-'.join(sorted(parts))}.json"), "w") as fo:
        json.dump(meta, fo, indent=1)
    print("failures:", len(meta["failures"]))
    for x in meta["failures"]:
        print("  ", x)


narrow() if a.weights == "narrow" else aion_base()
