"""Upstream's spectrum codec, in float64 and in its rounding ensemble, on few-channel
random weights for test_spectrum or on AION-1's for the parity part spectrum.

    uv run --locked spectrum.py --weights narrow OUT [--n 32]
    uv run --locked spectrum.py --weights aion-base --goldens GOLDENS --subset SUBSET \\
        SET OUT [--objects 16] [--n 32] [--chunk 32]

Both run upstream's SpectrumCodec (polymathic-aion at db7fba6d) in float64 (the
reference, torch's default dtype float64, weights cast exactly), and n times in float64
under perturb's dispatch mode at the order model's float32 scale, each sample with the
reference's normalisation token and decoding the reference's tokens, so that the spread
is conditional on equal tokens. The values are pre, the LFQ input (10, n); norm_pre,
the normalisation before its quantizer; norm, its value; the tokens; and flux and
logit, the decoding of the tokens on the latent grid and its mask logit. A value's
spread is the root mean square of the samples' distance from the reference. Every
file's metadata names this script, torch's version and upstream's commit.

--weights narrow: OUT is ../upstream. The codec has a small configuration (encoder
depths 1,1,2,1 and dims 4,4,4,8; decoder depths 1,1,1,1 and dims 4,4,4,2; 8 latent
channels; a latent grid of 64 wavelengths, 3500 + 0.8 k), every parameter random
(torch.manual_seed(0); uniform in [-0.5, 0.5], layer norms' and GRN's scales 1 more,
since upstream's initialisation leaves GRN the identity; biases and GRN's shift a tenth
of that). Designed so that the pre-quantizer norm's eps matters: the encoder's last layer
norm is scaled by 3e-3, which puts the pre-quantizer norm's lane variance near its eps
1e-5. It runs on 16 short spectra (40 pixels from 3482-3533 A in steps of 1.1-1.3 A, a
NaN flux, a NaN and a negative inverse variance, masked pixels), as the base Spectrum
class (unpadded) and as DESI and SDSS spectra (padded), also in float32 (a positive
control). Writes:

- OUT/spectrum_codec.safetensors: the codec's state dict, float32, upstream's names.
- OUT/spectrum_golden.safetensors:
  - input.{flux,ivar,mask,lambda};
  - for each <v> of base, desi and sdss: f64.<v>.{pre,norm_pre,tokens,flux,logit,mask},
    f32.<v>.{pre,tokens,flux,mask}, and s.<v>.{pre,norm_pre,flux,logit};
  - grn.{x,gamma,beta,ref,f32,s}: one GRN, designed so that its eps 1e-6 matters
    (x of 16 positions and 8 channels, entries in +-2.5e-6, so the channels' mean
    norm is near 1e-5), its float64 and float32 outputs and its ensemble's spread.
  - n, the ensemble's size.

--weights aion-base: AION-1's codec, on SET's first --objects rows: golden (upstream's
tests/test_data SPECTRUM input, base Spectrum class, unpadded), ls_desi or desi (the
parity subset's DESI rows, DESISpectrum), or sdss (the subset's SDSS rows trimmed of
their -1 pads, SDSSSpectrum; rows 0-14 and 41, the row whose token the port flipped).
Samples at order (u = 2^-24) and order64 (u = 2^-53, the float64 check's), in batches of
--chunk copies, each element drawing its own noise. OUT holds rows, n, ref.<v>,
ref.tokens and ref.mask, and s.<scale>.<v>, as perturb.per_object records them.
"""
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from aion import modalities as M
from aion.codecs.config import HF_REPO_ID, MODALITY_CODEC_MAPPING
from aion.codecs.modules.utils import GRN
from aion.codecs.spectrum import SpectrumCodec
from perturb import U32, U64, Ensemble, per_object, samples, spread

p = argparse.ArgumentParser()
p.add_argument("--weights", choices=["narrow", "aion-base"], required=True)
p.add_argument("set", nargs="?", choices=["golden", "ls_desi", "desi", "sdss"])
p.add_argument("out")
p.add_argument("--n", type=int, default=32)
p.add_argument("--objects", type=int, default=16)
p.add_argument("--chunk", type=int, default=32, help="samples per batch")
p.add_argument("--goldens", help="goldens.py's directory")
p.add_argument("--subset", help="subset.py's directory")
a = p.parse_args()

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
KEYS = ("pre", "norm_pre", "flux", "logit")
META = {"script": "apps/aion/test/gen/spectrum.py", "torch": torch.__version__,
        "upstream": "PolymathicAI/AION db7fba6df53b"}


def run(c, dt, spectra, tokens=None, mode=None, norm=None):
    """pre, norm_pre, norm, tokens, flux, logit and mask of spectra(), with the
    normalisation norm when given."""
    torch.set_default_dtype(dt)
    nq = c.normalization_quantizer
    orig = nq.quantize
    seen = {}

    def quantize(z):
        seen["norm_pre"] = z.clone()
        if mode is not None:
            mode.active = False
        try:
            return orig(z) if norm is None else norm.expand(z.shape).clone()
        finally:
            if mode is not None:
                mode.active = True
    nq.quantize = quantize
    logits = {}
    hook = c.decoder.register_forward_hook(lambda m_, i_, o: logits.__setitem__("out", o))
    try:
        ctx = mode if mode is not None else torch.no_grad()
        with ctx:
            h, nm = c._encode(spectra())
        norm_pre = seen["norm_pre"]
        if tokens is None:
            tokens = c.encode(spectra())
        with ctx:
            d = c.decode(tokens)
    finally:
        hook.remove()
        del nq.quantize
        torch.set_default_dtype(torch.float32)
    f = lambda x: x.detach().double().numpy()
    return dict(pre=f(h), norm_pre=f(norm_pre), norm=nm, tokens=tokens, flux=f(d.flux),
                logit=f(logits["out"][:, 1]), mask=d.mask.numpy().astype(np.uint8))


def narrow():
    torch.manual_seed(0)
    config = dict(encoder_depths=(1, 1, 2, 1), encoder_dims=(4, 4, 4, 8),
                  decoder_depths=(1, 1, 1, 1), decoder_dims=(4, 4, 4, 2),
                  latent_channels=8, embedding_dim=10, num_pixels=64)
    codec = SpectrumCodec(**config).eval()
    for name, prm in codec.named_parameters():
        v = torch.rand(prm.shape) - 0.5
        if name.endswith("bias") or name.endswith("grn.beta"):
            v = v / 10
        scale = name.endswith("norm.weight") or name.endswith("grn.gamma") or (
            prm.ndim == 1 and name.endswith("weight") and ("sample_layers" in name))
        prm.copy_(v + 1.0 if scale else v)
    codec.encoder.norm.weight.mul_(3e-3)
    codec.encoder.norm.bias.mul_(3e-3)
    state = {k: v.detach().clone().contiguous().numpy()
             for k, v in codec.state_dict().items()}

    rng = np.random.default_rng(0)
    b, m = 16, 40
    lam = np.stack([3482 + 1.5 * i + np.cumsum(rng.uniform(1.1, 1.3, m))
                    for i in range(b)]).astype(np.float32)
    flux = rng.uniform(0.5, 3.0, (b, m)).astype(np.float32)
    ivar = rng.uniform(0.0, 150.0, (b, m)).astype(np.float32)
    mask = rng.uniform(size=(b, m)) < 0.15
    flux[1, 5] = np.nan
    ivar[2, 7] = np.nan
    ivar[3, 9] = -1.0
    data = {"input.flux": flux, "input.ivar": ivar, "input.mask": mask.astype(np.uint8),
            "input.lambda": lam, "n": np.array([a.n], np.int64)}

    def spectra(cls, dt, copies=1):
        f = lambda x: torch.tensor(np.tile(x, (copies, 1))).to(dt)
        return lambda: cls(flux=f(flux), ivar=f(ivar),
                           mask=torch.tensor(np.tile(mask, (copies, 1))), wavelength=f(lam))

    torch.set_default_dtype(torch.float64)
    codec64 = SpectrumCodec(**config).to(torch.float64).eval()
    codec64.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
    codec64 = codec64.to(torch.float64)
    torch.set_default_dtype(torch.float32)

    variants = {"base": M.Spectrum, "desi": M.DESISpectrum, "sdss": M.SDSSSpectrum}
    for si, (name, cls) in enumerate(variants.items()):
        ref = run(codec64, torch.float64, spectra(cls, torch.float64))
        for k in ("pre", "norm_pre", "flux", "logit", "mask"):
            data[f"f64.{name}.{k}"] = ref[k]
        data[f"f64.{name}.tokens"] = ref["tokens"].numpy()
        f32 = run(codec, torch.float32, spectra(cls, torch.float32), ref["tokens"])
        data[f"f32.{name}.tokens"] = codec.encode(spectra(cls, torch.float32)()).numpy()
        for k in ("pre", "flux", "mask"):
            data[f"f32.{name}.{k}"] = f32[k]
        s = samples(lambda k, mode: run(codec64, torch.float64, spectra(cls, torch.float64, k),
                                        ref["tokens"].repeat(k, 1), mode,
                                        ref["norm"].repeat(k)),
                    KEYS, a.n, a.n, 2000 + si, U32)
        for k in KEYS:
            data[f"s.{name}.{k}"] = spread(s[k], ref[k])
        z = np.abs(f32["pre"] - ref["pre"]) / data[f"s.{name}.pre"]
        print(f"{name}: tokens {ref['tokens'][:3].tolist()}...; float32 tokens differ:"
              f" {int((data[f'f32.{name}.tokens'] != data[f'f64.{name}.tokens']).sum())};"
              f" float32 LFQ input max z {z.max():.2f}", flush=True)

    # One GRN, its eps 1e-6 against a mean norm near 1e-5.
    g = torch.Generator().manual_seed(1)
    x = ((torch.rand(1, 16, 8, generator=g) - 0.5) * 5e-6).float()
    gamma = (torch.rand(8, generator=g) + 0.5).float()
    beta = ((torch.rand(8, generator=g) - 0.5) * 1e-6).float()

    def grn(dt, copies=1, mode=None):
        torch.set_default_dtype(dt)
        m = GRN(8).to(dt)
        m.gamma.copy_(gamma.to(dt).reshape(1, 1, 8))
        m.beta.copy_(beta.to(dt).reshape(1, 1, 8))
        xs = x.to(dt).repeat(copies, 1, 1)
        try:
            if mode is None:
                return m(xs).double().numpy()
            with mode:
                return m(xs).double().numpy()
        finally:
            torch.set_default_dtype(torch.float32)

    gref = grn(torch.float64)
    gs = grn(torch.float64, a.n, Ensemble(3000, U32))
    data.update({"grn.x": x.numpy(), "grn.gamma": gamma.numpy(), "grn.beta": beta.numpy(),
                 "grn.ref": gref, "grn.f32": grn(torch.float32), "grn.s": spread(gs, gref)})
    print("grn: mean norm", float(torch.linalg.vector_norm(x.double(), dim=1).mean()),
          "float32 max z", float((np.abs(data["grn.f32"] - gref) / data["grn.s"]).max()))

    meta = {**META, "seed": "torch 0, numpy 0"}
    os.makedirs(a.out, exist_ok=True)
    save_file(state, os.path.join(a.out, "spectrum_codec.safetensors"), metadata=meta)
    save_file({k: np.ascontiguousarray(v) for k, v in data.items()},
              os.path.join(a.out, "spectrum_golden.safetensors"), metadata=meta)
    print("wrote", a.out)


def aion_base():
    if a.set == "golden":
        t = load_file(f"{a.goldens}/SPECTRUM_input_batch.safetensors")
        cls, pre = M.Spectrum, "spectrum."
        rows = list(range(a.objects))
    elif a.set in ("ls_desi", "desi"):
        t = load_file(f"{a.subset}/{a.set}.safetensors")
        cls, pre = M.DESISpectrum, ("desi.spectrum_" if a.set == "ls_desi" else "spectrum_")
        rows = list(range(a.objects))
    else:
        t = load_file(f"{a.subset}/sdss.safetensors")
        cls, pre = M.SDSSSpectrum, "spectrum_"
        rows = list(range(a.objects - 1)) + [41]
    lam = t[pre + "lambda"]
    lengths = lam.shape[1] - (lam < 0).sum(1)
    scales = {"order": U32, "order64": U64}

    def spectrum(i, copies=1):
        k = lengths[i]
        row = lambda key: np.repeat(np.asarray(t[pre + key][i:i + 1, :k], np.float32), copies, 0)
        f = lambda key: torch.tensor(row(key)).to(torch.float64)
        m = torch.tensor(np.repeat(t[pre + "mask"][i:i + 1, :k].astype(bool), copies, 0))
        return lambda: cls(flux=f("flux"), ivar=f("ivar"), mask=m, wavelength=f("lambda"))

    torch.set_default_dtype(torch.float64)
    c64 = MODALITY_CODEC_MAPPING[M.SDSSSpectrum].from_pretrained(HF_REPO_ID, modality=cls)
    c64 = c64.to(torch.float64).eval()
    torch.set_default_dtype(torch.float32)

    def one(i):
        ref = run(c64, torch.float64, spectrum(i))
        obj = {"ref." + k: ref[k] for k in KEYS}
        obj["ref.norm"] = ref["norm"].detach().double().numpy()
        obj["ref.tokens"] = ref["tokens"].numpy()
        logit = torch.tensor(ref["logit"])
        obj["ref.mask"] = (torch.round(torch.sigmoid(logit)) != 0).numpy().astype(np.uint8)
        for si, (name, u) in enumerate(scales.items()):
            s = samples(lambda k, mode: run(c64, torch.float64, spectrum(i, k),
                                            ref["tokens"].repeat(k, 1), mode, ref["norm"]),
                        KEYS, a.n, a.chunk, 1000 + 100003 * si + 1009 * i, u)
            for k in KEYS:
                obj[f"s.{name}.{k}"] = spread(s[k], ref[k])
        return obj

    meta = {**META, "n": str(a.n), "scales": ",".join(scales), "set": a.set}
    per_object(a.out, rows, one, meta)


narrow() if a.weights == "narrow" else aion_base()
