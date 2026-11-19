# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.14.1",
#   "numpy==2.5.5",
#   "safetensors==0.8.0",
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
"""Fixtures for test_spectrum: a few-channel upstream spectrum codec, its outputs and
their rounding ensemble.

    uv run test/gen/spectrum.py OUT [--n 32]

OUT is test/upstream.

Builds upstream's SpectrumCodec (polymathic-aion at db7fba6d) in a small configuration
(encoder depths 1,1,2,1 and dims 4,4,4,8; decoder depths 1,1,1,1 and dims 4,4,4,2; 8
latent channels; a latent grid of 64 wavelengths, 3500 + 0.8 k) and gives every
parameter a random value (torch.manual_seed(0); uniform in [-0.5, 0.5], layer norms'
and GRN's scales 1 more, since upstream's initialisation leaves GRN the identity;
biases and GRN's shift a tenth of that). Designed so that the pre-quantizer norm's
eps matters: the encoder's last layer norm is scaled by 3e-3, which puts the
pre-quantizer norm's lane variance near its eps 1e-5.

It runs on 16 short spectra (40 pixels from 3482-3533 A in steps of 1.1-1.3 A, a NaN
flux, a NaN and a negative inverse variance, masked pixels), as the base Spectrum
class (unpadded) and as DESI and SDSS spectra (padded), in float64 (the reference,
torch's default dtype float64, weights cast exactly) and float32 (a positive
control), and n times in float64 under perturb's dispatch mode at the order
model's float32 scale (each sample with the reference's normalisation token, and
decoding the reference's tokens). Writes:

- OUT/spectrum_codec.safetensors: the codec's state dict, float32, upstream's names.
- OUT/spectrum_golden.safetensors:
  - input.{flux,ivar,mask,lambda};
  - for each <v> of base, desi and sdss: f64.<v>.{pre,norm_pre,tokens,flux,logit,mask}
    (pre the LFQ input (b, 10, 2); flux, logit and mask the decoding of the
    reference's tokens on the grid), f32.<v>.{pre,tokens,flux,mask}, and
    s.<v>.{pre,norm_pre,flux,logit}: the ensemble's root mean square distance from
    the reference, per value;
  - grn.{x,gamma,beta,ref,f32,s}: one GRN, designed so that its eps 1e-6 matters
    (x of 16 positions and 8 channels, entries in +-2.5e-6, so the channels' mean
    norm is near 1e-5), its float64 and float32 outputs and its ensemble's spread.
  - n, the ensemble's size.

Its metadata names this script, torch's version and upstream's commit.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from safetensors.numpy import save_file

from aion.codecs.modules.utils import GRN
from aion.codecs.spectrum import SpectrumCodec
from aion import modalities as M
from perturb import U32, Ensemble

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--n", type=int, default=32)
a = p.parse_args()

torch.set_num_threads(2)
torch.set_grad_enabled(False)
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
state = {k: v.detach().clone().contiguous().numpy() for k, v in codec.state_dict().items()}

rng = np.random.default_rng(0)
b, m = 16, 40
lam = np.stack([3482 + 1.5 * i + np.cumsum(rng.uniform(1.1, 1.3, m)) for i in range(b)]).astype(np.float32)
flux = rng.uniform(0.5, 3.0, (b, m)).astype(np.float32)
ivar = rng.uniform(0.0, 150.0, (b, m)).astype(np.float32)
mask = rng.uniform(size=(b, m)) < 0.15
flux[1, 5] = np.nan
ivar[2, 7] = np.nan
ivar[3, 9] = -1.0
data = {"input.flux": flux, "input.ivar": ivar, "input.mask": mask.astype(np.uint8),
        "input.lambda": lam, "n": np.array([a.n], np.int64)}

variants = {"base": M.Spectrum, "desi": M.DESISpectrum, "sdss": M.SDSSSpectrum}


def spectra(cls, dt, copies=1):
    f = lambda x: torch.tensor(np.tile(x, (copies, 1))).to(dt)
    return cls(flux=f(flux), ivar=f(ivar), mask=torch.tensor(np.tile(mask, (copies, 1))),
               wavelength=f(lam))


def run(c, dt, cls, tokens=None, copies=1, mode=None, norm=None):
    torch.set_default_dtype(dt)
    nq = c.normalization_quantizer
    orig = nq.quantize
    seen = {}

    def quantize(z):
        seen["norm_pre"] = z.clone()
        if mode is not None:
            mode.active = False
        try:
            return orig(z) if norm is None else norm.clone()
        finally:
            if mode is not None:
                mode.active = True
    nq.quantize = quantize
    logits = {}
    hook = c.decoder.register_forward_hook(lambda m_, i_, o: logits.__setitem__("out", o))
    try:
        ctx = mode if mode is not None else torch.no_grad()
        with ctx:
            h, nm = c._encode(spectra(cls, dt, copies))
        norm_pre = seen["norm_pre"]
        if tokens is None:
            tokens = c.encode(spectra(cls, dt, copies))
        with ctx:
            d = c.decode(tokens)
    finally:
        hook.remove()
        del nq.quantize
        torch.set_default_dtype(torch.float32)
    f = lambda x: x.detach().double().numpy()
    return dict(pre=f(h), norm_pre=f(norm_pre), norm=nm, tokens=tokens, flux=f(d.flux),
                logit=f(logits["out"][:, 1]), mask=d.mask.numpy().astype(np.uint8))


torch.set_default_dtype(torch.float64)
codec64 = SpectrumCodec(**config).to(torch.float64).eval()
codec64.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
codec64 = codec64.to(torch.float64)
torch.set_default_dtype(torch.float32)

for si, (name, cls) in enumerate(variants.items()):
    ref = run(codec64, torch.float64, cls)
    for k in ("pre", "norm_pre", "flux", "logit", "mask"):
        data[f"f64.{name}.{k}"] = ref[k]
    data[f"f64.{name}.tokens"] = ref["tokens"].numpy()
    f32 = run(codec, torch.float32, cls, tokens=ref["tokens"])
    data[f"f32.{name}.tokens"] = codec.encode(spectra(cls, torch.float32)).numpy()
    for k in ("pre", "flux", "mask"):
        data[f"f32.{name}.{k}"] = f32[k]
    mode = Ensemble(2000 + si, U32)
    s = run(codec64, torch.float64, cls, tokens=ref["tokens"].repeat(a.n, 1), copies=a.n, mode=mode,
            norm=ref["norm"].repeat(a.n))
    for k in ("pre", "norm_pre", "flux", "logit"):
        d = s[k].reshape((a.n,) + ref[k].shape) - ref[k]
        data[f"s.{name}.{k}"] = np.sqrt((d ** 2).mean(0))
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
             "grn.ref": gref, "grn.f32": grn(torch.float32),
             "grn.s": np.sqrt(((gs - gref) ** 2).mean(0, keepdims=True))})
print("grn: mean norm", float(torch.linalg.vector_norm(x.double(), dim=1).mean()),
      "float32 max z", float((np.abs(data["grn.f32"] - gref) / data["grn.s"]).max()))

meta = {"script": "apps/aion/test/gen/spectrum.py", "torch": torch.__version__,
        "upstream": "PolymathicAI/AION db7fba6df53b", "seed": "torch 0, numpy 0"}
os.makedirs(a.out, exist_ok=True)
save_file(state, os.path.join(a.out, "spectrum_codec.safetensors"), metadata=meta)
save_file({k: np.ascontiguousarray(v) for k, v in data.items()},
          os.path.join(a.out, "spectrum_golden.safetensors"), metadata=meta)
print("wrote", a.out)
