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
"""The rounding ensemble of upstream's spectrum codec, per parity set, for
spectrum_parity.exe.

    uv run test/gen/spectrum_ensemble.py --goldens GOLDENS --subset SUBSET \
        SET OUT [--objects 16] [--n 32]

SET is golden (upstream's tests/test_data SPECTRUM input, base Spectrum class,
unpadded), ls_desi or desi (the parity subset's DESI rows, DESISpectrum), or sdss
(the subset's SDSS rows trimmed of their -1 pads, SDSSSpectrum). The objects are the
set's first rows; for sdss, rows 0-14 and 41, the row whose token the port flipped.

For each object it runs upstream's SpectrumCodec (aion-base, polymathic-aion
db7fba6d):
  - in float64, unmoved: the reference, ref.*: the LFQ input pre (10, 272), the
    normalisation before its quantizer norm_pre, its value norm, the tokens, and the
    decoding of those tokens on the latent grid, flux and the mask logit;
  - in float64 under perturb's dispatch mode, n times per scale, every sample
    with the reference's normalisation token and decoding the reference's tokens, so
    that the spread is conditional on equal tokens; s.<scale>.* is the root mean
    square of the samples' distance from the reference, per value. The n samples
    run as one batch of n copies of the object, each element drawing its own noise.
Scales: order (u = 2^-24, the float32 check's) and order64 (u = 2^-53, the float64
check's).

Each object is saved as it finishes, under OUT's name with .parts for .safetensors,
and a rerun skips the objects saved there with the same n; OUT, with the set's rows
and n, is assembled from them. The n samples run in batches of --chunk copies.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from aion import modalities as Mod
from aion.codecs.config import HF_REPO_ID, MODALITY_CODEC_MAPPING
from perturb import U32, U64, Ensemble

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

p = argparse.ArgumentParser()
p.add_argument("set", choices=["golden", "ls_desi", "desi", "sdss"])
p.add_argument("out")
p.add_argument("--objects", type=int, default=16)
p.add_argument("--n", type=int, default=32)
p.add_argument("--chunk", type=int, default=32, help="samples per batch")
p.add_argument("--goldens", required=True, help="goldens.py's directory")
p.add_argument("--subset", required=True, help="subset.py's directory")
a = p.parse_args()
SCALES = {"order": U32, "order64": U64}

# Inputs: (class, flux, ivar, mask, lambda) rows, float32 values.

if a.set == "golden":
    t = load_file(f"{a.goldens}/SPECTRUM_input_batch.safetensors")
    cls, pre = Mod.Spectrum, "spectrum."
    rows = list(range(a.objects))
elif a.set in ("ls_desi", "desi"):
    t = load_file(f"{a.subset}/{a.set}.safetensors")
    cls, pre = Mod.DESISpectrum, ("desi.spectrum_" if a.set == "ls_desi" else "spectrum_")
    rows = list(range(a.objects))
else:
    t = load_file(f"{a.subset}/sdss.safetensors")
    cls, pre = Mod.SDSSSpectrum, "spectrum_"
    rows = list(range(a.objects - 1)) + [41]
lam = t[pre + "lambda"]
lengths = lam.shape[1] - (lam < 0).sum(1)


def spectrum(i, dt, copies=1):
    k = lengths[i]
    f = lambda key: torch.tensor(np.repeat(np.asarray(t[pre + key][i:i + 1, :k], np.float32), copies, 0)).to(dt)
    m = torch.tensor(np.repeat(t[pre + "mask"][i:i + 1, :k].astype(bool), copies, 0))
    return cls(flux=f("flux"), ivar=f("ivar"), mask=m, wavelength=f("lambda"))


def make(dt):
    torch.set_default_dtype(dt)
    c = MODALITY_CODEC_MAPPING[Mod.SDSSSpectrum].from_pretrained(HF_REPO_ID, modality=cls)
    c = c.to(dt).eval()
    torch.set_default_dtype(torch.float32)
    return c


def run(codec, dt, i, tokens=None, copies=1, mode=None, norm=None):
    """pre, norm_pre, norm, tokens, flux and logit of object i (copies of it)."""
    torch.set_default_dtype(dt)
    nq = codec.normalization_quantizer
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
    hook = codec.decoder.register_forward_hook(lambda m, i_, o: logits.__setitem__("out", o))
    try:
        ctx = mode if mode is not None else torch.no_grad()
        with ctx:
            h, nm = codec._encode(spectrum(i, dt, copies))
        norm_pre = seen["norm_pre"]
        if tokens is None:
            tokens = codec.encode(spectrum(i, dt, copies))
        with ctx:
            d = codec.decode(tokens)
        out = logits["out"]
    finally:
        hook.remove()
        del nq.quantize
        torch.set_default_dtype(torch.float32)
    f = lambda x: x.detach().double().numpy()
    return dict(pre=f(h), norm_pre=f(norm_pre), norm=f(nm), tokens=tokens,
                flux=f(d.flux), logit=f(out[:, 1]))


def ensemble(i, ref, name, seed):
    """The n perturbed runs of object i at scale name, in batches of --chunk."""
    u = SCALES[name]
    parts, census = [], {}
    for lo in range(0, a.n, a.chunk):
        k = min(a.chunk, a.n - lo)
        mode = Ensemble(seed + lo, u)
        parts.append(run(c64, torch.float64, i, tokens=ref["tokens"].repeat(k, 1), copies=k,
                         mode=mode, norm=torch.tensor(ref["norm"])))
        for key, v in mode.census.items():
            census[key] = census.get(key, 0) + v
    unknown = {key: v for key, v in census.items() if key.startswith("UNKNOWN")}
    if unknown:
        print("unknown ops", unknown, flush=True)
    return {key: np.concatenate([q[key] for q in parts], 0)
            for key in ("pre", "norm_pre", "flux", "logit")}


c64 = make(torch.float64)

# Each object is saved as it finishes, to PARTS/row<i>.safetensors beside OUT, and a
# rerun skips the objects already saved there with the same n and scales; OUT is
# assembled from the parts once every object has one.
parts = os.path.splitext(os.path.abspath(a.out))[0] + ".parts"
os.makedirs(parts, exist_ok=True)
meta = {"script": "apps/aion/test/gen/spectrum_ensemble.py", "torch": torch.__version__,
        "upstream": "PolymathicAI/AION db7fba6df53b", "n": str(a.n), "scales": ",".join(SCALES),
        "set": a.set}


def saved(i):
    path = f"{parts}/row{i}.safetensors"
    if not os.path.exists(path):
        return None
    from safetensors import safe_open
    with safe_open(path, "np") as f:
        m = f.metadata() or {}
    return path if (m.get("n"), m.get("scales")) == (str(a.n), ",".join(SCALES)) else None


t0 = time.time()
for j, i in enumerate(rows):
    if saved(i):
        print(f"{a.set}: object {j + 1}/{len(rows)} (row {i}) already saved", flush=True)
        continue
    obj = {}
    ref = run(c64, torch.float64, i)
    for k in ("pre", "norm_pre", "norm", "flux", "logit"):
        obj["ref." + k] = ref[k]
    obj["ref.tokens"] = ref["tokens"].numpy()
    obj["ref.mask"] = (torch.round(torch.sigmoid(torch.tensor(ref["logit"]))) != 0).numpy().astype(np.uint8)
    for si, name in enumerate(SCALES):
        s = ensemble(i, ref, name, 1000 + 100003 * si + 1009 * i)
        for k in ("pre", "norm_pre", "flux", "logit"):
            d = s[k] - ref[k]
            obj[f"s.{name}.{k}"] = np.sqrt((d ** 2).mean(0, keepdims=True))
    tmp = f"{parts}/row{i}.tmp"
    save_file({k: np.ascontiguousarray(v) for k, v in obj.items()}, tmp, metadata=meta)
    os.replace(tmp, f"{parts}/row{i}.safetensors")
    print(f"{a.set}: object {j + 1}/{len(rows)} (row {i}) saved at {time.time() - t0:.0f} s", flush=True)

objs = [load_file(saved(i)) for i in rows]
out = {"rows": np.array(rows, np.int64), "n": np.array([a.n], np.int64)}
for k in objs[0]:
    out[k] = np.ascontiguousarray(np.concatenate([o[k] for o in objs], 0))
save_file(out, a.out, metadata=meta)
print("wrote", a.out, "in", f"{time.time() - t0:.0f} s", flush=True)
