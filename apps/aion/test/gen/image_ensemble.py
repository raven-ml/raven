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
"""The rounding ensemble of upstream's image codec, per parity set, for
image_parity.exe.

    uv run test/gen/image_ensemble.py --goldens GOLDENS --subset SUBSET \
        SET OUT [--objects 16] [--n 32] [--scales order,order64,contract,measured]
        [--chunk 8] [--time R]

SET is golden (upstream's tests/test_data image input, channels 5-8, the Legacy
Survey's), ls_desi or ls_catalog (the parity subset's Legacy Survey images), or hsc
(its HSC images). The objects are the set's first rows.

For each object it runs upstream's ImageCodec (aion-base, polymathic-aion db7fba6d):
  - in float64, unmoved: the reference, ref.*: the FSQ input pre (5, 576), its bound
    bounded (576, 5), the tokens (576) and the decoding of those tokens in the set's
    bands, decoded (c, 96, 96);
  - in float32 (PyTorch's own float32), f32.*: pre, bounded, tokens, and decoded
    from the reference's tokens;
  - in float64 under perturb's dispatch mode, n times per scale, each sample
    decoding the reference's tokens; s.<scale>.* is the root mean square of the
    samples' distance from the reference, per value (pre, bounded, decoded).
Scales: order (u = 2^-24, the float32 check's gate), order64 (the order model at
u = 2^-53, the float64 check's gate), contract and measured (reported only). The
n samples of an object run in batches of --chunk copies of it, each element
drawing its own noise. Each object is saved as it finishes, in OUT.parts/, and a
rerun skips the objects saved there (a part holds one object's arrays at the
scales given; delete a set's parts to change them). --time R prints the elapsed
and CPU time of R ensembles of one object at each scale and exits (run it on a
quiet machine).

Upstream encodes in place, so each run gets a fresh copy of its input.
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
p.add_argument("set", choices=["golden", "ls_desi", "ls_catalog", "hsc"])
p.add_argument("out")
p.add_argument("--objects", type=int, default=16)
p.add_argument("--n", type=int, default=32)
p.add_argument("--scales", default="order,order64,contract,measured")
p.add_argument("--time", type=int, default=0)
p.add_argument("--chunk", type=int, default=8, help="samples per batch")
p.add_argument("--goldens", required=True, help="goldens.py's directory")
p.add_argument("--subset", required=True, help="subset.py's directory")
a = p.parse_args()
SCALES = {"order": ("order", U32), "order64": ("order", U64), "contract": ("contract", U32),
          "measured": ("measured", U32)}
LS = ["DES-G", "DES-R", "DES-I", "DES-Z"]
HSC = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]

if a.set == "golden":
    images = load_file(f"{a.goldens}/image_codec_input_batch.safetensors")["image.array"][:, 5:]
    bands, cls = LS, Mod.LegacySurveyImage
elif a.set == "hsc":
    images = load_file(f"{a.subset}/hsc.safetensors")["image_array"]
    bands, cls = HSC, Mod.HSCImage
else:
    t = load_file(f"{a.subset}/{a.set}.safetensors")
    images = t["legacysurvey.image_array" if a.set == "ls_desi" else "image_array"]
    bands, cls = LS, Mod.LegacySurveyImage
rows = list(range(a.objects))


def image(i, dt, copies=1):
    x = np.repeat(np.asarray(images[i:i + 1], np.float32), copies, 0)
    return cls(flux=torch.tensor(x).to(dt), bands=bands)


def make(dt):
    torch.set_default_dtype(dt)
    c = MODALITY_CODEC_MAPPING[cls].from_pretrained(HF_REPO_ID, modality=cls)
    c = c.to(dt).eval()
    torch.set_default_dtype(torch.float32)
    return c


def run(codec, dt, i, tokens=None, copies=1, mode=None):
    """pre, bounded, tokens and decoded of object i (copies of it)."""
    torch.set_default_dtype(dt)
    try:
        ctx = mode if mode is not None else torch.no_grad()
        with ctx:
            z = codec._encode(image(i, dt, copies))
            y = codec.quantizer._bound(z.moveaxis(1, -1))
        if tokens is None:
            tokens = codec.encode(image(i, dt, copies))
        with ctx:
            d = codec.decode(tokens, bands=bands).flux
    finally:
        torch.set_default_dtype(torch.float32)
    f = lambda x: x.detach().double().numpy()
    return dict(pre=f(z), bounded=f(y), tokens=tokens, decoded=f(d))


KEYS = ("pre", "bounded", "decoded")


def ensemble(i, ref, name, seed):
    """The n perturbed runs of object i at scale name, in batches of --chunk."""
    scale, u = SCALES[name]
    parts, census = [], {}
    for lo in range(0, a.n, a.chunk):
        k = min(a.chunk, a.n - lo)
        mode = Ensemble(scale, seed + lo, u)
        parts.append(run(c64, torch.float64, i, tokens=ref["tokens"].repeat(k, 1), copies=k,
                         mode=mode))
        for key, v in mode.census.items():
            census[key] = census.get(key, 0) + v
    unknown = {key: v for key, v in census.items() if key.startswith("UNKNOWN")}
    if unknown:
        print("unknown ops", unknown, flush=True)
    return {key: np.concatenate([q[key] for q in parts], 0) for key in KEYS}


c64 = make(torch.float64)
c32 = make(torch.float32)

if a.time:
    ref = run(c64, torch.float64, rows[0])
    for name in a.scales.split(","):
        w, c = [], []
        for r in range(a.time):
            t0, p0 = time.perf_counter(), time.process_time()
            ensemble(rows[0], ref, name, 7 + 1000 * r)
            w.append(time.perf_counter() - t0)
            c.append(time.process_time() - p0)
        print(f"{a.set} row {rows[0]}, {name}, n = {a.n} samples in batches of {a.chunk}: elapsed "
              f"median {np.median(w):.1f} s (min {min(w):.1f}, max {max(w):.1f}), cpu median "
              f"{np.median(c):.1f} s; {np.median(w) / a.n:.2f} s a sample", flush=True)
    sys.exit(0)

out = {"rows": np.array(rows, np.int64), "n": np.array([a.n], np.int64)}
acc = {}
t0 = time.time()
# Each object is saved as it finishes, under OUT.parts/, and a rerun reads the
# objects already saved instead of computing them again.
parts = a.out + ".parts"
os.makedirs(parts, exist_ok=True)
for j, i in enumerate(rows):
    part = os.path.join(parts, f"{i}.safetensors")
    if os.path.exists(part):
        one = load_file(part)
        print(f"{a.set}: object {j + 1}/{len(rows)} (row {i}) read from {part}", flush=True)
    else:
        one = {}
        ref = run(c64, torch.float64, i)
        f32 = run(c32, torch.float32, i, tokens=ref["tokens"])
        for k in KEYS:
            one["ref." + k] = ref[k]
            one["f32." + k] = f32[k]
        one["ref.tokens"] = ref["tokens"].to(torch.int64).numpy()
        one["f32.tokens"] = c32.encode(image(i, torch.float32)).to(torch.int64).numpy()
        for si, name in enumerate(a.scales.split(",")):
            s = ensemble(i, ref, name, 1000 + 100003 * si + 1009 * i)
            for k in KEYS:
                d = s[k] - ref[k]
                one[f"s.{name}.{k}"] = np.sqrt((d ** 2).mean(0, keepdims=True))
        one = {k: np.ascontiguousarray(v) for k, v in one.items()}
        save_file(one, part + ".tmp", metadata={"set": a.set, "row": str(i), "n": str(a.n),
                                                "scales": a.scales})
        os.replace(part + ".tmp", part)
        print(f"{a.set}: object {j + 1}/{len(rows)} (row {i}) done at {time.time() - t0:.0f} s",
              flush=True)
    for k, v in one.items():
        acc.setdefault(k, []).append(v)

for k, v in acc.items():
    out[k] = np.ascontiguousarray(np.concatenate(v, 0))
meta = {"script": "apps/aion/test/gen/image_ensemble.py", "torch": torch.__version__,
        "upstream": "PolymathicAI/AION db7fba6df53b", "n": str(a.n), "scales": a.scales,
        "set": a.set, "bands": ",".join(bands)}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
save_file(out, a.out, metadata=meta)
print("wrote", a.out, "in", f"{time.time() - t0:.0f} s")
