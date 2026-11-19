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
"""Fixtures for test_image: a narrow upstream image codec, its outputs and their
rounding ensembles.

    uv run test/gen/image.py OUT [--n 32]

OUT is test/upstream.

Builds upstream's ImageCodec (polymathic-aion at db7fba6d) in a narrow configuration
(hidden_dims 8, so widths 2, 4 and 8; multisurvey_projection_dims 3; the shipped
n_compressions 2, num_consecutive 4, embedding_dim 5 and levels [7, 5, 5, 5, 5]) and
gives every parameter a random value (torch.manual_seed(0); uniform with variance
1 / fan-in for weights, uniform in [-0.5, 0.5] for biases, layer norm scales 1 more).
Designed so that each constant matters:
  - the squeeze-excite gates are open (upstream initialises their bias to -10, which
    closes them), and the SE's hidden activations take both signs, so the softmax,
    the LeakyReLU's slope and the sigmoid all move the latent;
  - the encoder's last convolution and the second convolution of each of its last
    four residual units are scaled by 3e-3, which puts the final layer norm's lane
    variance near its eps 1e-5;
  - two bands of nine, so the projection's scale is sqrt(9 / 2);
  - pixels normal at half each band's clamp, so the ELUs see both signs.

It runs on two images, Legacy Survey bands DES-G and DES-Z and HSC bands HSC-I and
HSC-Y, 96 x 96, in float64 (the reference, torch's default dtype float64, weights
cast exactly) and float32 (a positive control, printed), and n times in float64
under perturb's dispatch mode at the order model's float32 scale (order) and at
float64's (order64), each sample decoding the reference's tokens into the first
band. Writes:

- OUT/image_codec.safetensors: the codec's state dict, float32, upstream's names.
- OUT/image_golden.safetensors: input.<s> for <s> ls and hsc, float32 [1, 2, 96, 96];
  f64.<s>.{pre,bounded,tokens,decoded}: the FSQ input (1, 5, 576), its bound
  (1, 576, 5), the tokens (1, 576) and the decoding of the tokens in the first band
  (1, 1, 96, 96); s.<scale>.<s>.{pre,bounded,decoded}: the ensemble's root mean
  square distance from the reference, per value, rounded to float32; n.

Its metadata names this script, torch's version and upstream's commit.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from safetensors.numpy import save_file

from aion.codecs.image import ImageCodec
from aion import modalities as M
from perturb import U32, U64, Ensemble

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--n", type=int, default=32)
a = p.parse_args()

torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.manual_seed(0)

config = dict(quantizer_levels=[7, 5, 5, 5, 5], hidden_dims=8, multisurvey_projection_dims=3,
              n_compressions=2, num_consecutive=4, embedding_dim=5)
codec = ImageCodec(**config).eval()
for name, prm in codec.named_parameters():
    if name.endswith("bias"):
        v = torch.rand(prm.shape) - 0.5
    elif prm.ndim == 1:  # the layer norm's scale
        v = torch.rand(prm.shape) - 0.5 + 1.0
    else:
        v = (torch.rand(prm.shape) * 2 - 1) * math.sqrt(3.0 / prm[0].numel())
    prm.copy_(v)
small = ["model.encoder_layers.5.conv"] + [f"model.encoder_layers.6.{i}.net.2" for i in range(4)]
for name, prm in codec.named_parameters():
    if any(name.startswith(s + ".") for s in small):
        prm.mul_(3e-3)
state = {k: v.detach().clone().contiguous().numpy() for k, v in codec.state_dict().items()}

rng = np.random.default_rng(0)
CLAMP = {"DES-G": 6, "DES-Z": 25, "HSC-I": 200, "HSC-Y": 500}
sets = {"ls": ["DES-G", "DES-Z"], "hsc": ["HSC-I", "HSC-Y"]}
inputs = {}
for s, bands in sets.items():
    scale = np.array([CLAMP[b] for b in bands], dtype=np.float64)[None, :, None, None]
    inputs[s] = (rng.standard_normal((1, len(bands), 96, 96)) * 0.5 * scale).astype(np.float32)
data = {"n": np.array([a.n], np.int64)}
for s, x in inputs.items():
    data[f"input.{s}"] = x

torch.set_default_dtype(torch.float64)
codec64 = ImageCodec(**config).eval()
codec64.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
codec64 = codec64.to(torch.float64)
torch.set_default_dtype(torch.float32)


def run(c, dt, s, tokens=None, copies=1, mode=None):
    bands = sets[s]
    torch.set_default_dtype(dt)

    def image():  # upstream's encode writes into its input
        return M.Image(flux=torch.tensor(np.repeat(inputs[s], copies, 0)).to(dt), bands=bands)
    try:
        ctx = mode if mode is not None else torch.no_grad()
        with ctx:
            z = c._encode(image())
            y = c.quantizer._bound(z.moveaxis(1, -1))
        if tokens is None:
            tokens = c.encode(image())
        with ctx:
            d = c.decode(tokens, bands=bands[:1]).flux
    finally:
        torch.set_default_dtype(torch.float32)
    f = lambda t: t.detach().double().numpy()
    return dict(pre=f(z), bounded=f(y), tokens=tokens, decoded=f(d))


KEYS = ("pre", "bounded", "decoded")
for si, s in enumerate(sets):
    ref = run(codec64, torch.float64, s)
    for k in KEYS:
        data[f"f64.{s}.{k}"] = ref[k]
    data[f"f64.{s}.tokens"] = ref["tokens"].to(torch.int64).numpy()
    f32 = run(codec, torch.float32, s, tokens=ref["tokens"])
    k32 = codec.encode(M.Image(flux=torch.tensor(inputs[s]), bands=sets[s]))
    for name, u in [("order", U32), ("order64", U64)]:
        mode = Ensemble("order", 2000 + 10 * si + (u == U64), u)
        e = run(codec64, torch.float64, s, tokens=ref["tokens"].repeat(a.n, 1), copies=a.n, mode=mode)
        unknown = {k: v for k, v in mode.census.items() if k.startswith("UNKNOWN")}
        if unknown:
            print("unknown ops", unknown)
        for k in KEYS:
            d = e[k] - ref[k]
            data[f"s.{name}.{s}.{k}"] = np.sqrt((d ** 2).mean(0, keepdims=True)).astype(np.float32)
    z = {k: float((np.abs(f32[k] - ref[k]) / data[f"s.order.{s}.{k}"]).max()) for k in KEYS}
    y = ref["pre"]
    print(f"{s}: float32 tokens differ: {int((k32 != ref['tokens']).sum())}; float32 max "
          f"|diff| / s at order: {z}; distinct tokens {len(np.unique(ref['tokens']))}", flush=True)

# The final layer norm's lane variance, which the design puts near its eps.
hook = {}
h = codec64.model.encoder_layers[7].register_forward_hook(lambda m, i, o: hook.__setitem__("x", i[0]))
torch.set_default_dtype(torch.float64)
codec64._encode(M.Image(flux=torch.tensor(inputs["ls"]).double(), bands=sets["ls"]))
torch.set_default_dtype(torch.float32)
h.remove()
var = hook["x"].moveaxis(1, -1).var(-1, unbiased=False)
print("final norm's lane variance: median", float(var.median()), "min", float(var.min()),
      "max", float(var.max()))

meta = {"script": "apps/aion/test/gen/image.py", "torch": torch.__version__,
        "upstream": "PolymathicAI/AION db7fba6df53b", "seed": "torch 0, numpy 0"}
os.makedirs(a.out, exist_ok=True)
save_file(state, os.path.join(a.out, "image_codec.safetensors"), metadata=meta)
save_file({k: np.ascontiguousarray(v) for k, v in data.items()},
          os.path.join(a.out, "image_golden.safetensors"), metadata=meta)
print("wrote", a.out)
