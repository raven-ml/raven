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
"""Fixtures for test_transformer: a narrow upstream AION, its outputs and their rounding
ensemble.

    uv run test/gen/transformer.py OUT [--n 32]

OUT is test/upstream.

Builds upstream's AION (polymathic-aion at db7fba6d) with aion-base's structure (SwiGLU
without biases, query-key norm, layer norms without shift at eps 1e-6, a context
projection with a bias) at small sizes: width 16, 2 heads, 2 encoder and 2 decoder
blocks, over four modalities (tok_xp_bp, tok_flux_g, tok_z, tok_xp_rp). Every weight
is random (torch.manual_seed(0)), designed so that each constant matters:

- token, positional, modality and mask embeddings are scaled by 1e-3, so the first
  block's layer norm, in the encoder and in the decoder, sees a variance near its
  eps 1e-6;
- query and key projections (self- and cross-attention) are scaled by 1e-3, so the
  query-key norms see a variance near their eps, and without them the scores would be
  near zero;
- layer norms' and query-key norms' scales are 1 + U(-0.5, 0.5);
- the context projection's bias is U(-0.5, 0.5);
- the decoder's token tables, which no path reads, hold N(0, 1); the file keeps
  tok_z's alone.

The positional buffers are upstream's sine-cosine tables times 1e-3, the same in the
encoder and the decoder; the modality embeddings are shared, and the weights file
stores each under its decoder name only, as aion-base's file does.

Inputs: four objects, tok_xp_bp (55 tokens), tok_flux_g and tok_z, some absent
(upstream's input mask): object 1 lacks tok_xp_bp, object 2 lacks tok_flux_g and
tok_xp_bp's tokens 10-19, object 3 lacks tok_z. Outputs, in float64 (the reference,
torch's default dtype float64, weights cast exactly):

- embed: AION.encode of the four objects, with num_encoder_tokens the input's width;
- logits.tok_z, logits.tok_xp_rp: AION.forward of objects 1 and 2 to both targets,
  with the decoder mask of training (the target count at each target's first
  token), tok_xp_rp's for object 2 alone;

and for each, s.<name>: the root mean square distance from the reference of n float64
runs under perturb's dispatch mode at the order model's float32 scale. Writes:

- OUT/transformer.safetensors: the weights, float32, upstream's names.
- OUT/transformer_golden.safetensors: input.<key>.{tokens,absent} (int64 and uint8,
  [4; tokens]), f64.embed, f64.logits.<key>, s.embed, s.logits.<key> (float32), n.

Its metadata names this script, torch's version and upstream's commit.
"""
import argparse
import copy
import os
import random
import sys
import types
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from safetensors.numpy import save_file

from aion.model import AION
import perturb
from perturb import U32, Ensemble

# An embedding lookup rounds nothing.
perturb.NOROUND.add("embedding")

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--n", type=int, default=32)
a = p.parse_args()

torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)

MODS = ["tok_xp_bp", "tok_flux_g", "tok_z", "tok_xp_rp"]
INPUTS = ["tok_xp_bp", "tok_flux_g", "tok_z"]
TARGETS = [types.SimpleNamespace(token_key="tok_z", num_tokens=1),
           types.SimpleNamespace(token_key="tok_xp_rp", num_tokens=55)]
D = 16
config = dict(act_layer="SiLU", decoder_depth=2, encoder_depth=2, dim=D, num_heads=2,
              mlp_ratio=4, gated_mlp=True, mlp_bias=False, norm_bias=False, proj_bias=False,
              qk_norm=True, qkv_bias=False, domains_in=MODS, domains_out=MODS)
model = AION(config).eval()

SMALL = 1e-3


def u(shape):
    return torch.rand(shape) - 0.5


for name, prm in model.named_parameters():
    if name.endswith("mod_emb") or name == "mask_token" or (
            name.startswith("encoder_embeddings") and name.endswith("token_emb.weight")):
        v = torch.randn(prm.shape) * SMALL
    elif name.startswith("decoder_embeddings") and name.endswith("token_emb.weight"):
        v = torch.randn(prm.shape)
    elif prm.ndim == 1 and "norm" in name:
        v = u(prm.shape) + 1.0
    else:
        v = u(prm.shape)
        if name.endswith("qkv.weight"):
            v[: 2 * D] *= SMALL
        elif name.endswith("cross_attn.q.weight"):
            v *= SMALL
        elif name.endswith("cross_attn.kv.weight"):
            v[:D] *= SMALL
    prm.copy_(v)
for name, buf in model.named_buffers():
    if name.endswith("pos_emb"):
        buf.mul_(SMALL)

# The file stores each modality embedding under its decoder name only, as aion-base's
# does, and one of the decoder's token tables, which no path reads, to keep it small.
state = {k: v.detach().clone().contiguous().numpy() for k, v in model.state_dict().items()
         if not (k.startswith("encoder_embeddings") and k.endswith("mod_emb"))
         and not (k.startswith("decoder_embeddings") and k.endswith("token_emb.weight")
                  and k != "decoder_embeddings.tok_z.token_emb.weight")}

rng = np.random.default_rng(0)
B = 4
tokens = OrderedDict()
absent = OrderedDict()
for key in INPUTS:
    n = 55 if key == "tok_xp_bp" else 1
    vocab = 1025 if key == "tok_z" else 1024
    tokens[key] = rng.integers(0, vocab, (B, n), dtype=np.int64)
    absent[key] = np.zeros((B, n), dtype=bool)
absent["tok_xp_bp"][1, :] = True
absent["tok_flux_g"][2, :] = True
absent["tok_xp_bp"][2, 10:20] = True
absent["tok_z"][3, :] = True
WIDTH = sum(t.shape[1] for t in tokens.values())
FWD = [1, 2]


def embed_targets_patched(self, target_mask, num_decoder_tokens=256):
    """AION.embed_targets with the training decoder mask, as reference.py."""
    mods = {}
    for mod, mask in target_mask.items():
        mask = mask.to(torch.bool)
        tensor = torch.zeros_like(mask).to(torch.long)
        compressed = torch.zeros_like(mask).to(torch.long)
        compressed[:, 0] = (~mask).sum(dim=1)
        mods[mod] = self.decoder_embeddings[mod].forward_embed(
            {"tensor": tensor, "target_mask": mask, "decoder_attention_mask": compressed})
    return self.forward_mask_decoder(mods, num_decoder_tokens)


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
                         num_encoder_tokens=WIDTH)
            lg = m(OrderedDict((k, rep(v, FWD)) for k, v in tokens.items()),
                   target_modality=TARGETS,
                   input_mask={k: rep(v, FWD) for k, v in absent.items()})
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
s = run(model64, torch.float64, copies=a.n, mode=Ensemble("order", 4000, U32))
data = {"n": np.array([a.n], np.int64)}
for k in INPUTS:
    data[f"input.{k}.tokens"] = tokens[k]
    data[f"input.{k}.absent"] = absent[k].astype(np.uint8)
# The tok_xp_rp logits are kept for the forward pass's second object (object 2) only.
KEEP = {"logits.tok_xp_rp": slice(1, 2)}
for k, r in ref.items():
    d = s[k].reshape((a.n,) + r.shape) - r
    spread = np.sqrt((d ** 2).mean(0))
    rows = KEEP.get(k, slice(None))
    r, spread, f32[k] = r[rows], spread[rows], f32[k][rows]
    data[f"f64.{k}"] = r
    data[f"s.{k}"] = spread.astype(np.float32)
    z = np.abs(f32[k] - r) / spread
    print(f"{k}: shape {r.shape}, |ref| max {np.abs(r).max():.3g}, spread median"
          f" {np.median(spread):.3g}, float32 torch max z {z.max():.2f}", flush=True)

meta = {"script": "apps/aion/test/gen/transformer.py", "torch": torch.__version__,
        "upstream": "PolymathicAI/AION db7fba6df53b", "seed": "torch 0, numpy 0, random 0"}
os.makedirs(a.out, exist_ok=True)
save_file(state, os.path.join(a.out, "transformer.safetensors"), metadata=meta)
save_file({k: np.ascontiguousarray(v) for k, v in data.items()},
          os.path.join(a.out, "transformer_golden.safetensors"), metadata=meta)
print("wrote", a.out)
