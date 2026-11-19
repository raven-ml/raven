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
"""Fixture for test_roar: upstream's ROAR step on the narrow transformer, and the rounding
ensemble of its logits.

    uv run test/gen/roar.py DIR [--n 32]

DIR is test/upstream, which holds transformer.py's narrow AION
(transformer.safetensors) and its inputs (transformer_golden.safetensors). The model is
rebuilt from that file with upstream's code (polymathic-aion at db7fba6d): the encoder's
modality embeddings from the decoder's, as the file aliases them; the decoder's token
tables no path reads stay random. The rebuild is checked against transformer.py's float64 logits.

A ROAR step (fourm/generate.py, forward_enc_dec_roar_batched) encodes the inputs followed
by the target, present where revealed, and decodes the target's mask token at k chosen
positions, the decoder holding those k rows only, with no self-attention mask. Target
tok_xp_rp (55 tokens), from the fixture's inputs (tok_xp_bp, tok_flux_g, tok_z, with
their absent tokens), four objects:

  roar.0  the first step of linear 8 (k = 7), nothing revealed, the positions upstream
          draws (one order shared by the batch, seed 0);
  roar.1  a later step, k = 7: each object has its own 28 revealed positions holding
          random tokens, and its own 7 of the others.

roar.0 is upstream's forward_enc_dec_roar_batched; roar.1, whose positions differ by
object, which upstream's step cannot take, is the same computation with each row's
positions gathered (`step` below), checked equal to upstream's on roar.0 bit for bit.

Writes DIR/roar_golden.safetensors: roar.<i>.tokens (int64 [4; 55], the target's tokens,
0 where unrevealed), roar.<i>.revealed (uint8 [4; 55]), roar.<i>.positions (int64
[4; 7]), f64.roar.<i> (the float64 logits [4; 7; 1024]), s.roar.<i> (float32, the root
mean square distance from it of n float64 runs under perturb's dispatch mode at the
order model's float32 scale), and n. It prints PyTorch float32's largest distance from
the reference over the spread. Its metadata names
this script, torch's version and upstream's commit.
"""
import argparse
import os
import sys
import types
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from aion.model import AION
from aion.fourm.generate import GenerationSampler, init_full_input_modality
import perturb
from perturb import U32, Ensemble

perturb.NOROUND.add("embedding")

p = argparse.ArgumentParser()
p.add_argument("dir")
p.add_argument("--n", type=int, default=32)
a = p.parse_args()

torch.set_num_threads(2)
torch.set_grad_enabled(False)

MODS = ["tok_xp_bp", "tok_flux_g", "tok_z", "tok_xp_rp"]
INPUTS = ["tok_xp_bp", "tok_flux_g", "tok_z"]
TARGET = "tok_xp_rp"
N = 55
K = 7
D = 16
config = dict(act_layer="SiLU", decoder_depth=2, encoder_depth=2, dim=D, num_heads=2,
              mlp_ratio=4, gated_mlp=True, mlp_bias=False, norm_bias=False, proj_bias=False,
              qk_norm=True, qkv_bias=False, domains_in=MODS, domains_out=MODS)
model = AION(config).eval()
weights = load_file(os.path.join(a.dir, "transformer.safetensors"))
state = model.state_dict()
for k in state:
    if k in weights:
        src = weights[k]
    elif k.startswith("encoder_embeddings") and k.endswith("mod_emb"):
        src = weights["decoder_embeddings" + k[len("encoder_embeddings"):]]
    elif k.startswith("decoder_embeddings") and k.endswith("token_emb.weight"):
        continue
    else:
        raise SystemExit(f"no weight for {k}")
    state[k] = torch.from_numpy(src.copy())
model.load_state_dict(state)
golden = load_file(os.path.join(a.dir, "transformer_golden.safetensors"))
tokens = OrderedDict((k, golden[f"input.{k}.tokens"]) for k in INPUTS)
absent = OrderedDict((k, golden[f"input.{k}.absent"].astype(bool)) for k in INPUTS)
B = tokens["tok_xp_bp"].shape[0]

torch.set_default_dtype(torch.float64)
model64 = AION(config).eval()
model64.load_state_dict({k: v.double() if v.is_floating_point() else v
                         for k, v in model.state_dict().items()})
torch.set_default_dtype(torch.float32)


def check_rebuild():
    """p3's forward of objects 1 and 2 to tok_z, one token, where the decoder's mask
    changes nothing: it must give p3's float64 logits."""
    torch.set_default_dtype(torch.float64)
    try:
        rows = [1, 2]
        lg = model64(OrderedDict((k, torch.tensor(v[rows])) for k, v in tokens.items()),
                     target_modality=[types.SimpleNamespace(token_key="tok_z", num_tokens=1)],
                     input_mask={k: torch.tensor(v[rows]) for k, v in absent.items()})
    finally:
        torch.set_default_dtype(torch.float32)
    d = np.abs(lg["tok_z"].numpy() - golden["f64.logits.tok_z"]).max()
    print(f"rebuild: |tok_z logits - p3's f64| max {d:.3g}", flush=True)
    if d > 1e-12:
        raise SystemExit("the rebuilt model is not p3's")


def mod_dict(m, tgt_tokens, revealed, copies=1):
    rep = lambda x: torch.tensor(np.tile(x, (copies, 1)))
    d = OrderedDict()
    for k in INPUTS:
        d[k] = {"tensor": rep(tokens[k])}
        d = init_full_input_modality(d, m.modality_info, k, "cpu")
        d[k]["input_mask"] = rep(absent[k])
    d[TARGET] = {"tensor": rep(tgt_tokens), "input_mask": rep(~revealed),
                 "target_mask": rep(revealed),
                 "decoder_attention_mask": torch.zeros((B * copies, N), dtype=torch.bool)}
    return d


def step(m, d, rows):
    """Upstream's forward_enc_dec_roar_batched with the decoder holding each object's
    own positions `rows` (B, k), gathered row by row."""
    s = GenerationSampler(m)
    enc = {k: m.encoder_embeddings[k](v) for k, v in d.items() if k in m.encoder_embeddings}
    et, ee, em, _ = s.forward_mask_encoder_generation(enc)
    x = m.forward_encoder(et + ee, em)
    context = m.decoder_proj_context(x) + ee
    dd = {TARGET: m.decoder_embeddings[TARGET].forward_embed(d[TARGET])}
    emb = dd[TARGET]["emb"]
    idx = rows.unsqueeze(-1).expand(-1, -1, emb.shape[-1])
    g = torch.gather(emb, 1, idx)
    y = torch.zeros_like(g) + m.mask_token + g
    y = m.forward_decoder(y, context, em, None)
    mm = torch.full(rows.shape, m.modality_info[TARGET]["id"], dtype=torch.int16)
    return m.forward_logits(y, dd, mm)[TARGET].reshape(rows.shape[0], rows.shape[1], -1)


check_rebuild()
rng = np.random.default_rng(4)
states = []

# roar.0: upstream's own step, nothing revealed.
tgt0 = np.zeros((B, N), np.int64)
rev0 = np.zeros((B, N), bool)
torch.set_default_dtype(torch.float64)
try:
    lg_up, pos = GenerationSampler(model64).forward_enc_dec_roar_batched(
        mod_dict(model64, tgt0, rev0), TARGET, K, seed=0)
    lg_step = step(model64, mod_dict(model64, tgt0, rev0), pos)
finally:
    torch.set_default_dtype(torch.float32)
dd = (lg_up - lg_step).abs().max().item()
print(f"roar.0: upstream's step against the row gather, max |diff| {dd:.3g}", flush=True)
if dd != 0.0:
    raise SystemExit("the row gather is not upstream's step")
states.append((tgt0, rev0, pos.numpy()))

# roar.1: each object's own revealed set and positions.
tgt1 = np.zeros((B, N), np.int64)
rev1 = np.zeros((B, N), bool)
pos1 = np.zeros((B, K), np.int64)
for i in range(B):
    perm = rng.permutation(N)
    rev1[i, perm[:28]] = True
    tgt1[i, perm[:28]] = rng.integers(0, 1024, 28)
    pos1[i] = perm[28:28 + K]
states.append((tgt1, rev1, pos1))


def run(m, dt, tgt, rev, pos, copies=1, mode=None):
    torch.set_default_dtype(dt)
    ctx = mode if mode is not None else torch.no_grad()
    try:
        with ctx:
            lg = step(m, mod_dict(m, tgt, rev, copies), torch.tensor(np.tile(pos, (copies, 1))))
    finally:
        torch.set_default_dtype(torch.float32)
    return lg.detach().double().numpy()


data = {"n": np.array([a.n], np.int64)}
for i, (tgt, rev, pos) in enumerate(states):
    name = f"roar.{i}"
    ref = run(model64, torch.float64, tgt, rev, pos)
    f32 = run(model, torch.float32, tgt, rev, pos)
    s = run(model64, torch.float64, tgt, rev, pos, copies=a.n,
            mode=Ensemble(4100 + i, U32))
    spread = np.sqrt(((s.reshape((a.n,) + ref.shape) - ref) ** 2).mean(0))
    data[f"{name}.tokens"] = tgt
    data[f"{name}.revealed"] = rev.astype(np.uint8)
    data[f"{name}.positions"] = pos.astype(np.int64)
    data[f"f64.{name}"] = ref
    data[f"s.{name}"] = spread.astype(np.float32)
    z = np.abs(f32 - ref) / spread
    print(f"{name}: positions {pos.tolist()}, |ref| max {np.abs(ref).max():.3g}, spread"
          f" median {np.median(spread):.3g}, float32 torch max z {z.max():.2f}", flush=True)

meta = {"script": "apps/aion/test/gen/roar.py", "torch": torch.__version__,
        "upstream": "PolymathicAI/AION db7fba6df53b", "seed": "numpy 4, torch 0, ensemble 4100+i"}
save_file({k: np.ascontiguousarray(v) for k, v in data.items()},
          os.path.join(a.dir, "roar_golden.safetensors"), metadata=meta)
print("wrote", os.path.join(a.dir, "roar_golden.safetensors"))
