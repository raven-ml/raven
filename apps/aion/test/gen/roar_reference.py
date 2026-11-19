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
"""Upstream's random-order generation (ROAR) on aion-base, recorded for the port's parity
program roar_parity.exe. Upstream polymathic-aion at db7fba6d, torch CPU.

    uv run test/gen/roar_reference.py g1 CODECS OUT
    uv run test/gen/roar_reference.py z CODECS OUT COND
    uv run test/gen/roar_reference.py ensemble-time CODECS OUT

CODECS is the float64 reference's codecs.safetensors (its tokens); OUT a directory.

g1: four Gaia stars (the subset's gaia rows 0-3, their XP coefficients' tokens) to a
DESI spectrum, upstream's ROAR at linear 8, temperature 1, seed 0, run in float64
(GenerationSampler.generate, one order shared by the batch, as upstream draws it). The
states before steps 0 (k = 35, nothing revealed) and 4 (k = 34, 137 revealed) are kept:
the target's tokens, which are revealed, and the step's positions. Each state's logits
are recomputed from the state in float64 and float32 (checked against the run's own
float64 logits, bit for bit), as the decoder holding exactly the step's positions.
Writes OUT/g1.safetensors: s<i>.tokens (int64 [4; 273]), s<i>.revealed (uint8),
s<i>.positions (int64 [4; k]), f64.s<i> and f32.s<i> ([4; k; 1024]), final (the run's
tokens after its last step, int64 [4; 273]).

z COND: the redshift posterior of AION-1 §7.1.1 on the subset's 64 Legacy Survey objects
with DESI redshifts, from one of three conditionings: phot (the four g, r, i, z fluxes),
image_phot (the image and the four fluxes) and desi (the DESI spectrum). Logits of tok_z
from upstream's ROAR step (one token) with the float32 model as shipped, and 50 draws
per object by upstream's generate at temperature 1 with seeds j * 10^6 (j = 0..49):
consecutive seeds correlate upstream's runs (seed + step), so the seeds are spaced. The
logits are computed once and served to generate's step, which still seeds, draws its
order and calls torch.multinomial as upstream does. Writes OUT/z.COND.safetensors: f32
([64; 1025]), draws (int64 [64; 50]), z_true_token (int64 [64]). The float64 logits of
these sets are the transformer reference's (eight objects), which G1 reads.

ensemble-time: the cost of the rounding ensemble for G1. Times one float64 run of
each state for one object under perturb's dispatch mode (order scale, float32's u),
after one untimed run, and prints the cost of n = 32 runs plus the reference for 4
objects and 2 states. A timing: run it on a quiet machine.
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from aion.model import AION
from aion.fourm.generate import (GenerationSampler, build_chained_generation_schedules,
                                 init_empty_target_modality, init_full_input_modality)

torch.set_num_threads(2)
torch.set_grad_enabled(False)
mode, codecs, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
toks = load_file(codecs)
BP, RP, DESI, Z = "tok_xp_bp", "tok_xp_rp", "tok_spectrum_desi", "tok_z"
META = {"script": "apps/aion/test/gen/roar_reference.py", "torch": torch.__version__,
        "upstream": "PolymathicAI/AION db7fba6df53b"}


def load(dt):
    torch.set_default_dtype(dt)
    try:
        return AION.from_pretrained("polymathic-ai/aion-base").to(dt).eval()
    finally:
        torch.set_default_dtype(torch.float32)


def schedule(info, cond, target, n, steps):
    return build_chained_generation_schedules(
        cond_domains=cond, target_domains=[target], tokens_per_target=[n],
        autoregression_schemes=["roar"], decoding_steps=[steps],
        token_decoding_schedules=["linear"], temps=[1.0], temp_schedules=["constant"],
        cfg_scales=[1.0], cfg_schedules=["constant"], modality_info=info)


def mod_dict(info, inputs, target, n_target):
    d = {}
    b = None
    for k, v in inputs.items():
        d[k] = {"tensor": v.clone()}
        d = init_full_input_modality(d, info, k, "cpu")
        b = v.shape[0]
    return init_empty_target_modality(d, info, target, b, n_target, "cpu")


def step(m, d, target, rows):
    """Upstream's forward_enc_dec_roar_batched with the decoder holding each object's
    positions `rows` (B, k), gathered row by row; equal to upstream's on shared rows."""
    s = GenerationSampler(m)
    enc = {k: m.encoder_embeddings[k](v) for k, v in d.items() if k in m.encoder_embeddings}
    et, ee, em, _ = s.forward_mask_encoder_generation(enc)
    x = m.forward_encoder(et + ee, em)
    context = m.decoder_proj_context(x) + ee
    dd = {target: m.decoder_embeddings[target].forward_embed(d[target])}
    emb = dd[target]["emb"]
    idx = rows.unsqueeze(-1).expand(-1, -1, emb.shape[-1])
    g = torch.gather(emb, 1, idx)
    y = torch.zeros_like(g) + m.mask_token + g
    y = m.forward_decoder(y, context, em, None)
    mm = torch.full(rows.shape, m.modality_info[target]["id"], dtype=torch.int16)
    return m.forward_logits(y, dd, mm)[target].reshape(rows.shape[0], rows.shape[1], -1)


def gaia(n):
    return {k: torch.as_tensor(toks[f"gaia.{k}.tokens"][:n]).long() for k in (BP, RP)}


def state_dict(info, xp, tokens, revealed):
    d = mod_dict(info, xp, DESI, 273)
    d[DESI]["tensor"] = torch.as_tensor(tokens).to(d[DESI]["tensor"].dtype)
    d[DESI]["input_mask"] = torch.as_tensor(~revealed)
    d[DESI]["target_mask"] = torch.as_tensor(revealed)
    return d


KEEP = (0, 4)

if mode == "g1":
    m64 = load(torch.float64)
    xp = gaia(4)
    sampler = GenerationSampler(m64)
    rec = []
    orig = sampler.forward_enc_dec_roar_batched

    def spy(md, target, num_select, seed=None):
        tokens = md[target]["tensor"].clone().long().numpy()
        revealed = (~md[target]["input_mask"]).numpy().copy()
        lg, pos = orig(md, target, num_select, seed=seed)
        rec.append((tokens, revealed, pos.clone().numpy(), lg.clone().double().numpy()))
        return lg, pos

    sampler.forward_enc_dec_roar_batched = spy
    torch.set_default_dtype(torch.float64)
    try:
        final = sampler.generate(mod_dict(m64.modality_info, xp, DESI, 273),
                                 schedule(m64.modality_info, [BP, RP], DESI, 273, 8), seed=0)
    finally:
        torch.set_default_dtype(torch.float32)
    print("steps", len(rec), "counts", [r[2].shape[1] for r in rec], flush=True)
    data = {"final": final[DESI]["tensor"].long().numpy()}
    m32 = load(torch.float32)
    for s in KEEP:
        tokens, revealed, pos, lg = rec[s]
        torch.set_default_dtype(torch.float64)
        try:
            r64 = step(m64, state_dict(m64.modality_info, xp, tokens, revealed), DESI,
                       torch.as_tensor(pos)).numpy()
        finally:
            torch.set_default_dtype(torch.float32)
        r32 = step(m32, state_dict(m32.modality_info, xp, tokens, revealed), DESI,
                   torch.as_tensor(pos)).double().numpy()
        d = np.abs(r64 - lg).max()
        print(f"s{s}: k {pos.shape[1]}, revealed {int(revealed[0].sum())}, recomputed vs the"
              f" run's logits max |diff| {d:.3g}, f32 vs f64 max {np.abs(r32 - r64).max():.3g}",
              flush=True)
        if d != 0.0:
            raise SystemExit("the recomputed step is not the run's")
        data[f"s{s}.tokens"] = tokens.astype(np.int64)
        data[f"s{s}.revealed"] = revealed.astype(np.uint8)
        data[f"s{s}.positions"] = pos.astype(np.int64)
        data[f"f64.s{s}"] = r64
        data[f"f32.s{s}"] = r32.astype(np.float32)
    save_file({k: np.ascontiguousarray(v) for k, v in data.items()},
              os.path.join(out, "g1.safetensors"), metadata=META)
    print("wrote", os.path.join(out, "g1.safetensors"))

elif mode == "z":
    N, DRAWS, SPACING, CHUNK = 64, 50, 10 ** 6, 8
    flux = ["tok_flux_g", "tok_flux_r", "tok_flux_i", "tok_flux_z"]
    c = sys.argv[4]
    keys = {"phot": flux, "image_phot": ["tok_image"] + flux, "desi": [DESI]}[c]

    def inputs(lo, hi):
        res = {}
        for k in keys:
            t = torch.as_tensor(toks[f"ls_desi.{k}.tokens"][lo:hi]).long()
            res[k] = t.reshape(hi - lo, -1)
        return res

    data = {"z_true_token": toks["ls_desi.tok_z.tokens"][:N].astype(np.int64)}
    m = load(torch.float32)
    sampler = GenerationSampler(m)
    parts = []
    for lo in range(0, N, CHUNK):
        d = mod_dict(m.modality_info, inputs(lo, lo + CHUNK), Z, 1)
        lg, _ = sampler.forward_enc_dec_roar_batched(d, Z, 1, seed=0)
        parts.append(lg.reshape(lg.shape[0], -1).numpy())
        print(f"{c}: logits of objects {lo}-{lo + CHUNK - 1}", flush=True)
    data["f32"] = np.concatenate(parts).astype(np.float32)
    # Upstream's draws, by its own generate, the step served these logits.
    lg32 = torch.as_tensor(data["f32"]).reshape(N, 1, -1)

    def served(md, target, num_select, seed=None):
        dd = {target: m.decoder_embeddings[target].forward_embed(md[target])}
        *_, pos = sampler.forward_mask_decoder_roar(dd, target, num_select, seed=seed)
        return lg32, pos

    sampler.forward_enc_dec_roar_batched = served
    md = mod_dict(m.modality_info, inputs(0, N), Z, 1)
    sch = schedule(m.modality_info, keys, Z, 1, 1)
    draws = np.stack([sampler.generate(md, sch, seed=j * SPACING)[Z]["tensor"]
                      .long().numpy().reshape(N) for j in range(DRAWS)], axis=1)
    data["draws"] = draws.astype(np.int64)
    print(f"{c}: draws {draws.shape}, of class 1024: {int((draws == 1024).sum())}",
          flush=True)
    save_file({k: np.ascontiguousarray(v) for k, v in data.items()},
              os.path.join(out, f"z.{c}.safetensors"), metadata=META)
    print("wrote", os.path.join(out, f"z.{c}.safetensors"))

elif mode == "ensemble-time":
    import perturb
    from perturb import U32, Ensemble
    perturb.NOROUND.add("embedding")
    g1 = load_file(os.path.join(out, "g1.safetensors"))
    m64 = load(torch.float64)
    xp = gaia(1)
    for s in KEEP:
        tokens = g1[f"s{s}.tokens"][:1]
        revealed = g1[f"s{s}.revealed"][:1].astype(bool)
        pos = torch.as_tensor(g1[f"s{s}.positions"][:1])
        times = []
        for rep in range(3):
            torch.set_default_dtype(torch.float64)
            try:
                t0 = time.perf_counter()
                with Ensemble("order", 4200 + rep, U32):
                    step(m64, state_dict(m64.modality_info, xp, tokens, revealed), DESI, pos)
                times.append(time.perf_counter() - t0)
            finally:
                torch.set_default_dtype(torch.float32)
        t = sorted(times[1:])
        print(f"s{s}: one perturbed float64 run of one object, untimed first then"
              f" {t} s; median {np.median(t):.2f} s", flush=True)
        print(f"  G1 ensemble, 33 runs x 4 objects: {33 * 4 * np.median(t) / 60:.1f} min",
              flush=True)
else:
    raise SystemExit("mode: g1, z or ensemble-time")
