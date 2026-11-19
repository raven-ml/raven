# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.14.1",
#   "numpy==2.5.5",
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
"""Time bench_aion.ml's rows with upstream's code (PolymathicAI/AION db7fba6df53b).

    uv run bench/pytorch.py

Each row is the median of five calls after two, on the CPU at two threads, for one
object, at random weights of aion-base's shapes: the codecs built from their configs
and the transformer from aion-base's config.json, read from the Hub at the revision
Hub.revision names (no weights).
- image encode: ImageCodec.encode of a Legacy Survey cutout [1, 4, 160, 160];
- spectrum encode: SpectrumCodec.encode of a DESI spectrum of 7781 pixels from 3600 A at
  0.8 A, which the codec pads to 7808;
- encode: AION.encode of an image's 576 tokens and the g, r, i, z fluxes' four;
- decode 273: from that encoding, the decoder and logits of a DESI spectrum's 273 tokens,
  as AION._forward computes them after its encode;
- roar step: GenerationSampler.generate of tok_z from those inputs in one linear step.
"""
import json
import statistics
import time

import torch
from huggingface_hub import hf_hub_download

from aion.codecs import ImageCodec, SpectrumCodec
from aion.fourm.generate import (GenerationSampler, build_chained_generation_schedules,
                                 init_empty_target_modality, init_full_input_modality)
from aion.modalities import DESISpectrum, LegacySurveyImage
from aion.model import AION

torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.manual_seed(0)
REVISION = "40541618104bab0fa85c8af68daeb867a720bb8c"
DESI, Z = "tok_spectrum_desi", "tok_z"


def config(name):
    with open(hf_hub_download("polymathic-ai/aion-base", f"codecs/{name}/config.json",
                              revision=REVISION)) as f:
        return json.load(f)


def timed(f):
    for _ in range(2):
        f()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def row(name, f):
    med, lo, hi = timed(f)
    print(f"{name:<16} {med * 1e3:9.1f} ms  (min {lo * 1e3:.1f}, max {hi * 1e3:.1f})",
          flush=True)


image = ImageCodec(**config("image")).eval()
cutout = torch.randn(1, 4, 160, 160)
bands = ["DES-G", "DES-R", "DES-I", "DES-Z"]
row("image encode", lambda: image.encode(LegacySurveyImage(flux=cutout, bands=bands)))
del image

spectrum = SpectrumCodec(**config("spectrum")).eval()
m = 7781
flux = torch.randn(1, m)
wavelength = (3600.0 + 0.8 * torch.arange(m, dtype=torch.float32)).reshape(1, m)


def desi():
    x = DESISpectrum(flux=flux.clone(), ivar=torch.ones(1, m),
                     mask=torch.zeros(1, m, dtype=torch.bool), wavelength=wavelength.clone())
    return spectrum.encode(x)


row("spectrum encode", desi)
del spectrum

with open(hf_hub_download("polymathic-ai/aion-base", "config.json", revision=REVISION)) as f:
    model = AION(json.load(f)).eval()
info = model.modality_info
TOKENS = {"tok_image": 576, "tok_flux_g": 1, "tok_flux_r": 1, "tok_flux_i": 1, "tok_flux_z": 1}
inputs = {k: torch.randint(0, info[k]["vocab_size"], (1, n)) for k, n in TOKENS.items()}
n_in = sum(v.shape[1] for v in inputs.values())
row("encode", lambda: model.encode(inputs, num_encoder_tokens=n_in))

enc_tokens, enc_emb, enc_mask, _ = model.embed_inputs(inputs, num_encoder_tokens=n_in)
context = model._encode(enc_tokens, enc_emb, enc_mask)


def decode():
    target = {DESI: torch.zeros(1, 273, dtype=torch.bool)}
    dec_tokens, dec_emb, _, _, dec_attn, dec_mod = model.embed_targets(
        target, num_decoder_tokens=273)
    y = model._decode(context, enc_mask, dec_tokens, dec_emb, dec_attn)
    idx = info[DESI]["id"]
    return model.decoder_embeddings[DESI].forward_logits(y[dec_mod == idx])


row("decode 273", decode)

schedule = build_chained_generation_schedules(
    cond_domains=list(TOKENS), target_domains=[Z], tokens_per_target=[1],
    autoregression_schemes=["roar"], decoding_steps=[1], token_decoding_schedules=["linear"],
    temps=[1.0], temp_schedules=["constant"], cfg_scales=[1.0], cfg_schedules=["constant"],
    modality_info=info)
sampler = GenerationSampler(model)


def roar():
    d = {}
    for k, v in inputs.items():
        d[k] = {"tensor": v.clone()}
        d = init_full_input_modality(d, info, k, "cpu")
    d = init_empty_target_modality(d, info, Z, 1, 1, "cpu")
    return sampler.generate(d, schedule, seed=0)


row("roar step", roar)
print(f"torch {torch.__version__}, {torch.get_num_threads()} threads", flush=True)
