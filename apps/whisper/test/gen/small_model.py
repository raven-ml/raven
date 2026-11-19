# /// script
# requires-python = "==3.13.*"
# dependencies = [
#   "openai-whisper @ git+https://github.com/openai/whisper.git@86098128c0b4f24f0e2aa2994de830614b474227",
#   "torch==2.15.0",
#   "numpy==2.5.5",
#   "tiktoken==0.14.0",
# ]
#
# [tool.uv]
# # whisper declares triton on Linux x86-64 for its CUDA kernels; the
# # generators run on the CPU.
# override-dependencies = ["triton; sys_platform == 'never'"]
# sources = { torch = { index = "pytorch-cpu" } }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""A small Whisper for test_model: upstream's Whisper class (whisper/model.py
at 86098128) at tiny's architecture with small dims, randomly initialised,
saved as OpenAI saves its checkpoints ({"dims", "model_state_dict"}, every
tensor float16), then loaded back by whisper.load_model, and run on the CPU
in float32 on fixed inputs.

    uv run apps/whisper/test/gen/small_model.py OUT

OUT is apps/whisper/test/upstream.

Writes OUT/small.pt and OUT/small.npz (inputs and upstream's outputs), and
prints upstream's own float32 error against the same model in float64.
"""
import dataclasses, os, sys

import numpy as np
import torch
import whisper
from whisper.model import ModelDimensions, Whisper, disable_sdpa

# The fixtures were recorded on two threads.
torch.set_num_threads(2)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

dims = ModelDimensions(n_mels=80, n_audio_ctx=24, n_audio_state=64, n_audio_head=4,
                       n_audio_layer=2, n_vocab=96, n_text_ctx=16, n_text_state=64,
                       n_text_head=4, n_text_layer=2)
# Cross-attention K/V are cached by upstream's hooks only when the audio
# context is longer than the text context (model.py, save_to_cache).
assert dims.n_audio_ctx > dims.n_text_ctx

g = torch.Generator().manual_seed(20261118)


def randn(*shape, std=1.0):
    return torch.randn(*shape, generator=g) * std


model = Whisper(dims)
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("ln.weight") or name.endswith("ln_post.weight"):
            p.copy_(1 + randn(*p.shape, std=0.2))
        elif p.ndim == 1:
            p.copy_(randn(*p.shape, std=0.1))
        elif name == "decoder.positional_embedding":
            p.copy_(randn(*p.shape, std=0.1))
        elif name == "decoder.token_embedding.weight":
            p.copy_(randn(*p.shape, std=0.3))
        else:  # Linear and Conv1d weights: unit gain over their fan-in
            fan_in = p[0].numel()
            p.copy_(randn(*p.shape, std=fan_in ** -0.5))
    # The stored positional table runs, not a recomputed one: move it off the
    # sinusoids so a port that recomputes them fails.
    model.encoder.positional_embedding.add_(randn(dims.n_audio_ctx, dims.n_audio_state, std=0.05))

state = {k: v.half() for k, v in model.state_dict().items()}
ckpt = os.path.join(out, "small.pt")
torch.save({"dims": dataclasses.asdict(dims), "model_state_dict": state}, ckpt)

# Upstream's own load: fp32 modules holding the fp16 values.
model = whisper.load_model(ckpt, device="cpu")
model.eval()
assert all(p.dtype == torch.float32 for p in model.parameters())

# Every position of the text context, so the cache's last row and the last
# positional row are computed.
batch, frames, n_tok, prefix = 2, 2 * dims.n_audio_ctx, dims.n_text_ctx, 3
mel = randn(batch, dims.n_mels, frames, std=0.5)
tokens = torch.randint(0, dims.n_vocab, (batch, n_tok), generator=g)

with torch.no_grad():
    enc = model.encoder(mel)
    with disable_sdpa():
        enc_manual = model.encoder(mel)
    logits = model.decoder(tokens, enc)
    # Upstream's decoding path: the prefix at once, then one token a call,
    # with the KV-cache hooks.
    cache, hooks = model.install_kv_cache_hooks()
    steps = [model.decoder(tokens[:, :prefix], enc, kv_cache=cache)]
    for i in range(prefix, n_tok):
        steps.append(model.decoder(tokens[:, i : i + 1], enc, kv_cache=cache))
    for h in hooks:
        h.remove()
    logits_cached = torch.cat(steps, dim=1)

    # In float64 upstream's LayerNorm would compute in float32 (x.float());
    # this reference computes it in float64 too.
    whisper.model.LayerNorm.forward = lambda self, x: torch.nn.LayerNorm.forward(self, x)
    m64 = whisper.load_model(ckpt, device="cpu").double()
    enc64 = m64.encoder(mel.double())
    logits64 = m64.decoder(tokens, enc64)

np.savez(os.path.join(out, "small.npz"), **{
    "mel": mel.numpy(),
    "tokens": tokens.numpy().astype(np.int64),
    "encoder": enc.numpy(),
    "encoder.manual": enc_manual.numpy(),
    "encoder.f64": enc64.numpy(),
    "logits": logits.numpy(),
    "logits.cached": logits_cached.numpy(),
    "logits.f64": logits64.numpy(),
})


def err(a, b):
    return float((a.double() - b.double()).abs().max())


print("encoder |max|", float(enc.abs().max()), "logits |max|", float(logits.abs().max()))
print("f32 - f64: encoder", err(enc, enc64), "logits", err(logits, logits64))
print("sdpa - manual: encoder", err(enc, enc_manual))
print("cached - whole: logits", err(logits_cached, logits))
print("file bytes", os.path.getsize(ckpt))
