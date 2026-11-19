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
"""Goldens of upstream's detect_language (whisper/decoding.py at 86098128)
for test_decoding: a small randomly initialised multilingual Whisper
(n_vocab 51865, so is_multilingual) saved as OpenAI saves its checkpoints
(float16), audio features of a few audios, and the language probabilities
upstream gives them over the trimmed multilingual tokenizer
(multilingual-1000.tiktoken, 99 languages, specials appended as
get_encoding appends them), at float32, with the model at float64 as the
reference.

    uv run apps/whisper/test/gen/languages.py TRIMMED_DIR OUT

TRIMMED_DIR and OUT are apps/whisper/test/upstream. It reads TRIMMED_DIR's
multilingual-1000.tiktoken and writes to OUT languages.pt, languages.npz
(features [n; ctx; state] at float32, probs32 and probs64 [n; 99]) and
languages.json (the codes in token order, each audio's detected code at
float32).
"""
import dataclasses, json, os, sys

import numpy as np
import torch
import whisper
from whisper.decoding import detect_language
from whisper.model import ModelDimensions, Whisper
from whisper.tokenizer import LANGUAGES, Tokenizer

import trimmed

# The fixtures were recorded on two threads.
torch.set_num_threads(2)
trimmed_dir, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)


def encoding(fname, languages):
    return trimmed.encoding(fname, trimmed.read_ranks(os.path.join(trimmed_dir, fname)), languages)


TOK = Tokenizer(encoding=encoding("multilingual-1000.tiktoken", 99), num_languages=99,
                language="en", task="transcribe")

DIMS = ModelDimensions(n_mels=80, n_audio_ctx=24, n_audio_state=4, n_audio_head=1,
                       n_audio_layer=1, n_vocab=51865, n_text_ctx=16, n_text_state=4,
                       n_text_head=1, n_text_layer=1)

g = torch.Generator().manual_seed(20261119)


def randn(*shape, std=1.0):
    return torch.randn(*shape, generator=g) * std


model = Whisper(DIMS)
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("ln.weight") or name.endswith("ln_post.weight"):
            p.copy_(1 + randn(*p.shape, std=0.2))
        elif p.ndim == 1:
            p.copy_(randn(*p.shape, std=0.1))
        elif name == "decoder.positional_embedding":
            p.copy_(randn(*p.shape, std=0.1))
        elif name == "decoder.token_embedding.weight":
            p.copy_(randn(*p.shape, std=0.6))
        else:
            fan_in = p[0].numel()
            p.copy_(randn(*p.shape, std=fan_in ** -0.5))
    # The language tokens louder, so that their probabilities spread.
    langs = list(TOK.all_language_tokens)
    model.decoder.token_embedding.weight[langs] *= 3.0

state = {k: v.half() for k, v in model.state_dict().items()}
ckpt = os.path.join(out, "languages.pt")
torch.save({"dims": dataclasses.asdict(DIMS), "model_state_dict": state}, ckpt)
m32 = whisper.load_model(ckpt, device="cpu").eval()
assert m32.is_multilingual

features = randn(4, DIMS.n_audio_ctx, DIMS.n_audio_state, std=1.0)
features[1] *= 3.0
features[3] *= 0.3


def probs(model, x):
    with torch.no_grad():
        ids, ps = detect_language(model, x, TOK)
    # Token order: all_language_codes follows a set's order.
    order = list(LANGUAGES.keys())[:99]
    tokens = [TOK.to_language_token(c) for c in order]
    assert tokens == sorted(tokens)
    codes = [order[tokens.index(i)] for i in ids.tolist()]
    return codes, np.array([[p[c] for c in order] for p in ps], np.float64)


codes32, p32 = probs(m32, features)

# The float64 reference, as decoding.py takes it: layer norms at the
# input's dtype, and TextDecoder.forward without its closing .float(), so the
# logits stay at float64.
whisper.model.LayerNorm.forward = lambda self, x: torch.nn.LayerNorm.forward(self, x)


def _forward64(self, x, xa, kv_cache=None):
    offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
    x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
    x = x.to(xa.dtype)
    for block in self.blocks:
        x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
    x = self.ln(x)
    return x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)


whisper.model.TextDecoder.forward = _forward64
m64 = whisper.load_model(ckpt, device="cpu").double().eval()
_, p64 = probs(m64, features.double())
np.savez(os.path.join(out, "languages.npz"), features=features.numpy(),
         probs32=p32.astype(np.float32), probs64=p64)
with open(os.path.join(out, "languages.json"), "w") as f:
    json.dump({"codes": list(LANGUAGES.keys())[:99], "detected": codes32}, f)
    f.write("\n")
print("detected", codes32, "max |p32 - p64|", np.abs(p32 - p64).max())
