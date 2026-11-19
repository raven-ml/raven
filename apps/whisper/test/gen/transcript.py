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
"""Goldens of upstream's transcribe loop (whisper/transcribe.py at 86098128)
for Transcript.loop's tests, and a small randomly initialised model for
Transcript.transcribe's laws.

transcript.json: cases of upstream's transcribe over a fake model whose
decode returns scripted results in call order, over the trimmed gpt2
tokenizer (gpt2-1000.tiktoken, specials appended as get_encoding appends
them). The log-mel is replaced by frames numbered from 1, so each call's
window says its seek and its size. Each case holds:
- the port's options (ladder, thresholds, condition, prompt ids, carry,
  clips in frames, null for a stop past the content, timestamps);
- the content's frames;
- each call in order: its seek, size, temperature, upstream's prompt and
  initial tokens (DecodingTask's), and the scripted result (tokens, text,
  avg_logprob, no_speech_prob);
- upstream's segments (seek, start and end in frames, tokens, text) and
  text.
A case marked "departure" is one whose prompts differ from upstream's by
the port's rule (apps/whisper/AGENTS.md).

transcript.pt: a Whisper of 1500 audio positions over the trimmed gpt2
tokenizer, saved as OpenAI saves its checkpoints (float16), small enough
for a test to transcribe a few windows.

    uv run apps/whisper/test/gen/transcript.py TRIMMED_DIR OUT

TRIMMED_DIR and OUT are apps/whisper/test/upstream. It reads TRIMMED_DIR's
gpt2-1000.tiktoken and writes OUT/transcript.json and OUT/transcript.pt.
"""
import dataclasses, json, os, sys
from types import SimpleNamespace

import numpy as np
import torch
import whisper
import whisper.decoding as wd
import whisper.transcribe
wt = sys.modules["whisper.transcribe"]
from whisper.decoding import DecodingOptions, DecodingResult, DecodingTask
from whisper.model import ModelDimensions, Whisper
from whisper.tokenizer import Tokenizer

from whisper.utils import compression_ratio

import trimmed

# The fixtures were recorded on two threads.
torch.set_num_threads(2)
trimmed_dir, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)


def encoding(fname, languages):
    return trimmed.encoding(fname, trimmed.read_ranks(os.path.join(trimmed_dir, fname)), languages)


ENC = encoding("gpt2-1000.tiktoken", 99)


def get_tokenizer(multilingual, *, num_languages=99, language=None, task=None):
    assert not multilingual
    return Tokenizer(encoding=ENC, num_languages=99, language=None, task=None)


wd.get_tokenizer = get_tokenizer
wt.get_tokenizer = get_tokenizer
TOK = get_tokenizer(False)
TS, EOT = TOK.timestamp_begin, TOK.eot
N_FRAMES = 3000


def ts(i):
    return TS + i


W = TOK.encode(" the cat sat on a mat and then it ran far away from home")
SP = TOK.encode(" ")
assert len(SP) == 1
LOOP = [W[0], W[1]] * 40


# The log-mel: frame f (from 0) holds f + 1 in its first band, so a window
# padded with zeros says its seek and its size.
def fake_mel(audio, n_mels=80, padding=0):
    frames = (len(audio) + padding) // 160
    mel = torch.zeros(n_mels, frames)
    mel[0] = torch.arange(1, frames + 1, dtype=torch.float32)
    return mel


wt.log_mel_spectrogram = fake_mel

DIMS = SimpleNamespace(n_mels=80, n_audio_ctx=1500, n_text_ctx=448)


class Fake:
    # A model whose decode returns the script's results in order and records
    # each call.
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.device = torch.device("cpu")
        self.dims = DIMS
        self.is_multilingual = False
        self.num_languages = 99
        self.decoder = SimpleNamespace(blocks=[])

    def decode(self, segment, options):
        row = segment[0]
        seek = int(row[0].item()) - 1
        size = int((row > 0).sum().item())
        task = DecodingTask(self, options)
        assert self.script, f"call {len(self.calls)} past the script at seek {seek}"
        r = self.script.pop(0)
        tokens = r["tokens"]
        text = TOK.decode(tokens).strip()
        self.calls.append({
            "seek": seek, "size": size, "temperature": options.temperature,
            "prompt": list(options.prompt or []),
            "initial_tokens": list(task.initial_tokens),
            "tokens": tokens, "text": text,
            "avg_logprob": r["lp"], "no_speech_prob": r["ns"],
        })
        return DecodingResult(
            audio_features=None, language="en", tokens=tokens, text=text,
            avg_logprob=r["lp"], no_speech_prob=r["ns"],
            temperature=options.temperature, compression_ratio=compression_ratio(text))


def res(tokens, lp=-0.3, ns=0.05):
    return {"tokens": tokens, "lp": lp, "ns": ns}


DEFAULT_T = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def ladder(temps, beam=None, best_of=None):
    out = []
    for t in temps:
        if t == 0:
            out.append({"kind": "beam", "size": beam, "patience": 1.0} if beam
                       else {"kind": "greedy"})
        else:
            out.append({"kind": "sample", "best_of": best_of or 1, "temperature": t})
    return out


def prompt_of(n):
    # A text whose " " + strip encodes to n ids.
    text = " ".join(["the"] * n)
    ids = TOK.encode(" " + text.strip())
    assert len(ids) == n, (n, len(ids))
    return text, ids


cases = []


def case(name, frames, script, temps=DEFAULT_T, crt=2.4, lpt=-1.0, nst=0.6,
         condition=True, prompt=None, carry=False, clips=None, port_clips=None,
         timestamps=True, beam=None, best_of=None, departure=None):
    model = Fake(script)
    audio = np.zeros(frames * 160, np.float32)
    kwargs = dict(verbose=None, temperature=temps, compression_ratio_threshold=crt,
                  logprob_threshold=lpt, no_speech_threshold=nst,
                  condition_on_previous_text=condition, fp16=False,
                  carry_initial_prompt=carry)
    prompt_ids = []
    if prompt is not None:
        text, prompt_ids = prompt_of(prompt)
        kwargs["initial_prompt"] = text
    if clips is not None:
        kwargs["clip_timestamps"] = ",".join(repr(f / 100) for f in clips)
    if not timestamps:
        kwargs["without_timestamps"] = True
    if beam:
        kwargs["beam_size"] = beam
    if best_of:
        kwargs["best_of"] = best_of
    r = wt.transcribe(model, audio, **kwargs)
    assert not model.script, f"{name}: {len(model.script)} results unused"
    if port_clips is None:
        pts = list(clips) if clips is not None else [0]
        if len(pts) % 2 == 1:
            pts.append(None)
        port_clips = [[pts[i], pts[i + 1]] for i in range(0, len(pts), 2)]
    cases.append({
        "name": name, "frames": frames,
        "ladder": ladder(temps, beam, best_of),
        "compression_ratio_threshold": crt, "logprob_threshold": lpt,
        "no_speech_threshold": nst, "condition": condition,
        "prompt": prompt_ids, "carry_prompt": carry, "clips": port_clips,
        "timestamps": timestamps, "departure": departure,
        "calls": model.calls,
        "segments": [{"seek": s["seek"], "start": round(s["start"] * 100),
                      "end": round(s["end"] * 100), "tokens": s["tokens"],
                      "text": s["text"]} for s in r["segments"]],
        "text": r["text"],
    })
    # Frames from upstream's float seconds, exactly.
    for s in r["segments"]:
        for k in ("start", "end"):
            assert abs(s[k] * 100 - round(s[k] * 100)) < 1e-9


a, b, c, d, e = W[:5]

case("pairs, an unfinished segment, a single ending", 4500, [
    res([ts(0), a, b, ts(50), ts(50), c, d, ts(90), ts(90), e]),
    res([ts(0), a, ts(100)]),
    res([ts(0), a, b, ts(20), ts(20), c, ts(60)]),
])
case("no timestamp, then Timestamp 0 last", 3500, [
    res([a, b, c]),
    res([a, ts(0)]),
])
case("blank and instantaneous segments cleared", 3500, [
    res([ts(5), SP[0], ts(10), ts(10), a, ts(30), ts(30), b, ts(30), ts(30), c, ts(40)]),
    res([d, e]),
])
case("fallback on the compression ratio", 2000, [
    res([ts(0), *LOOP, ts(100)]),
    res([ts(0), a, b, c, ts(100)]),
])
case("fallback on the log probability, a reset above 0.5", 4500, [
    res([ts(0), a, ts(10)], lp=-1.5),
    res([ts(0), b, ts(10)], lp=-1.2),
    res([ts(0), c, ts(10)], lp=-1.1),
    res([ts(0), d, ts(10)], lp=-0.5),
    res([ts(0), e, ts(10)]),
])
case("no reset at 0.4", 4500, [
    res([ts(0), a, ts(10)], lp=-1.5),
    res([ts(0), b, ts(10)], lp=-1.2),
    res([ts(0), c, ts(10)], lp=-0.5),
    res([ts(0), d, ts(10)]),
])
case("every rung fails, the last is chosen", 2000, [
    res([ts(0), a, ts(10)], lp=-2.0),
    res([ts(0), b, ts(10)], lp=-2.0),
    res([ts(0), c, ts(10)], lp=-2.0),
    res([ts(0), d, ts(10)], lp=-2.0),
    res([ts(0), e, ts(10)], lp=-2.0),
    res([ts(0), a, b, ts(10)], lp=-2.0),
])
case("silence", 4500, [
    res([ts(0), a, ts(10)], lp=-1.5, ns=0.9),
    res([ts(0), b, ts(10)]),
])
case("no silence where the log probability is high", 3500, [
    res([ts(0), a, ts(10)], lp=-0.5, ns=0.9),
    res([ts(0), b, ts(10)]),
])
case("silence without a log probability threshold", 4500, [
    res([ts(0), a, ts(10)], lp=-0.2, ns=0.7),
    res([ts(0), b, ts(10)], lp=-3.0, ns=0.1),
], lpt=None)
case("no silence without a no-speech threshold", 3500, [
    res([ts(0), a, ts(10)], lp=-1.5, ns=0.95),
    res([ts(0), b, ts(10)], lp=-1.4, ns=0.95),
    res([ts(0), c, ts(10)]),
], temps=(0.0, 0.2), nst=None)
case("no compression ratio threshold", 2000, [
    res([ts(0), *LOOP, ts(100)]),
], crt=None)
case("not conditioned, with a prompt", 7000, [
    res([ts(0), a, ts(10)]),
    res([ts(0), b, ts(10)]),
    res([ts(0), c, ts(10)]),
], condition=False, prompt=10)
case("a prompt", 7000, [
    res([ts(0), a, ts(10)]),
    res([ts(0), b, c, ts(10)]),
    res([ts(0), d, ts(10)]),
], prompt=10)
case("a carried prompt of 10 ids", 7000, [
    res([ts(0), a, ts(10)]),
    res([ts(0), b, c, ts(10)], lp=-1.5),
    res([ts(0), d, ts(10)], lp=-1.4),
    res([ts(0), d, ts(10)], lp=-1.3),
    res([ts(0), d, ts(10)], lp=-1.2),
    res([ts(0), d, ts(10)]),
    res([ts(0), e, ts(10)]),
], prompt=10, carry=True)
case("a carried prompt of 222 ids", 7000, [
    res([ts(0), a, ts(10)]),
    res([ts(0), b, c, ts(10)]),
    res([ts(0), d, ts(10)]),
], prompt=222, carry=True)
case("a carried prompt of 223 ids", 7000, [
    res([ts(0), a, ts(10)]),
    res([ts(0), b, c, ts(10)]),
    res([ts(0), d, ts(10)]),
], prompt=223, carry=True, departure="carry")
case("clips, one entered below the seek, the last past the content", 9000, [
    res([ts(0), a, ts(50), ts(50), b, ts(700), ts(700), c]),
    res([ts(0), d, ts(40)]),
    res([ts(0), e, ts(10)]),
], clips=[0, 1000, 1200, 2000, 8000],
   port_clips=[[0, 1000], [1200, 2000], [8000, 9500], [9600, None]])
case("without timestamps", 4000, [
    res([a, b, c]),
    res([d, e]),
], timestamps=False)
EN = ENC._special_tokens["<|en|>"]
case("a special token's text is no segment's text", 3500, [
    res([ts(0), EN, a, ts(10), ts(10), EN, ts(20)]),
    res([ts(0), b, ts(10)]),
])
case("beam then samples of five", 2000, [
    res([ts(0), a, ts(10)], lp=-1.5),
    res([ts(0), b, ts(10)]),
], beam=5, best_of=5)

with open(os.path.join(out, "transcript.json"), "w") as f:
    json.dump({"n_text_ctx": DIMS.n_text_ctx, "cases": cases}, f)
    f.write("\n")
print(len(cases), "cases,", sum(len(c["calls"]) for c in cases), "calls")

# The laws' model.
MDIMS = ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=16, n_audio_head=2,
                        n_audio_layer=1, n_vocab=ENC.n_vocab, n_text_ctx=16,
                        n_text_state=16, n_text_head=2, n_text_layer=1)
g = torch.Generator().manual_seed(20261119)


def randn(*shape, std=1.0):
    return torch.randn(*shape, generator=g) * std


model = Whisper(MDIMS)
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
state = {k: v.half() for k, v in model.state_dict().items()}
torch.save({"dims": dataclasses.asdict(MDIMS), "model_state_dict": state},
           os.path.join(out, "transcript.pt"))
