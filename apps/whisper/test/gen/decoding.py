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
"""Decoding goldens for test_decoding: upstream's own DecodingTask
(whisper/decoding.py at 86098128) over the trimmed tokenizers of
test/upstream (the first 1000 ranks of gpt2.tiktoken and
multilingual.tiktoken, as tokenizer.py cuts them, specials appended as
get_encoding appends them).

    uv run apps/whisper/test/gen/decoding.py TRIMMED_DIR OUT

TRIMMED_DIR and OUT are apps/whisper/test/upstream. It reads TRIMMED_DIR's
gpt2-1000.tiktoken and multilingual-1000.tiktoken and writes to OUT:
- options.json: initial tokens, sample_begin, sot_index and the suppress list
  of DecodingTask for a set of options.
- filters.json, filters.npz: logits recorded from decode.pt's greedy runs and
  from crafted histories (among them a row where the timestamp rule's two
  sides tie exactly and one whose last timestamp is below an earlier one),
  with the ids SuppressBlank, SuppressTokens and ApplyTimestampRules each set
  to -inf alone, and all three in upstream's order.
- decode.pt, decode.npz, decode.json: a small randomly initialised Whisper
  whose vocabulary is the trimmed gpt2 tokenizer's, saved as OpenAI saves its
  checkpoints (float16), audio features, and upstream's greedy decode of them
  in float32: tokens, text, avg_logprob, no_speech_prob, with float64
  references of the last two (the model in float64, teacher-forced along the
  float32 tokens, under the float32 run's masks).
- beam.json: upstream's beam search of each audio of decode.npz alone, at
  float32, for a set of beam sizes, patiences and length penalties: the
  finished candidates in the order they finished with their scores, the
  live beams and their sums at the end, and the result.
"""
import dataclasses, json, os, sys
from types import SimpleNamespace

import numpy as np
import torch
import whisper
import whisper.decoding as wd
from whisper.decoding import (ApplyTimestampRules, DecodingOptions, DecodingTask,
                              SuppressBlank, SuppressTokens)
from whisper.model import ModelDimensions, Whisper
from whisper.tokenizer import Tokenizer

import trimmed

# The fixtures were recorded on two threads.
torch.set_num_threads(2)
trimmed_dir, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)


def encoding(fname, languages):
    return trimmed.encoding(fname, trimmed.read_ranks(os.path.join(trimmed_dir, fname)), languages)


ENC = {
    "gpt2": encoding("gpt2-1000.tiktoken", 99),
    "multilingual": encoding("multilingual-1000.tiktoken", 99),
}


# DecodingTask builds its tokenizer by get_tokenizer from the model's
# vocabulary; here it is the trimmed one of the same kind.
def get_tokenizer(multilingual, *, num_languages=99, language=None, task=None):
    if multilingual:
        language, task = language or "en", task or "transcribe"
        enc = ENC["multilingual"]
    else:
        language = task = None
        enc = ENC["gpt2"]
    return Tokenizer(encoding=enc, num_languages=99, language=language, task=task)


wd.get_tokenizer = get_tokenizer


def fake_model(multilingual, n_text_ctx=448, n_audio_ctx=1500):
    return SimpleNamespace(
        is_multilingual=multilingual, num_languages=99,
        dims=SimpleNamespace(n_text_ctx=n_text_ctx, n_audio_ctx=n_audio_ctx),
        decoder=SimpleNamespace(blocks=[]))


# Item 1: options

def options_cases():
    gpt2 = get_tokenizer(False)
    long_prompt = gpt2.encode(" " + " ".join(["the quick brown fox jumps over"] * 60))
    assert len(long_prompt) > 223
    cases = []
    for name, multi, opts in [
        ("english", False, {}),
        ("english, without timestamps", False, {"without_timestamps": True}),
        ("multilingual", True, {"language": "en"}),
        ("multilingual de translate", True, {"language": "de", "task": "translate"}),
        ("multilingual fr, without timestamps", True,
         {"language": "fr", "without_timestamps": True}),
        ("prompt", False, {"prompt": [11, 22, 33]}),
        ("prompt longer than half the context", False, {"prompt": long_prompt}),
        ("prefix", False, {"prefix": [44, 55]}),
        ("prompt and prefix, without timestamps", False,
         {"prompt": [11, 22], "prefix": [44, 55, 66], "without_timestamps": True}),
        ("prompt and prefix, multilingual", True,
         {"language": "es", "prompt": [5, 6, 7], "prefix": [8]}),
        ("suppress extra ids", False, {"suppress_tokens": [-1, 5, 7, 900]}),
        ("suppress only given ids", False, {"suppress_tokens": [5, 7, 900]}),
        ("suppress only given ids, multilingual", True,
         {"language": "en", "suppress_tokens": "3,9"}),
    ]:
        given = opts.get("suppress_tokens", "-1")
        if isinstance(given, str):
            given = [int(x) for x in given.split(",")]
        case = {
            "name": name, "multilingual": multi,
            "language": opts.get("language"), "task": opts.get("task", "transcribe"),
            "timestamps": not opts.get("without_timestamps", False),
            "prompt": list(opts.get("prompt", [])), "prefix": list(opts.get("prefix", [])),
            "suppress": sorted({t for t in given if t >= 0}),
            "suppress_non_speech": -1 in given,
        }
        # _get_suppress_tokens extends a given list in place: give it a copy.
        if isinstance(opts.get("suppress_tokens"), list):
            opts = {**opts, "suppress_tokens": list(opts["suppress_tokens"])}
        task = DecodingTask(fake_model(multi), DecodingOptions(fp16=False, **opts))
        cases.append({
            **case,
            "initial_tokens": list(task.initial_tokens),
            "sample_begin": task.sample_begin, "sot_index": task.sot_index,
            "sample_len": task.sample_len,
            "suppress_tokens": list(task._get_suppress_tokens()),
        })
    with open(os.path.join(out, "options.json"), "w") as f:
        json.dump({"n_text_ctx": 448, "cases": cases}, f, indent=1)
    print("options", len(cases))


options_cases()


# Item 3's model, whose runs item 2 records logits from

DIMS = ModelDimensions(n_mels=80, n_audio_ctx=24, n_audio_state=32, n_audio_head=4,
                       n_audio_layer=1, n_vocab=ENC["gpt2"].n_vocab, n_text_ctx=16,
                       n_text_state=32, n_text_head=4, n_text_layer=2)
assert DIMS.n_vocab == 2608 and DIMS.n_audio_ctx > DIMS.n_text_ctx
TOK = get_tokenizer(False)
EOT, TS = TOK.eot, TOK.timestamp_begin
# max_initial_timestamp_index = round(m / (30 / n_audio_ctx)): 62.5 s is index
# 50, as 1.0 s is for the released models' 1500 audio positions.
MAX_INITIAL = 62.5
EOT_SCALE = 14.0

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
    # The end of text along the last layer norm's bias, so that the greedy
    # decodes end at it as well as at the length limits.
    b = model.decoder.ln.bias
    model.decoder.token_embedding.weight[EOT] = EOT_SCALE * b / b.norm()

state = {k: v.half() for k, v in model.state_dict().items()}
ckpt = os.path.join(out, "decode.pt")
torch.save({"dims": dataclasses.asdict(DIMS), "model_state_dict": state}, ckpt)
model = whisper.load_model(ckpt, device="cpu")
model.eval()


class Recorder:
    # Records, at each step of a DecodingTask, the logits before the filters,
    # the tokens so far and the mask after all filters.
    def __init__(self, task):
        self.steps = []
        first, update = task.logit_filters[0], task.decoder.update
        apply = first.apply

        def first_apply(logits, tokens):
            self.steps.append({"logits": logits.clone(), "tokens": tokens.clone()})
            apply(logits, tokens)

        def hooked_update(tokens, logits, sum_logprobs):
            self.steps[-1]["filtered"] = logits.clone()
            return update(tokens, logits, sum_logprobs)

        first.apply = first_apply
        task.decoder.update = hooked_update


def run(features, opts):
    task = DecodingTask(model, opts)
    rec = Recorder(task)
    with torch.no_grad():
        res = task.run(features)
    return task, res, rec.steps


# Features: random, a few scales, two lanes each.
features = randn(6, 2, DIMS.n_audio_ctx, DIMS.n_audio_state, std=1.0)
features[1] *= 3.0
features[4] *= 0.3

long_prompt = TOK.encode(" the quick brown fox jumps over the lazy dog")
assert len(long_prompt) > 7
DECODES = [
    ("without timestamps", 0, {"without_timestamps": True}),
    ("timestamps", 1, {}),
    ("timestamps, louder", 2, {}),
    ("without timestamps, prompt and prefix", 3,
     {"without_timestamps": True, "prompt": long_prompt, "prefix": [262, 318, 257]}),
    ("timestamps, prompt", 4, {"prompt": long_prompt[:3]}),
    ("no blank suppression, given ids", 5,
     {"without_timestamps": True, "suppress_blank": False, "suppress_tokens": [5, 7, 220]}),
]

decodes, all_steps = [], []
for name, fi, opts in DECODES:
    given = list(opts.get("suppress_tokens", [-1]))
    if "suppress_tokens" in opts:
        opts = {**opts, "suppress_tokens": list(opts["suppress_tokens"])}
    opts = DecodingOptions(language="en", fp16=False, max_initial_timestamp=MAX_INITIAL,
                           **opts)
    task, res, steps = run(features[fi], opts)
    all_steps.append((name, task, steps))
    decodes.append((name, fi, given, opts, task, res, steps))

# Item 2: filters


def banned(x):
    # The ids at -inf, as [start, stop) ranges.
    ids, ranges = torch.nonzero(torch.isneginf(x)).flatten().tolist(), []
    for i in ids:
        if ranges and ranges[-1][1] == i:
            ranges[-1][1] = i + 1
        else:
            ranges.append([i, i + 1])
    return ranges


class _F:
    # Stands in for torch.nn.functional inside whisper.decoding, so the
    # log_softmax of ApplyTimestampRules' logprob rule is seen.
    def __init__(self):
        self.last = None

    def __getattr__(self, name):
        return getattr(torch.nn.functional, name)

    def log_softmax(self, *a, **kw):
        self.last = torch.nn.functional.log_softmax(*a, **kw)
        return self.last


F_PROXY = _F()
wd.F = F_PROXY


def filter_case(name, logits, tokens, sample_begin, timestamps, suppress):
    # Upstream's filters, each alone and all in order, on one lane.
    logits, tokens = logits[None].clone(), tokens[None]
    blank = SuppressBlank(TOK, sample_begin)
    supp = SuppressTokens(suppress)
    rules = ApplyTimestampRules(TOK, sample_begin, 50)

    def alone(f):
        x = logits.clone()
        f.apply(x, tokens)
        return banned(x[0])

    x = logits.clone()
    for f in [blank, supp] + ([rules] if timestamps else []):
        f.apply(x, tokens)
    case = {
        "name": name, "sampled": tokens[0, sample_begin:].tolist(), "timestamps": timestamps,
        "suppress": list(suppress),
        "blank": alone(blank), "suppress_tokens": alone(supp),
        "timestamp_rules": alone(rules), "all": banned(x[0]),
    }
    # The logprob rule's margin, logsumexp of the timestamps less the greatest
    # text, on the timestamp rules alone: above 0 bans the text.
    alone(rules)
    lp = F_PROXY.last[0]
    m = float(lp[TS:].logsumexp(-1) - lp[:TS].max())
    case["rule_margin"] = m if np.isfinite(m) else None
    return case


def half(x):
    # Logits held at float16, so the goldens store them exactly at half size.
    return x.half().float()


suppress_default = list(DecodingTask(model, DecodingOptions(fp16=False))._get_suppress_tokens())
filter_cases, filter_logits = [], []


def add(name, logits, tokens, sample_begin, timestamps, suppress):
    logits = half(logits)
    filter_logits.append(logits.half().numpy())
    filter_cases.append(filter_case(name, logits, tokens, sample_begin, timestamps, suppress))


# Recorded: the first two steps of each decode and its last.
for name, task, steps in all_steps:
    timestamps = not task.options.without_timestamps
    for i in sorted({0, 1, len(steps) - 1}):
        if i < len(steps):
            s = steps[i]
            add(f"{name}, step {i}", s["logits"][0], s["tokens"][0], task.sample_begin,
                timestamps, task._get_suppress_tokens())

# Crafted histories over recorded logits: each branch of ApplyTimestampRules,
# and the logprob rule on both sides of its threshold.
base = all_steps[1][2][1]["logits"][0]
sot = [TOK.sot]
crafted = [
    ("empty, first step", [], [0.0, 12.0]),
    ("text", [400, 401], [0.0, 12.0, -12.0]),
    ("one timestamp", [TS + 3], [0.0, 12.0]),
    ("timestamp after text", [TS, 400, TS + 7], [0.0, 12.0, -12.0]),
    ("two timestamps", [TS, 400, TS + 7, TS + 7], [0.0, 12.0]),
    ("timestamp pair then text", [TS, 400, TS + 7, TS + 9, 401], [0.0, 12.0, -12.0]),
    ("timestamp at the end of the range", [TS, 400, TS + 1500], [0.0, 12.0]),
    ("text after the last timestamp", [TS + 1500, 400], [0.0, 12.0]),
    # The last timestamp sampled is below an earlier one: the rule reads the
    # last, timestamps[-1], so it bans [TS, TS + 11), not up to TS + 31.
    ("decreasing timestamps", [TS + 30, 400, TS + 10, 401], [0.0, 12.0]),
]
for name, sampled, shifts in crafted:
    for shift in shifts:
        x = base.clone()
        x[TS:] += shift
        side = "" if shift == 0 else f", timestamps {shift:+g}"
        add(name + side, x, torch.tensor(sot + sampled), 1, True, suppress_default)

# Ties: the greatest logit held twice.
x = base.clone()
x[600] = x[601] = x.max() + 1.0
add("tied greatest", x, torch.tensor(sot + [400]), 1, False, suppress_default)

# The timestamp rule's two sides equal: after text then a timestamp, only the
# end of text and that timestamp are finite, both 0, so both log probabilities
# are -ln 2 exactly. The rule's strict > keeps the end of text.
x = torch.full_like(base, -np.inf)
x[EOT] = x[TS + 3] = 0.0
add("timestamp rule tied", x, torch.tensor(sot + [400, TS + 3]), 1, True, suppress_default)

# Upstream reads the last sampled timestamp, not the greatest.
for c in filter_cases:
    if c["name"].startswith("decreasing timestamps"):
        hit = lambda i: any(a <= i < b for a, b in c["all"])
        assert all(hit(i) for i in range(TS, TS + 11)), c["name"]
        assert not any(hit(i) for i in range(TS + 11, TS + 32)), c["name"]

margins = [c["rule_margin"] for c in filter_cases if c["timestamps"] and c["rule_margin"] is not None]
fired = sum(m > 0 for m in margins)
assert filter_cases[-1]["rule_margin"] == 0.0, filter_cases[-1]["rule_margin"]
assert not any(a <= EOT < b for a, b in filter_cases[-1]["all"])
assert 0 < fired < len(margins), (fired, len(margins))
np.savez(os.path.join(out, "filters.npz"), logits=np.stack(filter_logits))
with open(os.path.join(out, "filters.json"), "w") as f:
    json.dump({"timestamp_begin": TS, "eot": EOT, "max_initial_timestamp": 50,
               "cases": filter_cases}, f, indent=0)
print("filters", len(filter_cases), "logprob rule fired in", fired, "of", len(margins))

# Item 3: greedy decodes, with float64 references


def reference64(m64, feats, task, res_list, steps):
    # Each lane's sampled tokens, teacher-forced through the float64 model in
    # one call, under the float32 run's masks: the float64 avg_logprob and
    # no_speech_prob of the same decisions.
    out = []
    for k, r in enumerate(res_list):
        n = len(r.tokens)
        ended = len(steps) > n  # an end of text was sampled
        seq = list(task.initial_tokens) + r.tokens + ([EOT] if ended else [])
        with torch.no_grad():
            logits = m64.decoder(torch.tensor([seq[:-1]]), feats[k:k + 1].double())[0]
        total = 0.0
        for i in range(n + (1 if ended else 0)):
            row = logits[task.sample_begin - 1 + i].clone()
            row[torch.isneginf(steps[i]["filtered"][k])] = -np.inf
            total += float(torch.log_softmax(row, -1)[seq[task.sample_begin + i]])
        p = torch.softmax(logits[task.sot_index], -1)[TOK.no_speech]
        out.append({"avg_logprob": total / (n + 1), "no_speech_prob": float(p)})
    return out


whisper.model.LayerNorm.forward = lambda self, x: torch.nn.LayerNorm.forward(self, x)


# TextDecoder.forward ends with .float(): the float64 model's logits would be
# rounded to float32, and torch's float32 log_softmax on the CPU is off by
# about 1e-3 of the probability outside the chosen token. At float64 the
# logits stay at float64.
_forward = whisper.model.TextDecoder.forward


def _forward64(self, x, xa, kv_cache=None):
    if xa.dtype != torch.float64:
        return _forward(self, x, xa, kv_cache)
    offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
    x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
    x = x.to(xa.dtype)
    for block in self.blocks:
        x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
    x = self.ln(x)
    return x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)


whisper.model.TextDecoder.forward = _forward64
m64 = whisper.load_model(ckpt, device="cpu").double()
m64.eval()

cases = []
covered = {"eot": 0, "sample_len": 0, "n_ctx": 0}
for name, fi, given, opts, task, res, steps in decodes:
    refs = reference64(m64, features[fi], task, res, steps)
    lanes = []
    for k, (r, ref) in enumerate(zip(res, refs)):
        ended = len(steps) > len(r.tokens)
        stop = ("eot" if ended else
                "n_ctx" if task.sample_begin + len(steps) > DIMS.n_text_ctx else "sample_len")
        covered[stop] += 1
        lanes.append({"tokens": r.tokens, "text": r.text, "avg_logprob": r.avg_logprob,
                      "no_speech_prob": r.no_speech_prob, "stop": stop,
                      "avg_logprob.f64": ref["avg_logprob"],
                      "no_speech_prob.f64": ref["no_speech_prob"]})
    o = task.options
    cases.append({
        "name": name, "features": fi, "timestamps": not o.without_timestamps,
        "prompt": list(o.prompt or []), "prefix": list(o.prefix or []),
        "suppress_blank": o.suppress_blank,
        "suppress": sorted({t for t in given if t >= 0}),
        "suppress_non_speech": -1 in given,
        "initial_tokens": list(task.initial_tokens), "steps": len(steps), "lanes": lanes,
    })
    print(name, [(l["stop"], len(l["tokens"])) for l in lanes])

print("stops", covered)
np.savez(os.path.join(out, "decode.npz"), features=features.numpy())
with open(os.path.join(out, "decode.json"), "w") as f:
    json.dump({"max_initial_timestamp": 50, "cases": cases}, f, indent=1)
print("checkpoint bytes", os.path.getsize(ckpt))


# Beam search: upstream decodes one audio a call (its cross-attention's
# broadcast assumes one audio when the group is larger than one).


class BeamRecorder:
    # The finished candidates in the order they finished, and the live beams
    # and their sums, as finalize receives them.
    def __init__(self, task):
        self.finished, self.live, self.sums = None, None, None
        finalize = task.decoder.finalize

        def hooked(tokens, sum_logprobs):
            self.finished = [list(d.items()) for d in task.decoder.finished_sequences]
            self.live, self.sums = tokens.clone(), sum_logprobs.clone()
            return finalize(tokens, sum_logprobs)

        task.decoder.finalize = hooked


BEAMS = [
    ("beam 5, without timestamps", 0, 5, None, None, {"without_timestamps": True}),
    ("beam 5, timestamps", 1, 5, None, None, {}),
    ("beam 3, patience 2, louder", 2, 3, 2.0, None, {}),
    ("beam 5, patience 0.5, prompt and prefix", 3, 5, 0.5, None,
     {"without_timestamps": True, "prompt": long_prompt, "prefix": [262, 318, 257]}),
    ("beam 2, patience 1.5, length penalty 0.6, prompt", 4, 2, 1.5, 0.6,
     {"prompt": long_prompt[:3]}),
    ("beam 4, length penalty 1, given ids", 5, 4, None, 1.0,
     {"without_timestamps": True, "suppress_tokens": [5, 7, 220]}),
    ("beam 1, without timestamps", 0, 1, None, None, {"without_timestamps": True}),
]


def seq(t, begin):
    return [int(x) for x in t[begin:]]


beams = []
for name, fi, size, patience, alpha, opts in BEAMS:
    given = list(opts.get("suppress_tokens", [-1]))
    if "suppress_tokens" in opts:
        opts = {**opts, "suppress_tokens": list(opts["suppress_tokens"])}
    audios = []
    for k in range(features.shape[1]):
        o = DecodingOptions(language="en", fp16=False, max_initial_timestamp=MAX_INITIAL,
                            beam_size=size, patience=patience, length_penalty=alpha, **opts)
        task = DecodingTask(model, o)
        rec = BeamRecorder(task)
        with torch.no_grad():
            res = task.run(features[fi][k:k + 1])[0]
        b = task.sample_begin
        audios.append({
            "finished": [{"tokens": seq(t, b), "score": sc} for t, sc in rec.finished[0]],
            "live": [seq(t, b) for t in rec.live[0]],
            "sums": [float(x) for x in rec.sums[0]],
            "tokens": res.tokens, "text": res.text, "avg_logprob": res.avg_logprob,
            "no_speech_prob": res.no_speech_prob,
        })
    o = task.options
    beams.append({
        "name": name, "features": fi, "size": size,
        "patience": patience, "length_penalty": alpha,
        "max_candidates": task.decoder.max_candidates,
        "timestamps": not o.without_timestamps,
        "prompt": list(o.prompt or []), "prefix": list(o.prefix or []),
        "suppress_blank": o.suppress_blank,
        "suppress": sorted({t for t in given if t >= 0}),
        "suppress_non_speech": -1 in given,
        "initial_tokens": list(task.initial_tokens), "audios": audios,
    })
    print(name, [(len(a["finished"]), len(a["tokens"])) for a in audios])

with open(os.path.join(out, "beam.json"), "w") as f:
    json.dump({"max_initial_timestamp": 50, "cases": beams}, f, indent=1)
