# /// script
# requires-python = "==3.13.*"
# dependencies = [
#   "openai-whisper @ git+https://github.com/openai/whisper.git@86098128c0b4f24f0e2aa2994de830614b474227",
#   "torch==2.15.0",
#   "numpy==2.5.5",
#   "tiktoken==0.14.0",
#   "soundfile==0.14.0",
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
"""Upstream's greedy decodes of subset.py's utterances, the inputs of
normalize.py and of check.exe's greedy: DecodingTask (whisper/decoding.py at
86098128) on the CPU in float32, one utterance a call, on the short-form
log-mel (pad_or_trim of the samples, then log_mel_spectrogram), in English.

    uv run apps/whisper/test/gen/greedy.py MODEL PROTOCOL LIBRISPEECH_DIR SUBSET_TSV WEIGHTS_DIR OUT_JSONL

MODEL is a release (tiny, tiny.en, base.en, ...), which whisper.load_model
reads from WEIGHTS_DIR or downloads there; PROTOCOL is notimestamps or
timestamps. Writes OUT_JSONL, a record per utterance: initial tokens, the
tokens, text, sums and probabilities upstream returns, and each step's
chosen token, its two greatest logits and the timestamp rule's margin.
"""
import hashlib, json, math, os, sys

import numpy as np
import soundfile as sf
import torch
import whisper
import whisper.decoding as wd
from whisper.decoding import ApplyTimestampRules, DecodingOptions, DecodingTask

# The records were made on two threads.
torch.set_num_threads(2)
model_name, protocol, root, subset_tsv, weights, out_jsonl = sys.argv[1:7]
assert protocol in ("notimestamps", "timestamps")


def load(path):
    x, sr = sf.read(path, dtype="int16")
    assert sr == 16000 and x.ndim == 1
    return x.astype(np.float32) / 32768.0


def rows():
    with open(subset_tsv, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(header, line.rstrip("\n").split("\t")))


def fnum(x):
    return None if math.isinf(x) or math.isnan(x) else x


class _F:
    # Stands in for torch.nn.functional inside whisper.decoding, so the
    # log_softmax that ApplyTimestampRules computes is seen unchanged.
    def __init__(self):
        self.active, self.last = False, None

    def __getattr__(self, name):
        return getattr(torch.nn.functional, name)

    def log_softmax(self, *a, **kw):
        out = torch.nn.functional.log_softmax(*a, **kw)
        if self.active:
            self.last = out
        return out


F_PROXY = _F()
wd.F = F_PROXY


def decode(model, mel, options):
    task = DecodingTask(model, options)
    steps = []
    update = task.decoder.update
    margins = []

    # The timestamp rule's decision: logsumexp(timestamps) - max(text), on
    # the logits after the earlier masks; > 0 bans text at this step.
    for flt in task.logit_filters:
        if isinstance(flt, ApplyTimestampRules):
            apply, tb = flt.apply, task.tokenizer.timestamp_begin

            def hooked_apply(logits, tokens, apply=apply, tb=tb):
                F_PROXY.active = True
                try:
                    apply(logits, tokens)
                finally:
                    F_PROXY.active = False
                lp = F_PROXY.last[0]
                margins.append(float(lp[tb:].logsumexp(dim=-1) - lp[:tb].max()))

            flt.apply = hooked_apply

    # GreedyDecoder.update sees the logits after SuppressBlank, SuppressTokens
    # and ApplyTimestampRules. Batch of one: row 0.
    def hooked(tokens, logits, sum_logprobs):
        v, i = torch.topk(logits[0].float(), 2)
        lp = torch.log_softmax(logits[0].float(), dim=-1)
        n_finite = int(torch.isfinite(logits[0]).sum())
        out = update(tokens, logits, sum_logprobs)
        chosen = int(out[0][0, -1])
        steps.append({
            "token": chosen, "logprob": float(lp[chosen]),
            "top1": int(i[0]), "top2": int(i[1]),
            "logit1": float(v[0]), "logit2": fnum(float(v[1])),
            "gap": fnum(float(v[0] - v[1])), "n_finite": n_finite,
            "sum_logprob": float(sum_logprobs[0]),
        })
        if margins:
            steps[-1]["ts_rule_margin"] = fnum(margins[-1])
        return out

    task.decoder.update = hooked
    with torch.no_grad():
        result = task.run(mel.unsqueeze(0))[0]
    return result, steps, task


def main():
    model = whisper.load_model(model_name, device="cpu", download_root=weights)
    model.eval()
    options = DecodingOptions(language="en", without_timestamps=(protocol == "notimestamps"),
                              fp16=False)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows():
            audio = torch.from_numpy(load(os.path.join(root, r["path"])))
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), model.dims.n_mels)
            res, steps, task = decode(model, mel, options)
            rec = {
                "set": r["set"], "id": r["id"], "model": model_name, "protocol": protocol,
                "mel_sha256": hashlib.sha256(mel.numpy().tobytes()).hexdigest(),
                "initial_tokens": list(task.initial_tokens), "sample_begin": task.sample_begin,
                "tokens": res.tokens, "text": res.text, "language": res.language,
                "avg_logprob": res.avg_logprob, "sum_logprob": steps[-1]["sum_logprob"],
                "no_speech_prob": res.no_speech_prob, "compression_ratio": res.compression_ratio,
                "temperature": res.temperature, "n_steps": len(steps),
                "min_gap": min((s["gap"] for s in steps if s["gap"] is not None), default=None),
                "steps": steps, "reference": r["transcript"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(model_name, protocol, "utterances", n)


main()
