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
"""Tokenizer goldens for test_tokenizer: upstream's own Tokenizer
(whisper/tokenizer.py at 86098128) over trimmed rank tables, the first 1000
ranks of gpt2.tiktoken and multilingual.tiktoken (the multilingual one with
the empty placeholder token appended at rank 1000, as the full file holds it
last), with Whisper's specials appended exactly as get_encoding appends them.

    uv run apps/whisper/test/gen/tokenizer.py OUT

OUT is apps/whisper/test/upstream.

Writes OUT/gpt2-1000.tiktoken, OUT/multilingual-1000.tiktoken and
OUT/tokenizer.json: each configuration's specials, start sequences,
non-speech tokens, and upstream's encodings and decodings of hand strings
and id sequences.
"""
import base64, json, os, sys

import whisper.tokenizer as wt
from whisper.tokenizer import LANGUAGES, Tokenizer

from trimmed import encoding

# The ranks kept of each table.
n = 1000
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
assets = os.path.join(os.path.dirname(wt.__file__), "assets")


def read_ranks(name):
    with open(os.path.join(assets, f"{name}.tiktoken")) as f:
        return [(tok, int(rank)) for tok, rank in (line.split() for line in f if line)]


def cut(name):
    rows = [(t, r) for t, r in read_ranks(name) if r < n]
    if name == "multilingual":
        rows.append(("=", n))
    text = "".join(f"{t} {r}\n" for t, r in rows)
    return text, {base64.b64decode(t): r for t, r in rows}


HAND = [
    "", " ", "  ", "\t", "\n", "\n\n", " \n ", "a  b", "a\t\tb", "end\n", " leading",
    "Hello, world!", " Hello world.", "HELLO WORLD", " And so my fellow Americans, ask not.",
    "It's a dog's life; isn't it? I'm sure we'll see, they've said, you'd know.",
    "IT'S I'M WE'LL", "O'Brien's", "rock 'n' roll", "\"Quoted,\" (aside) [note] {brace} <angle>.",
    "#hashtag @user a+b=c 5*3/2 x^2 a_b `code` a|b ~tilde", "- dash", " -", " '", "--", "<<", ">>",
    "♪", "♪♪♪", "♩♪♫♬♭♮♯", "「引用」", "0", "12", "123", "1234", "1,234,567.89", "-42", "$2.07",
    "naïve café", "Straße", "我们今天", "今日は", "مرحبا", "नमस्ते", "Привет, мир!", "안녕", "😀",
    "👨‍👩‍👧", " nbsp", "zero​width", "<|endoftext|>", "<|en|>", "<|yue|>", "<|0.00|>",
    "<|30.02|>", "text <|endoftext|> more", "<|startoftranscript|><|en|><|transcribe|> Hello",
    "<|endoftext", "<|>",
    " HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN",
    " when is a lobster not a lobster when it is a crayfish",
]


def config(name, fname, text, ranks, languages, multilingual):
    enc = encoding(fname, ranks, languages)
    langs = list(LANGUAGES)[:languages]

    def tok(language=None, task=None):
        return Tokenizer(encoding=enc, num_languages=languages, language=language, task=task)

    t = tok()
    picked = ["<|endoftext|>", "<|startoftranscript|>", *[f"<|{l}|>" for l in langs],
              "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
              "<|nospeech|>", "<|notimestamps|>",
              *[f"<|{i * 0.02:.2f}|>" for i in (0, 1, 2, 54, 1499, 1500)]]
    sot = {"none": list(tok().sot_sequence)}
    if multilingual:
        for lang in ("en", "zh", langs[-1]):
            for task in ("transcribe", "translate"):
                sot[f"{lang}/{task}"] = list(tok(lang, task).sot_sequence)
    encodes = []
    for s in HAND:
        try:
            t.encode(s)
            refused = False
        except ValueError:
            refused = True
        ids = enc.encode_ordinary(s)
        encodes.append({"text": s, "ids": ids, "upstream_refuses": refused,
                        "decode": t.decode(ids)})
    tb, eot = t.timestamp_begin, t.eot
    hello = enc.encode_ordinary(" Hello world.")
    cases = [
        [tb, *hello, tb + 54, eot],
        [t.sot, *(tok("en", "transcribe").sot_sequence[1:] if multilingual else []), t.no_timestamps,
         *hello, eot],
        [tb, tb + 1, tb + 1499, tb + 1500],
        [tb - 1, tb], [eot, t.sot_prev, t.sot_lm, t.no_speech, t.transcribe, t.translate],
        [enc.encode_single_token(b"\xe6"), enc.encode_single_token(b"\x88")],
        [enc.encode_single_token(b"\xe6"), enc.encode_single_token(b"\x88"),
         enc.encode_single_token(b"\x91")],
        [enc.encode_single_token(b"a"), enc.encode_single_token(b"\xff"),
         enc.encode_single_token(b"b")],
        [enc.encode_single_token(b"\xe6"), tb + 3, enc.encode_single_token(b"\x88"),
         enc.encode_single_token(b"\x91")],
    ]
    if multilingual:
        cases.append([enc.encode_single_token(b"a"), n, enc.encode_single_token(b"b")])
    decodes = [{"ids": c, "decode": t.decode(c), "decode_with_timestamps": t.decode_with_timestamps(c)}
               for c in cases]
    return {
        "name": name, "file": fname, "languages": languages, "n_vocab": enc.n_vocab,
        "specials": [[s, t.special_tokens[s]] for s in picked],
        "timestamp_begin": tb, "sot_sequences": sot,
        "non_speech_tokens": list(t.non_speech_tokens),
        "encode": encodes, "decode": decodes,
    }


def main():
    configs = []
    for name, multilingual, counts in (("gpt2", False, (99,)), ("multilingual", True, (99, 100))):
        text, ranks = cut(name)
        fname = f"{name}-{n}.tiktoken"
        with open(os.path.join(out, fname), "w") as f:
            f.write(text)
        for k in counts:
            configs.append(config(f"{name}{k}", fname, text, ranks, k, multilingual))
    with open(os.path.join(out, "tokenizer.json"), "w") as f:
        json.dump({"generator": "apps/whisper/test/gen/tokenizer.py", "n": n,
                   "configs": configs}, f, ensure_ascii=False, indent=0)
    print("configs", len(configs))


main()
