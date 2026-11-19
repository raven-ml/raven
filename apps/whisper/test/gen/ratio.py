# /// script
# # ratio.json records zlib 1.3.2, the zlib of uv's CPython 3.13.16.
# requires-python = "==3.13.16"
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
"""Goldens of upstream's compression_ratio (whisper/utils.py at 86098128) for
test_decoding: each text with its byte length n, len(zlib.compress(text))
and upstream's ratio, under the zlib this Python links (recorded). The
texts:
- decodes of LibriSpeech by tiny, tiny.en and base.en;
- the two long-form texts nearest 2.4;
- loops whose zlib ratio lies within 0.1 of 2.4;
- the 24 runs of at most 31 bytes that zlib scores above 2.4, and "";
- Hebrew and Hindi sentences written for this test, repeated near 2.4;
- a text ending in U+FFFD.

    uv run apps/whisper/test/gen/ratio.py OUT

OUT is apps/whisper/test/upstream.

Writes OUT/ratio.json.
"""
import json, os, sys, zlib

from whisper.utils import compression_ratio

T = 2.4

# Upstream's decodes with timestamps of the first three utterances of
# subset.py's subset, by tiny, tiny.en and base.en (greedy.py), each text
# once.
ORACLE = [
    "He began a confused complaint against the wizard, who had vanished behind the curtain on the left.",
    "Kiv not so earnest in mind to these memories child",
    "a golden fortune and a happy life.",
    "Give not so earnest a mind to these memories, child.",
]

# Upstream's transcribe by tiny of two LibriSpeech chapters with noise added
# (test-clean 4970-29095 and test-other 8131-117029): the windows at
# temperature 0 whose ratios lie nearest 2.4, 2.354 and 2.413.
NEAR = [
    "If you want to marry me, absolutely, I will not be able to fly the future of the city. If you are a little bit like a little bit, you will not find the future of the city. If you are a little bit like a little bit, you will not be able to fly the future of the city.",
    "I'm so tired of it. I'm tired of it. I'm tired of it. I'm tired of it. I'm tired of it. Finally, I got out of the world. It's much easier. It's a one-year-old, it's a nightmare. It's a... Let's say it's just a nightmare. I'm tired of it. It's just a nightmare. It's just a nightmare. It's just a block of pain. It's just a nightmare.",
]

texts = []


def add(kind, text):
    if any(t == text for _, t in texts):
        return
    texts.append((kind, text))


for t in ORACLE:
    add("oracle", t)
for t in NEAR:
    add("near", t)


def ratio(t):
    b = t.encode()
    return len(b) / len(zlib.compress(b))


# Loops: a phrase repeated, cut at a character count, whose ratio lies
# within 0.1 of 2.4, both sides.
phrases = [
    "I'm tired of it. ",
    "It's just a nightmare. ",
    "And then he said, ",
    "you know, ",
    "Thank you for watching. ",
    "The, the ",
]
for p in phrases:
    below = above = None
    for n in range(len(p), 40 * len(p)):
        s = (p * 40)[:n].strip()
        r = ratio(s)
        if T - 0.1 <= r <= T and below is None:
            below = s
        if T < r <= T + 0.1 and above is None:
            above = s
    for s in (below, above):
        if s is not None:
            add("loop", s)

# Runs of at most 31 bytes, the ones over 2.4.
units = ["!", ".", "-", "a", "ha", "Ha", "ha ", "Ha! ", "I ", "no ", "la ", "♪",
         "♪ ", "哈", "呵", "は", "ハ", "ㅡ", "ㅋ", "हा", "ха", "ха-", "הה"]
short = set()
for u in units:
    for k in range(1, 40):
        s = (u * k).strip()
        if 20 <= len(s.encode()) <= 31 and ratio(s) > T:
            short.add(s)
assert len(short) == 24, len(short)
for s in sorted(short):
    add("short", s)
add("empty", "")

# Hebrew and Hindi, written for this test: each sentence alone, then
# repeated as often as brings it nearest 2.4 from each side.
written = [
    "שלום, מה שלומך היום? אני בסדר, תודה רבה.",
    "הילדים הלכו לבית הספר בבוקר.",
    "नमस्ते, आज आप कैसे हैं? मैं ठीक हूँ, धन्यवाद।",
    "बच्चे सुबह स्कूल गए और शाम को लौटे।",
]
for w in written:
    add("written", w)
    below = above = None
    for k in range(1, 12):
        s = " ".join([w] * k)
        r = ratio(s)
        if r <= T:
            below = s
        elif above is None:
            above = s
    for s in (below, above):
        if s is not None:
            add("written", s)

# A window's last token can cut a character: tiktoken's replacement.
add("replacement", "And so it goes �")
add("replacement", "���")

rows = []
for kind, t in texts:
    b = t.encode()
    rows.append({
        "kind": kind,
        "text": t,
        "n": len(b),
        "zlib": len(zlib.compress(b)),
        "ratio": compression_ratio(t),
    })
    # upstream's own function, checked against the definition.
    assert rows[-1]["ratio"] == len(b) / len(zlib.compress(b))

with open(os.path.join(sys.argv[1], "ratio.json"), "w") as f:
    json.dump({"zlib": zlib.ZLIB_RUNTIME_VERSION, "texts": rows}, f,
              ensure_ascii=False, indent=0)
    f.write("\n")
print(len(rows), "texts, zlib", zlib.ZLIB_RUNTIME_VERSION)
