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
"""Goldens of upstream's writers (whisper/utils.py at 86098128: WriteTXT,
WriteVTT, WriteSRT, WriteTSV, WriteJSON) for test_writer. Each case is a
transcript as the port holds it (windows with their seek, size, chosen
attempt's temperature, average log probability, compression ratio and
no-speech probability, and segments in frames) and the result dict
upstream's transcribe would return for it: segments numbered from 0, each
with its window's seek and decision, times in seconds as frame / 100, a
cleared segment (no tokens, no text) carrying transcribe.py's "words": [].

    uv run apps/whisper/test/gen/writer.py OUT

OUT is apps/whisper/test/upstream.

Writes OUT/writer.json: each case with each format's text.
"""
import io, json, os, sys

from whisper.utils import WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

out = sys.argv[1]


def window(seek, size, temperature, lp, cr, ns, segments):
    return {"seek": seek, "size": size, "temperature": temperature,
            "avg_logprob": lp, "compression_ratio": cr, "no_speech_prob": ns,
            "segments": [{"start": a, "stop": b, "tokens": t, "text": x}
                         for a, b, t, x in segments]}


cases = [
    {"name": "english", "language": "en", "windows": [
        window(0, 3000, 0.0, -0.35100454213667887, 1.4871794871794872, 0.17127984762191772, [
            (0, 280, [50364, 634, 4283, 50504], " Hello world."),
            (280, 560, [50504, 467, 50644], " It's --> here\tnow, \"quoted\" \\ back"),
            (560, 560, [], ""),
        ]),
        window(2796, 3000, 0.6000000000000001, -1.2e-05, 2.0, 1.25e-06, [
            (2796, 2801, [50364, 291, 50367], " Then."),
            (2801, 5796, [50367, 300, 50364], "And so-->on  "),
        ]),
        window(5796, 613, 0.2, -0.1, 0.5, 0.9, []),
    ]},
    {"name": "hours and white space", "language": "en", "windows": [
        window(359900, 3000, 0.0, -0.25, 1.1, 0.01, [
            (359900, 360010, [50364, 11, 50419], "　spoken "),
            (360010, 3600000, [50419, 12, 50500], " ten hours on "),
        ]),
    ]},
    {"name": "non-ASCII", "language": "ja", "windows": [
        window(0, 1234, 0.0, -0.5, 0.75, 0.002, [
            (0, 312, [50364, 1, 2, 50520], "東京は晴れ。"),
            (312, 1234, [50520, 3, 50981], " 🎉 café \u0001 é�"),
        ]),
    ]},
    {"name": "no segment", "language": "en", "windows": []},
    # Every segment cleared: upstream's subtitles read their empty words and
    # write no cue.
    {"name": "every segment cleared", "language": "en", "windows": [
        window(0, 3000, 0.0, -0.4, 1.0, 0.3, [(0, 0, [], ""), (100, 100, [], "")]),
        window(3000, 500, 1.0, -1.5, 0.9, 2 ** -24, [(3000, 3500, [], "")]),
    ]},
    # 2^-24, below a power of two: repr's shortest decimal is not the
    # correctly rounded one of its length.
    {"name": "a power of two", "language": "en", "windows": [
        window(0, 3000, 0.0, -2.0 ** -10, 2.0 ** 3, 2.0 ** -24, [
            (0, 100, [50364, 1, 50414], " Power."),
        ]),
    ]},
]


def result(c):
    segs = []
    for w in c["windows"]:
        for s in w["segments"]:
            g = {"id": len(segs), "seek": w["seek"], "start": s["start"] / 100,
                 "end": s["stop"] / 100, "text": s["text"], "tokens": s["tokens"],
                 "temperature": w["temperature"], "avg_logprob": w["avg_logprob"],
                 "compression_ratio": w["compression_ratio"],
                 "no_speech_prob": w["no_speech_prob"]}
            if not s["tokens"]:
                assert s["text"] == ""
                g["words"] = []
            segs.append(g)
    text = "".join(s["text"] for w in c["windows"] for s in w["segments"])
    c["text"] = text
    return {"text": text, "segments": segs, "language": c["language"]}


for c in cases:
    r = result(c)
    # A cleared first segment sends upstream's subtitles to word timings,
    # which raise unless every segment is cleared.
    cleared = ["words" in g for g in r["segments"]]
    assert not cleared or not cleared[0] or all(cleared)
    c["outputs"] = {}
    for W in (WriteTXT, WriteVTT, WriteSRT, WriteTSV, WriteJSON):
        f = io.StringIO()
        W(".").write_result(r, file=f)
        c["outputs"][W.extension] = f.getvalue()

with open(os.path.join(out, "writer.json"), "w", encoding="utf-8") as f:
    json.dump({"cases": cases}, f, ensure_ascii=False, indent=0)
    f.write("\n")
print(len(cases), "cases")
