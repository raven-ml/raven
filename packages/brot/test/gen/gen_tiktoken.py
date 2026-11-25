# /// script
# requires-python = ">=3.11"
# dependencies = ["tiktoken==0.14.0"]
# ///
"""Generate the tiktoken goldens: tiktoken's own output for `Brot.of_tiktoken`.

    WHISPER_SRC=<dir> uv run packages/brot/test/gen/gen_tiktoken.py [--check]

With `--check` it writes nothing and fails if a fixture differs from what it
would write.

`WHISPER_SRC` is the `whisper` package directory of openai/whisper at commit
86098128, which the `openai-whisper` 20250625 wheel also ships. Its
`tokenizer.py` gives Whisper's special tokens and its `assets` the gpt2 and
multilingual rank files, each checked against the sha256 below. cl100k_base and
o200k_base are tiktoken's own, which it downloads and checks.

For each encoding (Whisper's gpt2 and multilingual, cl100k_base, o200k_base)
the rank table is trimmed to the 256 single bytes, the tokens tiktoken emits
on TEXTS, and every part of each such token's own merge run, closed under the
same rule. A run on a trimmed table then takes the same steps as on the full
one: every part the full run makes is kept, and the minimum over a subset that
holds the full minimum is that minimum. The goldens are tiktoken's encodings on
the trimmed table, checked equal to the full table's. `support/tiktoken`
receives the `<name>.tiktoken` files and `goldens.json`, which
`test_tiktoken.ml` reads.
"""

import base64
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import tiktoken

OUT = Path(__file__).resolve().parent.parent / "support/tiktoken"
WHISPER = os.environ.get("WHISPER_SRC", "")
# openai/whisper 86098128.
PINS = {
    "tokenizer.py": "3b48e361a7e95b4ec0356ca6d72bba635778aa10269153136ee7bc34cae30b85",
    "assets/gpt2.tiktoken": "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    "assets/multilingual.tiktoken": "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126",
}
# The published files, as tiktoken_ext/openai_public.py pins them.
PUBLISHED = {
    "cl100k_base": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    "o200k_base": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
}

WHISPER_PAT = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
# cl100k_base's pattern in brot's dialect: possessive quantifiers as greedy
# ones, which they equal here, and \s++$ as \s+(?![\s\S]).
CL100K_BROT = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+(?![\s\S])|\s*[\r\n]|\s+(?!\S)|\s"""
)

# Hand-written texts, then six speech transcripts in four casings each, then
# texts for the patterns' corners. None holds a character that Unicode 15.1
# leaves unassigned: tiktoken's engine (16.0) and brot (18.0) may classify such
# a character differently.
TEXTS = [
    "",
    " ",
    "  ",
    "   ",
    "\t",
    "\n",
    "\n\n",
    " \n ",
    "a  b",
    "a   b",
    "a\t\tb",
    "a \n b",
    "trailing space ",
    " leading space",
    "  two leading",
    "end\n",
    "Hello, world!",
    " Hello, world!",
    "HELLO WORLD",
    "hello world",
    "HeLLo WoRLD",
    " And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
    "It's a dog's life; isn't it? I'm sure we'll see, they've said, you'd know.",
    "IT'S ISN'T I'M WE'LL THEY'VE YOU'D",
    "'s 't 're 've 'm 'll 'd",
    "O'Brien's",
    "rock 'n' roll",
    '"Quoted," she said -- then (aside) [note] {brace} <angle>.',
    "#hashtag @user a+b=c 5*3/2 x^2 a_b `code` a|b ~tilde",
    "- dash",
    " -",
    " '",
    "--",
    "---",
    "<<",
    ">>",
    "((",
    "))",
    "[[",
    "]]",
    "{{",
    "}}",
    "♪",
    "♪♪",
    "♪♪♪",
    "♩♪♫♬♭♮♯",
    "♪ la la la ♪",
    "「引用」『二重』",
    "0",
    "1",
    "12",
    "123",
    "1234",
    "12345",
    "1,234,567.89",
    "3.14159",
    "-42",
    "+1 (555) 010-0199",
    "1st 2nd 3rd 4th",
    "$2.07",
    "£5",
    "€10",
    "¢7",
    "50%",
    "1990s",
    "10:30 a.m.",
    "2026-11-18",
    "Mr. Smith and Dr. Jones met St. John.",
    "naïve café résumé coöperate façade Ångström",
    "Straße Œuvre ðþ łódź æsir ø",
    "我们今天去公园散步。",
    "你好，世界！",
    "中文和English混合",
    "今日はいい天気ですね。",
    "カタカナとひらがなと漢字",
    "ｶﾀｶﾅ半角",
    "مرحبا بالعالم",
    "السلام عليكم ورحمة الله",
    "नमस\u094dत\u0947 द\u0941न\u093fय\u093e",
    "ह\u093fन\u094dद\u0940 भ\u093eष\u093e",
    "Привет, мир!",
    "Съешь же ещё этих мягких французских булок.",
    "안녕하세요",
    "สว\u0e31สด\u0e35คร\u0e31บ",
    "שלום עולם",
    "Γειά σου Κόσμε",
    "Xin chào thế giới",
    "😀",
    "👍🏽",
    "👨\u200d👩\u200d👧\u200d👦",
    "🇫🇷",
    "emoji 😀 in text 🎉!",
    "❤\ufe0f",
    "\u00a0nbsp",
    "zero\u200bwidth",
    "e\u0301 combining",
    "\ufeffbom",
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|en|>",
    "<|yue|>",
    "<|transcribe|>",
    "<|translate|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
    "<|0.00|>",
    "<|1.08|>",
    "<|30.00|>",
    "<|30.02|>",
    "text <|endoftext|> more",
    "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello",
    "<|endoftext",
    "endoftext|>",
    "<|>",
    "<||>",
    "HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT",
    " HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT",
    " he began a confused complaint against the wizard who had vanished behind the curtain on the left",
    " He began a confused complaint against the wizard who had vanished behind the curtain on the left",
    "GIVE NOT SO EARNEST A MIND TO THESE MUMMERIES CHILD",
    " GIVE NOT SO EARNEST A MIND TO THESE MUMMERIES CHILD",
    " give not so earnest a mind to these mummeries child",
    " Give not so earnest a mind to these mummeries child",
    "A GOLDEN FORTUNE AND A HAPPY LIFE",
    " A GOLDEN FORTUNE AND A HAPPY LIFE",
    " a golden fortune and a happy life",
    " A golden fortune and a happy life",
    "ALSO A POPULAR CONTRIVANCE WHEREBY LOVE MAKING MAY BE SUSPENDED BUT NOT STOPPED DURING THE PICNIC SEASON",
    " ALSO A POPULAR CONTRIVANCE WHEREBY LOVE MAKING MAY BE SUSPENDED BUT NOT STOPPED DURING THE PICNIC SEASON",
    " also a popular contrivance whereby love making may be suspended but not stopped during the picnic season",
    " Also a popular contrivance whereby love making may be suspended but not stopped during the picnic season",
    "HARANGUE THE TIRESOME PRODUCT OF A TIRELESS TONGUE",
    " HARANGUE THE TIRESOME PRODUCT OF A TIRELESS TONGUE",
    " harangue the tiresome product of a tireless tongue",
    " Harangue the tiresome product of a tireless tongue",
    "ANGOR PAIN PAINFUL TO HEAR",
    " ANGOR PAIN PAINFUL TO HEAR",
    " angor pain painful to hear",
    " Angor pain painful to hear",
    "a  \n  ",
    "line\r\nnext\r\n",
    "trailing\n  ",
    "x \n\n \t",
    "  \n",
    "\n ",
    "1234567890",
    "12345678 9",
    "ABCdef GHI'S",
    "don't DON'T 'S 'Ll",
    "  \u00a0 x",
    "tab\tsep",
    "multi\n\n\nlines",
    "code: def f(x):\n    return x**2\n",
    "https://example.com/a/b?c=d",
    "Hello\u3000world",
    "ℕ ≥ ∑ ∞",
    "ǅungla ǈ ǋ",
    "\u0301 alone",
    "path/to/file\n/",
    "A\u0301B\u0301 École",
]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pinned(rel):
    if not WHISPER:
        sys.exit("set WHISPER_SRC to openai/whisper's whisper directory")
    path = os.path.join(WHISPER, rel)
    got = sha256(path)
    if got != PINS[rel]:
        sys.exit(f"{path}: sha256 {got}, expected {PINS[rel]}")
    return path


def whisper_module():
    spec = importlib.util.spec_from_file_location(
        "whisper_tokenizer", pinned("tokenizer.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_ranks(path):
    # As whisper's get_encoding reads them.
    with open(path, "rb") as f:
        return {
            base64.b64decode(t): int(r) for t, r in (line.split() for line in f if line)
        }


def whisper_specials(mod, num_languages):
    return [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(mod.LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]


def numbered(names, start):
    return {s: start + i for i, s in enumerate(names)}


def run(b, ranks):
    """tiktoken's merge on the bytes [b]: the final parts, the last pair
    joined, and every part made on the way."""
    parts = [b[i : i + 1] for i in range(len(b))]
    made = set()
    last = None
    while len(parts) > 1:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        last = (parts[i], parts[i + 1])
        parts[i : i + 2] = [parts[i] + parts[i + 1]]
        made.add(parts[i])
    return parts, last, made


def closure(tokens, ranks):
    kept = set(tokens)
    work = list(tokens)
    while work:
        t = work.pop()
        _, _, made = run(t, ranks)
        for p in made:
            if p not in kept:
                kept.add(p)
                work.append(p)
    return kept


def encoding(name, pat, ranks, specials):
    return tiktoken.Encoding(
        name, pat_str=pat, mergeable_ranks=ranks, special_tokens=specials
    )


def cases(enc, ts):
    rows = []
    for t in ts:
        ordinary = enc.encode_ordinary(t)
        rows.append(
            {
                "text": t,
                "ordinary": ordinary,
                "all": enc.encode(t, allowed_special="all"),
                "decode_each": [enc.decode([i], errors="replace") for i in ordinary],
            }
        )
    return rows


def rank_file(ranks):
    lines = []
    for tok, r in sorted(ranks.items(), key=lambda x: x[1]):
        # The empty token as multilingual.tiktoken writes it.
        field = base64.b64encode(tok) if tok else b"="
        lines.append(field + b" " + str(r).encode() + b"\n")
    return b"".join(lines)


def published(enc):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, enc.name + ".tiktoken")
        with open(path, "wb") as f:
            f.write(rank_file(enc._mergeable_ranks))
        if sha256(path) != PUBLISHED[enc.name]:
            sys.exit(f"{enc.name}: not the published file")


def main() -> int:
    mod = whisper_module()
    gpt2 = read_ranks(pinned("assets/gpt2.tiktoken"))
    multi = read_ranks(pinned("assets/multilingual.tiktoken"))
    cl100k = tiktoken.get_encoding("cl100k_base")
    o200k = tiktoken.get_encoding("o200k_base")
    published(cl100k)
    published(o200k)

    # name, rank file stem, full ranks, tiktoken's pattern, brot's, specials
    specs = [
        ("gpt2", "gpt2", gpt2, WHISPER_PAT, WHISPER_PAT,
         numbered(whisper_specials(mod, 99), len(gpt2))),
        ("multilingual99", "multilingual", multi, WHISPER_PAT, WHISPER_PAT,
         numbered(whisper_specials(mod, 99), len(multi))),
        ("multilingual100", "multilingual", multi, WHISPER_PAT, WHISPER_PAT,
         numbered(whisper_specials(mod, 100), len(multi))),
        ("cl100k", "cl100k", cl100k._mergeable_ranks, cl100k._pat_str, CL100K_BROT,
         cl100k._special_tokens),
        ("o200k", "o200k", o200k._mergeable_ranks, o200k._pat_str, o200k._pat_str,
         o200k._special_tokens),
    ]  # fmt: skip

    # Per rank file, the tokens every encoding of it reaches.
    reached = {}
    for name, stem, ranks, pat, _, specials in specs:
        full_enc = encoding(name, pat, ranks, specials)
        inverse = {r: b for b, r in ranks.items()}
        toks = reached.setdefault(stem, set())
        for t in TEXTS:
            toks.update(inverse[i] for i in full_enc.encode_ordinary(t))
    trimmed = {}
    files = {}
    for stem, toks in reached.items():
        ranks = next(s[2] for s in specs if s[1] == stem)
        kept = closure(toks, ranks)
        kept.update(bytes([b]) for b in range(256))
        if b"" in ranks:
            kept.add(b"")
        trimmed[stem] = {b: ranks[b] for b in kept}
        files[f"{stem}.tiktoken"] = rank_file(trimmed[stem])

    # Whisper's composition on a table whose ranks are contiguous, so that the
    # specials are numbered from its end as whisper numbers them.
    by_rank = sorted(trimmed["multilingual"].items(), key=lambda x: x[1])
    compact = {b: i for i, (b, _) in enumerate(by_rank)}
    files["whisper.tiktoken"] = rank_file(compact)
    specs.append(("whisper", "whisper", compact, WHISPER_PAT, WHISPER_PAT,
                  numbered(whisper_specials(mod, 99), len(compact))))  # fmt: skip
    trimmed["whisper"] = compact

    goldens = []
    for name, stem, ranks, pat, brot_pat, specials in specs:
        enc = encoding(name, pat, trimmed[stem], specials)
        rows = cases(enc, TEXTS)
        if stem != "whisper":
            full_enc = encoding(name, pat, ranks, specials)
            for row in rows:
                text = row["text"]
                assert full_enc.encode_ordinary(text) == row["ordinary"], (name, text)
                assert full_enc.encode(text, allowed_special="all") == row["all"], (
                    name,
                    text,
                )
        goldens.append(
            {
                "name": name,
                "ranks": f"{stem}.tiktoken",
                "pattern": brot_pat,
                "specials": sorted(specials.items(), key=lambda x: x[1]),
                "cases": rows,
            }
        )
    files["goldens.json"] = (
        json.dumps({"encodings": goldens}, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode()

    if "--check" in sys.argv[1:]:
        stale = [n for n, b in files.items() if (OUT / n).read_bytes() != b]
        for n in stale:
            print(f"{OUT / n} differs from what this would write", file=sys.stderr)
        return 1 if stale else 0
    for n, b in files.items():
        (OUT / n).write_bytes(b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
