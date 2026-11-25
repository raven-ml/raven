# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.1"]
# ///
"""Generate the goldens of test_pre_tokenizer.ml that only HuggingFace
tokenizers can compute: the pretokens of regular-expression splits and of
UnicodeScripts.

    uv run packages/brot/test/gen/gen_split_regex_expected.py [--check]

It writes `support/pre_tokenizer/split_regex.json`, the pieces
`pre_tokenizers.Split(Regex(pattern), behavior, invert)` gives for each
pattern, behavior and text below, and `support/pre_tokenizer/unicode_scripts.json`,
those of `pre_tokenizers.UnicodeScripts()`, with tokenizers' character
offsets turned into the byte offsets brot reports. `split_regex.json` also
holds the patterns of tokenizer files that brot matches by name
(`PATTERNS`). With `--check` it writes nothing and fails, naming the file,
if a committed golden differs from what it would write.
"""

import json
import sys
from pathlib import Path

from tokenizers import Regex, pre_tokenizers

SUPPORT = Path(__file__).resolve().parents[1] / "support" / "pre_tokenizer"

# The pattern of Llama 3's tokenizer.json.
CL100K = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# The pattern of the o200k family (GPT-4o, gpt-oss): `o200k_base` in tiktoken's
# tiktoken_ext/openai_public.py, as gpt-oss's tokenizer.json writes it.
O200K = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

# Tekken's, Mistral's: the "pattern" of tekken.json in
# mistralai/Mistral-Nemo-Instruct-2407, which its tokenizer.json repeats.
TEKKEN = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
        r"\p{N}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

# Kimi K2's: `pat_str` in tokenization_kimi.py of moonshotai/Kimi-K2-Instruct.
KIMI = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

CL100K_TEXTS = [
    # The worked examples: a run before a letter, newlines, a contraction.
    "a  b", "x\n\n  y", "I'M 123  ",
    # Whitespace runs of 1, 2, 3 before a letter, a digit, punctuation, the end.
    " a", "  a", "   a", " 1", "  1", "   1", " !", "  !", "   !", " ", "  ", "   ",
    # Other whitespace characters as the prefix of a word.
    "\ta", " a", "　a", "\t\ta", "  a", "a　　",
    # Newlines.
    "a\r\nb", "a\n\rb", "\r\r\n  x", "a \n b", "a  \n  b", "a\n", "\n  ",
    # Contractions, case-insensitive, U+017F and U+212A included.
    "I'M", "HE'LL", "it'ſ", "'Sx", "''s", "'", "a'", "'re're", "x'Ve", "'K",
    # Digit groups of at most three, over every kind of number.
    "1", "12", "123", "1234", "1234567", "١٢٣٤",
    "１２３４", "1²2", "12ab34",
    # Punctuation, its optional space and the newlines it absorbs.
    "!!!\n\nfoo", " !!!", "a!?\r\n\r\nb", "!\n ", "!a", "$100", "a.b", "...a",
    " ...\n\n\na", "a_b-c", "_a", "-a", "\x00a", "\x7fa",
    # Marks are not letters: precomposed and decomposed text differ.
    "été", "été", "́a", "á", " ́",
    # Emoji, joiners and variation selectors.
    "\U0001F600", "a\U0001F600b", "\U0001F468‍\U0001F469‍\U0001F467",
    "❤️ ok",
    # Scripts.
    "日本語 123", "مرحبا بك",
    "नमस्ते",
    # Source code and a URL.
    "def f(x):\r\n\treturn x+1\r\n", "https://a.b/c?d=1&e=2",
]

O200K_TEXTS = [
    "Hello World", "helloWorld", "HelloWORLDfoo", "XMLHttpRequest",
    "I'M he'll DON'T", "a's'T", "a//b", "!!/\n/x", "x =/ y", " /", "12345",
    "éÁb", "ǅa",
]

# The texts the o200k, Tekken and Kimi patterns are compared on: case runs
# and the letters of both classes ([Lm], [Lo], marks) between them, CJK and
# mixed scripts, contractions, numbers, punctuation with newlines and
# slashes, emoji and whitespace runs.
WORD_TEXTS = [
    # Case: capitals give back up to the last letter of both classes.
    "iPhone eBay HTTPServer", "ÉCOLEécole İstanbul", "日本A", "A日B",
    "ʰA Aʰ AʰB", "ǅungla ǄA", "𝐀𝐁c 𝐀b", "ﬁne Straße",
    # Marks as a prefix, inside words and in punctuation.
    "a\u0301B \u0301A", "\u0301", " \u0301x", "!\u0301!", "e\u0301\u0301E",
    "\u0301Z", "!\u0301Z", "\u0301日Z",
    "नमस्ते दुनिया",
    # Scripts, with and without spaces.
    "Привет Мир", "Γειά σου", "مرحبا123 שלום", "한국어 텍스트",
    "ひらがなカタカナ漢字", "中文，标点。English混合", "日本語abc", "Abc日's",
    "〇一二 々 a々b 、。", "⺀⺁ 𠀀𠀁 ｶﾀｶﾅ", "\u4e00\u0301 \u0301日 !日 123日",
    # Contractions.
    "eBay's McDONALD'S OK'd won't' a'ſ 'll x'LL",
    # Numbers.
    "1234567 ١٢٣٤٥ １２３ ½¾ x² 3.14159 12,345",
    # Punctuation, newlines and slashes.
    "a/b/c http://x.y/z \\/", " //\n//", "!/\r\n/a", "(a) [b] {c}",
    # Emoji.
    "😀😀 a😀b", "👨‍👩‍👧 ❤️ok", "🇫🇷 👍🏽",
    # Whitespace runs.
    "a\u3000b \u00a0\u00a0x", "  \t\n  x", "x \n", "\n\n\n", "\r\n\r\n",
    " \r\n x", "a   ", "\u2028a \u0085a",
]

# Nested classes, intersection and scripts, isolated.
CLASS_CASES = [
    (r"[a-z&&[^aeiou]]+", ["abcdefe"]),
    (r"[^a-z&&[^aeiou]]+", ["abcdefe1"]),
    (r"[a-z&&b-d&&c-z]+", ["abcdefe"]),
    # An empty side of [&&] is the empty set.
    (r"[a&&]+", ["a&b"]),
    (r"[&&a]+", ["a&b"]),
    (r"[a[bc]d]+", ["abcdex"]),
    (r"[[^a]&&[^b]]+", ["abcab"]),
    (r"[\p{L}&&\p{Greek}]+", ["aβγd1"]),
    # Han is the Script property: U+3001 is Common, U+3005 and U+3007 Han.
    (r"\p{Han}+", ["a日本b〇、々ー"]),
    (r"\p{Hani}+|\p{Latin}+", ["ab日本Ωc"]),
    (r"\P{Han}+", ["日a本"]),
    (r"[^\p{Han}]+", ["日a本"]),
    (r"\p{Hiragana}+|\p{Katakana}+", ["ひらがなカタカナー"]),
    (r"\p{Common}+", ["a1!、b"]),
    (r"\p{Inherited}+", ["a\u0301\u0302b"]),
    (r"\p{Arabic}+", ["abc مرحبا"]),
]

BEHAVIOR_CASES = [
    (r"\d+", ["a1b22c", "1a2"]),
    (r"[,;]\s*", ["a, b;c", ",,a"]),
    # Empty matches cut the text where they stand.
    (r"x*", ["axxb", "ab", "éxé"]),
    (r"(?=b)", ["abab"]),
    (r"a(?=b)|b", ["abab"]),
    # Alternatives that open alike are still tried in turn.
    (r"x?x|x?y", ["xy"]),
]

BEHAVIORS = ["isolated", "removed", "merged_with_previous", "merged_with_next", "contiguous"]


# Qwen2's, in the tokenizer.json of Qwen/Qwen2.5-7B: cl100k's with numbers one
# at a time.
QWEN2 = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Qwen3.5's: Qwen2's with the combining marks in its words.
QWEN35 = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# cl100k as some tokenizer files spell it, its newline alternative without the
# `+`, which matches the same text.
CL100K_HF = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

PATTERNS = {
    "cl100k": CL100K,
    "cl100k_hf": CL100K_HF,
    "qwen2": QWEN2,
    "qwen35": QWEN35,
    "o200k": O200K,
    "tekken": TEKKEN,
    "kimi": KIMI,
}

# The texts UnicodeScripts is compared on. Hiragana, Katakana and U+30FC
# folding into Han is stated by brot.mli and tested there by hand.
SCRIPT_TEXTS = [
    "Hello world", "abc!def", "a b", "",
    # A leading run of spaces opens no piece.
    "  !  ", " abc", "  文", "   ", "abc  ",
    # Only U+0020 joins the run around it; other white space has a script of
    # its own: Common, or Ogham for U+1680.
    "a\tb", "a\rb", "\n\nabc def", "  \n  abc", "a 　b",
    "a b", "a b", "a b", "a b", " abc",
    # A character of no known script joins the run around it too: U+E000 is
    # private use.
    "ab", "abc", "    abc",
    # U+0558 is Armenian since Unicode 18.0.0, and has no script in 17.0.0 or
    # in tokenizers' table.
    "a՘b", "՘abc", "日՘本",
    # The iteration mark U+3005 is Han.
    "中々中 abc",
    # A script change cuts; digits are Common.
    "中文 abc", "Hello世界", "a１２３",
    " 123 中文 abc ",
    # One script, one piece, however many words.
    "Ελληνικά abc",
    "Привет мир",
    "abc Привет",
    "مرحبا بالعالم",
    "שלום abc", "안녕 abc",
    "नमस्ते abc",
    # Inherited is a script of its own: a combining mark and a variation
    # selector cut off what they attach to.
    "éx", "a❤️b",
]


def pretokens(pre, text):
    """The pieces of `text` as [piece, start, stop], offsets in bytes."""
    rows = []
    for piece, (start, stop) in pre.pre_tokenize_str(text):
        rows.append(
            [piece, len(text[:start].encode()), len(text[:stop].encode())]
        )
    return rows


def split_set(group, pattern, behavior, invert, texts):
    pre = pre_tokenizers.Split(Regex(pattern), behavior, invert=invert)
    return {
        "group": group,
        "pattern": pattern,
        "behavior": behavior,
        "invert": invert,
        "cases": [
            {"text": text, "pretokens": pretokens(pre, text)} for text in texts
        ],
    }


def split_regex():
    sets = [
        split_set("cl100k", CL100K, "isolated", False, CL100K_TEXTS),
        split_set("o200k", O200K, "isolated", False, O200K_TEXTS + WORD_TEXTS),
        split_set("tekken", TEKKEN, "isolated", False, WORD_TEXTS),
        split_set("kimi", KIMI, "isolated", False, WORD_TEXTS),
    ]
    sets += [
        split_set("classes", pattern, "isolated", False, texts)
        for pattern, texts in CLASS_CASES
    ]
    sets += [
        split_set("behaviors", pattern, behavior, invert, texts)
        for pattern, texts in BEHAVIOR_CASES
        for behavior in BEHAVIORS
        for invert in (False, True)
    ]
    return {"patterns": PATTERNS, "sets": sets}


def unicode_scripts():
    pre = pre_tokenizers.UnicodeScripts()
    return [{"text": text, "pretokens": pretokens(pre, text)} for text in SCRIPT_TEXTS]


def line(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def cases(cases, indent) -> str:
    """One case per line, so that a change reads as a diff of cases."""
    pad = " " * indent
    return "[\n" + ",\n".join(pad + line(c) for c in cases) + "\n" + pad[2:] + "]"


def dump_split_regex(golden) -> str:
    patterns = ",\n".join(
        f"    {line(name)}: {line(pattern)}"
        for name, pattern in golden["patterns"].items()
    )
    sets = ",\n".join(
        "    " + line({k: v for k, v in s.items() if k != "cases"})[:-1]
        + ', "cases": ' + cases(s["cases"], 6) + "}"
        for s in golden["sets"]
    )
    return '{\n  "patterns": {\n' + patterns + '\n  },\n  "sets": [\n' + sets + "\n  ]\n}\n"


def main() -> int:
    goldens = {
        "split_regex.json": dump_split_regex(split_regex()),
        "unicode_scripts.json": cases(unicode_scripts(), 2) + "\n",
    }
    if "--check" in sys.argv[1:]:
        stale = [
            name
            for name, text in goldens.items()
            if not (SUPPORT / name).exists()
            or (SUPPORT / name).read_text(encoding="utf-8") != text
        ]
        for name in stale:
            print(f"support/pre_tokenizer/{name} differs from the generator's")
        return 1 if stale else 0
    SUPPORT.mkdir(parents=True, exist_ok=True)
    for name, text in goldens.items():
        (SUPPORT / name).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
