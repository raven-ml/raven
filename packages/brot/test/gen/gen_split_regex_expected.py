# Generates the expected rows of the regular-expression Split tests in
# test_pretokenizers.ml (`test_split_regex_cl100k`, `test_split_regex_o200k`,
# `test_split_regex_behaviors`): the pieces HuggingFace tokenizers gives for
# `pre_tokenizers.Split(Regex(pattern), behavior, invert)`, with its character
# offsets turned into the byte offsets brot reports. Run with
#
#   uv run --with tokenizers==0.23.1 python3 gen_split_regex_expected.py
#
# from this directory and paste the printed fragments over the case lists.

from tokenizers import Regex, pre_tokenizers

# The pattern of Llama 3's tokenizer.json.
CL100K = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# The pattern of the o200k family (GPT-4o, gpt-oss).
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

BEHAVIOR_CASES = [
    (r"\d+", ["a1b22c", "1a2"]),
    (r"[,;]\s*", ["a, b;c", ",,a"]),
    # Empty matches cut the text where they stand.
    (r"x*", ["axxb", "ab", "éxé"]),
    (r"(?=b)", ["abab"]),
    (r"a(?=b)|b", ["abab"]),
]

BEHAVIORS = {
    "isolated": "`Isolated",
    "removed": "`Removed",
    "merged_with_previous": "`Merged_with_previous",
    "merged_with_next": "`Merged_with_next",
    "contiguous": "`Contiguous",
}


def ocaml(b: bytes) -> str:
    out = []
    for c in b:
        if c == 0x22:
            out.append('\\"')
        elif c == 0x5C:
            out.append("\\\\")
        elif 0x20 <= c < 0x7F:
            out.append(chr(c))
        else:
            out.append("\\x%02X" % c)
    return '"' + "".join(out) + '"'


def pieces(pattern, behavior, invert, text):
    pre = pre_tokenizers.Split(Regex(pattern), behavior, invert=invert)
    rows = []
    for piece, (start, stop) in pre.pre_tokenize_str(text):
        assert text[start:stop] == piece
        rows.append(
            (piece.encode(), len(text[:start].encode()), len(text[:stop].encode()))
        )
    return rows


def emit(name, pattern, behavior, invert, text):
    body = "; ".join(
        "(%s, (%d, %d))" % (ocaml(p), a, b)
        for p, a, b in pieces(pattern, behavior, invert, text)
    )
    print(
        "  case %s %s ~invert:%s %s [ %s ];"
        % (name, BEHAVIORS[behavior], str(invert).lower(), ocaml(text.encode()), body)
    )


print("(* test_split_regex_cl100k *)")
for text in CL100K_TEXTS:
    emit("~pattern:cl100k", CL100K, "isolated", False, text)
print("\n(* test_split_regex_o200k *)")
for text in O200K_TEXTS:
    emit("~pattern:o200k", O200K, "isolated", False, text)
print("\n(* test_split_regex_behaviors *)")
for pattern, texts in BEHAVIOR_CASES:
    for behavior in BEHAVIORS:
        for invert in (False, True):
            for text in texts:
                emit("~pattern:" + ocaml(pattern.encode()), pattern, behavior, invert, text)
