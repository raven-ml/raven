# Generates the expected rows of the regular-expression Split tests in
# test_pretokenizers.ml (`test_split_regex_cl100k`, `test_split_regex_o200k`,
# `test_split_regex_tekken`, `test_split_regex_kimi`,
# `test_split_regex_classes`, `test_split_regex_behaviors`) and the
# generated rows of `test_unicode_scripts`: the pieces HuggingFace tokenizers
# gives for `pre_tokenizers.Split(Regex(pattern), behavior, invert)` and
# `pre_tokenizers.UnicodeScripts()`, with its character offsets turned into
# the byte offsets brot reports. Run with
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

# Characters Unicode 18.0.0 assigns, which have no script in Unicode 17.0.0 or
# in HuggingFace's table (U+0558 is Armenian since 18.0.0).
SCRIPT_TEXTS = ["a\u0558b", "\u0558abc", "\u65e5\u0558\u672c"]

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


def emit_scripts(text):
    rows = []
    for piece, (start, stop) in pre_tokenizers.UnicodeScripts().pre_tokenize_str(text):
        rows.append(
            "(%s, (%d, %d))"
            % (ocaml(piece.encode()), len(text[:start].encode()), len(text[:stop].encode()))
        )
    print("  test_case %s [ %s ];" % (ocaml(text.encode()), "; ".join(rows)))


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
for name, pattern in (("o200k", O200K), ("tekken", TEKKEN), ("kimi", KIMI)):
    print("\n(* test_split_regex_%s, the word texts *)" % name)
    for text in WORD_TEXTS:
        emit("~pattern:" + name, pattern, "isolated", False, text)
print("\n(* test_split_regex_classes *)")
for pattern, texts in CLASS_CASES:
    for text in texts:
        emit("~pattern:" + ocaml(pattern.encode()), pattern, "isolated", False, text)
print("\n(* test_split_regex_behaviors *)")
for pattern, texts in BEHAVIOR_CASES:
    for behavior in BEHAVIORS:
        for invert in (False, True):
            for text in texts:
                emit("~pattern:" + ocaml(pattern.encode()), pattern, behavior, invert, text)
print("\n(* test_unicode_scripts *)")
for text in SCRIPT_TEXTS:
    emit_scripts(text)
