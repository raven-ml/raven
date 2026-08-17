#!/usr/bin/env python3
"""Print the alignment HuggingFace `tokenizers` gives each normalizer.

The expectations of the "alignment" group in test/test_unicode.ml come from
this script. Run it with:

    uv run --with tokenizers python3 packages/brot/test/scripts/hf_alignments.py

`NormalizedString` keeps, for every byte of the normalized text, the byte range
of the original it stands for, and reports a span of normalized bytes as the
range of its first byte joined with the range of its last. The Python bindings
do not expose that vector, but `PreTokenizedString.get_splits` runs a split
through the same conversion, so splitting the normalized text into single
characters exposes the alignment of every one of them.

Each case prints the input, the normalized text and one span per normalized
character, escaped so that the lines can be pasted into the OCaml test.
"""

from tokenizers import NormalizedString, PreTokenizedString, Regex, normalizers

ANY = Regex(r"[\s\S]")


def align(normalizer, text):
    """(character, original byte span) for every character of the result."""
    pretokenized = PreTokenizedString(text)
    pretokenized.normalize(lambda string: normalizer.normalize(string))
    pretokenized.split(lambda _, string: string.split(ANY, "isolated"))
    return [
        (character, span)
        for character, span, _ in pretokenized.get_splits(
            offset_referential="original", offset_type="byte"
        )
    ]


def quote(text):
    out = []
    for byte in text.encode():
        if byte == 0x22:
            out.append('\\"')
        elif byte == 0x5C:
            out.append("\\\\")
        elif byte == 0x0A:
            out.append("\\n")
        elif byte == 0x09:
            out.append("\\t")
        elif 0x20 <= byte < 0x7F:
            out.append(chr(byte))
        else:
            out.append("\\x%02X" % byte)
    return '"' + "".join(out) + '"'


def case(label, normalizer, text):
    rows = align(normalizer, text)
    normalized = "".join(character for character, _ in rows)
    spans = " ".join(f"{start},{stop}" for _, (start, stop) in rows)
    print(f'  case "{label}" {quote(text)}')
    print(f"    {quote(normalized)}")
    print(f'    "{spans}";')


NFD, NFC = normalizers.NFD(), normalizers.NFC()
NFKD, NFKC = normalizers.NFKD(), normalizers.NFKC()
LOWERCASE, STRIP_ACCENTS = normalizers.Lowercase(), normalizers.StripAccents()
BERT = normalizers.BertNormalizer()
LLAMA = normalizers.Sequence(
    [normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")]
)

CASES = [
    ("nfd", NFD, ["café", "ẛ̣xy", "ḍ́", "한글",
                  "á̖"]),
    ("nfc", NFC, ["café", "ẹ́", "á̖",
                  "각Z", "ẹ̖́",
                  "ẛ̣xy"]),
    ("nfkc", NFKC, ["ﬁx ① Ａ", "½⁵"]),
    ("nfkd", NFKD, ["ﷺ!", "ﬁx ① Ǆ"]),
    ("lowercase", LOWERCASE, ["AİBß"]),
    ("strip accents", STRIP_ACCENTS, ["ááb́́c"]),
    ("strip", normalizers.Strip(), ["  a b  "]),
    ("strip left", normalizers.Strip(left=True, right=False), ["\t\n a b \n"]),
    ("strip right", normalizers.Strip(left=False, right=True), [" a b \n"]),
    ("replace string", normalizers.Replace(" ", "▁"), ["a  b"]),
    ("replace collapse", normalizers.Replace(Regex(r"\s+"), " "), ["a \t\n b"]),
    ("replace grow", normalizers.Replace("a", "xyz"), ["za z"]),
    ("replace shrink", normalizers.Replace(Regex(r"ab+"), "X"), ["zabbbz"]),
    ("replace delete", normalizers.Replace(Regex(r"\s+"), ""), ["a \t b"]),
    ("prepend", normalizers.Prepend("▁"), ["Hello", " x"]),
    ("prepend multi", normalizers.Prepend("<<"), ["Hello"]),
    ("bert", BERT, ["Ça 漢 áx", "a漢b", "a­b\tc",
                    "नमस्ते", "Café"]),
    ("llama", LLAMA, ["\n\nNot", "a  b"]),
    ("nfd strip lower", normalizers.Sequence([NFD, STRIP_ACCENTS, LOWERCASE]),
     ["Café"]),
]

for label, normalizer, texts in CASES:
    print(f"(* {label} *)")
    for text in texts:
        case(label, normalizer, text)
