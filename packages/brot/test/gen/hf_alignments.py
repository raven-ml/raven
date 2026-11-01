# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.1"]
# ///
"""Print the alignment HuggingFace `tokenizers` gives each normalizer.

The expectations of the "alignment" group in test/test_unicode.ml come from
this script:

    uv run packages/brot/test/gen/hf_alignments.py [--check]

`NormalizedString` keeps, for every byte of the normalized text, the byte range
of the original it stands for, and reports a span of normalized bytes as the
range of its first byte joined with the range of its last. The Python bindings
do not expose that vector, but `PreTokenizedString.get_splits` runs a split
through the same conversion, so splitting the normalized text into single
characters exposes the alignment of every one of them.

Each case prints the input, the normalized text and one span per normalized
character, escaped so that the lines can be pasted into the OCaml test. With
`--check` it prints nothing and compares the cases with the test's `aligned`
calls instead, failing on each one that differs or is missing.
"""

import base64
import json
import sys
from pathlib import Path

from tokenizers import PreTokenizedString, Regex, normalizers

import ocaml_cases

TEST = Path(__file__).resolve().parents[1] / "test_unicode.ml"

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
    return [
        f'  case "{label}" {quote(text)}',
        f"    {quote(normalized)}",
        f'    "{spans}";',
    ]


NFD, NFC = normalizers.NFD(), normalizers.NFC()
NFKD, NFKC = normalizers.NFKD(), normalizers.NFKC()
LOWERCASE, STRIP_ACCENTS = normalizers.Lowercase(), normalizers.StripAccents()
BERT = normalizers.BertNormalizer()
STRIP, PREPEND = normalizers.Strip(), normalizers.Prepend("▁")
REPLACE = normalizers.Replace(" ", "▁")
LLAMA = normalizers.Sequence(
    [normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")]
)

# The charsmap gen_precompiled.py writes.
CHARSMAP = json.loads(
    (Path(__file__).resolve().parents[1] / "support" / "precompiled"
     / "charsmap.json").read_text(encoding="utf-8")
)["normalizer"]["precompiled_charsmap"]
PRECOMPILED = normalizers.Precompiled(base64.b64decode(CHARSMAP))

CASES = [
    ("nfd", NFD, ["café", "ẛ̣xy", "ḍ́", "한글",
                  "á̖"]),
    ("nfc", NFC, ["café", "ẹ́", "á̖",
                  "각Z", "ẹ̖́",
                  "ẛ̣xy", "\uac00\u11c3"]),
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
    ("replace empty", normalizers.Replace(Regex(r"x*"), "_"), ["aé b", "xa"]),
    ("replace empty string", normalizers.Replace("", "_"), ["aé"]),
    ("replace class", normalizers.Replace(Regex(r"\p{L}+"), "L"), ["ab1é日"]),
    ("replace anchor", normalizers.Replace(Regex(r"$"), "!"), ["a\nb"]),
    ("prepend", normalizers.Prepend("▁"), ["Hello", " x"]),
    ("prepend multi", normalizers.Prepend("<<"), ["Hello"]),
    ("bert", BERT, ["Ça 漢 áx", "a漢b", "a­b\tc",
                    "नमस्ते", "Café",
                    "\u1e9b\u0323 x", "\uac01Z", "\u00a0\u0130\u2028",
                    # A dropped control does not separate the marks around it.
                    "a\u0301\x01\U0001D165",
                    "a\U0001D16D\x7f\u0301\U0001D165",
                    "a\u0301\x00\U0001D165b"]),
    ("bert nolower", normalizers.BertNormalizer(lowercase=False),
     ["Ça 漢 áx"]),
    ("bert noclean", normalizers.BertNormalizer(clean_text=False),
     ["a\x01É\tb"]),
    ("bert nostrip", normalizers.BertNormalizer(strip_accents=False), ["Éa"]),
    ("llama", LLAMA, ["\n\nNot", "a  b"]),
    ("nfd strip lower", normalizers.Sequence([NFD, STRIP_ACCENTS, LOWERCASE]),
     ["Café"]),
    ("strip prepend", normalizers.Sequence([STRIP, PREPEND]), ["  ab ", " x"]),
    ("prepend strip",
     normalizers.Sequence([normalizers.Prepend("<<"), STRIP]), [" a ", "a "]),
    ("prepend strip left",
     normalizers.Sequence([normalizers.Prepend("  x"),
                           normalizers.Strip(left=True, right=False)]),
     [" a "]),
    ("prepend prepend",
     normalizers.Sequence([PREPEND, normalizers.Prepend("x")]), ["ab"]),
    ("strip strip",
     normalizers.Sequence([normalizers.Strip(left=True, right=False),
                           normalizers.Strip(left=False, right=True)]),
     ["  a b  "]),
    ("prepend strip replace",
     normalizers.Sequence([PREPEND, STRIP, REPLACE]), ["  a b "]),
    ("strip prepend replace",
     normalizers.Sequence([STRIP, PREPEND, REPLACE]), ["  a b "]),
    ("strip lowercase", normalizers.Sequence([STRIP, LOWERCASE]), [" AÉ "]),
    ("prepend nfd", normalizers.Sequence([PREPEND, NFD]), ["é"]),
    ("byte level", normalizers.ByteLevel(), ["aé", " x"]),
    ("nmt", normalizers.Nmt(), ["a\x01b\tc\u200bd"]),
    ("clip",
     normalizers.Sequence([NFC, normalizers.Replace(Regex(r"\s+"), " "),
                           LOWERCASE]),
     ["Á  B\tc"]),
    # Full-width forms, ligatures, controls; a cluster under 6 bytes takes the
    # shortest source it starts with and drops the rest; longer clusters go
    # character by character; CR LF is one cluster.
    ("precompiled", PRECOMPILED,
     ["", "abc", "\uff21\uff45\uff11\uff01", "\ufb01x \ufb03 \u01c4",
      "\u2122 \u00bd \u2026", "\x01ab", "a\x01b", "ab\x01", "a\x01\x1f",
      "\u2122\x01g", "e\u0301", "e\u0301\u0301", "e\u0302\u0300",
      "\uff45\u0301", "\uff45\u0301\u0301", "\uff76\uff9e", "C\u0327a",
      "\u1100\u1161", "a\r\nb", "\r\r\n", "a\tb\n", "x\u3000\u200by",
      "\ufffd", "\U0001F600\u2122",
      # Clusters whose rules look past an extend: a spacing mark (GB9a), a
      # prepend (GB9b), a regional indicator pair (GB12); U+05C8 is
      # unassigned in Unicode 17, so a cluster of its own.
      "\u00bd\u0903", "\u0600\u00bd", "\U0001F1EB\U0001F1F7\u00bd",
      "\u00bd\u05c8"]),
]

# `let case = aligned`, which binds `case` to a normalizer and a label.
ALIGNED = [("word", w) for w in ("let", "case", "=", "aligned")]


def recorded(source: str) -> list[ocaml_cases.Case]:
    """The (label, input), (normalized, spans) of every alignment the test
    states: `aligned normalizer label input normalized spans`, or `case input
    normalized spans` after `let case = aligned normalizer label in`."""
    out = []
    for body in ocaml_cases.definitions(source).values():
        label = None
        for bindings, statement in ocaml_cases.statements(body):
            if ("word", "let") in bindings:
                label = None
                if bindings[:4] == ALIGNED:
                    label = [bindings[-2]]
            words = ocaml_cases.arguments(statement)
            if not words:
                continue
            if words[0] == [("word", "aligned")] and len(words) >= 6:
                words = [words[0], *words[-4:]]
            elif words[0] == [("word", "case")] and label and len(words) == 4:
                words = [words[0], label, *words[1:]]
            else:
                continue
            if all(len(w) == 1 and w[0][0] == "str" for w in words[1:]):
                out.append(ocaml_cases.case(words, 2))
    return out


def check(lines: list[str]) -> int:
    generated = ocaml_cases.cases(ocaml_cases.tokens("\n".join(lines)), "case", 2)
    found = recorded(ocaml_cases.read(TEST))
    problems = []
    for label, _, _ in CASES:
        key = ((("str", label.encode()),),)
        problems += ocaml_cases.differences(
            label,
            [c for c in generated if c[0][:1] == key],
            [c for c in found if c[0][:1] == key],
            partial=False,
        )
    return ocaml_cases.report(problems)


def main() -> int:
    lines = []
    for label, normalizer, texts in CASES:
        lines.append(f"(* {label} *)")
        for text in texts:
            lines.extend(case(label, normalizer, text))
    if "--check" in sys.argv[1:]:
        return check(lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
