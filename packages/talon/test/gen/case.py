# /// script
# requires-python = ">=3.15"
# dependencies = []
# ///
"""Generates talon's case-mapping tables and their conformance golden.

    uv run packages/talon/test/gen/case.py [--check]

Str.lower and Str.upper follow Unicode's Default Case Conversion, chapter 3,
section 3.13, at the pinned VERSION. Their scalar mappings are full case
mappings: SpecialCasing.txt's unconditional entries, or else UnicodeData.txt's
simple mapping (fields 12 and 13). The conditional entries of SpecialCasing are
the language-specific ones, which talon does not apply, and Final_Sigma, which
the kernel states, reading the Cased and Case_Ignorable properties of
DerivedCoreProperties.txt, from which it writes:

- lib/case_tables.ml: lower and upper mappings as two-stage tables, and the
  ranges of the two properties.
- test/golden/case.txt: the same mappings and ranges, one fact a line, as data
  a test reads, printed from the parsed files directly.

Each file is derived from Unicode's data files, under the Unicode License v3
in packages/talon/LICENSE-unicode, and each is rewritten; --check rewrites
nothing and fails if one differs. The Unicode Character Database publishes no
case-mapping conformance file, so golden/case.txt plays that part: expected
values read from the files themselves.

Before writing, the mappings are checked against Python's str.lower and
str.upper, an implementation with its own tables, wherever Python's Unicode
version knows the scalar (`unicodedata.unidata_version`, older than VERSION).
Scalars whose mapping changed since Python's version are printed.
"""

from __future__ import annotations

import filecmp
import sys
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

VERSION = "18.0.0"
UCD = "https://www.unicode.org/Public/{}/ucd/{}"

# Each table maps a scalar through a block of 2^BLOCK_BITS bytes, chosen as the
# smallest total of index and blocks, 5.6 KB for lower and 6.0 KB for upper.
BLOCK_BITS = 6
SIGMA = 0x03A3


def fetch(name: str) -> str:
    with urllib.request.urlopen(UCD.format(VERSION, name)) as response:
        return response.read().decode("utf-8")


def data_lines(text: str):
    """The semicolon-separated fields of each data line."""
    for line in text.splitlines():
        data = line.split("#", 1)[0].strip()
        if data:
            yield [f.strip() for f in data.split(";")]


def scalars(field: str) -> tuple[int, ...]:
    return tuple(int(x, 16) for x in field.split())


def mappings() -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    """The full lower and upper mapping of each scalar that has one."""
    lower: dict[int, tuple[int, ...]] = {}
    upper: dict[int, tuple[int, ...]] = {}
    for line in fetch("UnicodeData.txt").splitlines():
        fields = line.split(";")
        scalar = int(fields[0], 16)
        if fields[12]:
            upper[scalar] = scalars(fields[12])
        if fields[13]:
            lower[scalar] = scalars(fields[13])
    # Fields of SpecialCasing: code; lower; title; upper; (condition;)
    for fields in data_lines(fetch("SpecialCasing.txt")):
        condition = fields[4] if len(fields) > 4 else ""
        if condition:
            continue
        scalar = int(fields[0], 16)
        lower[scalar] = scalars(fields[1])
        upper[scalar] = scalars(fields[3])
    return lower, upper


def properties() -> tuple[set[int], set[int]]:
    """The scalars that are Cased and that are Case_Ignorable."""
    cased: set[int] = set()
    ignorable: set[int] = set()
    for fields in data_lines(fetch("DerivedCoreProperties.txt")):
        if fields[1] == "Cased":
            into = cased
        elif fields[1] == "Case_Ignorable":
            into = ignorable
        else:
            continue
        first, _, last = fields[0].partition("..")
        into.update(range(int(first, 16), int(last or first, 16) + 1))
    return cased, ignorable


def utf_8(mapped: tuple[int, ...]) -> bytes:
    return "".join(chr(u) for u in mapped).encode("utf-8")


# Tables


def entry(scalar: int, mapped: tuple[int, ...] | None):
    """What a table holds for a scalar: its delta, or the bytes of a longer map."""
    if mapped is None or mapped == (scalar,):
        return ("delta", 0)
    if len(mapped) == 1:
        return ("delta", mapped[0] - scalar)
    return ("full", utf_8(mapped))


def table(mapping: dict[int, tuple[int, ...]]) -> dict:
    """The two-stage table: scalars below limit, index and blocks, values."""
    limit = max(mapping) + 1
    values = [entry(u, mapping.get(u)) for u in range(limit)]
    deltas = sorted({v[1] for v in values if v[0] == "delta"} - {0})
    full = sorted({v[1] for v in values if v[0] == "full"})
    numbers: dict[object, int] = {("delta", 0): 0}
    numbers.update({("delta", d): i + 1 for i, d in enumerate(deltas)})
    first_full = len(deltas) + 1
    numbers.update({("full", s): first_full + i for i, s in enumerate(full)})
    assert len(numbers) <= 256
    encoded = bytes(numbers[v] for v in values)
    size = 1 << BLOCK_BITS
    index = bytearray()
    blocks: dict[bytes, int] = {}
    for start in range(0, limit, size):
        block = encoded[start : start + size].ljust(size, b"\0")
        number = blocks.setdefault(block, len(blocks))
        assert number < 256
        index.append(number)
    return {
        "limit": limit,
        "index": bytes(index),
        "blocks": b"".join(blocks),
        "deltas": [0] + deltas,
        "full": full,
    }


def ranges(members: set[int]) -> list[int]:
    """The members as ascending bounds: each start, then the end after it."""
    bounds = []
    for u in sorted(members):
        if bounds and bounds[-1] == u:
            bounds[-1] = u + 1
        else:
            bounds += [u, u + 1]
    return bounds


# Writing


def literal(data: bytes) -> str:
    return "".join(
        # A space would be lost after a line continuation, which skips blanks.
        chr(b) if 0x20 < b < 0x7F and chr(b) not in '"\\' else f"\\x{b:02x}"
        for b in data
    )


def wrapped(data: bytes, width: int = 76) -> str:
    """An OCaml string literal of data, cut at scalar escapes."""
    out, line = [], ""
    for b in data:
        c = literal(bytes([b]))
        if len(line) + len(c) > width:
            out.append(line)
            line = ""
        line += c
    out.append(line)
    return '"' + "\\\n     ".join(out) + '"'


def ints(values: list[int]) -> str:
    return "\n".join(
        "    " + " ".join(f"{v};" for v in values[i : i + 12])
        for i in range(0, len(values), 12)
    )


def ocaml_table(name: str, t: dict) -> str:
    full = "".join(f'      "{literal(s)}";\n' for s in t["full"])
    return f"""let {name} =
  {{
    limit = 0x{t['limit']:X};
    index = {wrapped(t['index'])};
    blocks = {wrapped(t['blocks'])};
    deltas = [|
{ints(t['deltas'])}
    |];
    full = [|
{full}    |];
  }}
"""


def ocaml(lower, upper, cased, ignorable) -> str:
    return f"""(* Generated by packages/talon/test/gen/case.py from the Unicode {VERSION}
   UCD (UnicodeData.txt, SpecialCasing.txt, DerivedCoreProperties.txt); do not
   edit. Run [uv run packages/talon/test/gen/case.py] to regenerate it.

   The data is derived from Unicode's data files, under the Unicode License v3
   in packages/talon/LICENSE-unicode. *)

type table = {{
  limit : int;
  index : string;
  blocks : string;
  deltas : int array;
  full : string array;
}}

let unicode_version = "{VERSION}"
let block_bits = {BLOCK_BITS}

{ocaml_table("lower", table(lower))}
{ocaml_table("upper", table(upper))}
let cased =
  [|
{ints(ranges(cased))}
  |]

let case_ignorable =
  [|
{ints(ranges(ignorable))}
  |]
"""


def golden(lower, upper, cased, ignorable) -> str:
    lines = [
        f"# Unicode {VERSION}: `S scalar;lower;upper` for each scalar whose full",
        "# mapping is not itself, `C first..last` for Cased, `I first..last` for",
        "# Case_Ignorable. From UnicodeData.txt, SpecialCasing.txt and",
        "# DerivedCoreProperties.txt, under the Unicode License v3.",
    ]
    hex_ = lambda t: " ".join(f"{u:04X}" for u in t)
    for scalar in sorted(set(lower) | set(upper)):
        low, up = lower.get(scalar, (scalar,)), upper.get(scalar, (scalar,))
        if low != (scalar,) or up != (scalar,):
            lines.append(f"S {scalar:04X};{hex_(low)};{hex_(up)}")
    for name, members in (("C", cased), ("I", ignorable)):
        bounds = ranges(members)
        for start, end in zip(bounds[0::2], bounds[1::2]):
            lines.append(f"{name} {start:04X}..{end - 1:04X}")
    return "\n".join(lines) + "\n"


# Cross-check against Python


def lower_text(s: str, lower, cased, ignorable) -> str:
    """toLowercase (R2) with Final_Sigma (Table 3-17), from the parsed files."""
    out = []
    for i, c in enumerate(s):
        u = ord(c)
        if u == SIGMA and final_sigma(s, i, cased, ignorable):
            out.append(chr(0x03C2))
        else:
            out.append("".join(chr(x) for x in lower.get(u, (u,))))
    return "".join(out)


def final_sigma(s, i, cased, ignorable) -> bool:
    """The possessive reading the standard states after Table 3-17."""
    j = i - 1
    while j >= 0 and ord(s[j]) in ignorable:
        j -= 1
    before = j >= 0 and ord(s[j]) in cased
    j = i + 1
    while j < len(s) and ord(s[j]) in ignorable:
        j += 1
    after = j == len(s) or ord(s[j]) not in cased
    return before and after


def cross_check(lower, upper, cased, ignorable) -> None:
    known = [
        u
        for u in range(0x110000)
        if not 0xD800 <= u < 0xE000 and unicodedata.category(chr(u)) != "Cn"
    ]
    for name, method, mapping in (
        ("lower", str.lower, lower),
        ("upper", str.upper, upper),
    ):
        differs = [
            u
            for u in known
            # A lone capital sigma lowers to sigma; Python agrees.
            if method(chr(u))
            != "".join(chr(x) for x in mapping.get(u, (u,)))
        ]
        print(f"{name}: {len(differs)} scalars differ from Python's "
              f"Unicode {unicodedata.unidata_version}: "
              + " ".join(f"U+{u:04X}" for u in differs[:40]))
    # Sigma in context, over an alphabet old enough for Python's data to hold.
    alphabet = ["Α", "Σ", "σ", "a", "A", " ", "1", "'", "ͅ",
                "́", "­", "ʰ", "."]
    disagreements = []
    for a in alphabet:
        for b in alphabet:
            for c in alphabet:
                for d in alphabet:
                    s = a + b + "Σ" + c + d
                    if s.lower() != lower_text(s, lower, cased, ignorable):
                        disagreements.append(s)
    assert not disagreements, disagreements[:20]


def write(talon: Path) -> list[Path]:
    lower, upper = mappings()
    cased, ignorable = properties()
    cross_check(lower, upper, cased, ignorable)
    tables = talon / "lib" / "case_tables.ml"
    tables.parent.mkdir(parents=True, exist_ok=True)
    tables.write_text(
        ocaml(lower, upper, cased, ignorable), encoding="utf-8", newline="\n"
    )
    case = talon / "test" / "golden" / "case.txt"
    case.parent.mkdir(parents=True, exist_ok=True)
    case.write_text(
        golden(lower, upper, cased, ignorable), encoding="utf-8", newline="\n"
    )
    return [tables.relative_to(talon), case.relative_to(talon)]


def main() -> int:
    talon = Path(__file__).resolve().parents[2]
    if "--check" not in sys.argv[1:]:
        for path in write(talon):
            print(f"wrote {path}")
        return 0
    with tempfile.TemporaryDirectory() as scratch:
        stale = [
            path
            for path in write(Path(scratch))
            if not filecmp.cmp(Path(scratch) / path, talon / path, shallow=False)
        ]
    for path in stale:
        print(f"{path} differs from what the generator writes", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
