# /// script
# requires-python = ">=3.15"
# dependencies = []
# ///
"""Generates talon's Unicode tables and their conformance golden.

    uv run packages/talon/test/gen/case.py [--check]

Every table reads one Unicode version, the pinned VERSION. lib/unicode_tables.ml
holds what Str.regex matches against: White_Space, which Str.trim also trims,
General_Category, Script and Script_Extensions with their property value
aliases, the binary properties of RL1.2 and the compatibility properties of
Annex C of UTS #18, and the orbits of simple case folding. golden/unicode.txt
states the membership of each class, printed from the parsed files directly: it
is the expected data of the conformance test.

Str.lower and Str.upper follow Unicode's Default Case Conversion, chapter 3,
section 3.13, at the pinned VERSION. Their scalar mappings are full case
mappings: SpecialCasing.txt's unconditional entries, or else UnicodeData.txt's
simple mapping (fields 12 and 13). The conditional entries of SpecialCasing are
the language-specific ones, which talon does not apply, and Final_Sigma, which
the kernel states, reading the Cased and Case_Ignorable properties of
DerivedCoreProperties.txt, from which it writes:

- lib/case_tables.ml: lower and upper mappings as two-stage tables, and the
  ranges of the two properties.
- test/golden/case.txt: the same mappings and ranges, and White_Space's, one
  fact a line, as data a test reads, printed from the parsed files directly.

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


def white_space() -> set[int]:
    """The scalars with PropList.txt's White_Space property."""
    members: set[int] = set()
    for fields in data_lines(fetch("PropList.txt")):
        if fields[1] == "White_Space":
            first, _, last = fields[0].partition("..")
            members.update(range(int(first, 16), int(last or first, 16) + 1))
    return members


def strip_name(name: str) -> str:
    """A property or value name loosely, by UAX44-LM3: case, spaces, `_`, `-`."""
    return "".join(c for c in name.lower() if c not in " _-")


def code_ranges(field: str) -> range:
    first, _, last = field.partition("..")
    return range(int(first, 16), int(last or first, 16) + 1)


ALL = range(0x110000)


def value_aliases(prop: str) -> dict[str, list[str]]:
    """Each value of [prop] by its short name: all names, stripped of case."""
    aliases = {}
    for fields in data_lines(fetch("PropertyValueAliases.txt")):
        if fields[0] == prop:
            names = [strip_name(f) for f in fields[1:] if f != "n/a"]
            aliases[fields[1]] = list(dict.fromkeys(names))
    return aliases


def general_categories() -> dict[str, set[int]]:
    """The members of every General_Category value and of the groups."""
    cats: dict[str, set[int]] = {}
    for fields in data_lines(fetch("extracted/DerivedGeneralCategory.txt")):
        cats.setdefault(fields[1], set()).update(code_ranges(fields[0]))
    groups = {
        "L": ["Lu", "Ll", "Lt", "Lm", "Lo"],
        "LC": ["Lu", "Ll", "Lt"],
        "M": ["Mn", "Mc", "Me"],
        "N": ["Nd", "Nl", "No"],
        "P": ["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"],
        "S": ["Sm", "Sc", "Sk", "So"],
        "Z": ["Zs", "Zl", "Zp"],
        "C": ["Cc", "Cf", "Cs", "Co", "Cn"],
    }
    for name, members in groups.items():
        cats[name] = set().union(*(cats[m] for m in members))
    return cats


def scripts() -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """The members of each Script value and of each Script_Extensions value,
    which holds a scalar with no ScriptExtensions.txt line under its Script,
    and a scalar with one under each listed Script."""
    long_to_short = {}
    for fields in data_lines(fetch("PropertyValueAliases.txt")):
        if fields[0] == "sc":
            for name in fields[1:]:
                long_to_short[name] = fields[1]
    sc: dict[str, set[int]] = {}
    for fields in data_lines(fetch("Scripts.txt")):
        sc.setdefault(long_to_short[fields[1]], set()).update(code_ranges(fields[0]))
    explicit: dict[int, list[str]] = {}
    for fields in data_lines(fetch("ScriptExtensions.txt")):
        values = [long_to_short[v] for v in fields[1].split()]
        for u in code_ranges(fields[0]):
            explicit[u] = values
    sc["Zzzz"] = set(ALL) - set().union(*sc.values())
    # Script values no scalar has, as Katakana_Or_Hiragana, match nothing.
    for name in long_to_short.values():
        sc.setdefault(name, set())
    scx = {name: {u for u in members if u not in explicit} for name, members in sc.items()}
    for u, values in explicit.items():
        for value in values:
            scx[value].add(u)
    return sc, scx


def binary_properties() -> dict[str, set[int]]:
    """The binary properties of RL1.2 and the sets Annex C names, each under
    every alias PropertyAliases.txt gives it."""
    derived: dict[str, set[int]] = {}
    for name in ("DerivedCoreProperties.txt", "PropList.txt"):
        for fields in data_lines(fetch(name)):
            if len(fields) == 2:
                derived.setdefault(fields[1], set()).update(code_ranges(fields[0]))
    cats = general_categories()
    alpha = derived["Alphabetic"]
    white = derived["White_Space"]
    digit = cats["Nd"]
    graph = (
        set(ALL)
        - white
        - cats["Cc"]
        - cats["Cs"]
        - cats["Cn"]
    )
    blank = cats["Zs"] | {0x09}
    word = alpha | cats["M"] | digit | cats["Pc"] | derived["Join_Control"]
    return {
        # RL1.2, with Unicode's short aliases.
        "alphabetic": alpha,
        "alpha": alpha,
        "uppercase": derived["Uppercase"],
        "upper": derived["Uppercase"],
        "lowercase": derived["Lowercase"],
        "lower": derived["Lowercase"],
        "whitespace": white,
        "wspace": white,
        "space": white,
        "noncharactercodepoint": derived["Noncharacter_Code_Point"],
        "nchar": derived["Noncharacter_Code_Point"],
        "defaultignorablecodepoint": derived["Default_Ignorable_Code_Point"],
        "di": derived["Default_Ignorable_Code_Point"],
        "any": set(ALL),
        "ascii": set(range(0x80)),
        "assigned": set(ALL) - cats["Cn"],
        # Annex C, its Standard Recommendation column.
        "punct": cats["P"],
        "digit": digit,
        "xdigit": digit | derived["Hex_Digit"],
        "alnum": alpha | digit,
        "blank": blank,
        "cntrl": cats["Cc"],
        "graph": graph,
        "print": graph | (blank - cats["Cc"]),
        "word": word,
    }


def case_orbits() -> list[list[int]]:
    """The orbits of simple case folding: each set of two or more scalars
    that fold (CaseFolding.txt, statuses C and S) to one scalar."""
    orbits: dict[int, set[int]] = {}
    for fields in data_lines(fetch("CaseFolding.txt")):
        if fields[1] in ("C", "S"):
            source, target = int(fields[0], 16), int(fields[2], 16)
            orbits.setdefault(target, {target}).add(source)
    return [sorted(o) for o in sorted(orbits.values(), key=min) if len(o) > 1]


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


def encoded(bounds: list[int]) -> bytes:
    """Ascending bounds as the LEB128 gaps between them."""
    out = bytearray()
    last = 0
    for b in bounds:
        gap = b - last
        last = b
        while gap >= 0x80:
            out.append(gap & 0x7F | 0x80)
            gap >>= 7
        out.append(gap)
    return bytes(out)


def ocaml_list(items: list[str]) -> str:
    return "[ " + "; ".join(items) + " ]"


def ocaml_string_list(names: list[str]) -> str:
    return ocaml_list([f'"{n}"' for n in names])


def ocaml_unicode(white, gc_aliases, gcs, sc_aliases, sc, scx, binary, orbits) -> str:
    tables: dict[bytes, str] = {}
    names = []
    def named(members: set[int]) -> str:
        data = encoded(ranges(members))
        if data not in tables:
            tables[data] = f"t{len(tables)}"
        return tables[data]

    white_data = encoded(ranges(white))
    tables[white_data] = "white_space"

    general_category = [
        f"({ocaml_string_list(gc_aliases[name])}, {named(gcs[name])})"
        for name in sorted(gc_aliases)
    ]
    script = [
        f"({ocaml_string_list(sc_aliases[name])}, {named(sc[name])}, {named(scx[name])})"
        for name in sorted(sc_aliases)
    ]
    by_members: dict[bytes, list[str]] = {}
    for name in sorted(binary):
        by_members.setdefault(encoded(ranges(binary[name])), []).append(name)
    binaries = [
        f"({ocaml_string_list(names)}, {named(binary[names[0]])})"
        for names in by_members.values()
    ]
    bindings = "".join(
        f"let {name} = {wrapped(data)}\n"
        for data, name in tables.items()
        if name != "white_space"
    )
    flat = [u for orbit in orbits for u in orbit + [-1]]
    # One entry per line would be 1200 lines: three to a line.
    def entries(xs):
        return "\n".join(
            "    " + " ".join(f"{x};" for x in xs[i : i + 3])
            for i in range(0, len(xs), 3)
        )
    return f"""(* Generated by packages/talon/test/gen/case.py from the Unicode {VERSION}
   UCD (PropList.txt, DerivedGeneralCategory.txt, Scripts.txt,
   ScriptExtensions.txt, PropertyValueAliases.txt, DerivedCoreProperties.txt,
   CaseFolding.txt); do not edit. Run
   [uv run packages/talon/test/gen/case.py] to regenerate it.

   The data is derived from Unicode's data files, under the Unicode License v3
   in packages/talon/LICENSE-unicode. *)

(* [bounds s] decodes a table of this file: its ranges as ascending bounds, each
   range [bounds.(2 * i)] to [bounds.(2 * i + 1) - 1], stored as the LEB128 gaps
   between them. *)
let bounds s =
  let n = ref 0 in
  String.iter (fun c -> if Char.code c < 0x80 then incr n) s;
  let b = Array.make !n 0 and last = ref 0 and at = ref 0 in
  let value = ref 0 and shift = ref 0 in
  let add c =
    let c = Char.code c in
    value := !value lor ((c land 0x7f) lsl !shift);
    if c < 0x80 then begin
      last := !last + !value;
      b.(!at) <- !last;
      incr at;
      value := 0;
      shift := 0
    end
    else shift := !shift + 7
  in
  String.iter add s;
  b

(* The scalars with Unicode's White_Space property. *)
let white_space = {wrapped(white_data)}

(* The other classes, read through the lookups below. *)
{bindings}
(* Every General_Category value and group, under its names by UAX44-LM3. *)
let general_category =
  {ocaml_list(general_category)}

(* Every Script value, under its names by UAX44-LM3: its members, then those of
   Script_Extensions at that value. *)
let script =
  {ocaml_list(script)}

(* The binary properties of RL1.2 and the classes of Annex C (UTS #18) under
   their names, each case-folded like UAX44-LM3. *)
let binary =
  {ocaml_list(binaries)}

(* The orbits of simple case folding, each two or more scalars that fold to one
   value (CaseFolding.txt, statuses C and S), each orbit ended by [-1]. *)
let case_orbits =
  [|
{entries(flat)}
  |]
"""


def golden(lower, upper, cased, ignorable, white) -> str:
    lines = [
        f"# Unicode {VERSION}: `S scalar;lower;upper` for each scalar whose full",
        "# mapping is not itself, `C first..last` for Cased, `I first..last` for",
        "# Case_Ignorable, `W first..last` for White_Space. From UnicodeData.txt,",
        "# SpecialCasing.txt, DerivedCoreProperties.txt and PropList.txt, under",
        "# the Unicode License v3.",
    ]
    hex_ = lambda t: " ".join(f"{u:04X}" for u in t)
    for scalar in sorted(set(lower) | set(upper)):
        low, up = lower.get(scalar, (scalar,)), upper.get(scalar, (scalar,))
        if low != (scalar,) or up != (scalar,):
            lines.append(f"S {scalar:04X};{hex_(low)};{hex_(up)}")
    for name, members in (("C", cased), ("I", ignorable), ("W", white)):
        bounds = ranges(members)
        for start, end in zip(bounds[0::2], bounds[1::2]):
            lines.append(f"{name} {start:04X}..{end - 1:04X}")
    return "\n".join(lines) + "\n"


def unicode_golden(gc_aliases, gcs, sc_aliases, sc, scx, binary, orbits) -> str:
    def spans(members):
        bounds = ranges(members)
        return " ".join(
            f"{start:04X}..{end - 1:04X}"
            for start, end in zip(bounds[0::2], bounds[1::2])
        )

    lines = [
        f"# Unicode {VERSION}: `C name ranges` gives the members of a class, for",
        "# gc=Value of General_Category, sc=Value of Script, scx=Value of",
        "# Script_Extensions and the name of a binary or Annex C class; `A",
        "# class name` an alias the class also answers to; `F a b ...` an orbit",
        "# of simple case folding. Printed from the UCD files, under the",
        "# Unicode License v3.",
    ]
    for name in sorted(gc_aliases):
        lines.append(f"C gc={name} {spans(gcs[name])}")
        lines += [f"A gc={name} {alias}" for alias in gc_aliases[name]]
    for name in sorted(sc_aliases):
        lines.append(f"C sc={name} {spans(sc[name])}")
        lines.append(f"C scx={name} {spans(scx[name])}")
        lines += [f"A sc={name} {alias}" for alias in sc_aliases[name]]
    for name in sorted(binary):
        lines.append(f"C {name} {spans(binary[name])}")
    lines += ["F " + " ".join(f"{u:04X}" for u in orbit) for orbit in orbits]
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
    white = white_space()
    gc_aliases = value_aliases("gc")
    gcs = general_categories()
    sc_aliases = value_aliases("sc")
    sc, scx = scripts()
    binary = binary_properties()
    orbits = case_orbits()
    unicode_tables = talon / "lib" / "unicode_tables.ml"
    unicode_tables.write_text(
        ocaml_unicode(white, gc_aliases, gcs, sc_aliases, sc, scx, binary, orbits),
        encoding="utf-8",
        newline="\n",
    )
    case = talon / "test" / "golden" / "case.txt"
    case.parent.mkdir(parents=True, exist_ok=True)
    case.write_text(
        golden(lower, upper, cased, ignorable, white),
        encoding="utf-8",
        newline="\n",
    )
    unicode_data = talon / "test" / "golden" / "unicode.txt"
    unicode_data.write_text(
        unicode_golden(gc_aliases, gcs, sc_aliases, sc, scx, binary, orbits),
        encoding="utf-8",
        newline="\n",
    )
    return [
        tables.relative_to(talon),
        unicode_tables.relative_to(talon),
        case.relative_to(talon),
        unicode_data.relative_to(talon),
    ]


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
