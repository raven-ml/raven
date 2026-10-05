# /// script
# requires-python = ">=3.11"
# ///
"""Transcribe the CODATA releases' measured constants from NIST's tables.

    uv run contrib/ymir/test/units/gen/codata.py [--check]

NIST publishes each CODATA adjustment as a plain-text table, one constant
per row: its name, value, standard uncertainty and unit, digits grouped by
spaces, and "(exact)" for an exact uncertainty. The sources:

    2018  https://physics.nist.gov/cuu/Constants/ArchiveASCII/allascii_2018.txt
    2022  https://physics.nist.gov/cuu/Constants/Table/allascii.txt

The second URL serves the latest adjustment, so the script checks that each
table's header names the release it is read for, and records each source's
URL and the SHA-256 of the text it read above that release's rows: a table
NIST changes in place then fails --check, and the checked-in rows name the
exact text they came from.

The script writes lib/units/codata_tables.ml: for each release, every
measured row of its table, in the table's order, as (name, text, unit), with
the name as NIST writes it, the value and uncertainty in Constant.v's text,
the uncertainty's digits in parentheses counting units of the value's last
digit ("6.67430(15)e-11"), and the unit as an expression over Unit's names.
Each text is read back and checked against the table's value and
uncertainty as exact fractions. Every line of the table must read as a row,
and a comment above each release tallies its rows: exact, restating another,
and kept.

Left out are the exact rows, whose constants are units, and the rows that
restate another row in other units: energy equivalents and relationships,
and values given again in MeV, eV, u, Hz, K or m^-1.

NIST writes the radian as 1. A unit gets the radian where its quantity is
an angle per something or something per angle by definition: a gyromagnetic
ratio is an angular frequency per tesla, rad s^-1 T^-1, and a reduced
Compton wavelength is a wavelength per radian, m rad^-1. Such a row carries
NIST's unit in a comment.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import re
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "lib" / "units" / "codata_tables.ml"

SOURCES = {
    2018: "https://physics.nist.gov/cuu/Constants/ArchiveASCII/allascii_2018.txt",
    2022: "https://physics.nist.gov/cuu/Constants/Table/allascii.txt",
}

# The rows Codata's accessors name, which every release must hold.
NAMED = [
    "Newtonian constant of gravitation",
    "fine-structure constant",
    "vacuum mag. permeability",
    "vacuum electric permittivity",
    "atomic mass constant",
    "electron mass",
    "muon mass",
    "tau mass",
    "proton mass",
    "neutron mass",
    "deuteron mass",
    "triton mass",
    "helion mass",
    "alpha particle mass",
    "Rydberg constant",
    "Bohr radius",
    "classical electron radius",
    "Compton wavelength",
    "Thomson cross section",
    "Hartree energy",
    "Bohr magneton",
    "nuclear magneton",
    "electron mag. mom.",
    "proton mag. mom.",
    "electron g factor",
]

# Rows that restate another row in other units.
RESTATED = re.compile(
    r"relationship|energy equivalent|times c in Hz|times hc in|over h-bar c"
    r"| in (MeV|eV|u|Hz|MHz|K|inverse meter)\b")


def radian(name):
    """The exponent of the radian NIST's unit leaves out of the row's."""
    if "gyromag. ratio" in name:
        return 1
    if name.startswith("reduced ") and name.endswith("Compton wavelength"):
        return -1
    return 0


# NIST's unit symbols and Unit's names for them.
UNITS = {
    "m": "metre",
    "kg": "kilogram",
    "s": "second",
    "A": "ampere",
    "K": "kelvin",
    "mol": "mole",
    "N": "newton",
    "J": "joule",
    "C": "coulomb",
    "F": "farad",
    "T": "tesla",
    "V": "volt",
    "ohm": "ohm",
    "GeV": "(giga electronvolt)",
}

# A number is digits in groups split by a space or the point, as in
# "7294.299 541 71" or "6.674 30 e-11", with "..." after a truncated exact one.
NUMBER = r"-?[0-9]+(?:[ .][0-9]+)*(?:\.\.\.)?(?: e-?[0-9]+)?"
ROW = re.compile(rf"^(?P<name>\S.*?)\s{{2,}}(?P<value>{NUMBER})"
                 rf"\s{{2,}}(?P<unc>\(exact\)|{NUMBER})\s*(?P<unit>.*)$")


def fetch(year):
    # NIST refuses Python's default user agent.
    req = urllib.request.Request(SOURCES[year], headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req) as r:
        raw = r.read()
    text = raw.decode("ascii")
    if f"{year} CODATA adjustment" not in text:
        sys.exit(f"{SOURCES[year]} is not the CODATA {year} table")
    # Every non-blank line after the header's rule is a row, and each must
    # read as one, so none is dropped unseen.
    lines = text.split("-" * 20)[-1].lstrip("-").splitlines()
    rows = {}
    for line in filter(str.strip, lines):
        m = ROW.match(line)
        if not m:
            sys.exit(f"CODATA {year}: unread row {line!r}")
        rows[m["name"]] = (m["value"], m["unc"], m["unit"].strip())
    if len(rows) != len(list(filter(str.strip, lines))):
        sys.exit(f"CODATA {year}: a name is given twice")
    return rows, hashlib.sha256(raw).hexdigest()


def split(number):
    """The digits and exponent of NIST's number, spaces removed."""
    mantissa, _, exp = number.replace(" ", "").partition("e")
    return mantissa, exp


def decimals(mantissa):
    return len(mantissa.partition(".")[2])


def fraction(mantissa, exp):
    return Fraction(mantissa) * Fraction(10) ** int(exp or "0")


def text(name, value, unc):
    """Constant.v's text of the row, checked against the row exactly."""
    if unc == "(exact)" or "..." in value:
        sys.exit(f"{name} is exact; it is a unit")
    vm, ve = split(value)
    um, ue = split(unc)
    if ue != ve or decimals(um) != decimals(vm):
        sys.exit(f"{name}: the uncertainty {unc} is not aligned with {value}")
    digits = um.replace(".", "").lstrip("0")
    out = f"{vm}({digits})" + (f"e{ve}" if ve else "")
    # Read the text back as Constant.v reads it.
    m = re.fullmatch(r"(-?)([0-9]+)(?:\.([0-9]+))?\(([0-9]+)\)(?:e(-?[0-9]+))?", out)
    sign, whole, frac, u, e = m.groups()
    scale = Fraction(10) ** (int(e or "0") - len(frac or ""))
    back_value = (-1 if sign else 1) * int(whole + (frac or "")) * scale
    back_unc = int(u) * scale
    if back_value != fraction(vm, ve) or back_unc != fraction(um, ue):
        sys.exit(f"{name}: {out} does not read back as {value} +- {unc}")
    return out


def unit(name, symbols):
    """The OCaml expression of the row's unit."""
    powers = []
    rad = radian(name)
    if rad:
        powers.append("radian" if rad == 1 else f"(radian ** {rad})")
    for item in symbols.split():
        base, _, exp = item.partition("^")
        if base not in UNITS:
            sys.exit(f"{name}: no Unit name for {base!r} in {symbols!r}")
        powers.append(UNITS[base] if exp == "" else f"({UNITS[base]} ** {exp})")
    if not powers:
        return "Unit.one"
    if len(powers) == 1:
        p = powers[0]
        return f"Unit.({p[1:-1]})" if p.startswith("(") else f"Unit.{p}"
    return f"Unit.({' * '.join(powers)})"


def release(year):
    rows, sha256 = fetch(year)
    for name in NAMED:
        if name not in rows:
            sys.exit(f"CODATA {year} has no {name}")
    exact = [n for n, (_, unc, _) in rows.items() if unc == "(exact)"]
    restated = [n for n in rows if n not in exact and RESTATED.search(n)]
    kept = len(rows) - len(exact) - len(restated)
    out = [
        f"(* CODATA {year}: {len(rows)} rows, {len(exact)} exact, "
        f"{len(restated)} restating another, {kept} kept.",
        f"   Source: {SOURCES[year]}",
        f"   SHA-256: {sha256} *)",
        f"let v{year} =",
        "  [",
    ]
    for name, (value, unc, symbols) in rows.items():
        if name in exact or name in restated:
            continue
        row = (f'"{name}"', f'"{text(name, value, unc)}"', unit(name, symbols))
        if radian(name):
            out.append(f"    (* NIST: {symbols} *)")
        # A row on one line when it fits in 80 columns.
        line = f"    ({', '.join(row)});"
        if len(line) <= 80:
            out.append(line)
        else:
            out += [f"    ( {row[0]},", f"      {row[1]},", f"      {row[2]} );"]
    out.append("  ]")
    return out


def render():
    header = [
        "(*---------------------------------------------------------------------------",
        "  Copyright (c) 2026 The Raven authors. All rights reserved.",
        "  SPDX-License-Identifier: ISC",
        "  ---------------------------------------------------------------------------*)",
        "",
        "(* Written by test/units/gen/codata.py from NIST's tables of the CODATA",
        "   adjustments; do not edit. Each row is a constant's name, value with",
        "   uncertainty, and unit. A comment gives NIST's unit where the row's",
        "   unit holds the radian, which NIST writes as 1. Each release names its",
        "   source and the SHA-256 of the text read from it. *)",
        "",
        '[@@@ocamlformat "disable"]',
        "",
    ]
    body = []
    for year in SOURCES:
        body += release(year) + [""]
    return "\n".join(header + body[:-1]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} would change; run without --check")
        return
    OUT.write_text(text)


if __name__ == "__main__":
    main()
