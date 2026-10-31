# /// script
# requires-python = ">=3.11"
# dependencies = ["pyerfa==2.0.1.5"]
# ///
"""Generate the time scales' tables and the times' test references.

    uv run contrib/ymir/test/gen/times.py [--check | --refresh]

The sources, each pinned by its SHA-256 below:

    iers/UTC-TAI.history, a copy of
    https://hpiers.obspm.fr/iers/bul/bulc/UTC-TAI.history
        The IERS's history of TAI - UTC: from 1961 to 1972 a linear
        expression in the UTC date per interval, from 1972 whole seconds.
    iers/Leap_Second.dat, a copy of
    https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat
        IERS Bulletin C's table of TAI - UTC since 1972 and its expiry.
    https://raw.githubusercontent.com/liberfa/erfa/v2.0.1/src/dtdb.c
        ERFA's eraDtdb, the IAU SOFA library's iauDtdb under ERFA's BSD
        license: the Fairhead & Bretagnon (1990) series for TDB - TT.

The IERS files are read from the copies beside this script, so a run
gives the same output until they change; dtdb.c is read from its tagged
URL. A source whose SHA-256 is not its pin fails the run with "source
changed: update pins and regenerate". --refresh fetches the IERS files
into the copies and fails the same way if either changed: Bulletin C is
reissued every six months, if only for its expiry.

The rows of the history from 1972 must equal Bulletin C's rows, date and
value.

It writes:

lib/time_tables.ml
    - TAI - UTC per interval: its first day, offset, reference day and
      rate, as the history writes them;
    - the day Bulletin C says its table expires;
    - eraDtdb's 787 terms, grouped by power of T, and its four
      adjustments to JPL's planetary masses, as ERFA writes them, with
      ERFA's copyright notice and license, which its terms ask for.

test/support/times_reference.ml
    - UTC instants with ERFA's TAI, TT and TDB of each (eraUtctai,
      eraTaitt, eraDtdb and eraTttdb), drawn on every leap-second day,
      each day that ends a 1961-1972 interval, and at random from 1961 to
      2035;
    - each day that ends an interval: its length in UTC seconds and TAI -
      UTC at the next day's 0h, exact from the history and rounded once;
    - TDB - TT from eraDtdb at the geocentre, at TT 0.5 s past 0h of days
      drawn from 1600 to 2400.

An instant there is a modified Julian date's day and the seconds since its
0h: ERFA's two-part Julian date read exactly and the seconds rounded once.
TDB is evaluated at TDB, as eraDtdb's argument states: the TDB of a TT is
the fixed point of tdb = tt + dtdb(tdb), reached in two steps.

Without --check both files are written; with it, nothing is written and the
run fails if a file would change.
"""

import argparse
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import random
import re
import sys
import urllib.request
import warnings

import erfa

# ERFA flags a UTC year five or more past its release as dubious and computes
# it as any other: the references draw UTC past Bulletin C's expiry on purpose.
warnings.filterwarnings("ignore", message=".*dubious year.*")

HERE = Path(__file__).resolve().parent
TABLES = HERE.parents[1] / "lib" / "time_tables.ml"
REFERENCE = HERE.parent / "support" / "times_reference.ml"

HISTORY = "https://hpiers.obspm.fr/iers/bul/bulc/UTC-TAI.history"
BULLETIN = "https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat"
DTDB = "https://raw.githubusercontent.com/liberfa/erfa/v2.0.1/src/dtdb.c"

# Each source's SHA-256, and the copy an IERS file is read from.
PINS = {
    HISTORY: "54e702abdc388ae3bf8cfc5f126900a5277829ad90e80f6773df6e714a133642",
    BULLETIN: "6cb6f5d4b819f2e568e25db4b0b26d89dedf031fdffb18bc94d40f4e94e268d7",
    DTDB: "228bd7c7c1c88dda44c29a076f289c968f64d77a2dc389138b03ab5d1d442105",
}
COPIES = {
    HISTORY: HERE / "iers" / "UTC-TAI.history",
    BULLETIN: HERE / "iers" / "Leap_Second.dat",
}

HEADER = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)
"""

TABLES_HEADER = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Copyright (C) 2013-2023, NumFOCUS Foundation (ERFA's terms of TDB - TT).
  SPDX-License-Identifier: ISC AND BSD-3-Clause
  ---------------------------------------------------------------------------*)
"""

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}

# eraDtdb's groups by power of T, as its header comment numbers its terms.
GROUPS = [(1, 474), (475, 679), (680, 764), (765, 784), (785, 787)]


def changed(url):
    sys.exit(f"source changed: {url}: update pins and regenerate")


def pinned(url, data):
    if hashlib.sha256(data).hexdigest() != PINS[url]:
        changed(url)
    return data.decode("latin-1")


def download(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def source(url):
    """The text of a source, checked against its pin."""
    copy = COPIES.get(url)
    return pinned(url, copy.read_bytes() if copy else download(url))


def refresh():
    """Fetch the IERS files into their copies; fail if either changed."""
    for url, copy in COPIES.items():
        data = download(url)
        copy.write_bytes(data)
        pinned(url, data)


# Calendar

def mjd(y, m, d):
    """The modified Julian date of 0h of a proleptic Gregorian date."""
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468 + 40587


def civil(n):
    """The proleptic Gregorian date of the modified Julian date n."""
    z = n - 40587 + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + (3 if mp < 10 else -9)
    return (yoe + era * 400 + (m <= 2), m, d)


# TAI - UTC

def history(text):
    """The intervals of TAI - UTC: (first day, offset, reference, rate),
    the numbers as the history writes them, digit groups closed up."""
    rows = []
    year = None
    drift = None
    row = re.compile(
        r"^\s*(\d{4})?\s+([A-Za-z]+)\.?\s+(\d+)\.?\s*-(.*)$")
    for line in text.splitlines():
        match = row.match(line)
        if not match:
            continue
        if match.group(1):
            year = int(match.group(1))
        month = MONTHS[match.group(2)[:3].lower()]
        day = int(match.group(3))
        rest = match.group(4)
        value = re.search(r"\s(\d+(?:\.[\d ]+)?)s", rest)
        if not value:
            sys.exit(f"no TAI - UTC in the history's row: {line!r}")
        offset = value.group(1).replace(" ", "")
        tail = rest[value.end():]
        linear = re.search(r"\(MJD - ([\d ]+)\) x ([\d. ]+)s", tail)
        if linear:
            drift = (linear.group(1).replace(" ", "") + ".",
                     linear.group(2).replace(" ", ""))
        elif '""' in tail:
            if drift is None:
                sys.exit(f"the history's row repeats no rate: {line!r}")
        else:
            drift = None
        reference, rate = drift if drift else ("0.", "0.")
        rows.append((mjd(year, month, day), offset, reference, rate))
    if not rows:
        sys.exit("the history has no rows")
    starts = [r[0] for r in rows]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        sys.exit("the history's intervals do not ascend")
    return rows


def bulletin(text):
    """Bulletin C's rows (MJD, TAI - UTC) and its expiry day."""
    rows = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            sys.exit(f"Bulletin C's row has {len(fields)} fields: {line!r}")
        day = Fraction(fields[0])
        d, m, y = (int(f) for f in fields[1:4])
        if day != mjd(y, m, d):
            sys.exit(f"Bulletin C's MJD {fields[0]} is not {y}-{m}-{d}")
        rows.append((int(day), fields[4]))
    match = re.search(r"File expires on (\d+) (\w+) (\d{4})", text)
    if not match:
        sys.exit("Bulletin C states no expiry")
    d, month, y = match.groups()
    expires = mjd(int(y), MONTHS[month[:3].lower()], int(d))
    return rows, expires


def agree(intervals, leaps):
    """The history's rows from 1972 are Bulletin C's."""
    since = [(s, o) for s, o, ref, rate in intervals if Fraction(rate) == 0]
    stated = [(s, str(int(Fraction(o)))) for s, o in since]
    if stated != leaps:
        sys.exit("the history and Bulletin C disagree from 1972")
    first = since[0][0]
    if any(Fraction(rate) == 0 for s, _, _, rate in intervals if s < first):
        sys.exit("the history has a whole-second interval before 1972")


# eraDtdb

def dtdb_terms(text):
    body = text.split("fairhd[787][3] = {", 1)
    if len(body) != 2:
        sys.exit("dtdb.c has no fairhd table")
    table = body[1].split("};", 1)[0]
    number = r"([-+]?\d*\.\d+(?:e[-+]?\d+)?)"
    terms = re.findall(r"\{\s*" + number + r",\s*" + number + r",\s*"
                       + number + r"\s*\}", table)
    if len(terms) != 787:
        sys.exit(f"dtdb.c's table has {len(terms)} terms, not 787")
    loops = re.findall(r"for \(j = (\d+); j >= (\d+); j--\)", text)
    bounds = [(int(lo) + 1, int(hi) + 1) for hi, lo in loops]
    if bounds != GROUPS:
        sys.exit(f"dtdb.c groups its terms as {bounds}")
    jpl = re.search(r"wj =(.*?);", text, re.S)
    if not jpl:
        sys.exit("dtdb.c has no JPL adjustment")
    sines = re.findall(
        r"\(?\s*" + number + r" \* sin\(\s*" + number + r" \* t \+ "
        + number + r"\)\)?", jpl.group(1))
    square = re.findall(r"\+\s*" + number + r" \* t \* t", jpl.group(1))
    if len(sines) != 4 or len(square) != 1:
        sys.exit("dtdb.c's JPL adjustment is not four sines and a T^2 term")
    return terms, sines, square[0]


# Instants

def instant(j1, j2):
    """(MJD, seconds since its 0h) of an exact two-part JD."""
    x = Fraction(j1) + Fraction(j2) - Fraction(4800001, 2)
    day = math.floor(x)
    return day, float((x - day) * 86400)


def tdb_of_tt(t1, t2):
    dtr = erfa.dtdb(t1, t2, 0.0, 0.0, 0.0, 0.0)
    dtr = erfa.dtdb(t1, t2 + dtr / 86400.0, 0.0, 0.0, 0.0, 0.0)
    return erfa.tttdb(t1, t2, dtr)


def offset(intervals, n):
    """TAI - UTC at 0h UTC of day n, and its rate, exactly."""
    return next((Fraction(o) + (n - Fraction(ref)) * Fraction(rate),
                 Fraction(rate))
                for s, o, ref, rate in reversed(intervals) if s <= n)


def ends(intervals):
    """Each day that ends an interval: (day, TAI - UTC at its 0h, its rate,
    its jump), the jump as SOFA's convention reads it, exactly."""
    out = []
    for start, _, _, _ in intervals[1:]:
        dat0, rate = offset(intervals, start - 1)
        after, _ = offset(intervals, start)
        out.append((start - 1, dat0, rate, after - dat0 - rate))
    return out


def utc_instants(intervals):
    """(date, clock) of UTC instants: around each leap second, around the
    end of each interval before 1972, and at random."""
    out = []
    starts = [s for s, _, _, _ in intervals]
    for last, _, _, jump in ends(intervals):
        before = civil(last)
        after = civil(last + 1)
        out.append((before, (23, 59, 59.25)))
        if jump >= 1:
            out += [(before, (23, 59, s)) for s in (60.0, 60.5, 60.999)]
        elif jump > 0:
            out.append((before, (23, 59, 60 + float(jump) / 2)))
        out += [(after, (0, 0, 0.0)), (after, (0, 0, 0.75))]
    rng = random.Random(2026)
    first, last = starts[0], mjd(2035, 1, 1)
    for _ in range(120):
        day = civil(rng.randrange(first, last))
        clock = (rng.randrange(24), rng.randrange(60),
                 round(rng.uniform(0, 60), 6) % 60)
        out.append((day, clock))
    return out


def utc_references(intervals):
    rows = []
    for (y, m, d), (h, mi, s) in utc_instants(intervals):
        u1, u2 = erfa.dtf2d("UTC", y, m, d, h, mi, s)
        a1, a2 = erfa.utctai(u1, u2)
        t1, t2 = erfa.taitt(a1, a2)
        b1, b2 = tdb_of_tt(t1, t2)
        rows.append(((y, m, d), (h, mi, s),
                     instant(a1, a2), instant(t1, t2), instant(b1, b2)))
    return rows


def dtdb_references():
    rng = random.Random(1990)
    rows = []
    for _ in range(60):
        day = rng.randrange(mjd(1600, 1, 1), mjd(2400, 1, 1))
        t1, t2 = 2400000.5 + day, 0.5 / 86400.0
        b1, b2 = tdb_of_tt(t1, t2)
        dtr = erfa.dtdb(b1, b2, 0.0, 0.0, 0.0, 0.0)
        rows.append((day, float(dtr)))
    return rows


# Rendering

def ocaml_float(x):
    """The shortest decimal that reads back as x."""
    text = repr(x)
    return text if any(c in text for c in ".en") else text + "."


def literal(text):
    """An OCaml float literal of a C decimal literal, the same digits."""
    text = text.lstrip("+")
    if text.startswith("-."):
        text = "-0." + text[2:]
    elif text.startswith("."):
        text = "0." + text[1:]
    if "." not in text and "e" not in text:
        text += "."
    return text


def rows_of(values, per_line):
    lines = []
    for i in range(0, len(values), per_line):
        lines.append("    " + "; ".join(values[i:i + per_line]) + ";")
    return "\n".join(lines)


def erfa_notice(text):
    """ERFA's copyright notice and license, as dtdb.c ends."""
    start = text.rfind("Copyright (C)")
    end = text.rfind("*/")
    lines = [line.removeprefix("**").rstrip()
             for line in text[start:end].splitlines()]
    return "\n".join(("  " + line.strip(" ")) if line.strip() else ""
                     for line in lines).rstrip()


def render_tables(sources, intervals, expires, dtdb_text):
    terms, sines, square = dtdb_terms(dtdb_text)
    out = [TABLES_HEADER, f"""
(* Written by test/gen/times.py; do not edit. Regenerate with
   uv run contrib/ymir/test/gen/times.py, which reads the IERS files from
   its copies in test/gen/iers. Each source and its SHA-256:
""".lstrip("\n")]
    out[-1] = "\n" + out[-1]
    for url, sha in sources:
        out.append(f"     {url}\n       {sha}\n")
    y, m, d = civil(expires)
    out.append(f"""
   The terms of TDB - TT are ERFA's eraDtdb's, the IAU SOFA library's
   iauDtdb as ERFA distributes it under the license at the end of this
   file and in LICENSE-erfa. They are derived from ERFA and are neither
   ERFA nor SOFA software, nor endorsed by either. {{!Time}} sums the same
   series at the geocentre: without eraDtdb's topocentric terms, and
   evaluated in nx. *)

[@@@ocamlformat "disable"]

(* TAI - UTC. From 0h UTC of the modified Julian date [utc_starts.(i)]
   until the next start, TAI - UTC is
   [utc_offsets.(i) + (MJD - utc_references.(i)) * utc_rates.(i)] seconds,
   MJD the UTC date with its fraction of a day. Starts ascend; the rows
   from 1972 have rate 0. *)
""")
    out.append("let utc_starts = [|\n"
               + rows_of([str(s) for s, _, _, _ in intervals], 8) + "\n|]\n\n")
    out.append("let utc_offsets = [|\n"
               + rows_of([literal(o) for _, o, _, _ in intervals], 6)
               + "\n|]\n\n")
    out.append("let utc_references = [|\n"
               + rows_of([literal(r) for _, _, r, _ in intervals], 6)
               + "\n|]\n\n")
    out.append("let utc_rates = [|\n"
               + rows_of([literal(r) for _, _, _, r in intervals], 6)
               + "\n|]\n\n")
    out.append(f"""(* The modified Julian date on which Bulletin C's table expires,
   {y}-{m:02d}-{d:02d}. *)
let utc_expires = {expires}

""")
    out.append("""(* TDB - TT at the geocentre, in seconds, is
   [sum_k T^k sum_j a_kj sin (f_kj T + p_kj)] plus [jpl]'s terms, T in
   Julian millennia of TDB since J2000.0. [fairhead.(k)] holds the terms
   of power k as triples (a, f, p): a in seconds, f in radians per Julian
   millennium, p in radians, flattened. *)
let fairhead = [|
""")
    for k, (lo, hi) in enumerate(GROUPS):
        group = terms[lo - 1:hi]
        values = [f"{literal(a)}; {literal(f)}; {literal(p)}"
                  for a, f, p in group]
        out.append(f"  (* T^{k}: terms {lo} to {hi} *)\n  [|\n"
                   + "\n".join("    " + v + ";" for v in values) + "\n  |];\n")
    out.append("""|]

(* eraDtdb's adjustment to JPL's planetary masses: triples (a, f, p) as
   [fairhead]'s, of power 0, then [jpl_t2 * T^2] seconds. *)
let jpl = [|
""")
    out.append("\n".join(
        f"  {literal(a)}; {literal(f)}; {literal(p)};" for a, f, p in sines))
    out.append(f"\n|]\n\nlet jpl_t2 = {literal(square)}\n")
    out.append("\n(* ERFA's notice, from eraDtdb's source:\n\n")
    out.append(erfa_notice(dtdb_text).replace("(*", "( *").replace("*)", "* )"))
    out.append("\n*)\n")
    return "".join(out)


def pair(p):
    return f"({p[0]}, {ocaml_float(p[1])})"


def render_reference(sources, intervals):
    out = [HEADER, "\n(* Written by test/gen/times.py; do not edit. Sources:\n"]
    for url, sha in sources:
        out.append(f"     {url}\n       {sha}\n")
    out.append("""   and pyerfa's ERFA. *)

[@@@ocamlformat "disable"]

(* A UTC date and clock, and ERFA's TAI, TT and TDB of it, each as
   (modified Julian date, seconds since its 0h), the seconds rounded once
   from ERFA's two-part Julian date. TDB is at the geocentre. *)
let utc = [
""")
    for (y, m, d), (h, mi, s), tai, tt, tdb in utc_references(intervals):
        out.append(f"  (({y}, {m}, {d}), ({h}, {mi}, {ocaml_float(s)}), "
                   f"{pair(tai)}, {pair(tt)}, {pair(tdb)});\n")
    out.append("""]

(* A modified Julian date and eraDtdb's TDB - TT at the geocentre, in
   seconds, at the TDB of TT 0.5 s past the day's 0h. *)
let dtdb = [
""")
    for day, dtr in dtdb_references():
        out.append(f"  ({day}, {ocaml_float(dtr)});\n")
    out.append("""]

(* Each day that ends an interval of TAI - UTC: its modified Julian date,
   its length in UTC seconds, TAI - UTC at its 0h, its rate in seconds a
   day, and TAI - UTC at the next day's 0h, each exact from the history and
   rounded once. *)
let ends = [
""")
    for day, dat0, rate, jump in ends(intervals):
        values = [86400 + jump, dat0, rate, dat0 + rate + jump]
        out.append(f"  ({day}, "
                   + ", ".join(ocaml_float(float(v)) for v in values)
                   + ");\n")
    out.append("]\n")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        refresh()
        return
    history_text = source(HISTORY)
    bulletin_text = source(BULLETIN)
    dtdb_text = source(DTDB)
    sources = [(url, PINS[url]) for url in (HISTORY, BULLETIN, DTDB)]
    intervals = history(history_text)
    leaps, expires = bulletin(bulletin_text)
    agree(intervals, leaps)
    files = [
        (TABLES, render_tables(sources, intervals, expires, dtdb_text)),
        (REFERENCE, render_reference(sources, intervals)),
    ]
    if args.check:
        for path, text in files:
            if not path.exists() or path.read_text() != text:
                sys.exit(f"{path} would change; run without --check")
        return
    for path, text in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    main()
