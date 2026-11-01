# /// script
# requires-python = ">=3.11"
# dependencies = ["spiceypy==6.0.0", "jplephem==2.22", "numpy==2.1.3"]
# ///
"""Generate the SPK test files and their references.

    uv run contrib/ymir/test/spk/gen/spk.py [--check | --refresh]

golden/de440s-2024.bsp
    JPL's DE440s, as NAIF distributes it, cut to 2024 and 2025 by
    jplephem's excerpt tool, which reads the parts of the file it needs
    over HTTP:

        python -m jplephem excerpt 2024/1/1 2026/1/1 \\
            https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp

    then annotated here. NAIF's rules for kernels
    (https://naif.jpl.nasa.gov/naif/rules.html) permit redistributing a
    kernel NAIF distributes only unmodified; "If a kernel distributed by
    NAIF has been modified in any way, any embedded or otherwise allied
    attribution of the original kernel producer must be replaced" by the
    modifier's, and the file renamed. So the cut's internal file name and
    comment area are replaced by a note naming the Raven authors' cut, and
    the file is named for it. Its 14 type 2 segments are the barycentres of
    the planets and Pluto and the Sun from the solar system's barycentre,
    the Moon and the Earth from the Earth-Moon barycentre, and Mercury and
    Venus from theirs. The annotated copy is pinned by its SHA-256 below;
    --refresh cuts and annotates it again, and fails if the cut changed.

golden/types.bsp
    Written here with CSPICE's spkw03 and spkw09, as spiceypy carries it:
    in file order, a type 3 segment of the Moon from the Earth-Moon
    barycentre over 64 days, its position coefficients the excerpt's and
    its velocity coefficients their derivatives; a later type 3 segment of
    the same pair over 16 days inside those, from the Earth's coefficients,
    so that the later segment's priority shows; and a type 9 segment of
    body 1000, a type ymir does not read.

golden/types-big.bsp
    types.bsp in BIG-IEEE: each double and integer of its file, summary
    and data records with its bytes reversed. CSPICE reads it as a
    non-native file, and its states from it at the types instants must
    be its states from types.bsp, bit for bit.

golden/many.bsp
    Written with spkw02 from the excerpt's coefficients: 30 type 2
    segments of the Moon from the Earth-Moon barycentre, 4 records each,
    whose summaries take two summary records; one of the Earth from it; a
    later segment of the Moon from the Earth over 32 of those days, its
    coefficients the Moon's less the Earth's, which serves the Moon there
    whatever the centre asked; and a segment of body 1001 on frame
    ECLIPJ2000.

spk_reference.ml
    TDB instants, drawn and at the records' ends and the segments' ends,
    with CSPICE's geometric states (spkgeo, frame J2000) for pairs of
    bodies in each file, in km and km/s.

Without --check every file is written; with it, nothing is written and the
run fails if a file would change.
"""

import argparse
import hashlib
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile

import numpy as np
import spiceypy as sp
from jplephem.spk import SPK

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"
EXCERPT = GOLDEN / "de440s-2024.bsp"
REFERENCE = HERE.parent / "spk_reference.ml"

DE440S = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"
# The SHA-256 of jplephem's cut, and of the annotated copy.
CUT_SHA = "e691193969a0f8b6379603efc0a21bd83e9e1de39f64d6ee721013b1694cc440"
EXCERPT_SHA = "baa75ccd1d9a5e0908e11b4509341cc5331bc2d0004c5f6f4526351e24edf30c"

HEADER = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)
"""

DAY = 86400.0
RECORD = 1024

NOTE = [
    "de440s-2024.bsp: a test file of the ymir library of Raven, by the",
    "Raven authors: JPL's DE440s, as NAIF distributes it as de440s.bsp,",
    "cut to 2023-12-26 .. 2026-02-01 by jplephem 2.22's excerpt tool,",
    "with this comment and the internal file name replaced. It is not a",
    "NAIF or JPL product; use NAIF's de440s.bsp for any other purpose.",
]


def et_of(mjd, second):
    """TDB seconds past J2000 of [second] past 0h of the modified Julian
    date [mjd]."""
    return (mjd - 51544.5) * DAY + second


def digest(data):
    return hashlib.sha256(data).hexdigest()


# The excerpt

def annotate(cut):
    """[cut] with its internal file name and comment area replaced by
    NOTE: DAF comments are lines ended by NUL in the first 1000 bytes of
    the records between the file record and FWARD, ended by EOT."""
    b = bytearray(cut)
    b[16:76] = b"ymir test cut of DE440s".ljust(60)
    fward = struct.unpack("<i", b[76:80])[0]
    text = b"".join(line.encode() + b"\0" for line in NOTE) + b"\4"
    if fward - 2 < 1 or len(text) > 1000:
        sys.exit("the cut has no room for the note")
    for r in range(2, fward):
        o = (r - 1) * RECORD
        b[o:o + 1000] = (text if r == 2 else b"").ljust(1000, b" ")
    return bytes(b)


def refresh():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        out = Path(d) / "excerpt.bsp"
        subprocess.run([sys.executable, "-m", "jplephem", "excerpt",
                        "2024/1/1", "2026/1/1", DE440S, str(out)], check=True)
        cut = out.read_bytes()
        if digest(cut) != CUT_SHA:
            sys.exit("source changed: the excerpt of de440s.bsp: update the "
                     "pin and regenerate")
        EXCERPT.write_bytes(annotate(cut))


# Written files

MOON, EARTH, EMB = 301, 399, 3


def type3_records(segment, first, count):
    """[count] type 3 records from record [first] of a type 2 segment, as
    spkw03 takes them: the x, y, z position then velocity coefficients;
    spkw03 adds each record's MID and RADIUS."""
    init, intlen, coef = segment.load_array()
    radius = intlen * DAY / 2
    init_s = (init - 2451545.0) * DAY
    out = []
    for r in range(first, first + count):
        pos = [coef[i, r, :] for i in range(3)]
        vel = [np.append(np.polynomial.chebyshev.chebder(p) / radius, 0.0)
               for p in pos]
        out.append(np.concatenate(pos + vel))
    return init_s + first * intlen * DAY, intlen * DAY, np.array(out)


def type2_records(coef, init, intlen, first, count):
    """[count] type 2 records from record [first] of the coefficients
    [coef], (3, records, n), as spkw02 takes them, with their first epoch."""
    init_s = (init - 2451545.0) * DAY
    recs = np.concatenate(
        [np.concatenate([coef[i, r, :] for i in range(3)])
         for r in range(first, first + count)])
    return init_s + first * intlen * DAY, recs


def segments_of(excerpt):
    k = SPK.open(str(excerpt))
    moon = next(s for s in k.segments if s.target == MOON)
    earth = next(s for s in k.segments if s.target == EARTH)
    return k, moon, earth


def write_types(excerpt, path):
    k, moon, earth = segments_of(excerpt)
    handle = sp.spkopn(str(path), "ymir types", 0)
    start, intlen, recs = type3_records(moon, 10, 16)
    degree = recs.shape[1] // 6 - 1
    sp.spkw03(handle, MOON, EMB, "J2000", start, start + 16 * intlen,
              "moon type 3", intlen, len(recs), degree, recs.flatten(), start)
    start2, _, recs2 = type3_records(earth, 14, 4)
    sp.spkw03(handle, MOON, EMB, "J2000", start2, start2 + 4 * intlen,
              "later type 3", intlen, len(recs2), degree, recs2.flatten(),
              start2)
    epochs = [start, start + DAY, start + 2 * DAY]
    states = [[1e5, 2e5, 3e5, 1.0, 2.0, 3.0]] * 3
    sp.spkw09(handle, 1000, EARTH, "J2000", epochs[0], epochs[-1],
              "type 9", 1, 3, states, epochs)
    sp.spkcls(handle)
    k.close()


def write_many(excerpt, path):
    k, moon, earth = segments_of(excerpt)
    init, intlen, cm = moon.load_array()
    _, _, ce = earth.load_array()
    step = intlen * DAY
    degree = cm.shape[2] - 1
    handle = sp.spkopn(str(path), "ymir many", 0)
    for j in range(30):
        start, recs = type2_records(cm, init, intlen, 4 * j, 4)
        sp.spkw02(handle, MOON, EMB, "J2000", start, start + 4 * step,
                  f"moon {j}", step, 4, degree, recs, start)
    start, recs = type2_records(ce, init, intlen, 0, 120)
    sp.spkw02(handle, EARTH, EMB, "J2000", start, start + 120 * step,
              "earth", step, 120, degree, recs, start)
    start, recs = type2_records(cm - ce, init, intlen, 40, 8)
    sp.spkw02(handle, MOON, EARTH, "J2000", start, start + 8 * step,
              "moon from earth", step, 8, degree, recs, start)
    start, recs = type2_records(cm, init, intlen, 0, 1)
    sp.spkw02(handle, 1001, EARTH, "ECLIPJ2000", start, start + step,
              "ecliptic", step, 1, degree, recs, start)
    sp.spkcls(handle)
    k.close()


def big_endian(data):
    """A little-endian DAF/SPK file in BIG-IEEE: the file record's integers,
    the summary records' doubles and integers and the data records'
    doubles each with its bytes reversed; comment and name records kept."""
    b = bytearray(data)
    if b[88:96] != b"LTL-IEEE":
        sys.exit("big_endian: not a little-endian DAF")
    b[88:96] = b"BIG-IEEE"

    def swap(o, n):
        b[o:o + n] = b[o:o + n][::-1]

    nd, ni = struct.unpack("<ii", data[8:16])
    fward = struct.unpack("<i", data[76:80])[0]
    for o in (8, 12, 76, 80, 84):
        swap(o, 4)
    kept = set(range(1, fward))
    r = fward
    while r:
        kept |= {r, r + 1}
        o = (r - 1) * RECORD
        nxt, _, nsum = struct.unpack("<3d", data[o:o + 24])
        for w in range(3):
            swap(o + 8 * w, 8)
        ss = nd + (ni + 1) // 2
        for k in range(int(nsum)):
            s = o + 24 + 8 * ss * k
            for w in range(nd):
                swap(s + 8 * w, 8)
            for j in range(ni):
                swap(s + 8 * nd + 4 * j, 4)
        r = int(nxt)
    for r in range(1, len(data) // RECORD + 1):
        if r in kept:
            continue
        o = (r - 1) * RECORD
        for w in range(RECORD // 8):
            swap(o + 8 * w, 8)
    return bytes(b)


# References

PAIRS = {
    "excerpt": [(10, 0), (399, 0), (301, 399), (301, 10), (5, 3), (199, 0),
                (3, 301)],
    "types": [(MOON, EMB)],
    "many": [(MOON, EMB), (MOON, EARTH), (EARTH, MOON)],
}


def instant(et):
    """(MJD, seconds since its 0h) of TDB seconds past J2000, which et_of
    rebuilds within an ulp."""
    mjd = int(np.floor(et / DAY + 51544.5))
    return mjd, float(et - (mjd - 51544.5) * DAY)


def coverage(path):
    """The segments of types 2 and 3 in [path], in file order, as
    (start, stop, (first epoch as a Julian date, record length in days))."""
    k = SPK.open(str(path))
    segs = [(s.start_second, s.end_second, s.load_array()[:2])
            for s in k.segments if s.data_type in (2, 3)]
    k.close()
    return segs


def instants(name, path):
    """Drawn instants the file's first pair covers, the ends of records,
    and the ends of the segments; instants in a later segment too."""
    segs = coverage(path)
    rng = random.Random(440)
    if name == "excerpt":
        lo = max(s[0] for s in segs)
        hi = min(s[1] for s in segs)
        ets = [rng.uniform(lo, hi) for _ in range(12)]
        _, _, (init, intlen) = segs[10]
    elif name == "types":
        (lo, hi, _), (lo2, hi2, (init, intlen)) = segs[0], segs[1]
        ets = [rng.uniform(lo, hi) for _ in range(8)]
        ets += [rng.uniform(lo2, hi2) for _ in range(4)] + [lo2, hi2]
    else:
        lo, hi = segs[0][0], segs[29][1]
        lo2, hi2, (init, intlen) = segs[31]
        ets = [rng.uniform(lo, hi) for _ in range(8)]
        ets += [rng.uniform(lo2, hi2) for _ in range(4)] + [lo2, hi2]
        ets += [segs[j][1] for j in (25, 26, 29)]
    base = (init - 2451545.0) * DAY
    ets += [base + j * intlen * DAY for j in range(1, 4)] + [lo, hi]
    return list(dict.fromkeys(instant(et) for et in ets))


def references(name, path, at=None):
    """CSPICE's states of [name]'s pairs from [path], at [at] or the
    instants of [path]."""
    sp.furnsh(str(path))
    rows = []
    try:
        for mjd, second in at or instants(name, path):
            et = et_of(mjd, second)
            states = []
            for target, center in PAIRS[name]:
                st, _ = sp.spkgeo(target, et, "J2000", center)
                states.append((target, center, [float(x) for x in st]))
            rows.append((mjd, second, states))
    finally:
        sp.kclear()
    return rows


def ocaml_float(x):
    text = repr(float(x))
    return text if any(c in text for c in ".en") else text + "."


def render(paths):
    out = [HEADER, f"""
(* Written by gen/spk.py; do not edit. CSPICE {sp.tkvrsn("TOOLKIT")}'s
   spkgeo, through spiceypy 6.0.0, of the files in golden. *)

[@@@ocamlformat "disable"]

(* Per instant: a TDB modified Julian date and the seconds since its 0h,
   then per pair (target, centre, state), the state the target's position
   then velocity from the centre on J2000's axes, in km and km/s. *)
"""]
    for name in ("excerpt", "types", "many"):
        out.append(f"let {name} = [\n")
        for mjd, second, states in references(name, paths[name]):
            out.append(f"  ({mjd}, {ocaml_float(second)}, [\n")
            for t, c, st in states:
                vals = "; ".join(ocaml_float(x) for x in st)
                out.append(f"    ({t}, {c}, [| {vals} |]);\n")
            out.append("  ]);\n")
        out.append("]\n\n")
    return "".join(out).rstrip("\n") + "\n"


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
    if digest(EXCERPT.read_bytes()) != EXCERPT_SHA:
        sys.exit(f"source changed: {EXCERPT}: update the pin and regenerate")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        d = Path(d)
        paths = {"excerpt": EXCERPT, "types": d / "types.bsp",
                 "many": d / "many.bsp"}
        write_types(EXCERPT, paths["types"])
        write_many(EXCERPT, paths["many"])
        big = big_endian(paths["types"].read_bytes())
        big_path = d / "types-big.bsp"
        big_path.write_bytes(big)
        at = instants("types", paths["types"])
        if references("types", big_path, at) != references("types", paths["types"], at):
            sys.exit("CSPICE reads types-big.bsp otherwise than types.bsp")
        files = [
            (GOLDEN / "types.bsp", paths["types"].read_bytes()),
            (GOLDEN / "types-big.bsp", big),
            (GOLDEN / "many.bsp", paths["many"].read_bytes()),
            (REFERENCE, render(paths).encode()),
        ]
    if args.check:
        for path, data in files:
            if not path.exists() or path.read_bytes() != data:
                sys.exit(f"{path} would change; run without --check")
        return
    for path, data in files:
        path.write_bytes(data)


if __name__ == "__main__":
    main()
