# /// script
# requires-python = ">=3.11"
# dependencies = ["tzdata==2026.3"]
# ///
"""Writes the TZif fixtures of the Tz suite into golden/tz/.

    uv run packages/talon/test/gen/tz.py [--check]

The tz database is the tzdata package's copy of IANA tzdata 2026c: its
tzdata.zi and leapseconds. zic and zdump are the system's, looked up in PATH,
then /usr/sbin. With --check nothing is written, and the run fails if
golden/tz/ would change.

- zoneinfo/: zones built with `zic -b fat`, the slim and leap-second builds of
  Europe/Paris, a copy whose version byte reads '5', hand-made zones, and files
  that are not TZif.
- malformed/<case>/Zone: one file per way of breaking RFC 9636.
- transitions.txt: `zone instant offset-before offset-after`, the
  transitions of zoneinfo/'s zones in POSIX seconds, but for
  right/Europe/Paris, which has Europe/Paris's. A leap second, 23:59:60,
  reads as the POSIX second of 23:59:59, and transitions within one POSIX
  second merge into one.

The transitions of the built zones are zdump's, from 1800 to 2200, read from
the fixture files. zdump steps half a day at a time and misses transitions
closer than that, so the hand-made zones are read every second near their
transitions through the readers that model them, which must agree:
- glibc's localtime, the reader zdump uses, on the file. glibc ignores the
  footer of a file without transitions, and reads daylight saving time all
  year as a rule that changes at the new year.
- glibc's localtime on the footer alone, as a POSIX TZ string.
- CPython's zoneinfo.ZoneInfo.from_file. It ignores leap seconds, starts a
  zero-based day `n` one day early, and in leap years starts `J59` one day
  late, on February 29.
Both glibc and zoneinfo apply a footer that disagrees with the last
transition from that transition on. tzcode's localtime.c, and Tz.offset_s,
hold the last transition's offset until the rule's next transition.
StdDisagrees, Disagrees and PermDisagrees therefore get no rows here: test_tz.ml
states theirs from Tz's contract.
"""

import argparse
import calendar
import filecmp
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zoneinfo
from datetime import datetime, timedelta
from pathlib import Path

import tzdata

SOURCE = Path(tzdata.__file__).parent / "zoneinfo"
TZDATA_VERSION = "2026c"
ZONES = [
    "Europe/Paris",
    "America/Ojinaga",
    "America/New_York",
    "Australia/Lord_Howe",
    "America/Nuuk",
    "Asia/Jerusalem",
    "UTC",
    "Etc/GMT+5",
]
BUILT = ZONES + ["slim/Europe/Paris", "v5/Europe/Paris"]
# The last explicit transition of right/Europe/Paris, in 2037.
RIGHT_END = 2140045200
GOLDEN = Path(__file__).parent.parent / "golden" / "tz"
DAY = 86_400
EPOCH = datetime(1970, 1, 1)


def block(transitions, types, leaps, isstd, isut, size):
    """A TZif header and data block. [types] are (utoff, isdst, abbr)."""
    chars = b""
    idx = []
    for _, _, abbr in types:
        a = abbr.encode() + b"\0"
        pos = chars.find(a)
        if pos < 0:
            pos = len(chars)
            chars += a
        idx.append(pos)
    t = "q" if size == 8 else "l"
    data = b"".join(struct.pack(">" + t, at) for at, _ in transitions)
    data += bytes(ty for _, ty in transitions)
    data += b"".join(
        struct.pack(">lBB", off, dst, i) for (off, dst, _), i in zip(types, idx)
    )
    data += chars
    data += b"".join(struct.pack(">" + t + "l", o, c) for o, c in leaps)
    data += bytes(isstd) + bytes(isut)
    counts = (len(isut), len(isstd), len(leaps), len(transitions), len(types), len(chars))
    return counts, data


def header(version, counts):
    return b"TZif" + version + b"\0" * 15 + struct.pack(">6L", *counts)


def tzif(transitions, types, footer, version=b"2", leaps=(), isstd=(), isut=()):
    """A version 2+ file with a placeholder version 1 block."""
    c1, d1 = block([], [(0, 0, "UTC")], [], [], [], 4)
    c2, d2 = block(transitions, types, list(leaps), list(isstd), list(isut), 8)
    return header(version, c1) + d1 + header(version, c2) + d2 + b"\n" + footer + b"\n"


def tzif1(transitions, types):
    c, d = block(transitions, types, [], [], [], 4)
    return header(b"\0", c) + d


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def tool(name):
    path = shutil.which(name, path=os.environ.get("PATH", "") + ":/usr/sbin:/sbin")
    if path is None:
        sys.exit(f"{name} (tzcode) not found")
    return path


def zic(args, dest):
    subprocess.run(
        [tool("zic"), *args, "-d", str(dest), str(SOURCE / "tzdata.zi")], check=True
    )


def leap_lines():
    lines = (SOURCE / "leapseconds").read_text().splitlines()
    return [l for l in lines if l.startswith("Leap")]


def posix(y, mo, d, h, mi, s):
    """The POSIX second of a UTC time; 23:59:60 reads as 23:59:59."""
    return calendar.timegm((y, mo, d, h, mi, min(s, 59)))


def merge(rows):
    """[rows] in time order, those within one POSIX second merged."""
    out = []
    for at, before, after in rows:
        if out and out[-1][0] == at:
            out[-1] = (at, out[-1][1], after)
        else:
            out.append((at, before, after))
    return out


MONTHS = {m: i for i, m in enumerate(calendar.month_abbr) if m}
UT = re.compile(r"\s+\w{3} (\w{3}) +(\d+) (\d+):(\d+):(\d+) (-?\d+) UT = (.*)$")


def zdump(path):
    """zdump's transitions of the file [path] from 1800 to 2200. A pair of
    lines whose local time types are equal, which zdump prints around a leap
    second, is not a transition."""
    out = subprocess.run(
        [tool("zdump"), "-V", "-c", "1800,2200", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    rows = []
    for before, after in zip(out[0::2], out[1::2]):
        m = UT.match(after[len(str(path)) :])
        mo, d, h, mi, s, y = m.group(1), *map(int, m.groups()[1:6])
        kind = lambda line: line.split(" UT = ", 1)[1].split(None, 5)[5]
        if kind(before) == kind(after):
            continue
        offset = lambda line: int(line.rsplit("gmtoff=", 1)[1])
        rows.append((posix(y, MONTHS[mo], d, h, mi, s), offset(before), offset(after)))
    return merge(rows)


def glibc(path):
    """glibc's reading of the file [path]: the offset and POSIX second at a
    second of the file's time scale, which counts leap seconds in a file with
    leap records."""

    os.environ["TZ"] = str(path.resolve())
    time.tzset()

    def read(t):
        tm = time.localtime(t)
        at = posix(*tm[:6]) - tm.tm_gmtoff
        return tm.tm_gmtoff, at

    return read


def footer(path):
    """glibc's reading of the footer of the file [path], at a POSIX second."""
    os.environ["TZ"] = path.read_bytes().rstrip(b"\n").rsplit(b"\n", 1)[1].decode()
    time.tzset()
    return lambda t: (time.localtime(t).tm_gmtoff, t)


def python(path):
    """CPython's zoneinfo reading of the file [path], at a POSIX second."""
    with open(path, "rb") as f:
        zone = zoneinfo.ZoneInfo.from_file(f)

    def read(t):
        utc = EPOCH + timedelta(seconds=t)
        wall = zone.fromutc(utc.replace(tzinfo=zone)).replace(tzinfo=None)
        return int((wall - utc).total_seconds()), t

    return read


def scan(read, lo, hi, step):
    """The transitions of [read] from [lo] to [hi], sampled every [step]
    seconds and located by bisection."""
    rows = []
    prev, _ = read(lo)
    t = lo
    while t < hi:
        n = min(t + step, hi)
        if read(n)[0] == prev:
            t = n
            continue
        a, b = t, n
        while b - a > 1:
            m = (a + b) // 2
            if read(m)[0] == prev:
                a = m
            else:
                b = m
        after, at = read(b)
        rows.append((at, prev, after))
        prev, t = after, b
    return merge(rows)


# Two time types around one transition at 1000 s: AAA at +1 h, then BBB at +2 h.
BASE = ([(1000, 1)], [(3600, 0, "AAA"), (7200, 0, "BBB")], b"BBB-2")

MALFORMED_CASES = {
    "truncated": tzif(*BASE)[:-20],
    "second-magic": (lambda f: f[:f.index(b"TZif", 4)] + b"TZjf" + f[f.index(b"TZif", 4) + 4 :])(tzif(*BASE)),
    "version": tzif(*BASE, version=b"1"),
    "versions-differ": (lambda f: f[:4] + b"3" + f[5:])(tzif(*BASE)),
    "no-types": tzif([], [], b"UTC0"),
    "isutcnt": tzif(*BASE, isut=[0]),
    "not-ascending": tzif([(1000, 1), (1000, 0)], BASE[1], b"AAA-1"),
    "type-index": tzif([(1000, 2)], BASE[1], b"BBB-2"),
    "utoff": tzif([(1000, 1)], [(3600, 0, "AAA"), (-(2**31), 0, "BBB")], b"BBB-2"),
    "isdst": tzif([(1000, 1)], [(3600, 0, "AAA"), (7200, 2, "BBB")], b"BBB-2"),
    "ut-not-std": tzif(*BASE, isstd=[0, 0], isut=[0, 1]),
    "indicator": tzif(*BASE, isstd=[0, 2]),
    "footer-start": tzif(*BASE)[:-7] + b"xBBB-2\n",
    "footer-end": tzif(*BASE)[:-1],
    "footer-nul": tzif(BASE[0], BASE[1], b"BBB\0-2"),
    "footer-syntax": tzif(BASE[0], BASE[1], b"BBB-2x"),
    "footer-short": tzif(BASE[0], BASE[1], b"BB-2"),
    "footer-no-rule": tzif([(1000, 1)], [(3600, 0, "AAA"), (7200, 0, "BBB")], b"BBB-2CCC"),
    "footer-hour-v2": tzif(BASE[0], BASE[1], b"BBB-2CCC,M3.5.0/26,M10.5.0"),
    "footer-hour-v3": tzif(BASE[0], BASE[1], b"BBB-2CCC,M3.5.0/168,M10.5.0", version=b"3"),
    "after-footer": tzif(*BASE) + b"x",
    "v1-after-data": tzif1(*BASE[:2]) + b"x",
    "leap-first": tzif(*BASE, leaps=[(78796800, 2)]),
    "leap-order": tzif(*BASE, leaps=[(78796800, 1), (78796800, 2)]),
    "leap-step": tzif(*BASE, leaps=[(78796800, 1), (94694401, 3)]),
    "leap-month": tzif(*BASE, leaps=[(78796801, 1)]),
    "leap-negative": tzif(*BASE, leaps=[(-1, 1)]),
    "leap-unknown": tzif(*BASE, version=b"4", leaps=[(1483228826, 27)]),
    "leap-expiry-v2": tzif(*BASE, leaps=[(78796800, 1), (94694401, 1)]),
    "leap-equal-v4": tzif(
        *BASE, version=b"4", leaps=[(78796800, 1), (94694401, 1), (126230401, 2)]
    ),
    "ut-no-std": tzif(*BASE, isut=[0, 1]),
    "footer-digits": tzif(BASE[0], BASE[1], b"BBB-123"),
    "designation": (lambda f: f.replace(b"\x00\x00\x1c\x20\x00\x04", b"\x00\x00\x1c\x20\x00\x08"))(tzif(*BASE)),
    "designation-nul": (lambda f: f.replace(b"BBB\x00\n", b"BBBB\n"))(tzif(*BASE)),
}


# The hand-made zones: [name: (content, readers, windows)]. A window is
# (lo, hi, step), which readers read in the file's time scale.
BOTH = (glibc, python)
NEAR = lambda *ts: [(min(ts) - DAY, max(ts) + DAY, 1)]
RULES = [(calendar.timegm((2023, 6, 1, 0, 0, 0)), calendar.timegm((2025, 6, 1, 0, 0, 0)), 600)]

HAND = {
    "Version1": (tzif1(*BASE[:2]), BOTH, NEAR(1000)),
    "Fixed": (tzif1([], [(19800, 0, "IST")]), BOTH, NEAR(0)),
    "AllYearDst": (
        tzif([], [(-4 * 3600, 1, "EDT")], b"XXX3EDT4,0/0,J365/23", version=b"3"),
        (python,),
        RULES,
    ),
    "Truncated": (
        tzif(
            [(1500000027, 1)],
            [(0, 0, "-00"), (3600, 0, "AAA")],
            b"AAA-1",
            version=b"4",
            leaps=[(1483228826, 27), (1814140827, 27)],
        ),
        (glibc,),
        NEAR(1500000027),
    ),
    "Rules": (
        tzif([], [(3600, 0, "AAA")], b"AAA-1BBB-2,M12.5.0/1:30:15,J60/-2:15", version=b"3"),
        (footer, python),
        RULES,
    ),
    # Transitions closer than the offsets they change: a local time skipped
    # twice, and one read three times.
    "Skips": (
        tzif([(1000, 1), (1500, 0), (2000, 1)], [(0, 0, "AAA"), (3600, 0, "BBB")], b""),
        BOTH,
        NEAR(1000, 2000),
    ),
    "Folds": (
        tzif([(1000, 1), (1500, 2)], [(7200, 0, "AAA"), (3600, 0, "BBB"), (0, 0, "CCC")], b""),
        BOTH,
        NEAR(1000, 1500),
    ),
    # A fold within a gap's span, and two gaps around a fold below the first.
    "FoldGap": (
        tzif([(1000, 1), (1100, 2)], [(3600, 0, "AAA"), (0, 0, "BBB"), (7200, 0, "CCC")], b""),
        BOTH,
        NEAR(1000, 1100),
    ),
    "GapFoldGap": (
        tzif([(0, 1), (10, 2), (20, 0)], [(0, 0, "AAA"), (3600, 0, "BBB"), (-3600, 0, "CCC")], b""),
        BOTH,
        NEAR(0, 20),
    ),
    # Footers that disagree with the last transition, as zic 2022g writes
    # America/Ojinaga in slim files: one without daylight saving time, and one
    # whose rule gives daylight saving time at the last transition.
    "StdDisagrees": (tzif(BASE[0], BASE[1], b"BBB-3"), (), []),
    "Disagrees": (
        tzif([(15638400, 1)], [(0, 0, "UTC"), (-21600, 0, "CST")], b"CST6CDT,M3.2.0,M11.1.0"),
        (),
        [],
    ),
    # A footer of daylight saving time all year that disagrees with the last
    # transition: its rule never changes the offset, so the last one holds.
    "PermDisagrees": (
        tzif([(1000, 1)], [(0, 0, "AAA"), (-10800, 0, "XXX")], b"XXX3EDT4,0/0,J365/23", version=b"3"),
        (),
        [],
    ),
    # Transitions at 23:59:59 and at the leap second 23:59:60 that follows it,
    # which fall in one POSIX second.
    "LeapPair": (
        tzif(
            [(78796799, 1), (78796800, 2)],
            [(0, 0, "AAA"), (3600, 0, "BBB"), (7200, 0, "CCC")],
            b"CCC-2",
            leaps=[(78796800, 1)],
        ),
        (glibc,),
        NEAR(78796799, 78796800),
    ),
    # Daylight saving time from February 28 to March 1, which spans February 29
    # in leap years: Julian days never count it.
    "Julian": (tzif([], [(3600, 0, "AAA")], b"AAA-1BBB-2,J59/0,J60/0"), (footer,), RULES),
    "Days": (
        tzif([], [(3600, 0, "AAA")], b"AAA-1BBB-2,59/+3,M2.4.6/-2", version=b"3"),
        (footer,),
        RULES,
    ),
    # A leap second at the epoch, and a transition at the same leap time.
    "LeapAtEpoch": (
        tzif([(0, 1)], [(0, 0, "AAA"), (3600, 0, "BBB")], b"BBB-1", leaps=[(0, 1)]),
        (glibc,),
        NEAR(0),
    ),
}


def hand_rows(name, path):
    _, readers, windows = HAND[name]
    found = None
    for reader in readers:
        rows = [r for lo, hi, step in windows for r in scan(reader(path), lo, hi, step)]
        if found is not None and rows != found:
            sys.exit(f"hand/{name}: readers disagree: {found} and {rows}")
        found = rows
    return found or []


def generate(out):
    zoneinfo_dir = out / "zoneinfo"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zic(["-b", "fat"], tmp / "fat")
        for zone in ZONES:
            write(zoneinfo_dir / zone, (tmp / "fat" / zone).read_bytes())
        zic(["-b", "slim"], tmp / "slim")
        write(zoneinfo_dir / "slim/Europe/Paris", (tmp / "slim/Europe/Paris").read_bytes())
        (tmp / "leap").write_text(
            "\n".join(leap_lines() + ["Expires 2027 Jun 28 00:00:00", ""])
        )
        zic(["-b", "fat", "-L", str(tmp / "leap")], tmp / "right")
        write(zoneinfo_dir / "right/Europe/Paris", (tmp / "right/Europe/Paris").read_bytes())
    write(zoneinfo_dir / "zone.tab", b"# A file of a zoneinfo directory that is not TZif.\nFR\t+4852+00220\tEurope/Paris\n")
    write(zoneinfo_dir / "tiny", b"TZ")
    paris = (zoneinfo_dir / "Europe/Paris").read_bytes()
    second = paris.index(b"TZif", 4)
    v5 = bytearray(paris)
    v5[4] = v5[second + 4] = ord("5")
    write(zoneinfo_dir / "v5/Europe/Paris", bytes(v5) + b"data of a later version")
    for name, (content, _, _) in HAND.items():
        write(zoneinfo_dir / "hand" / name, content)
    for case, content in MALFORMED_CASES.items():
        write(out / "malformed" / case / "Zone", content)
    rows = [(zone, zdump(zoneinfo_dir / zone)) for zone in BUILT]
    # glibc adds no leap seconds to the times of a footer's rule, so past its
    # explicit transitions zdump reads right/Europe/Paris 27 s early. The test
    # takes Europe/Paris's transitions for it.
    right = [r for r in zdump(zoneinfo_dir / "right/Europe/Paris") if r[0] <= RIGHT_END]
    if right != [r for r in dict(rows)["Europe/Paris"] if r[0] <= RIGHT_END]:
        sys.exit("zdump reads right/Europe/Paris and Europe/Paris apart")
    rows += [(f"hand/{name}", hand_rows(name, zoneinfo_dir / "hand" / name)) for name in HAND]
    lines = [f"{zone} {at} {b} {a}" for zone, zrows in rows for at, b, a in zrows]
    (out / "transitions.txt").write_text("\n".join(lines) + "\n")


def same(a, b):
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    return not mismatch and not errors and all(same(a / d, b / d) for d in cmp.common_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if tzdata.IANA_VERSION != TZDATA_VERSION:
        sys.exit(f"tzdata {tzdata.IANA_VERSION}, expected {TZDATA_VERSION}")
    if not args.check:
        shutil.rmtree(GOLDEN / "zoneinfo", ignore_errors=True)
        shutil.rmtree(GOLDEN / "malformed", ignore_errors=True)
        generate(GOLDEN)
        return
    with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp))
        if not same(Path(tmp), GOLDEN):
            sys.exit("golden/tz/ is stale: run gen/tz.py")


if __name__ == "__main__":
    main()
