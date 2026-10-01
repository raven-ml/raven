"""Writes the TZif fixtures of the Tz suite.

Run from this directory with `uv run fixtures.py`. It needs zic and zdump
(tzcode) and a system zoneinfo with tzdata.zi; the fixtures are committed, so
the suite itself needs neither.

- zoneinfo/: zones copied from the system, the slim and leap-second builds of
  Europe/Paris, a copy whose version byte reads '5', hand-made zones, and files
  that are not TZif.
- malformed/<case>/Zone: one file per way of breaking RFC 9636.
- transitions.txt: zdump's transitions of the copied zones from 1800 to 2200,
  as `zone instant offset-before offset-after`.
"""

import calendar
import shutil
import struct
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

SYSTEM = Path("/usr/share/zoneinfo")
ZONES = [
    "Europe/Paris",
    "America/New_York",
    "Australia/Lord_Howe",
    "America/Nuuk",
    "Asia/Jerusalem",
    "UTC",
    "Etc/GMT+5",
]
HERE = Path(__file__).parent
ZONEINFO = HERE / "zoneinfo"
MALFORMED = HERE / "malformed"


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


def zic(args, dest):
    subprocess.run(["zic", *args, "-d", str(dest), str(SYSTEM / "tzdata.zi")], check=True)


def leap_lines():
    lines = (SYSTEM / "leapseconds").read_text().splitlines()
    return [l for l in lines if l.startswith("Leap")]


def transitions(zone):
    out = subprocess.run(
        ["zdump", "-V", "-c", "1800,2200", zone], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    rows = []
    for before, after in zip(out[0::2], out[1::2]):
        ut = " ".join(after.split()[1:6])
        at = calendar.timegm(datetime.strptime(ut, "%a %b %d %H:%M:%S %Y").timetuple())
        offset = lambda line: int(line.rsplit("gmtoff=", 1)[1])
        rows.append(f"{zone} {at} {offset(before)} {offset(after)}")
    return rows


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


def main():
    shutil.rmtree(ZONEINFO, ignore_errors=True)
    shutil.rmtree(MALFORMED, ignore_errors=True)
    for zone in ZONES:
        write(ZONEINFO / zone, (SYSTEM / zone).read_bytes())
    write(ZONEINFO / "zone.tab", b"# A file of a zoneinfo directory that is not TZif.\nFR\t+4852+00220\tEurope/Paris\n")
    write(ZONEINFO / "tiny", b"TZ")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zic(["-b", "slim"], tmp / "slim")
        write(ZONEINFO / "slim/Europe/Paris", (tmp / "slim/Europe/Paris").read_bytes())
        (tmp / "leap").write_text(
            "\n".join(leap_lines() + ["Expires 2027 Jun 28 00:00:00", ""])
        )
        zic(["-b", "fat", "-L", str(tmp / "leap")], tmp / "right")
        write(ZONEINFO / "right/Europe/Paris", (tmp / "right/Europe/Paris").read_bytes())
    paris = (SYSTEM / "Europe/Paris").read_bytes()
    second = paris.index(b"TZif", 4)
    v5 = bytearray(paris)
    v5[4] = v5[second + 4] = ord("5")
    write(ZONEINFO / "v5/Europe/Paris", bytes(v5) + b"data of a later version")
    write(ZONEINFO / "hand/Version1", tzif1(*BASE[:2]))
    write(ZONEINFO / "hand/Fixed", tzif1([], [(19800, 0, "IST")]))
    write(ZONEINFO / "hand/AllYearDst", tzif([], [(-4 * 3600, 1, "EDT")], b"XXX3EDT4,0/0,J365/23", version=b"3"))
    write(
        ZONEINFO / "hand/Truncated",
        tzif(
            [(1500000027, 1)],
            [(0, 0, "-00"), (3600, 0, "AAA")],
            b"AAA-1",
            version=b"4",
            leaps=[(1483228826, 27), (1814140827, 27)],
        ),
    )
    write(
        ZONEINFO / "hand/Rules",
        tzif([], [(3600, 0, "AAA")], b"AAA-1BBB-2,M12.5.0/1:30:15,J60/-2:15", version=b"3"),
    )
    # Transitions closer than the offsets they change: a local time skipped
    # twice, and one read three times.
    write(ZONEINFO / "hand/Skips", tzif([(1000, 1), (1500, 0), (2000, 1)], [(0, 0, "AAA"), (3600, 0, "BBB")], b""))
    write(ZONEINFO / "hand/Folds", tzif([(1000, 1), (1500, 2)], [(7200, 0, "AAA"), (3600, 0, "BBB"), (0, 0, "CCC")], b""))
    # A fold within a gap's span, and two gaps around a fold below the first.
    write(ZONEINFO / "hand/FoldGap", tzif([(1000, 1), (1100, 2)], [(3600, 0, "AAA"), (0, 0, "BBB"), (7200, 0, "CCC")], b""))
    write(ZONEINFO / "hand/GapFoldGap", tzif([(0, 1), (10, 2), (20, 0)], [(0, 0, "AAA"), (3600, 0, "BBB"), (-3600, 0, "CCC")], b""))
    # Footers that disagree with the last transition, as zic 2022g writes
    # America/Ojinaga in slim files: one without daylight saving time, and one
    # whose rule gives daylight saving time at the last transition.
    write(ZONEINFO / "hand/StdDisagrees", tzif(BASE[0], BASE[1], b"BBB-3"))
    write(ZONEINFO / "hand/Disagrees", tzif([(15638400, 1)], [(0, 0, "UTC"), (-21600, 0, "CST")], b"CST6CDT,M3.2.0,M11.1.0"))
    # A footer of daylight saving time all year that disagrees with the last
    # transition: its rule never changes the offset, so the last one holds.
    write(
        ZONEINFO / "hand/PermDisagrees",
        tzif([(1000, 1)], [(0, 0, "AAA"), (-10800, 0, "XXX")], b"XXX3EDT4,0/0,J365/23", version=b"3"),
    )
    # Transitions at 23:59:59 and at the leap second 23:59:60 that follows it,
    # which fall in one POSIX second.
    write(
        ZONEINFO / "hand/LeapPair",
        tzif(
            [(78796799, 1), (78796800, 2)],
            [(0, 0, "AAA"), (3600, 0, "BBB"), (7200, 0, "CCC")],
            b"CCC-2",
            leaps=[(78796800, 1)],
        ),
    )
    # Daylight saving time from February 28 to March 1, which spans February 29
    # in leap years: Julian days never count it.
    write(ZONEINFO / "hand/Julian", tzif([], [(3600, 0, "AAA")], b"AAA-1BBB-2,J59/0,J60/0"))
    write(
        ZONEINFO / "hand/Days",
        tzif([], [(3600, 0, "AAA")], b"AAA-1BBB-2,59/+3,M2.4.6/-2", version=b"3"),
    )
    # A leap second at the epoch, and a transition at the same leap time.
    write(
        ZONEINFO / "hand/LeapAtEpoch",
        tzif([(0, 1)], [(0, 0, "AAA"), (3600, 0, "BBB")], b"BBB-1", leaps=[(0, 1)]),
    )
    for case, content in MALFORMED_CASES.items():
        write(MALFORMED / case / "Zone", content)
    rows = [row for zone in ZONES for row in transitions(zone)]
    (HERE / "transitions.txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
