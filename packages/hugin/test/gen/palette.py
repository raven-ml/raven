"""Derives the palette of hugin's default theme, Tableau 10 darkened to read
on white paper, and checks the one `lib/theme.ml` holds.

    uv run packages/hugin/test/gen/palette.py [--check]

A mark needs a contrast of 3:1 against its background (WCAG 2.2, 1.4.11).
Five colours of Tableau 10 fall below it on white. Each colour keeps its
Oklch hue and chroma, its chroma reduced only to stay in sRGB, and moves
in lightness alone:

1. A colour may take its Tableau 10 lightness, or the lightness where it
   reaches 3:1 on white if that is darker, or darker ones by steps of STEP
   down to FLOOR.
2. Neighbours, the colours of classes i and i + 1 with the last next to the
   first, stay apart by an Oklab distance of at least NEIGHBOURS_NORMAL for
   normal vision and NEIGHBOURS under protanopia and deuteranopia,
   simulated with the matrices of Machado, Oliveira and Fernandes (2009) at
   severity 1 on linear sRGB, each with MARGIN to spare. Darkening to 3:1
   alone compresses the lightnesses and breaks some of these.
3. Of the palettes that meet 2, the derived one darkens least: the least
   sum of the squares of the steps past 1's first lightness, found exactly
   by a shortest path around the cycle of neighbours. Any two colours of
   it must then stay apart by ANY for normal vision.

Every colour is rounded to 8 bits before it is measured. Without --check
the palette and its measures are printed; with --check the script fails
if `lib/theme.ml`'s palette differs. runtest checks the laws the
`.mli` states (contrast and distances, in `test/test_figure.ml`) but not
that the palette darkens least, which needs this search and Python; run
--check after changing the palette.
"""

import argparse
import itertools
import math
import pathlib
import re
import sys

TABLEAU10 = ["4e79a7", "f28e2c", "e15759", "76b7b2", "59a14f",
             "edc949", "af7aa1", "ff9da7", "9c755f", "bab0ab"]
NAMES = ["blue", "orange", "red", "teal", "green",
         "yellow", "purple", "pink", "brown", "grey"]

CONTRAST = 3.0
NEIGHBOURS = 0.07
NEIGHBOURS_NORMAL = 0.11
ANY = 0.07
MARGIN = 1.01
STEP = 0.005
FLOOR = 0.45

WHITE = (1.0, 1.0, 1.0)

# Machado, Oliveira and Fernandes 2009, severity 1, on linear sRGB.
MACHADO = {
    "protanopia": [[0.152286, 1.052583, -0.204868],
                   [0.114503, 0.786281, 0.099216],
                   [-0.003882, -0.048116, 1.051998]],
    "deuteranopia": [[0.367322, 0.860646, -0.227968],
                     [0.280085, 0.672501, 0.047413],
                     [-0.011820, 0.042940, 0.968881]],
    "tritanopia": [[1.255528, -0.076749, -0.178779],
                   [-0.078411, 0.930809, 0.147602],
                   [0.004733, 0.691367, 0.303900]],
}


def rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def linear(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def encode(v):
    return 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def hex_of_linear(lin):
    return "".join(f"{round(encode(min(1, max(0, v))) * 255):02x}" for v in lin)


def luminance(c):
    r, g, b = map(linear, c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(c, c2):
    l, l2 = luminance(c), luminance(c2)
    return (max(l, l2) + 0.05) / (min(l, l2) + 0.05)


def oklab(c):
    r, g, b = map(linear, c)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (math.copysign(abs(x) ** (1 / 3), x) for x in (l, m, s))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def linear_of_oklch(lightness, chroma, hue):
    a, b = chroma * math.cos(hue), chroma * math.sin(hue)
    l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def in_gamut(lin):
    return all(-1e-9 <= v <= 1 + 1e-9 for v in lin)


def bisect(ok, lo, hi):
    """The largest x in [lo, hi] with ok x, for ok true at lo and monotone."""
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def at(lightness, chroma, hue):
    """The 8-bit colour at an Oklch point, its chroma reduced into sRGB."""
    if not in_gamut(linear_of_oklch(lightness, chroma, hue)):
        chroma = bisect(lambda c: in_gamut(linear_of_oklch(lightness, c, hue)),
                        0.0, chroma)
    return hex_of_linear(linear_of_oklch(lightness, chroma, hue))


def simulate(c, kind):
    m = MACHADO[kind]
    lin = [linear(v) for v in c]
    out = [sum(m[i][j] * lin[j] for j in range(3)) for i in range(3)]
    return tuple(encode(min(1, max(0, v))) for v in out)


def distance(c, c2):
    return math.dist(oklab(c), oklab(c2))


def neighbours(cs):
    return [(cs[i], cs[(i + 1) % len(cs)]) for i in range(len(cs))]


def measures(hexes):
    """The least distances the rule bounds, as (name, distance, minimum)."""
    cs = [rgb(h) for h in hexes]
    out = [("neighbours, normal vision",
            min(distance(a, b) for a, b in neighbours(cs)), NEIGHBOURS_NORMAL)]
    for kind in ("protanopia", "deuteranopia"):
        sim = [simulate(c, kind) for c in cs]
        out.append((f"neighbours, {kind}",
                    min(distance(a, b) for a, b in neighbours(sim)), NEIGHBOURS))
    out.append(("any two, normal vision",
                min(distance(a, b) for a, b in itertools.combinations(cs, 2)),
                ANY))
    return out


def apart(c, c2):
    """Whether neighbours c and c2 meet their minimums with MARGIN."""
    if distance(c, c2) < NEIGHBOURS_NORMAL * MARGIN:
        return False
    return all(distance(simulate(c, k), simulate(c2, k)) >= NEIGHBOURS * MARGIN
               for k in ("protanopia", "deuteranopia"))


def derive():
    lch = []
    for h in TABLEAU10:
        lightness, a, b = oklab(rgb(h))
        lch.append((lightness, math.hypot(a, b), math.atan2(b, a)))

    def colour(i, lightness):
        _, chroma, hue = lch[i]
        return at(lightness, chroma, hue)

    def reads(i):
        return lambda l: contrast(rgb(colour(i, l)), WHITE) >= CONTRAST

    # The candidates of colour i: its lightness capped at 3:1, then darker
    # by STEP down to FLOOR, each with its cost, the square of the step.
    n = len(TABLEAU10)
    top = [min(lch[i][0], bisect(reads(i), 0.0, 1.0)) for i in range(n)]
    cands = []
    for i in range(n):
        ks = range(int((top[i] - FLOOR) / STEP) + 1)
        cands.append([(colour(i, top[i] - k * STEP), (k * STEP) ** 2) for k in ks])

    ok = {}
    def edge(i, a, b):
        key = (i, a, b)
        if key not in ok:
            j = (i + 1) % n
            ok[key] = apart(rgb(cands[i][a][0]), rgb(cands[j][b][0]))
        return ok[key]

    # Neighbours form a cycle: fix the first colour's candidate, run a
    # shortest path along the others, and close the cycle on the first.
    best = None
    for first in range(len(cands[0])):
        cost = {first: (cands[0][first][1], [first])}
        for i in range(1, n):
            nxt = {}
            for b, (_, cb) in enumerate(cands[i]):
                options = [(c + cb, path + [b]) for a, (c, path) in cost.items()
                           if edge(i - 1, a, b)]
                if options:
                    nxt[b] = min(options)
            cost = nxt
        closed = [(c, path) for last, (c, path) in cost.items()
                  if edge(n - 1, last, first)]
        if closed and (best is None or min(closed) < best):
            best = min(closed)
    if best is None:
        sys.exit("no palette meets the minimums")
    hexes = [cands[i][k][0] for i, k in enumerate(best[1])]
    cs = [rgb(h) for h in hexes]
    if min(distance(a, b) for a, b in itertools.combinations(cs, 2)) < ANY:
        sys.exit("the closest palette has two colours nearer than ANY")
    return hexes


def held(theme):
    """The palette `theme.ml` holds: the hex integers of `light_palette`."""
    m = re.search(r"let light_palette =(.*?)\|\]", theme.read_text(), re.S)
    if m is None:
        sys.exit(f"{theme}: no light_palette")
    return [h.lower() for h in re.findall(r"0x([0-9a-fA-F]{6})", m.group(1))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    hexes = derive()
    theme = pathlib.Path(__file__).resolve().parents[2] / "lib" / "theme.ml"
    if args.check:
        if held(theme) != hexes:
            sys.exit(f"{theme}: light_palette is not " +
                     " ".join("0x" + h for h in hexes))
        return
    for name, old, new in zip(NAMES, TABLEAU10, hexes):
        print(f"{name:7} #{old} {contrast(rgb(old), WHITE):.2f}:1"
              f"  ->  #{new} {contrast(rgb(new), WHITE):.2f}:1")
    for name, d, m in measures(hexes):
        print(f"{name}: {d:.4f} (minimum {m})")
    for kind in ("tritanopia",):
        sim = [simulate(rgb(h), kind) for h in hexes]
        print(f"neighbours, {kind} (unbounded): "
              f"{min(distance(a, b) for a, b in neighbours(sim)):.4f}")
    print("[| " + "; ".join("0x" + h for h in hexes) + " |]")


if __name__ == "__main__":
    main()
