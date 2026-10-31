# /// script
# dependencies = ["fonttools==4.60.1"]
# ///
"""Writes the CFF and CFF2 fixtures of the font tests and their outlines.

    uv run packages/hugin/test/font/gen/gen.py [--check]

The fixtures are subsets of Source Sans 3 (release 3.052R), whose upright
OpenType face has CFF outlines with subroutines and hints, and whose
variable face has CFF2 outlines with blends. Each is downloaded, checked
against its SHA-256, cut down to the glyphs of SAMPLE with its subroutines
kept, and renamed: Source is a Reserved Font Name of the SIL Open Font
License they are under, so the modified fonts take another name and keep the
copyright notice and the license, which LICENSE.md copies.

`cff-2048.otf` has an em of 2048 units: the CFF outlines of the glyphs of
SMALL of the bundled Inter Regular, its quadratic curves written as the cubic
ones they are, renamed. Inter is under the SIL Open Font License without a
Reserved Font Name; LICENSE-Inter.txt copies its license.

Each face also gives `<name>-small.otf`, of the glyphs of SMALL and the
names the license asks to keep, small enough that the robustness tests
overwrite each of its bytes.

For each fixture, `<name>.expected` lists every glyph's outline as fontTools
decodes it, at the default instance for CFF2: `M`, `L`, `C` and `Z` with
coordinates in font units, y up, each a multiple of 2^-16 written as that
multiple's integer so that it reads back exactly.

Without --check the files are written; with --check nothing is written and
the run fails if a file would change.
"""

import argparse
import hashlib
import io
import os
import sys
import urllib.request

from fontTools import subset
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont

RELEASE = "https://github.com/adobe-fonts/source-sans/raw/3.052R"
FACES = {
    "cff": (
        "OTF/SourceSans3-Regular.otf",
        "08df266400933d3178d081a45f94a08814c3e55b4b7dd2e0ff69cb1329f13ab6",
    ),
    "cff2": (
        "VF/SourceSans3VF-Upright.otf",
        "3d0dfd6a3a644ab3d462a737923ffac41fb0ae007ce9ba83c24e6bfa76aa56c7",
    ),
}
LICENSE = "https://raw.githubusercontent.com/adobe-fonts/source-sans/3.052R/LICENSE.md"
NAME = "Hugin Fixture"
# The glyphs of the fixtures, and of the small ones every byte of which the
# robustness tests overwrite.
SAMPLE = "Hamburgefontsiv 0123456789 &@%?!éÅßœ()[]{}"
SMALL = "Ho"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "golden")
FONTS = os.path.join(HERE, "..", "..", "..", "lib", "font", "fonts")


def fetch(url, sha256=None):
    with urllib.request.urlopen(url) as r:
        data = r.read()
    if sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha256:
            sys.exit(f"{url}: SHA-256 {digest}, expected {sha256}")
    return data


def renamed(font, postscript):
    """`font` with every name that holds the Reserved Font Name replaced."""
    names = {1: NAME, 3: f"{NAME} {postscript}", 4: NAME, 6: postscript,
             16: NAME, 17: "Regular", 21: NAME, 22: "Regular", 25: postscript}
    table = font["name"]
    for record in table.names:
        if record.nameID in names:
            record.string = names[record.nameID]
        elif record.nameID not in (0, 7):
            # Instance names name the family; the copyright and trademark
            # notices stay.
            text = record.toUnicode()
            for old, new in (("Source Sans 3", NAME), ("SourceSans3", "HuginFixture")):
                text = text.replace(old, new)
            record.string = text
    if "CFF " in font:
        cff = font["CFF "].cff
        cff.fontNames = [postscript]
        top = cff.topDictIndex[0]
        for key in ("FullName", "FamilyName"):
            if hasattr(top, key):
                setattr(top, key, NAME)
    return font


def subset_of(data, postscript, text, names):
    # The release's timestamps stay, so that the fixtures do not change.
    font = TTFont(io.BytesIO(data), recalcTimestamp=False)
    options = subset.Options()
    options.desubroutinize = False
    options.hinting = True
    options.notdef_outline = True
    options.name_IDs = names
    options.name_languages = ["*"]
    options.layout_features = []
    options.drop_tables += ["GSUB", "GPOS", "GDEF", "STAT"]
    if names != ["*"]:
        options.drop_tables += ["BASE", "HVAR", "MVAR", "avar", "fvar"]
    sub = subset.Subsetter(options)
    sub.populate(text=text)
    sub.subset(font)
    renamed(font, postscript)
    out = io.BytesIO()
    font.save(out)
    return out.getvalue()


def inter_cff():
    """The CFF font of the glyphs of SMALL of the bundled Inter Regular."""
    src = TTFont(os.path.join(FONTS, "Inter-Regular.ttf"))
    options = subset.Options()
    options.notdef_outline = True
    options.layout_features = []
    sub = subset.Subsetter(options)
    sub.populate(text=SMALL)
    sub.subset(src)
    order = src.getGlyphOrder()
    glyphs = src.getGlyphSet()
    charstrings = {}
    for name in order:
        # Cubic curves hold the quadratic ones exactly, up to the 16.16
        # numbers charstrings write their control points in.
        pen = T2CharStringPen(None, glyphs, roundTolerance=0)
        glyphs[name].draw(pen)
        charstrings[name] = pen.getCharString()
    postscript = "HuginFixture-cff2048"
    fb = FontBuilder(src["head"].unitsPerEm, isTTF=False)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(src.getBestCmap())
    fb.setupCFF(postscript, {"FullName": NAME}, charstrings, {})
    fb.setupHorizontalMetrics({n: src["hmtx"][n] for n in order})
    hhea = src["hhea"]
    fb.setupHorizontalHeader(ascent=hhea.ascent, descent=hhea.descent)
    fb.setupNameTable(
        {"familyName": NAME, "styleName": "Regular", "psName": postscript,
         "copyright": src["name"].getDebugName(0)}
    )
    os2 = src["OS/2"]
    fb.setupOS2(
        version=os2.version,
        sCapHeight=os2.sCapHeight,
        sxHeight=os2.sxHeight,
        usWeightClass=os2.usWeightClass,
        sTypoAscender=os2.sTypoAscender,
        sTypoDescender=os2.sTypoDescender,
        sTypoLineGap=os2.sTypoLineGap,
    )
    fb.setupPost()
    fb.font["head"].created = fb.font["head"].modified = 0
    out = io.BytesIO()
    fb.font.recalcTimestamp = False
    fb.save(out)
    return out.getvalue()


def fixed(v):
    n = v * 65536
    if n != int(n):
        sys.exit(f"the coordinate {v!r} is not a multiple of 2^-16")
    return str(int(n))


def outlines(data):
    font = TTFont(io.BytesIO(data))
    glyphs = font.getGlyphSet()
    lines = []
    for g, name in enumerate(font.getGlyphOrder()):
        pen = RecordingPen()
        glyphs[name].draw(pen)
        lines.append(f"glyph {g}")
        for op, pts in pen.value:
            if op == "moveTo":
                lines.append("M " + " ".join(fixed(v) for v in pts[0]))
            elif op == "lineTo":
                lines.append("L " + " ".join(fixed(v) for v in pts[0]))
            elif op == "curveTo":
                lines.append("C " + " ".join(fixed(v) for p in pts for v in p))
            elif op == "closePath":
                lines.append("Z")
            else:
                sys.exit(f"glyph {g}: the pen operation {op}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = {"LICENSE.md": fetch(LICENSE)}
    for name, (path, sha256) in FACES.items():
        face = fetch(f"{RELEASE}/{path}", sha256)
        data = subset_of(face, f"HuginFixture-{name}", SAMPLE, ["*"])
        files[f"{name}.otf"] = data
        files[f"{name}.expected"] = outlines(data).encode()
        # The copyright notice and the family's names stay.
        small = subset_of(face, f"HuginFixture-{name}", SMALL, [0, 1, 2, 6])
        files[f"{name}-small.otf"] = small
    data = inter_cff()
    files["cff-2048.otf"] = data
    files["cff-2048.expected"] = outlines(data).encode()
    with open(os.path.join(FONTS, "LICENSE.txt"), "rb") as f:
        files["LICENSE-Inter.txt"] = f.read()
    changed = []
    for name, data in files.items():
        path = os.path.join(OUT, name)
        old = open(path, "rb").read() if os.path.exists(path) else None
        if old != data:
            changed.append(name)
            if not args.check:
                with open(path, "wb") as f:
                    f.write(data)
    for name in changed:
        print(("would change " if args.check else "wrote ") + name)
    if args.check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
