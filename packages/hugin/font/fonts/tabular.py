# /// script
# dependencies = ["fonttools"]
# ///
"""Bakes Inter's tabular figures into the bundled faces.

The bundled faces are subsets of Inter 4.1 without a GSUB table, so a
renderer cannot ask for the `tnum` feature. This script copies the glyphs
that `tnum` substitutes for the digits from the full release into each
bundled face and points the digits' cmap entries at them, so every digit
has one advance and numbers align in columns.

Usage, from the repository root, with Inter 4.1's release unpacked
(https://github.com/rsms/inter/releases/tag/v4.1):

    uv run packages/hugin/font/fonts/tabular.py <release>/extras/ttf

It rewrites the faces in place and refuses a face whose digits already share
one advance.
"""

import os
import sys

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import ttProgram

FACES = ["Inter-Regular.ttf", "Inter-Bold.ttf"]
DIGITS = range(0x30, 0x3A)


def single_substitutions(font, tag):
    """The glyph substitutions of the single-substitution lookups of `tag`."""
    gsub = font["GSUB"].table
    lookups = set()
    for record in gsub.FeatureList.FeatureRecord:
        if record.FeatureTag == tag:
            lookups.update(record.Feature.LookupListIndex)
    mapping = {}
    for i in sorted(lookups):
        for sub in gsub.LookupList.Lookup[i].SubTable:
            if sub.LookupType == 7:
                sub = sub.ExtSubTable
            if sub.LookupType != 1:
                sys.exit(f"{tag} lookup {i} is not a single substitution")
            mapping.update(sub.mapping)
    return mapping


def copy_glyph(release, face, name):
    """Copies the glyph `name` and its metrics from `release` into `face`."""
    glyph = release["glyf"][name]
    glyph.expand(release["glyf"])
    if glyph.isComposite():
        for component in glyph.components:
            if component.glyphName not in face["glyf"]:
                sys.exit(f"{name} uses {component.glyphName}, not in the face")
    if hasattr(glyph, "program"):
        # The bundled faces carry no hinting.
        glyph.program = ttProgram.Program()
        glyph.program.fromBytecode(b"")
    order = face.getGlyphOrder()
    if name not in order:
        face.setGlyphOrder(order + [name])
    face["glyf"][name] = glyph
    face["hmtx"][name] = release["hmtx"][name]


def bake(release_dir, face_path):
    release = TTFont(os.path.join(release_dir, os.path.basename(face_path)))
    # The face keeps the full font's head box and maxima.
    face = TTFont(face_path, recalcBBoxes=False, recalcTimestamp=False)
    cmap = face.getBestCmap()
    if len({face["hmtx"][cmap[c]][0] for c in DIGITS}) == 1:
        sys.exit(f"{face_path} has tabular digits already")
    tnum = single_substitutions(release, "tnum")
    best = release.getBestCmap()
    tabular = {c: tnum[best[c]] for c in DIGITS}
    for name in tabular.values():
        copy_glyph(release, face, name)
    classes = face["GDEF"].table.GlyphClassDef
    for c, name in tabular.items():
        if classes is not None:
            classes.classDefs[name] = classes.classDefs.get(best[c], 1)
        for table in face["cmap"].tables:
            if table.isUnicode() and c in table.cmap:
                table.cmap[c] = name
    face.save(face_path)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    for face in FACES:
        bake(sys.argv[1], os.path.join(here, face))


if __name__ == "__main__":
    main()
