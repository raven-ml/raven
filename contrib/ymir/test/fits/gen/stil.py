# /// script
# requires-python = ">=3.13"
# ///
"""Write survey/stil-fits-plus.fits with STILTS.

    uv run contrib/ymir/test/fits/gen/stil.py

STIL, the table library behind TOPCAT and STILTS, writes fits-plus: a
primary byte array holding a VOTable that describes the table, then the
table as a binary table. The columns are synthetic, functions of the row
number, with NaN floats and blank integers. STIL stamps the time of
writing in the VOTable and the table header, so a run writes new bytes:
the file is committed once and gen/survey.py pins its SHA-256.
"""

import pathlib
import subprocess
import sys

STILTS = "3.5-4"
OUT = pathlib.Path(__file__).resolve().parent.parent / "survey" / "stil-fits-plus.fits"

# Column name, unit and STILTS expression of the row number i (0 to 47).
COLUMNS = [
    ("source_id", None, "(long) i * 4398046511104L + 1729L"),
    ("ra", "deg", "(i * 137.50776405003785) % 360"),
    ("dec", "deg", "asinDeg(2 * ((i * 0.6180339887498949) % 1) - 1)"),
    ("parallax", "mas", "i % 3 == 0 ? NaN : 0.125 + 0.25 * i"),
    ("phot_g_mean_mag", "mag", "i % 5 == 0 ? (float) NaN : (float) (12 + 0.75 * (i % 7))"),
    ("n_obs", None, "i % 4 == 0 ? NULL : 3 * i"),
    ("n_bad", None, "(short) (i - 24)"),
    ("duplicated", None, "i % 2 == 0"),
    ("name", None, '"STIL-" + (1000 + i)'),
]


def main():
    version = subprocess.run(["stilts", "-version"], capture_output=True, text=True, check=True).stdout
    if f"STILTS version {STILTS}" not in version:
        sys.exit(f"needs STILTS {STILTS}, found:\n{version}")
    args = ["stilts", "tpipe", "in=:loop:48"]
    for name, unit, expr in COLUMNS:
        units = ["-units", unit] if unit else []
        args.append("cmd=" + " ".join(["addcol", *units, name, quote(expr)]))
    args += ["cmd=delcols i", "ofmt=fits-plus", f"out={OUT}"]
    subprocess.run(args, check=True)
    print(f"{OUT.name}: {OUT.stat().st_size} bytes")


def quote(expr):
    """[expr] as one word of a STILTS filter command."""
    return "'" + expr.replace("\\", "\\\\").replace("'", "\\'") + "'"


main()
