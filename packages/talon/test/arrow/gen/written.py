# /// script
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Reads talon's own Arrow files with pinned pyarrow and records what it reads.

Run from this directory with `uv run written.py [--check]`, after writing
../support/written/ with the executable gen_written. It fails if pyarrow cannot
read a file, and records per file and column the talon type and every value of
pyarrow's read, as arrow.py renders them, in ../golden/written.txt. With --check
it fails if the golden would change.
"""

import argparse
import sys
from pathlib import Path

import arrow as corpus
import pyarrow as pa

HERE = Path(__file__).parent
WRITTEN = HERE.parent / "support" / "written"
GOLDEN = HERE.parent / "golden" / "written.txt"


def main():
    pa.register_extension_type(corpus.Kg())
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    lines = []
    for path in sorted(WRITTEN.iterdir()):
        table = corpus.read(path)
        for column, name in zip(table.columns, table.column_names):
            ty = (
                corpus.dict_reads_as(column)
                if pa.types.is_dictionary(column.type)
                else corpus.talon_type(column.type)
            )
            lines.append("|".join([path.name, name, ty, " ".join(corpus.column_tokens(column))]) + "\n")
    text = "".join(lines)
    if not args.check:
        GOLDEN.write_text(text, encoding="utf-8")
    elif text != GOLDEN.read_text(encoding="utf-8"):
        sys.exit("golden/written.txt differs")


if __name__ == "__main__":
    main()
