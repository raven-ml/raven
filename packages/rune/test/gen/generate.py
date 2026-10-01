# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "numpy"]
# ///
"""Generate the graph-parity goldens of rune's lowering from tinygrad.

    uv run packages/rune/test/gen/generate.py [--check] [MODULE...]

Each generator `gen/<module>.py` declares its goldens with the decorators of
tolk's `gen/golden.py`, and each golden is written to
`test/golden/<module>/<name>.golden` in tolk's graph format
(`gen/graph.py`), so that both trees read and write graphs one way. The goldens are made as
tolk's are: in a copy of the checkout at TINYGRAD with tolk's patch
applied, each in a process of its own.
Without --check they are written, with the manifest; with it, nothing is
written and the run fails if a golden would change.
"""

import argparse
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
TOLK_GEN = HERE.parents[2] / "tolk" / "test" / "gen"

sys.path.insert(0, str(TOLK_GEN))
from generate import CHILD, HEADER, TINYGRAD, check_checkout, default_tinygrad, patched, read, write  # noqa: E402

MANIFEST = HERE / "manifest"


def generators(modules):
    found = {path.stem: path for path in sorted(HERE.glob("*.py")) if path.name != "generate.py"}
    unknown = [module for module in modules if module not in found]
    if unknown:
        sys.exit(f"no generator for {', '.join(unknown)}; generators: {', '.join(found)}")
    return {module: found[module] for module in (modules or found)}


def run(module, generator, tinygrad):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG") if key in os.environ}
    env.update(PYTHONPATH=os.pathsep.join([str(tinygrad), str(TOLK_GEN)]), PYTHONHASHSEED="0",
               PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
    with tempfile.TemporaryDirectory(prefix="rune-gen-") as scratch:
        out = Path(scratch) / "goldens.json"
        result = subprocess.run([sys.executable, "-c", CHILD, str(generator), str(out)], cwd=scratch,
                                env=env, capture_output=True, text=True)
        if result.returncode:
            sys.exit(f"gen/{module}.py failed:\n{result.stdout}{result.stderr}")
        return {f"golden/{module}/{name}.golden": HEADER + body + "\n" for name, body in json.loads(out.read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="write nothing; fail if a golden would change")
    parser.add_argument("modules", nargs="*", metavar="MODULE")
    args = parser.parse_args()
    tinygrad = default_tinygrad().resolve()
    check_checkout(tinygrad)
    goldens = {}
    with tempfile.TemporaryDirectory(prefix="rune-tinygrad-") as scratch:
        tree = patched(tinygrad, scratch)
        for module, generator in generators(args.modules).items():
            goldens.update(run(module, generator, tree))
    changed = {path: body for path, body in goldens.items() if read(TEST / path) != body}
    listed = HEADER + "".join(f"{path}\n" for path in sorted(goldens))
    if args.check:
        for path, body in changed.items():
            sys.stdout.writelines(difflib.unified_diff(
                (read(TEST / path) or "").splitlines(keepends=True), body.splitlines(keepends=True),
                fromfile=f"committed/{path}", tofile=f"generated/{path}"))
        if changed or (not args.modules and listed != (read(MANIFEST) or "")):
            sys.exit(1)
        print(f"{len(goldens)} goldens match tinygrad {TINYGRAD}")
        return
    for path, body in changed.items():
        write(TEST / path, body)
        print(f"wrote {path}")
    if not args.modules:
        write(MANIFEST, listed)


if __name__ == "__main__":
    main()
