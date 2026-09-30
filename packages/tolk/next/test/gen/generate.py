# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate the goldens of tolk.next from tinygrad.

    uv run packages/tolk/next/test/gen/generate.py [--check] [MODULE...]

Each generator file `gen/<path>.py` runs in a fresh interpreter against
the tinygrad checkout, which must be clean and at TINYGRAD. MODULE, such as
`dtype` or `uop/op`, limits the run to those generators; by default all run.

Without --check, the goldens and the manifest are written, goldens that a
generator no longer declares are removed, and each change is listed: review
them with `git diff` before committing. With --check nothing is written: each
golden that would change is printed as a diff, and the run fails if any would.
"""

import argparse
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

TINYGRAD = "79af1ca70e7021f504919c4ff5631245acc33ed6"

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
MANIFEST = HERE / "manifest"
HEADER = f"# tinygrad {TINYGRAD}\n"

# The child runs the generator, then makes each golden in a process of its own,
# forked from the generator's, so that no golden sees the state another left in
# tinygrad (buffer numbering, caches). It writes the goldens to a file, out of
# reach of anything tinygrad prints.
CHILD = """
import json, os, runpy, sys, golden
runpy.run_path(sys.argv[1])
goldens = []
for name, body in golden.GOLDENS:
    read, write = os.pipe()
    if (pid := os.fork()) == 0:
        os.close(read)
        with os.fdopen(write, "w") as out: out.write(body())
        os._exit(0)
    os.close(write)
    with os.fdopen(read) as out: text = out.read()
    if os.waitpid(pid, 0)[1] != 0: sys.exit(f"golden {name} failed")
    goldens.append((name, text))
with open(sys.argv[2], "w") as out:
    json.dump(goldens, out)
"""


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def default_tinygrad():
    # The checkout sits in the main working tree, which every worktree shares.
    common = git(HERE, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common).parent / "_tinygrad_next"


def check_checkout(tinygrad):
    head = git(tinygrad, "rev-parse", "HEAD")
    if head != TINYGRAD:
        sys.exit(f"{tinygrad} is at {head}, not at {TINYGRAD}")
    if git(tinygrad, "status", "--porcelain", "--untracked-files=no"):
        sys.exit(f"{tinygrad} has local changes")


def generators(modules):
    found = {path.relative_to(HERE).with_suffix("").as_posix(): path
             for path in sorted(HERE.rglob("*.py"))
             if path.parent != HERE or path.name not in ("generate.py", "golden.py", "graph.py")}
    unknown = [module for module in modules if module not in found]
    if unknown:
        sys.exit(f"no generator for {', '.join(unknown)}; generators: {', '.join(found)}")
    return {module: found[module] for module in (modules or found)}


def run(module, generator, tinygrad):
    """The files of the goldens `generator` declares, by path under test/."""
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG") if key in os.environ}
    # Scrub everything tinygrad reads (DEBUG, DEV, BEAM, ...) and fix string
    # hashing, so that a golden depends on tinygrad alone.
    env.update(PYTHONPATH=os.pathsep.join([str(tinygrad), str(HERE)]), PYTHONHASHSEED="0",
               PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
    with tempfile.TemporaryDirectory(prefix="tolk-next-gen-") as scratch:
        out = Path(scratch) / "goldens.json"
        result = subprocess.run([sys.executable, "-c", CHILD, str(generator), str(out)], cwd=scratch,
                                env=env, capture_output=True, text=True)
        if result.returncode:
            sys.exit(f"gen/{module}.py failed:\n{result.stdout}{result.stderr}")
        declared = json.loads(out.read_text())
    files = {}
    for name, body in declared:
        path = f"{module}/{name}.golden"
        if path in files:
            sys.exit(f"gen/{module}.py declares {name} twice")
        files[path] = HEADER + body + "\n"
    if not files:
        sys.exit(f"gen/{module}.py declares no golden")
    return files


def read(path):
    return path.read_text(encoding="utf-8") if path.is_file() else None


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="write nothing; fail if a golden would change")
    parser.add_argument("--tinygrad", type=Path, default=None, help="the tinygrad checkout")
    parser.add_argument("modules", nargs="*", metavar="MODULE")
    args = parser.parse_args()
    tinygrad = (args.tinygrad or default_tinygrad()).resolve()
    check_checkout(tinygrad)
    selected = generators(args.modules)
    manifest = read(MANIFEST) or HEADER
    if args.modules and not manifest.startswith(HEADER):
        sys.exit("the goldens are from another tinygrad commit: regenerate them all")
    goldens = {}
    for module, generator in selected.items():
        goldens.update(run(module, generator, tinygrad))
    # A golden belongs to the generator of its directory. A full run owns every
    # golden, including those of a generator that no longer exists.
    recorded = set(manifest.splitlines()[1:])
    removed = sorted(path for path in recorded if path not in goldens
                     and (not args.modules or Path(path).parent.as_posix() in selected))
    kept = recorded - set(removed)
    files = {path: body for path, body in goldens.items() if read(TEST / path) != body}
    listed = HEADER + "".join(f"{path}\n" for path in sorted(kept | set(goldens)))
    if args.check:
        for path, body in files.items():
            before = read(TEST / path) or ""
            sys.stdout.writelines(difflib.unified_diff(
                before.splitlines(keepends=True), body.splitlines(keepends=True),
                fromfile=f"committed/{path}", tofile=f"generated/{path}"))
        for path in removed:
            print(f"would remove {path}")
        if listed != manifest:
            print("would rewrite gen/manifest")
        if files or removed or listed != manifest:
            sys.exit(1)
        print(f"{len(goldens)} goldens match tinygrad {TINYGRAD}")
        return
    for path, body in files.items():
        write(TEST / path, body)
        print(f"wrote {path}")
    for path in removed:
        (TEST / path).unlink(missing_ok=True)
        print(f"removed {path}")
    write(MANIFEST, listed)
    if not files and not removed:
        print(f"no golden changed ({len(goldens)} generated)")


if __name__ == "__main__":
    main()
