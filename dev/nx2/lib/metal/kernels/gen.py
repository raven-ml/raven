#!/usr/bin/env python3
"""Compiles nx.metal's kernels ahead of time into the metallib nx.metal
embeds, and the Metal suite's harness kernels into the metallib the suite
and the bench load: one metallib per artifact.

Run from the repository root, on a Mac with the pinned Metal toolchain:

  uv run dev/nx2/lib/metal/kernels/gen.py
  uv run dev/nx2/lib/metal/kernels/gen.py --check
  uv run dev/nx2/lib/metal/kernels/gen.py --pin

An artifact is every .metal file of its sources' directory, each compiled
to AIR and linked by metallib. AIR is compiled for the GPU when a pipeline
is made, so one metallib serves every Apple GPU. The sources compile in a
directory of their own beside copies of the pinned headers, the only ones
they can include, so the output depends on the pinned inputs alone.

pins.json records the toolchain (the compiler's version, and the digest of
each program that runs: the metal and metallib drivers, the compiler and
the linker they start), the options, and the digest of every input (the
sources, the headers, this script) and of every output, each path from
dev/nx2. A digest is BLAKE2b with 32 bytes of output, which OCaml's
Digest.BLAKE256 computes. The script refuses another toolchain; --pin moves
the pin and regenerates everything. --check regenerates into a temporary
directory and fails if a committed file differs: the compiler is
deterministic for a version and options. A dune rule checks the digests on
every machine, with no toolchain.

Each generation checks that an artifact's functions are exactly the kernels
its list names (the X macro of its header).

Python's standard library only.
"""

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

NX2 = pathlib.Path(__file__).resolve().parents[3]
PINS = NX2 / "lib/metal/kernels/pins.json"

# Each artifact: the directory of its sources, its metallib, and the
# header whose X macro lists its kernels, with the macro's name. nx.metal's
# own joins with its first kernel.
ARTIFACTS = [
    ("test/metal/support", "test/metal/support/harness.metallib",
     "test/metal/support/harness.h", "NX_HARNESS_KERNELS"),
]

HEADERS = ["lib/array/nx_dtype.h", "lib/metal/kernels.h", "test/metal/support/harness.h"]

# No reassociation, no contraction of a product into a sum, and the
# precise float32 functions; the AIR targets the oldest macOS rig.metal
# opens a device on.
OPTIONS = ["-std=metal3.1", "-mmacosx-version-min=15.0", "-fmetal-math-mode=safe",
           "-fmetal-math-fp32-functions=precise", "-ffp-contract=off", "-Wall", "-Werror"]


def digest(path):
    return hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()


def run(*args, cwd=None):
    p = subprocess.run(["xcrun", "-sdk", "macosx", *args], capture_output=True, text=True, cwd=cwd)
    if p.returncode != 0:
        sys.exit(p.stderr)
    return p.stdout


def toolchain():
    """The compiler's version and the digest of every program that runs."""
    out = run("metal", "--version")
    installed = pathlib.Path(re.search(r"^InstalledDir: (.*)$", out, re.M).group(1))
    programs = {"metal": pathlib.Path(run("-f", "metal").strip()),
                "metallib": pathlib.Path(run("-f", "metallib").strip()).resolve(),
                "compiler": installed / "metal", "linker": installed / "air-lld"}
    return {"version": out.splitlines()[0],
            "programs": {name: digest(p) for name, p in programs.items()}}


def inputs():
    files = [NX2 / h for h in HEADERS] + [pathlib.Path(__file__).resolve()]
    for d, _, _, _ in ARTIFACTS:
        files += sorted((NX2 / d).glob("*.metal"))
    return {str(f.relative_to(NX2)): digest(f) for f in sorted(set(files))}


def kernels(header, macro):
    """The names the X macro [macro] of [header] lists."""
    text = (NX2 / header).read_text()
    m = re.search(rf"#define {macro}\(X\)((?:.*\\\n)*.*)", text)
    if m is None:
        sys.exit(f"{header}: no {macro}")
    return sorted(re.findall(r"X\((\w+)\)", m.group(1)))


def compile_artifact(directory, out):
    """Compiles the .metal files of [directory] and links them to [out]."""
    with tempfile.TemporaryDirectory() as d:
        work = pathlib.Path(d)
        for h in HEADERS:
            shutil.copy(NX2 / h, work)
        airs = []
        for src in sorted((NX2 / directory).glob("*.metal")):
            shutil.copy(src, work)
            run("metal", *OPTIONS, "-I.", "-c", src.name, "-o", src.stem + ".air", cwd=work)
            airs.append(src.stem + ".air")
        run("metallib", *airs, "-o", "out.metallib", cwd=work)
        shutil.copy(work / "out.metallib", out)


def generate(outdir):
    """Writes every metallib under [outdir]; their digests by path."""
    outputs = {}
    for directory, path, header, macro in ARTIFACTS:
        out = outdir / path
        out.parent.mkdir(parents=True, exist_ok=True)
        compile_artifact(directory, out)
        functions = sorted(l.split()[-1] for l in run("metal-nm", str(out)).splitlines() if " T " in l)
        if functions != kernels(header, macro):
            sys.exit(f"{path}: functions {functions}, expected {kernels(header, macro)}")
        outputs[path] = digest(out)
    return dict(sorted(outputs.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if a committed file differs")
    ap.add_argument("--pin", action="store_true", help="move the toolchain pin and regenerate")
    args = ap.parse_args()
    tool = toolchain()
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    if not args.pin and pins.get("toolchain") != tool:
        sys.exit(f"toolchain {tool} is not the pinned {pins.get('toolchain')}; --pin moves the pin")
    if args.check:
        if inputs() != pins.get("inputs"):
            sys.exit("an input differs from its pin: regenerate")
        with tempfile.TemporaryDirectory() as d:
            outputs = generate(pathlib.Path(d))
        for path, dg in outputs.items():
            if not (NX2 / path).exists() or digest(NX2 / path) != dg:
                sys.exit(f"{path} differs from what gen.py generates")
        if outputs != pins.get("outputs"):
            sys.exit("pins.json's outputs differ from what gen.py generates")
        print("up to date")
        return
    outputs = generate(NX2)
    PINS.write_text(json.dumps({"toolchain": tool, "options": OPTIONS, "inputs": inputs(),
                                "outputs": outputs}, indent=1) + "\n")
    for path in outputs:
        print(f"{path}: {(NX2 / path).stat().st_size} bytes")


if __name__ == "__main__":
    main()
