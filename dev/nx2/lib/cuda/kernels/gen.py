#!/usr/bin/env python3
"""Compiles nx.cuda's kernels ahead of time into the cubins nx.cuda embeds,
and the CUDA suite's harness kernels into the cubin the suite and the bench
load: one cubin per artifact and architecture, <artifact>/<arch>.cubin.

Run from the repository root, on a machine with the pinned CUDA toolkit,
which it finds at $CUDA_HOME, else /usr/local/cuda:

  uv run dev/nx2/lib/cuda/kernels/gen.py
  uv run dev/nx2/lib/cuda/kernels/gen.py --check
  uv run dev/nx2/lib/cuda/kernels/gen.py --pin

An artifact is one translation unit that includes every .cu file of its
sources' directory in name order. nvcc compiles it to a cubin for each
architecture and nothing else: no PTX, which the driver would compile at
load for GPUs no machine here measured.

pins.json records the toolchain (nvcc's version, and the digest of each
program that reads the sources: nvcc, cudafe++, cicc, ptxas, and the host
compiler, whose preprocessor nvcc runs), the options, and the digest of
every input (the sources, the headers they include, this script) and of
every output, each path from dev/nx2. A digest is BLAKE2b with 32 bytes of
output, which OCaml's Digest.BLAKE256 computes. The script refuses another
toolchain; --pin moves the pin and regenerates everything. --check
regenerates into a temporary directory and fails if a committed file
differs: nvcc is deterministic for a version and options. A dune rule checks
the digests on every machine, with no toolchain, so a source changed
without regenerating, or a cubin edited, fails the tests.

Each generation checks that each kernel the artifact's list names (the X
macro of its header, by its first argument) is a symbol of its cubin.

Python's standard library only.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

NX2 = pathlib.Path(__file__).resolve().parents[3]
PINS = NX2 / "lib/cuda/kernels/pins.json"
ARCHS = ["sm_89"]

# Each artifact: the directory of its sources, the directory of its cubins,
# the header whose X macro lists its kernels by their first argument, and
# that macro's name.
ARTIFACTS = [
    ("lib/cuda/kernels/src", "lib/cuda/kernels", "lib/cuda/kernels.h", "NX_CUDA_KERNELS"),
    ("test/cuda/support", "test/cuda/support", "test/cuda/support/harness.h", "NX_HARNESS_KERNELS"),
]

# The headers the sources may include, by name: the only ones copied beside
# them.
HEADERS = ["lib/array/nx_dtype.h", "lib/cuda/kernels.h", "test/cuda/support/harness.h"]

# Floats as the kernel contract states them: no contraction of a product
# into a sum, division and square roots correctly rounded, subnormals kept.
OPTIONS = [
    "-cubin", "-std=c++17", "-O3", "--fmad=false", "-prec-div=true",
    "-prec-sqrt=true", "-ftz=false", "-Werror", "all-warnings",
]


def digest(path):
    return hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()


def cuda_home():
    return pathlib.Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def toolchain():
    """The toolkit's version and the digest of every program that reads the
    sources."""
    home = cuda_home()
    nvcc = home / "bin/nvcc"
    if not nvcc.exists():
        sys.exit(f"no nvcc at {nvcc}")
    gcc = pathlib.Path(shutil.which("gcc") or sys.exit("no gcc")).resolve()
    cc1plus = pathlib.Path(run([str(gcc), "-print-prog-name=cc1plus"]).strip()).resolve()
    programs = {"nvcc": nvcc, "cudafe++": home / "bin/cudafe++", "cicc": home / "nvvm/bin/cicc",
                "ptxas": home / "bin/ptxas", "gcc": gcc, "cc1plus": cc1plus}
    return {
        "nvcc": run([str(nvcc), "--version"]).strip().splitlines()[-1],
        "gcc": run([str(gcc), "-dumpfullversion"]).strip(),
        "programs": {name: digest(p) for name, p in programs.items()},
    }


def inputs():
    files = [NX2 / h for h in HEADERS] + [pathlib.Path(__file__).resolve()]
    for d, _, _, _ in ARTIFACTS:
        files += sources(d)
    return {str(f.relative_to(NX2)): digest(f) for f in sorted(set(files))}


def kernels(header, macro):
    """The names the X macro [macro] of [header] lists."""
    text = (NX2 / header).read_text()
    m = re.search(rf"#define {macro}\(X\)((?:.*\\\n)*.*)", text)
    if m is None:
        sys.exit(f"{header}: no {macro}")
    return re.findall(r"X\((\w+)", m.group(1))


def sources(directory):
    """The pinned sources of an artifact: its .cu and .cuh files."""
    d = NX2 / directory
    return sorted(d.glob("*.cu")) + sorted(d.glob("*.cuh"))


def compile_artifact(directory, arch, out):
    """Compiles the artifact of [directory] for [arch] to [out]: a unit that
    includes its .cu files, compiled in a directory of its own beside
    copies of its sources and of the pinned headers, the only files it can
    include. A header outside the pins fails to compile, and nvcc sees the
    same paths wherever the repository is."""
    with tempfile.TemporaryDirectory() as d:
        work = pathlib.Path(d)
        for f in sources(directory) + [NX2 / h for h in HEADERS]:
            shutil.copy(f, work / f.name)
        unit = "".join(f'#include "{f.name}"\n' for f in sources(directory) if f.suffix == ".cu")
        (work / "unit.cu").write_text(unit)
        nvcc = str(cuda_home() / "bin/nvcc")
        subprocess.run([nvcc, *OPTIONS, f"-arch={arch}", "-I", str(work), "-o", str(out),
                        "unit.cu"], check=True, cwd=work)


def generate(outdir):
    """Writes every cubin under [outdir]; their digests by path."""
    outputs = {}
    for directory, cubins, header, macro in ARTIFACTS:
        for arch in ARCHS:
            path = f"{cubins}/{arch}.cubin"
            out = outdir / path
            out.parent.mkdir(parents=True, exist_ok=True)
            compile_artifact(directory, arch, out)
            data = out.read_bytes()
            missing = [k for k in kernels(header, macro) if b"\0" + k.encode() + b"\0" not in data]
            if missing:
                sys.exit(f"{path}: no kernel {', '.join(missing)}")
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
