# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nvidia-cuda-nvcc==13.4.92",
#   "nvidia-cuda-crt==13.4.92",
#   "nvidia-cuda-runtime==13.4.92",
#   "nvidia-nvvm==13.4.92",
# ]
# ///
"""Compile nx.cuda's kernels and the CUDA suite's harness to their cubins,
and write kernels.ml, kernels.h as the host plans with it.

    uv run packages/nx/lib/cuda/kernels/gen.py [--check]

uv installs the pinned toolchain, one CUDA release: ptxas fails on a mixed
one. Each cubin is compiled for sm_89 with nx.cuda's flags, then loses
`.debug_frame` and its relocations, call-frame records for a debugger that
neither loader reads. nvcc's output is the same bytes on every run, from any
directory, so with --check nothing is written: the run compiles both cubins
and kernels.ml and fails if any differs from the committed one.

Each cubin also carries an empty kernel named after its sources' digest,
which the dune build checks (fresh.ml) in every tree, with or without CUDA.
The digest covers this script and kernels.ml's sources, so a change to them
needs a regenerated cubin, and with it kernels.ml.

kernels.ml states each fact of kernels.h as clang reads it: clang's
preprocessor expands the X-macros and lists the macros, its syntax tree
names the enumerations' constants and the structs' fields, and a program
it compiles prints their values, sizes and offsets on this 64-bit host,
whose layout of the structs is the device's.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import sysconfig
import tempfile

nx = Path(__file__).resolve().parents[3]

# The contract on floats: no product contracted into a sum, division and
# square roots correctly rounded, subnormals kept. A cubin holds no PTX,
# which the driver would compile at load for a GPU no machine here measured.
FLAGS = ["-cubin", "-arch=sm_89", "-std=c++17", "-O3", "--fmad=false",
         "-prec-div=true", "-prec-sqrt=true", "-ftz=false",
         "-Werror", "all-warnings"]

INCLUDES = ["lib/array", "lib/kernel", "lib/cuda"]

# Each cubin: its path, the file nvcc compiles, and the sources its stamp
# covers, in the order of the dune rule that checks it.
CUBINS = [
    ("lib/cuda/kernels/sm_89.cubin", "lib/cuda/kernels/src/kernels.cu", [
        "lib/cuda/kernels/gen.py",
        "lib/cuda/kernels.h",
        "lib/array/nx_dtype.h",
        "lib/kernel/nx_kernel.h",
        "lib/kernel/nx_kinds.h",
        "lib/kernel/nx_kinds_real.h",
        "lib/kernel/nx_kinds_complex.h",
        "lib/cuda/kernels/src/kernels.cu",
        "lib/cuda/kernels/src/contract.cu",
        "lib/cuda/kernels/src/combine.cuh",
        "lib/cuda/kernels/src/mma.cuh",
        "lib/cuda/kernels/src/kinds.cuh",
        "lib/cuda/kernels/src/words.cuh",
        "lib/cuda/kernels/src/loop.cu",
        "lib/cuda/kernels/src/factor.cu",
        "lib/cuda/kernels/src/spectral.cu",
        "lib/cuda/kernels/src/sp_common.cuh",
        "lib/cuda/kernels/src/sp_block.cuh",
        "lib/cuda/kernels/src/sp_jacobi.cuh",
        "lib/cuda/kernels/src/sp_tridiag.cuh",
        "lib/cuda/kernels/src/sp_stedc.cuh",
        "lib/cuda/kernels/src/sp_bd.cuh",
        "lib/cuda/kernels/src/sp_eig.cuh",
        "lib/cuda/kernels/src/moves.cu",
        "lib/cuda/kernels/src/sort.cu",
        "lib/cuda/kernels/src/fft.cu",
        "lib/cuda/kernels/src/attention.cu",
    ]),
    ("test/cuda/support/sm_89.cubin", "test/cuda/support/harness.cu", [
        "lib/cuda/kernels/gen.py",
        "test/cuda/support/harness.h",
        "lib/array/nx_dtype.h",
        "test/cuda/support/harness.cu",
    ]),
]

STRIP = {".debug_frame", ".rel.debug_frame"}

# ELF64 constants.
SHT_NOBITS = 8
E_PHOFF, E_SHOFF, E_PHENTSIZE, E_SHENTSIZE = 0x20, 0x28, 0x36, 0x3a
EHDR_SIZE = 64
# Every section and table keeps its alignment if bytes move by this.
STEP = 256


def stamp(sources):
    """The kernel named after the sources' digest: an MD5 of the lines of
    each file's hex MD5, as fresh.ml computes it."""
    lines = "".join(hashlib.md5((nx / s).read_bytes()).hexdigest() + "\n"
                    for s in sources)
    digest = hashlib.md5(lines.encode()).hexdigest()
    return f'extern "C" __global__ void nx_cuda_sources_{digest}() {{}}\n'


def strip(blob):
    """[blob] with the STRIP sections emptied: size 0, header kept, so no
    section or symbol index moves. The bytes after an emptied section move
    back by the largest multiple of STEP it held. nvcc places these
    sections outside every segment, so no segment's size changes."""
    shoff, = struct.unpack_from("<Q", blob, E_SHOFF)
    phoff, = struct.unpack_from("<Q", blob, E_PHOFF)
    phentsize, phnum = struct.unpack_from("<HH", blob, E_PHENTSIZE)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, E_SHENTSIZE)
    heads = [struct.unpack_from("<IIQQQQ", blob, shoff + i * shentsize)
             for i in range(shnum)]
    names = heads[shstrndx][4]

    def name(at):
        return blob[names + at:blob.index(b"\0", names + at)].decode()

    dropped = [name(h[0]) in STRIP and h[1] != SHT_NOBITS for h in heads]
    cuts = [(h[4], h[5]) for h, d in zip(heads, dropped) if d]

    def moved(offset):
        return offset - sum(size // STEP * STEP for at, size in cuts
                            if at + size <= offset)

    out = bytearray(moved(len(blob)))
    out[:EHDR_SIZE] = blob[:EHDR_SIZE]
    for (_, kind, _, _, offset, size), d in zip(heads[1:], dropped[1:]):
        if kind != SHT_NOBITS and not d:
            out[moved(offset):moved(offset) + size] = blob[offset:offset + size]
    for i in range(phnum):
        at = phoff + i * phentsize
        head = bytearray(blob[at:at + phentsize])
        offset, = struct.unpack_from("<Q", head, 8)
        struct.pack_into("<Q", head, 8, moved(offset))
        to = moved(phoff) + i * phentsize
        out[to:to + phentsize] = head
    for i, ((_, _, _, _, offset, _), d) in enumerate(zip(heads, dropped)):
        at = shoff + i * shentsize
        head = bytearray(blob[at:at + shentsize])
        if i:
            struct.pack_into("<Q", head, 24, moved(offset))
        if d:
            struct.pack_into("<Q", head, 32, 0)
        to = moved(shoff) + i * shentsize
        out[to:to + shentsize] = head
    struct.pack_into("<Q", out, E_PHOFF, moved(phoff))
    struct.pack_into("<Q", out, E_SHOFF, moved(shoff))
    return bytes(out)


def compile_cubin(nvcc, source, sources, directory):
    header = Path(directory) / "stamp.h"
    header.write_text(stamp(sources))
    out = Path(directory) / "out.cubin"
    includes = [arg for d in INCLUDES for arg in ("-I", str(nx / d))]
    subprocess.run([str(nvcc), *FLAGS, *includes, "-include", str(header),
                    "-o", str(out), str(nx / source)], check=True)
    return strip(out.read_bytes())


# kernels.ml

KERNELS_ML = "lib/cuda/kernels.ml"
HEADER = nx / "lib/cuda/kernels.h"
CLANG = ["clang", "-x", "c", "-std=c11", "-I", str(nx / "lib/kernel")]

# The arguments the host reads of a family's rows, in order: int, or a
# variant type whose constructors are the identifiers the rows give,
# capitalised. A family with a record's name takes a record of these
# fields. Another family takes each argument as an int or a string.
FAMILIES = {
    "GEMM": ("gemm", [("t", "ct"), ("bm", "int"), ("bn", "int"),
                      ("bk", "int"), ("tm", "int"), ("tn", "int"),
                      ("lm", "int"), ("wm", "int"), ("wn", "int"),
                      ("minb", "int"), ("occ", "int"), ("load", "load")]),
    "GEMV": (None, [("t", "ct")]),
    "MMA": (None, [("kind", "kind"), ("a", "axis"), ("b", "axis"),
                   ("tile", "tile")]),
}

# NX_CUDA_TILES's columns after the tile's name, the record shape.
SHAPE = ["bm", "bn", "bkb", "wm", "wn", "stages"]

KEYWORDS = set("""and as assert begin class constraint do done downto else end
    exception external false for fun function functor if in include inherit
    initializer lazy let match method module mutable new nonrec object of open
    or private rec sig struct then to true try type val virtual when while
    with mod land lor lxor lsl lsr asr effect""".split())


PREAMBLE = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generated by kernels/gen.py from kernels.h; do not edit. Regenerate it
   from the repository root with
   [uv run packages/nx/lib/cuda/kernels/gen.py].

   kernels.h as the host plans with it: the cubin's kernels, each a name and
   an instance, in NX_CUDA_KERNELS's order; the mma tiles of NX_CUDA_TILES;
   each NX_ macro of an integer and each enumeration's constant,
   NX_LOOP_WORDS as [loop_words]; each struct's size and its fields' byte
   offsets, pack_params's as [Pack_params]; and the ops NX_ATT_KINDS lists.
   Of the enumerations, NX_CUDA_KERNELS's is [kernels]'s indices. *)
"""


def run(args, **kw):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          **kw).stdout


def is_header(f):
    return f is not None and Path(f).resolve() == HEADER


def macros():
    """kernels.h's object-like macros of an integer, each (line, name): a
    body whose identifiers are all NX_ macros or constants."""
    out, file, line = [], None, 0
    for text in run([*CLANG, "-E", "-dD", str(HEADER)]).splitlines():
        marker = re.match(r'# (\d+) "(.*)"', text)
        if marker:
            line, file = int(marker[1]), marker[2]
            continue
        define = re.match(r"#define (NX_\w+) (.+)", text)
        if define and is_header(file):
            names = re.findall(r"(?<!\w)[A-Za-z_]\w*", define[2])
            if all(n.startswith("NX_") for n in names):
                out.append((line, define[1]))
        line += 1
    return out


def declarations():
    """kernels.h's anonymous enumerations' constants, each (line, name), and
    its structs, each (line, name, fields), from clang's syntax tree. Its
    JSON states a location's file and line where they change from the last
    location it wrote, so a walk in its order tracks them."""
    tree = json.loads(run([*CLANG, "-Xclang", "-ast-dump=json",
                           "-fsyntax-only", str(HEADER)]))
    at, where = {"file": None, "line": 0}, {}

    def walk(node):
        if isinstance(node, list):
            for x in node:
                walk(x)
        elif isinstance(node, dict):
            for key, x in node.items():
                if key in ("file", "line"):
                    at[key] = x
                elif key != "includedFrom":
                    walk(x)
                if key == "loc":
                    where[node["id"]] = (at["file"], at["line"])

    walk(tree)
    constants, structs, records = [], [], {}
    for node in tree["inner"]:
        file, line = where[node["id"]]
        if not is_header(file):
            continue
        if node["kind"] == "EnumDecl" and "name" not in node:
            constants += [(where[c["id"]][1], c["name"]) for c in node["inner"]]
        elif node["kind"] == "RecordDecl":
            records[node["id"]] = [f["name"] for f in node["inner"]
                                   if f["kind"] == "FieldDecl"]
        elif node["kind"] == "TypedefDecl":
            tag = node["inner"][0].get("ownedTagDecl", {}).get("id")
            if tag in records:
                structs.append((line, node["name"], records[tag]))
    return constants, structs


def expand(macro):
    """The rows of the X-macro [macro], each a list of its arguments."""
    source = (f'#include "kernels.h"\n#define ROW(...) NX_ROW_ __VA_ARGS__ '
              f'NX_ROW_END_\n{macro}(ROW)\n')
    text = run([*CLANG, "-E", "-P", "-I", str(HEADER.parent), "-"],
               input=source)
    return [[a.strip() for a in row.split(",")]
            for row in re.findall(r"NX_ROW_ (.*?) NX_ROW_END_", text, re.S)]


def values(names, structs):
    """The values of [names], then each of [structs]' size and its fields'
    offsets, then NX_ATT_KINDS's ops, printed by a program clang compiles."""
    lines = [f'printf("%lld\\n", (long long)({n}));' for n in names]
    for _, s, fields in structs:
        lines.append(f'printf("%zu\\n", sizeof({s}));')
        lines += [f'printf("%zu\\n", offsetof({s}, {f}));' for f in fields]
    source = "\n".join([
        "#include <stdio.h>", '#include "nx_kernel.h"', '#include "kernels.h"',
        '#define KIND(S, T, A, U, n) printf("%d\\n", NX_ATT_OP(S, A, U));',
        "int main(void) {", *lines, "NX_ATT_KINDS(KIND)", "return 0;", "}", ""])
    with tempfile.TemporaryDirectory() as directory:
        exe = Path(directory) / "values"
        subprocess.run([*CLANG, "-I", str(HEADER.parent), "-o", str(exe), "-"],
                       input=source, text=True, check=True)
        return [int(x) for x in run([str(exe)]).split()]


def constant_name(name):
    """The OCaml value of the constant [name]: NX_LOOP_WORDS is loop_words."""
    lower = name.removeprefix("NX_").lower()
    return lower + "_" if lower in KEYWORDS else lower


def field_name(name):
    """The OCaml value of the field [name]: lowercase, with _ after a keyword
    or a name that lowercasing changes, so that P and p stay apart."""
    lower = name.lower()
    return lower + "_" if lower in KEYWORDS or lower != name else lower


def constructor(name):
    return name[0].upper() + name[1:].lower()


def number(a):
    return re.fullmatch(r"-?\d+", a) is not None


def kernels_ml():
    """kernels.ml's text."""
    rows, tiles = expand("NX_CUDA_KERNELS"), expand("NX_CUDA_TILES")
    constants, structs = declarations()
    named = macros() + constants
    numbers = iter(values([n for _, n in named], structs))
    value = {n: next(numbers) for _, n in named}
    layout = {s: [next(numbers) for _ in range(len(fs) + 1)]
              for _, s, fs in structs}

    # Each family's rows, and its arguments: FAMILIES's, or each column an
    # int or a string.
    families = {}
    for name, family, *args in rows:
        families.setdefault(family, []).append(args)
    fields = {}
    for family, rows_of in families.items():
        if family in FAMILIES:
            fields[family] = FAMILIES[family][1]
        else:
            columns = list(zip(*rows_of))
            fields[family] = [(None, "int" if all(map(number, c)) else "string")
                              for c in columns]
        if any(len(args) != len(fields[family]) for args in rows_of):
            sys.exit(f"kernels.h: {family} rows of other arities than gen.py's")

    # Each variant type's constructors in the rows' order, the tiles first.
    variants = {"tile": [t[0] for t in tiles]}
    for family, rows_of in families.items():
        for args in rows_of:
            for (_, ty), a in zip(fields[family], args):
                if ty == "tile" and a not in variants["tile"]:
                    sys.exit(f"kernels.h: no tile {a} in NX_CUDA_TILES")
                if ty not in ("int", "string") and a not in variants.setdefault(ty, []):
                    variants[ty].append(a)

    def arg(ty, a):
        if ty == "int":
            return a
        return f'"{a}"' if ty == "string" else constructor(a)

    def instance(family, args):
        c, fs = constructor(family), fields[family]
        if not fs:
            return c
        record = FAMILIES.get(family, (None,))[0]
        if record:
            return c + " { " + "; ".join(
                f"{f} = {arg(ty, a)}" for (f, ty), a in zip(fs, args)) + " }"
        xs = [arg(ty, a) for (_, ty), a in zip(fs, args)]
        return f"{c} {xs[0]}" if len(xs) == 1 else f"{c} ({', '.join(xs)})"

    out = [PREAMBLE, "(* Kernels *)\n"]
    out += [f"type {ty} = {' | '.join(map(constructor, cs))}"
            for ty, cs in variants.items()]
    decls = []
    for family in families:
        c, fs = constructor(family), fields[family]
        record = FAMILIES.get(family, (None,))[0]
        if record:
            body = "".join(f"  {f} : {ty};\n" for f, ty in fs)
            out.append(f"\ntype {record} = {{\n{body}}}")
            decls.append(f"  | {c} of {record}")
        elif fs:
            decls.append(f"  | {c} of {' * '.join(ty for _, ty in fs)}")
        else:
            decls.append(f"  | {c}")
    out.append("\ntype instance =\n" + "\n".join(decls) + "\n")
    out.append("let kernels =\n  [|")
    out += [f'    ("{name}", {instance(family, args)});'
            for name, family, *args in rows]
    out.append("  |]\n\n(* Tiles *)\n")
    out.append("type shape = { " + "; ".join(f"{f} : int" for f in SHAPE) + " }")
    out.append("let tiles = [| " + "; ".join(constructor(t[0]) for t in tiles) + " |]\n")
    out.append("let shape = function")
    for name, *columns in tiles:
        if len(columns) != len(SHAPE):
            sys.exit(f"kernels.h: tile {name} has {len(columns)} columns")
        body = "; ".join(f"{f} = {c}" for f, c in zip(SHAPE, columns))
        out.append(f"  | {constructor(name)} -> {{ {body} }}")
    out.append("\n(* Constants and structs *)\n")

    items = [(line, name, None) for line, name in named]
    items += [(line, name, fs) for line, name, fs in structs]
    seen = set()
    for _, name, fs in sorted(items, key=lambda x: x[0]):
        if fs is None:
            ocaml = constant_name(name)
            if ocaml in seen:
                sys.exit(f"kernels.h: two constants named {ocaml}")
            seen.add(ocaml)
            v = value[name]
            out.append(f"let {ocaml} = {v if v >= 0 else f'({v})'}")
            continue
        names = [field_name(f) for f in fs]
        if "size" in names or len(set(names)) != len(names):
            sys.exit(f"kernels.h: {name}'s fields clash as OCaml values")
        size, *offsets = layout[name]
        out.append(f"\nmodule {name[0].upper() + name[1:]} = struct")
        out.append(f"  let size = {size}")
        out += [f"  let {n} = {at}" for n, at in zip(names, offsets)]
        out.append("end\n")
    out.append("\n(* Attention's ops *)\n")
    out.append("let att_kinds = [ " + "; ".join(map(str, numbers)) + " ]")
    return "\n".join(out) + "\n"


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--check", action="store_true",
                    help="fail if a committed file differs; write nothing")
args = parser.parse_args()
nvcc = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/bin/nvcc"
stale = []
for cubin, source, sources in CUBINS:
    with tempfile.TemporaryDirectory() as directory:
        blob = compile_cubin(nvcc, source, sources, directory)
    path = nx / cubin
    if path.exists() and path.read_bytes() == blob:
        continue
    stale.append(cubin)
    if not args.check:
        path.write_bytes(blob)
text = kernels_ml()
path = nx / KERNELS_ML
if not (path.exists() and path.read_text() == text):
    stale.append(KERNELS_ML)
    if not args.check:
        path.write_text(text)
for file in stale:
    verb = "differs" if args.check else "written"
    print(f"packages/nx/{file} {verb}")
if args.check and stale:
    sys.exit("regenerate with: uv run packages/nx/lib/cuda/kernels/gen.py")
