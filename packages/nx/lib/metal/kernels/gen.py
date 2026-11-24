# /// script
# requires-python = ">=3.10"
# ///
"""Compile nx.metal's kernels, its specialised steps and the Metal suite's
harness to their metallibs, and write kernels.ml, src/kernels.h as the
host plans with it.

    uv run packages/nx/lib/metal/kernels/gen.py [--check | --sizes]

The toolchain is Xcode's Metal toolchain, through xcrun, at the version
VERSION pins: the script refuses another metal or metallib, since another
compiler gives other bytes, and may give other bits. Every unit compiles
as MSL 3.2 for macOS 15, rig's floor, with -fno-fast-math and
-ffp-contract=off (src/device.h adds the pragma that keeps products from
fusing), and metallib links a library's units into one metallib.

Before it writes a metallib, the script refuses one whose units the rules
below forbid, reading each unit's IR:

- static threadgroup memory: a group's threadgroup memory is its one
  dynamic binding, of the bytes its launch states;
- for the library and the steps, a kernel whose parameter bytes differ from
  its row of kernels.h, or whose [[max_total_threads_per_threadgroup(n)]],
  where it states one, differs from its row's threads, or a kernel the
  table lacks or a row the metallib lacks: the steps' metallib holds the
  rows of NX_FAMILY_FFT_STEPS, the library every other. A kernel that
  states no bound lets its registers set its pipeline's limit, which the
  host reads and checks against its row's threads.

It prints each kernel's bytes in the metallib, its function's module, and
each metallib's total.

Each metallib also holds an empty kernel named after its sources' digest,
which the dune build checks (fresh.ml) in every tree, without the
toolchain. The digest covers this script and the sources each metallib
reads, so a change to them needs a regenerated metallib, and with it
kernels.ml.

kernels.ml states src/kernels.h's facts as the pinned metal reads them:
its preprocessor expands the X-macro NX_METAL_KERNELS, the kernel table,
and the anonymous enumerations, whose constants are the shared constants
and the parameters' word indices; the values of every constant and of
every row's columns come from a constant array metal compiles to IR.

The stamp's sources are listed by hand, so the script refuses a unit that
includes a file outside the SDK that its metallib's stamp does not
cover, from the dependencies metal reports.

metal and metallib give the same bytes on every run and from any
directory: the script builds each metallib twice, the second time from
a copy of its sources under another directory, and refuses one whose
builds differ. --check builds every metallib and kernels.ml in memory,
writes nothing, and fails if any differs from the committed file.

--sizes compiles each kernel of the committed library for an M1 Max's
GPU, applegpu_g13s, with the toolchain's metal-tt, as Metal compiles a
pipeline at a call's first launch. It prints each kernel's bytes of GPU
code and each family's sum, writes nothing, and fails if a kernel has
more than 100 KiB. A pipeline's first compile takes about 5-13 ms a KiB
on an M1 Max, so the bound keeps a kernel's first call near a second.
It takes about 25 s and several cores.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

nx = Path(__file__).resolve().parents[3]

# Toolchain

VERSION = "32023.864"
METAL_VERSION = f"Apple metal version {VERSION} (metalfe-{VERSION})"
METALLIB_VERSION = (f"AIR-LLD {VERSION} (metalfe-{VERSION}) "
                    "(compatible with legacy metallib linker)")

# MSL 3.2 for macOS 15. The contract on floats: no fast math, and no
# product contracted into a sum. Warnings are errors.
METAL = ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2",
         "-mmacosx-version-min=15.0", "-fno-fast-math", "-ffp-contract=off",
         "-Wall", "-Werror"]
METALLIB = ["xcrun", "-sdk", "macosx", "metallib"]

INCLUDES = ["lib/metal/kernels/src", "lib/array", "lib/kernel"]

# Each metallib: its path, its units, the sources its stamp covers, in the
# order of the dune rule that checks it, and the families of kernels.h's
# rows it holds: every family but STEPS, STEPS alone, or no row (the
# harness). A unit compiles alone.
LIBRARY, STEPS = "library", "steps"
STEPS_FAMILY = "NX_FAMILY_FFT_STEPS"

METALLIBS = [
    ("lib/metal/kernels/nx_metal.metallib", [
        "lib/metal/kernels/src/copy.metal",
        "lib/metal/kernels/src/loop.metal",
        "lib/metal/kernels/src/direct.metal",
        "lib/metal/kernels/src/contract.metal",
        "lib/metal/kernels/src/fft.metal",
        "lib/metal/kernels/src/sort.metal",
        "lib/metal/kernels/src/factor.metal",
        "lib/metal/kernels/src/index.metal",
        "lib/metal/kernels/src/spectral.metal",
        "lib/metal/kernels/src/attention.metal",
    ], [
        "lib/metal/kernels/gen.py",
        "lib/metal/kernels/src/kernels.h",
        "lib/metal/kernels/src/device.h",
        "lib/metal/kernels/src/tape.h",
        "lib/metal/kernels/src/fold.h",
        "lib/metal/kernels/src/values.h",
        "lib/metal/kernels/src/copy.metal",
        "lib/metal/kernels/src/loop.metal",
        "lib/metal/kernels/src/direct.metal",
        "lib/metal/kernels/src/contract.metal",
        "lib/metal/kernels/src/fft.h",
        "lib/metal/kernels/src/fft.metal",
        "lib/metal/kernels/src/sort.metal",
        "lib/metal/kernels/src/factor.metal",
        "lib/metal/kernels/src/index.metal",
        "lib/metal/kernels/src/spectral.metal",
        "lib/metal/kernels/src/sp_common.h",
        "lib/metal/kernels/src/sp_block.h",
        "lib/metal/kernels/src/sp_jacobi.h",
        "lib/metal/kernels/src/sp_tridiag.h",
        "lib/metal/kernels/src/sp_stedc.h",
        "lib/metal/kernels/src/sp_bd.h",
        "lib/metal/kernels/src/sp_eig.h",
        "lib/metal/kernels/src/attention.metal",
        "lib/kernel/nx_kernel.h",
        "lib/kernel/nx_kinds.h",
        "lib/kernel/nx_kinds_real.h",
        "lib/kernel/nx_kinds_complex.h",
        "lib/array/nx_dtype.h",
    ], LIBRARY),
    ("lib/metal/kernels/fft_steps.metallib", [
        "lib/metal/kernels/src/fft_steps.metal",
    ], [
        "lib/metal/kernels/gen.py",
        "lib/metal/kernels/src/kernels.h",
        "lib/metal/kernels/src/device.h",
        "lib/metal/kernels/src/fft.h",
        "lib/metal/kernels/src/fft_steps.metal",
    ], STEPS),
    ("test/metal/support/harness.metallib", [
        "test/metal/support/harness.metal",
    ], [
        "lib/metal/kernels/gen.py",
        "lib/metal/kernels/src/kernels.h",
        "lib/metal/kernels/src/device.h",
        "test/metal/support/harness.metal",
    ], None),
]

KERNELS_ML = "lib/metal/kernels.ml"
HEADER = nx / "lib/metal/kernels/src/kernels.h"


def run(args, **kw):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          **kw).stdout


def pinned():
    """Refuses a metal or a metallib of another version than VERSION."""
    for tool, want in [("metal", METAL_VERSION),
                       ("metallib", METALLIB_VERSION)]:
        got = run(["xcrun", "-sdk", "macosx", tool, "--version"])
        first = got.splitlines()[0] if got else ""
        if first != want:
            sys.exit(f"{tool} is {first!r}, not {want!r}: install the Metal "
                     f"toolchain {VERSION} (Xcode 26.3)")


def includes(root):
    return [arg for d in INCLUDES for arg in ("-I", str(root / d))]


# Stamps


def digest(sources):
    """An MD5 of the lines of each file's hex MD5, as fresh.ml computes it."""
    lines = "".join(hashlib.md5((nx / s).read_bytes()).hexdigest() + "\n"
                    for s in sources)
    return hashlib.md5(lines.encode()).hexdigest()


def stamp_name(sources):
    return f"nx_metal_sources_{digest(sources)}"


def stamp(sources):
    """A unit of one empty kernel named after the sources' digest."""
    return f"[[kernel]] void {stamp_name(sources)}() {{}}\n"


# Building


def depends(path, root):
    """The files under [root] that the make rule at [path] names as its
    target's prerequisites, relative to [root]."""
    text = path.read_text().replace("\\\n", " ")
    _, _, prerequisites = text.partition(": ")
    root = root.resolve()
    out = set()
    for name in prerequisites.split():
        p = Path(name).resolve()
        if p.is_relative_to(root):
            out.add(p.relative_to(root).as_posix())
    return out


def build(units, sources, directory, root=nx):
    """The metallib of [units], read under [root], and the stamp of
    [sources]; each unit's textual IR; and the files under [root] the
    units include, themselves among them."""
    d = Path(directory)
    stamp_unit = d / "stamp.metal"
    stamp_unit.write_text(stamp(sources))
    airs, irs, closure = [], [], set()
    for unit in [*map(lambda u: root / u, units), stamp_unit]:
        air = d / f"{unit.stem}.air"
        deps = d / f"{unit.stem}.d"
        subprocess.run([*METAL, *includes(root), "-MD", "-MF", str(deps),
                        "-c", str(unit), "-o", str(air)], check=True)
        closure |= depends(deps, root)
        airs.append(str(air))
        irs.append(run([*METAL, *includes(root), "-S", "-emit-llvm", "-o",
                        "-", str(unit)]))
    lib = d / "out.metallib"
    subprocess.run([*METALLIB, *airs, "-o", str(lib)], check=True)
    return lib.read_bytes(), irs, closure


def copied(files, directory):
    """A root under [directory] holding [files], read under nx, at their
    paths, with every include directory."""
    root = Path(directory) / "elsewhere" / "nx"
    for d in INCLUDES:
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(nx / f, root / f)
    return root


# The metallib and the IR


def tags(lib):
    """Each function of the metallib [lib]: its tags, by tag. A
    metallib's header is followed by its function list: a count, then for
    each function its byte length and its tags, a 4-byte name and a 16-bit
    length before each value, until ENDT."""
    if lib[:4] != b"MTLB":
        sys.exit("not a metallib")
    list_at, = struct.unpack_from("<Q", lib, 24)
    count, = struct.unpack_from("<I", lib, list_at)
    at = list_at + 4
    out = []
    for _ in range(count):
        length, = struct.unpack_from("<I", lib, at)
        i, end = at + 4, at + length
        fields = {}
        while i < end:
            tag = lib[i:i + 4]
            if tag == b"ENDT":
                break
            n, = struct.unpack_from("<H", lib, i + 4)
            fields[tag] = lib[i + 6:i + 6 + n]
            i += 6 + n
        out.append(fields)
        at = end
    return out


def functions(lib):
    """Each function of the metallib [lib], by name: the bytes of its
    module."""
    out = {}
    for fields in tags(lib):
        if b"NAME" not in fields or b"MDSZ" not in fields:
            sys.exit("a function of the metallib has no name or size")
        name = fields[b"NAME"].rstrip(b"\0").decode()
        out[name], = struct.unpack_from("<Q", fields[b"MDSZ"])
    return out


def hashes(lib):
    """Each function of the metallib [lib], by name: its module's hash, in
    upper-case hex, as metal-objdump prints a module's."""
    return {fields[b"NAME"].rstrip(b"\0").decode(): fields[b"HASH"].hex().upper()
            for fields in tags(lib) if b"NAME" in fields and b"HASH" in fields}


def kernels_of(ir):
    """Each kernel of the unit's IR [ir]: (threads, parameter bytes), where
    threads is None for a kernel without its attribute and parameter bytes
    None for a kernel with no buffer 0."""
    meta = dict(re.findall(r"^(![0-9]+) = (.*)$", ir, re.M))
    entries = re.search(r"^!air\.kernel = !\{(.*)\}$", ir, re.M)
    out = {}
    for ref in re.findall(r"![0-9]+", entries[1]) if entries else []:
        node = meta[ref]
        name = re.search(r"@([A-Za-z_0-9]+)", node)[1]
        refs = re.findall(r"![0-9]+", node)
        threads, params = None, None
        for r in refs:
            m = re.search(r'"air\.max_work_group_size", i32 ([0-9]+)', meta[r])
            if m:
                threads = int(m[1])
        args = meta[refs[1]] if len(refs) > 1 else ""
        for r in re.findall(r"![0-9]+", args):
            a = meta[r]
            buffer = '"air.buffer"' in a or '"air.indirect_buffer"' in a
            threadgroup = '"air.address_space", i32 3' in a
            if (buffer and not threadgroup
                    and '"air.location_index", i32 0, i32 1' in a):
                m = re.search(r'"air\.arg_type_size", i32 ([0-9]+)', a)
                params = int(m[1]) if m else None
        out[name] = (threads, params)
    return out


def check(path, lib, irs, rows, stamp):
    """Refuses [lib] as the module's doc says; [rows], each kernel's
    (threads, words) by name, or None for the harness, whose kernels are
    checked against no table. Prints each kernel's bytes."""
    kernels = {}
    for ir in irs:
        if re.search(r"^@\S+ = .*addrspace\(3\) global", ir, re.M):
            sys.exit(f"{path}: static threadgroup memory")
        kernels.update(kernels_of(ir))
    kernels.pop(stamp, None)
    sizes = functions(lib)
    if set(sizes) != set(kernels) | {stamp}:
        sys.exit(f"{path}: functions {sorted(set(sizes) ^ set(kernels))} are "
                 "not kernels")
    if rows is not None and set(kernels) != set(rows):
        sys.exit(f"{path}: kernels {sorted(set(kernels) ^ set(rows))} are "
                 "in one of kernels.h's table and the metallib alone")
    print(f"{path}")
    print(f"  {'kernel':<32} {'threads':>7} {'params':>6} {'bytes':>9}")
    for name in sorted(kernels):
        threads, params = kernels[name]
        if rows is not None:
            want_threads, words = rows[name]
            if threads is not None and threads != want_threads:
                sys.exit(f"{path}: {name}: threads {threads}, expected "
                         f"{want_threads}")
            if params != 8 * words:
                sys.exit(f"{path}: {name}: parameter bytes {params}, "
                         f"expected {8 * words}")
        bound = "-" if threads is None else threads
        print(f"  {name:<32} {bound:>7} {params if params else 0:>6} "
              f"{sizes[name]:>9}")
    print(f"  {len(lib)} bytes")


# GPU code

# --sizes compiles every kernel of the library for the GPU of an M1 Max,
# as Metal would at a first call, and measures each one's code. A
# pipeline's first compile takes time about proportional to its code,
# 5-13 ms a KiB on an M1 Max, so a kernel of more than MOST_CODE bytes
# makes a first call wait for about a second or more.
SIZES_ARCH = "applegpu_g13s"
MOST_CODE = 100 * 1024
METAL_TT = ["xcrun", "-sdk", "macosx", "metal-tt"]
OBJDUMP = ["xcrun", "-sdk", "macosx", "metal-objdump"]


def code_sizes(lib, names):
    """Each kernel of [names] in the metallib [lib], by name: the bytes of
    its SIZES_ARCH code, from an archive of their pipelines that metal-tt
    compiles offline. The archive's objects name their modules, whose
    hashes are the metallib's functions' HASH tags."""
    with tempfile.TemporaryDirectory() as directory:
        d = Path(directory)
        source, archive, spec = (d / "in.metallib", d / "out.metallib",
                                 d / "pipelines.mtlp-json")
        source.write_bytes(lib)
        spec.write_text(json.dumps({
            "libraries": {"paths": [{"label": "nx", "path": str(source)}]},
            "pipelines": {"compute_pipelines": [
                {"compute_function": f"alias:nx#{n}",
                 "support_indirect_command_buffers": True} for n in names]}}))
        run([*METAL_TT, "-arch", SIZES_ARCH, str(source), str(spec), "-o",
             str(archive)])
        modules = run([*OBJDUMP, "--build-table=module", str(archive)])
        objects = run([*OBJDUMP, "--build-table=object", str(archive)])
    module_hash = dict(re.findall(r"^\[(\d+)\]\s+([0-9A-F]{64})\s", modules,
                                  re.M))
    by_hash = {h: name for name, h in hashes(lib).items()}
    out = {}
    for module, size in re.findall(
            r"^\[\d+\]\s+(\d+)\s+\d+\s+\(file offset:\d+, size:(\d+)\)",
            objects, re.M):
        name = by_hash.get(module_hash.get(module))
        if name is None:
            sys.exit(f"{SIZES_ARCH}: an object of module {module} is no "
                     "kernel's")
        out[name] = out.get(name, 0) + int(size)
    if set(out) != set(names):
        sys.exit(f"{SIZES_ARCH}: no code for {sorted(set(names) - set(out))}")
    return out


def sizes(families):
    """Prints the code of each kernel of the committed library and each
    family's sum, the largest first, and fails if a kernel's passes
    MOST_CODE. [families] holds each kernel's family name, by name."""
    path = METALLIBS[0][0]
    code = code_sizes((nx / path).read_bytes(), sorted(families))
    print(f"{path}, {SIZES_ARCH} code")
    print(f"  {'kernel':<36} {'family':<28} {'KiB':>7}")
    for name in sorted(code, key=lambda n: -code[n]):
        print(f"  {name:<36} {families[name]:<28} {code[name] / 1024:>7.1f}")
    sums = {}
    for name, c in code.items():
        sums[families[name]] = sums.get(families[name], 0) + c
    print(f"  {'family':<36} {'kernels':>7} {'KiB':>7}")
    for f in sorted(sums, key=lambda f: -sums[f]):
        count = sum(1 for n in families if families[n] == f)
        print(f"  {f:<36} {count:>7} {sums[f] / 1024:>7.1f}")
    over = sorted(n for n in code if code[n] > MOST_CODE)
    if over:
        sys.exit(f"{len(over)} kernels have more than {MOST_CODE // 1024} KiB "
                 f"of {SIZES_ARCH} code: {' '.join(over)}")


# kernels.ml

PREAMBLE = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generated by kernels/gen.py from kernels/src/kernels.h; do not edit.
   Regenerate it from the repository root with
   [uv run packages/nx/lib/metal/kernels/gen.py].

   kernels.h as the host plans with it: the metallib's kernels, in
   NX_METAL_KERNELS's order, each with its family, its group's threads and
   its parameters' words; the cast kernel of each source and target dtype
   code, of the source's class and the target's width
   (NX_METAL_CAST_KERNELS); and each constant of its enumerations,
   the words' indices among them, NX_COPY_DST as [copy_dst]. *)
"""

KEYWORDS = set("""and as assert begin class constraint do done downto else end
    exception external false for fun function functor if in include inherit
    initializer lazy let match method module mutable new nonrec object of open
    or private rec sig struct then to true try type val virtual when while
    with mod land lor lxor lsl lsr asr effect""".split())


def preprocess(source):
    return run([*METAL, "-I", str(HEADER.parent), "-E", "-P", "-x",
                "metal", "-"], input=source)


def table():
    """NX_METAL_KERNELS's rows, each a list of its arguments."""
    text = preprocess('#include "kernels.h"\n#define ROW(...) NX_ROW_ '
                      '__VA_ARGS__ NX_ROW_END_\nNX_METAL_KERNELS(ROW)\n')
    return [[a.strip() for a in row.split(",")]
            for row in re.findall(r"NX_ROW_ (.*?) NX_ROW_END_", text, re.S)]


def listed(call):
    """The items of a list macro's call, such as NX_CAST_TARGETS_8(L)."""
    text = preprocess('#include "kernels.h"\n#define L(x) NX_ROW_ x NX_ROW_END_\n'
                      + call + "\n")
    return [a.strip() for a in re.findall(r"NX_ROW_ (.*?) NX_ROW_END_", text,
                                          re.S)]


def casts():
    """NX_METAL_CAST_KERNELS's rows: name, class and width; each class of
    tables' sources and each width's targets."""
    text = preprocess('#include "kernels.h"\n#define ROW(X, name, k, e, w) '
                      'NX_ROW_ name, k, w NX_ROW_END_\n'
                      'NX_METAL_CAST_KERNELS(ROW, 0)\n')
    rows = [[a.strip() for a in row.split(",")]
            for row in re.findall(r"NX_ROW_ (.*?) NX_ROW_END_", text, re.S)]
    classes = {"NX_CAST_BYTES": listed("NX_CAST_BYTE_SOURCES(L)"),
               "NX_CAST_NIBBLES": listed("NX_CAST_NIBBLE_SOURCES(L)")}
    widths = {w: listed(f"NX_CAST_TARGETS_{w}(L)") for w in (8, 16, 32, 64)}
    return rows, classes, widths


def cast_match(rows, classes, widths, code):
    """kernels.ml's [cast]: each source code and target code to the
    kernel of the source's class and the target's width, a cast into the
    source's own dtype to none."""
    arms = []
    for name, k, w in rows:
        sources = classes.get(k, [k])
        targets = widths[int(w)]
        s = " | ".join(str(code[x]) for x in sources)
        t = " | ".join(str(code[x]) for x in targets)
        arms.append(f"    | ({s}), ({t}) -> Some {constructor(name)}\n")
    return ("let cast s t =\n  if s = t then None\n  else\n"
            "    match (s, t) with\n" + "".join(arms) + "    | _ -> None\n")


def enumerators():
    """The constants of kernels.h's anonymous enumerations, in order."""
    text = preprocess('#include "kernels.h"\n')
    names = []
    for body in re.findall(r"enum\s*\{([^}]*)\}", text):
        for item in body.split(","):
            name = item.split("=")[0].strip()
            if name:
                if not re.fullmatch(r"NX_[A-Z0-9_]+", name):
                    sys.exit(f"kernels.h: constant {name} is not NX_ and "
                             "upper case")
                names.append(name)
    return names


def mode_flags():
    """The names of NX_LOOP_MODE's flags: the constants of the enumeration
    that holds NX_LOOP_WIDE, and every NX_LOOP_ constant defined as a
    shift of 1, wherever it lies."""
    text = preprocess('#include "kernels.h"\n')
    flags = []
    for body in re.findall(r"enum\s*\{([^}]*)\}", text):
        items = [[x.strip() for x in item.split("=", 1)]
                 for item in body.split(",") if item.strip()]
        mode = any(i[0] == "NX_LOOP_WIDE" for i in items)
        for i in items:
            shift = len(i) == 2 and re.match(r"\(?\s*1u?\s*<<", i[1])
            if i[0].startswith("NX_LOOP_") and (mode or shift):
                flags.append(i[0])
    return flags


def check_flags(value):
    """Refuses NX_LOOP_MODE's flags unless each is one bit past v's 8 and
    no two are one: a merge can keep two definitions of one bit."""
    flags = mode_flags()
    seen = {}
    for f in flags:
        v = value[f]
        if v < 256 or v & (v - 1) != 0:
            sys.exit(f"kernels.h: mode flag {f} = {v} is not one bit past "
                     "v's 8")
        if v in seen:
            sys.exit(f"kernels.h: mode flags {seen[v]} and {f} are both {v}")
        seen[v] = f


def values(exprs):
    """The values of the constant expressions [exprs], as metal compiles
    them into an array of the device's IR."""
    source = ('#include "nx_dtype.h"\n#include "kernels.h"\nextern "C" '
              'constant long nx_metal_values[] = {\n' +
              "".join(f"  (long)({e}),\n" for e in exprs) + "};\n")
    ir = run([*METAL, *includes(nx), "-S", "-emit-llvm", "-x",
              "metal", "-o", "-", "-"], input=source)
    m = re.search(r"@nx_metal_values = .*?\[(\d+) x i64\] (\[.*?\]|"
                  r"zeroinitializer)", ir)
    if m is None or int(m[1]) != len(exprs):
        sys.exit("kernels.h: no array of the constants in metal's IR")
    if m[2] == "zeroinitializer":
        return [0] * len(exprs)
    return [int(v) for v in re.findall(r"i64 (-?\d+)", m[2])]


def constant_name(name):
    lower = name.removeprefix("NX_").lower()
    return lower + "_" if lower in KEYWORDS else lower


def constructor(name):
    return name[0].upper() + name[1:]


def kernels_ml():
    """kernels.ml's text; each kernel's (threads, words) by name, for the
    library and the steps; and each library kernel's family, by name, as
    its NX_FAMILY_ constant's name."""
    rows = table()
    if any(len(r) != 4 for r in rows):
        sys.exit("kernels.h: a row of NX_METAL_KERNELS is not "
                 "X(name, family, threads, words)")
    names = enumerators()
    cast_rows, classes, widths = casts()
    listed_codes = sorted({x for r in cast_rows for x in r[1:2]}
                          | {x for l in classes.values() for x in l}
                          | {x for l in widths.values() for x in l})
    numbers = values(names + [c for r in rows for c in r[1:]] + listed_codes)
    value = dict(zip(names, numbers))
    check_flags(value)
    columns = numbers[len(names):len(names) + 3 * len(rows)]
    code = dict(zip(listed_codes, numbers[len(names) + 3 * len(rows):]))
    facts = {r[0]: tuple(columns[3 * i:3 * i + 3]) for i, r in enumerate(rows)}
    out = [PREAMBLE, "(* Kernels *)\n"]
    out.append("type t =\n" + "".join(f"  | {constructor(r[0])}\n"
                                       for r in rows))
    out.append("let all = [| " + "; ".join(constructor(r[0]) for r in rows)
               + " |]\n")
    for what, f in [("index", lambda i, r: str(i)),
                    ("name", lambda i, r: f'"{r[0]}"'),
                    ("family", lambda i, r: str(facts[r[0]][0])),
                    ("threads", lambda i, r: str(facts[r[0]][1])),
                    ("words", lambda i, r: str(facts[r[0]][2]))]:
        out.append(f"let {what} = function\n" + "".join(
            f"  | {constructor(r[0])} -> {f(i, r)}\n"
            for i, r in enumerate(rows)))
    out.append(cast_match(cast_rows, classes, widths, code))
    out.append("(* Constants *)\n")
    seen = set()
    for name in names:
        ocaml = constant_name(name)
        if ocaml in seen:
            sys.exit(f"kernels.h: two constants named {ocaml}")
        seen.add(ocaml)
        v = value[name]
        out.append(f"let {ocaml} = {v if v >= 0 else f'({v})'}")
    text = "\n".join(out) + "\n"
    steps = value[STEPS_FAMILY]
    family_name = {value[n]: n for n in names if n.startswith("NX_FAMILY_")}
    families = {name: family_name[family]
                for name, (family, _, _) in facts.items() if family != steps}
    return text, {holds: {name: (threads, words)
                          for name, (family, threads, words) in facts.items()
                          if (family == steps) == (holds == STEPS)}
                  for holds in (LIBRARY, STEPS)}, families


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true",
                       help="fail if a committed file differs; write nothing")
    modes.add_argument("--sizes", action="store_true",
                       help="print each kernel's GPU code and fail past "
                       f"{MOST_CODE // 1024} KiB; write nothing")
    args = parser.parse_args()
    pinned()
    text, rows, families = kernels_ml()
    if args.sizes:
        sizes(families)
        return
    built = {KERNELS_ML: text.encode()}
    for path, units, sources, holds in METALLIBS:
        with tempfile.TemporaryDirectory() as directory:
            lib, irs, closure = build(units, sources, directory)
        missing = sorted(closure - set(sources))
        if missing:
            sys.exit(f"{path}: its stamp's sources lack {missing}, which "
                     "its units include")
        with tempfile.TemporaryDirectory() as directory:
            root = copied(closure, directory)
            again, _, _ = build(units, sources, directory, root)
        if lib != again:
            sys.exit(f"{path}: builds from two directories differ")
        check(path, lib, irs, rows.get(holds), stamp_name(sources))
        built[path] = lib
    stale = []
    for path, blob in built.items():
        target = nx / path
        if target.exists() and target.read_bytes() == blob:
            continue
        stale.append(path)
        if not args.check:
            target.write_bytes(blob)
    for path in stale:
        print(f"packages/nx/{path} {'differs' if args.check else 'written'}")
    if args.check and stale:
        sys.exit("regenerate with: uv run packages/nx/lib/metal/kernels/gen.py")


if __name__ == "__main__":
    main()
