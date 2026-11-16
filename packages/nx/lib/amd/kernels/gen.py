# /// script
# requires-python = ">=3.10"
# ///
"""Compile nx.amd's kernels and the AMD suite's harness to their code
objects, and write kernels.ml, src/kernels.h as the host plans with it.

    uv run packages/nx/lib/amd/kernels/gen.py [--check]

The toolchain is LLVM's release archive for this host, fetched once by URL
into the user's cache (XDG_CACHE_HOME, else ~/.cache, under raven/llvm),
checked against its SHA-256 and refused on a mismatch. Only clang, lld,
llvm-objcopy and clang's own headers are unpacked from it. No ROCm and no
device library: every kernel is HIP compiled for the device alone, with
nx.amd's C library shims (libc/) and its own builtins (src/device.h).

Each unit of an object compiles with -fgpu-rdc to bitcode under a cuid of
its own, and ld.lld links the units' bitcode into one shared code object
for gfx12-generic, compiling it as one module, so that the object holds
one metadata note that lists every kernel. Before it strips the object, the
script refuses one whose kernels the rules below forbid, and prints each
kernel's registers, SGPR spills, LDS and bytes:

- a private segment, which rig would grow device-wide scratch for, or a
  VGPR spill, or a dynamic stack. An SGPR spill goes to a lane of a VGPR,
  and to memory only through a VGPR spilled there, which the private
  segment shows: lane spills are allowed and printed, since each costs a
  v_writelane and a v_readlane at every use;
- an implicit argument, which rig writes only in part, or a system value
  rig gives no kernel (the dispatch packet, the queue, the dispatch id): a
  kernel reads its parameters, 64-bit words, through its kernarg pointer,
  and its arguments are exactly those words;
- a function symbol without its kernel descriptor (name.kd): a function
  that is not a kernel is a call, which keeps its frame in scratch;
- waves of other than 32 lanes, or groups of other than its row's threads.

The strip removes the static symbol table and every section rig's loader
does not read; the dynamic symbols keep each name.kd, by which rig resolves
a kernel.

Each object also carries a symbol named after its sources' digest, which
the dune build checks (fresh.ml) in every tree, without the toolchain. The
digest covers this script and kernels.ml's sources, so a change to them
needs a regenerated object, and with it kernels.ml.

kernels.ml states src/kernels.h's facts as the pinned clang reads them:
its preprocessor expands the X-macro NX_AMD_KERNELS, the kernel table, and
the anonymous enumerations, whose constants are the shared constants and
the parameters' word indices; the values of every constant and of every
row's columns come from a constant array clang compiles to LLVM IR. No
struct crosses: a kernel's parameters are words. Each kernel's static LDS
comes from the library object's metadata, so the host knows how much
dynamic LDS a group may take beside it; each contraction tile's dynamic
LDS comes from kernels.h's formula (NX_GEMM_LDS, NX_WMMA_LDS), which the
device code checks its tile's layout against.

clang and lld give the same bytes on every run, from any directory and on
either host, so --check compiles every object and writes kernels.ml in
memory, writes nothing, and fails if any differs from the committed file.
"""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

nx = Path(__file__).resolve().parents[3]

# Toolchain

LLVM_VERSION = "22.1.8"
LLVM_RELEASE = ("https://github.com/llvm/llvm-project/releases/download/"
                f"llvmorg-{LLVM_VERSION}/")

# Each host's release archive and its SHA-256, as the release lists them.
LLVM_ARCHIVES = {
    ("Darwin", "arm64"): (
        f"LLVM-{LLVM_VERSION}-macOS-ARM64",
        "f260f4f7c0d430828a81ae8a3826a1d63fc0963ec2459489308cc23b1f7eab4f"),
    ("Linux", "x86_64"): (
        f"LLVM-{LLVM_VERSION}-Linux-X64",
        "df0e1ecf16caf3489a272a5eea4eec9b0d82878f6477fa309504f918a0006384"),
}

# The archive's members the script runs, and clang's own headers
# (stdint.h, stddef.h), which -nostdlibinc keeps.
TOOLS = ["bin/clang", "bin/clang-22", "bin/ld.lld", "bin/lld",
         "bin/llvm-objcopy"]
RESOURCE = "lib/clang/22/include/"


def cache():
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "raven" / "llvm"


def fetch(name, sha256):
    """The archive [name], downloaded into the cache and checked."""
    archive = cache() / f"{name}.tar.xz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        part = archive.with_suffix(".part")
        print(f"fetching {LLVM_RELEASE}{name}.tar.xz", file=sys.stderr)
        with urllib.request.urlopen(f"{LLVM_RELEASE}{name}.tar.xz") as r, \
                open(part, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        part.rename(archive)
    digest = hashlib.sha256()
    with open(archive, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        sys.exit(f"{archive}: SHA-256 {digest.hexdigest()}, expected {sha256};"
                 " delete it to fetch it again")
    return archive


def toolchain():
    """The directory of the unpacked toolchain for this host."""
    host = (platform.system(), platform.machine())
    if host not in LLVM_ARCHIVES:
        sys.exit(f"no LLVM {LLVM_VERSION} archive for {host[0]} {host[1]}")
    name, sha256 = LLVM_ARCHIVES[host]
    root = cache() / name
    done = root / ".unpacked"
    if done.exists() and done.read_text() == sha256:
        return root
    archive = fetch(name, sha256)
    if root.exists():
        shutil.rmtree(root)
    wanted = {f"{name}/{t}" for t in TOOLS}
    resource = f"{name}/{RESOURCE}"
    with tarfile.open(archive, "r:xz") as tar:
        for m in tar:
            if m.name in wanted or m.name.startswith(resource):
                tar.extract(m, cache(), filter="data")
    missing = [t for t in TOOLS if not (cache() / name / t).exists()]
    if missing:
        sys.exit(f"{archive} lacks {', '.join(missing)}")
    done.write_text(sha256)
    return root


# Compiling

# HIP for the device alone, C++17, against no system header and no device
# library. The contract on floats: no product contracted into a sum,
# division and square roots correctly rounded, subnormals kept. The host
# triple is fixed so that the build host changes nothing, and a config
# file of a packaged clang chooses nothing. Loops stay loops: there is no
# memcpy or memset to call. A unit's cuid is its own, fixed, so that two
# builds name its symbols alike and two units never alike.
TARGET = "gfx12-generic"
HIP = ["--no-default-config", "-x", "hip", "--offload-device-only",
       f"--offload-arch={TARGET}", "--target=x86_64-unknown-linux-gnu",
       "-nogpuinc", "-nogpulib", "-nostdlibinc", "-std=c++17", "-O3",
       "-ffp-contract=off", "-fhip-fp32-correctly-rounded-divide-sqrt",
       "-fno-gpu-flush-denormals-to-zero", "-mcode-object-version=6",
       "-mllvm", "-disable-loop-idiom-all", "-Wall", "-Werror"]

# The units' bitcode to one code object, compiled for the target at link.
# The early (pre-RA) machine LICM is off: it hoisted the constants of every
# kind inlined into the loops' tape interpreter out of the interpreter's
# loop, each then live across its whole run, a register each, spilled to
# lanes past the SGPRs. The IR's LICM and the post-RA machine LICM remain.
# LTO compiles one module, so this holds for every kernel of the object;
# none spills more for it.
LINK = ["-shared", f"-plugin-opt=mcpu={TARGET}", "-plugin-opt=O3", "-mllvm",
        "-disable-loop-idiom-all", "-mllvm", "-disable-machine-licm"]

INCLUDES = ["lib/amd/kernels/libc", "lib/amd/kernels/src", "lib/array",
            "lib/kernel"]

# Each code object: its path, its units and the sources its stamp covers,
# in the order of the dune rule that checks it. A unit compiles alone.
OBJECTS = [
    ("lib/amd/kernels/gfx12-generic.co", [
        "lib/amd/kernels/src/zero.hip",
        "lib/amd/kernels/src/loop.hip",
        "lib/amd/kernels/src/moves.hip",
        "lib/amd/kernels/src/fft.hip",
        "lib/amd/kernels/src/sort.hip",
        "lib/amd/kernels/src/contract.hip",
        "lib/amd/kernels/src/factor.hip",
        "lib/amd/kernels/src/spectral.hip",
    ], [
        "lib/amd/kernels/gen.py",
        "lib/amd/kernels/libc/math.h",
        "lib/amd/kernels/libc/string.h",
        "lib/amd/kernels/src/kernels.h",
        "lib/amd/kernels/src/device.h",
        "lib/amd/kernels/src/dtypes.h",
        "lib/array/nx_dtype.h",
        "lib/kernel/nx_kernel.h",
        "lib/kernel/nx_kinds.h",
        "lib/kernel/nx_kinds_real.h",
        "lib/kernel/nx_kinds_complex.h",
        "lib/amd/kernels/src/zero.hip",
        "lib/amd/kernels/src/loop.hip",
        "lib/amd/kernels/src/moves.hip",
        "lib/amd/kernels/src/fft.hip",
        "lib/amd/kernels/src/sort.hip",
        "lib/amd/kernels/src/contract.hip",
        "lib/amd/kernels/src/wmma.h",
        "lib/amd/kernels/src/factor.hip",
        "lib/amd/kernels/src/sp_common.h",
        "lib/amd/kernels/src/sp_block.h",
        "lib/amd/kernels/src/sp_jacobi.h",
        "lib/amd/kernels/src/sp_tridiag.h",
        "lib/amd/kernels/src/sp_stedc.h",
        "lib/amd/kernels/src/sp_bd.h",
        "lib/amd/kernels/src/sp_eig.h",
        "lib/amd/kernels/src/spectral.hip",
    ]),
    ("test/amd/support/gfx12-generic.co", [
        "test/amd/support/harness.hip",
        "test/amd/support/staged.hip",
    ], [
        "lib/amd/kernels/gen.py",
        "lib/amd/kernels/src/kernels.h",
        "lib/amd/kernels/src/device.h",
        "lib/amd/kernels/libc/math.h",
        "lib/amd/kernels/libc/string.h",
        "lib/array/nx_dtype.h",
        "test/amd/support/harness.hip",
        "test/amd/support/staged.hip",
    ]),
]

KERNELS_ML = "lib/amd/kernels.ml"
HEADER = nx / "lib/amd/kernels/src/kernels.h"


def run(args, **kw):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          **kw).stdout


def digest(sources):
    """An MD5 of the lines of each file's hex MD5, as fresh.ml computes it."""
    lines = "".join(hashlib.md5((nx / s).read_bytes()).hexdigest() + "\n"
                    for s in sources)
    return hashlib.md5(lines.encode()).hexdigest()


def stamp(sources):
    """A unit of one byte named after the sources' digest."""
    return ("extern \"C\" __attribute__((device, used)) const char "
            f"nx_amd_sources_{digest(sources)} = 1;\n")


def cuid(unit):
    return "nx_amd_" + Path(unit).stem


def link(tools, units, sources, directory):
    """The unstripped code object of [units] and the stamp of [sources]."""
    clang, lld = tools / "bin/clang", tools / "bin/ld.lld"
    includes = [arg for d in INCLUDES for arg in ("-I", str(nx / d))]
    stamp_unit = Path(directory) / "stamp.hip"
    stamp_unit.write_text(stamp(sources))
    objects = []
    for unit in [*map(lambda u: nx / u, units), stamp_unit]:
        bc = Path(directory) / f"{unit.stem}.bc"
        subprocess.run([clang, *HIP, *includes, f"-cuid={cuid(unit)}",
                        "-fgpu-rdc", "-c", str(unit), "-o", str(bc)],
                       check=True)
        objects.append(str(bc))
    co = Path(directory) / "out.co"
    subprocess.run([lld, *LINK, *objects, "-o", str(co)], check=True)
    return co


def strip(tools, co):
    """[co] without its static symbols and the sections rig does not read."""
    out = co.with_suffix(".stripped")
    subprocess.run([tools / "bin/llvm-objcopy", "--strip-all",
                    "--remove-section=.comment", str(co), str(out)],
                   check=True)
    return out.read_bytes()


# ELF and the AMDGPU metadata

SHT_SYMTAB, SHT_NOTE = 2, 7
STT_FUNC = 2
NT_AMDGPU_METADATA = 32

# A kernel descriptor's kernel_code_properties (AMDGPUUsage, "Kernel
# Descriptor"): its byte, and the bits of the system values a kernel may not
# read: the private segment buffer, since a kernel has no scratch, and what
# rig does not give it, the dispatch packet's and the queue's addresses and
# the dispatch's id.
KD_CODE_PROPERTIES = 56
KD_SYSTEM_VALUES = {
    1 << 0: "private segment buffer",
    1 << 1: "dispatch packet",
    1 << 2: "queue",
    1 << 4: "dispatch id",
}


def sections(elf):
    """Each section of the ELF64 [elf]: (name, type, address, offset, size,
    link, entry size)."""
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3a)
    heads = [struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize)
             for i in range(shnum)]
    names = heads[shstrndx][4]

    def name(at):
        return elf[names + at:elf.index(b"\0", names + at)].decode()

    return [(name(h[0]), h[1], h[3], h[4], h[5], h[6], h[9]) for h in heads]


def symbols(elf):
    """Each symbol of the static symbol table: name, (type, address,
    size)."""
    ss = sections(elf)
    out = {}
    for _, t, _, offset, size, link_, entsize in ss:
        if t != SHT_SYMTAB:
            continue
        strings = ss[link_][3]
        for i in range(1, size // entsize):
            at = offset + i * entsize
            name, info = struct.unpack_from("<IB", elf, at)
            value, sym_size = struct.unpack_from("<QQ", elf, at + 8)
            end = elf.index(b"\0", strings + name)
            out[elf[strings + name:end].decode()] = (info & 0xf, value,
                                                     sym_size)
    return out


def read_at(elf, address, fmt):
    """The value of [fmt] at the virtual [address] of [elf]."""
    for _, _, base, offset, size, _, _ in sections(elf):
        if base <= address < base + size:
            return struct.unpack_from(fmt, elf, offset + address - base)[0]
    sys.exit(f"no section holds address 0x{address:x}")


# MessagePack's fixed-size numbers, and the strings, arrays and maps whose
# length follows their type byte.
NUMBERS = {0xcc: ">B", 0xcd: ">H", 0xce: ">I", 0xcf: ">Q",
           0xd0: ">b", 0xd1: ">h", 0xd2: ">i", 0xd3: ">q"}
SIZED = {0xd9: ("str", ">B"), 0xda: ("str", ">H"), 0xdb: ("str", ">I"),
         0xdc: ("array", ">H"), 0xdd: ("array", ">I"),
         0xde: ("map", ">H"), 0xdf: ("map", ">I")}


def unpack(b, i):
    """The MessagePack value at b[i] and the offset after it."""
    t = b[i]
    i += 1
    if t <= 0x7f or t >= 0xe0:
        return (t if t <= 0x7f else t - 256), i
    if t in (0xc2, 0xc3):
        return t == 0xc3, i
    if t in NUMBERS:
        f = NUMBERS[t]
        return struct.unpack_from(f, b, i)[0], i + struct.calcsize(f)
    if t in SIZED:
        kind, f = SIZED[t]
        n, = struct.unpack_from(f, b, i)
        i += struct.calcsize(f)
    elif 0xa0 <= t <= 0xbf:
        kind, n = "str", t & 0x1f
    elif 0x80 <= t <= 0x9f:
        kind, n = ("map" if t < 0x90 else "array"), t & 0x0f
    else:
        sys.exit(f"MessagePack type 0x{t:02x} in the metadata")
    if kind == "str":
        return b[i:i + n].decode(), i + n
    items = []
    for _ in range(2 * n if kind == "map" else n):
        v, i = unpack(b, i)
        items.append(v)
    return (dict(zip(items[::2], items[1::2])) if kind == "map" else items), i


def metadata(elf):
    """The AMDGPU metadata note of [elf]."""
    for _, t, _, offset, size, _, _ in sections(elf):
        if t != SHT_NOTE:
            continue
        i = offset
        while i < offset + size:
            namesz, descsz, ntype = struct.unpack_from("<III", elf, i)
            name = elf[i + 12:i + 12 + namesz].rstrip(b"\0")
            desc = i + 12 + (namesz + 3) // 4 * 4
            if name == b"AMDGPU" and ntype == NT_AMDGPU_METADATA:
                return unpack(elf[desc:desc + descsz], 0)[0]
            i = desc + (descsz + 3) // 4 * 4
    sys.exit("no AMDGPU metadata note")


def check(path, elf, rows):
    """Refuses [elf] as the module's doc says; [rows], each kernel's
    (threads, words) by name, or None for the harness, whose kernels are
    checked against no table. Prints each kernel's facts, and is each
    kernel's (static LDS bytes, VGPRs) by name."""
    meta = metadata(elf)
    if meta["amdhsa.target"] != f"amdgcn-amd-amdhsa--{TARGET}":
        sys.exit(f"{path}: built for {meta['amdhsa.target']}")
    syms = symbols(elf)
    calls = sorted(n for n, (t, _, _) in syms.items()
                   if t == STT_FUNC and n + ".kd" not in syms)
    if calls:
        sys.exit(f"{path}: functions that are no kernel: {', '.join(calls)}")
    kernels = {k[".name"]: k for k in meta["amdhsa.kernels"]}
    if rows is not None and set(kernels) != set(rows):
        sys.exit(f"{path}: kernels {sorted(set(kernels) ^ set(rows))} are "
                 "in one of kernels.h's table and the object alone")
    print(f"{path}")
    print(f"  {'kernel':<32} {'VGPR':>5} {'SGPR':>5} {'spills':>6} "
          f"{'LDS':>6} {'bytes':>8}")
    for name in sorted(kernels):
        k = kernels[name]
        hidden = [a[".value_kind"] for a in k[".args"]
                  if a[".value_kind"] not in ("by_value", "global_buffer")]
        properties = read_at(elf, syms[name + ".kd"][1] + KD_CODE_PROPERTIES,
                             "<H")
        system = [v for bit, v in KD_SYSTEM_VALUES.items() if properties & bit]
        facts = [
            ("private segment", k[".private_segment_fixed_size"], 0),
            ("VGPR spills", k[".vgpr_spill_count"], 0),
            ("dynamic stack", k.get(".uses_dynamic_stack", False), False),
            ("implicit arguments", hidden, []),
            ("system values", system, []),
            ("wavefront size", k[".wavefront_size"], 32),
        ]
        if rows is not None:
            threads, words = rows[name]
            facts += [("group size", k[".max_flat_workgroup_size"], threads),
                      ("argument bytes", k[".kernarg_segment_size"],
                       8 * words)]
        for what, got, want in facts:
            if got != want:
                sys.exit(f"{path}: {name}: {what} {got}, expected {want}")
        print(f"  {name:<32} {k['.vgpr_count']:>5} {k['.sgpr_count']:>5} "
              f"{k['.sgpr_spill_count']:>6} "
              f"{k['.group_segment_fixed_size']:>6} {syms[name][2]:>8}")
    return {name: (k[".group_segment_fixed_size"], k[".vgpr_count"])
            for name, k in kernels.items()}


# kernels.ml

PREAMBLE = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generated by kernels/gen.py from kernels/src/kernels.h; do not edit.
   Regenerate it from the repository root with
   [uv run packages/nx/lib/amd/kernels/gen.py].

   kernels.h as the host plans with it: the code object's kernels, in
   NX_AMD_KERNELS's order, each with its group's threads, its
   parameters' words, and the static LDS bytes and VGPRs the code object
   gives it; the SIMT contraction tiles of NX_AMD_GEMM_TILES and the
   matrix-unit ones of NX_AMD_WMMA_TILES, each with the dynamic LDS bytes
   kernels.h states; and each constant of its enumerations, the words'
   indices among them, NX_ZERO_DST as [zero_dst]. *)
"""

KEYWORDS = set("""and as assert begin class constraint do done downto else end
    exception external false for fun function functor if in include inherit
    initializer lazy let match method module mutable new nonrec object of open
    or private rec sig struct then to true try type val virtual when while
    with mod land lor lxor lsl lsr asr effect""".split())


def preprocess(tools, source):
    return run([tools / "bin/clang", *HIP, "-I", str(HEADER.parent), "-E",
                "-P", "-"], input=source)


def table(tools):
    """NX_AMD_KERNELS's rows, each a list of its arguments."""
    text = preprocess(tools, '#include "kernels.h"\n#define ROW(...) NX_ROW_ '
                      '__VA_ARGS__ NX_ROW_END_\nNX_AMD_KERNELS(ROW)\n')
    return [[a.strip() for a in row.split(",")]
            for row in re.findall(r"NX_ROW_ (.*?) NX_ROW_END_", text, re.S)]


def tiles(tools):
    """NX_AMD_GEMM_TILES's rows: name, compute type, eight numbers and the
    load."""
    text = preprocess(tools, '#include "kernels.h"\n#define T(X, ...) '
                      'NX_TILE_ __VA_ARGS__ NX_TILE_END_\n'
                      'NX_AMD_GEMM_TILES(T, _)\n')
    rows = [[a.strip() for a in row.split(",")]
            for row in re.findall(r"NX_TILE_ (.*?) NX_TILE_END_", text, re.S)]
    if any(len(r) != 11 or r[1] not in ("f32", "f64", "u32", "u64")
           or r[10] not in ("native", "convert") for r in rows):
        sys.exit("kernels.h: a row of NX_AMD_GEMM_TILES is not "
                 "R(X, name, t, BM, BN, BK, TM, TN, LM, WM, WN, load)")
    return rows


TILE_FIELDS = ["bm", "bn", "bk", "tm", "tn", "lm", "wm", "wn", "lds"]

WMMA_KINDS = {"f16": "F16", "bf16": "Bf16", "8bit": "Bits8", "any": "Any"}


def wmma_tiles(tools):
    """NX_AMD_WMMA_TILES's rows: name, kind and five numbers."""
    text = preprocess(tools, '#include "kernels.h"\n#define M(X, ...) '
                      'NX_TILE_ __VA_ARGS__ NX_TILE_END_\n'
                      'NX_AMD_WMMA_TILES(M, _)\n')
    rows = [[a.strip() for a in row.split(",")]
            for row in re.findall(r"NX_TILE_ (.*?) NX_TILE_END_", text, re.S)]
    if any(len(r) != 7 or r[1] not in WMMA_KINDS for r in rows):
        sys.exit("kernels.h: a row of NX_AMD_WMMA_TILES is not "
                 "M(X, name, kind, BM, BN, BKB, WM, WN)")
    return rows


WMMA_FIELDS = ["bm", "bn", "bkb", "wm", "wn", "lds"]


def enumerators(tools):
    """The constants of kernels.h's anonymous enumerations, in order."""
    text = preprocess(tools, '#include "kernels.h"\n')
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


def values(tools, exprs):
    """The values of the C++ constant expressions [exprs], as clang
    compiles them into an array of the device's IR."""
    source = ('#include "kernels.h"\nextern "C" __attribute__((device, used))'
              ' const long long nx_amd_values[] = {\n' +
              "".join(f"  (long long)({e}),\n" for e in exprs) + "};\n")
    ir = run([tools / "bin/clang", *HIP, "-I", str(HEADER.parent),
              "-emit-llvm", "-S", "-o", "-", "-"], input=source)
    m = re.search(r"@nx_amd_values = .*?\[(\d+) x i64\] (\[.*?\]|"
                  r"zeroinitializer)", ir)
    if m is None or int(m[1]) != len(exprs):
        sys.exit("kernels.h: no array of the constants in clang's IR")
    if m[2] == "zeroinitializer":
        return [0] * len(exprs)
    return [int(v) for v in re.findall(r"i64 (-?\d+)", m[2])]


def constant_name(name):
    lower = name.removeprefix("NX_").lower()
    return lower + "_" if lower in KEYWORDS else lower


def constructor(name):
    return name[0].upper() + name[1:]


def kernels_ml(tools):
    """A function of each kernel's (static LDS bytes, VGPRs) by name to
    kernels.ml's text, and each kernel's (threads, words) by name."""
    rows = table(tools)
    if any(len(r) != 3 for r in rows):
        sys.exit("kernels.h: a row of NX_AMD_KERNELS is not "
                 "X(name, threads, words)")
    simt = tiles(tools)
    units = wmma_tiles(tools)
    names = enumerators(tools)
    # A tile's columns, then its dynamic LDS bytes by kernels.h's formula.
    simt_cols = [c for r in simt for c in r[2:10]
                 + [f"NX_GEMM_LDS({r[1]}, {r[2]}, {r[3]}, {r[4]})"]]
    unit_cols = [c for r in units for c in r[2:7]
                 + [f"NX_WMMA_LDS({r[2]}, {r[3]}, {r[4]})"]]
    numbers = values(tools, names + [c for r in rows for c in r[1:]]
                     + simt_cols + unit_cols)
    value = dict(zip(names, numbers))
    columns = numbers[len(names):]
    facts = {r[0]: (columns[2 * i], columns[2 * i + 1])
             for i, r in enumerate(rows)}
    shapes = columns[2 * len(rows):]
    wshapes = shapes[len(TILE_FIELDS) * len(simt):]
    if any(r[0] not in facts for r in simt + units):
        sys.exit("kernels.h: a tile of NX_AMD_GEMM_TILES or NX_AMD_WMMA_TILES "
                 "is no kernel")

    def render(meta):
        out = [PREAMBLE, "(* Kernels *)\n"]
        out.append("type t =\n" + "".join(f"  | {constructor(r[0])}\n"
                                           for r in rows))
        out.append("let all = [| " + "; ".join(constructor(r[0])
                                               for r in rows) + " |]\n")
        for what, f in [("index", lambda i, r: str(i)),
                        ("name", lambda i, r: f'"{r[0]}"'),
                        ("threads", lambda i, r: str(facts[r[0]][0])),
                        ("words", lambda i, r: str(facts[r[0]][1])),
                        ("lds", lambda i, r: str(meta[r[0]][0])),
                        ("vgprs", lambda i, r: str(meta[r[0]][1]))]:
            out.append(f"let {what} = function\n" + "".join(
                f"  | {constructor(r[0])} -> {f(i, r)}\n"
                for i, r in enumerate(rows)))
        out.append("(* SIMT contraction tiles *)\n")
        out.append("type ct = F32 | F64 | U32 | U64")
        out.append("type load = Native | Convert\n")
        out.append("type tile = {\n  t : ct;\n" + "".join(
            f"  {f} : int;\n" for f in TILE_FIELDS) + "  load : load;\n}\n")
        cases = []
        for i, r in enumerate(simt):
            fields = "; ".join(f"{f} = {shapes[len(TILE_FIELDS) * i + j]}"
                               for j, f in enumerate(TILE_FIELDS))
            cases.append(f"  | {constructor(r[0])} ->\n      Some {{ t = "
                         f"{r[1].upper()}; {fields}; load = "
                         f"{constructor(r[10])} }}\n")
        out.append("let tile = function\n" + "".join(cases)
                   + "  | _ -> None\n")
        out.append("(* Matrix-unit contraction tiles *)\n")
        out.append("type wkind = " + " | ".join(WMMA_KINDS.values()) + "\n")
        out.append("type wtile = {\n  kind : wkind;\n" + "".join(
            f"  {f} : int;\n" for f in WMMA_FIELDS) + "}\n")
        cases = []
        for i, r in enumerate(units):
            fields = "; ".join(f"{f} = {wshapes[len(WMMA_FIELDS) * i + j]}"
                               for j, f in enumerate(WMMA_FIELDS))
            cases.append(f"  | {constructor(r[0])} ->\n      Some {{ kind = "
                         f"{WMMA_KINDS[r[1]]}; {fields} }}\n")
        out.append("let wtile = function\n" + "".join(cases)
                   + "  | _ -> None\n")
        out.append("(* Constants *)\n")
        seen = set()
        for name in names:
            ocaml = constant_name(name)
            if ocaml in seen:
                sys.exit(f"kernels.h: two constants named {ocaml}")
            seen.add(ocaml)
            v = value[name]
            out.append(f"let {ocaml} = {v if v >= 0 else f'({v})'}")
        return "\n".join(out) + "\n"

    return render, facts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="fail if a committed file differs; write nothing")
    args = parser.parse_args()
    tools = toolchain()
    render, rows = kernels_ml(tools)
    built = {}
    for path, units, sources in OBJECTS:
        library = path.startswith("lib/")
        with tempfile.TemporaryDirectory() as directory:
            co = link(tools, units, sources, directory)
            meta = check(path, co.read_bytes(), rows if library else None)
            built[path] = strip(tools, co)
        if library:
            built[KERNELS_ML] = render(meta).encode()
        print(f"  {len(built[path])} bytes")
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
        sys.exit("regenerate with: uv run packages/nx/lib/amd/kernels/gen.py")


if __name__ == "__main__":
    main()
