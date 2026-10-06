#!/usr/bin/env python3
"""Compiles nx.amd's kernels ahead of time into the code objects nx.amd embeds:
one per (kind, dtypes) and target, gfx12-generic/<key>.co, from the sources
in src/.

Run from the repository root, on a machine with the pinned comgr (ROCm's code
object manager), which it loads from $COMGR_PATH, else
/opt/rocm/lib/libamd_comgr.so:

  uv run packages/nx/lib/amd/kernels/gen/gen.py
  uv run packages/nx/lib/amd/kernels/gen/gen.py --check
  uv run packages/nx/lib/amd/kernels/gen/gen.py --pin

pins.json records the toolchain (the comgr library, by name and SHA-256), the
options, and the SHA-256 of every input (the sources, nx's dtype codecs and
this script) and of every output. The script refuses another comgr; --pin
moves the pin and regenerates everything. --check regenerates into a
temporary directory and fails if a committed file differs: comgr is
deterministic for a version and options. A dune rule checks the digests on
every machine, with no toolchain, so a source changed without regenerating,
or a code object edited, fails the tests.

comgr 3.0 writes out the device libraries a generic target links under a name
its clang does not read, so the script hands clang a copy under the right name
(device_libs).

Python's standard library only.
"""

import argparse
import ctypes
import hashlib
import json
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
KERNELS = HERE.parent
SRC = KERNELS / "src"
DTYPE_H = KERNELS.parents[1] / "dtype" / "nx_dtype.h"
PINS = HERE / "pins.json"

TARGETS = ["gfx12-generic"]

# The served dtypes, by nx's name and the name of their type in common.h.
DTYPES = [
    ("bool", "bool_"), ("int8", "int8"), ("uint8", "uint8"), ("int16", "int16"),
    ("uint16", "uint16"), ("int32", "int32"), ("uint32", "uint32"), ("int64", "int64"),
    ("uint64", "uint64"), ("float16", "float16"), ("bfloat16", "bfloat16"),
    ("float32", "float32"), ("float64", "float64"), ("float8_e4m3", "float8_e4m3"),
    ("float8_e5m2", "float8_e5m2"),
]
# Kinds that only move bytes are keyed by element width, in bytes.
WIDTHS = {1: "uint8_t", 2: "uint16_t", 4: "uint32_t", 8: "uint64_t"}

# A unit's ID names its symbol __hip_cuid_<hash>. comgr otherwise derives it
# from every include it is handed, so that a new source would change every code
# object; each code object is a program of its own, which one ID serves.
COMPILE = [
    "-O3", "-ffp-contract=off", "-fhip-fp32-correctly-rounded-divide-sqrt",
    "-fno-gpu-flush-denormals-to-zero", "-nogpuinc", "-mcode-object-version=6",
    "-std=c++17", "-cuid=nx_amd", "-Wall", "-Werror", "-Xclang", "-disable-llvm-passes",
    "-Xclang", "-aux-triple", "-Xclang", "x86_64-unknown-linux-gnu",
]
CODEGEN = ["-O3", "-ffp-contract=off", "-mcode-object-version=6", "-mllvm",
           "-amdgpu-internalize-symbols"]


# The kinds of each family and the dtypes they serve, as nx.cpu's tables
# (cpu/nx_c_map.c) give them: integers (signed or not), floats and booleans.
INTS = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]
FLOATS = ["float16", "bfloat16", "float32", "float64", "float8_e4m3", "float8_e5m2"]
NUMERIC = INTS + FLOATS
UNARY = [(["neg", "recip", "abs", "sign"], NUMERIC),
         (["sqrt", "exp", "log", "log1p", "expm1", "sin", "cos", "tan", "asin", "acos", "atan", "sinh",
           "cosh", "tanh", "erf", "trunc", "ceil", "floor", "round"], FLOATS)]
BINARY = [(["add", "sub", "mul", "pow", "idiv", "mod"], NUMERIC), (["fdiv", "atan2"], FLOATS),
          (["maximum", "minimum"], NUMERIC + ["bool"]), (["and", "or", "xor"], INTS + ["bool"])]
COMPARE = [(["equal", "not_equal", "less", "less_equal"], NUMERIC + ["bool"])]
# As nx.cpu's fold tables (cpu/nx_c_fold.c); scan's kinds are reduce's, and
# arg_reduce's the extremes of reduce's.
REDUCE = [(["sum", "prod"], NUMERIC), (["max", "min"], NUMERIC + ["bool"])]
ARG_REDUCE = [(["max", "min"], NUMERIC + ["bool"])]
# A kind whose name C++ reserves takes a trailing underscore.
C_NAMES = {"and": "and_", "or": "or_", "xor": "xor_", "bool": "bool_"}


def c_name(n):
    return C_NAMES.get(n, n)


def modules():
    """Each module: its key, and the source that instantiates it."""
    for w, t in WIDTHS.items():
        yield f"contiguous.{w}", f'#include "contiguous.hip"\nCONTIGUOUS({t})\n'
        yield f"where.{w}", f'#include "where.hip"\nWHERE({t})\n'
        yield f"gather.{w}", f'#include "gather.hip"\nGATHER({t})\n'
        yield f"pad.{w}", f'#include "pad.hip"\nPAD({t})\n'
        yield f"place.{w}", f'#include "place.hip"\nPLACE({t})\n'
        yield f"scatter_set.{w}", f'#include "scatter.hip"\nSCATTER_SET({t})\n'
    for s, cs in DTYPES:
        for d, cd in DTYPES:
            if s != d:
                yield f"cast.{s}.{d}", f'#include "cast.hip"\nCAST({cs}, {cd})\n'
    for family, macro, table in (("unary", "UNARY", UNARY), ("binary", "BINARY", BINARY),
                                 ("compare", "COMPARE", COMPARE)):
        for kinds, dtypes in table:
            for k in kinds:
                for d in dtypes:
                    yield f"{family}.{k}.{d}", f'#include "{family}.hip"\n{macro}({c_name(k)}, {c_name(d)})\n'
    for d in NUMERIC:
        yield f"fma.{d}", f'#include "fma.hip"\nFMA({c_name(d)})\n'
    yield "threefry", '#include "random.hip"\nTHREEFRY()\n'
    for d, cd in DTYPES:
        yield f"sort.{d}", f'#include "sort.hip"\nSORT({cd})\n'
        yield f"scatter_add.{d}", f'#include "scatter.hip"\nSCATTER_ADD({cd})\n'
        for k in ("max", "min"):
            yield f"scatter_{k}.{d}", f'#include "scatter.hip"\nSCATTER_EXTREME({k}, {cd})\n'
    for d in NUMERIC:
        yield f"matmul.{d}", f'#include "matmul.hip"\nMATMUL({c_name(d)})\n'
    for kinds, dtypes in REDUCE:
        for k in kinds:
            for d in dtypes:
                yield f"scan.{k}.{d}", f'#include "scan.hip"\nSCAN({k}, {c_name(d)})\n'
    for prefix, macro, table in (("", "REDUCE", REDUCE), ("arg", "ARG_REDUCE", ARG_REDUCE)):
        for kinds, dtypes in table:
            for k in kinds:
                for d in dtypes:
                    yield (f"{macro.lower()}.{prefix}{k}.{d}",
                           f'#include "reduce.hip"\n{macro}({k}, {c_name(d)})\n')


def inputs():
    """The files the code objects are made of, by their path from kernels/."""
    files = sorted(SRC.rglob("*")) + [DTYPE_H, pathlib.Path(__file__).resolve()]
    return {os.path.relpath(f, KERNELS): f for f in files if f.is_file()}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# comgr

DATA_KIND_SOURCE, DATA_KIND_INCLUDE, DATA_KIND_LOG, DATA_KIND_EXECUTABLE = 1, 2, 5, 8
# Version 3's languages and actions.
LANGUAGE_HIP = 3
ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = 12
ACTION_CODEGEN_BC_TO_RELOCATABLE = 4
ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 7


def comgr_path():
    return pathlib.Path(os.environ.get("COMGR_PATH", "/opt/rocm/lib/libamd_comgr.so")).resolve()


def toolchain():
    lib = comgr_path()
    if not lib.exists():
        sys.exit(f"no comgr at {lib}")
    c = ctypes.CDLL(str(lib))
    major, minor = ctypes.c_uint64(), ctypes.c_uint64()
    c.amd_comgr_get_version(ctypes.byref(major), ctypes.byref(minor))
    if major.value != 3:
        sys.exit(f"comgr {major.value}.{minor.value}: the script speaks version 3")
    return {"library": lib.name, "version": f"{major.value}.{minor.value}", "sha256": sha256(lib)}


class Comgr:
    def __init__(self):
        self.c = ctypes.CDLL(str(comgr_path()))

    def check(self, status, what):
        if status != 0:
            raise RuntimeError(f"comgr: {what} failed with status {status}")

    def handle(self):
        return ctypes.c_uint64()

    def data(self, kind, name, data):
        d = self.handle()
        self.check(self.c.amd_comgr_create_data(kind, ctypes.byref(d)), "create_data")
        self.check(self.c.amd_comgr_set_data(d, ctypes.c_size_t(len(data)), data), "set_data")
        self.check(self.c.amd_comgr_set_data_name(d, name.encode()), "set_data_name")
        return d

    def get(self, data_set, kind):
        d, n = self.handle(), ctypes.c_size_t()
        if self.c.amd_comgr_action_data_get_data(data_set, kind, ctypes.c_size_t(0), ctypes.byref(d)) != 0:
            return b""
        self.check(self.c.amd_comgr_get_data(d, ctypes.byref(n), None), "get_data")
        buf = ctypes.create_string_buffer(n.value)
        self.check(self.c.amd_comgr_get_data(d, ctypes.byref(n), buf), "get_data")
        return buf.raw[: n.value]

    def options(self, info, opts):
        arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        self.check(self.c.amd_comgr_action_info_set_option_list(info, arr, ctypes.c_size_t(len(opts))),
                   "set_option_list")

    def compile(self, target, name, source, includes, libs):
        c = self.c
        info = self.handle()
        self.check(c.amd_comgr_create_action_info(ctypes.byref(info)), "create_action_info")
        self.check(c.amd_comgr_action_info_set_language(info, LANGUAGE_HIP), "set_language")
        self.check(c.amd_comgr_action_info_set_isa_name(info, f"amdgcn-amd-amdhsa--{target}".encode()),
                   "set_isa_name")
        self.check(c.amd_comgr_action_info_set_logging(info, True), "set_logging")
        sets = [self.handle() for _ in range(4)]  # source, bitcode, relocatable, executable
        for s in sets:
            self.check(c.amd_comgr_create_data_set(ctypes.byref(s)), "create_data_set")
        self.check(c.amd_comgr_data_set_add(sets[0], self.data(DATA_KIND_SOURCE, name, source)), "add")
        for n, d in includes.items():
            self.check(c.amd_comgr_data_set_add(sets[0], self.data(DATA_KIND_INCLUDE, n, d)), "add")

        def action(kind, opts, src, dst, what):
            self.options(info, opts)
            if c.amd_comgr_do_action(kind, info, src, dst) != 0:
                raise RuntimeError(f"{name}: {what} failed\n{self.get(dst, DATA_KIND_LOG).decode()}")

        action(ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
               COMPILE + [f"--offload-arch={target}", f"--rocm-device-lib-path={libs}"],
               sets[0], sets[1], "compile")
        action(ACTION_CODEGEN_BC_TO_RELOCATABLE, CODEGEN, sets[1], sets[2], "codegen")
        action(ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, [""], sets[2], sets[3], "link")
        return self.get(sets[3], DATA_KIND_EXECUTABLE)


# Generation

COMGR = None


def compile_one(job):
    global COMGR
    if COMGR is None:
        COMGR = Comgr()
    target, key, source, includes, libs = job
    return target, key, COMGR.compile(target, key + ".hip", source.encode(), includes, libs)


def extract_device_libs():
    """A compile that keeps its temporary files, where comgr writes its embedded
    device libraries: run in a process of its own, whose environment comgr
    reads when it loads."""
    try:
        Comgr().compile("gfx1201", "probe.hip", b'extern "C" __attribute__((global)) void k() {}', {}, "")
    except RuntimeError:
        pass  # the libraries are written out before anything can fail


def device_libs(workdir):
    """comgr's device libraries, as its clang looks for them, in a directory of
    [workdir]. comgr writes them out for each compile, naming the ISA library
    of a generic target oclc_isa_version_12_generic.bc where its clang reads
    oclc_isa_version_12-generic.bc, so a compile for a generic target links no
    device library at all (comgr 3.0, ROCm 7.0). They are written out once, by
    a compile that keeps its temporary files, in a process of its own, and each
    generic ISA library is copied under the name clang reads."""
    tmp = workdir / "extract"
    tmp.mkdir()
    env = {**os.environ, "TMPDIR": str(tmp), "AMD_COMGR_SAVE_TEMPS": "1"}
    subprocess.run([sys.executable, __file__, "--extract-device-libs"], env=env, check=True)
    found = sorted(tmp.glob("comgr-*/rocm/amdgcn/bitcode"))
    if not found:
        sys.exit("comgr wrote out no device library")
    libs = workdir / "bitcode"
    shutil.copytree(found[0], libs)
    for f in libs.glob("oclc_isa_version_*_generic.bc"):
        version = f.name[len("oclc_isa_version_"):-len("_generic.bc")]
        shutil.copy(f, libs / f"oclc_isa_version_{version.replace('_', '-')}-generic.bc")
    return libs


def generate(outdir, jobs):
    """Writes every code object under [outdir]; its relative path from
    kernels/ and its digest, by path."""
    includes = {f.name: f.read_bytes() for f in [*SRC.glob("*.h"), *SRC.glob("*.hip"), *(SRC / "libc").glob("*.h"),
                                                 DTYPE_H]}
    with tempfile.TemporaryDirectory() as d:
        libs = device_libs(pathlib.Path(d))
        work = [(t, k, s, includes, libs) for t in TARGETS for k, s in modules()]
        return compile_all(outdir, work, jobs)


def compile_all(outdir, work, jobs):
    outputs = {}
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        for target, key, co in pool.imap_unordered(compile_one, work):
            if not co:
                sys.exit(f"{target}/{key}: comgr made no code object")
            path = outdir / target / f"{key}.co"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(co)
            outputs[f"{target}/{key}.co"] = hashlib.sha256(co).hexdigest()
    return dict(sorted(outputs.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if a committed file differs")
    ap.add_argument("--pin", action="store_true", help="move the toolchain pin and regenerate")
    ap.add_argument("-j", type=int, default=4, help="compiles at once")
    ap.add_argument("--extract-device-libs", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.extract_device_libs:
        extract_device_libs()
        return
    tool = toolchain()
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    if not args.pin and pins.get("comgr") != tool:
        sys.exit(f"comgr {tool} is not the pinned {pins.get('comgr')}; --pin moves the pin")
    if args.check:
        if {p: sha256(f) for p, f in inputs().items()} != pins.get("inputs"):
            sys.exit("an input differs from its pin: regenerate")
        with tempfile.TemporaryDirectory() as d:
            outputs = generate(pathlib.Path(d), args.j)
            for path, digest in outputs.items():
                if not (KERNELS / path).exists() or sha256(KERNELS / path) != digest:
                    sys.exit(f"{path} differs from what gen.py generates")
            if outputs != pins.get("outputs"):
                sys.exit("pins.json's outputs differ from what gen.py generates")
        print("up to date")
        return
    for target in TARGETS:
        for old in (KERNELS / target).glob("*.co"):
            old.unlink()
    outputs = generate(KERNELS, args.j)
    pins = {
        "comgr": tool,
        "compile": COMPILE,
        "codegen": CODEGEN,
        "inputs": {p: sha256(f) for p, f in inputs().items()},
        "outputs": outputs,
    }
    PINS.write_text(json.dumps(pins, indent=1) + "\n")
    total = sum((KERNELS / p).stat().st_size for p in outputs)
    print(f"{len(outputs)} code objects, {total} bytes")


if __name__ == "__main__":
    main()
