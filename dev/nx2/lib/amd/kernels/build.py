#!/usr/bin/env python3
"""Compiles nx.amd's kernels and the AMD suite's harness kernels to the
code objects they embed, through comgr (ROCm's code object manager), which
it loads from $COMGR_PATH, else /opt/rocm/lib/libamd_comgr.so. Run by hand,
from anywhere:

  uv run dev/nx2/lib/amd/kernels/build.py

A unit includes every .hip file of its directory in name order. comgr
compiles it as HIP for one processor, links no device library, and links
the code object. The unit sees no system header: it reads its sources and
the headers below, the libc subset among them, by name.

Python's standard library only.
"""

import ctypes
import os
import pathlib
import sys

NX2 = pathlib.Path(__file__).resolve().parents[3]
PROCESSOR = "gfx1201"

# Each unit: the directory of its sources and its code object.
UNITS = [
    ("lib/amd/kernels/src", "lib/amd/kernels/gfx1201.co"),
    ("test/amd/support", "test/amd/support/gfx1201.co"),
]

HEADERS = [
    "lib/array/nx_dtype.h", "lib/amd/kernels.h", "lib/amd/kernels/src/combine.h",
    "lib/amd/kernels/src/device.h", "lib/amd/kernels/libc/math.h",
    "lib/amd/kernels/libc/stdint.h", "lib/amd/kernels/libc/string.h",
    "test/amd/support/harness.h",
]

# Floats as the kernel contract states them: no contraction of a product
# into a sum, division and square roots correctly rounded, subnormals kept.
# The unit's ID names its symbol __hip_cuid_<id>: fixed, since comgr would
# otherwise derive it from the paths it is handed.
COMPILE = [
    "-O3", "-ffp-contract=off", "-fhip-fp32-correctly-rounded-divide-sqrt",
    "-fno-gpu-flush-denormals-to-zero", "-nogpuinc", "-nogpulib", "-nostdinc",
    "-mcode-object-version=6", "-std=c++17", "-cuid=nx2_amd", "-Wall", "-Werror",
    "-Xclang", "-disable-llvm-passes", "-Xclang", "-aux-triple", "-Xclang",
    "x86_64-unknown-linux-gnu", f"--offload-arch={PROCESSOR}",
]
CODEGEN = ["-O3", "-ffp-contract=off", "-mcode-object-version=6", "-mllvm",
           "-amdgpu-internalize-symbols"]

# comgr 3 (amd_comgr.h)

DATA_KIND_SOURCE, DATA_KIND_INCLUDE, DATA_KIND_LOG, DATA_KIND_EXECUTABLE = 1, 2, 5, 8
LANGUAGE_HIP = 3
ACTION_COMPILE_SOURCE_TO_BC = 2
ACTION_CODEGEN_BC_TO_RELOCATABLE = 4
ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 7

comgr = ctypes.CDLL(os.environ.get("COMGR_PATH", "/opt/rocm/lib/libamd_comgr.so"))


def check(status, what):
    if status != 0:
        sys.exit(f"comgr: {what} failed with status {status}")


def handle():
    return ctypes.c_uint64()


def data(kind, name, contents):
    d = handle()
    check(comgr.amd_comgr_create_data(kind, ctypes.byref(d)), "create_data")
    check(comgr.amd_comgr_set_data(d, ctypes.c_size_t(len(contents)), contents), "set_data")
    check(comgr.amd_comgr_set_data_name(d, name.encode()), "set_data_name")
    return d


def get(data_set, kind):
    d, n = handle(), ctypes.c_size_t()
    if comgr.amd_comgr_action_data_get_data(data_set, kind, ctypes.c_size_t(0), ctypes.byref(d)) != 0:
        return b""
    check(comgr.amd_comgr_get_data(d, ctypes.byref(n), None), "get_data")
    buf = ctypes.create_string_buffer(n.value)
    check(comgr.amd_comgr_get_data(d, ctypes.byref(n), buf), "get_data")
    return buf.raw[: n.value]


def compile_unit(unit, includes):
    """The code object of the HIP source [unit], which sees the files
    [includes], by name, and nothing else."""
    info = handle()
    check(comgr.amd_comgr_create_action_info(ctypes.byref(info)), "create_action_info")
    check(comgr.amd_comgr_action_info_set_language(info, LANGUAGE_HIP), "set_language")
    isa = f"amdgcn-amd-amdhsa--{PROCESSOR}".encode()
    check(comgr.amd_comgr_action_info_set_isa_name(info, isa), "set_isa_name")
    check(comgr.amd_comgr_action_info_set_logging(info, True), "set_logging")
    source, bitcode, relocatable, executable = sets = [handle() for _ in range(4)]
    for s in sets:
        check(comgr.amd_comgr_create_data_set(ctypes.byref(s)), "create_data_set")
    check(comgr.amd_comgr_data_set_add(source, data(DATA_KIND_SOURCE, "unit.hip", unit)), "add")
    for name, contents in includes.items():
        check(comgr.amd_comgr_data_set_add(source, data(DATA_KIND_INCLUDE, name, contents)), "add")

    def action(kind, opts, src, dst, what):
        arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        check(comgr.amd_comgr_action_info_set_option_list(info, arr, ctypes.c_size_t(len(opts))),
              "set_option_list")
        if comgr.amd_comgr_do_action(kind, info, src, dst) != 0:
            sys.exit(f"{what} failed\n{get(dst, DATA_KIND_LOG).decode()}")

    action(ACTION_COMPILE_SOURCE_TO_BC, COMPILE, source, bitcode, "compile")
    action(ACTION_CODEGEN_BC_TO_RELOCATABLE, CODEGEN, bitcode, relocatable, "codegen")
    action(ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, [""], relocatable, executable, "link")
    return get(executable, DATA_KIND_EXECUTABLE)


for directory, out in UNITS:
    sources = sorted((NX2 / directory).glob("*.hip"))
    includes = {f.name: f.read_bytes() for f in sources + [NX2 / h for h in HEADERS]}
    unit = "".join(f'#include "{f.name}"\n' for f in sources).encode()
    (NX2 / out).write_bytes(compile_unit(unit, includes))
