#!/usr/bin/env python3
"""Compiles nx.amd's kernels ahead of time into the code objects nx.amd
embeds, and the AMD suite's harness kernels into the code object the suite
and the bench load: one code object per artifact and processor,
<artifact>/<processor>.co.

Run from the repository root, on a machine with the pinned comgr (ROCm's
code object manager), which it loads from $COMGR_PATH, else
/opt/rocm/lib/libamd_comgr.so:

  uv run dev/nx2/lib/amd/kernels/gen.py
  uv run dev/nx2/lib/amd/kernels/gen.py --check
  uv run dev/nx2/lib/amd/kernels/gen.py --pin

An artifact is one translation unit that includes every .hip file of its
directory in name order. comgr compiles it as HIP for one processor, links
no device library, and links the code object. The unit sees no system
header: it reads the copies of its sources and of the pinned headers, the
libc subset below among them, and nothing else, so a code object depends on
the pinned inputs alone, whatever machine compiles it.

pins.json records the toolchain (the comgr library, by name, version and
digest), the options, and the digest of every input (the sources, the
headers they include, this script) and of every output, each path from
dev/nx2. A digest is BLAKE2b with 32 bytes of output, which OCaml's
Digest.BLAKE256 computes. The script refuses another comgr; --pin moves the
pin and regenerates everything. --check regenerates into a temporary
directory and fails if a committed file differs: comgr is deterministic for
a version and options. A dune rule checks the digests on every machine,
with no toolchain, so a source changed without regenerating, or a code
object edited, fails the tests.

Each generation checks that each kernel the artifact's list names (the X
macro of its header) is a kernel of its code object.

Python's standard library only.
"""

import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile

NX2 = pathlib.Path(__file__).resolve().parents[3]
PINS = NX2 / "lib/amd/kernels/pins.json"
PROCESSORS = ["gfx1201"]

# Each artifact: its directory, the header whose X macro lists its kernels,
# and that macro's name.
ARTIFACTS = [
    ("test/amd/support", "test/amd/support/harness.h", "NX_HARNESS_KERNELS"),
]

# The headers the sources may include, by name: the only ones the unit sees.
HEADERS = [
    "lib/array/nx_dtype.h", "lib/amd/kernels/libc/math.h", "lib/amd/kernels/libc/stdint.h",
    "lib/amd/kernels/libc/string.h", "test/amd/support/harness.h",
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
    "x86_64-unknown-linux-gnu",
]
CODEGEN = ["-O3", "-ffp-contract=off", "-mcode-object-version=6", "-mllvm",
           "-amdgpu-internalize-symbols"]


def digest(path):
    return hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()


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
    return {"library": lib.name, "version": f"{major.value}.{minor.value}", "digest": digest(lib)}


def sources(directory):
    """The pinned sources of an artifact: its .hip files."""
    return sorted((NX2 / directory).glob("*.hip"))


def inputs():
    files = [NX2 / h for h in HEADERS] + [pathlib.Path(__file__).resolve()]
    for d, _, _ in ARTIFACTS:
        files += sources(d)
    return {str(f.relative_to(NX2)): digest(f) for f in sorted(set(files))}


def kernels(header, macro):
    """The names the X macro [macro] of [header] lists."""
    text = (NX2 / header).read_text()
    m = re.search(rf"#define {macro}\(X\)((?:.*\\\n)*.*)", text)
    if m is None:
        sys.exit(f"{header}: no {macro}")
    return re.findall(r"X\((\w+)\)", m.group(1))


# comgr 3.0 (amd_comgr.h)

DATA_KIND_SOURCE, DATA_KIND_INCLUDE, DATA_KIND_LOG, DATA_KIND_EXECUTABLE = 1, 2, 5, 8
LANGUAGE_HIP = 3
ACTION_COMPILE_SOURCE_TO_BC = 2
ACTION_CODEGEN_BC_TO_RELOCATABLE = 4
ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 7


class Comgr:
    def __init__(self):
        self.c = ctypes.CDLL(str(comgr_path()))

    def check(self, status, what):
        if status != 0:
            sys.exit(f"comgr: {what} failed with status {status}")

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

    def compile(self, processor, unit, includes):
        """The code object of the HIP source [unit] for [processor], which
        sees the files [includes], by name, and nothing else."""
        c = self.c
        info = self.handle()
        self.check(c.amd_comgr_create_action_info(ctypes.byref(info)), "create_action_info")
        self.check(c.amd_comgr_action_info_set_language(info, LANGUAGE_HIP), "set_language")
        self.check(c.amd_comgr_action_info_set_isa_name(info, f"amdgcn-amd-amdhsa--{processor}".encode()),
                   "set_isa_name")
        self.check(c.amd_comgr_action_info_set_logging(info, True), "set_logging")
        sets = [self.handle() for _ in range(4)]  # source, bitcode, relocatable, executable
        for s in sets:
            self.check(c.amd_comgr_create_data_set(ctypes.byref(s)), "create_data_set")
        self.check(c.amd_comgr_data_set_add(sets[0], self.data(DATA_KIND_SOURCE, "unit.hip", unit)), "add")
        for name, data in includes.items():
            self.check(c.amd_comgr_data_set_add(sets[0], self.data(DATA_KIND_INCLUDE, name, data)), "add")

        def action(kind, opts, src, dst, what):
            arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
            self.check(c.amd_comgr_action_info_set_option_list(info, arr, ctypes.c_size_t(len(opts))),
                       "set_option_list")
            if c.amd_comgr_do_action(kind, info, src, dst) != 0:
                sys.exit(f"{what} failed\n{self.get(dst, DATA_KIND_LOG).decode()}")

        action(ACTION_COMPILE_SOURCE_TO_BC, COMPILE + [f"--offload-arch={processor}"], sets[0], sets[1],
               "compile")
        action(ACTION_CODEGEN_BC_TO_RELOCATABLE, CODEGEN, sets[1], sets[2], "codegen")
        action(ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, [""], sets[2], sets[3], "link")
        return self.get(sets[3], DATA_KIND_EXECUTABLE)


def generate(outdir):
    """Writes every code object under [outdir]; their digests by path."""
    comgr = Comgr()
    outputs = {}
    for directory, header, macro in ARTIFACTS:
        files = sources(directory) + [NX2 / h for h in HEADERS]
        includes = {f.name: f.read_bytes() for f in files}
        if len(includes) != len(files):
            sys.exit(f"{directory}: two inputs share a name")
        unit = "".join(f'#include "{f.name}"\n' for f in sources(directory)).encode()
        for processor in PROCESSORS:
            path = f"{directory}/{processor}.co"
            co = comgr.compile(processor, unit, includes)
            missing = [k for k in kernels(header, macro) if b"\0" + k.encode() + b".kd\0" not in co]
            if missing:
                sys.exit(f"{path}: no kernel {', '.join(missing)}")
            out = outdir / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(co)
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
        sys.exit(f"comgr {tool} is not the pinned {pins.get('toolchain')}; --pin moves the pin")
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
    PINS.write_text(json.dumps({"toolchain": tool, "compile": COMPILE, "codegen": CODEGEN,
                                "inputs": inputs(), "outputs": outputs}, indent=1) + "\n")
    for path in outputs:
        print(f"{path}: {(NX2 / path).stat().st_size} bytes")


if __name__ == "__main__":
    main()
