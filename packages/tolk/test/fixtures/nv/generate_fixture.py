"""Generates the real-cubin fixture for test_runtime_nv.ml's skip-gated
"cubin fixture" test.

Compiles a trivial kernel to a cubin for sm_89 through the pinned clone's
nvrtc binding (no GPU needed, only the CUDA toolkit's libnvrtc), then
records the fields tolk's Program.load must parse from it. The test
compares its parse of simple_add_sm89.cubin against simple_add_sm89.fields
and skips when either file is absent, so this script only needs to run on
a Linux box with the CUDA toolkit; commit both outputs.

Run (from anywhere):
  uv run python packages/tolk/test/fixtures/nv/generate_fixture.py
"""

import os
import re
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "..", "_tinygrad"))

from tinygrad.helpers import round_up  # noqa: E402
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler  # noqa: E402
from tinygrad.runtime.support.elf import elf_loader  # noqa: E402

NAME, ARCH = "simple_add", "sm_89"
SRC = """
extern "C" __global__ void simple_add(int* out, const int* a, const int* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
"""


def parse_elf_info(sh, start_off=0):
    # REPLICATED BLOCK -- NVProgram._parse_elf_info
    # (_tinygrad/tinygrad/runtime/ops_nv.py); re-sync on pin moves.
    while start_off < sh.header.sh_size:
        typ, param, sz = struct.unpack_from("BBH", sh.content, start_off)
        yield typ, param, sh.content[start_off + 4 : start_off + sz + 4] if typ == 0x4 else sz
        start_off += (sz if typ == 0x4 else 0) + 4
    # END REPLICATED BLOCK


def parse_fields(lib):
    # REPLICATED BLOCK -- the non-NAK section walk of NVProgram.__init__
    # (_tinygrad/tinygrad/runtime/ops_nv.py, the sections loop), with the
    # address bookkeeping dropped: only the sizes tolk's test asserts on.
    # Re-sync on pin moves.
    _image, sections, _relocs = elf_loader(lib, force_section_align=128)
    regs_usage, shmem_usage, lcmem_usage, cbuf0_size = 0, 0x400, 0x240, 0
    constbufs = {0: 0x160}
    for sh in sections:
        if sh.name == f".nv.shared.{NAME}":
            shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
        if m := re.match(r"\.nv\.constant(\d+)", sh.name):
            constbufs[int(m.group(1))] = sh.header.sh_size
        elif sh.name.startswith(".nv.info"):
            for typ, param, data in parse_elf_info(sh):
                if sh.name == f".nv.info.{NAME}" and param == 0xA:
                    cbuf0_size = struct.unpack_from("IH", data)[1]
                elif sh.name == ".nv.info" and param == 0x12:
                    lcmem_usage = struct.unpack_from("II", data)[1] + 0x240
                elif sh.name == ".nv.info" and param == 0x2F:
                    regs_usage = struct.unpack_from("II", data)[1]
    # END REPLICATED BLOCK
    return {
        "name": NAME,
        "regs_usage": regs_usage,
        "shmem_usage": shmem_usage,
        "lcmem_usage": lcmem_usage,
        "cbuf0_size": cbuf0_size,
        "constbuf0_size": constbufs[0],
        "kernargs_alloc_size": round_up(constbufs[0], 1 << 8) + (8 << 8),
    }


def main():
    lib = NVRTCCompiler(ARCH, ptx=False, cache_key="nv").compile(SRC)
    with open(os.path.join(_HERE, f"{NAME}_{ARCH}.cubin"), "wb") as f:
        f.write(lib)
    fields = parse_fields(lib)
    with open(os.path.join(_HERE, f"{NAME}_{ARCH}.fields"), "w") as f:
        for k, v in fields.items():
            f.write(f"{k} {v}\n")
    print("\n".join(f"{k} {v}" for k, v in fields.items()))


if __name__ == "__main__":
    main()
