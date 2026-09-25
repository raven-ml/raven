"""Compile the checked-in cubin and record expectations using the frozen target.

Requires libnvrtc, Python 3.11+, and a tinygrad checkout/archive supplied through
--tinygrad-root. No GPU is opened. See README.md for the reproducible container
command and the compiler image digest.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

REFERENCE = "471a3aeb6924257d5e9bf321f5ff0a519163f18e"
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinygrad-root", type=Path, required=True)
    parser.add_argument("--compiler-image", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tinygrad_root))

    from tinygrad.dtype import dtypes
    from tinygrad.runtime.autogen import nvrtc
    from tinygrad.runtime.ops_nv import NVProgramData, nv_gpu
    from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler, nvrtc_check

    source = (HERE / "simple_add.cu").read_bytes()
    compiler = NVRTCCompiler("sm_89", ptx=False)
    lib = compiler.compile(source.decode())
    major, minor = ctypes.c_int(), ctypes.c_int()
    nvrtc_check(nvrtc.nvrtcVersion(major, minor))
    dev = SimpleNamespace(
        iface=SimpleNamespace(compute_class=nv_gpu.AMPERE_COMPUTE_B),
        renderer=None, shared_mem_window=0x729400000000,
        local_mem_window=0x729300000000, sass_version=0x89,
    )
    dev._ensure_has_local_memory = lambda size: setattr(dev, "slm_per_thread", size)
    obj = SimpleNamespace(name="simple_add", lib=lib, signature=[
        (None, None, dtypes.uint8, None), (None, None, dtypes.uint8, None),
        (None, None, dtypes.uint8, None), ("n", None, dtypes.int32, None),
    ])
    data = NVProgramData(dev, obj)
    fields = {
        "name": obj.name,
        "regs_usage": data.qmd.read("register_count_v"),
        "shmem_usage": data.qmd.read("shared_memory_size"),
        "lcmem_usage": dev.slm_per_thread,
        "cbuf0_size": len(data.cbuf_0) * 4,
        "constbuf0_size": data.constbufs[0][1],
        "kernargs_size": data.kernargs_size,
    }
    stem = HERE / "simple_add_sm89"
    stem.with_suffix(".cubin").write_bytes(lib)
    stem.with_suffix(".fields").write_text("".join(f"{k} {v}\n" for k, v in fields.items()))
    digest = lambda data: hashlib.sha256(data).hexdigest()
    provenance = {
        "reference": REFERENCE,
        "compiler_image": args.compiler_image,
        "compiler": f"NVRTC {major.value}.{minor.value}",
        "options": compiler.compile_options,
        "source_sha256": digest(source),
        "binary_sha256": digest(lib),
        "oracle_sha256": {
            name: digest((args.tinygrad_root / name).read_bytes()) for name in (
                "tinygrad/runtime/support/compiler_cuda.py", "tinygrad/runtime/ops_nv.py",
                "tinygrad/runtime/support/elf.py",
            )
        },
    }
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(fields)


if __name__ == "__main__":
    main()
