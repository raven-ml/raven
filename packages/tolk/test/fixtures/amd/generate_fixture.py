"""Build a linked AMD code object with LLVM and parse it with the frozen target.

Requires an AMDGPU-capable clang, ld.lld and Python 3.11+. No GPU is opened.
The source uses only compiler builtins, so ROCm device libraries are unnecessary.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REFERENCE = "471a3aeb6924257d5e9bf321f5ff0a519163f18e"
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinygrad-root", type=Path, required=True)
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--lld", default="ld.lld")
    args = parser.parse_args()
    sys.path.insert(0, str(args.tinygrad_root))
    from tinygrad.runtime.ops_amd import _amd_program_image
    from tinygrad.runtime.support.elf import elf_loader

    compile_options = ["-x", "hip", "--cuda-device-only", "--offload-arch=gfx1100",
                       "-nogpuinc", "-nogpulib", "-O3", "-mcumode",
                       "-mcode-object-version=5", "-cuid=tolk_fixture", "-fgpu-rdc", "-c"]
    codegen_options = ["-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1100",
                       "-nogpulib", "-O3", "-mcode-object-version=5", "-c"]
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        subprocess.run([args.clang, *compile_options, str(HERE / "simple_add.cpp"),
                        "-o", str(tmp / "kernel.bc")], check=True)
        subprocess.run([args.clang, *codegen_options, str(tmp / "kernel.bc"),
                        "-o", str(tmp / "kernel.o")], check=True)
        subprocess.run([args.lld, "-shared", str(tmp / "kernel.o"),
                        "-o", str(tmp / "kernel.hsaco")], check=True)
        lib = (tmp / "kernel.hsaco").read_bytes()
    # A hashable stand-in supplies only the target and LDS capacity used by the
    # parser; no device constructor, allocator, firmware or queue is involved.
    class Device:
        target = (11, 0, 0)
        iface = SimpleNamespace(props={"lds_size_in_kb": 64})
    data, image = _amd_program_image(Device(), lib)
    fields = asdict(data)
    del fields["libhash"]
    fields["image_size"] = len(image)
    sections = elf_loader(lib)[1]
    text = next(section for section in sections if section.name == ".text")
    fields["code_offset"], fields["code_size"] = text.header.sh_addr, text.header.sh_size
    stem = HERE / "simple_add_gfx1100"
    stem.with_suffix(".hsaco").write_bytes(lib)
    stem.with_suffix(".image").write_bytes(image)
    stem.with_suffix(".fields").write_text("".join(f"{k} {int(v)}\n" for k, v in fields.items()))
    digest = lambda data: hashlib.sha256(data).hexdigest()
    version = lambda command: subprocess.check_output([command, "--version"], text=True).strip()
    provenance = {
        "reference": REFERENCE,
        "compiler": version(args.clang),
        "linker": version(args.lld),
        "compile_options": compile_options,
        "codegen_options": codegen_options,
        "link_options": ["-shared"],
        "source_sha256": digest((HERE / "simple_add.cpp").read_bytes()),
        "binary_sha256": digest(lib),
        "image_sha256": digest(image),
        "oracle_sha256": {
            name: digest((args.tinygrad_root / name).read_bytes()) for name in (
                "tinygrad/runtime/ops_amd.py", "tinygrad/runtime/support/elf.py",
            )
        },
    }
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(fields)


if __name__ == "__main__":
    main()
