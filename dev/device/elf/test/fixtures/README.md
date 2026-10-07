# ELF fixtures

Made in this directory with Homebrew clang 22.1.7 (`clang`) and Homebrew
LLD 21.1.8 (`ld.lld`). `H` is `-c -x c -O2 -fPIC -ffreestanding
-fno-math-errno -nostdlib -fno-ident`; `A` is `-c -x cl -cl-std=CL2.0
-target amdgcn-amd-amdhsa -mcpu=gfx1100 -mcode-object-version=5 -nogpulib -O2`.

- `host_x86_64.o`, a host object with RELA calls and addends: `clang $H --target=x86_64-none-unknown-elf host.c -o host_x86_64.o`
- `host_aarch64.o`, the same for aarch64, and the bench's host object: `clang $H -ffixed-x18 --target=aarch64-none-unknown-elf host.c -o host_aarch64.o`
- `amd_gfx1100.o`, a relocatable GPU object whose descriptor relocates to its kernel: `clang $A amd.cl -o amd_gfx1100.o`
- `amd_gfx1100.hsaco`, a linked code object with sections at addresses: `ld.lld -shared amd_gfx1100.o -o amd_gfx1100.hsaco`
- `amd_many_gfx1100.hsaco`, 128 kernels without a symbol table, for the bench: `clang $A -DMANY amd.cl -o many.o && ld.lld -shared --strip-all many.o -o amd_many_gfx1100.hsaco`
- `simple_add_sm89.cubin`, an NVIDIA executable whose sections the reader finds by name; rebuilding needs CUDA (kimchi): NVRTC 12.8 `nvrtcCompileProgram` on `simple_add.cu` with `--gpu-architecture=sm_89 --minimal`, then `nvrtcGetCUBIN`
