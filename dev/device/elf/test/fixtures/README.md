# ELF fixtures

Made in this directory with Homebrew clang 22.1.7 (`clang`) and Homebrew
LLD 21.1.8 (`ld.lld`). `H` is `-c -x c -O2 -fPIC -ffreestanding
-fno-math-errno -nostdlib -fno-ident`; `A` is `-c -x cl -cl-std=CL2.0
-target amdgcn-amd-amdhsa -mcpu=gfx1100 -mcode-object-version=5 -nogpulib -O2`.

- `host_x86_64.o`, a host object with RELA calls and addends: `clang $H --target=x86_64-none-unknown-elf host.c -o host_x86_64.o`
- `host_aarch64.o`, the same for aarch64, and the bench's host object: `clang $H -ffixed-x18 --target=aarch64-none-unknown-elf host.c -o host_aarch64.o`
- `amd_gfx1100.o`, a relocatable GPU object whose descriptor relocates to its kernel: `clang $A amd.cl -o amd_gfx1100.o`
- `amd_gfx1100.hsaco`, a linked code object with sections at addresses: `ld.lld -shared amd_gfx1100.o -o amd_gfx1100.hsaco`
- `amd_<k>_gfx1100.hsaco` for `<k>` of 16 and 128, `<k>` kernels without a symbol table, for the bench: `clang $A -DKERNELS=<k> amd.cl -o amd_<k>.o && ld.lld -shared --strip-all amd_<k>.o -o amd_<k>_gfx1100.hsaco`
- `simple_add_sm89.cubin`, an NVIDIA executable whose sections the reader finds by name, made by `nvrtc.c` with NVRTC 12.8.93 (CUDA 12.8, the PyPI package `nvidia-cuda-nvrtc-cu12==12.8.93`), whose files are under `$NVRTC`: `cc nvrtc.c -I$NVRTC/include -L$NVRTC/lib -l:libnvrtc.so.12 -Wl,-rpath,$NVRTC/lib -o nvrtc && ./nvrtc simple_add.cu sm_89 simple_add_sm89.cubin`; another release of NVRTC writes its own version into the file
- `global_sm89.cubin`, a cubin whose kernel uses an uninitialised `__device__` global, which NVRTC puts in `.nv.global` (SHT_NOBITS), relocated to from `.nv.constant4`, made as `simple_add_sm89.cubin` is, with the same NVRTC: `./nvrtc global.cu sm_89 global_sm89.cubin` (SHA-256 95ce0268f3e108f1ec9922c19c0a65822fdb74620b0165ecdd4e581cede397d0)
