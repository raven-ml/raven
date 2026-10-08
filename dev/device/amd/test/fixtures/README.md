# AMD driver fixtures

Made in this directory from `kernels.cl` with Homebrew clang 22.1.7
(`clang`) and Homebrew LLD 21.1.8 (`ld.lld`):

- `kernels_gfx1201.hsaco`, the kernels `empty`, `double_index`, `spin` and
  `wild` for gfx1201:
  `clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 -mcode-object-version=5 -nogpulib -O2 kernels.cl -o kernels.o && ld.lld -shared kernels.o -o kernels_gfx1201.hsaco && rm kernels.o`
